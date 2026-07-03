from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload
from typing import Optional, List, Any
import json
import uuid
import os
import time
from datetime import datetime
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path

from services.rag_setup import stream_legal_answer
from core.database import get_db, SessionLocal
from core.models import ChatSession, Message, User, Report, generate_uuid
from services.security import get_current_user
from core.config import settings

router = APIRouter(prefix="/rag", tags=["Legal Assistant"])

NO_ANSWER_PHRASE = "Je suis désolé, je n'ai pas la réponse"


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class MessageFeedback(BaseModel):
    feedback: Optional[str] = None  # "like", "dislike", ou null


class IncomingMessage(BaseModel):
    id: str
    role: str
    parts: List[Any]


class ChatCompletionBody(BaseModel):
    chatSessionId: str
    isNewChat: bool
    messages: List[IncomingMessage]


class MessageReport(BaseModel):
    reason: str
    details: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_text_from_message(message: IncomingMessage) -> str:
    """
    Extracts raw text from the complex message structure.
    """
    for part in message.parts:
        if isinstance(part, dict) and part.get("type") == "text":
            return part.get("text", "")

        if hasattr(part, "type") and part.type == "text":
            return part.text or ""

    return "Nouvelle Conversation"


def now_perf_ms(start_perf: float) -> float:
    return round((time.perf_counter() - start_perf) * 1000, 2)


def dedupe_sources_by_page(sources: list[dict]) -> list[dict]:
    """
    Removes duplicate chunks from the same PDF page.
    Keeps the highest score for each (title, page).
    """
    unique = {}

    for source in sources or []:
        title = source.get("title", "")
        page = str(source.get("page", "Inconnu"))
        key = (title, page)

        if key not in unique:
            unique[key] = source
            continue

        old_score = unique[key].get("score", 0)
        new_score = source.get("score", 0)

        try:
            if float(new_score) > float(old_score):
                unique[key] = source
        except Exception:
            pass

    return list(unique.values())


# ── Vercel AI SDK v5 SSE helpers ──────────────────────────────────────────────

def _event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def sse_start() -> str:
    return _event({"type": "start"})


def sse_start_step() -> str:
    return _event({"type": "start-step"})


def sse_text_start(part_id: str) -> str:
    return _event({"type": "text-start", "id": part_id})


def sse_text_delta(part_id: str, delta: str) -> str:
    return _event({"type": "text-delta", "id": part_id, "delta": delta})


def sse_text_end(part_id: str) -> str:
    return _event({"type": "text-end", "id": part_id})


def sse_finish_step() -> str:
    return _event({"type": "finish-step"})


def sse_finish(finish_reason: str = "stop") -> str:
    return _event({"type": "finish", "finishReason": finish_reason})


def sse_done() -> str:
    return "data: [DONE]\n\n"


# ── Chat endpoint ─────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat_completion_handler(
    body: ChatCompletionBody,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user_message = body.messages[-1]
    query = get_text_from_message(user_message)
    session_id = body.chatSessionId

    if body.isNewChat:
        new_session = ChatSession(
            id=session_id,
            title=query[:100],
            user_id=current_user.id,
        )
        db.add(new_session)

        try:
            db.commit()
        except Exception:
            db.rollback()

    user_msg_record = Message(
        id=user_message.id,
        session_id=session_id,
        role="user",
        parts=user_message.parts,
    )

    db.add(user_msg_record)

    try:
        db.commit()
    except Exception:
        db.rollback()

    async def generate():
        full_text = ""
        answer_text_only = ""
        sources = []
        final_title = query[:100]
        part_id = str(uuid.uuid4())

        started_perf = time.perf_counter()

        optimized_query_ms = None
        sources_received_ms = None
        first_token_ms = None

        chunk_count = 0

        yield sse_start()
        yield sse_start_step()
        yield sse_text_start(part_id)

        async for event in stream_legal_answer(query):
            event_type = event.get("type")

            if event_type == "optimized_query":
                final_title = event.get("text", final_title)
                optimized_query_ms = now_perf_ms(started_perf)

            elif event_type == "sources":
                raw_sources = event.get("sources", [])
                sources = dedupe_sources_by_page(raw_sources)
                sources_received_ms = now_perf_ms(started_perf)

            elif event_type == "chunk":
                token = event.get("text", "")

                if first_token_ms is None:
                    first_token_ms = now_perf_ms(started_perf)

                chunk_count += 1
                answer_text_only += token
                full_text += token

                yield sse_text_delta(part_id, token)

        total_duration_ms = now_perf_ms(started_perf)

        if sources:
            sources_suffix = "\n\n**Documents pertinents :**\n"

            for source in sources:
                page = source.get("page", "Inconnu")
                title = source.get("title", "Document inconnu")
                score = source.get("score", 0)

                sources_suffix += f"- {title} - Page {page} (Pertinence : {score}%)\n"

            full_text += sources_suffix
            yield sse_text_delta(part_id, sources_suffix)

        health_metrics = {
            "total_duration_ms": total_duration_ms,
            "total_duration_seconds": round(total_duration_ms / 1000, 2),

            "time_to_optimized_query_ms": optimized_query_ms,
            "time_to_sources_ms": sources_received_ms,
            "time_to_first_token_ms": first_token_ms,

            "chunk_count": chunk_count,
            "answer_chars": len(answer_text_only),
            "source_count": len(sources),
            "refused": NO_ANSWER_PHRASE in answer_text_only,

            "model_id": settings.LLM_MODEL,
            "created_at": datetime.utcnow().isoformat(),
        }

        yield sse_text_end(part_id)
        yield sse_finish_step()
        yield sse_finish("stop")
        yield sse_done()

        # ==========================================
        # SAVE ASSISTANT MESSAGE + METRICS
        # ==========================================
        gen_db = SessionLocal()

        try:
            assistant_parts = [
                {
                    "type": "text",
                    "text": full_text,
                },
                {
                    "type": "metrics",
                    "metrics": health_metrics,
                },
            ]

            if sources:
                assistant_parts.append({
                    "type": "sources",
                    "sources": sources,
                })

            new_assistant_msg = Message(
                id=generate_uuid(),
                session_id=session_id,
                model_id=settings.LLM_MODEL,
                role="assistant",
                parts=assistant_parts,
            )

            gen_db.add(new_assistant_msg)

            if body.isNewChat:
                session_to_update = (
                    gen_db.query(ChatSession)
                    .filter(ChatSession.id == session_id)
                    .first()
                )

                if session_to_update:
                    session_to_update.title = final_title[:100]

            gen_db.commit()

        except Exception as e:
            print(f"[STREAM DB ERROR] {e}")
            gen_db.rollback()

        finally:
            gen_db.close()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Sessions ──────────────────────────────────────────────────────────────────

@router.get("/sessions")
async def get_user_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )

    return [
        {
            "id": session.id,
            "title": session.title,
            "createdAt": session.created_at.isoformat(),
            "archived": session.archived,
        }
        for session in sessions
    ]


@router.get("/session/{session_id}")
async def get_chat_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = (
        db.query(ChatSession)
        .options(joinedload(ChatSession.messages))
        .filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id)
        .first()
    )

    if not session:
        raise HTTPException(status_code=404, detail="Session introuvable")

    return {
        "id": session.id,
        "title": session.title,
        "createdAt": session.created_at.isoformat(),
        "archived": session.archived,
        "chatMessages": [
            {
                "id": message.id,
                "role": message.role,
                "parts": message.parts,
                "createdAt": message.created_at.isoformat(),
                "feedback": message.feedback,
            }
            for message in sorted(session.messages, key=lambda x: x.created_at)
        ],
    }


@router.post("/message/{message_id}/feedback")
async def update_message_feedback(
    message_id: str,
    body: MessageFeedback,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    message = (
        db.query(Message)
        .join(ChatSession)
        .filter(Message.id == message_id, ChatSession.user_id == current_user.id)
        .first()
    )

    if not message:
        raise HTTPException(status_code=404, detail="Message introuvable")

    message.feedback = body.feedback
    db.commit()

    return {
        "status": "success",
        "feedback": message.feedback,
    }


@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id)
        .first()
    )

    if not session:
        raise HTTPException(status_code=404, detail="Session introuvable")

    db.delete(session)
    db.commit()

    return {
        "message": "Session supprimée",
        "id": session_id,
    }


@router.post("/session/{session_id}/archive")
async def archive_chat_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id)
        .first()
    )

    if not session:
        raise HTTPException(status_code=404, detail="Session introuvable")

    session.archived = True
    db.commit()

    return {
        "message": "Session archivée",
        "id": session_id,
    }


# ── PDF service ────────────────────────────────────────────────────────────────

@router.get("/pdf")
async def get_pdf(title: str):
    clean_filename = os.path.basename(title)

    if not clean_filename.lower().endswith(".pdf"):
        clean_filename += ".pdf"

    pdf_dir = Path("../") / settings.PDF_PATH
    pdf_old_dir = Path("../") / settings.PDF_OLD_PATH

    search_directories = [pdf_dir, pdf_old_dir]
    found_file_path = None

    for directory in search_directories:
        if directory.exists():
            matches = list(directory.rglob(clean_filename))

            if matches:
                found_file_path = matches[0]
                break

    if not found_file_path or not found_file_path.exists():
        raise HTTPException(status_code=404, detail="Document PDF introuvable.")

    return FileResponse(
        path=str(found_file_path),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{clean_filename}"'
        },
    )


# ── Reports ───────────────────────────────────────────────────────────────────

@router.post("/message/{message_id}/report")
async def report_message_handler(
    message_id: str,
    body: MessageReport,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    message = db.query(Message).filter(Message.id == message_id).first()

    if not message:
        raise HTTPException(status_code=404, detail="Message introuvable")

    existing_report = (
        db.query(Report)
        .filter(
            Report.message_id == message_id,
            Report.user_id == current_user.id,
        )
        .first()
    )

    if existing_report:
        raise HTTPException(status_code=400, detail="Vous avez déjà signalé ce message")

    new_report = Report(
        id=generate_uuid(),
        message_id=message_id,
        user_id=current_user.id,
        reason=body.reason,
        details=body.details,
    )

    db.add(new_report)

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Erreur lors de l'enregistrement du signalement",
        )

    return {
        "status": "success",
        "message": "Le signalement a été enregistré avec succès",
    }

@router.get("/admin/pdfs")
async def list_indexed_pdfs(
    current_user: User = Depends(get_current_user),
):
    """
    Returns a flat list of every PDF found under PDF_PATH and PDF_OLD_PATH,
    ignoring the year subfolders they're organized in.
    """
    pdf_dirs = {
        "pdf": Path("../") / settings.PDF_PATH,
        "pdf_old": Path("../") / settings.PDF_OLD_PATH,
    }

    pdfs = []

    for source_label, base_dir in pdf_dirs.items():
        if not base_dir.exists():
            continue

        for pdf_path in base_dir.rglob("*.pdf"):
            stat = pdf_path.stat()
            pdfs.append({
                "filename": pdf_path.name,
                "year_folder": pdf_path.parent.name,
                "source": source_label,  # "pdf" or "pdf_old"
                "size_bytes": stat.st_size,
                "modified_at": datetime.utcfromtimestamp(stat.st_mtime).isoformat(),
            })

    pdfs.sort(key=lambda p: p["filename"].lower())

    return {"count": len(pdfs), "pdfs": pdfs}