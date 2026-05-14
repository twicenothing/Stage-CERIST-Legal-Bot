from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload
from typing import Optional, List, Any
import json
import uuid
import os
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import insert
from pathlib import Path
from services.rag_setup import stream_legal_answer
from core.database import get_db, SessionLocal
from core.models import ChatSession, Message, User, generate_uuid
from services.security import get_current_user
from core.config import settings

router = APIRouter(prefix="/rag", tags=["Legal Assistant"])

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class MessageFeedback(BaseModel):
    feedback: Optional[str] = None # "like", "dislike", ou null

class IncomingMessage(BaseModel):
    id: str
    role: str
    parts: List[Any]

class ChatCompletionBody(BaseModel):
    chatSessionId: str
    isNewChat: bool
    messages: List[IncomingMessage]

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_text_from_message(message: IncomingMessage) -> str:
    """Extrait le texte brut de la structure complexe du message."""
    for part in message.parts:
        if isinstance(part, dict) and part.get("type") == "text":
            return part.get("text", "")
        if hasattr(part, "type") and part.type == "text":
            return part.text or ""
    return "Nouvelle Conversation"

# ── Vercel AI SDK v5 SSE helpers ──────────────────────────────────────────────

def _event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

def sse_start() -> str: return _event({"type": "start"})
def sse_start_step() -> str: return _event({"type": "start-step"})
def sse_text_start(part_id: str) -> str: return _event({"type": "text-start", "id": part_id})
def sse_text_delta(part_id: str, delta: str) -> str: return _event({"type": "text-delta", "id": part_id, "delta": delta})
def sse_text_end(part_id: str) -> str: return _event({"type": "text-end", "id": part_id})
def sse_finish_step() -> str: return _event({"type": "finish-step"})
def sse_finish(finish_reason: str = "stop") -> str: return _event({"type": "finish", "finishReason": finish_reason})
def sse_done() -> str: return "data: [DONE]\n\n"

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
            user_id=current_user.id
        )
        db.add(new_session)
        try:
            db.commit()
        except:
            db.rollback() 

    user_msg_record = Message(
        id=user_message.id,
        session_id=session_id,
        role="user",
        parts=user_message.parts
    )
    db.add(user_msg_record)
    try:
        db.commit()
    except:
        db.rollback()

    async def generate():
        full_text = ""
        sources = []
        part_id = str(uuid.uuid4())

        yield sse_start()
        yield sse_start_step()
        yield sse_text_start(part_id)

        async for event in stream_legal_answer(query):
            if event["type"] == "sources":
                sources = event["sources"]
            elif event["type"] == "chunk":
                token = event["text"]
                full_text += token
                yield sse_text_delta(part_id, token)

        if sources:
            sources_suffix = "\n\n**Documents pertinents :**\n"
            for s in sources:
                page = s.get('page', 'Inconnu')
                sources_suffix += f"- {s['title']} - Page {page} (Pertinence : {s['score']}%)\n"
            
            full_text += sources_suffix
            yield sse_text_delta(part_id, sources_suffix)

        yield sse_text_end(part_id)
        yield sse_finish_step()
        yield sse_finish("stop")
        yield sse_done()

        gen_db = SessionLocal()
        try:
            assistant_parts = [{"type": "text", "text": full_text}]
            if sources:
                assistant_parts.append({"type": "sources", "sources": sources})

            new_assistant_msg = Message(
                id=generate_uuid(),
                session_id=session_id,
                model_id=settings.LLM_MODEL,
                role="assistant",
                parts=assistant_parts
            )
            gen_db.add(new_assistant_msg)
            gen_db.commit()
        except Exception as e:
            print(f"[STREAM DB ERROR] {e}")
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

# ── Gestion des Sessions ──────────────────────────────────────────────────────

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
        {"id": s.id, "title": s.title, "createdAt": s.created_at.isoformat()}
        for s in sessions
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
        "chatMessages": [
            {
                "id": m.id,
                "role": m.role,
                "parts": m.parts,
                "createdAt": m.created_at.isoformat(),
                "feedback": m.feedback,
            }
            for m in sorted(session.messages, key=lambda x: x.created_at)
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
    return {"status": "success", "feedback": message.feedback}

@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id, 
        ChatSession.user_id == current_user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session introuvable")

    db.delete(session)
    db.commit()
    return {"message": "Session supprimée", "id": session_id}

# ── Service PDF ───────────────────────────────────────────────────────────────


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
        headers={"Content-Disposition": f'inline; filename="{clean_filename}"'}
    )