from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, distinct
from core.database import get_db
from core.models import User, ChatSession, Message
from services.security import get_admin_user
from datetime import datetime, timedelta
import logging
import os
import json
from collections import defaultdict
from core.config import settings

router = APIRouter(prefix="/stats", tags=["Admin Statistics"])

logger = logging.getLogger(__name__)

NO_ANSWER_PHRASE = "Je suis désolé, je n'ai pas la réponse"


# ==============================================================================
# GENERIC HELPERS
# ==============================================================================

def has_field(model, field_name: str) -> bool:
    return hasattr(model, field_name)


def safe_percentage(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0

    return round((numerator / denominator) * 100, 2)


def serialize_datetime(value):
    if value is None:
        return None

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return str(value)


def avg(values: list[float]) -> float:
    if not values:
        return 0.0

    return round(sum(values) / len(values), 2)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0

    values = sorted(values)
    index = int(round((p / 100) * (len(values) - 1)))

    return round(values[index], 2)


def get_message_text_field():
    """
    Detects a simple text/content column in Message.
    Your app mainly uses parts, but this keeps fallback compatibility.
    """
    possible_fields = ["content", "text", "message", "body", "answer"]

    for field in possible_fields:
        if has_field(Message, field):
            return getattr(Message, field)

    return None


def get_session_fk_field():
    possible_fields = ["chat_session_id", "session_id", "conversation_id"]

    for field in possible_fields:
        if has_field(Message, field):
            return getattr(Message, field)

    return None


# ==============================================================================
# PARTS / METRICS HELPERS
# ==============================================================================

def parse_parts(parts):
    if parts is None:
        return []

    if isinstance(parts, str):
        try:
            return json.loads(parts)
        except Exception:
            return []

    if isinstance(parts, list):
        return parts

    return []


def extract_text_from_parts(parts) -> str:
    parsed_parts = parse_parts(parts)
    texts = []

    for part in parsed_parts:
        if isinstance(part, dict) and part.get("type") == "text":
            texts.append(str(part.get("text", "")))

    return "\n".join(texts)


def extract_metrics_from_parts(parts):
    parsed_parts = parse_parts(parts)

    for part in parsed_parts:
        if isinstance(part, dict) and part.get("type") == "metrics":
            metrics = part.get("metrics", {})

            if isinstance(metrics, dict):
                return metrics

    return None


def get_message_preview(message: Message) -> str:
    parts_text = extract_text_from_parts(getattr(message, "parts", None))

    if parts_text:
        return parts_text[:250]

    text_field = get_message_text_field()

    if text_field is not None:
        value = getattr(message, text_field.key, "") or ""
        return str(value)[:250]

    return ""


# ==============================================================================
# TRAFFIC HELPERS
# ==============================================================================

def fill_daily_traffic(raw_data: dict, days: int):
    today = datetime.utcnow().date()
    start_day = today - timedelta(days=days - 1)

    result = []

    for i in range(days):
        current_day = start_day + timedelta(days=i)
        date_str = current_day.strftime("%Y-%m-%d")

        result.append({
            "date": date_str,
            "count": raw_data.get(date_str, 0),
        })

    return result


def build_daily_latency(raw_data: dict, days: int):
    today = datetime.utcnow().date()
    start_day = today - timedelta(days=days - 1)

    result = []

    for i in range(days):
        current_day = start_day + timedelta(days=i)
        date_str = current_day.strftime("%Y-%m-%d")
        values = raw_data.get(date_str, [])

        result.append({
            "date": date_str,
            "avg_total_duration_seconds": avg(values),
            "count": len(values),
        })

    return result


def subtract_months(date_obj: datetime, months: int):
    month = date_obj.month - months
    year = date_obj.year

    while month <= 0:
        month += 12
        year -= 1

    return datetime(year, month, 1)


def fill_monthly_traffic(raw_data: dict, months: int = 12):
    current_month_start = datetime.utcnow().replace(day=1)

    month_keys = []

    for i in reversed(range(months)):
        month_date = subtract_months(current_month_start, i)
        month_keys.append(month_date.strftime("%Y-%m"))

    return [
        {
            "month": month_key,
            "count": raw_data.get(month_key, 0),
        }
        for month_key in month_keys
    ]


# ==============================================================================
# SETTINGS HELPERS
# ==============================================================================

def get_setting_value(name: str, default=None):
    value = getattr(settings, name, None)

    if value is not None:
        return value

    return os.getenv(name, default)


def get_bool_config(name: str, default: bool = False) -> bool:
    value = get_setting_value(name, str(default).lower())

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def get_int_config(name: str, default: int) -> int:
    value = get_setting_value(name, default)

    try:
        return int(value)
    except Exception:
        return default


def get_float_config(name: str, default: float) -> float:
    value = get_setting_value(name, default)

    try:
        return float(value)
    except Exception:
        return default


# ==============================================================================
# ADMIN APP CONFIG
# ==============================================================================

@router.get("/app-config")
def get_app_config(
    admin: User = Depends(get_admin_user),
):
    """
    Returns non-sensitive runtime configuration for the admin dashboard.
    Never expose SECRET_KEY or database credentials.
    """
    return {
        "rag": {
            "collection_name": get_setting_value("COLLECTION_NAME", "legal_algeria"),
            "chroma_path": get_setting_value("CHROMA_PATH", "./data/chroma_db"),
            "embedding_model": get_setting_value("EMBEDDING_MODEL", "BAAI/bge-m3"),
            "reranker_model": get_setting_value("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            "llm_model": get_setting_value("LLM_MODEL", "mistral-small3.1:latest"),
            "ollama_host": get_setting_value("OLLAMA_HOST", "http://127.0.0.1:11434"),
        },
        "generation": {
            "rag_num_ctx": get_int_config("RAG_NUM_CTX", 32768),
            "rag_num_predict": get_int_config("RAG_NUM_PREDICT", 800),
            "rag_temperature": get_float_config("RAG_TEMPERATURE", 0.0),
            "rag_think": get_bool_config("RAG_THINK", False),
        },
        "retrieval": {
            "rag_top_k_retrieve": get_int_config("RAG_TOP_K_RETRIEVE", 30),
            "rag_top_k_rerank": get_int_config("RAG_TOP_K_RERANK", 5),
        },
        "vision": {
            "vision_model": get_setting_value(
                "VISION_MODEL",
                get_setting_value(
                    "VISION_TABLE_MODEL",
                    get_setting_value("LLM_MODEL", "mistral-small3.1:latest"),
                ),
            ),
            "vision_table_model": get_setting_value("VISION_TABLE_MODEL", "mistral-small3.1:latest"),
            "use_pdf_vision_for_tables": get_bool_config("USE_PDF_VISION_FOR_TABLES", True),
            "vision_max_pages": get_int_config("VISION_MAX_PAGES", 3),
            "vision_page_zoom": get_float_config("VISION_PAGE_ZOOM", 3.0),
            "vision_num_ctx": get_int_config("VISION_NUM_CTX", 32768),
            "vision_num_predict": get_int_config("VISION_NUM_PREDICT", 800),
        },
        "documents": {
            "pdf_path": get_setting_value("PDF_PATH", "./data/pdf"),
            "pdf_old_path": get_setting_value("PDF_OLD_PATH", "./data/pdf_old"),
        },
        "security": {
            "algorithm": get_setting_value("ALGORITHM", "HS256"),
            "access_token_expire_minutes": get_int_config("ACCESS_TOKEN_EXPIRE_MINUTES", 10080),
            "secret_key_configured": bool(get_setting_value("SECRET_KEY", "")),
        },
        "hidden": [
            "SECRET_KEY",
            "SQLALCHEMY_DATABASE_URL",
        ],
    }


# ==============================================================================
# DASHBOARD STATS + HEALTH
# ==============================================================================

@router.get("/")
def get_dashboard_stats(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    try:
        now = datetime.utcnow()
        period_start = now - timedelta(days=days)
        year_start = now - timedelta(days=365)

        # ============================================================
        # Basic counts
        # ============================================================
        total_users = db.query(User).count()
        total_sessions = db.query(ChatSession).count()
        total_messages = db.query(Message).count()

        liked_messages = db.query(Message).filter(Message.feedback == "like").count()
        disliked_messages = db.query(Message).filter(Message.feedback == "dislike").count()
        feedback_total = liked_messages + disliked_messages

        # ============================================================
        # Period counts
        # ============================================================
        new_users_period = 0

        if has_field(User, "created_at"):
            new_users_period = (
                db.query(User)
                .filter(User.created_at >= period_start)
                .count()
            )

        sessions_period = 0

        if has_field(ChatSession, "created_at"):
            sessions_period = (
                db.query(ChatSession)
                .filter(ChatSession.created_at >= period_start)
                .count()
            )

        messages_period = 0

        if has_field(Message, "created_at"):
            messages_period = (
                db.query(Message)
                .filter(Message.created_at >= period_start)
                .count()
            )

        # ============================================================
        # Active users
        # ============================================================
        active_users_period = 0
        message_session_fk = get_session_fk_field()

        if (
            message_session_fk is not None
            and has_field(ChatSession, "id")
            and has_field(ChatSession, "user_id")
            and has_field(Message, "created_at")
        ):
            active_users_period = (
                db.query(func.count(distinct(ChatSession.user_id)))
                .select_from(Message)
                .join(ChatSession, message_session_fk == ChatSession.id)
                .filter(Message.created_at >= period_start)
                .scalar()
                or 0
            )

        elif has_field(Message, "user_id") and has_field(Message, "created_at"):
            active_users_period = (
                db.query(func.count(distinct(Message.user_id)))
                .select_from(Message)
                .filter(Message.created_at >= period_start)
                .scalar()
                or 0
            )

        elif has_field(ChatSession, "user_id") and has_field(ChatSession, "created_at"):
            active_users_period = (
                db.query(func.count(distinct(ChatSession.user_id)))
                .select_from(ChatSession)
                .filter(ChatSession.created_at >= period_start)
                .scalar()
                or 0
            )

        # ============================================================
        # Message role split
        # ============================================================
        user_messages = None
        assistant_messages = None

        if has_field(Message, "role"):
            user_messages = db.query(Message).filter(Message.role == "user").count()
            assistant_messages = db.query(Message).filter(Message.role == "assistant").count()

        # ============================================================
        # Daily traffic
        # ============================================================
        daily_traffic = []

        if has_field(Message, "created_at"):
            daily_rows = (
                db.query(
                    func.strftime("%Y-%m-%d", Message.created_at).label("date"),
                    func.count(Message.id).label("message_count"),
                )
                .filter(Message.created_at >= period_start)
                .group_by(func.strftime("%Y-%m-%d", Message.created_at))
                .order_by(func.strftime("%Y-%m-%d", Message.created_at))
                .all()
            )

            daily_raw = {
                row.date: row.message_count
                for row in daily_rows
            }

            daily_traffic = fill_daily_traffic(daily_raw, days)

        # ============================================================
        # Monthly traffic
        # ============================================================
        monthly_traffic = []

        if has_field(Message, "created_at"):
            monthly_rows = (
                db.query(
                    func.strftime("%Y-%m", Message.created_at).label("month"),
                    func.count(Message.id).label("message_count"),
                )
                .filter(Message.created_at >= year_start)
                .group_by(func.strftime("%Y-%m", Message.created_at))
                .order_by(func.strftime("%Y-%m", Message.created_at))
                .all()
            )

            monthly_raw = {
                row.month: row.message_count
                for row in monthly_rows
            }

            monthly_traffic = fill_monthly_traffic(monthly_raw, 12)

        # ============================================================
        # Assistant messages in period for health
        # ============================================================
        assistant_period_query = db.query(Message)

        if has_field(Message, "role"):
            assistant_period_query = assistant_period_query.filter(Message.role == "assistant")

        if has_field(Message, "created_at"):
            assistant_period_query = assistant_period_query.filter(Message.created_at >= period_start)

        assistant_period_messages = assistant_period_query.all()

        health_metrics_rows = []
        assistant_texts = []
        daily_latency_raw = defaultdict(list)

        for message in assistant_period_messages:
            parts = getattr(message, "parts", None)

            text_value = extract_text_from_parts(parts)

            if text_value:
                assistant_texts.append(text_value)

            metrics = extract_metrics_from_parts(parts)

            if not metrics:
                continue

            created_at = getattr(message, "created_at", None)

            if created_at is not None and hasattr(created_at, "strftime"):
                day_key = created_at.strftime("%Y-%m-%d")
            else:
                day_key = None

            try:
                total_seconds = float(metrics.get("total_duration_seconds") or 0)
            except Exception:
                total_seconds = 0.0

            if total_seconds <= 0:
                continue

            try:
                first_token_seconds = round(float(metrics.get("time_to_first_token_ms") or 0) / 1000, 2)
            except Exception:
                first_token_seconds = 0.0

            row = {
                "message_id": getattr(message, "id", None),
                "created_at": serialize_datetime(created_at),
                "total_duration_seconds": total_seconds,
                "time_to_first_token_seconds": first_token_seconds,
                "source_count": int(metrics.get("source_count") or 0),
                "chunk_count": int(metrics.get("chunk_count") or 0),
                "answer_chars": int(metrics.get("answer_chars") or 0),
                "refused": bool(metrics.get("refused", False)),
                "model_id": metrics.get("model_id"),
            }

            health_metrics_rows.append(row)

            if day_key:
                daily_latency_raw[day_key].append(total_seconds)

        total_durations = [
            row["total_duration_seconds"]
            for row in health_metrics_rows
            if row["total_duration_seconds"] > 0
        ]

        first_token_durations = [
            row["time_to_first_token_seconds"]
            for row in health_metrics_rows
            if row["time_to_first_token_seconds"] > 0
        ]

        source_counts = [
            row["source_count"]
            for row in health_metrics_rows
        ]

        answer_lengths = [
            row["answer_chars"]
            for row in health_metrics_rows
        ]

        slow_threshold_seconds = get_float_config("HEALTH_SLOW_QUERY_SECONDS", 60.0)

        slow_queries = [
            row for row in health_metrics_rows
            if row["total_duration_seconds"] >= slow_threshold_seconds
        ]

        recent_slow_queries = sorted(
            slow_queries,
            key=lambda x: x.get("created_at") or "",
            reverse=True,
        )[:5]

        assistant_answer_count = len(assistant_texts)

        no_answer_count_from_parts = sum(
            1 for text in assistant_texts
            if NO_ANSWER_PHRASE in text
        )

        health = {
            "tracked_answers": len(health_metrics_rows),
            "slow_threshold_seconds": slow_threshold_seconds,

            "avg_total_duration_seconds": avg(total_durations),
            "p50_total_duration_seconds": percentile(total_durations, 50),
            "p95_total_duration_seconds": percentile(total_durations, 95),
            "max_total_duration_seconds": max(total_durations) if total_durations else 0.0,

            "avg_time_to_first_token_seconds": avg(first_token_durations),
            "p95_time_to_first_token_seconds": percentile(first_token_durations, 95),

            "slow_queries_count": len(slow_queries),
            "slow_queries_rate": safe_percentage(len(slow_queries), len(health_metrics_rows)),

            "avg_source_count": avg(source_counts),
            "avg_answer_chars": avg(answer_lengths),

            "daily_latency": build_daily_latency(daily_latency_raw, days),
            "recent_slow_queries": recent_slow_queries,
        }

        # ============================================================
        # Most disliked messages
        # ============================================================
        most_disliked_messages = []

        disliked_query = db.query(Message).filter(Message.feedback == "dislike")

        if has_field(Message, "created_at"):
            disliked_query = disliked_query.order_by(Message.created_at.desc())

        disliked_rows = disliked_query.limit(5).all()

        for message in disliked_rows:
            most_disliked_messages.append({
                "id": getattr(message, "id", None),
                "preview": get_message_preview(message),
                "created_at": serialize_datetime(getattr(message, "created_at", None)),
                "feedback": getattr(message, "feedback", None),
            })

        # ============================================================
        # Rates
        # ============================================================
        avg_messages_per_session = (
            round(total_messages / total_sessions, 2)
            if total_sessions
            else 0.0
        )

        feedback_rate = safe_percentage(feedback_total, total_messages)
        satisfaction_rate = safe_percentage(liked_messages, feedback_total)

        return {
            "period": {
                "days": days,
                "from": period_start.isoformat(),
                "to": now.isoformat(),
            },
            "overview": {
                "total_users": total_users,
                "new_users_period": new_users_period,
                "active_users_period": active_users_period,

                "total_sessions": total_sessions,
                "sessions_period": sessions_period,

                "total_messages": total_messages,
                "messages_period": messages_period,

                "user_messages": user_messages,
                "assistant_messages": assistant_messages,

                "avg_messages_per_session": avg_messages_per_session,
            },
            "feedback": {
                "liked_messages": liked_messages,
                "disliked_messages": disliked_messages,
                "feedback_total": feedback_total,
                "feedback_rate": feedback_rate,
                "satisfaction_rate": satisfaction_rate,
                "most_disliked_messages": most_disliked_messages,
            },
            "traffic": {
                "daily": daily_traffic,
                "monthly": monthly_traffic,
            },
            "rag_quality": {
                "no_answer_count": no_answer_count_from_parts,
                "no_answer_rate": safe_percentage(no_answer_count_from_parts, assistant_answer_count),
            },
            "health": health,
        }

    except Exception:
        logger.exception("[STATS API ERROR] Could not retrieve dashboard statistics")

        raise HTTPException(
            status_code=500,
            detail="Could not retrieve statistics",
        )