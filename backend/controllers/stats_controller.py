from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, distinct
from core.database import get_db
from core.models import User, ChatSession, Message
from services.security import get_admin_user
from datetime import datetime, timedelta
import logging

router = APIRouter(prefix="/stats", tags=["Admin Statistics"])

logger = logging.getLogger(__name__)

NO_ANSWER_PHRASE = "Je suis désolé, je n'ai pas la réponse"


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


def get_message_text_field():
    """
    Detects the text/content column in the Message model.
    Adjust the list if your Message model uses another field name.
    """
    possible_fields = ["content", "text", "message", "body", "answer"]

    for field in possible_fields:
        if has_field(Message, field):
            return getattr(Message, field)

    return None


def get_session_fk_field():
    """
    Detects the foreign key from Message to ChatSession.
    Adjust the list if your Message model uses another field name.
    """
    possible_fields = ["chat_session_id", "session_id", "conversation_id"]

    for field in possible_fields:
        if has_field(Message, field):
            return getattr(Message, field)

    return None


def fill_daily_traffic(raw_data: dict, days: int):
    today = datetime.utcnow().date()
    start_day = today - timedelta(days=days - 1)

    result = []

    for i in range(days):
        current_day = start_day + timedelta(days=i)
        date_str = current_day.strftime("%Y-%m-%d")

        result.append({
            "date": date_str,
            "count": raw_data.get(date_str, 0)
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
            "count": raw_data.get(month_key, 0)
        }
        for month_key in month_keys
    ]


@router.get("/")
def get_dashboard_stats(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user)
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
        # Daily traffic - SQLite
        # ============================================================
        daily_traffic = []

        if has_field(Message, "created_at"):
            daily_rows = (
                db.query(
                    func.strftime("%Y-%m-%d", Message.created_at).label("date"),
                    func.count(Message.id).label("message_count")
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
        # Monthly traffic - SQLite
        # ============================================================
        monthly_traffic = []

        if has_field(Message, "created_at"):
            monthly_rows = (
                db.query(
                    func.strftime("%Y-%m", Message.created_at).label("month"),
                    func.count(Message.id).label("message_count")
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
        # RAG quality: no-answer count
        # ============================================================
        no_answer_count = 0
        no_answer_rate = 0.0

        message_text_field = get_message_text_field()

        if message_text_field is not None:
            no_answer_count = (
                db.query(Message)
                .filter(message_text_field.ilike(f"%{NO_ANSWER_PHRASE}%"))
                .count()
            )

            no_answer_rate = safe_percentage(no_answer_count, total_messages)

        # ============================================================
        # Most disliked messages
        # ============================================================
        most_disliked_messages = []

        if message_text_field is not None:
            disliked_query = db.query(Message).filter(Message.feedback == "dislike")

            if has_field(Message, "created_at"):
                disliked_query = disliked_query.order_by(Message.created_at.desc())

            disliked_rows = disliked_query.limit(5).all()

            for msg in disliked_rows:
                text_value = getattr(msg, message_text_field.key, "") or ""

                most_disliked_messages.append({
                    "id": getattr(msg, "id", None),
                    "preview": text_value[:250],
                    "created_at": serialize_datetime(getattr(msg, "created_at", None)),
                    "feedback": getattr(msg, "feedback", None)
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
                "to": now.isoformat()
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

                "avg_messages_per_session": avg_messages_per_session
            },
            "feedback": {
                "liked_messages": liked_messages,
                "disliked_messages": disliked_messages,
                "feedback_total": feedback_total,
                "feedback_rate": feedback_rate,
                "satisfaction_rate": satisfaction_rate,
                "most_disliked_messages": most_disliked_messages
            },
            "traffic": {
                "daily": daily_traffic,
                "monthly": monthly_traffic
            },
            "rag_quality": {
                "no_answer_count": no_answer_count,
                "no_answer_rate": no_answer_rate
            }
        }

    except Exception:
        logger.exception("[STATS API ERROR] Could not retrieve dashboard statistics")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve statistics"
        )