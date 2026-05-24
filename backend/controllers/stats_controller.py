from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from core.database import get_db
from core.models import User, ChatSession, Message
from services.security import get_admin_user
from datetime import datetime, timedelta

router = APIRouter(prefix="/stats", tags=["Admin Statistics"])



@router.get("/")
def get_dashboard_stats(db: Session = Depends(get_db), admin: User = Depends(get_admin_user)):
    try:
        # Basic Counts
        total_users = db.query(User).count()
        total_sessions = db.query(ChatSession).count()
        total_messages = db.query(Message).count()
        
        liked_messages = db.query(Message).filter(Message.feedback == "like").count()
        disliked_messages = db.query(Message).filter(Message.feedback == "dislike").count()

        # Daily Traffic (Last 30 Days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        
        # SQLite uses strftime for date formatting
        daily_traffic_query = (
            db.query(
                func.strftime('%Y-%m-%d', Message.created_at).label('date'),
                func.count(Message.id).label('count')
            )
            .filter(Message.created_at >= thirty_days_ago)
            .group_by(func.strftime('%Y-%m-%d', Message.created_at))
            .order_by(func.strftime('%Y-%m-%d', Message.created_at))
            .all()
        )
        
        daily_traffic = [{"date": row.date, "count": row.count} for row in daily_traffic_query]

        # Monthly Traffic (Last 12 Months)
        one_year_ago = datetime.utcnow() - timedelta(days=365)
        monthly_traffic_query = (
            db.query(
                func.strftime('%Y-%m', Message.created_at).label('month'),
                func.count(Message.id).label('count')
            )
            .filter(Message.created_at >= one_year_ago)
            .group_by(func.strftime('%Y-%m', Message.created_at))
            .order_by(func.strftime('%Y-%m', Message.created_at))
            .all()
        )
        
        monthly_traffic = [{"month": row.month, "count": row.count} for row in monthly_traffic_query]

        return {
            "overview": {
                "total_users": total_users,
                "total_sessions": total_sessions,
                "total_messages": total_messages,
                "liked_messages": liked_messages,
                "disliked_messages": disliked_messages
            },
            "traffic": {
                "daily": daily_traffic,
                "monthly": monthly_traffic
            }
        }
    except Exception as e:
        print(f"[STATS API ERROR] {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve statistics")
