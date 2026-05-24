from sqlalchemy import Column, String, Text, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from core.database import Base

def generate_uuid():
    return str(uuid.uuid4())
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    title = Column(String, default="Nouvelle Conversation")
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    archived = Column(Boolean, default=False)

    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("chat_sessions.id"))
    model_id = Column(String, nullable=True)
    role = Column(String)           
    parts = Column(JSON)            
    attachments = Column(JSON)      
    created_at = Column(DateTime, default=datetime.utcnow)
    feedback = Column(String, nullable=True)

    session = relationship("ChatSession", back_populates="messages")
    reports = relationship("Report", back_populates="message", cascade="all, delete-orphan")


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=generate_uuid)
    first_name = Column(String)
    last_name = Column(String)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="utilisateur")
    created_at = Column(DateTime, default=datetime.utcnow)
    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="user", cascade="all, delete-orphan")
class Report(Base):
    __tablename__ = "reports"

    id = Column(String, primary_key=True, default=generate_uuid)
    message_id = Column(String, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    reason = Column(String, nullable=False)      # Ex: "hallucination", "loi_obsolete", "contexte_invalide"
    details = Column(Text, nullable=True)        # Commentaire libre soumis par l'utilisateur
    status = Column(String, default="en_attente") # "en_attente", "traite", "rejete"
    created_at = Column(DateTime, default=datetime.utcnow)

    message = relationship("Message", back_populates="reports")
    user = relationship("User", back_populates="reports")