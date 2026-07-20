"""
SQLAlchemy models for chat sessions and messages.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, Integer, Float, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from src.db.database import Base


def generate_id():
    return str(uuid.uuid4())


class Session(Base):
    """
    A research session — groups multiple messages about a topic.
    Each session is tied to a document (or searches all documents).

    Example:
        Session(
            title="Termination and notice periods",
            doc_id="saudi_labor_law_abc123",
            doc_name="Saudi Labor Law.pdf"
        )
    """
    __tablename__ = "sessions"

    id          = Column(String, primary_key=True, default=generate_id)
    title       = Column(String(200), nullable=False)
    doc_id      = Column(String(200), nullable=True)   # None = search all docs
    doc_name    = Column(String(200), nullable=True)   # Display name
    created_at  = Column(DateTime, default=datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    message_count = Column(Integer, default=0)

    messages = relationship(
        "Message",
        back_populates="session",
        order_by="Message.created_at",
        cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "id":            self.id,
            "title":         self.title,
            "doc_id":        self.doc_id,
            "doc_name":      self.doc_name,
            "created_at":    self.created_at.isoformat(),
            "updated_at":    self.updated_at.isoformat(),
            "message_count": self.message_count,
        }


class Message(Base):
    """
    A single message in a session.
    Role is either 'user' (question) or 'assistant' (answer).

    User messages contain only the query text.
    Assistant messages contain the full agent response.
    """
    __tablename__ = "messages"

    id          = Column(String, primary_key=True, default=generate_id)
    session_id  = Column(String, ForeignKey("sessions.id"), nullable=False)
    role        = Column(String(20), nullable=False)   # 'user' or 'assistant'
    content     = Column(Text, nullable=False)          # The message text

    # Agent metadata (only for assistant messages)
    query_type          = Column(String(50),  nullable=True)
    citations           = Column(Text,        nullable=True)   # JSON list stored as string
    groundedness_score  = Column(Float,       nullable=True)
    citation_score      = Column(Float,       nullable=True)
    relevance_score     = Column(Float,       nullable=True)
    critique_passed     = Column(Boolean,     nullable=True)
    regeneration_count  = Column(Integer,     nullable=True)
    chunks_used         = Column(Integer,     nullable=True)

    created_at  = Column(DateTime, default=datetime.utcnow)
    session     = relationship("Session", back_populates="messages")

    def to_dict(self):
        import json
        citations = []
        if self.citations:
            try:
                citations = json.loads(self.citations)
            except Exception:
                citations = []
        return {
            "id":                  self.id,
            "session_id":          self.session_id,
            "role":                self.role,
            "content":             self.content,
            "query_type":          self.query_type,
            "citations":           citations,
            "groundedness_score":  self.groundedness_score,
            "citation_score":      self.citation_score,
            "relevance_score":     self.relevance_score,
            "critique_passed":     self.critique_passed,
            "regeneration_count":  self.regeneration_count,
            "chunks_used":         self.chunks_used,
            "created_at":          self.created_at.isoformat(),
        }
