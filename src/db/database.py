"""
SQLite database setup for LexMind chat sessions.
Uses SQLAlchemy for ORM and connection management.
"""

import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from loguru import logger

DB_PATH = os.getenv("SESSIONS_DB_PATH", "./data/sessions.db")
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """
    Dependency that provides a database session.
    Used with FastAPI's Depends() to inject DB sessions into routes.
    Automatically closes the session when the request is done.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Create all database tables if they don't exist.
    Called once on application startup.
    """
    from src.db.models import Session, Message
    Base.metadata.create_all(bind=engine)
    logger.info(f"Database initialised at {DB_PATH}")
