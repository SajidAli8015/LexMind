"""
Sessions Router
Handles research session creation, listing, and chat messages
with full conversation history.
"""

import json
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session as DBSession
from loguru import logger

from src.api.schemas import (
    CreateSessionRequest, SendMessageRequest,
    SessionResponse, SessionDetailResponse, MessageResponse
)
from src.db.database import get_db
from src.db.models import Session, Message

router = APIRouter()


@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    request: CreateSessionRequest,
    db: DBSession = Depends(get_db),
):
    session = Session(
        title=request.title,
        doc_id=request.doc_id,
        doc_name=request.doc_name,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return SessionResponse(**session.to_dict())


@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(db: DBSession = Depends(get_db)):
    sessions = (
        db.query(Session)
        .order_by(Session.updated_at.desc())
        .all()
    )
    return [SessionResponse(**s.to_dict()) for s in sessions]


@router.get(
    "/sessions/recent-queries",
    summary="Get recent queries across all sessions",
)
async def get_recent_queries(
    limit: int = 10,
    db: DBSession = Depends(get_db),
):
    """
    Return the most recent user messages across all sessions.
    Used by the Dashboard to show query history.
    """
    from src.db.models import Message
    messages = (
        db.query(Message)
        .filter(Message.role == "user")
        .order_by(Message.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": m.id,
            "session_id": m.session_id,
            "content": m.content,
            "created_at": m.created_at.isoformat(),
        }
        for m in messages
    ]


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: str,
    db: DBSession = Depends(get_db),
):
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = [MessageResponse(**m.to_dict()) for m in session.messages]
    data = session.to_dict()
    data["messages"] = messages
    return SessionDetailResponse(**data)


@router.post("/sessions/{session_id}/message", response_model=MessageResponse)
async def send_message(
    session_id: str,
    request: SendMessageRequest,
    db: DBSession = Depends(get_db),
):
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Save user message
    user_msg = Message(
        session_id=session_id,
        role="user",
        content=request.content,
    )
    db.add(user_msg)
    db.commit()

    # Build conversation history from previous messages
    previous = session.messages[:-1]  # exclude the one we just added
    history_text = _build_history_context(previous[-6:])

    try:
        # Import inside function — prevents model loading at startup
        from src.graph.graph import run_query

        # ALWAYS use raw question for retrieval
        # Pass history separately for reasoning only
        result = run_query(
            query=request.content,
            doc_id=session.doc_id,
            conversation_history=history_text or None,
        )

        final_answer = result.get("final_answer") or result.get("answer") or ""

        if not final_answer:
            raise HTTPException(500, "Agent did not produce an answer")

        # Save assistant response
        assistant_msg = Message(
            session_id=session_id,
            role="assistant",
            content=final_answer,
            query_type=result.get("query_type"),
            citations=json.dumps(result.get("citations") or []),
            groundedness_score=result.get("groundedness_score"),
            citation_score=result.get("citation_score"),
            relevance_score=result.get("relevance_score"),
            critique_passed=result.get("critique_passed"),
            regeneration_count=result.get("regeneration_count") or 0,
            chunks_used=len(result.get("retrieved_chunks") or []),
        )
        db.add(assistant_msg)

        # Update session
        session.message_count = len(session.messages) + 1
        session.updated_at = datetime.utcnow()
        if session.title == "New research session" and len(session.messages) == 1:
            session.title = _generate_title(request.content)

        db.commit()
        db.refresh(assistant_msg)

        return MessageResponse(**assistant_msg.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Message failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/sessions/{session_id}/export",
    summary="Export session as markdown",
)
async def export_session(
    session_id: str,
    db: DBSession = Depends(get_db),
):
    """
    Export a research session as a markdown document.
    Returns a markdown string with all Q&A and citations.
    """
    from fastapi.responses import Response
    from datetime import datetime

    session = db.query(Session).filter(
        Session.id == session_id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    lines = []
    lines.append(f"# {session.title}")
    lines.append(f"")
    if session.doc_name:
        lines.append(f"**Document:** {session.doc_name}")
    lines.append(
        f"**Date:** {session.created_at.strftime('%B %d, %Y')}"
    )
    lines.append(f"**Messages:** {session.message_count}")
    lines.append(f"")
    lines.append("---")
    lines.append("")

    for msg in session.messages:
        if msg.role == "user":
            lines.append(f"## Q: {msg.content}")
            lines.append("")
        else:
            lines.append(f"**Answer:**")
            lines.append("")
            lines.append(msg.content)
            lines.append("")
            import json
            citations = []
            if msg.citations:
                try:
                    citations = json.loads(msg.citations)
                except Exception:
                    pass
            if citations:
                lines.append(
                    f"*Cited: {', '.join(citations)}*"
                )
                lines.append("")
            scores = []
            if msg.groundedness_score is not None:
                scores.append(
                    f"Grounded {int(msg.groundedness_score * 100)}%"
                )
            if msg.relevance_score is not None:
                scores.append(
                    f"Relevant {int(msg.relevance_score * 100)}%"
                )
            if scores:
                lines.append(f"*Quality: {' · '.join(scores)}*")
                lines.append("")
            lines.append("---")
            lines.append("")

    markdown = "\n".join(lines)
    filename = session.title.replace(" ", "_")[:50] + ".md"

    return Response(
        content=markdown,
        media_type="text/markdown",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    db: DBSession = Depends(get_db),
):
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    db.delete(session)
    db.commit()
    return {"success": True}


def _build_history_context(messages: list) -> str:
    if not messages:
        return ""
    lines = []
    for msg in messages:
        if msg.role == "user":
            lines.append(f"User: {msg.content}")
        else:
            lines.append(f"Assistant: {msg.content}")
    return "\n".join(lines)


def _generate_title(question: str) -> str:
    q = question.strip()
    if len(q) <= 60:
        return q.rstrip("?")
    truncated = q[:60]
    last_space = truncated.rfind(" ")
    if last_space > 30:
        truncated = truncated[:last_space]
    return truncated.rstrip("?") + "..."
