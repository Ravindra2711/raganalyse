from datetime import datetime
import logging
from fastapi import APIRouter, Depends, HTTPException, Query, Body, Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from src.db.init_db import session_factory
from src.db.schemas.models import ModelUsage
from src.managers.ai_manager import AI_Manager
from src.managers.chat_manager import ChatManager
from src.managers.user_manager import get_current_user, User
from src.schemas.chat_schema import *
from src.orchestration.autogen_orchestrator import Orchestrator
from src.utils.logger import Logger
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize logger with console logging disabled
logger = Logger("chat_routes", see_time=True, console_log=False)

# Initialize router
router = APIRouter(prefix="/chats", tags=["chats"])

# Initialize chat manager
chat_manager = ChatManager(db_url=os.getenv("DATABASE_URL"))

# Initialize AI manager
ai_manager = AI_Manager()

# Orchestrator
orchestrator = Orchestrator()


# Routes
@router.post("/", response_model=ChatResponse)
async def create_chat(chat_create: ChatCreate):
    """Create a new chat session"""
    try:
        chat = chat_manager.create_chat(chat_create.user_id)
        return chat
    except Exception as e:
        logger.log_message(f"Error creating chat: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=f"Failed to create chat: {str(e)}")

@router.post("/{chat_id}/messages", response_model=MessageResponse)
async def add_message(chat_id: int, message: MessageCreate, user_id: Optional[int] = None):
    """Add a message to a chat"""
    try:
        result = chat_manager.add_message(chat_id, message.content, message.sender, user_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.log_message(f"Error adding message: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=f"Failed to add message: {str(e)}")

@router.get("/{chat_id}", response_model=ChatDetailResponse)
async def get_chat(chat_id: int, user_id: Optional[int] = None):
    """Get a chat by ID with all messages"""
    try:
        chat = chat_manager.get_chat(chat_id, user_id)
        return chat
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.log_message(f"Error retrieving chat: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat: {str(e)}")

@router.get("/", response_model=List[ChatResponse])
async def get_chats(
    user_id: Optional[int] = None,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get recent chats, optionally filtered by user_id"""
    try:
        chats = chat_manager.get_user_chats(user_id, limit, offset)
        return chats
    except Exception as e:
        logger.log_message(f"Error retrieving chats: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chats: {str(e)}")

@router.delete("/{chat_id}")
async def delete_chat(chat_id: int, user_id: Optional[int] = None):
    """Delete a chat and all its messages while preserving model usage data"""
    try:
        # Delete the chat using the updated chat_manager method
        # which now preserves ModelUsage records
        success = chat_manager.delete_chat(chat_id, user_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Chat with ID {chat_id} not found or access denied")
        return {"message": f"Chat {chat_id} deleted successfully", "preserved_model_usage": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.log_message(f"Error deleting chat: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=f"Failed to delete chat: {str(e)}")


@router.post("/users", response_model=dict)
async def create_or_get_user(user_info: UserInfo):
    """Create a new user or get an existing one by email"""
    try:
        user = chat_manager.get_or_create_user(
            username=user_info.username,
            email=user_info.email
        )
        return user
    except Exception as e:
        logger.log_message(f"Error creating/getting user: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=f"Failed to process user: {str(e)}")

@router.put("/{chat_id}", response_model=ChatResponse)
async def update_chat(chat_id: int, chat_update: ChatUpdate):
    """Update a chat's title or user_id"""
    try:
        chat = chat_manager.update_chat(
            chat_id=chat_id,
            title=chat_update.title,
            user_id=chat_update.user_id
        )
        return chat
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.log_message(f"Error updating chat: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=f"Failed to update chat: {str(e)}")

@router.post("/cleanup-empty", response_model=dict)
async def cleanup_empty_chats(request: ChatCreate):
    """Delete empty chats for a user"""
    try:
        deleted_count = chat_manager.delete_empty_chats(request.user_id, request.is_admin)
        return {"message": f"Deleted {deleted_count} empty chats"}
    except Exception as e:
        logger.log_message(f"Error cleaning up empty chats: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=f"Failed to clean up empty chats: {str(e)}")


def _session_df_info(request: Request) -> Dict[str, Any]:
    try:
        app_state = request.app.state
        # get session id from SessionManager; fall back to header
        session_id = request.headers.get("X-Session-ID")
        if not session_id:
            return {"name": "dataset", "columns": []}
        ss = app_state.get_session_state(session_id)
        datasets = ss.get("datasets", {})
        if not datasets:
            return {"name": ss.get("name", "dataset"), "columns": []}
        first_name = next(iter(datasets.keys()))
        df = datasets[first_name]
        cols = list(getattr(df, "columns", []))
        return {"name": first_name, "columns": cols, "session_id": session_id}
    except Exception:
        return {"name": "dataset", "columns": []}


@router.post("/{chat_id}/query", response_model=MessageResponse)
async def query_chat(
    chat_id: int,
    payload: ChatQueryRequest,
    request: Request,
    user_id: Optional[int] = None,
):
    """Query the chat using the new Autogen-based orchestrator (minimal path)."""
    try:
        # Save user message
        user_msg = chat_manager.add_message(chat_id, payload.content, "user", user_id)

        # Build df info from session
        df_info = _session_df_info(request)
        session_id = df_info.get("session_id") or request.headers.get("X-Session-ID", "default")

        # Orchestrate
        answer_text, meta = orchestrator.answer_query(
            chat_id=chat_id,
            session_id=session_id,
            df_info={"name": df_info.get("name", "dataset"), "columns": df_info.get("columns", [])},
            query=payload.content,
            model_name=payload.model_name or "gpt-5-mini",
            temperature=float(payload.temperature or 0.1),
            max_tokens=int(payload.max_tokens or 800),
            embedding_model=payload.embedding_model or "text-embedding-3-small",
            k=int(payload.k or 8),
            offset=int(payload.offset or 0),
        )

        # Save AI response
        ai_msg = chat_manager.add_message(chat_id, answer_text, "ai", user_id)

        # Record usage
        ai_manager.save_usage_to_db(
            user_id=user_id,
            chat_id=chat_id,
            model_name=payload.model_name or "gpt-5-mini",
            provider=meta.get("provider", "Unknown"),
            prompt_tokens=meta.get("prompt_tokens", 0),
            completion_tokens=meta.get("completion_tokens", 0),
            total_tokens=meta.get("total_tokens", 0),
            query_size=len(payload.content or ""),
            response_size=len(answer_text or ""),
            cost=ai_manager.calculate_cost(payload.model_name or "gpt-5-mini", meta.get("prompt_tokens", 0), meta.get("completion_tokens", 0)),
            request_time_ms=meta.get("request_time_ms", 0),
            is_streaming=False,
        )

        return ai_msg
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.log_message(f"Error processing query: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")
