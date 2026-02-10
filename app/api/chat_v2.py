"""
UCSI Buddy Chat API v2 - Refactored with Hybrid Intent Classifier

Clean architecture:
1. Intent Classification (hybrid: keyword guard + vector + LLM)
2. Data Retrieval (based on category)
3. LLM Response Generation (always)
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request

from app.config import Config
from app.core.session import (
    append_conversation_message,
    get_conversation_history,
    high_security_sessions,
)
from app.engines.intent_classifier import intent_classifier
from app.engines.ai_engine_async import ai_engine_async
from app.engines.db_engine_async import db_engine_async
from app.engines.rag_engine_async import rag_engine_async
from app.engines.response_validator import (
    response_validator,
    detect_prompt_injection,
    sanitize_user_input,
)
from app.api.chat_helpers import (
    _build_suggestions,
    _capability_smalltalk_response,
    _capability_suggestions,
    _compose_no_data_response,
    _detect_language,
    _format_personal_info,
    _personal_info_suggestions,
    _user_display_name,
    _user_student_number,
)
from app.schemas import ChatRequest, FeedbackRequest
from app.utils.auth_utils import decode_access_token

router = APIRouter()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _resolve_session(user: Optional[dict], body: ChatRequest) -> tuple:
    """Resolve session key and conversation ID."""
    import secrets
    student_number = _user_student_number(user)
    if student_number:
        return f"user:{student_number}", student_number, False
    cid = body.conversation_id
    if cid:
        return f"guest:{cid}", cid, False
    new_id = secrets.token_hex(8)
    return f"guest:{new_id}", new_id, True


def _extract_rag_meta(rag_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract metadata from RAG result."""
    result = rag_result or {}
    confidence = float(result.get("confidence") or 0.0)
    sources = result.get("sources", [])
    if not isinstance(sources, list):
        sources = []
    return {
        "context": str(result.get("context") or ""),
        "has_relevant_data": bool(result.get("has_relevant_data")),
        "confidence": max(0.0, min(confidence, 1.2)),
        "sources": [str(s) for s in sources[:8]],
    }


async def _log_rag_miss(
    conversation_id: str,
    user_message: str,
    query: str,
    rag_meta: Dict[str, Any],
    reason: str,
) -> None:
    """Log unanswered questions for future improvement."""
    payload = {
        "timestamp": datetime.now().isoformat(),
        "conversation_id": conversation_id,
        "question": user_message,
        "search_query": query,
        "confidence": rag_meta.get("confidence", 0.0),
        "sources": rag_meta.get("sources", []),
        "reason": reason,
    }
    try:
        await db_engine_async.log_unanswered(payload)
    except Exception:
        pass


# =============================================================================
# ROUTE HANDLERS
# =============================================================================

async def handle_personal_query(
    user: Optional[dict],
    user_msg: str,
    user_lang: str,
    classification: Dict[str, Any],
    conversation_id: str,
) -> Dict[str, Any]:
    """Handle personal data queries (profile, grades)."""
    student_number = _user_student_number(user)

    # Check login
    if not student_number:
        return {
            "response": "Please login to access personal information.",
            "type": "login_hint",
            "conversation_id": conversation_id,
        }

    # Check high security for grades
    is_grade_query = classification.get("is_grade_query", False)
    expiry = high_security_sessions.get(student_number)
    high_security_active = bool(expiry and expiry >= datetime.now())

    if expiry and expiry < datetime.now():
        high_security_sessions.pop(student_number, None)

    if is_grade_query and not high_security_active:
        prompt_text = (
            "GPA/성적 정보를 보려면 비밀번호 인증이 필요해요."
            if user_lang == "ko"
            else "Please verify your password to access GPA/grade information."
        )
        return {
            "response": prompt_text,
            "type": "password_prompt",
            "conversation_id": conversation_id,
        }

    # Fetch student data
    student_data = await db_engine_async.get_student_by_number(student_number)

    if not student_data:
        return {
            "text": (
                "학생 정보를 찾지 못했어요."
                if user_lang == "ko"
                else "I could not find your student profile."
            ),
            "retrieval": {
                "route": "personal_db_direct",
                "used": True,
                "confidence": 0.0,
                "sources": ["MongoDB:Student"],
            },
        }

    # Format response
    response_text = _format_personal_info(
        student_data,
        user_msg,
        None,
        allow_sensitive=high_security_active,
    )

    return {
        "text": response_text,
        "suggestions": _personal_info_suggestions(user_lang, allow_sensitive=high_security_active),
        "retrieval": {
            "route": "personal_db_direct",
            "used": True,
            "confidence": 1.0,
            "sources": ["MongoDB:Student"],
        },
    }


async def handle_ucsi_query(
    user_msg: str,
    user_lang: str,
    classification: Dict[str, Any],
    conversation_id: str,
    history: List[Dict],
) -> Dict[str, Any]:
    """Handle UCSI domain queries using RAG."""
    search_term = classification.get("search_term") or user_msg

    # Check for learned response first
    learned_answer = await db_engine_async.search_learned_response(user_msg)
    if learned_answer:
        return {
            "text": learned_answer,
            "suggestions": [],
            "retrieval": {
                "route": "feedback_learned_qa",
                "used": True,
                "confidence": 1.0,
                "sources": ["MongoDB:LearnedQA"],
            },
        }

    # Check for bad response guard
    feedback_bad_guard = await db_engine_async.has_bad_response(user_msg)

    # Get RLHF policy
    rlhf_policy = {"has_signal": False}
    try:
        rlhf_policy = await db_engine_async.get_rlhf_policy(user_msg)
    except Exception:
        pass

    # Perform RAG search
    rag_result = await rag_engine_async.search_context(
        query=search_term,
        top_k=5,
    )
    rag_meta = _extract_rag_meta(rag_result)

    retrieval_meta = {
        "route": "rag",
        "used": True,
        "confidence": rag_meta["confidence"],
        "sources": rag_meta["sources"],
        "classification": {
            "intent": classification.get("intent"),
            "source": classification.get("source"),
        },
    }

    if feedback_bad_guard:
        retrieval_meta["feedback_guard"] = True

    if rlhf_policy.get("has_signal"):
        retrieval_meta["rlhf"] = {
            "has_signal": True,
            "strict_grounding": rlhf_policy.get("strict_grounding", False),
            "top_tags": rlhf_policy.get("top_tags", [])[:3],
        }

    # No relevant data found
    if not rag_meta["has_relevant_data"]:
        await _log_rag_miss(
            conversation_id=conversation_id,
            user_message=user_msg,
            query=search_term,
            rag_meta=rag_meta,
            reason="no_relevant_data",
        )
        return {
            "text": _compose_no_data_response(),
            "suggestions": [],
            "retrieval": retrieval_meta,
        }

    # Build context for LLM
    context_text = rag_meta["context"]

    if feedback_bad_guard:
        context_text = (
            "[FEEDBACK_GUARD] A previous answer was marked incorrect. "
            "Rely only on verified context.\n\n" + context_text
        )

    if rag_meta["confidence"] < float(Config.RAG_FAST_CONFIDENCE):
        context_text = (
            "[LOW_CONFIDENCE_CONTEXT] Some matches are weak. Avoid overclaiming.\n\n"
            + context_text
        )

    # Generate response with LLM
    ai_result = await ai_engine_async.process_message(
        user_message=user_msg,
        data_context=context_text,
        conversation_history=history,
        language=user_lang,
        rlhf_policy_hint=rlhf_policy.get("policy_hint"),
    )

    response_text = str(ai_result.get("response") or "I'm not sure how to respond.")
    suggestions = ai_result.get("suggestions", [])

    return {
        "text": response_text,
        "suggestions": suggestions[:3],
        "retrieval": retrieval_meta,
    }


async def handle_general_query(
    user_msg: str,
    user_lang: str,
    classification: Dict[str, Any],
    history: List[Dict],
) -> Dict[str, Any]:
    """Handle general knowledge queries (no RAG needed)."""

    # Generate response with LLM (no context)
    ai_result = await ai_engine_async.process_message(
        user_message=user_msg,
        data_context="",  # No context for general knowledge
        conversation_history=history,
        language=user_lang,
        query_scope_hint="general_person" if classification.get("intent") == "general_person" else None,
    )

    response_text = str(ai_result.get("response") or "I'm not sure how to respond.")
    suggestions = ai_result.get("suggestions", [])

    return {
        "text": response_text,
        "suggestions": suggestions[:3],
        "retrieval": {
            "route": "general_ai",
            "used": False,
            "confidence": classification.get("confidence", 0.5),
            "sources": [],
            "classification": {
                "intent": classification.get("intent"),
                "source": classification.get("source"),
            },
        },
    }


async def handle_capability_query(
    user_msg: str,
    user_lang: str,
) -> Dict[str, Any]:
    """Handle capability/smalltalk queries."""
    return {
        "text": _capability_smalltalk_response(user_msg),
        "suggestions": _capability_suggestions(user_lang),
        "retrieval": {
            "route": "capability_smalltalk",
            "used": False,
            "confidence": 1.0,
            "sources": [],
        },
    }


# =============================================================================
# MAIN CHAT ENDPOINT
# =============================================================================

@router.post("/chat")
async def chat(body: ChatRequest, request: Request):
    """
    Main chat endpoint with clean routing logic.

    Flow:
    1. Parse request & authenticate
    2. Classify intent (hybrid)
    3. Route to appropriate handler
    4. Return response
    """
    # [1] Parse request
    user = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        user = decode_access_token(auth_header.split(" ", 1)[1].strip())

    session_key, conversation_id, _ = _resolve_session(user, body)
    user_msg = (body.message or "").strip()

    if not user_msg:
        return {"error": "Message is required", "session_id": conversation_id}

    # Save user message to history
    append_conversation_message(session_key, "user", user_msg)
    history = get_conversation_history(session_key)
    user_lang = _detect_language(user_msg)

    # [2] Classify intent using hybrid classifier
    classification = await intent_classifier.classify(
        user_message=user_msg,
        conversation_history=history,
        language=user_lang,
        search_term=body.search_term,
    )

    category = classification.get("category", "unknown")

    # [3] Route to appropriate handler
    if category == "personal":
        result = await handle_personal_query(
            user=user,
            user_msg=user_msg,
            user_lang=user_lang,
            classification=classification,
            conversation_id=conversation_id,
        )

    elif category == "ucsi_domain":
        result = await handle_ucsi_query(
            user_msg=user_msg,
            user_lang=user_lang,
            classification=classification,
            conversation_id=conversation_id,
            history=history,
        )

    elif category == "capability":
        result = await handle_capability_query(
            user_msg=user_msg,
            user_lang=user_lang,
        )

    else:  # general_knowledge or unknown
        result = await handle_general_query(
            user_msg=user_msg,
            user_lang=user_lang,
            classification=classification,
            history=history,
        )

    # [4] Handle special response types (login_hint, password_prompt)
    if result.get("type") in ("login_hint", "password_prompt"):
        return {
            "response": result["response"],
            "type": result["type"],
            "conversation_id": conversation_id,
            "session_id": conversation_id,
        }

    # [5] Build final response
    response_payload = {
        "text": result.get("text", ""),
        "suggestions": result.get("suggestions", [])[:3],
        "retrieval": result.get("retrieval", {}),
    }

    # Add classification debug info in development
    if classification.get("debug"):
        response_payload["retrieval"]["classification_debug"] = classification["debug"]

    # Save to conversation history
    append_conversation_message(session_key, "model", json.dumps(response_payload))

    return {
        "response": json.dumps(response_payload),
        "session_id": conversation_id,
        "type": "message",
        "user": _user_display_name(user),
    }


# =============================================================================
# FEEDBACK ENDPOINT
# =============================================================================

@router.post("/feedback")
async def feedback(body: FeedbackRequest):
    """Save user feedback for RLHF learning."""
    user_message = str(body.user_message or "").strip()
    ai_response = str(body.ai_response or "").strip()
    comment = None if body.comment is None else str(body.comment).strip()
    rating = str(body.rating or "").strip().lower()
    session_id = str(body.session_id or "guest_session").strip() or "guest_session"

    payload = {
        "user_message": user_message,
        "ai_response": ai_response,
        "rating": rating,
        "comment": comment or None,
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
    }

    saved = await db_engine_async.save_feedback(payload)

    if saved and user_message and ai_response:
        if rating == "positive":
            await db_engine_async.save_learned_response(user_message, ai_response)
        elif rating == "negative":
            await db_engine_async.save_bad_response(
                user_message,
                ai_response,
                reason=comment or "User marked as incorrect",
            )

    return {"success": bool(saved), "status": "success" if saved else "error"}


# =============================================================================
# EXPORT CHAT ENDPOINT
# =============================================================================

@router.get("/export_chat")
async def export_chat(request: Request):
    """Export conversation history for logged-in user."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return {"success": False, "message": "Token is missing"}

    user = decode_access_token(auth_header.split(" ", 1)[1].strip())
    student_number = _user_student_number(user)

    if not student_number:
        return {"success": False, "message": "Invalid or expired token"}

    session_key = f"user:{student_number}"
    return {
        "success": True,
        "session_id": student_number,
        "messages": get_conversation_history(session_key),
    }
