import json
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request

from app.config import Config
from app.core.session import (
    append_conversation_message,
    get_conversation_history,
    high_security_sessions,
)
from app.engines.ai_engine_async import ai_engine_async
from app.engines.db_engine_async import db_engine_async
from app.engines.rag_engine_async import rag_engine_async
from app.api.chat_helpers import (
    _build_suggestions,
    _capability_smalltalk_response,
    _capability_suggestions,
    _compose_no_data_response,
    _detect_language,
    _extract_person_candidate,
    _extract_rag_meta,
    _format_personal_info,
    _has_ucsi_context,
    _infer_preferred_labels,
    _is_capability_smalltalk_query,
    _is_grade_query,
    _is_personal_query,
    _personal_info_suggestions,
    _resolve_session,
    _should_force_rag,
    _user_display_name,
    _user_student_number,
)
from app.schemas import ChatRequest, FeedbackRequest
from app.utils.auth_utils import decode_access_token

router = APIRouter()


async def _log_rag_miss(
    *,
    conversation_id: str,
    user_message: str,
    query: str,
    rag_meta: Dict[str, Any],
    reason: str,
) -> None:
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


@router.post("/chat")
async def chat(body: ChatRequest, request: Request):
    user = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        user = decode_access_token(auth_header.split(" ", 1)[1].strip())

    session_key, conversation_id, _ = _resolve_session(user, body)
    user_msg = (body.message or "").strip()
    if not user_msg:
        return {"error": "Message is required", "session_id": conversation_id}

    append_conversation_message(session_key, "user", user_msg)
    history = get_conversation_history(session_key)
    user_lang = _detect_language(user_msg)

    search_term = (body.search_term or "").strip() or None
    context_text = ""
    is_personal = _is_personal_query(user_msg, search_term)
    person_candidate: Optional[str] = None
    person_resolution: Optional[Dict[str, Any]] = None
    feedback_bad_guard = False
    retrieval_meta: Dict[str, Any] = {
        "route": "general_ai",
        "used": False,
        "confidence": 0.0,
        "sources": [],
    }

    if _is_capability_smalltalk_query(user_msg):
        capability_meta = {
            "route": "capability_smalltalk",
            "used": False,
            "confidence": 1.0,
            "sources": [],
        }
        response_payload = {
            "text": _capability_smalltalk_response(user_msg),
            "suggestions": _build_suggestions(
                user_message=user_msg,
                user_lang=user_lang,
                retrieval_meta=capability_meta,
                ai_suggestions=_capability_suggestions(user_lang),
                is_personal=False,
                search_term=search_term,
                person_resolution=person_resolution,
            ),
            "retrieval": capability_meta,
        }
        append_conversation_message(session_key, "model", json.dumps(response_payload))
        return {
            "response": json.dumps(response_payload),
            "session_id": conversation_id,
            "type": "message",
            "user": _user_display_name(user),
        }

    if not is_personal:
        learned_answer = await db_engine_async.search_learned_response(user_msg)
        if learned_answer:
            retrieval_meta = {
                "route": "feedback_learned_qa",
                "used": True,
                "confidence": 1.0,
                "sources": ["MongoDB:LearnedQA"],
            }
            response_payload = {
                "text": learned_answer,
                "suggestions": _build_suggestions(
                    user_message=user_msg,
                    user_lang=user_lang,
                    retrieval_meta=retrieval_meta,
                    ai_suggestions=[],
                    is_personal=False,
                    search_term=search_term,
                    person_resolution=person_resolution,
                ),
                "retrieval": retrieval_meta,
            }
            append_conversation_message(session_key, "model", json.dumps(response_payload))
            return {
                "response": json.dumps(response_payload),
                "session_id": conversation_id,
                "type": "message",
                "user": _user_display_name(user),
            }

    force_rag = bool(body.needs_context or search_term or _should_force_rag(user_msg))

    if not is_personal and not search_term:
        person_candidate = _extract_person_candidate(user_msg)
        if person_candidate:
            staff_match = await db_engine_async.find_staff_by_name(person_candidate)
            if staff_match:
                force_rag = True
                search_term = person_candidate
                person_resolution = {
                    "candidate": person_candidate,
                    "match_type": "db_staff_match",
                    "matched_name": staff_match.get("name"),
                }
            elif _has_ucsi_context(user_msg):
                force_rag = True
                search_term = person_candidate
                person_resolution = {
                    "candidate": person_candidate,
                    "match_type": "ucsi_context_no_match",
                }
            else:
                person_resolution = {
                    "candidate": person_candidate,
                    "match_type": "general_person",
                }

    if person_resolution:
        retrieval_meta["person_resolution"] = person_resolution

    if not is_personal:
        feedback_bad_guard = await db_engine_async.has_bad_response(user_msg)
        if feedback_bad_guard:
            force_rag = True

    if is_personal:
        student_number = _user_student_number(user)
        if not student_number:
            return {
                "response": "Please login to access personal information.",
                "type": "login_hint",
                "conversation_id": conversation_id,
                "session_id": conversation_id,
            }

        expiry = high_security_sessions.get(student_number)
        high_security_active = bool(expiry and expiry >= datetime.now())
        if expiry and expiry < datetime.now():
            high_security_sessions.pop(student_number, None)

        if _is_grade_query(user_msg, search_term) and not high_security_active:
            prompt_text = (
                "GPA/?깆쟻 ?뺣낫瑜?蹂대젮硫?鍮꾨?踰덊샇 ?몄쬆???꾩슂?댁슂."
                if user_lang == "ko"
                else "Please verify your password to access GPA/grade information."
            )
            return {
                "response": prompt_text,
                "type": "password_prompt",
                "conversation_id": conversation_id,
                "session_id": conversation_id,
            }

        student_data = await db_engine_async.get_student_by_number(student_number)
        if student_data:
            retrieval_meta = {
                "route": "personal_db_direct",
                "used": True,
                "confidence": 1.0,
                "sources": ["MongoDB:Student"],
            }
            response_payload = {
                "text": _format_personal_info(
                    student_data,
                    user_msg,
                    search_term,
                    allow_sensitive=high_security_active,
                ),
                "suggestions": _build_suggestions(
                    user_message=user_msg,
                    user_lang=user_lang,
                    retrieval_meta=retrieval_meta,
                    ai_suggestions=_personal_info_suggestions(
                        user_lang, allow_sensitive=high_security_active
                    ),
                    is_personal=True,
                    search_term=search_term,
                    person_resolution=person_resolution,
                ),
                "retrieval": retrieval_meta,
            }
            append_conversation_message(session_key, "model", json.dumps(response_payload))
            return {
                "response": json.dumps(response_payload),
                "session_id": conversation_id,
                "type": "message",
                "user": _user_display_name(user),
            }
        else:
            retrieval_meta = {
                "route": "personal_db_direct",
                "used": True,
                "confidence": 0.0,
                "sources": ["MongoDB:Student"],
            }
            missing_text = (
                "?숈깮 ?뺣낫瑜?李얠? 紐삵뻽?댁슂. ?숇쾲/?대쫫???ㅼ떆 ?뺤씤??二쇱꽭??"
                if user_lang == "ko"
                else "I could not find your student profile. Please verify your student account."
            )
            response_payload = {
                "text": missing_text,
                "suggestions": _build_suggestions(
                    user_message=user_msg,
                    user_lang=user_lang,
                    retrieval_meta=retrieval_meta,
                    ai_suggestions=_personal_info_suggestions(
                        user_lang, allow_sensitive=high_security_active
                    ),
                    is_personal=True,
                    search_term=search_term,
                    person_resolution=person_resolution,
                ),
                "retrieval": retrieval_meta,
            }
            append_conversation_message(session_key, "model", json.dumps(response_payload))
            return {
                "response": json.dumps(response_payload),
                "session_id": conversation_id,
                "type": "message",
                "user": _user_display_name(user),
            }

    elif force_rag:
        query = str(search_term or user_msg)
        preferred_labels = _infer_preferred_labels(user_msg, search_term)
        if person_resolution and person_resolution.get("match_type") == "db_staff_match":
            if "Staff" not in preferred_labels:
                preferred_labels.insert(0, "Staff")
        rag_result = await rag_engine_async.search_context(
            query=query,
            top_k=5,
            preferred_labels=preferred_labels,
        )
        rag_meta = _extract_rag_meta(rag_result)
        retrieval_meta = {
            "route": "rag",
            "used": True,
            "confidence": rag_meta["confidence"],
            "sources": rag_meta["sources"],
            "preferred_labels": preferred_labels,
        }
        if feedback_bad_guard:
            retrieval_meta["feedback_guard"] = True
        if person_resolution:
            retrieval_meta["person_resolution"] = person_resolution
        context_text = rag_meta["context"] or "[NO_RELEVANT_DATA_FOUND]"
        if feedback_bad_guard:
            context_text = (
                "[FEEDBACK_GUARD] A previous answer for this query was marked incorrect. "
                "Do not repeat it; rely only on verified context.\n\n"
                + context_text
            )

        if not rag_meta["has_relevant_data"]:
            await _log_rag_miss(
                conversation_id=conversation_id,
                user_message=user_msg,
                query=query,
                rag_meta=rag_meta,
                reason="no_relevant_data",
            )
            response_payload = {
                "text": _compose_no_data_response(),
                "suggestions": _build_suggestions(
                    user_message=user_msg,
                    user_lang=user_lang,
                    retrieval_meta=retrieval_meta,
                    ai_suggestions=[],
                    is_personal=False,
                    search_term=search_term,
                    person_resolution=person_resolution,
                ),
                "retrieval": retrieval_meta,
            }
            append_conversation_message(session_key, "model", json.dumps(response_payload))
            return {
                "response": json.dumps(response_payload),
                "session_id": conversation_id,
                "type": "message",
                "user": _user_display_name(user),
            }

    ai_result = None
    if context_text:
        if retrieval_meta.get("used") and retrieval_meta.get("confidence", 0.0) < float(
            Config.RAG_FAST_CONFIDENCE
        ):
            context_text = (
                "[LOW_CONFIDENCE_CONTEXT] Some matches are weak. Avoid overclaiming.\n\n"
                + context_text
            )
        ai_result = await ai_engine_async.process_message(
            user_message=user_msg,
            data_context=context_text,
            conversation_history=history,
            language=user_lang,
        )
    else:
        # Two-step flow: let AI decide intent first, then retrieve context if needed.
        initial_result = await ai_engine_async.process_message(
            user_message=user_msg,
            data_context="",
            conversation_history=history,
            language=user_lang,
        )
        if initial_result.get("needs_context"):
            query = (
                initial_result.get("search_term")
                or search_term
                or user_msg
            )
            preferred_labels = _infer_preferred_labels(user_msg, str(query))
            rag_result = await rag_engine_async.search_context(
                query=str(query),
                top_k=5,
                preferred_labels=preferred_labels,
            )
            rag_meta = _extract_rag_meta(rag_result)
            retrieval_meta = {
                "route": "planner_rag",
                "used": True,
                "confidence": rag_meta["confidence"],
                "sources": rag_meta["sources"],
                "preferred_labels": preferred_labels,
            }
            if feedback_bad_guard:
                retrieval_meta["feedback_guard"] = True
            if person_resolution:
                retrieval_meta["person_resolution"] = person_resolution
            context_text = rag_meta["context"] or "[NO_RELEVANT_DATA_FOUND]"
            if feedback_bad_guard:
                context_text = (
                    "[FEEDBACK_GUARD] A previous answer for this query was marked incorrect. "
                    "Do not repeat it; rely only on verified context.\n\n"
                    + context_text
                )

            if rag_meta["has_relevant_data"]:
                if rag_meta["confidence"] < float(Config.RAG_FAST_CONFIDENCE):
                    context_text = (
                        "[LOW_CONFIDENCE_CONTEXT] Some matches are weak. Avoid overclaiming.\n\n"
                        + context_text
                    )
                ai_result = await ai_engine_async.process_message(
                    user_message=user_msg,
                    data_context=context_text,
                    conversation_history=history,
                    language=user_lang,
                )
            else:
                await _log_rag_miss(
                    conversation_id=conversation_id,
                    user_message=user_msg,
                    query=str(query),
                    rag_meta=rag_meta,
                    reason="planner_no_relevant_data",
                )
                fallback_text = str(initial_result.get("response") or "").strip()
                if fallback_text and not force_rag:
                    ai_result = {
                        "response": fallback_text,
                        "suggestions": _build_suggestions(
                            user_message=user_msg,
                            user_lang=user_lang,
                            retrieval_meta=retrieval_meta,
                            ai_suggestions=initial_result.get("suggestions", []),
                            is_personal=False,
                            search_term=search_term,
                            person_resolution=person_resolution,
                        ),
                    }
                else:
                    ai_result = {
                        "response": _compose_no_data_response(),
                        "suggestions": _build_suggestions(
                            user_message=user_msg,
                            user_lang=user_lang,
                            retrieval_meta=retrieval_meta,
                            ai_suggestions=[],
                            is_personal=False,
                            search_term=search_term,
                            person_resolution=person_resolution,
                        ),
                    }
        else:
            ai_result = initial_result

    response_text = str((ai_result or {}).get("response") or "I'm not sure how to respond.")
    suggestions = _build_suggestions(
        user_message=user_msg,
        user_lang=user_lang,
        retrieval_meta=retrieval_meta,
        ai_suggestions=(ai_result or {}).get("suggestions") or [],
        is_personal=False,
        search_term=search_term,
        person_resolution=person_resolution,
    )

    response_payload = {
        "text": response_text,
        "suggestions": suggestions[:3],
        "retrieval": retrieval_meta,
    }
    append_conversation_message(session_key, "model", json.dumps(response_payload))

    return {
        "response": json.dumps(response_payload),
        "session_id": conversation_id,
        "type": "message",
        "user": _user_display_name(user),
    }


@router.post("/feedback")
async def feedback(body: FeedbackRequest):
    user_message = str(body.user_message or "").strip()
    ai_response = str(body.ai_response or "").strip()
    comment = None if body.comment is None else str(body.comment).strip()
    rating = str(body.rating or "").strip().lower()
    payload = {
        "user_message": user_message,
        "ai_response": ai_response,
        "rating": rating,
        "comment": comment or None,
        "query_norm": user_message.lower(),
    }
    payload["timestamp"] = datetime.now().isoformat()
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


@router.get("/export_chat")
async def export_chat(request: Request):
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return {
            "success": False,
            "message": "Token is missing",
        }

    user = decode_access_token(auth_header.split(" ", 1)[1].strip())
    student_number = _user_student_number(user)
    if not student_number:
        return {
            "success": False,
            "message": "Invalid or expired token",
        }

    session_key = f"user:{student_number}"
    return {
        "success": True,
        "session_id": student_number,
        "messages": get_conversation_history(session_key),
    }
