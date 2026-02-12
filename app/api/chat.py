"""
UCSI Buddy Chat API v2 - Refactored with Hybrid Intent Classifier

Clean architecture:
1. Intent Classification (hybrid: keyword guard + vector + LLM)
2. Data Retrieval (based on category)
3. LLM Response Generation (always)
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request

from app.config import Config
from app.core.session import (
    append_conversation_message,
    get_conversation_history,
    high_security_sessions,
)
from app.engines.intent_classifier import intent_classifier, _is_personal_query
from app.engines.intent_config import (
    has_ucsi_keywords as _config_has_ucsi_keywords,
    detect_entity_type as _config_detect_entity_type,
)
from app.engines.ai_engine_async import ai_engine_async
from app.engines.db_engine_async import db_engine_async
from app.engines.rag_engine_async import rag_engine_async
from app.engines.response_validator import (
    response_validator,
    detect_prompt_injection,
    sanitize_user_input,
)
from app.engines.monitoring import Monitor, record_feedback
from app.api.chat_helpers import (
    _compose_no_data_response,
    _detect_language,
    _extract_rich_content,
    _format_personal_info,
    _personal_info_suggestions,
    _user_display_name,
    _user_student_number,
)
from app.schemas import ChatRequest, FeedbackRequest
from app.utils.auth_utils import decode_access_token
from app.api.dependencies import get_current_user_optional
from fastapi import Depends

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


def _intent_to_preferred_labels(classification: Dict[str, Any]) -> Optional[List[str]]:
    """Map intent classification to preferred RAG labels for better search precision."""
    intent = str(classification.get("intent") or "").lower()
    label_map = {
        "ucsi_hostel": ["Hostel", "HostelFAQ"],
        "ucsi_facility": ["Facility", "CampusBlocks"],
        "ucsi_programme": ["Programme"],
        "ucsi_staff": ["Staff"],
        "ucsi_schedule": ["Schedule"],
    }
    return label_map.get(intent)


def _clean_context_for_llm(context: str) -> str:
    """Strip internal markers from RAG context before sending to LLM.

    Removes [Document], [Hostel], [Programme], [conf:X.XX] etc. so the
    LLM sees clean factual text and never echoes internal tags to users.
    """
    if not context:
        return context
    # Remove label prefixes like [Document], [Hostel], [Facility], [Programme], etc.
    cleaned = re.sub(r"\[(?:Document|Hostel|Facility|Programme|Staff|Schedule|HostelFAQ|CampusBlocks|Verified Answer)\]\s*", "", context)
    # Remove confidence markers like [conf:0.82]
    cleaned = re.sub(r"\s*\[conf:[0-9.]+\]", "", cleaned)
    # Remove [NO_RELEVANT_DATA_FOUND] marker
    cleaned = cleaned.replace("[NO_RELEVANT_DATA_FOUND]", "")
    # Remove [LOW_CONFIDENCE_CONTEXT] and [FEEDBACK_GUARD] markers
    cleaned = re.sub(r"\[LOW_CONFIDENCE_CONTEXT\][^\n]*\n*", "", cleaned)
    cleaned = re.sub(r"\[FEEDBACK_GUARD\][^\n]*\n*", "", cleaned)
    return cleaned.strip()


def _extract_staff_name_from_label(label: str) -> str:
    text = str(label or "").strip()
    text = text.replace("View ", "").replace("'s Profile", "").strip()
    return text


def _select_staff_link(links: List[Dict[str, Any]], query: str) -> Optional[Dict[str, Any]]:
    staff_links = [lnk for lnk in (links or []) if lnk.get("type") == "staff_profile"]
    if not staff_links:
        return None

    query_lower = str(query or "").lower()
    best_link = None
    best_score = 0

    for lnk in staff_links:
        staff_name = _extract_staff_name_from_label(lnk.get("label", ""))
        if not staff_name:
            continue
        name_lower = staff_name.lower()
        score = 0
        if name_lower in query_lower:
            score += 10
        words = [w for w in re.split(r"[^a-z0-9]+", name_lower) if len(w) >= 3]
        score += sum(1 for w in words if w in query_lower)
        if score > best_score:
            best_score = score
            best_link = lnk

    if best_link and best_score > 0:
        return best_link
    return staff_links[0] if len(staff_links) == 1 else None


def _promote_drive_map_to_image(links: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for lnk in links or []:
        if lnk.get("type") != "map":
            continue
        map_url = str(lnk.get("url") or "")
        if "drive.google.com" in map_url.lower():
            return {
                "url": map_url,
                "type": "building_image",
                "label": str(lnk.get("label") or "Campus Map"),
            }
    return None


def _select_building_image(
    images: List[Dict[str, Any]],
    links: List[Dict[str, Any]],
    user_msg: str,
    search_term: str,
    response_text: str,
) -> Optional[Dict[str, Any]]:
    img_list = list(images or [])
    query_lower = f"{str(user_msg or '')} {str(search_term or '')}".lower()
    response_lower = str(response_text or "").lower()

    for img in img_list:
        img_label = str(img.get("label") or "").lower()
        if img_label and img_label in query_lower:
            return img

    for img in img_list:
        img_label = str(img.get("label") or "").lower()
        if img_label and img_label in response_lower:
            return img

    if img_list:
        return img_list[0]

    return _promote_drive_map_to_image(links or [])


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
    user: Optional[dict] = None,
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

    # Perform RAG search with preferred labels from intent classification
    preferred_labels = _intent_to_preferred_labels(classification)
    rag_result = await rag_engine_async.search_context(
        query=search_term,
        top_k=5,
        preferred_labels=preferred_labels,
    )
    
    rag_meta = _extract_rag_meta(rag_result)

    # [SECURITY CHECK] Block unauthenticated access to student data
    # Only flag as student data when source explicitly contains "student"
    is_student_data = False
    if rag_meta.get("has_relevant_data"):
        is_student_data = any("student" in str(s).lower() for s in rag_meta.get("sources", []))

    if is_student_data and not user:
        print(f"[SECURITY] Blocking unauthenticated access to student data query='{user_msg}'")
        return {
            "text": "I found information about this student, but I need you to login first to view it." if user_lang != "ko" else "해당 학생 정보를 확인했지만, 상세 내용을 보려면 로그인이 필요해요.",
            "suggestions": ["Login"],
            "type": "login_hint",
            "conversation_id": conversation_id,
        }

    print(f"[DEBUG] handle_ucsi_query: intent='{classification.get('intent')}', search_term='{search_term}', found_data={rag_meta.get('has_relevant_data')}")

    retrieval_meta = {
        "route": "rag",
        "used": True,
        "confidence": rag_meta["confidence"],
        "sources": rag_meta["sources"],
        "has_relevant_data": rag_meta["has_relevant_data"],
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
            "text": _compose_no_data_response(user_lang),
            "suggestions": [],
            "retrieval": retrieval_meta,
        }

    # Build context for LLM — strip internal markers so they don't leak into responses
    context_text = _clean_context_for_llm(rag_meta["context"])

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

    response_text = str(ai_result.get("response") or ai_result.get("text") or "I'm not sure how to respond.")
    suggestions = ai_result.get("suggestions", [])

    # Include context for validation
    retrieval_meta["context"] = context_text

    # Extract rich content only when query intent is relevant
    # Uses classifier metadata instead of hardcoded keywords
    query_type = classification.get("query_type", "specific")
    entity_type = classification.get("entity_type") or _config_detect_entity_type(user_msg)
    rich_content = {"links": [], "images": []}
    raw_rich = _extract_rich_content(rag_meta["context"])

    # Aggregate queries → NEVER get rich content links
    # Specific queries → filter by entity_type from classifier
    if query_type != "aggregate" and entity_type in ("staff", "building"):

        if entity_type == "staff":
            selected_link = _select_staff_link(raw_rich.get("links", []), user_msg)
            if selected_link:
                rich_content["links"].append(selected_link)
            else:
                # Retry with Staff-only retrieval to avoid unrelated profile links.
                try:
                    staff_rag = await rag_engine_async.search_context(
                        query=search_term,
                        top_k=5,
                        preferred_labels=["Staff"],
                    )
                    staff_rich = _extract_rich_content(str(staff_rag.get("context") or ""))
                    retry_link = _select_staff_link(staff_rich.get("links", []), user_msg)
                    if retry_link:
                        rich_content["links"].append(retry_link)
                except Exception:
                    pass

        if entity_type == "building":
            selected_image = _select_building_image(
                images=raw_rich.get("images", []),
                links=raw_rich.get("links", []),
                user_msg=user_msg,
                search_term=search_term,
                response_text=response_text,
            )
            if selected_image is None:
                # Retry with CampusBlocks-only retrieval to guarantee block media.
                try:
                    block_rag = await rag_engine_async.search_context(
                        query=search_term,
                        top_k=5,
                        preferred_labels=["CampusBlocks"],
                    )
                    block_rich = _extract_rich_content(str(block_rag.get("context") or ""))
                    selected_image = _select_building_image(
                        images=block_rich.get("images", []),
                        links=block_rich.get("links", []),
                        user_msg=user_msg,
                        search_term=search_term,
                        response_text=response_text,
                    )
                except Exception:
                    selected_image = None

            if selected_image:
                rich_content["images"].append(selected_image)
            else:
                for lnk in raw_rich.get("links", []):
                    if lnk.get("type") != "map":
                        continue
                    rich_content["links"].append(lnk)
                    break  # Max 1 map link

    # Programme links: attach if entity_type is programme and not aggregate
    elif query_type != "aggregate" and entity_type == "programme":
        for lnk in raw_rich.get("links", []):
            if lnk.get("type") == "programme_info":
                rich_content["links"].append(lnk)
                if len(rich_content["links"]) >= 2:
                    break

    # Deduplicate by URL
    seen_urls = set()
    deduped_links = []
    for lnk in rich_content["links"]:
        if lnk["url"] not in seen_urls:
            seen_urls.add(lnk["url"])
            deduped_links.append(lnk)
    rich_content["links"] = deduped_links
    rich_content["images"] = rich_content["images"][:1]

    result = {
        "text": response_text,
        "suggestions": suggestions[:3],
        "retrieval": retrieval_meta,
    }
    if rich_content.get("links") or rich_content.get("images"):
        result["rich_content"] = rich_content
    return result


async def handle_llm_first_query(
    user_msg: str,
    user_lang: str,
    history: List[Dict],
    conversation_id: str,
    user: Optional[dict] = None,
) -> Dict[str, Any]:
    """RAG-first handler: always searches DB before calling LLM.

    Ensures person/entity queries (e.g. student names) find DB data
    even when no UCSI keywords are present in the message.
    """
    # Step 1: RAG (semantic) + MongoDB name search in parallel
    rag_result, db_result = await asyncio.gather(
        rag_engine_async.search_context(query=user_msg, top_k=5),
        db_engine_async.search_by_name(user_msg),
    )
    rag_meta = _extract_rag_meta(rag_result)

    # Merge contexts from both search paths
    all_contexts: List[str] = []
    all_sources: List[str] = []

    if rag_meta["has_relevant_data"]:
        all_contexts.append(rag_meta["context"])
        all_sources.extend(rag_meta["sources"])

    if db_result:
        all_contexts.append(db_result["context"])
        all_sources.extend(db_result["sources"])

    if all_contexts:
        # [SECURITY CHECK] Block unauthenticated access to student data
        is_student_data = any("student" in s.lower() for s in all_sources)

        if is_student_data and not user:
            return {
                "text": "I found information about this student, but I need you to login first to view it." if user_lang != "ko" else "해당 학생 정보를 확인했지만, 상세 내용을 보려면 로그인이 필요해요.",
                "suggestions": ["Login"],
                "type": "login_hint",
                "conversation_id": conversation_id,
            }

        context_text = _clean_context_for_llm("\n\n".join(all_contexts))
        ai_result = await ai_engine_async.process_message(
            user_message=user_msg,
            data_context=context_text,
            conversation_history=history,
            language=user_lang,
        )
        response_text = str(ai_result.get("response") or ai_result.get("text") or "")
        suggestions = ai_result.get("suggestions", [])
        result = {
            "text": response_text,
            "suggestions": suggestions[:3],
            "retrieval": {
                "route": "rag_first",
                "used": True,
                "confidence": rag_meta["confidence"],
                "sources": all_sources,
                "has_relevant_data": True,
                "context": context_text,
            },
        }
        raw_rich = _extract_rich_content("\n\n".join(all_contexts))
        rich_content = {"links": [], "images": []}
        entity_type = _config_detect_entity_type(user_msg)

        if entity_type == "staff":
            selected_link = _select_staff_link(raw_rich.get("links", []), user_msg)
            if selected_link:
                rich_content["links"].append(selected_link)

        elif entity_type == "building":
            selected_image = _select_building_image(
                images=raw_rich.get("images", []),
                links=raw_rich.get("links", []),
                user_msg=user_msg,
                search_term=user_msg,
                response_text=response_text,
            )

            if selected_image:
                rich_content["images"].append(selected_image)
            else:
                first_map = next(
                    (lnk for lnk in raw_rich.get("links", []) if lnk.get("type") == "map"),
                    None,
                )
                if first_map:
                    rich_content["links"].append(first_map)

        if rich_content["links"] or rich_content["images"]:
            result["rich_content"] = rich_content
        return result

    # Step 2: No DB data found → LLM answers from general knowledge
    ai_result = await ai_engine_async.process_message(
        user_message=user_msg,
        data_context="",
        conversation_history=history,
        language=user_lang,
    )
    response_text = str(ai_result.get("response") or ai_result.get("text") or "")
    suggestions = ai_result.get("suggestions", [])
    return {
        "text": response_text,
        "suggestions": suggestions[:3],
        "retrieval": {
            "route": "llm_first",
            "used": False,
            "confidence": 1.0,
            "sources": [],
        },
    }


# =============================================================================
# MAIN CHAT ENDPOINT
# =============================================================================

@router.post("/chat")
async def chat(
    body: ChatRequest,
    request: Request,
    user: Optional[dict] = Depends(get_current_user_optional),
):
    """
    Main chat endpoint with clean routing logic.

    Flow:
    1. Parse request & authenticate
    2. Sanitize input & detect injection
    3. Classify intent (hybrid)
    4. Route to appropriate handler
    5. Validate response
    6. Return response
    """
    async with Monitor.request("chat") as monitor:
        # [1] Parse request
        # User is already resolved by Depends(get_current_user_optional)

        session_key, conversation_id, _ = _resolve_session(user, body)
        raw_msg = (body.message or "").strip()

        if not raw_msg:
            return {
                "response": "Please type a message so I can help you!",
                "type": "message",
                "conversation_id": conversation_id,
                "session_id": conversation_id,
                "suggestions": ["What are the hostel fees?", "Show me programmes"],
            }

        # [1.5] Sanitize input and detect prompt injection
        is_injection, injection_types = detect_prompt_injection(raw_msg)
        user_msg = sanitize_user_input(raw_msg, max_length=2000)

        if is_injection:
            # Block the injection attempt immediately — never let it reach the LLM
            await _log_rag_miss(
                conversation_id=conversation_id,
                user_message=raw_msg[:200],
                query="[INJECTION_ATTEMPT]",
                rag_meta={"confidence": 0, "sources": []},
                reason=f"prompt_injection:{','.join(injection_types)}",
            )
            return {
                "response": "I'm sorry, I can't process that request.",
                "type": "message",
                "conversation_id": conversation_id,
                "session_id": conversation_id,
                "suggestions": [],
            }

        # Save user message to history
        append_conversation_message(session_key, "user", user_msg)
        history = get_conversation_history(session_key)
        user_lang = _detect_language(user_msg)

        # [2] LLM-First Routing
        # Only use the full intent classification pipeline for queries
        # that clearly need it (personal data, explicit UCSI keywords).
        # Everything else goes directly to the LLM for natural handling.

        if _is_personal_query(user_msg):
            # Personal queries need DB lookup + security checks
            classification = await intent_classifier.classify(
                user_message=user_msg,
                conversation_history=history,
                language=user_lang,
                search_term=body.search_term,
            )
            await monitor.record_intent(
                intent=classification.get("intent", "unknown"),
                confidence=classification.get("confidence", 0.0),
                source=classification.get("source", "unknown"),
            )
            result = await handle_personal_query(
                user=user,
                user_msg=user_msg,
                user_lang=user_lang,
                classification=classification,
                conversation_id=conversation_id,
            )

        elif _config_has_ucsi_keywords(user_msg):
            # UCSI keywords detected → classify to get query_type
            classification = await intent_classifier.classify(
                user_message=user_msg,
                conversation_history=history,
                language=user_lang,
                search_term=body.search_term,
            )
            await monitor.record_intent(
                intent=classification.get("intent", "unknown"),
                confidence=classification.get("confidence", 0.0),
                source=classification.get("source", "unknown"),
            )

            # Classifier now decides: aggregate → LLM-First, specific → RAG
            if classification.get("query_type") == "aggregate":
                result = await handle_llm_first_query(
                    user_msg=user_msg,
                    user_lang=user_lang,
                    history=history,
                    conversation_id=conversation_id,
                    user=user,
                )
            else:
                result = await handle_ucsi_query(
                    user_msg=user_msg,
                    user_lang=user_lang,
                    classification=classification,
                    conversation_id=conversation_id,
                    history=history,
                    user=user,
                )

        else:
            # [3] LLM-First: Let the LLM understand and respond naturally.
            # The LLM decides if UCSI data is needed (via needs_context flag).
            await monitor.record_intent(
                intent="llm_first",
                confidence=1.0,
                source="llm_first",
            )
            result = await handle_llm_first_query(
                user_msg=user_msg,
                user_lang=user_lang,
                history=history,
                conversation_id=conversation_id,
                user=user,
            )

        # [4] Handle special response types (login_hint, password_prompt)
        if result.get("type") in ("login_hint", "password_prompt"):
            return {
                "response": result.get("text") or result.get("response", ""),
                "type": result["type"],
                "conversation_id": conversation_id,
                "session_id": conversation_id,
            }

        # [4.5] Validate response (prevent hallucination for RAG-based responses)
        response_text = result.get("text", "")
        retrieval_info = result.get("retrieval", {})

        if retrieval_info.get("route") in ("rag", "llm_first_rag") and retrieval_info.get("used"):
            context_text = retrieval_info.get("context", "")
            sources = retrieval_info.get("sources", [])
            confidence = retrieval_info.get("confidence", 0.0)

            # Record RAG metrics
            await monitor.record_rag(
                query=user_msg,
                confidence=confidence,
                has_results=retrieval_info.get("has_relevant_data", False),
                sources=sources,
            )

            validation = response_validator.validate_response(
                response_text=response_text,
                context_text=context_text,
                sources=sources,
                confidence=confidence,
                language=user_lang,
                strict_grounding=(confidence < 0.3),
            )

            if not validation.get("is_valid", True):
                warning_list = [str(w) for w in (validation.get("warnings") or [])]
                # Log validation issues
                await _log_rag_miss(
                    conversation_id=conversation_id,
                    user_message=user_msg,
                    query=classification.get("search_term", user_msg),
                    rag_meta={"confidence": confidence, "sources": sources},
                    reason=(
                        f"validation_failed:{','.join(warning_list)}"
                        if warning_list
                        else "validation_failed"
                    ),
                )
                # Use safe response if validation failed
                response_text = validation.get("text", response_text)
                result["text"] = response_text

        # [5] Build final response — return fields directly (no double JSON encoding)
        final_text = result.get("text", "")
        final_suggestions = result.get("suggestions", [])[:3]

        # Save to conversation history (compact)
        append_conversation_message(session_key, "model", final_text)

        response_out = {
            "response": final_text,
            "suggestions": final_suggestions,
            "session_id": conversation_id,
            "type": "message",
            "user": _user_display_name(user),
        }

        # Include rich content (links, images) if available
        if result.get("rich_content"):
            response_out["rich_content"] = result["rich_content"]

        return response_out


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

    # Record feedback metric
    await record_feedback(rating, session_id)

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
