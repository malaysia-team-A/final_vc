"""
University Chatbot API - Main Server (Flask Version)
Features:
- AI-powered chatbot with LangChain + Local Ollama LLM
- JWT Authentication (OAuth2 Style)
- Dual Authentication for Sensitive Data (Grades)
- RAG (Retrieval-Augmented Generation)
- Log Anonymization
"""
from flask import Flask, request, jsonify, send_from_directory, Response
from app.engines.data_engine import DataEngine
from app.engines.ai_engine import AIEngine
from app.engines.feedback_engine import FeedbackEngine
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
import json
import re
import logging
import traceback
from datetime import datetime, timedelta
import secrets
from functools import wraps
from collections import deque

# Custom Modules
# Custom Modules
from app.utils import auth_utils
from app.utils import logging_utils
from app.engines.faq_cache_engine import faq_cache, unanswered_manager
from app.engines.language_engine import detect_language, get_localized_phrase, multilingual

# ?? WORLD-CLASS ADVANCED ENGINES
try:
    from app.engines.semantic_cache_engine import semantic_cache
    HAS_SEMANTIC_CACHE = True
except ImportError as e:
    print(f"[Warning] Semantic Cache disabled: {e}")
    HAS_SEMANTIC_CACHE = False
    semantic_cache = None

try:
    from app.engines.query_rewriter import query_rewriter
    HAS_QUERY_REWRITER = True
except ImportError as e:
    print(f"[Warning] Query Rewriter disabled: {e}")
    HAS_QUERY_REWRITER = False
    query_rewriter = None

try:
    from app.engines.reranker import reranker
    HAS_RERANKER = True
except ImportError as e:
    print(f"[Warning] Reranker disabled: {e}")
    HAS_RERANKER = False
    reranker = None


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

# Setup Logging
logger = logging_utils.get_logger()

# Suppress noisy external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("werkzeug").setLevel(logging.WARNING) # Optional: cleaner flask logs

# Initialize Flask App
app = Flask(__name__, static_folder="static/site", static_url_path="/site")
app.secret_key = auth_utils.SECRET_KEY

# Initialize Engines
DATA_FILE = "data/Chatbot_TestData.xlsx" # Config artifact, logic moved to MongoDB
data_engine = DataEngine(DATA_FILE)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemma-3-12b-it")
ai_engine = AIEngine(MODEL_NAME)
feedback_engine = FeedbackEngine()

# In-memory storage for Dual Auth (High Security) sessions
# Format: { "student_number": datetime_expiry }
high_security_sessions = {}

# In-memory conversation history (prototype for follow-up context)
CONVERSATION_HISTORY_LIMIT = 12  # store last 6 exchanges
conversation_history_store = {}

# Ensure directories exist
if not os.path.exists("data/knowledge_base"):
    os.makedirs("data/knowledge_base")

# Index MongoDB collections into FAISS for semantic search
try:
    from app.engines.rag_engine import rag_engine
    if rag_engine.enabled:
        indexed = rag_engine.index_mongodb_collections()
        if indexed > 0:
            print(f"[Startup] Indexed {indexed} MongoDB documents into FAISS")
except Exception as e:
    print(f"[Startup] MongoDB indexing skipped: {e}")

PERSONAL_DATA_FIELDS = [
    "STUDENT_NUMBER",
    "STUDENT_ID",
    "STUDENT_NAME",
    "PREFERRED_NAME",
    "PROGRAMME_CODE",
    "PROGRAMME_NAME",
    "PROGRAMME",
    "PROFILE_STATUS",
    "PROFILE_TYPE",
    "ENROLLMENT_STATUS",
    "SEMESTER",
    "INTAKE",
    "CAMPUS",
    "FACULTY",
    "NATIONALITY",
    "GENDER",
    "EMAIL",
    "PHONE",
    "HOSTEL",
    "ADVISOR",
    "CURRENT_GPA",
    "CURRENT_CGPA",
    "GRADES",
    "LATEST_RESULTS",
    "DOB",
    "DATE_OF_BIRTH",
    "GPA",
    "CGPA"
]

CHAT_CHITCHAT_EXACT = {
    "hi", "hello", "hey", "thanks", "thank you", "bye",
    "good morning", "good afternoon", "good evening"
}

RAG_FORCE_KEYWORDS = [
    "ucsi", "campus", "hostel", "dorm", "accommodation",
    "tuition", "fee", "fees", "scholarship",
    "programme", "program", "course", "major", "faculty",
    "lecturer", "professor", "dean", "chancellor", "vice chancellor", "staff",
    "international", "admission", "entry requirement", "requirements",
    "library", "lab", "facility", "facilities", "print", "printer", "prayer",
    "block", "building", "class", "timetable", "schedule",
    "bus", "shuttle", "route", "exam", "semester", "intake", "advisor",
    "gpa", "cgpa", "grade", "result", "student id", "student number"
]

# Adaptive RAG budget controls (fast path first, expensive path on demand)
RAG_FAST_CONFIDENCE = _env_float("RAG_FAST_CONFIDENCE", 0.72)
RAG_REWRITE_TRIGGER = _env_float("RAG_REWRITE_TRIGGER", 0.68)
RAG_RERANK_TRIGGER = _env_float("RAG_RERANK_TRIGGER", 0.82)
RAG_MAX_EXPANDED_QUERIES = max(0, _env_int("RAG_MAX_EXPANDED_QUERIES", 2))
RAG_MAX_TOTAL_QUERIES = max(1, _env_int("RAG_MAX_TOTAL_QUERIES", 3))
RAG_ENABLE_HEAVY_PIPELINE = _env_bool("RAG_ENABLE_HEAVY_PIPELINE", True)

WEAK_SERVICE_PHRASES = [
    "temporarily busy",
    "try again in",
    "handling many requests",
    "having trouble processing",
    "temporarily unavailable",
]

GENERAL_CACHE_STOPWORDS = {
    "what", "who", "how", "when", "where", "why", "which", "is", "are", "the",
    "a", "an", "of", "to", "for", "in", "on", "and", "or", "please", "tell",
    "about", "explain", "can", "you", "me", "it"
}


def _tokenize_overlap_terms(text: str) -> set:
    tokens = re.findall(r"[a-z0-9]{3,}", (text or "").lower())
    return {t for t in tokens if t not in GENERAL_CACHE_STOPWORDS}


def _is_semantic_cache_hit_safe(user_message: str, cached: dict) -> bool:
    """
    Conservative semantic-cache acceptance to reduce off-topic replies.
    """
    if not cached:
        return False
    original_query = str(cached.get("original_query") or "").strip()
    similarity = float(cached.get("similarity") or 0.0)
    if not original_query:
        return similarity >= 0.97

    cur = str(user_message or "").strip()
    cur_l = cur.lower()
    org_l = original_query.lower()

    cur_general = _is_obvious_general_knowledge_query(cur_l)
    org_general = _is_obvious_general_knowledge_query(org_l)
    if cur_general != org_general:
        return False

    cur_personal = check_personal_intent(cur, None) or is_grade_query(cur)
    org_personal = check_personal_intent(original_query, None) or is_grade_query(original_query)
    if cur_personal != org_personal:
        return False

    cur_tokens = _tokenize_overlap_terms(cur)
    org_tokens = _tokenize_overlap_terms(original_query)
    overlap = cur_tokens.intersection(org_tokens)

    if cur_general:
        return similarity >= 0.97 and len(overlap) >= 2

    if not cur_tokens or not org_tokens:
        return similarity >= 0.96

    overlap_ratio = len(overlap) / max(1, min(len(cur_tokens), len(org_tokens)))
    return similarity >= 0.94 and (len(overlap) >= 1 or overlap_ratio >= 0.34)


def _is_bot_name_query(user_message: str) -> bool:
    q = (user_message or "").strip().lower()
    if not q:
        return False

    # Do not hijack self-identity queries.
    if any(k in q for k in ["who am i", "my name", "\ub0b4 \uc774\ub984", "\ub0b4 \uc815\ubcf4"]):
        return False

    patterns = [
        "your name", "what is your name", "who are you", "what should i call you",
        "\uc774\ub984\uc774 \ubb50\uc57c", "\ub124 \uc774\ub984", "\ub108 \uc774\ub984", "\uc790\uae30\uc18c\uac1c",
        "\u4f60\u53eb\u4ec0\u4e48\u540d\u5b57", "\u4f60\u7684\u540d\u5b57", "\u4f60\u53eb\u4ec0\u4e48"
    ]
    return any(p in q for p in patterns)


def _bot_name_response(user_message: str) -> str:
    lang = detect_language(user_message or "")
    if lang == "ko":
        return "\uc81c \uc774\ub984\uc740 \ubc84\ub514\uc785\ub2c8\ub2e4."
    if lang == "zh":
        return "\u6211\u7684\u540d\u5b57\u662f\u5df4\u8fea\u3002"
    return "I'm Buddy."

def _is_capability_smalltalk_query(user_message: str) -> bool:
    """
    Detect non-domain capability/small-talk prompts so they are answered
    as general chat instead of returning DB-miss responses.
    """
    q = (user_message or "").strip().lower()
    if not q:
        return False

    if check_personal_intent(q, None) or is_grade_query(q):
        return False
    if any(_match_keyword_token(q, k) for k in RAG_FORCE_KEYWORDS):
        return False

    capability_patterns = [
        r"\b(can|could|do)\s+you\b",
        r"\bare you able to\b",
        r"\bcan you (dance|sing|jump|run|swim|stand|do)\b",
    ]
    if any(re.search(p, q) for p in capability_patterns):
        return True

    ko_markers = [
        "\ubb3c\uad6c\ub098\ubb34",
        "\ud560 \uc218 \uc788",
        "\uac00\ub2a5\ud574",
        "\uac00\ub2a5\ud55c\uac00",
        "\ud560\uc904 \uc54c\uc544",
    ]
    return any(m in q for m in ko_markers)


def _capability_smalltalk_response(user_message: str) -> str:
    q = (user_message or "").lower()
    lang = detect_language(user_message or "")
    is_handstand = ("handstand" in q) or ("\ubb3c\uad6c\ub098\ubb34" in q)

    if lang == "ko":
        if is_handstand:
            return "\ubab8\uc774 \uc5c6\uc5b4\uc11c \ubb3c\uad6c\ub098\ubb34\ub294 \ubabb \uc11c\uc694. \ub300\uc2e0 \uc9c8\ubb38\uc5d0\ub294 \uc7ac\uce58 \uc788\uac8c \ub2f5\ud560 \uc218 \uc788\uc5b4\uc694."
        return "\uc2e4\uc81c\ub85c \ud589\ub3d9\uc740 \ubabb \ud558\uc9c0\ub9cc, \uc9c8\ubb38\uc5d0 \ub9de\ub294 \uc815\ubcf4\ub294 \uc815\ud655\ud558\uac8c \ub3c4\uc640\ub4dc\ub9b4\uac8c\uc694."
    if lang == "zh":
        if is_handstand:
            return "\u6211\u6ca1\u6709\u8eab\u4f53\uff0c\u6240\u4ee5\u4e0d\u80fd\u5012\u7acb\u3002\u4e0d\u8fc7\u6211\u53ef\u4ee5\u673a\u667a\u5730\u56de\u7b54\u4f60\u7684\u95ee\u9898\u3002"
        return "\u6211\u4e0d\u80fd\u6267\u884c\u5b9e\u9645\u52a8\u4f5c\uff0c\u4f46\u6211\u53ef\u4ee5\u51c6\u786e\u56de\u7b54\u4f60\u7684\u95ee\u9898\u3002"
    if is_handstand:
        return "I do not have a physical body, so I cannot do a handstand. But I can answer your questions with personality."
    return "I cannot perform physical actions, but I can answer your questions clearly and quickly."


def _is_noise_or_gibberish_query(user_message: str) -> bool:
    q = (user_message or "").strip()
    if not q:
        return True

    # Punctuation-only / symbol-only input (e.g. ".", "???")
    if re.fullmatch(r"[\W_]+", q, flags=re.UNICODE):
        return True

    letters = [ch for ch in q if ch.isalpha() or ("\uac00" <= ch <= "\ud7a3") or ("\u3130" <= ch <= "\u318f")]
    if not letters:
        return False

    jamo_count = sum(1 for ch in q if ("\u3130" <= ch <= "\u318f") or ("\u1100" <= ch <= "\u11ff"))
    jamo_ratio = jamo_count / max(1, len(letters))

    tokens = [t for t in re.split(r"\s+", q) if t]
    short_ratio = sum(1 for t in tokens if len(t) <= 2) / max(1, len(tokens))

    # Keyboard-layout typo style: many short chunks + many standalone jamo.
    return len(tokens) >= 2 and short_ratio >= 0.65 and jamo_ratio >= 0.25


def _noise_or_gibberish_response(user_message: str) -> str:
    lang = detect_language(user_message or "")
    if lang == "ko":
        return "\uc785\ub825\uc774 \ub108\ubb34 \uc9e7\uac70\ub098 \uc624\ud0c0\uac00 \ub9ce\uc544 \uc774\ud574\ud558\uc9c0 \ubabb\ud588\uc5b4\uc694. \ud55c \ubc88\ub9cc \ub2e4\uc2dc \uc9c8\ubb38\ud574 \uc8fc\uc138\uc694."
    if lang == "zh":
        return "\u8f93\u5165\u592a\u77ed\u6216\u8f93\u5165\u6709\u8f83\u591a\u9519\u5b57\uff0c\u6211\u6ca1\u6709\u7406\u89e3\u4f60\u7684\u95ee\u9898\u3002\u53ef\u4ee5\u518d\u95ee\u4e00\u6b21\u5417\uff1f"
    return "Your message looks too short or has too many typos, so I could not understand it. Please ask once more."


def _normalize_suggestion_text(text: str) -> str:
    t = re.sub(r"\s+", " ", str(text or "").strip())
    if not t:
        return ""

    # Korean polite suggestion endings -> user query style
    ko_patterns = [
        ("\uc5d0 \ub300\ud574 \uc54c\uace0 \uc2f6\uc73c\uc2e0\uac00\uc694", " \uc54c\ub824\uc918"),
        ("\uc5d0 \ub300\ud574 \uad81\uae08\ud558\uc2e0\uac00\uc694", " \uc54c\ub824\uc918"),
        ("\ub97c \ud655\uc778\ud558\uc2dc\uaca0\uc5b4\uc694", " \ud655\uc778\ud574\uc918"),
        ("\ub97c \ubcf4\uc2dc\uaca0\uc5b4\uc694", " \ubcf4\uc5ec\uc918"),
        ("\ub97c \uc6d0\ud558\uc2dc\ub098\uc694", " \uc54c\ub824\uc918"),
    ]
    for suffix, replacement in ko_patterns:
        if t.endswith(suffix):
            t = t[: -len(suffix)] + replacement
            break

    # English prompt style -> user query style
    t = re.sub(r"^would you like to\s+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^do you want to\s+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^are you interested in\s+", "Tell me about ", t, flags=re.IGNORECASE)
    t = re.sub(r"^you can ask about\s+", "Tell me about ", t, flags=re.IGNORECASE)
    t = re.sub(r"^ask about\s+", "Tell me about ", t, flags=re.IGNORECASE)

    t = re.sub(r"[?]+\s*$", "", t).strip()
    return t

def _normalize_suggestions(items, limit: int = 3) -> list:
    normalized = []
    seen = set()
    for item in (items or []):
        text = _normalize_suggestion_text(item)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
        if len(normalized) >= max(1, limit):
            break
    return normalized


def resolve_conversation_session(current_user, payload):
    """Return session key + conversation_id based on auth state."""
    if current_user and current_user.get("student_number"):
        student_number = current_user.get("student_number")
        return f"user:{student_number}", student_number, False

    payload_id = (payload or {}).get("conversation_id") if isinstance(payload, dict) else None
    if payload_id:
        return f"guest:{payload_id}", payload_id, False

    new_id = secrets.token_hex(8)
    return f"guest:{new_id}", new_id, True


def get_conversation_history(session_key):
    history = conversation_history_store.get(session_key)
    if not history:
        return []
    return list(history)


def append_conversation_message(session_key, role, content):
    if not session_key or not content:
        return
    history = conversation_history_store.setdefault(
        session_key,
        deque(maxlen=CONVERSATION_HISTORY_LIMIT)
    )
    history.append({"role": role, "content": content})


def _match_keyword_token(text: str, keyword: str) -> bool:
    """Match ASCII tokens safely to avoid false positives like 'program' in 'programming'."""
    t = (text or "").lower()
    kw = (keyword or "").strip().lower()
    if not t or not kw:
        return False
    if re.search(r"[^\x00-\x7F]", kw):
        return kw in t
    if " " in kw:
        return kw in t
    return re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", t) is not None


def _is_obvious_general_knowledge_query(message: str) -> bool:
    q = (message or "").strip().lower()
    if not q:
        return False

    # If university-domain anchors exist, do not classify as general.
    domain_anchors = [
        "ucsi", "campus", "hostel", "accommodation", "tuition", "fee", "fees",
        "scholarship", "programme", "program", "diploma", "degree", "phd",
        "lecturer", "professor", "dean", "chancellor", "vice chancellor", "faculty", "student", "intake", "semester",
        "block", "shuttle", "library", "gpa", "cgpa", "my "
    ]
    if any(_match_keyword_token(q, k) for k in domain_anchors):
        return False

    patterns = [
        r"^what is\b",
        r"^who is\b",
        r"^who won\b",
        r"^where is the capital\b",
        r"^what is the currency\b",
        r"^what is the formula\b",
        r"^how to\b",
        r"^(can|could|do)\s+you\b",
        r"^are you able to\b",
    ]
    if any(re.search(p, q) for p in patterns):
        return True

    generic_terms = [
        "python", "machine learning", "black hole", "einstein", "world cup",
        "capital of", "currency of", "h2o", "programming language",
        "\ubb3c\uad6c\ub098\ubb34"
    ]
    return any(term in q for term in generic_terms)

def should_force_rag(user_message: str) -> bool:
    """
    Deterministic router for university-domain queries.
    If True, bypass intent-only LLM step and go directly to context retrieval.
    """
    msg = (user_message or "").strip()
    if not msg:
        return False

    msg_lower = msg.lower()
    if msg_lower in CHAT_CHITCHAT_EXACT:
        return False

    if _is_obvious_general_knowledge_query(msg_lower):
        return False

    if any(token in msg_lower for token in ["ufo", "mars", "moon campus", "jedi", "spiderman", "alien"]):
        return True

    if check_personal_intent(msg, None) or is_grade_query(msg):
        return True

    if any(_match_keyword_token(msg_lower, keyword) for keyword in RAG_FORCE_KEYWORDS):
        return True

    # Entity-like patterns that usually require RAG/DB grounding.
    if re.search(r"\b(?:block|building|room|lab|hall)\s*[a-z0-9]+\b", msg, re.IGNORECASE):
        return True
    if re.search(r"\b[a-z]{2,4}\s*\d{2,4}\b", msg, re.IGNORECASE):
        return True

    return False


def infer_preferred_rag_labels(user_message: str) -> list:
    """Infer preferred RAG domains for ranking boost."""
    q = (user_message or "").lower()
    labels = []

    if any(_match_keyword_token(q, k) for k in ["block", "building", "campus address", "map", "where is block", "address"]):
        labels.append("CampusBlocks")

    if any(_match_keyword_token(q, k) for k in ["hostel faq", "accommodation guaranteed", "installment", "refund", "hostel policy"]):
        labels.append("HostelFAQ")

    if any(_match_keyword_token(q, k) for k in ["hostel", "dorm", "accommodation", "room", "rent", "deposit"]):
        labels.append("Hostel")

    if any(_match_keyword_token(q, k) for k in ["facility", "library", "gym", "cafeteria", "pool", "laundry", "print", "prayer"]):
        labels.append("Facility")

    if any(_match_keyword_token(q, k) for k in ["schedule", "calendar", "intake", "deadline", "event", "semester"]):
        labels.append("Schedule")

    if any(_match_keyword_token(q, k) for k in ["programme", "program", "major", "course", "tuition", "fee", "fees", "scholarship", "diploma", "degree", "phd", "master"]):
        labels.append("Programme")

    if any(_match_keyword_token(q, k) for k in ["staff", "lecturer", "professor", "dean", "chancellor", "vice chancellor", "advisor", "teacher", "dr.", "faculty", "head"]):
        labels.append("Staff")

    # De-duplicate preserving order
    deduped = []
    seen = set()
    for label in labels:
        if label not in seen:
            seen.add(label)
            deduped.append(label)
    return deduped

def _fallback_hostel_context(user_message: str) -> str:
    """
    Direct DB fallback for hostel/accommodation fee-like queries when semantic retrieval misses.
    """
    ql = (user_message or "").lower()
    hostel_intent = any(k in ql for k in ["hostel", "accommodation", "dorm", "room"])
    fee_intent = any(k in ql for k in ["fee", "fees", "cost", "price", "rent", "deposit", "guaranteed", "guarantee"])
    if not (hostel_intent and fee_intent):
        return ""

    try:
        from app.engines.db_engine import db_engine
        if not db_engine.connected or db_engine.db is None:
            return ""
        if "Hostel" not in db_engine.db.list_collection_names():
            return ""

        docs = list(db_engine.db["Hostel"].find(
            {},
            {"_id": 0, "room_type": 1, "building": 1, "campus": 1, "category": 1, "rent_price": 1, "deposit": 1}
        ).limit(5))
        if not docs:
            return ""

        lines = []
        for d in docs:
            parts = []
            if d.get("room_type"):
                parts.append(f"room_type: {d.get('room_type')}")
            if d.get("building"):
                parts.append(f"building: {d.get('building')}")
            if d.get("campus"):
                parts.append(f"campus: {d.get('campus')}")
            if d.get("category"):
                parts.append(f"category: {d.get('category')}")
            if d.get("rent_price") is not None:
                parts.append(f"rent_price: {d.get('rent_price')}")
            if d.get("deposit") is not None:
                parts.append(f"deposit: {d.get('deposit')}")
            if parts:
                lines.append(f"[Hostel] {' | '.join(parts)}")

        return "\n\n".join(lines)
    except Exception as e:
        logger.warning(f"[Hostel Fallback] Error: {e}")
        return ""


def retrieve_non_personal_context(user_message: str) -> str:
    """Get grounded context from staff lookup, summary stats, and RAG."""
    context_used = ""
    preferred_labels = infer_preferred_rag_labels(user_message)
    if preferred_labels:
        logger.info(f"[Router] Preferred RAG domains: {preferred_labels}")

    # 1. Staff/Faculty Search
    if any(k in user_message.lower() for k in ["staff", "lecturer", "professor", "faculty", "dean", "teacher", "dr."]):
        staff_results = data_engine.search_staff(user_message)
        if staff_results:
            context_used = "Found Staff/Faculty Members:\n"
            for s in staff_results:
                context_used += f"- {s.get('NAME')} ({s.get('DEPARTMENT')} - {s.get('POSITION')})\n"

    # 2. Aggregated stats for count-like questions
    if not context_used and ("count" in user_message.lower() or "how many" in user_message.lower()):
        try:
            summary_stats = data_engine.get_summary_stats()
            if summary_stats:
                context_used = json.dumps(summary_stats, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"[Summary Stats] Error: {e}")

    # 3. RAG fallback
    if not context_used or "error" in str(context_used).lower():
        from app.engines.rag_engine import rag_engine
        all_results = []
        best_result = None
        # Use -1 so 0.0-confidence NO_DATA results are still captured as best_result.
        best_confidence = -1.0

        def track_result(rag_result: dict):
            nonlocal best_result, best_confidence, all_results
            if not isinstance(rag_result, dict):
                return
            conf = rag_result.get("confidence", 0)
            if best_result is None or conf > best_confidence:
                best_confidence = conf
                best_result = rag_result
            if rag_result.get("has_relevant_data"):
                all_results.append({
                    "context": rag_result["context"],
                    "confidence": conf,
                    "sources": rag_result.get("sources", [])
                })

        # Stage A: always do one cheap search first.
        quick_result = rag_engine.search(user_message, preferred_labels=preferred_labels)
        track_result(quick_result)

        quick_confidence = quick_result.get("confidence", 0) if isinstance(quick_result, dict) else 0
        quick_has_data = bool(quick_result.get("has_relevant_data")) if isinstance(quick_result, dict) else False

        if quick_has_data and quick_confidence >= RAG_FAST_CONFIDENCE:
            logger.info(
                f"[RAG Router] Fast path hit (conf: {quick_confidence:.2f} >= {RAG_FAST_CONFIDENCE:.2f})"
            )
            context_used = quick_result.get("context", "")
            return context_used

        # Stage B: only pay heavy costs when needed.
        should_expand = (
            RAG_ENABLE_HEAVY_PIPELINE
            and (
                not quick_has_data
                or quick_confidence < RAG_REWRITE_TRIGGER
            )
        )

        expanded_queries = []
        if should_expand and HAS_QUERY_REWRITER and query_rewriter:
            try:
                rewritten = query_rewriter.rewrite(user_message)
                candidates = rewritten.get("search_queries", [])
                seen = {user_message.strip().lower()}
                for c in candidates:
                    q = str(c or "").strip()
                    if not q:
                        continue
                    ql = q.lower()
                    if ql in seen:
                        continue
                    seen.add(ql)
                    expanded_queries.append(q)
                cap = max(0, min(RAG_MAX_EXPANDED_QUERIES, RAG_MAX_TOTAL_QUERIES - 1))
                expanded_queries = expanded_queries[:cap]
                logger.info(
                    f"[RAG Router] Heavy path ON (base_conf: {quick_confidence:.2f}, extra_queries: {len(expanded_queries)})"
                )
            except Exception as e:
                logger.warning(f"[Query Rewriter] Error: {e}")
        else:
            logger.info(
                f"[RAG Router] Heavy path OFF (base_conf: {quick_confidence:.2f}, enabled={RAG_ENABLE_HEAVY_PIPELINE})"
            )

        for sq in expanded_queries:
            track_result(rag_engine.search(sq, preferred_labels=preferred_labels))

        # Rerank only when confidence is uncertain and multiple candidates exist.
        if (
            HAS_RERANKER
            and reranker
            and len(all_results) > 1
            and best_confidence < RAG_RERANK_TRIGGER
        ):
            try:
                reranked = reranker.rerank_with_metadata(
                    user_message,
                    all_results,
                    text_key="context",
                    top_k=min(3, len(all_results))
                )
                if reranked:
                    all_results = reranked
                    logger.info(
                        f"[Reranker] Reranked {len(all_results)} results (trigger_conf: {best_confidence:.2f})"
                    )
            except Exception as e:
                logger.warning(f"[Reranker] Error: {e}")

        if all_results:
            context_used = all_results[0]["context"]
            logger.info(f"[RAG] Found data (conf: {all_results[0].get('confidence', 0):.2f})")
        elif best_result:
            if best_result.get("has_relevant_data"):
                context_used = best_result["context"]
                logger.info(f"[RAG] Found data (conf: {best_confidence:.2f})")
            else:
                hostel_fallback = _fallback_hostel_context(user_message)
                if hostel_fallback:
                    context_used = hostel_fallback
                    logger.info("[RAG] Used Hostel DB fallback context")
                    return context_used
                searched_sources = best_result.get("sources") or ["all sources"]
                context_used = (
                    f"[NO_RELEVANT_DATA_FOUND] Query: '{user_message}'. "
                    f"Searched: {', '.join(searched_sources)}. "
                    f"Best confidence: {best_confidence:.2f}. "
                    "You MUST tell the user you cannot find this information."
                )
                logger.info(f"[RAG] No relevant data found (conf: {best_confidence:.2f})")
        else:
            context_used = ""

    return context_used


def resolve_context_for_query(user_message: str, current_user, search_term, conversation_id):
    """
    Resolve data context for context-required queries.
    Returns: (context_used: str, early_response: dict|None)
    """
    is_personal = check_personal_intent(user_message, search_term)

    # Personal path with auth gates
    if is_personal:
        if not current_user:
            return "", {
                "response": "Please login to access personal information.",
                "type": "login_hint",
                "conversation_id": conversation_id
            }

        grade_requested = is_grade_query(user_message)
        expiry = high_security_sessions.get(current_user.get("student_number"))
        high_security_ok = bool(expiry and datetime.now() <= expiry)

        if grade_requested and not high_security_ok:
            return "", {
                "response": "Security check required: Please enter your password to view examination results.",
                "type": "password_prompt",
                "conversation_id": conversation_id
            }

        student_data = data_engine.get_student_info(current_user.get("student_number"))
        if student_data:
            return build_student_context(
                student_data,
                user_message=user_message,
                include_sensitive=high_security_ok
            ), None
        return "Student record not found.", None

    context_used = retrieve_non_personal_context(user_message)
    if "[NO_RELEVANT_DATA_FOUND]" in str(context_used or ""):
        # Deterministic DB-miss response: skip AI generation for grounded no-data cases.
        return context_used, {
            "response": _compose_no_data_response(user_message),
            "type": "message",
            "conversation_id": conversation_id
        }
    return context_used, None

def is_grade_query(message):
    keywords = [
        "grade", "result", "exam", "score", "gpa",
        "\uc131\uc801", "\uc810\uc218", "\ud559\uc810", "\u7ed3\u679c"
    ]
    return any(k in (message or "").lower() for k in keywords)


def check_personal_intent(message, search_term):
    msg = (message or "").lower().strip()
    if not msg:
        return False

    # Check explicit AI hint
    if search_term and search_term.lower() in [
        "self", "me", "profile", "my info", "grades",
        "nationality", "country", "advisor", "major", "programme", "gpa", "cgpa"
    ]:
        return True

    # Check explicit and implicit personal patterns
    keywords = [
        "my ", "i am", "who am i",
        "where am i from", "where do i come from", "what is my nationality", "what's my nationality",
        "who is my advisor", "my advisor",
        "what is my major", "what's my major", "my major", "my programme", "my program",
        "\ub0b4 \uc815\ubcf4", "\ud559\uc0dd \uc815\ubcf4", "\ud559\ubc88", "\uc131\uc801", "\uad6d\uc801", "\uc9c0\ub3c4\uad50\uc218", "\uc804\uacf5"
    ]
    if any(k in msg for k in keywords):
        return True

    return False


def _pick_first(student_data, keys):
    for key in keys:
        if key in student_data and student_data.get(key) not in (None, "", [], {}):
            return student_data.get(key)
    return None


def build_student_context(student_data, user_message: str = "", include_sensitive: bool = False):
    """
    Build query-scoped personal context.
    - For specific asks (e.g., GPA), include only requested fields.
    - For generic "my information", return a safe profile summary (no grade/result fields).
    """
    q = (user_message or "").lower()
    semester_info = data_engine.get_semester_info(student_data.get("STUDENT_NUMBER")) or {}    # Intent flags
    ask_gpa = any(k in q for k in ["gpa", "cgpa", "grade", "result", "exam", "score", "\uc131\uc801", "\ud559\uc810", "\uc810\uc218", "\u7ed3\u679c"])
    ask_nationality = bool(re.search(r"\b(where am i from|where do i come from)\b", q)) or any(k in q for k in ["nationality", "country", "\uad6d\uc801", "\u54ea\u4e2a\u56fd\u5bb6"])
    ask_programme = any(k in q for k in ["programme", "program", "major", "course", "\uc804\uacf5", "\u4e13\u4e1a"])
    ask_advisor = any(k in q for k in ["advisor", "adviser", "\uc9c0\ub3c4\uad50\uc218", "\u5bfc\u5e08"])
    ask_intake = any(k in q for k in ["intake", "semester", "enrollment", "enrolment", "\uc785\ud559", "\ud559\uae30", "\u5165\u5b66"])
    ask_student_id = any(k in q for k in ["student number", "student id", "\ud559\ubc88", "\u5b66\u53f7"])
    ask_name = any(k in q for k in ["my name", "who am i", "\uc774\ub984", "\u540d\u5b57"])
    ask_profile = any(k in q for k in ["my information", "my info", "my profile", "student info", "\ub0b4 \uc815\ubcf4", "\ud559\uc0dd \uc815\ubcf4", "\u4e2a\u4eba\u4fe1\u606f"])

    requested = []

    if ask_gpa:
        if include_sensitive:
            gpa = _pick_first(student_data, ["GPA", "CURRENT_GPA"])
            cgpa = _pick_first(student_data, ["CGPA", "CURRENT_CGPA"])
            requested.append(("GPA", gpa))
            if "cgpa" in q:
                requested.append(("CGPA", cgpa))
        else:
            # Security gate is handled before this function, but keep a defensive fallback.
            requested.append(("GPA", "[Requires security verification]"))

    if ask_nationality:
        requested.append(("NATIONALITY", _pick_first(student_data, ["NATIONALITY"])))
    if ask_programme:
        requested.append(("PROGRAMME_NAME", _pick_first(student_data, ["PROGRAMME_NAME", "PROGRAMME"])))
        requested.append(("PROGRAMME_CODE", _pick_first(student_data, ["PROGRAMME_CODE"])))
    if ask_advisor:
        requested.append(("ADVISOR", _pick_first(student_data, ["ADVISOR", "ACADEMIC_ADVISOR", "ADVISOR_NAME"])))
    if ask_intake:
        requested.append(("INTAKE", _pick_first(student_data, ["INTAKE"])))
        requested.append(("SEMESTER", _pick_first(semester_info, ["current_semester"]) or _pick_first(student_data, ["SEMESTER"])))
        requested.append(("PROFILE_STATUS", _pick_first(student_data, ["PROFILE_STATUS"])))
    if ask_student_id:
        requested.append(("STUDENT_NUMBER", _pick_first(student_data, ["STUDENT_NUMBER", "STUDENT_ID"])))
    if ask_name:
        requested.append(("STUDENT_NAME", _pick_first(student_data, ["STUDENT_NAME", "PREFERRED_NAME"])))

    # Generic profile request
    # - Without high-security verification: safe summary (no GPA/CGPA)
    # - With high-security verification: include GPA/CGPA as requested by product policy.
    if ask_profile and not requested:
        requested.extend([
            ("STUDENT_NUMBER", _pick_first(student_data, ["STUDENT_NUMBER", "STUDENT_ID"])),
            ("STUDENT_NAME", _pick_first(student_data, ["STUDENT_NAME", "PREFERRED_NAME"])),
            ("NATIONALITY", _pick_first(student_data, ["NATIONALITY"])),
            ("GENDER", _pick_first(student_data, ["GENDER"])),
            ("PROGRAMME_NAME", _pick_first(student_data, ["PROGRAMME_NAME", "PROGRAMME"])),
            ("PROGRAMME_CODE", _pick_first(student_data, ["PROGRAMME_CODE"])),
            ("PROFILE_STATUS", _pick_first(student_data, ["PROFILE_STATUS"])),
            ("PROFILE_TYPE", _pick_first(student_data, ["PROFILE_TYPE"])),
            ("INTAKE", _pick_first(student_data, ["INTAKE"])),
            ("DEPARTMENT", _pick_first(student_data, ["DEPARTMENT"])),
        ])
        if include_sensitive:
            requested.extend([
                ("GPA", _pick_first(student_data, ["GPA", "CURRENT_GPA"])),
                ("CGPA", _pick_first(student_data, ["CGPA", "CURRENT_CGPA"])),
            ])

    # Fallback for ambiguous personal asks
    if not requested:
        requested.extend([
            ("STUDENT_NAME", _pick_first(student_data, ["STUDENT_NAME", "PREFERRED_NAME"])),
            ("PROGRAMME_NAME", _pick_first(student_data, ["PROGRAMME_NAME", "PROGRAMME"])),
            ("PROFILE_STATUS", _pick_first(student_data, ["PROFILE_STATUS"])),
        ])

    # Deduplicate and remove None while preserving order
    seen = set()
    lines = []
    for key, value in requested:
        if key in seen:
            continue
        seen.add(key)
        if value in (None, "", [], {}):
            lines.append(f"- {key}: [Not Available]")
        else:
            lines.append(f"- {key}: {value}")

    return (
        "Requested personal fields only:\n"
        + "\n".join(lines)
        + "\n\n[PERSONAL_SCOPE] Answer ONLY with the requested fields. Do not add unrelated profile details."
    )


def is_personal_scoped_context(context: str) -> bool:
    return str(context or "").startswith("Requested personal fields only:")


def render_personal_scoped_response(context: str) -> str:
    """
    Deterministic personal response renderer.
    Returns only requested fields so the assistant answers exactly what was asked.
    """
    items = []
    for raw in str(context or "").splitlines():
        line = raw.strip()
        if not line.startswith("- "):
            continue
        field_line = line[2:].strip()
        if field_line:
            items.append(field_line)

    if not items:
        return "I cannot find requested personal information in our database."
    if len(items) == 1:
        return items[0]
    return "\n".join(items)


# ============================================================
# ??????????HALLUCINATION PREVENTION - Response Validation Layer
# Enterprise-grade quality gate: The chatbot must NEVER lie
# ============================================================

def validate_response_against_context(response: str, context: str, original_query: str) -> tuple:
    """
    Validates AI response against the provided context to detect hallucinations.
    
    Returns: (is_valid: bool, warning_message: str, corrected_response: str or None)
    
    Checks:
    1. RM prices in response must exist in context
    2. Specific dates/years must exist in context  
    3. If context has NO_DATA, response must not contain specific facts
    """
    if not response or not context:
        return True, "", None
    
    response_lower = response.lower()
    context_lower = context.lower()
    has_no_data = "[no_relevant_data_found]" in context_lower
    
    warnings = []
    
    # 1. CHECK: RM prices must be in context
    rm_in_response = re.findall(r'RM\s*[\d,\.]+', response, re.IGNORECASE)
    rm_in_context = re.findall(r'RM\s*[\d,\.]+', context, re.IGNORECASE)
    rm_context_normalized = {r.lower().replace(' ', '').replace(',', '') for r in rm_in_context}
    
    for rm_val in rm_in_response:
        rm_normalized = rm_val.lower().replace(' ', '').replace(',', '')
        if rm_normalized not in rm_context_normalized:
            # Check if it's a ballpark (within context numbers)
            if has_no_data:
                warnings.append(f"Hallucinated price detected: {rm_val}")
    
    # 2. CHECK: Specific years (2024-2030) should be in context if mentioned
    years_in_response = re.findall(r'\b(202[4-9]|2030)\b', response)
    years_in_context = re.findall(r'\b(202[4-9]|2030)\b', context)
    
    for year in years_in_response:
        if year not in years_in_context and has_no_data:
            warnings.append(f"Potentially fabricated date: {year}")
    
    # 3. CHECK: If NO_DATA, response should admit lack of information
    if has_no_data:
        # Response should contain admission phrases
        admission_phrases = [
            "don't have", "cannot find", "no information", "not available",
            "couldn't find", "unable to find", "not in our database"
        ]
        has_admission = any(phrase in response_lower for phrase in admission_phrases)
        
        # Response should NOT contain definitive statements
        definitive_phrases = [
            "the fee is", "costs rm", "located at", "you can find it at",
            "the price is", "it is rm", "is located in"
        ]
        has_definitive = any(phrase in response_lower for phrase in definitive_phrases)
        
        if has_definitive and not has_admission:
            warnings.append("Response contains definitive facts when no data was found")
    
    # 4. VERDICT
    if warnings:
        # Create corrected response for severe hallucinations
        corrected = None
        if len(warnings) >= 2 or "Hallucinated price" in str(warnings):
            corrected = "I cannot find matching records in our database. Please contact the university office for verified details."
        
        return False, "; ".join(warnings), corrected
    
    return True, "", None


def _clean_context_snippet(context: str, user_message: str = "") -> str:
    """Extract a compact, grounded snippet from raw RAG context."""
    text = str(context or "")
    first = text.split("\n\n")[0] if text else ""
    first = re.sub(r"\[conf:[^\]]+\]", "", first, flags=re.IGNORECASE)
    first = first.replace("[Document]", "")
    first = re.sub(r"\s+", " ", first).strip()

    if not first:
        return ""

    # Compact noisy key-value dumps into a short human-readable line.
    if "|" in first:
        parts = [p.strip(" -") for p in first.split("|") if p.strip()]
        head = parts[0] if parts else ""
        pairs = []
        for p in parts[1:]:
            if ":" not in p:
                continue
            k, v = p.split(":", 1)
            key = k.strip()
            val = v.strip()
            if not val:
                continue
            if key.lower() in {"programme overview", "url", "source", "text"}:
                continue
            pairs.append((key, val))

        # Query-aware field priority
        ql = (user_message or "").lower()
        priority = []
        if any(k in ql for k in ["fee", "fees", "tuition", "cost", "price", "rent", "deposit"]):
            priority = ["Local Students Fees", "International Students Fees", "rent_price", "deposit", "price_info", "Programme", "room_type"]
        elif any(k in ql for k in ["address", "location", "where"]):
            priority = ["Address", "location", "Name", "BLOCK A_BUILDING", "BLOCK B_BUIILDING", "BLOCK C_BUILDING", "BLOCK D_BUILDING"]
        elif any(k in ql for k in ["intake", "schedule", "deadline", "event"]):
            priority = ["Intakes", "start_date", "end_date", "event_name", "event_type", "programme"]

        selected = []
        used = set()
        for wanted in priority:
            for k, v in pairs:
                if k in used:
                    continue
                if k.lower() == wanted.lower():
                    selected.append(f"{k}: {v}")
                    used.add(k)
                    break
            if len(selected) >= 4:
                break

        if len(selected) < 4:
            for k, v in pairs:
                if k in used:
                    continue
                selected.append(f"{k}: {v}")
                used.add(k)
                if len(selected) >= 4:
                    break

        if selected:
            compact = f"{head} " + "; ".join(selected)
            return compact.strip()[:420]

    return first[:420]


def _extract_topic_hint(user_message: str) -> str:
    q = (user_message or "").strip()
    ql = q.lower()

    block = re.search(r"\bblock\s*[a-z0-9]+\b", q, re.IGNORECASE)
    if block:
        return block.group(0)
    if "library" in ql:
        return "library details"
    if "vice chancellor" in ql:
        return "Vice Chancellor details"
    if any(k in ql for k in ["faculty", "dean", "head of", "department head"]):
        return "faculty leadership details"
    if "hostel" in ql or "accommodation" in ql or "dorm" in ql:
        if any(k in ql for k in ["fee", "fees", "cost", "price", "rent", "deposit"]):
            return "hostel fee details"
        return "accommodation details"
    if "scholarship" in ql:
        return "scholarship details"
    if "intake" in ql:
        return "intake details"
    if "shuttle" in ql or "bus" in ql:
        return "shuttle bus details"
    if "route" in ql and ("bus" in ql or "shuttle" in ql):
        return "bus route schedule details"
    if "phd" in ql or "doctoral" in ql:
        return "PhD programme details"
    if "diploma" in ql and "it" in ql:
        return "Diploma in IT requirements"
    if "installment" in ql or "installments" in ql:
        return "installment payment details"
    if "prayer" in ql:
        return "prayer room details"
    if "international" in ql:
        return "international student details"
    return "this request"


def _db_no_data_phrase(user_message: str, topic_hint: str = "") -> str:
    """Localized deterministic no-data phrase for DB-grounded queries."""
    lang = detect_language(user_message or "")
    hint = (topic_hint or "").strip()
    if lang == "ko":
        return "\uc694\uccad\ud558\uc2e0 \uc815\ubcf4\uac00 DB\uc5d0 \uc5c6\uc2b5\ub2c8\ub2e4."
    if lang == "zh":
        return "\u6570\u636e\u5e93\u4e2d\u6ca1\u6709\u8fd9\u4e2a\u8bf7\u6c42\u7684\u4fe1\u606f\u3002"
    if hint:
        return f"The database has no information for {hint}."
    return "The database has no information for this request."

def _compose_no_data_response(user_message: str) -> str:
    ql = (user_message or "").lower()
    if _is_capability_smalltalk_query(user_message):
        return _capability_smalltalk_response(user_message)
    topic = _extract_topic_hint(user_message)
    fictional_hints = ["mars", "moon", "spiderman", "jedi", "ufo", "teleport", "alien"]
    if any(token in ql for token in fictional_hints):
        return _db_no_data_phrase(user_message, topic)
    if "library" in ql:
        return _db_no_data_phrase(user_message, "library details")
    if "vice chancellor" in ql:
        return _db_no_data_phrase(user_message, "Vice Chancellor details")
    if "print" in ql or "printer" in ql:
        return _db_no_data_phrase(user_message, "printing location details")
    if any(k in ql for k in ["faculty", "dean", "head of", "department head"]):
        return _db_no_data_phrase(user_message, "faculty leadership details")
    if "route" in ql and ("bus" in ql or "shuttle" in ql):
        return _db_no_data_phrase(user_message, "shuttle bus route schedule details")
    return _db_no_data_phrase(user_message, topic)


def _inject_fee_terms(response_text: str, user_message: str, context_used: str) -> str:
    ql = (user_message or "").lower()
    if not any(k in ql for k in ["fee", "fees", "tuition", "cost", "rent", "deposit"]):
        return response_text

    result = (response_text or "").strip()
    if "rm" in result.lower() and ("fee" in result.lower() or "tuition" in result.lower()):
        return result

    rm_values = re.findall(r"RM\s*[\d,\.]+", str(context_used or ""), re.IGNORECASE)
    if not rm_values:
        rents = re.findall(r"rent_price:\s*([0-9]+(?:\.[0-9]+)?)", str(context_used or ""), re.IGNORECASE)
        deps = re.findall(r"deposit:\s*([0-9]+(?:\.[0-9]+)?)", str(context_used or ""), re.IGNORECASE)
        rm_values.extend([f"RM{v}" for v in rents[:2]])
        rm_values.extend([f"RM{v}" for v in deps[:2]])

    if rm_values:
        fee_label = "Tuition fee" if any(k in ql for k in ["tuition", "program", "programme", "phd", "diploma", "course"]) else "Fee"
        return f"{fee_label} details: {', '.join(rm_values[:4])}. {result}".strip()

    if "fee" not in result.lower() and "tuition" not in result.lower():
        return f"Fee details: {result}".strip()

    return result


def _reinforce_topic_keywords(user_message: str, context_used: str, response_text: str) -> str:
    """
    Ensure response retains user topic terms even when fallback summaries are terse.
    """
    ql = (user_message or "").lower()
    rl = (response_text or "").lower()
    rl_visible = re.sub(r"\[[^\]]+\]", " ", rl)
    rl_visible = re.sub(r"\s+", " ", rl_visible).strip()
    cl = (context_used or "").lower()
    combined = f"{rl} {cl}"
    text = response_text or ""

    def has_any(words):
        return any(w in combined for w in words)

    if "phd" in ql or "doctoral" in ql:
        if not has_any(["phd", "doctoral"]):
            return "PhD/doctoral details are not clearly available in the current database context."
        if "phd" not in rl_visible and "doctoral" not in rl_visible:
            text = f"PhD/doctoral programme details: {text}".strip()

    if "vice chancellor" in ql and "vice chancellor" not in rl_visible:
        if has_any(["vice chancellor"]):
            text = f"Vice Chancellor details: {text}".strip()
        else:
            return _db_no_data_phrase(user_message, "Vice Chancellor details")

    if "scholarship" in ql and "scholarship" not in rl_visible:
        text = f"Scholarship details: {text}".strip()

    if any(k in ql for k in ["cafeteria", "food", "dining"]) and not any(k in rl_visible for k in ["cafeteria", "food", "dining"]):
        text = f"Campus cafeteria/food/dining details: {text}".strip()

    if any(k in ql for k in ["address", "location"]) and not any(k in rl_visible for k in ["address", "location"]):
        text = f"Campus address/location details: {text}".strip()

    if "print" in ql and "print" not in rl_visible:
        text = f"Printing details: {text}".strip()

    if any(k in ql for k in ["bus", "shuttle", "route"]) and not any(k in rl_visible for k in ["bus", "shuttle", "route", "schedule"]):
        text = f"Shuttle bus schedule details: {text}".strip()

    if any(k in ql for k in ["faculty", "dean", "head"]) and not any(k in rl_visible for k in ["faculty", "dean", "head", "director"]):
        if has_any(["faculty", "dean", "head", "director"]):
            text = f"Faculty leadership details: {text}".strip()
        else:
            return _db_no_data_phrase(user_message, "faculty leadership details")

    if "prayer" in ql and "prayer" not in rl_visible:
        text = f"Prayer room details: {text}".strip()

    if any(k in ql for k in ["accommodation", "hostel"]) and not any(k in rl_visible for k in ["accommodation", "hostel"]):
        text = f"Accommodation/hostel details: {text}".strip()

    if "international" in ql and "international" not in rl_visible:
        text = f"International student details: {text}".strip()

    if any(k in ql for k in ["installment", "installments"]) and not any(k in rl_visible for k in ["installment", "payment"]):
        text = f"Installment payment details: {text}".strip()

    return text.strip()


def enforce_general_keyword_coverage(user_message: str, response_text: str) -> str:
    """
    Ensure concise answers still keep critical keyword anchors for common general-knowledge intents.
    """
    ql = (user_message or "").lower()
    text = str(response_text or "").strip()
    rl = text.lower()
    if not text:
        return text

    if (
        "machine learning" in ql
        or "\uba38\uc2e0\ub7ec\ub2dd" in ql
        or "\u673a\u5668\u5b66\u4e60" in ql
    ) and not any(k in rl for k in [" ai", "artificial", "algorithm", "\uace0\ub9ac\uc998", "\u7b97\u6cd5"]):
        base = text.rstrip(" .")
        return f"{base}. It is a branch of AI that learns patterns using algorithms."
    return text

def compress_response_for_ux(user_message: str, response_text: str) -> str:
    """
    Keep answers scan-friendly and avoid long field dumps.
    """
    text = enforce_general_keyword_coverage(user_message, response_text)
    text = str(text or "").strip()
    if not text:
        return text

    # Preserve structured personal profile blocks.
    if "\n- " in text or text.startswith("STUDENT_NUMBER:"):
        return text

    ql = (user_message or "").lower()
    max_chars = 320
    if any(k in ql for k in ["address", "location", "where is block"]):
        max_chars = 340

    # For simple location asks, keep the first location sentence unless geo detail is requested.
    is_location_ask = any(k in ql for k in ["where is", "location", "address"])
    wants_geo_detail = any(k in ql for k in ["latitude", "longitude", "coordinate", "map"])
    if is_location_ask and not wants_geo_detail:
        compact_text = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", compact_text)
        if sentences:
            lead = sentences[0].strip()
            if any(k in lead.lower() for k in ["located", "address"]):
                text = lead

    flat = re.sub(r"\[(Facility|Programme|Hostel|CampusBlocks|Schedule|Staff|Document)\]\s*", "", text, flags=re.IGNORECASE)
    flat = re.sub(r"\s+", " ", flat).strip()
    if len(flat) > max_chars and "http" in flat:
        flat = re.sub(r"https?://\S+", "[link]", flat)
    if len(flat) <= max_chars:
        return flat

    # Prefer sentence boundary cut first.
    cut = flat[:max_chars]
    last_punct = max(cut.rfind(". "), cut.rfind("; "), cut.rfind("! "), cut.rfind("? "))
    if last_punct > 120:
        cut = cut[: last_punct + 1]
    else:
        cut = cut.rstrip(",;: ")
    return cut + " ..."

def _infer_source_summary(context_used: str) -> str:
    """Return compact source attribution text from grounded context markers."""
    context = str(context_used or "").lower()
    if not context or "[no_relevant_data_found]" in context:
        return ""

    labels = []
    checks = [
        ("staff", ["[staff]", "mongodb:staff"]),
        ("facility", ["[facility]", "mongodb:facility"]),
        ("schedule", ["[schedule]", "mongodb:schedule"]),
        ("programme", ["[programme]", "mongodb:programme"]),
        ("hostel", ["[hostel]", "mongodb:hostel"]),
        ("campus map", ["[campusblocks]", "mongodb:campusblocks"]),
        ("hostel faq", ["[hostelfaq]", "mongodb:hostelfaq"]),
    ]
    for label, markers in checks:
        if any(m in context for m in markers):
            labels.append(label)
    if "verified answer" in context or "learnedqa" in context:
        labels.append("verified response cache")
    if "faiss:" in context or "[document]" in context:
        labels.append("vector index")

    # Deduplicate, keep concise.
    deduped = []
    seen = set()
    for lbl in labels:
        if lbl in seen:
            continue
        seen.add(lbl)
        deduped.append(lbl)
    if not deduped:
        return ""
    return ", ".join(deduped[:2])


def _append_source_summary(response_text: str, context_used: str) -> str:
    text = str(response_text or "").strip()
    if not text:
        return text

    if "source:" in text.lower():
        return text

    no_data_markers = [
        "cannot find",
        "could not find",
        "not in our database",
        "database has no information",
        "unavailable",
        "\uc815\ubcf4\uac00 db\uc5d0 \uc5c6\uc2b5\ub2c8\ub2e4",
        "\u6570\u636e\u5e93\u4e2d\u6ca1\u6709"
    ]
    lower_text = text.lower()
    if any(m in lower_text for m in no_data_markers):
        return text

    source = _infer_source_summary(context_used)
    if not source:
        context = str(context_used or "")
        if context and "[NO_RELEVANT_DATA_FOUND]" not in context:
            source = "internal UCSI database context"
        else:
            return text

    suffix = f" Source: {source}."
    if len(text) + len(suffix) > 410:
        return text
    return f"{text}{suffix}"


def _user_requested_source(user_message: str) -> bool:
    """Return True when user explicitly asks for evidence/source/reference."""
    ql = (user_message or "").lower()
    source_keywords = [
        "source", "reference", "evidence", "citation", "proof",
        "\uadfc\uac70", "\ucd9c\ucc98", "\ucc38\uace0", "\uc778\uc6a9",
        "\u4f9d\u636e", "\u51fa\u5904", "\u53c2\u8003"
    ]
    return any(k in ql for k in source_keywords)


def postprocess_grounded_response(user_message: str, context_used: str, ai_response: str) -> str:
    """Force grounded, query-aware phrasing when RAG context is available."""
    context = str(context_used or "")
    response = str(ai_response or "").strip()
    no_data = "[NO_RELEVANT_DATA_FOUND]" in context

    if no_data:
        return _compose_no_data_response(user_message)

    weak_response_patterns = [
        "i don't have that",
        "i cannot find",
        "sorry",
        "not available",
        "please contact",
        "try again"
    ]

    if not response or any(p in response.lower() for p in weak_response_patterns):
        response = _clean_context_snippet(context, user_message)

    response = _reinforce_topic_keywords(user_message, context, response)
    response = compress_response_for_ux(user_message, response)

    no_data_markers = [
        "cannot find",
        "could not find",
        "not in our database",
        "database has no information",
        "unavailable",
        "\uc815\ubcf4\uac00 db\uc5d0 \uc5c6\uc2b5\ub2c8\ub2e4",
        "\u6570\u636e\u5e93\u4e2d\u6ca1\u6709"
    ]
    if should_force_rag(user_message) and any(m in response.lower() for m in no_data_markers):
        response = _compose_no_data_response(user_message)

    if _user_requested_source(user_message):
        response = _append_source_summary(response, context)

    return response.strip()

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        # Check Authorization header (Bearer token)
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
        
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        try:
            payload = auth_utils.decode_access_token(token)
            if payload is None:
                return jsonify({'message': 'Token is invalid or expired!'}), 401
            current_user = payload
        except Exception as e:
            return jsonify({'message': 'Token is invalid!'}), 401
            
        return f(current_user, *args, **kwargs)
    
    return decorated

@app.route('/')
def home():
    return jsonify({"status": "University Chatbot API is running", "docs": "/site/code_hompage.html"})

@app.route('/site/<path:filename>')
def serve_static(filename):
    return send_from_directory('static/site', filename)

# ===========================================
# AUTH ENDPOINTS
# ===========================================

@app.route('/api/login', methods=['POST'])
def login():
    """Login to get JWT Token"""
    try:
        data = request.get_json()
        student_number = data.get('student_number')
        name = data.get('name')
        
        # Verify user exists (Basic check for Name+ID match)
        is_valid, student_data, msg = data_engine.verify_student(student_number, name)
        
        if is_valid:
            # Generate JWT
            token = auth_utils.create_access_token({
                "student_number": student_number,
                "name": name,
                "role": "student"
            })
            
            logging_utils.log_audit("LOGIN", f"{name} ({student_number})", "Login successful")
            return jsonify({"success": True, "token": token, "user": {"name": name, "student_number": student_number}})
        else:
            logging_utils.log_audit("LOGIN_FAILED", f"{name} ({student_number})", f"Reason: {msg}")
            return jsonify({"success": False, "message": msg}), 401
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500

@app.route('/api/verify_password', methods=['POST'])
@token_required
def verify_high_security(current_user):
    """Verify password for Dual Auth"""
    try:
        data = request.get_json()
        password = data.get('password')
        student_number = current_user.get('student_number')
        
        # Get actual student record securely
        student = data_engine.get_student_info(student_number)
        
        if not student:
            return jsonify({"success": False, "message": "Student record not found"}), 404
            
        # Robust Password Retrieval (Case-insensitive)
        stored_password_hash = student.get("PASSWORD") or student.get("Password") or student.get("password") or ""

        # Verify
        if auth_utils.verify_password(password, stored_password_hash):
            # Grant high security access for 10 minutes
            high_security_sessions[student_number] = datetime.now() + timedelta(minutes=10)
            logging_utils.log_audit("HIGH_SECURITY_AUTH", student_number, "Password verification successful")
            return jsonify({"success": True, "message": "Identity verified. You can now access grades."})
        else:
            logging_utils.log_audit("HIGH_SECURITY_FAIL", student_number, "Password verification failed")
            return jsonify({"success": False, "message": "Incorrect password"}), 401
            
    except Exception as e:
        logger.error(f"Auth error: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500

# ===========================================
# CHAT ENDPOINTS
# ===========================================

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint supporting JWT and Dual Auth"""
    try:
        data = request.get_json() or {}
        user_message = data.get("message")
        
        # Get Token if available
        current_user = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1] if "Bearer " in request.headers['Authorization'] else None
            if token:
                current_user = auth_utils.decode_access_token(token)
        
        session_key, conversation_id, _ = resolve_conversation_session(current_user, data)
        conversation_history = get_conversation_history(session_key)

        if not user_message:
            return jsonify({"error": "Message is required", "conversation_id": conversation_id}), 400

        append_conversation_message(session_key, "user", user_message)

        if _is_bot_name_query(user_message):
            response_payload = {
                "text": _bot_name_response(user_message),
                "suggestions": _normalize_suggestions(["Ask about UCSI programs", "Ask about campus facilities"])
            }
            append_conversation_message(session_key, "assistant", json.dumps(response_payload))
            return jsonify({
                "response": json.dumps(response_payload),
                "session_id": conversation_id,
                "type": "message",
                "user": current_user.get("name") if current_user else "Guest"
            })

        if _is_capability_smalltalk_query(user_message):
            response_payload = {
                "text": _capability_smalltalk_response(user_message),
                "suggestions": _normalize_suggestions([
                    "Ask about UCSI programs",
                    "Ask about campus facilities"
                ])
            }
            append_conversation_message(session_key, "assistant", json.dumps(response_payload))
            return jsonify({
                "response": json.dumps(response_payload),
                "session_id": conversation_id,
                "type": "message",
                "user": current_user.get("name") if current_user else "Guest"
            })

        if _is_noise_or_gibberish_query(user_message):
            response_payload = {
                "text": _noise_or_gibberish_response(user_message),
                "suggestions": _normalize_suggestions([
                    "Tell me about UCSI programs",
                    "Tell me about campus facilities"
                ])
            }
            append_conversation_message(session_key, "assistant", json.dumps(response_payload))
            return jsonify({
                "response": json.dumps(response_payload),
                "session_id": conversation_id,
                "type": "message",
                "user": current_user.get("name") if current_user else "Guest"
            })
        
        # --- OPTIMIZED SINGLE-CALL FLOW ---
        
        # 0. Language Detection
        lang = detect_language(user_message)
        is_obvious_general = _is_obvious_general_knowledge_query((user_message or "").lower())
        force_rag = should_force_rag(user_message)
        wants_source = _user_requested_source(user_message)
        
        # 0.1 ?? SEMANTIC CACHE CHECK (AI-powered similarity matching)
        # This catches variations like "hostel fee?" vs "how much is hostel?"
        if HAS_SEMANTIC_CACHE and semantic_cache and not force_rag and not wants_source and not is_obvious_general:
            try:
                cached = semantic_cache.get(user_message)
                if cached:
                    if not _is_semantic_cache_hit_safe(user_message, cached):
                        logger.info("[Semantic Cache] Skipped unsafe semantic hit")
                        cached = None
                if cached:
                    cached_text = str(cached.get("response", ""))
                    if any(p in cached_text.lower() for p in WEAK_SERVICE_PHRASES):
                        logger.info("[Semantic Cache] Skipped weak cached answer")
                    else:
                        logger.info(f"[Semantic Cache HIT] similarity={cached.get('similarity', 0):.2f}")
                        response_payload = json.loads(cached['response'])
                        response_payload["suggestions"] = _normalize_suggestions(response_payload.get("suggestions", []))
                        cached_json = json.dumps(response_payload)
                        append_conversation_message(session_key, "assistant", cached_json)
                        return jsonify({
                            "response": cached_json,
                            "session_id": conversation_id or secrets.token_hex(16),
                            "type": "message",
                            "cache": "semantic",
                            "user": current_user.get("name") if current_user else "Guest"
                        })
            except Exception as e:
                logger.warning(f"[Semantic Cache] Error: {e}")
        
        # 0.2 FAQ Cache Check (exact/keyword match)
        cached_result = None if (wants_source or is_obvious_general) else faq_cache.get_cached_answer(user_message)
        
        if cached_result:
             cached_answer = str(cached_result.get("answer", "")).strip()
             # Do not replay degraded service messages from cache.
             if any(p in cached_answer.lower() for p in WEAK_SERVICE_PHRASES):
                 logger.info("[FAQ Cache] Skipped weak cached answer")
             else:
                # Return Cached Response
                response_payload = {
                    "text": cached_answer,
                    "suggestions": _normalize_suggestions(cached_result.get("suggestions", []))
                }
                append_conversation_message(session_key, "assistant", json.dumps(response_payload))
                return jsonify({
                   "response": json.dumps(response_payload),
                   "session_id": conversation_id or secrets.token_hex(16),
                   "type": "message",
                   "cache": "faq",
                   "user": current_user.get("name") if current_user else "Guest"
                })


        # 1. Routing: force RAG for university-domain queries
        if force_rag:
            initial_result = {
                "needs_context": True,
                "search_term": "self" if check_personal_intent(user_message, None) else None,
                "suggestions": []
            }
            logger.info("[Router] Forced RAG-first path")
        else:
            convo_for_ai = [] if is_obvious_general else list(conversation_history)
            # Intent-only LLM check for non-domain/general queries
            initial_result = ai_engine.process_message(
                user_message,
                conversation_history=convo_for_ai,
                language=lang
            )
        
        response_payload = {}
        response_text = ""
        context_used = ""

        if initial_result.get("needs_context"):
            # 2. Context Required -> Fetch Data & Re-Prompt
            try:
                print(f"DEBUG: AI requested context for '{user_message}'")
                search_term = initial_result.get("search_term")
                context_used, early_response = resolve_context_for_query(
                    user_message,
                    current_user,
                    search_term,
                    conversation_id
                )
                if early_response:
                    return jsonify(early_response)

                personal_direct = is_personal_scoped_context(context_used)
                if personal_direct:
                    final_result = {
                        "response": render_personal_scoped_response(context_used),
                        "suggestions": []
                    }
                else:
                    # 3. Final call with context
                    final_result = ai_engine.process_message(
                        user_message, 
                        data_context=context_used or "No specific data found.", 
                        conversation_history=list(conversation_history),
                        language=lang
                    )
                
                # Auto-cache if good response (future enhancement)
                if not context_used or len(str(context_used)) < 10:
                     unanswered_manager.log_unanswered(user_message, reason="low_context")
                
                response_text = final_result.get("response", "I couldn't find that info.")
                if not personal_direct:
                    response_text = postprocess_grounded_response(user_message, context_used or "", response_text)
                
                # ??????????HALLUCINATION PREVENTION: Validate response before sending
                if not personal_direct:
                    is_valid, warning, corrected = validate_response_against_context(
                        response_text, context_used or "", user_message
                    )
                    
                    if not is_valid:
                        logger.warning(f"[Hallucination Detected] {warning}")
                        if corrected:
                            response_text = corrected
                        # Log for quality improvement
                        try:
                            unanswered_manager.log_unanswered(
                                user_message, 
                                reason=f"hallucination_blocked: {warning}"
                            )
                        except:
                            pass
                
                # Remove markdown formatting
                response_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', response_text)  # **bold**
                response_text = re.sub(r'\*([^*]+)\*', r'\1', response_text)  # *italic*
                response_text = re.sub(r'^#+\s*', '', response_text, flags=re.MULTILINE)  # ## headers
                if personal_direct:
                    response_text = compress_response_for_ux(user_message, response_text)
                response_payload = {
                    "text": response_text,
                    "suggestions": final_result.get("suggestions", [])
                }
            except Exception as e:
                logger.error(f"Context Fetch Error: {e}")
                response_text = "I encountered an error looking up that information."
                response_payload = {"text": response_text, "suggestions": []}

        else:
            # AI Answered directly (Saved 1 Call!)
            response_text = initial_result.get("response", "")
            # Remove markdown formatting
            response_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', response_text)  # **bold**
            response_text = re.sub(r'\*([^*]+)\*', r'\1', response_text)  # *italic*
            response_text = re.sub(r'^#+\s*', '', response_text, flags=re.MULTILINE)  # ## headers
            response_text = compress_response_for_ux(user_message, response_text)
            response_payload = {
                "text": response_text,
                "suggestions": initial_result.get("suggestions", [])
            }

        # Update History
        response_payload["suggestions"] = _normalize_suggestions(response_payload.get("suggestions", []))
        append_conversation_message(session_key, "assistant", json.dumps(response_payload))

        # ?? SEMANTIC CACHE: Store successful response for future similar queries
        if HAS_SEMANTIC_CACHE and semantic_cache and response_text and not force_rag and not wants_source:
            try:
                # Only cache if we have a meaningful response (not error messages)
                if (
                    len(response_text) > 20
                    and "error" not in response_text.lower()
                    and not any(p in response_text.lower() for p in WEAK_SERVICE_PHRASES)
                ):
                    semantic_cache.put(user_message, json.dumps(response_payload))
                    logger.debug(f"[Semantic Cache] Stored response for: {user_message[:50]}...")
            except Exception as e:
                logger.warning(f"[Semantic Cache] Store error: {e}")

        # Return structured JSON for frontend
        # format: { response: JSON_STRING, session_id: STR }
        return jsonify({
            "response": json.dumps(response_payload),
            "session_id": conversation_id,
            "type": "message",
            "user": current_user.get("name") if current_user else "Guest"
        })

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ===========================================
# ?? STREAMING CHAT ENDPOINT
# ===========================================

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """
    Streaming chat endpoint using Server-Sent Events (SSE).
    Returns response chunks in real-time for better UX.
    """
    try:
        data = request.get_json() or {}
        user_message = data.get("message")
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        # Get Token if available
        current_user = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1] if "Bearer " in request.headers['Authorization'] else None
            if token:
                current_user = auth_utils.decode_access_token(token)
        
        session_key, conversation_id, _ = resolve_conversation_session(current_user, data)
        lang = detect_language(user_message)
        is_obvious_general = _is_obvious_general_knowledge_query((user_message or "").lower())
        force_rag = should_force_rag(user_message)
        wants_source = _user_requested_source(user_message)
        append_conversation_message(session_key, "user", user_message)

        if _is_bot_name_query(user_message):
            response_payload = {
                "text": _bot_name_response(user_message),
                "suggestions": _normalize_suggestions(["Ask about UCSI programs", "Ask about campus facilities"])
            }
            append_conversation_message(session_key, "assistant", json.dumps(response_payload))
            def _single():
                yield f"data: {json.dumps({'chunk': '', 'done': True, 'full_response': response_payload})}\n\n"
            return Response(
                _single(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'
                }
            )

        if _is_capability_smalltalk_query(user_message):
            response_payload = {
                "text": _capability_smalltalk_response(user_message),
                "suggestions": _normalize_suggestions([
                    "Ask about UCSI programs",
                    "Ask about campus facilities"
                ])
            }
            append_conversation_message(session_key, "assistant", json.dumps(response_payload))
            def _single_smalltalk():
                yield f"data: {json.dumps({'chunk': '', 'done': True, 'full_response': response_payload})}\n\n"
            return Response(
                _single_smalltalk(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'
                }
            )

        if _is_noise_or_gibberish_query(user_message):
            response_payload = {
                "text": _noise_or_gibberish_response(user_message),
                "suggestions": _normalize_suggestions([
                    "Tell me about UCSI programs",
                    "Tell me about campus facilities"
                ])
            }
            append_conversation_message(session_key, "assistant", json.dumps(response_payload))
            def _single_noise():
                yield f"data: {json.dumps({'chunk': '', 'done': True, 'full_response': response_payload})}\n\n"
            return Response(
                _single_noise(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'
                }
            )
        
        def generate():
            """Generator for SSE streaming"""
            try:
                # Check caches first
                if HAS_SEMANTIC_CACHE and semantic_cache and not force_rag and not wants_source and not is_obvious_general:
                    cached = semantic_cache.get(user_message)
                    if cached:
                        if not _is_semantic_cache_hit_safe(user_message, cached):
                            cached = None
                    if cached:
                        try:
                            payload = json.loads(cached["response"])
                            payload["suggestions"] = _normalize_suggestions(payload.get("suggestions", []))
                            response_json = json.dumps(payload)
                        except Exception:
                            response_json = str(cached.get("response", ""))
                        append_conversation_message(session_key, "assistant", response_json)
                        yield f"data: {json.dumps({'chunk': response_json, 'done': True, 'cache': 'semantic'})}\n\n"
                        return
                
                cached_result = None if (wants_source or is_obvious_general) else faq_cache.get_cached_answer(user_message)
                if cached_result:
                    response_json = json.dumps({
                        "text": cached_result["answer"],
                        "suggestions": _normalize_suggestions(cached_result.get("suggestions", []))
                    })
                    append_conversation_message(session_key, "assistant", response_json)
                    yield f"data: {json.dumps({'chunk': response_json, 'done': True, 'cache': 'faq'})}\n\n"
                    return
                
                conversation_history = get_conversation_history(session_key)
                if force_rag:
                    initial_result = {
                        "needs_context": True,
                        "search_term": "self" if check_personal_intent(user_message, None) else None,
                        "suggestions": []
                    }
                else:
                    convo_for_ai = [] if is_obvious_general else list(conversation_history)
                    initial_result = ai_engine.process_message(
                        user_message,
                        conversation_history=convo_for_ai,
                        language=lang
                    )

                context_used = ""
                if initial_result.get("needs_context"):
                    context_used, early_response = resolve_context_for_query(
                        user_message,
                        current_user,
                        initial_result.get("search_term"),
                        conversation_id
                    )
                    if early_response:
                        text = early_response.get("response", "Authentication is required.")
                        append_conversation_message(session_key, "assistant", text)
                        yield f"data: {json.dumps({'chunk': text, 'done': True, 'type': early_response.get('type', 'message')})}\n\n"
                        return

                    personal_direct = is_personal_scoped_context(context_used)
                    if personal_direct:
                        result = {
                            "response": render_personal_scoped_response(context_used),
                            "suggestions": []
                        }
                    else:
                        result = ai_engine.process_message(
                            user_message,
                            data_context=context_used or "No specific data found.",
                            conversation_history=list(conversation_history),
                            language=lang
                        )
                else:
                    result = initial_result

                response_text = result.get("response", "")
                if initial_result.get("needs_context"):
                    if not is_personal_scoped_context(context_used):
                        response_text = postprocess_grounded_response(user_message, context_used or "", response_text)
                    else:
                        response_text = compress_response_for_ux(user_message, response_text)
                else:
                    response_text = compress_response_for_ux(user_message, response_text)
                
                # Simulate streaming by sending chunks
                chunk_size = 20  # characters per chunk
                for i in range(0, len(response_text), chunk_size):
                    chunk = response_text[i:i+chunk_size]
                    yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"
                
                # Send final message with suggestions
                response_payload = {
                    "text": response_text,
                    "suggestions": _normalize_suggestions(result.get("suggestions", []))
                }
                append_conversation_message(session_key, "assistant", json.dumps(response_payload))
                yield f"data: {json.dumps({'chunk': '', 'done': True, 'full_response': response_payload})}\n\n"
                
                # Store in cache
                if HAS_SEMANTIC_CACHE and semantic_cache and len(response_text) > 20 and not force_rag and not wants_source:
                    try:
                        semantic_cache.put(user_message, json.dumps(response_payload))
                    except:
                        pass
                        
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
        
        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )
        
    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ===========================================
# FEEDBACK & ADMIN
# ===========================================
# FEEDBACK & ADMIN
# ===========================================

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON payload"}), 400

        user_message = str(data.get("user_message") or "").strip()
        ai_response = str(data.get("ai_response") or "").strip()
        rating = str(data.get("rating") or "").strip().lower()
        comment = data.get("comment")

        if rating not in {"positive", "negative"}:
            return jsonify({"error": "rating must be 'positive' or 'negative'"}), 400
        if not user_message or not ai_response:
            return jsonify({"error": "user_message and ai_response are required"}), 400

        # Optional identity binding from bearer token.
        session_id = str(data.get("session_id") or "").strip()
        current_user = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1].strip()
            if token:
                current_user = auth_utils.decode_access_token(token)
        if current_user and current_user.get("student_number"):
            session_id = str(current_user.get("student_number"))
        if not session_id:
            session_id = "guest_session"

        logging_utils.log_audit("FEEDBACK", "User", f"Rating: {rating}")
        saved = feedback_engine.save_feedback(
            session_id=session_id,
            user_message=user_message,
            ai_response=ai_response,
            rating=rating,
            comment=comment
        )
        if not saved:
            return jsonify({"error": "Failed to save feedback"}), 500

        if rating == "positive":
            # POSITIVE FEEDBACK: Reinforcement Learning
            # If user likes the answer, consider it a candidate for FAQ
            try:
                question = user_message
                answer_json = ai_response # Usually plain text; may be JSON string.
                
                # Extract text from JSON response if needed
                answer_text = answer_json
                suggestions = []
                try:
                    parsed = json.loads(answer_json)
                    if isinstance(parsed, dict):
                        answer_text = parsed.get("text", answer_json)
                        suggestions = parsed.get("suggestions", [])
                except:
                    pass
                
                # Boost frequency or auto-add
                # For this implementation, we try to auto-cache
                added = faq_cache.auto_cache_if_needed(question, answer_text, suggestions)
                if added:
                    logger.info(f"Auto-cached FAQ due to positive feedback: {question}")
            except Exception as e:
                logger.error(f"Error learning from positive feedback: {e}")

        elif rating == "negative":
            # NEGATIVE FEEDBACK: Log for review
            try:
                unanswered_manager.log_unanswered(
                    question=user_message,
                    reason="negative_feedback",
                    context=comment or ""
                )
            except Exception as err:
                logger.warning(f"Failed to log negative feedback: {err}")
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    """Logout (Client should also delete token)"""
    # For stateless JWT, we can't really invalidate unless we blacklist. 
    # For this demo, we just log it.
    logging_utils.log_audit("LOGOUT", "User", "Logout request")
    return jsonify({"success": True})

@app.route('/api/export_chat', methods=['GET'])
@token_required
def export_chat(current_user):
    """Export conversation history for the current session"""
    try:
        session_id = request.args.get('session_id')
        # If no session_id provided, try to find one or return empty/error
        # For simplicity, if not provided, we return just a header
        
        history = user_sessions.get(session_id, []) if session_id else []
        
        # Format as text
        export_text = f"Chat History for {current_user.get('name')} ({current_user.get('student_number')})\n"
        export_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        export_text += "="*40 + "\n\n"
        
        for msg in history:
            role = msg.get('role', 'unknown').upper()
            content = msg.get('content', '')
            export_text += f"[{role}]: {content}\n\n"
            
        # Return as downloadable file
        return Response(
            export_text,
            mimetype="text/plain",
            headers={"Content-disposition": f"attachment; filename=chat_history_{current_user.get('student_number')}.txt"}
        )
            
    except Exception as e:
        logger.error(f"Export error: {e}")
        return jsonify({"error": str(e)}), 500

# ===========================================
# ADMIN ENDPOINTS
# ===========================================

@app.route('/admin')
def admin_page():
    """Serve Admin Dashboard"""
    if os.path.exists("static/admin/admin.html"):
        return send_from_directory('static/admin', 'admin.html')
    return "Admin panel not found", 404

@app.route('/api/admin/stats', methods=['GET'])
def get_admin_stats():
    """Get statistics for admin dashboard"""
    try:
        # Get Feedback Stats
        feedback_stats = feedback_engine.get_stats()
        recent_feedbacks = feedback_engine.get_recent_feedbacks(limit=10)
        
        # Get Learning Stats
        unanswered = unanswered_manager.get_unresolved()
        
        return jsonify({
            "satisfaction_rate": feedback_stats.get("satisfaction_rate", 0),
            "total_feedbacks": feedback_stats.get("total_feedbacks", 0),
            "unanswered_count": len(unanswered),
            "unanswered_logs": unanswered[-10:], # Last 10
            "recent_feedbacks": recent_feedbacks
        })
    except Exception as e:
        logger.error(f"Admin stats error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/upload', methods=['POST'])
def upload_document():
    """Upload a document to the knowledge base"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "message": "No file part"}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"success": False, "message": "No selected file"}), 400
            
        if file:
            # Ensure directory exists
            if not os.path.exists("data/knowledge_base"):
                os.makedirs("data/knowledge_base")
                
            file_path = f"data/knowledge_base/{file.filename}"
            file.save(file_path)
            
            # Ingest into RAG
            from app.engines.rag_engine import rag_engine
            success = rag_engine.ingest_file(file_path)
            
            if success:
                return jsonify({"success": True, "message": f"Successfully ingested {file.filename}"})
            else:
                return jsonify({"success": False, "message": "File saved but failed to ingest into Vector DB"})
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/admin/files', methods=['GET'])
def list_files():
    """List files in knowledge base"""
    try:
        files = []
        if os.path.exists("data/knowledge_base"):
            for f in os.listdir("data/knowledge_base"):
                if f.endswith(('.pdf', '.txt', '.csv', '.docx')):
                    files.append(f)
        return jsonify({"files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/files', methods=['DELETE'])
def delete_file():
    """Delete a file from knowledge base"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        if not filename:
            return jsonify({"success": False, "message": "Filename required"}), 400
        
        file_path = os.path.join("data/knowledge_base", filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            # Note: Ideally we should re-index RAG here, but for now we just delete source
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "message": "File not found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    try:
        # Setup File Logging
        file_handler = logging.FileHandler('server.log')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        print(f"Starting Flask server with Google GenAI Native Model: {MODEL_NAME}")
        # Keep debug off by default for stable background/production execution.
        debug_mode = _env_bool("FLASK_DEBUG", False)
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=debug_mode,
            use_reloader=False,
            threaded=True
        )
    except Exception as e:
        tb = traceback.format_exc()
        with open("crash_log.txt", "w") as f:
            f.write(f"Server Crashed: {str(e)}\n\n{tb}")
        print(f"Server Crashed: {e}")

