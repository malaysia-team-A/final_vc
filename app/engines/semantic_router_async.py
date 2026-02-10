import asyncio
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from app.engines.db_engine_async import db_engine_async

DEFAULT_MODEL_NAME = os.getenv("RAG_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
DEFAULT_COLLECTION = os.getenv("SEMANTIC_ROUTER_COLLECTION", "semantic_intents")
DEFAULT_INDEX = os.getenv("SEMANTIC_ROUTER_INDEX", "semantic_intent_vector_idx")
MIN_CONFIDENCE = float(os.getenv("SEMANTIC_ROUTER_MIN_CONFIDENCE", "0.53"))
ENABLED = str(os.getenv("SEMANTIC_ROUTER_ENABLED", "true")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Seed data is intentionally data-driven so intent behavior can be tuned in DB
# without continuously patching application routing rules.
DEFAULT_INTENT_EXAMPLES: Dict[str, List[str]] = {
    "personal_profile": [
        "show my profile",
        "show my student information",
        "what is my nationality",
        "내 정보 보여줘",
        "내 프로필 알려줘",
        "내 전공 알려줘",
    ],
    "personal_grade": [
        "show my gpa",
        "show my grades",
        "what is my cgpa",
        "내 gpa 알려줘",
        "내 성적 보여줘",
        "내 학점 알려줘",
    ],
    "ucsi_programme": [
        "tell me about ucsi programme",
        "tuition fee for this programme",
        "entry requirement for ucsi",
        "UCSI 전공 정보 알려줘",
        "등록금 알려줘",
        "입학 조건 알려줘",
    ],
    "ucsi_hostel": [
        "ucsi hostel fee",
        "hostel deposit",
        "hostel policy",
        "기숙사 비용 알려줘",
        "기숙사 보증금 얼마야",
        "기숙사 정책 알려줘",
    ],
    "ucsi_staff": [
        "who is the dean of this faculty",
        "staff contact in ucsi",
        "professor in computer science ucsi",
        "UCSI 교수 정보 알려줘",
        "학장 누구야",
        "직원 연락처 알려줘",
    ],
    "ucsi_facility": [
        "where is the library on campus",
        "campus facilities",
        "is there a gym in ucsi",
        "도서관 위치 알려줘",
        "캠퍼스 시설 알려줘",
        "프린터 어디 있어",
    ],
    "ucsi_schedule": [
        "semester schedule ucsi",
        "intake dates",
        "exam period",
        "학사 일정 알려줘",
        "입학 시기 알려줘",
        "시험 기간 언제야",
    ],
    "general_person": [
        "who is taylor swift",
        "tell me about albert einstein",
        "what do you know about cristiano ronaldo",
        "이순신에 대해서 알려줘",
        "vicky yiran에 대해 알려줘",
        "김연아 누구야",
    ],
    "general_world": [
        "what is machine learning",
        "capital of malaysia",
        "explain python",
        "양자역학이 뭐야",
        "말레이시아 수도는 어디야",
        "파이썬이 뭐야",
    ],
    "capability_smalltalk": [
        "can you do a handstand",
        "can you dance",
        "can you sing",
        "물구나무 설 수 있어",
        "춤출 수 있어",
        "노래할 수 있어",
    ],
}

PERSONAL_INTENTS = {"personal_profile", "personal_grade"}
UCSI_CONTEXT_INTENTS = {
    "ucsi_programme",
    "ucsi_hostel",
    "ucsi_staff",
    "ucsi_facility",
    "ucsi_schedule",
}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class AsyncSemanticRouter:
    def __init__(self):
        self.model = None
        self.dimension = 0
        self.collection = None
        self._ready = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> bool:
        if not ENABLED:
            return False
        if self._ready:
            return True
        async with self._init_lock:
            if self._ready:
                return True
            if db_engine_async.db is None:
                return False
            if SentenceTransformer is None:
                return False
            try:
                self.model = SentenceTransformer(DEFAULT_MODEL_NAME)
                self.dimension = int(self.model.get_sentence_embedding_dimension() or 384)
                self.collection = db_engine_async.db[DEFAULT_COLLECTION]
                await self._ensure_seed_data()
                await self._ensure_vector_index()
                self._ready = True
                return True
            except Exception:
                self._ready = False
                return False

    def _embed(self, text: str) -> Optional[List[float]]:
        if not self.model:
            return None
        try:
            vec = self.model.encode([str(text or "").strip()])[0]
            return np.asarray(vec, dtype=np.float32).tolist()
        except Exception:
            return None

    def _extract_turn_text(self, raw_content: Any) -> str:
        text = str(raw_content or "").strip()
        if not text:
            return ""
        if text.startswith("{") and text.endswith("}"):
            try:
                payload = json.loads(text)
                if isinstance(payload, dict):
                    candidate = payload.get("text")
                    if isinstance(candidate, str) and candidate.strip():
                        text = candidate.strip()
            except Exception:
                pass
        text = re.sub(r"\s+", " ", text).strip()
        return text[:280]

    def _build_history_hint(
        self,
        conversation_history: Optional[List[Dict[str, Any]]],
        user_message: str,
    ) -> str:
        if not isinstance(conversation_history, list) or not conversation_history:
            return ""

        history_items = [item for item in conversation_history if isinstance(item, dict)]
        if not history_items:
            return ""

        # Current turn is already appended by chat.py; remove duplicate tail.
        if str(history_items[-1].get("role") or "").strip().lower() == "user":
            tail = self._extract_turn_text(history_items[-1].get("content"))
            if tail and tail == str(user_message or "").strip():
                history_items = history_items[:-1]

        if not history_items:
            return ""

        picked: List[str] = []
        user_count = 0
        model_count = 0

        for item in reversed(history_items):
            role = str(item.get("role") or "").strip().lower()
            content = self._extract_turn_text(item.get("content"))
            if not content:
                continue

            if role == "user" and user_count < 2:
                picked.append(content)
                user_count += 1
            elif role != "user" and model_count < 1:
                picked.append(content)
                model_count += 1

            if user_count >= 2 and model_count >= 1:
                break

        if not picked:
            return ""

        picked.reverse()
        hint = re.sub(r"\s+", " ", " ".join(picked)).strip()
        return hint[:240]

    def _should_blend_history(self, user_message: str) -> bool:
        text = re.sub(r"\s+", " ", str(user_message or "").strip())
        if not text:
            return False
        word_count = len(text.split())
        char_count = len(text)
        return word_count <= 7 or char_count <= 40

    async def _ensure_seed_data(self) -> None:
        if self.collection is None:
            return
        try:
            existing = await self.collection.count_documents({"source": "default_seed_v1"})
            if existing > 0:
                return
            docs = []
            for intent, examples in DEFAULT_INTENT_EXAMPLES.items():
                for text in examples:
                    vec = self._embed(text)
                    if not vec:
                        continue
                    docs.append(
                        {
                            "intent": intent,
                            "example": text,
                            "embedding": vec,
                            "source": "default_seed_v1",
                            "created_at": datetime.now().isoformat(),
                        }
                    )
            if docs:
                await self.collection.insert_many(docs, ordered=False)
        except Exception:
            return

    async def _ensure_vector_index(self) -> None:
        if self.collection is None:
            return
        try:
            await db_engine_async.db.command(
                {
                    "createSearchIndexes": DEFAULT_COLLECTION,
                    "indexes": [
                        {
                            "name": DEFAULT_INDEX,
                            "definition": {
                                "fields": [
                                    {
                                        "type": "vector",
                                        "path": "embedding",
                                        "numDimensions": int(self.dimension),
                                        "similarity": "cosine",
                                    }
                                ]
                            },
                        }
                    ],
                }
            )
        except Exception:
            # Atlas permissions/tier may not allow automatic index creation.
            # Query path has a local cosine fallback below.
            return

    def _extract_entity(self, user_message: str) -> Optional[str]:
        text = str(user_message or "").strip()
        if not text:
            return None
        patterns = [
            r"^\s*who(?:'s| is)\s+(?P<name>.+?)\s*\??\s*$",
            r"^\s*tell me (?:more )?about\s+(?P<name>.+?)\s*\??\s*$",
            r"^\s*what do you know about\s+(?P<name>.+?)\s*\??\s*$",
            r"^\s*(?P<name>[A-Za-z0-9가-힣.\-\'\s]+?)\s*(?:에 대해|에대해|에 대해서|에대해서|관련해서|에 관한)\s*(?:정보를\s*)?(?:알려줘|알려줄래|알려주세요|말해줘|설명해줘|소개해줘)?\s*\??\s*$",
            r"^\s*(?P<name>[A-Za-z0-9가-힣.\-\'\s]+?)\s*(?:누구야|누구예요|누군가요)\s*\??\s*$",
        ]
        candidate = None
        for pattern in patterns:
            m = re.match(pattern, text, flags=re.IGNORECASE)
            if m:
                candidate = m.group("name")
                break
        if not candidate:
            return None
        candidate = re.sub(r"\s+", " ", str(candidate).strip())
        candidate = candidate.strip(" \t\r\n?!.,\"'`")
        candidate = re.sub(r"\s*(?:은|는|이|가|을|를|와|과|의)$", "", candidate)
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if len(candidate) < 2:
            return None
        if len(candidate.split()) > 10:
            return None
        lowered = candidate.lower()
        reject_terms = {
            "ucsi",
            "university",
            "hostel",
            "tuition",
            "fee",
            "fees",
            "programme",
            "program",
            "course",
            "faculty",
            "campus",
            "building",
            "schedule",
            "gpa",
            "cgpa",
            "기숙사",
            "등록금",
            "전공",
            "학과",
            "학부",
            "프로그램",
            "시설",
            "도서관",
            "일정",
            "입학",
            "장학금",
            "학점",
            "성적",
        }
        if any(term in lowered for term in reject_terms):
            return None
        return candidate

    async def _vector_search_intents(self, query_vector: List[float], limit: int = 8) -> List[Dict[str, Any]]:
        if self.collection is None:
            return []
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": DEFAULT_INDEX,
                        "path": "embedding",
                        "queryVector": query_vector,
                        "numCandidates": max(40, int(limit) * 10),
                        "limit": int(limit),
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "intent": 1,
                        "example": 1,
                        "score": {"$meta": "vectorSearchScore"},
                    }
                },
            ]
            return await self.collection.aggregate(pipeline).to_list(length=int(limit))
        except Exception:
            return []

    async def _local_similarity_fallback(self, query_vector: List[float], limit: int = 8) -> List[Dict[str, Any]]:
        if self.collection is None:
            return []
        try:
            docs = await self.collection.find(
                {},
                {"_id": 0, "intent": 1, "example": 1, "embedding": 1},
            ).to_list(length=500)
            qv = np.asarray(query_vector, dtype=np.float32)
            ranked: List[Dict[str, Any]] = []
            for doc in docs:
                emb = doc.get("embedding")
                if not isinstance(emb, list) or not emb:
                    continue
                sv = _cosine(qv, np.asarray(emb, dtype=np.float32))
                ranked.append(
                    {
                        "intent": str(doc.get("intent") or ""),
                        "example": str(doc.get("example") or ""),
                        "score": float(max(min(sv, 1.0), -1.0)),
                    }
                )
            ranked.sort(key=lambda x: x["score"], reverse=True)
            return ranked[: int(limit)]
        except Exception:
            return []

    def _aggregate_intent_score(self, rows: List[Dict[str, Any]]) -> Dict[str, float]:
        intent_stats: Dict[str, Dict[str, float]] = {}
        for row in rows:
            intent = str(row.get("intent") or "").strip()
            if not intent:
                continue
            score = float(row.get("score") or 0.0)
            score = max(min(score, 1.0), -1.0)
            stats = intent_stats.setdefault(intent, {"max": -1.0, "sum": 0.0, "count": 0.0})
            stats["max"] = max(stats["max"], score)
            stats["sum"] += score
            stats["count"] += 1.0

        out: Dict[str, float] = {}
        for intent, stats in intent_stats.items():
            avg = stats["sum"] / max(1.0, stats["count"])
            mixed = (stats["max"] * 0.75) + (avg * 0.25)
            out[intent] = float(max(min(mixed, 1.0), -1.0))
        return out

    async def classify(
        self,
        user_message: str,
        search_term: Optional[str] = None,
        language: str = "en",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not str(user_message or "").strip():
            return None
        ok = await self.initialize()
        if not ok:
            return None

        query = str(user_message or "").strip()
        if search_term:
            query = f"{query} {str(search_term).strip()}".strip()
        query_vector = self._embed(query)
        if not query_vector:
            return None

        history_hint = self._build_history_hint(conversation_history, user_message=user_message)
        used_history_context = False
        if history_hint and self._should_blend_history(user_message):
            history_vector = self._embed(history_hint)
            if history_vector and len(history_vector) == len(query_vector):
                qv = np.asarray(query_vector, dtype=np.float32)
                hv = np.asarray(history_vector, dtype=np.float32)
                blended = (qv * np.float32(0.80)) + (hv * np.float32(0.20))
                norm = np.linalg.norm(blended)
                if float(norm) > 0.0:
                    blended = blended / norm
                query_vector = blended.astype(np.float32).tolist()
                used_history_context = True

        rows = await self._vector_search_intents(query_vector, limit=8)
        if not rows:
            rows = await self._local_similarity_fallback(query_vector, limit=8)
        if not rows:
            return None

        scores = self._aggregate_intent_score(rows)
        if not scores:
            return None
        best_intent, best_score = max(scores.items(), key=lambda kv: kv[1])

        result: Dict[str, Any] = {
            "intent": best_intent,
            "confidence": float(best_score),
            "language": language,
            "top_matches": rows[:3],
            "is_personal": best_intent in PERSONAL_INTENTS,
            "needs_context": best_intent in UCSI_CONTEXT_INTENTS,
            "entity": None,
            "search_term": None,
            "used_history_context": used_history_context,
        }

        if best_intent == "general_person":
            result["entity"] = self._extract_entity(user_message)
        elif best_intent in UCSI_CONTEXT_INTENTS:
            token = str(search_term or "").strip() or str(user_message or "").strip()
            if not search_term and used_history_context and history_hint:
                token = f"{history_hint} {token}".strip()
            token = re.sub(r"\s+", " ", token).strip()
            result["search_term"] = token[:80] if token else None

        if result["confidence"] < MIN_CONFIDENCE:
            result["intent"] = "unknown"
            result["is_personal"] = False
            result["needs_context"] = False
            result["search_term"] = None
            result["entity"] = None
            result["used_history_context"] = False

        return result


semantic_router_async = AsyncSemanticRouter()
