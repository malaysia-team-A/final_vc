import os
import re
import motor.motor_asyncio
from typing import Optional, Dict, List, Any
from datetime import datetime
from app.config import Config

_QUERY_TOKEN_RE = re.compile(r"[a-z0-9가-힣]{2,}")
_QUERY_STOPWORDS = {
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "the",
    "a",
    "an",
    "to",
    "for",
    "of",
    "in",
    "on",
    "about",
    "tell",
    "show",
    "please",
    "can",
    "you",
    "do",
    "did",
    "me",
    "my",
    "정보",
    "알려줘",
    "말해줘",
    "대해",
    "대해서",
    "관련",
    "좀",
    "해줘",
    "해줘요",
    "부탁",
}


def _normalize_query_text(query: str) -> str:
    text = str(query or "").strip().lower()
    if not text:
        return ""
    text = text.replace("_", " ")
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"[^\w가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_query_tokens(query: str) -> List[str]:
    norm = _normalize_query_text(query)
    if not norm:
        return []
    out: List[str] = []
    seen = set()
    for tok in _QUERY_TOKEN_RE.findall(norm):
        if tok in _QUERY_STOPWORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out[:24]


def _token_overlap_score(query_tokens: List[str], candidate_tokens: List[str]) -> float:
    q = [str(t).strip().lower() for t in (query_tokens or []) if str(t).strip()]
    c = [str(t).strip().lower() for t in (candidate_tokens or []) if str(t).strip()]
    if not q or not c:
        return 0.0

    q_set = set(q)
    c_set = set(c)
    intersection = len(q_set.intersection(c_set))
    if intersection == 0:
        return 0.0

    precision = intersection / max(1, len(c_set))
    recall = intersection / max(1, len(q_set))
    jaccard = intersection / max(1, len(q_set.union(c_set)))

    score = (precision * 0.35) + (recall * 0.45) + (jaccard * 0.20)
    if intersection >= 3:
        score += 0.08
    return min(score, 1.0)


class AsyncDatabaseEngine:
    def __init__(self):
        self.client = None
        self.db = None
        self.student_collection_name = "UCSI"  # Default
        
    async def connect(self):
        """Establish connection to MongoDB Atlas"""
        try:
            uri = Config.MONGO_URI
            if not uri:
                print("Error: MONGO_URI not found in .env")
                return

            self.client = motor.motor_asyncio.AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
            
            # Verify connection
            await self.client.admin.command('ping')
            
            # Get Database
            db_name = uri.split('/')[-1].split('?')[0] or "UCSI_DB"
            self.db = self.client[db_name]
            print(f"[AsyncDB] Successfully connected to MongoDB: {db_name}")

            # Smart detection of Student Collection
            try:
                colls = await self.db.list_collection_names()
                candidates = ["UCSI", "students", "Students", "UCSI_STUDENTS"]
                found = False
                for c in candidates:
                    if c in colls:
                        self.student_collection_name = c
                        found = True
                        break
                
                if not found:
                    for c in colls:
                        if "student" in c.lower() or "ucsi" in c.lower():
                            self.student_collection_name = c
                            found = True
                            print(f"DEBUG: Auto-detected student collection: {c}")
                            break
                            
                if found or colls:
                    target = self.student_collection_name if found else colls[0]
                    # Check if empty? No, just assume it's valid for now
                    print(f"DEBUG: Using collection '{target}' for student data.")
                    self.student_collection_name = target
            except Exception as e:
                print(f"Warning: Could not auto-detect collections: {e}")
            
        except Exception as e:
            print(f"[AsyncDB][ERROR] Database connection failed: {e}")
    
    @property
    def student_coll(self):
        if self.db is not None:
            return self.db[self.student_collection_name]
        return None

    def query_metadata(self, query: str) -> Dict[str, Any]:
        query_norm = _normalize_query_text(query)
        return {
            "query_norm": query_norm,
            "query_tokens": _extract_query_tokens(query_norm),
        }

    async def get_student_by_number(self, student_number: str) -> Optional[Dict]:
        if self.student_coll is None: return None
        try:
            # Try string
            student = await self.student_coll.find_one({"STUDENT_NUMBER": str(student_number)})
            if student: return student
            
            # Try int
            if str(student_number).isdigit():
                student = await self.student_coll.find_one({"STUDENT_NUMBER": int(student_number)})
                if student: return student
                    
            return None
        except Exception as e:
            print(f"DB Error: {e}")
            return None

    async def get_student_by_name(self, name: str) -> Optional[Dict]:
        if self.student_coll is None: return None
        try:
            escaped_name = re.escape(str(name or "").strip())
            if not escaped_name:
                return None
            return await self.student_coll.find_one({
                "STUDENT_NAME": {"$regex": f"^{escaped_name}$", "$options": "i"}
            })
        except Exception as e:
            print(f"DB Error: {e}")
            return None

    async def search_programme_by_keywords(self, keywords: List[str], limit: int = 5) -> List[Dict]:
        if self.student_coll is None: return []
        
        sanitized = []
        for k in (keywords or []):
            kw_clean = str(k).strip()
            if len(kw_clean) >= 3:
                sanitized.append(kw_clean)
        if not sanitized: return []

        or_conditions = []
        fields = ["PROGRAMME_NAME", "PROGRAMME", "PROGRAMME_TITLE"]
        for kw in sanitized:
            regex = {"$regex": re.escape(kw), "$options": "i"}
            for field in fields:
                or_conditions.append({field: regex})
        
        if not or_conditions: return []

        try:
            cursor = self.student_coll.find({"$or": or_conditions}, {"_id": 0}).limit(limit * 3)
            return await cursor.to_list(length=limit * 3)
        except Exception as e:
            print(f"DB Error: {e}")
            return []

    async def find_staff_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Find a public staff profile candidate from UCSI_STAFF by name.
        Returns minimal safe profile fields or None.
        """
        if self.db is None:
            return None

        candidate = str(name or "").strip()
        if len(candidate) < 2:
            return None

        coll = self.db["UCSI_STAFF"]

        try:
            escaped = re.escape(candidate)
            # Try to match name exactly first, then as a substring
            patterns = [rf"^{escaped}$", escaped]

            for pattern in patterns:
                name_filter = {"$regex": pattern, "$options": "i"}
                # Search across all documents and unwind to find ALL matching members
                pipeline = [
                    {"$unwind": "$staff_members"},
                    {"$match": {"staff_members.name": name_filter}},
                    {
                        "$project": {
                            "_id": 0,
                            "major": 1,
                            "name": "$staff_members.name",
                            "role": "$staff_members.role",
                            "email": "$staff_members.email",
                            "profile_url": "$staff_members.profile_url"
                        }
                    },
                    {"$limit": 1} # For now, keep returning best single match to satisfy existing API
                ]
                
                cursor = coll.aggregate(pipeline)
                docs = await cursor.to_list(length=1)
                
                if docs:
                    doc = docs[0]
                    return {
                        "name": str(doc.get("name") or "").strip(),
                        "role": str(doc.get("role") or "").strip(),
                        "email": str(doc.get("email") or "").strip(),
                        "major": str(doc.get("major") or "").strip(),
                        "profile_url": str(doc.get("profile_url") or "").strip()
                    }
        except Exception as e:
            print(f"Staff lookup error: {e}")

        return None

    async def save_feedback(self, feedback_data: Dict) -> bool:
        if self.db is None: return False
        try:
            await self.db.Feedback.insert_one(feedback_data)
            return True
        except Exception as e:
            print(f"Feedback save error: {e}")
            return False

    async def log_unanswered(self, question_data: Dict) -> bool:
        if self.db is None:
            return False
        try:
            await self.db.unanswered.insert_one(question_data)
            return True
        except Exception as e:
            print(f"Unanswered log error: {e}")
            return False

    async def save_learned_response(self, query: str, answer: str) -> bool:
        if self.db is None: return False
        try:
            meta = self.query_metadata(query)
            query_norm = meta["query_norm"] or str(query or "").strip().lower()
            await self.db.LearnedQA.update_one(
                {"query_norm": query_norm},
                {
                    "$set": {
                        "query": query_norm,
                        "query_norm": query_norm,
                        "query_tokens": meta["query_tokens"],
                        "answer": answer,
                        "timestamp": datetime.now(),
                        "source": "user_feedback",
                    }
                },
                upsert=True
            )
            return True
        except Exception as e:
            print(f"Learned response save error: {e}")
            return False

    async def search_learned_response(self, query: str) -> Optional[str]:
        if self.db is None:
            return None
        meta = self.query_metadata(query)
        q_norm = meta["query_norm"]
        q_tokens = meta["query_tokens"]
        if not q_norm:
            return None
        try:
            doc = await self.db.LearnedQA.find_one(
                {"$or": [{"query_norm": q_norm}, {"query": q_norm}]},
                {"_id": 0, "answer": 1},
            )
            if not doc:
                # Fuzzy pass: score recent learned answers by token overlap.
                cursor = self.db.LearnedQA.find(
                    {},
                    {"_id": 0, "answer": 1, "query_norm": 1, "query_tokens": 1},
                ).sort("timestamp", -1).limit(120)
                candidates = await cursor.to_list(length=120)
                best_answer = None
                best_score = 0.0
                for item in candidates:
                    answer = str(item.get("answer") or "").strip()
                    if not answer:
                        continue
                    candidate_norm = str(item.get("query_norm") or "").strip().lower()
                    candidate_tokens = item.get("query_tokens")
                    if not isinstance(candidate_tokens, list):
                        candidate_tokens = _extract_query_tokens(candidate_norm)
                    score = _token_overlap_score(q_tokens, candidate_tokens)
                    if candidate_norm and (q_norm in candidate_norm or candidate_norm in q_norm):
                        score = max(score, 0.9 if min(len(q_norm), len(candidate_norm)) >= 8 else 0.78)
                    if score > best_score:
                        best_score = score
                        best_answer = answer

                threshold = 0.86 if len(q_tokens) <= 2 else 0.62
                if best_answer and best_score >= threshold:
                    return best_answer
                return None
            answer = str(doc.get("answer") or "").strip()
            return answer or None
        except Exception as e:
            print(f"Learned response search error: {e}")
            return None

    async def save_bad_response(self, query: str, bad_answer: str, reason: str = "") -> bool:
        if self.db is None: return False
        try:
            meta = self.query_metadata(query)
            query_norm = meta["query_norm"] or str(query or "").strip().lower()
            await self.db.BadQA.update_one(
                {"query_norm": query_norm},
                {
                    "$set": {
                        "query": query_norm,
                        "query_norm": query_norm,
                        "query_tokens": meta["query_tokens"],
                        "bad_answer": bad_answer,
                        "reason": reason,
                        "timestamp": datetime.now(),
                    },
                    "$inc": {"count": 1},
                },
                upsert=True
            )
            return True
        except Exception as e:
            print(f"Bad response save error: {e}")
            return False

    async def has_bad_response(self, query: str) -> bool:
        if self.db is None:
            return False
        meta = self.query_metadata(query)
        q_norm = meta["query_norm"]
        q_tokens = meta["query_tokens"]
        if not q_norm:
            return False
        try:
            bad_exact = await self.db.BadQA.find_one(
                {"$or": [{"query_norm": q_norm}, {"query": q_norm}]},
                {"_id": 1},
            )
            if bad_exact:
                return True
            feedback_hit = await self.db.Feedback.find_one(
                {
                    "rating": "negative",
                    "$or": [
                        {"query_norm": q_norm},
                        {"user_message": {"$regex": f"^{re.escape(str(query or '').strip())}$", "$options": "i"}},
                    ],
                },
                {"_id": 1},
            )
            if feedback_hit:
                return True

            if not q_tokens:
                return False

            bad_rows = await self.db.BadQA.find(
                {},
                {"_id": 0, "query_norm": 1, "query_tokens": 1},
            ).sort("timestamp", -1).limit(120).to_list(length=120)
            for row in bad_rows:
                candidate_norm = str(row.get("query_norm") or "").strip().lower()
                candidate_tokens = row.get("query_tokens")
                if not isinstance(candidate_tokens, list):
                    candidate_tokens = _extract_query_tokens(candidate_norm)
                score = _token_overlap_score(q_tokens, candidate_tokens)
                if candidate_norm and (q_norm in candidate_norm or candidate_norm in q_norm):
                    score = max(score, 0.9 if min(len(q_norm), len(candidate_norm)) >= 8 else 0.76)
                if score >= 0.66:
                    return True

            fb_rows = await self.db.Feedback.find(
                {"rating": "negative"},
                {"_id": 0, "query_norm": 1, "query_tokens": 1, "user_message": 1},
            ).sort("timestamp", -1).limit(160).to_list(length=160)
            for row in fb_rows:
                candidate_norm = str(row.get("query_norm") or row.get("user_message") or "").strip().lower()
                candidate_tokens = row.get("query_tokens")
                if not isinstance(candidate_tokens, list):
                    candidate_tokens = _extract_query_tokens(candidate_norm)
                score = _token_overlap_score(q_tokens, candidate_tokens)
                if candidate_norm and (q_norm in candidate_norm or candidate_norm in q_norm):
                    score = max(score, 0.88 if min(len(q_norm), len(candidate_norm)) >= 8 else 0.74)
                if score >= 0.7:
                    return True

            return False
        except Exception as e:
            print(f"Bad response check error: {e}")
            return False

    async def get_student_stats(self) -> Dict:
        """Aggregate high-level student statistics."""
        if self.student_coll is None:
            return {"total_students": 0, "gender": {}, "top_nationalities": {}}

        try:
            total_students = await self.student_coll.count_documents({})

            gender_pipeline = [
                {"$group": {"_id": "$GENDER", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            nationality_pipeline = [
                {"$group": {"_id": "$NATIONALITY", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 10}
            ]

            gender_rows = await self.student_coll.aggregate(gender_pipeline).to_list(length=None)
            nationality_rows = await self.student_coll.aggregate(nationality_pipeline).to_list(length=10)

            gender = {str(r.get("_id", "Unknown")): r.get("count", 0) for r in gender_rows}
            top_nationalities = {str(r.get("_id", "Unknown")): r.get("count", 0) for r in nationality_rows}

            return {
                "total_students": total_students,
                "gender": gender,
                "top_nationalities": top_nationalities
            }
        except Exception as e:
            print(f"Student stats error: {e}")
            return {"total_students": 0, "gender": {}, "top_nationalities": {}}

# Singleton
db_engine_async = AsyncDatabaseEngine()
