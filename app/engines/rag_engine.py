"""
RAG Engine - Lightweight FAISS Version
Uses FAISS for vector search and Sentence Transformers for embeddings.
"""
import os
# Suppress HuggingFace Warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

import pickle
import re
from typing import List, Dict, Optional, Set
import numpy as np

# Conditional imports
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    import PyPDF2
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    print("Warning: RAG dependencies missing. Install faiss-cpu, sentence-transformers, PyPDF2")

KNOWLEDGE_BASE_DIR = "data/knowledge_base"
INDEX_FILE = "data/knowledge_base/faiss_index.bin"
METADATA_FILE = "data/knowledge_base/faiss_metadata.pkl"
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL",
    "paraphrase-multilingual-MiniLM-L12-v2"
)

FICTIONAL_OR_INVALID_KEYWORDS = {
    "ufo", "jedi", "spiderman", "mars campus", "moon campus",
    "time machine", "teleport pad", "alien faculty"
}

class RAGEngine:
    def __init__(self):
        self.index = None
        self.metadata = []  # List of dicts: [{"text": "...", "source": "..."}, ...]
        self.model = None
        self.enabled = HAS_DEPENDENCIES
        
        if self.enabled:
            try:
                # Multilingual MiniLM: better cross-lingual retrieval for KO/EN/ZH queries.
                self.model_name = DEFAULT_EMBEDDING_MODEL
                self.model = SentenceTransformer(self.model_name)
                self.dimension = int(self.model.get_sentence_embedding_dimension() or 384)
                self.metric_type = faiss.METRIC_INNER_PRODUCT
                
                # Load or Create Index
                self._load_index()
                
            except Exception as e:
                print(f"RAG Init Error: {e}")
                self.enabled = False

    def _load_index(self):
        """Load index from disk if exists"""
        if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
            try:
                self.index = faiss.read_index(INDEX_FILE)
                if getattr(self.index, "d", None) != self.dimension:
                    print(
                        f"[RAG] Index dimension mismatch: index={getattr(self.index, 'd', None)} "
                        f"model={self.dimension}. Rebuilding index."
                    )
                    self._create_new_index()
                    return
                if getattr(self.index, "metric_type", None) != self.metric_type:
                    print(
                        f"[RAG] Index metric mismatch: index={getattr(self.index, 'metric_type', None)} "
                        f"expected={self.metric_type}. Rebuilding index."
                    )
                    self._create_new_index()
                    return
                with open(METADATA_FILE, 'rb') as f:
                    self.metadata = pickle.load(f)
            except Exception as e:
                print(f"Error loading index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Create a new empty index"""
        # Cosine similarity via normalized vectors + inner-product index.
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []

    def _save_index(self):
        """Save index to disk"""
        if not os.path.exists(KNOWLEDGE_BASE_DIR):
            os.makedirs(KNOWLEDGE_BASE_DIR)
        faiss.write_index(self.index, INDEX_FILE)
        with open(METADATA_FILE, 'wb') as f:
            pickle.dump(self.metadata, f)

    def _infer_preferred_labels(self, query: str) -> Set[str]:
        """Infer domain focus labels from the query."""
        q = (query or "").lower()
        preferred = set()

        if any(k in q for k in ["block", "building", "campus address", "map", "where is block", "address"]):
            preferred.add("CampusBlocks")
        if any(k in q for k in ["블록", "건물", "주소", "지도"]):
            preferred.add("CampusBlocks")
        if any(k in q for k in ["hostel faq", "accommodation guaranteed", "installment", "refund", "hostel policy"]):
            preferred.add("HostelFAQ")
        if any(k in q for k in ["기숙사", "환불", "보증금", "분할 납부"]):
            preferred.add("HostelFAQ")
        if any(k in q for k in ["hostel", "dorm", "accommodation", "room", "rent", "deposit"]):
            preferred.add("Hostel")
        if any(k in q for k in ["기숙사", "숙소", "방", "월세", "보증금"]):
            preferred.add("Hostel")
        if any(k in q for k in ["facility", "library", "gym", "cafeteria", "pool", "laundry", "print", "prayer"]):
            preferred.add("Facility")
        if any(k in q for k in ["도서관", "시설", "체육관", "수영장", "세탁", "프린트", "인쇄", "기도실", "카페테리아"]):
            preferred.add("Facility")
        if any(k in q for k in ["schedule", "calendar", "intake", "deadline", "event", "semester"]):
            preferred.add("Schedule")
        if any(k in q for k in ["일정", "학사", "학기", "입학", "셔틀", "버스", "노선"]):
            preferred.add("Schedule")
        if any(k in q for k in ["programme", "program", "major", "course", "tuition", "fee", "fees", "scholarship", "diploma", "degree", "phd", "master"]):
            preferred.add("Programme")
        if any(k in q for k in ["전공", "학과", "프로그램", "과정", "학비", "등록금", "장학금", "박사", "석사"]):
            preferred.add("Programme")
        if any(k in q for k in ["staff", "lecturer", "professor", "dean", "advisor", "teacher", "dr.", "faculty", "head"]):
            preferred.add("Staff")
        if any(k in q for k in ["교수", "교직원", "직원", "학장", "부총장", "지도교수", "학부장", "학과장"]):
            preferred.add("Staff")

        return preferred

    def _expand_query_variants(self, query: str) -> List[str]:
        """
        Expand user query with cross-lingual/domain aliases so retrieval is not
        blocked by exact keyword mismatch.
        """
        q = (query or "").strip()
        if not q:
            return []
        ql = q.lower()

        alias_map = {
            "블록": ["block", "building"],
            "위치": ["location", "where", "address"],
            "비용": ["fee", "cost", "price", "rent", "deposit"],
            "가격": ["price", "cost", "fee"],
            "도서관": ["library"],
            "기숙사": ["hostel", "accommodation", "dorm"],
            "숙소": ["hostel", "accommodation"],
            "학비": ["tuition", "fee", "fees"],
            "등록금": ["tuition", "fee", "fees"],
            "장학금": ["scholarship"],
            "교수": ["professor", "lecturer", "staff"],
            "학장": ["dean", "staff"],
            "부총장": ["vice chancellor", "staff"],
            "전공": ["major", "programme", "program"],
            "학과": ["major", "programme", "program"],
            "입학": ["intake", "admission"],
            "일정": ["schedule", "calendar"],
            "셔틀": ["shuttle", "bus"],
            "버스": ["bus", "shuttle"],
            "인쇄": ["print", "printer"],
            "프린트": ["print", "printer"],
            "식당": ["cafeteria", "dining"],
            "카페테리아": ["cafeteria", "dining"],
            "체육관": ["gym", "fitness"],
            "수영장": ["pool"],
            "세탁": ["laundry"],
            "기도실": ["prayer room", "prayer"],
            "hostel": ["accommodation", "dorm"],
            "accommodation": ["hostel", "dorm"],
            "tuition": ["fee", "fees"],
            "fee": ["tuition", "cost", "price"],
            "programme": ["program", "major", "course"],
            "program": ["programme", "major", "course"],
        }

        alias_terms = []
        for trigger, aliases in alias_map.items():
            if trigger in ql:
                alias_terms.extend(aliases)

        variants = [q]
        if alias_terms:
            dedup_aliases = []
            seen_alias = set()
            for term in alias_terms:
                tl = str(term).strip().lower()
                if not tl or tl in seen_alias or tl in ql:
                    continue
                seen_alias.add(tl)
                dedup_aliases.append(term)

            if dedup_aliases:
                variants.append(f"{q} {' '.join(dedup_aliases[:4])}")
                variants.extend(dedup_aliases[:6])

        # Preserve order and cap expansion budget.
        out = []
        seen = set()
        for v in variants:
            key = str(v).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(str(v).strip())
            if len(out) >= 7:
                break
        return out

    def _is_ascii_token(self, token: str) -> bool:
        return re.fullmatch(r"[A-Za-z0-9_\-\s]+", str(token or "")) is not None

    def _source_to_label(self, source: str, text: str = "") -> Optional[str]:
        """Map source metadata to a normalized domain label."""
        source = source or ""
        source_lower = source.lower()
        text = text or ""

        collection_to_label = {
            "hostel": "Hostel",
            "ucsi_facility": "Facility",
            "usci_schedual": "Schedule",
            "ucsi_ major": "Programme",
            "ucsi_staff": "Staff",
            "ucsi_hostel_faq": "HostelFAQ",
            "ucsi_university_blocks_data": "CampusBlocks",
        }

        for coll, label in collection_to_label.items():
            if coll in source_lower:
                return label

        # Try to infer from bracketed prefixes in indexed text: [Hostel], [Facility], etc.
        m = re.search(r"\[([A-Za-z]+)\]", text)
        if m:
            token = m.group(1).lower()
            token_map = {
                "hostel": "Hostel",
                "facility": "Facility",
                "schedule": "Schedule",
                "programme": "Programme",
                "staff": "Staff",
                "hostelfaq": "HostelFAQ",
                "campusblocks": "CampusBlocks",
            }
            if token in token_map:
                return token_map[token]

        return None

    def _apply_domain_boost(self, score: float, label: Optional[str], preferred_labels: Set[str]) -> float:
        """
        Bias ranking toward the intended domain when query intent is clear.
        """
        if not preferred_labels or not label:
            # If intent is clear but label is unknown, apply conservative penalty.
            if preferred_labels and not label:
                return score * 0.55
            return score
        if label == "LearnedQA":
            return score

        related = {
            "HostelFAQ": {"Hostel"},
            "Hostel": {"HostelFAQ"},
            "CampusBlocks": {"Facility"},
            "Facility": {"CampusBlocks"},
            "Programme": {"Staff"},
            "Staff": {"Programme"},
            "Schedule": {"Programme"},
        }

        if label in preferred_labels:
            boosted = score * 1.45
        elif any(label in related.get(p, set()) for p in preferred_labels):
            boosted = score * 1.0
        else:
            boosted = score * 0.55
        return min(boosted, 1.2)

    def _expand_domain_scope(self, preferred_labels: Set[str], query: str = "") -> Set[str]:
        """Expand preferred labels with near-neighbor related domains."""
        if not preferred_labels:
            return set()
        q = (query or "").lower()
        related = {
            "HostelFAQ": {"Hostel"},
            "Hostel": {"HostelFAQ"},
            "CampusBlocks": {"Facility"},
            "Facility": {"CampusBlocks"},
            "Programme": {"Staff"},
            "Staff": {"Programme"},
            "Schedule": {"Programme"},
        }
        expanded = set(preferred_labels)
        for label in list(preferred_labels):
            # Avoid broad Facility<->CampusBlocks expansion for specific facility asks.
            if label == "Facility":
                specific_facility_intent = any(
                    k in q for k in [
                        "library", "gym", "cafeteria", "pool", "laundry", "print", "printer", "prayer",
                        "도서관", "체육관", "카페테리아", "수영장", "세탁", "인쇄", "프린트", "기도실"
                    ]
                )
                if specific_facility_intent:
                    continue
            expanded.update(related.get(label, set()))
        return expanded

    def _is_forced_no_data_query(self, query: str) -> bool:
        """
        Detect obviously fictional/out-of-scope entities to prevent false positives.
        """
        q = (query or "").lower()
        if any(token in q for token in FICTIONAL_OR_INVALID_KEYWORDS):
            return True

        # Role + explicit person name query: if no matching staff name exists, prefer NO_DATA.
        role_name = re.search(r"\b(?:professor|lecturer|dr\.?)\s+([a-z][a-z0-9\-']+)\b", q, re.IGNORECASE)
        if role_name:
            candidate = role_name.group(1)
            # Ignore generic words that are not real names.
            if candidate not in {"for", "of", "in", "the", "a", "an"}:
                try:
                    from .db_engine import db_engine
                    if db_engine.connected and db_engine.db is not None:
                        exists = db_engine.db["UCSI_STAFF"].find_one(
                            {"staff_members.name": {"$regex": re.escape(candidate), "$options": "i"}},
                            {"_id": 1}
                        )
                        if not exists:
                            return True
                except Exception:
                    # If staff lookup fails, do not hard-fail the query.
                    pass

        return False

    def _search_facility_direct(self, db, query: str, limit: int = 3) -> list:
        """
        Dedicated facility retriever for high-intent queries such as
        library opening/closing hours.
        Returns: [(text, score, source, label), ...]
        """
        q = (query or "").lower()
        if not any(
            k in q for k in [
                "library", "gym", "cafeteria", "pool", "laundry", "prayer", "print", "printer", "facility",
                "도서관", "체육관", "카페테리아", "식당", "수영장", "세탁", "기도실", "인쇄", "프린트", "시설"
            ]
        ):
            return []
        if "UCSI_FACILITY" not in db.list_collection_names():
            return []

        term_map = {
            "library": "library",
            "gym": "gym",
            "cafeteria": "cafeteria",
            "pool": "pool",
            "laundry": "laundry",
            "prayer": "prayer",
            "print": "print",
            "printer": "print",
            "도서관": "library",
            "체육관": "gym",
            "카페테리아": "cafeteria",
            "식당": "cafeteria",
            "수영장": "pool",
            "세탁": "laundry",
            "기도실": "prayer",
            "인쇄": "print",
            "프린트": "print",
            "시설": "facility",
        }
        matched_terms = [v for k, v in term_map.items() if k in q]
        if not matched_terms:
            matched_terms = ["facility"]

        regex_or = []
        for term in matched_terms:
            reg = {"$regex": re.escape(term), "$options": "i"}
            regex_or.append({"name": reg})
            regex_or.append({"category": reg})
            regex_or.append({"tags": reg})
            regex_or.append({"location": reg})

        cursor = db["UCSI_FACILITY"].find({"$or": regex_or}, {"_id": 0}).limit(limit)
        results = []
        for doc in cursor:
            name = doc.get("name") or "Facility"
            details = []
            if doc.get("category"):
                details.append(f"category: {doc.get('category')}")
            if doc.get("location"):
                details.append(f"location: {doc.get('location')}")
            if doc.get("opening_hours"):
                details.append(f"opening_hours: {doc.get('opening_hours')}")
            if doc.get("price_info"):
                details.append(f"price_info: {doc.get('price_info')}")
            if doc.get("tags"):
                details.append(f"tags: {doc.get('tags')}")
            text = f"[Facility] {name}: {', '.join(details)}"
            score = 0.95 if any(k in q for k in ["library", "도서관"]) else 0.88
            results.append((text, score, "MongoDB:Facility:direct", "Facility"))
        return results

    def _search_staff_role_direct(self, db, query: str, limit: int = 15) -> list:
        """
        Dedicated staff-role retriever for leadership queries
        (e.g., Vice Chancellor, Dean).
        Returns: [(text, score, source, label), ...]
        """
        q = (query or "").lower()
        if "UCSI_STAFF" not in db.list_collection_names():
            return []

        role_patterns = [
            ("vice chancellor", r"vice\s*chancellor"),
            ("chancellor", r"\bchancellor\b"),
            ("dean", r"\bdean\b"),
            ("director", r"\bdirector\b"),
            ("head", r"\bhead\b"),
            ("professor", r"\bprofessor\b"),
            ("lecturer", r"\blecturer\b"),
            ("advisor", r"\badvisor\b"),
            ("vice chancellor", r"부총장"),
            ("dean", r"학장"),
            ("head", r"학부장|학과장|책임자"),
            ("professor", r"교수"),
            ("lecturer", r"강사"),
            ("advisor", r"지도교수|지도"),
        ]
        matched_role = None
        matched_regex = None
        for role_name, role_regex in role_patterns:
            # Match by regex so localized queries (e.g., Korean role names) work.
            if re.search(role_regex, q, re.IGNORECASE):
                matched_role = role_name
                matched_regex = role_regex
                break

        if not matched_role:
            return []

        faculty_hints = {
            "it": r"(computer|digital|it\b|informatics)",
            "computer": r"(computer|digital|informatics)",
            "engineering": r"engineering",
            "business": r"business",
            "medicine": r"medicine|health",
            "nursing": r"nursing",
            "music": r"music",
            "design": r"design|creative",
        }
        faculty_regex = None
        for token, regex in faculty_hints.items():
            if token in q:
                faculty_regex = regex
                break

        pipeline = [
            {"$unwind": "$staff_members"},
            {"$match": {"staff_members.role": {"$regex": matched_regex, "$options": "i"}}},
        ]
        if faculty_regex:
            pipeline.append({"$match": {"major": {"$regex": faculty_regex, "$options": "i"}}})
        pipeline.extend([
            {
                "$project": {
                    "_id": 0,
                    "major": 1,
                    "name": "$staff_members.name",
                    "role": "$staff_members.role",
                    "email": "$staff_members.email",
                    "profile_url": "$staff_members.profile_url",
                }
            },
            {"$limit": limit},
        ])

        docs = list(db["UCSI_STAFF"].aggregate(pipeline))
        results = []
        for doc in docs:
            name = doc.get("name") or "Staff Member"
            role = doc.get("role") or "N/A"
            major = doc.get("major") or "N/A"
            details = [f"role: {role}", f"major: {major}"]
            if doc.get("email"):
                details.append(f"email: {doc.get('email')}")
            if doc.get("profile_url"):
                details.append(f"profile_url: {doc.get('profile_url')}")
            text = f"[Staff] {name}: {', '.join(details)}"
            score = 0.97 if matched_role == "vice chancellor" else 0.90
            results.append((text, score, "MongoDB:Staff:direct", "Staff"))
        return results

    def _is_route_specific_query(self, query: str) -> bool:
        q = (query or "").lower()
        return "route" in q and any(k in q for k in ["bus", "shuttle"])

    def _is_route_relevant_text(self, text: str) -> bool:
        tl = (text or "").lower()
        return any(k in tl for k in ["bus", "shuttle", "route", "transport"])

    def ingest_file(self, file_path: str) -> bool:
        """
        Ingest a file (PDF or TXT) into the vector DB
        """
        if not self.enabled: return False
        
        try:
            filename = os.path.basename(file_path)
            ext = filename.split('.')[-1].lower()
            text = ""
            
            # Extract Text
            if ext == 'pdf':
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
            elif ext == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif ext == 'csv':
                import csv
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    text = "\n".join([",".join(row) for row in rows])
            else:
                return False
            
            if not text.strip(): return False
            
            # Chunking (Simple)
            chunk_size = 500
            overlap = 50
            chunks = []
            
            for i in range(0, len(text), chunk_size - overlap):
                batch = text[i:i+chunk_size]
                if len(batch) > 50: # Ignore very small validation chunks
                    chunks.append(batch)
            
            if not chunks: return False

            # Embed and Add to FAISS
            embeddings = np.array(self.model.encode(chunks)).astype('float32')
            if getattr(self.index, "metric_type", None) == faiss.METRIC_INNER_PRODUCT:
                faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            
            # Update Metadata
            for chunk in chunks:
                self.metadata.append({"text": chunk, "source": filename})
            
            # Save persistently
            self._save_index()
            return True
            
        except Exception as e:
            print(f"Error ingesting file {file_path}: {e}")
            return False

    def index_mongodb_collections(self) -> int:
        """
        Index MongoDB collection data into FAISS for semantic search.
        This enables queries like "dormitory cost" to find "hostel fee".
        Returns: number of MongoDB documents indexed.
        """
        if not self.enabled:
            return 0
        
        try:
            from .db_engine import db_engine
            if not db_engine.connected or db_engine.db is None:
                print("[RAG] MongoDB not connected, skipping collection indexing")
                return 0
            
            indexed_count = 0
            
            # Define collections and their text fields to index
            collections_config = [
                {
                    "name": "Hostel",
                    "fields": ["room_type", "building", "campus", "category", "rent_price", "deposit", "features"],
                    "label": "Hostel"
                },
                {
                    "name": "UCSI_FACILITY",
                    "fields": ["name", "location", "category", "opening_hours", "price_info", "tags"],
                    "label": "Facility"
                },
                {
                    "name": "UCSI_ MAJOR",
                    "fields": [
                        "Programme", "Fields of Study", "Course Duration", "Course Mode",
                        "Course Location", "Intakes", "Local Students Fees",
                        "International Students Fees", "Programme Overview", "Url"
                    ],
                    "label": "Programme"
                },
                {
                    "name": "USCI_SCHEDUAL",
                    "fields": ["event_name", "event_type", "start_date", "end_date", "programme", "campus_scope"],
                    "label": "Schedule"
                },
                {
                    "name": "UCSI_STAFF",
                    "fields": ["major", "staff_members"],
                    "label": "Staff"
                },
                {
                    "name": "UCSI_HOSTEL_FAQ",
                    "fields": ["question", "answer", "category", "tags"],
                    "label": "HostelFAQ"
                },
                {
                    "name": "UCSI_University_Blocks_Data",
                    "fields": [
                        "Name", "Address", "BLOCK A_BUILDING", "BLOCK B_BUIILDING", "BLOCK C_BUILDING",
                        "BLOCK D_BUILDING", "BLOCK E_BUILDING", "BLOCKG_BUILDING", "KUCHING_BUILDING",
                        "SPRINGHILL_BUILDING", "Latitude", "Longitude"
                    ],
                    "label": "CampusBlocks"
                }
            ]
            
            mongo_texts = []
            mongo_metadata_entries = []
            
            for config in collections_config:
                try:
                    coll = db_engine.db[config["name"]]
                    # Pull full collection so campus answers are not missed by arbitrary limits.
                    docs = list(coll.find({}, {"_id": 0}))
                    
                    for doc in docs:
                        # SPECIAL HANDLING: Staff members should be indexed individually for precision
                        if config["name"] == "UCSI_STAFF" and "staff_members" in doc:
                            members = doc["staff_members"]
                            if isinstance(members, list):
                                for member in members:
                                    if not isinstance(member, dict):
                                        continue
                                    
                                    m_text_parts = [f"[{config['label']}]", f"major: {doc.get('major', 'N/A')}"]
                                    for m_key, m_val in member.items():
                                        # Clean up invalid URL/values
                                        val_str = str(m_val).strip()
                                        if val_str.lower() in ["n/a", "none", "null", "not available", ""]:
                                            continue
                                        m_text_parts.append(f"{m_key}: {m_val}")
                                    
                                    m_full_text = " | ".join(m_text_parts)
                                    if len(m_full_text) > 30:
                                        mongo_texts.append(m_full_text)
                                        mongo_metadata_entries.append({
                                            "text": m_full_text,
                                            "source": f"MongoDB:{config['name']}:member",
                                            "type": "collection"
                                        })
                                continue

                        # Build searchable text from document
                        text_parts = [f"[{config['label']}]"]
                        
                        for field in config["fields"]:
                            if field in doc:
                                value = doc[field]
                                if isinstance(value, str):
                                    # Clean up
                                    if value.strip().lower() not in ["n/a", "none", "null", ""]:
                                        text_parts.append(f"{field}: {value}")
                                elif isinstance(value, list):
                                    # Handle arrays
                                    for item in value:
                                        if isinstance(item, dict):
                                            # Flatten dict to string with cleaning
                                            item_parts = []
                                            for k, v in item.items():
                                                if str(v).strip().lower() not in ["n/a", "none", "null", ""]:
                                                    item_parts.append(f"{k}: {v}")
                                            if item_parts:
                                                text_parts.append(", ".join(item_parts))
                                        else:
                                            if str(item).strip().lower() not in ["n/a", "none", "null", ""]:
                                                text_parts.append(str(item))
                                elif isinstance(value, (int, float)):
                                    text_parts.append(f"{field}: {value}")
                        
                        # Also add any other string/number fields
                        for k, v in doc.items():
                            if k not in config["fields"] and isinstance(v, (str, int, float)):
                                text_parts.append(f"{k}: {v}")
                        
                        full_text = " | ".join(text_parts)
                        
                        if len(full_text) > 50:  # Only meaningful documents
                            mongo_texts.append(full_text)
                            mongo_metadata_entries.append({
                                "text": full_text,
                                "source": f"MongoDB:{config['name']}",
                                "type": "collection"
                            })
                            
                except Exception as e:
                    print(f"[RAG] Error indexing {config['name']}: {e}")
                    continue
            
            if mongo_texts:
                # Preserve non-Mongo knowledge entries and rebuild index to avoid Mongo duplication on restart.
                preserved_entries = [
                    m for m in self.metadata
                    if not str(m.get("source", "")).startswith("MongoDB:")
                ]
                preserved_entries = [m for m in preserved_entries if m.get("text")]

                rebuilt_texts = [m["text"] for m in preserved_entries] + mongo_texts
                rebuilt_metadata = preserved_entries + mongo_metadata_entries

                # Rebuild FAISS from scratch so Mongo docs stay idempotent across startups.
                self._create_new_index()
                embeddings = np.array(self.model.encode(rebuilt_texts)).astype('float32')
                if getattr(self.index, "metric_type", None) == faiss.METRIC_INNER_PRODUCT:
                    faiss.normalize_L2(embeddings)
                self.index.add(embeddings)
                self.metadata = rebuilt_metadata

                # Save persistently
                self._save_index()
                
                indexed_count = len(mongo_texts)
                print(f"[RAG] Indexed {indexed_count} MongoDB documents (total index size: {len(self.metadata)})")
            
            return indexed_count
            
        except Exception as e:
            print(f"[RAG] MongoDB indexing error: {e}")
            return 0

    def search(self, query: str, n_results=3, preferred_labels: Optional[List[str]] = None) -> dict:
        """
        Search for relevant context from multiple sources with confidence scoring.
        
        Returns: {
            "context": str,            # Combined search results
            "has_relevant_data": bool, # True if relevant data found
            "confidence": float,       # 0~1 confidence score
            "sources": list            # List of sources that matched
        }
        """
        CONFIDENCE_THRESHOLD = 0.35  # Below this = NO_DATA_FOUND
        results_with_scores = []  # [(text, score, source), ...]
        sources_matched = []
        query_variants = self._expand_query_variants(query)
        if not query_variants:
            query_variants = [query]
        effective_preferred = set(preferred_labels or [])
        effective_preferred.update(self._infer_preferred_labels(query))
        scoped_labels = self._expand_domain_scope(effective_preferred, query)
        route_specific = self._is_route_specific_query(query)

        if self._is_forced_no_data_query(query):
            return {
                "context": "[NO_RELEVANT_DATA_FOUND] Query appears fictional or out-of-scope.",
                "has_relevant_data": False,
                "confidence": 0.0,
                "sources": []
            }
        
        # 0. SELF-LEARNING CHECK (Highest Priority, score=1.0)
        try:
            from .db_engine import db_engine
            if db_engine.connected:
                for qv in query_variants[:3]:
                    learned_ans = db_engine.search_learned_response(qv)
                    if learned_ans:
                        results_with_scores.append((
                            f"[Verified Answer]\n{learned_ans}",
                            self._apply_domain_boost(1.0, "LearnedQA", effective_preferred),
                            "LearnedQA"
                        ))
                        sources_matched.append("LearnedQA")
                        break
        except Exception as e:
            print(f"LearnedQA Search Error: {e}")

        # 1. FAISS Document Search (score = 1/(1+distance))
        if self.enabled and self.index is not None and self.index.ntotal > 0:
            try:
                vector_queries = query_variants[:4]
                query_vectors = np.array(self.model.encode(vector_queries)).astype('float32')
                if getattr(self.index, "metric_type", None) == faiss.METRIC_INNER_PRODUCT:
                    faiss.normalize_L2(query_vectors)
                faiss_k = max(5, n_results)
                D_all, I_all = self.index.search(query_vectors, k=faiss_k)

                for qi, qv in enumerate(vector_queries):
                    qv_l = str(qv or "").lower()
                    variant_bonus = 1.0 if qi == 0 else 0.96
                    for i, idx in enumerate(I_all[qi]):
                        if idx == -1 or idx >= len(self.metadata):
                            continue
                        distance = D_all[qi][i]
                        metric = getattr(self.index, "metric_type", None)
                        if metric == faiss.METRIC_INNER_PRODUCT:
                            # Map cosine similarity (-1~1) to 0~1.
                            score = ((float(distance) + 1.0) / 2.0) * variant_bonus
                        else:
                            # Legacy fallback for L2 indices.
                            score = (1.0 / (1.0 + float(distance))) * variant_bonus

                        text_payload = self.metadata[idx].get("text", "")
                        if not text_payload:
                            continue

                        # Small lexical overlap bonus to reward close paraphrases.
                        q_tokens = set(re.findall(r"[a-z0-9가-힣]{2,}", qv_l))
                        t_tokens = set(re.findall(r"[a-z0-9가-힣]{2,}", str(text_payload).lower()))
                        if q_tokens and t_tokens:
                            overlap = len(q_tokens.intersection(t_tokens)) / max(1, min(len(q_tokens), len(t_tokens)))
                            score += min(0.12, overlap * 0.12)

                        # Keep slightly lower floor to allow semantic matches that are not exact-keyword hits.
                        if score > 0.26:
                            source_name = self.metadata[idx].get('source', 'doc')
                            label = self._source_to_label(source_name, text_payload)
                            if scoped_labels and label and label not in scoped_labels:
                                continue
                            if route_specific and label in {"Schedule", "Facility"} and not self._is_route_relevant_text(text_payload):
                                continue
                            boosted_score = self._apply_domain_boost(score, label, effective_preferred)
                            results_with_scores.append((
                                f"[Document] {text_payload}",
                                boosted_score,
                                f"FAISS:{source_name}"
                            ))
                            if "FAISS" not in sources_matched:
                                sources_matched.append("FAISS")
            except Exception as e:
                print(f"FAISS Search Error: {e}")
        
        # 2. MongoDB Collection Search (with score normalization)
        try:
            from .db_engine import db_engine
            if db_engine.connected and db_engine.db is not None:
                direct_results = []
                for qv in query_variants[:4]:
                    try:
                        direct_results.extend(self._search_facility_direct(db_engine.db, qv))
                    except Exception as e:
                        print(f"Direct facility search error: {e}")
                    try:
                        direct_results.extend(self._search_staff_role_direct(db_engine.db, qv))
                    except Exception as e:
                        print(f"Direct staff role search error: {e}")

                for text, score, source, label in direct_results:
                    if scoped_labels and label and label not in scoped_labels:
                        continue
                    if route_specific and label in {"Schedule", "Facility"} and not self._is_route_relevant_text(text):
                        continue
                    boosted_score = self._apply_domain_boost(score, label, effective_preferred)
                    results_with_scores.append((text, boosted_score, source))
                    if source not in sources_matched:
                        sources_matched.append(source)
                
                def smart_search_collection(
                    coll_name: str,
                    text_fields: list,
                    label: str,
                    queries: List[str],
                    limit: int = 15
                ) -> list:
                    """
                    Enhanced search with multiple strategies:
                    - Text Search: MongoDB $text with scoring
                    - Keyword Search: Extract key terms and search each
                    - Regex Search: Flexible pattern matching
                    """
                    coll = db_engine.db[coll_name]
                    found_items = []
                    seen_ids = set()
                    
                    # 🚀 ENHANCED Keyword Extraction (Enterprise-grade)
                    def extract_keywords(q):
                        """
                        World-class keyword extraction supporting:
                        - Location patterns: Block A, Room 101, Lab 3
                        - Course codes: CS101, IT201
                        - Korean keywords: 기숙사, 도서관, 학비
                        - Multi-word entities: Computer Science, Kuala Lumpur
                        """
                        stopwords = {'what', 'where', 'which', 'how', 'is', 'are', 'the', 'a', 'an', 
                                    'of', 'in', 'on', 'at', 'to', 'for', 'do', 'does', 'can', 'i',
                                    'you', 'there', 'have', 'has', 'my', 'about', 'tell', 'me',
                                    '은', '는', '이', '가', '을', '를', '에', '에서', '으로', '로'}
                        
                        extracted = []
                        
                        # 1. PRIORITY: Location/Building patterns (Block A, Room 101, Lab 3)
                        location_patterns = re.findall(
                            r'\b(?:Block|Room|Lab|Building|Hall|Wing|Level|Floor)\s*[A-Z0-9]+\b', 
                            q, re.IGNORECASE
                        )
                        extracted.extend([p.lower() for p in location_patterns])
                        
                        # 2. Course codes (CS101, IT201, UCSI001)
                        course_codes = re.findall(r'\b[A-Z]{2,4}\s*\d{2,4}\b', q, re.IGNORECASE)
                        extracted.extend([c.lower().replace(' ', '') for c in course_codes])
                        
                        # 3. Alphanumeric IDs (A1, B2, Level 3)
                        alphanumeric = re.findall(r'\b[A-Z]\s*\d+\b|\b\d+\s*[A-Z]\b', q, re.IGNORECASE)
                        extracted.extend([a.lower().replace(' ', '') for a in alphanumeric])
                        
                        # 4. Multi-word entities (Title Case phrases)
                        title_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', q)
                        extracted.extend([p.lower() for p in title_phrases])
                        
                        # 5. Korean keywords (숙소, 학비, 도서관, 기숙사, etc.)
                        korean_words = re.findall(r'[가-힣]{2,}', q)
                        extracted.extend(korean_words)
                        
                        # 6. Basic word extraction (fallback)
                        words = re.findall(r'\b\w+\b', q.lower())
                        basic_keywords = [w for w in words if w not in stopwords and len(w) > 1]
                        extracted.extend(basic_keywords)
                        
                        # 7. Also keep original case versions for exact matching
                        original_patterns = re.findall(r'\b[A-Z]+\s*[A-Z0-9]*\b', q)  # "BLOCK A", "IT"
                        extracted.extend([p for p in original_patterns if len(p) > 1])
                        
                        # Deduplicate while preserving order
                        seen = set()
                        result = []
                        for kw in extracted:
                            kw_lower = kw.lower() if isinstance(kw, str) else kw
                            if kw_lower not in seen and kw_lower:
                                seen.add(kw_lower)
                                result.append(kw)
                        
                        return result

                    generic_keywords = {
                        "campus", "fee", "fees", "tuition", "program", "programme", "course",
                        "information", "details", "what", "where", "when", "who", "how",
                        "tell", "about", "is", "are", "for", "with", "service"
                    }

                    for qx in (queries or [])[:4]:
                        if len(found_items) >= limit:
                            break
                        keywords = extract_keywords(qx)

                        # A. Text Search (with score)
                        try:
                            cursor = coll.find(
                                {"$text": {"$search": qx}},
                                {"score": {"$meta": "textScore"}, "_id": 0}
                            ).sort([("score", {"$meta": "textScore"})]).limit(limit)

                            for doc in cursor:
                                text_score = doc.get('score', 0)
                                normalized_score = min(text_score / 3.0, 1.0)

                                if normalized_score >= 0.25:  # Lowered threshold
                                    s_doc = str(doc)
                                    if s_doc not in seen_ids:
                                        seen_ids.add(s_doc)
                                        found_items.append((doc, normalized_score, "text"))
                        except Exception:
                            pass  # Text index might not exist

                        # B. Keyword-based Regex Search (more precise)
                        if len(found_items) < limit and keywords:
                            try:
                                for keyword in keywords[:8]:
                                    token = str(keyword or "").strip()
                                    if len(token) < 2:
                                        continue
                                    is_generic = token.lower() in generic_keywords

                                    if self._is_ascii_token(token):
                                        pattern = rf"\b{re.escape(token)}\b"
                                    else:
                                        # For CJK/non-latin terms, word boundaries are unreliable.
                                        pattern = re.escape(token)

                                    regex_conditions = []
                                    for field in text_fields:
                                        regex_conditions.append({
                                            field: {"$regex": pattern, "$options": "i"}
                                        })

                                    if regex_conditions:
                                        cursor = coll.find({"$or": regex_conditions}, {"_id": 0}).limit(limit)
                                        for doc in cursor:
                                            s_doc = str(doc)
                                            if s_doc not in seen_ids:
                                                seen_ids.add(s_doc)
                                                # Penalize generic keywords to reduce false positives.
                                                match_score = 0.65 if not is_generic else 0.30
                                                found_items.append((doc, match_score, f"keyword:{token}"))
                            except Exception:
                                pass

                        # C. Fuzzy Regex Search (fallback)
                        if len(found_items) < limit:
                            try:
                                # Use only specific tokens for fallback regex to avoid broad false matches.
                                specific_tokens = [
                                    str(k) for k in keywords
                                    if len(str(k)) >= 2 and str(k).lower() not in generic_keywords
                                ]
                                cleaned_q = re.escape(specific_tokens[0]) if specific_tokens else ""
                                if not cleaned_q:
                                    continue
                                regex_conditions = []

                                for field in text_fields:
                                    regex_conditions.append({field: {"$regex": cleaned_q, "$options": "i"}})

                                if regex_conditions:
                                    cursor = coll.find({"$or": regex_conditions}, {"_id": 0}).limit(limit)
                                    for doc in cursor:
                                        s_doc = str(doc)
                                        if s_doc not in seen_ids:
                                            seen_ids.add(s_doc)
                                            found_items.append((doc, 0.32, "regex"))
                            except Exception:
                                pass

                    return found_items

                # Define Search Domains
                domains = [
                    ('UCSI_ MAJOR', ['Programme', 'Fields of Study', 'Local Students Fees', 'International Students Fees', 'Url'], 'Programme'),
                    ('Hostel', ['room_type', 'building', 'campus', 'category', 'features'], 'Hostel'),
                    ('UCSI_FACILITY', ['name', 'location', 'category', 'opening_hours', 'tags'], 'Facility'),
                    ('USCI_SCHEDUAL', ['event_name', 'event_type', 'programme', 'campus_scope'], 'Schedule'),
                    ('UCSI_STAFF', ['staff_members.name', 'major', 'staff_members.role', 'staff_members.email'], 'Staff'),
                    ('UCSI_HOSTEL_FAQ', ['question', 'answer', 'category', 'tags'], 'HostelFAQ'),
                    ('UCSI_University_Blocks_Data', ['Name', 'Address', 'BLOCK A_BUILDING', 'BLOCK B_BUIILDING', 'BLOCK C_BUILDING', 'BLOCK D_BUILDING', 'BLOCK E_BUILDING', 'BLOCKG_BUILDING', 'KUCHING_BUILDING', 'SPRINGHILL_BUILDING'], 'CampusBlocks')
                ]

                if scoped_labels:
                    domains = [d for d in domains if d[2] in scoped_labels]

                # Execute Universal Search
                for coll_name, fields, label in domains:
                    try:
                        domain_results = smart_search_collection(
                            coll_name,
                            fields,
                            label,
                            query_variants,
                        )
                        
                        for doc, score, match_type in domain_results:
                            # Format document
                            name = (
                                doc.get('name')
                                or doc.get('Programme')
                                or doc.get('room_type')
                                or doc.get('event_name')
                                or doc.get('question')
                                or doc.get('Name')
                                or "Item"
                            )
                            details_parts = []
                            for k, v in doc.items():
                                if k in ['_id', 'score', 'name', 'Programme', 'room_type', 'event_name']:
                                    continue
                                if isinstance(v, (str, int, float)):
                                    details_parts.append(f"{k}: {v}")
                                elif k == "staff_members" and isinstance(v, list):
                                    # Flatten all staff entities to include all relevant links.
                                    for member in v:
                                        if not isinstance(member, dict):
                                            continue
                                        m_name = member.get("name")
                                        m_role = member.get("role")
                                        if m_name and m_role:
                                            details_parts.append(f"staff: {m_name} ({m_role})")
                                        elif m_name:
                                            details_parts.append(f"staff: {m_name}")
                                        elif m_role:
                                            details_parts.append(f"role: {m_role}")
                                        m_email = member.get("email")
                                        if m_email:
                                            details_parts.append(f"email: {m_email}")
                                        m_profile_url = member.get("profile_url")
                                        if m_profile_url:
                                            details_parts.append(f"profile_url: {m_profile_url}")
                            details = ", ".join(details_parts)
                            
                            text = f"[{label}] {name}: {details}"
                            if route_specific and label in {"Schedule", "Facility"} and not self._is_route_relevant_text(text):
                                continue
                            boosted_score = self._apply_domain_boost(score, label, effective_preferred)
                            results_with_scores.append((text, boosted_score, f"MongoDB:{label}"))
                            
                            if f"MongoDB:{label}" not in sources_matched:
                                sources_matched.append(f"MongoDB:{label}")
                                
                    except Exception as e:
                        print(f"Error searching {coll_name}: {e}")

        except Exception as e:
            print(f"MongoDB RAG Search Error: {e}")
        
        if results_with_scores:
            # Keep highest score per (text, source) to avoid duplicate hits from query variants.
            deduped = {}
            for text, score, source in results_with_scores:
                key = (text, source)
                prev = deduped.get(key)
                if prev is None or score > prev[1]:
                    deduped[key] = (text, score, source)
            results_with_scores = list(deduped.values())

        # 3. Calculate Final Confidence and Build Context
        if not results_with_scores:
            return {
                "context": "[NO_RELEVANT_DATA_FOUND]",
                "has_relevant_data": False,
                "confidence": 0.0,
                "sources": []
            }
        
        # Sort by score (highest first)
        results_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get max confidence
        max_confidence = results_with_scores[0][1]
        
        # Check if we have relevant data
        has_relevant = max_confidence >= CONFIDENCE_THRESHOLD
        
        if has_relevant:
            # Build context from top results (limit to avoid token overflow)
            context_parts = []
            for text, score, source in results_with_scores[:20]:  # Top 20 (Expanded from 5)
                if score >= CONFIDENCE_THRESHOLD:
                    context_parts.append(f"{text} [conf:{score:.2f}]")
            
            return {
                "context": "\n\n".join(context_parts),
                "has_relevant_data": True,
                "confidence": max_confidence,
                "sources": sources_matched
            }
        else:
            return {
                "context": f"[NO_RELEVANT_DATA_FOUND] Searched: {', '.join(sources_matched) if sources_matched else 'all sources'}. Best match confidence: {max_confidence:.2f}",
                "has_relevant_data": False,
                "confidence": max_confidence,
                "sources": sources_matched
            }

# Singleton
rag_engine = RAGEngine()

if __name__ == "__main__":
    if HAS_DEPENDENCIES:
        print("Dependencies found. Initializing FAISS RAG...")
    else:
        print("RAG dependencies missing.")
