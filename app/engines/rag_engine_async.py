import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from app.engines.db_engine_async import db_engine_async
from app.engines.rag_engine import rag_engine as _sync_rag_engine

_MONGO_COLLECTIONS_CONFIG: List[Dict[str, Any]] = [
    {
        "name": "Hostel",
        "aliases": ["HOSTEL"],
        "fields": ["room_type", "building", "campus", "category", "rent_price", "deposit", "features"],
        "label": "Hostel",
    },
    {
        "name": "UCSI_FACILITY",
        "aliases": ["UCSI_FACILITIES"],
        "fields": ["name", "location", "category", "opening_hours", "price_info", "tags"],
        "label": "Facility",
    },
    {
        "name": "UCSI_ MAJOR",
        "aliases": ["UCSI_MAJOR", "UCSI_MAJORS"],
        "fields": [
            "Programme",
            "Fields of Study",
            "Course Duration",
            "Course Mode",
            "Course Location",
            "Intakes",
            "Local Students Fees",
            "International Students Fees",
            "Programme Overview",
        ],
        "label": "Programme",
    },
    {
        "name": "USCI_SCHEDUAL",
        "aliases": ["UCSI_SCHEDUAL", "UCSI_SCHEDULE", "UCSI_SCHEDULES"],
        "fields": ["event_name", "event_type", "start_date", "end_date", "programme", "campus_scope"],
        "label": "Schedule",
    },
    {
        "name": "UCSI_STAFF",
        "aliases": ["UCSI_STAFFS"],
        "fields": ["major", "staff_members"],
        "label": "Staff",
    },
    {
        "name": "UCSI_HOSTEL_FAQ",
        "aliases": ["UCSI_HOSTEL_FAQS"],
        "fields": ["question", "answer", "category", "tags"],
        "label": "HostelFAQ",
    },
    {
        "name": "UCSI_University_Blocks_Data",
        "aliases": ["UCSI_UNIVERSITY_BLOCKS_DATA"],
        "fields": ["campus"],
        "label": "CampusBlocks",
    },
]


class AsyncRAGEngine:
    def __init__(self):
        # CPU-bound embedding/FAISS tasks stay in thread pool.
        self.executor = ThreadPoolExecutor(max_workers=4)
        print("[AsyncRAG] Initialized ThreadPool for FAISS operations")

    async def _collect_mongo_index_payload(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        db = db_engine_async.db
        if db is None:
            return [], []

        mongo_texts: List[str] = []
        mongo_metadata_entries: List[Dict[str, Any]] = []
        collection_names = set(await db.list_collection_names())

        for config in _MONGO_COLLECTIONS_CONFIG:
            coll_name = str(config["name"])
            aliases = [str(x).strip() for x in (config.get("aliases") or []) if str(x).strip()]
            candidate_names = [coll_name] + aliases
            matched_collection = next((n for n in candidate_names if n in collection_names), None)
            if not matched_collection:
                continue
            try:
                docs = await db[matched_collection].find({}, {"_id": 0}).to_list(length=None)
            except Exception as e:
                print(f"[AsyncRAG] Error reading {matched_collection}: {e}")
                continue

            if coll_name == "UCSI_University_Blocks_Data":
                for doc in docs:
                    for entry in _sync_rag_engine._iter_campus_block_entries(doc):
                        full_text = _sync_rag_engine._build_campus_block_text(entry)
                        if len(full_text) <= 50:
                            continue
                        mongo_texts.append(full_text)
                        mongo_metadata_entries.append(
                            {
                                "text": full_text,
                                "source": f"MongoDB:{matched_collection}",
                                "type": "collection",
                            }
                        )
                continue

            fields = list(config.get("fields") or [])
            label = str(config.get("label") or "Document")
            for doc in docs:
                text_parts: List[str] = [f"[{label}]"]
                for field in fields:
                    if field not in doc:
                        continue
                    value = doc[field]
                    if isinstance(value, str):
                        text_parts.append(f"{field}: {value}")
                    elif isinstance(value, list):
                        for item in value[:5]:
                            text_parts.append(str(item))
                    elif isinstance(value, (int, float)):
                        text_parts.append(f"{field}: {value}")

                for key, value in doc.items():
                    if key in fields:
                        continue
                    if isinstance(value, (str, int, float)):
                        text_parts.append(f"{key}: {value}")

                full_text = " | ".join(text_parts)
                if len(full_text) <= 50:
                    continue
                mongo_texts.append(full_text)
                mongo_metadata_entries.append(
                    {
                        "text": full_text,
                        "source": f"MongoDB:{matched_collection}",
                        "type": "collection",
                    }
                )

        return mongo_texts, mongo_metadata_entries

    async def search_context(
        self,
        query: str,
        top_k: int = 5,
        preferred_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        base_result = await loop.run_in_executor(
            self.executor,
            lambda: _sync_rag_engine.search(
                query=query,
                n_results=top_k,
                preferred_labels=preferred_labels,
            ),
        )

        learned_answer = None
        try:
            learned_answer = await db_engine_async.search_learned_response(str(query or ""))
        except Exception:
            learned_answer = None

        if not learned_answer:
            return base_result

        base = dict(base_result or {})
        context = str(base.get("context") or "").strip()
        learned_context = f"[Verified Answer]\n{learned_answer} [conf:1.00]"
        merged_context = learned_context
        if context and "[NO_RELEVANT_DATA_FOUND]" not in context:
            merged_context = f"{learned_context}\n\n{context}"

        sources = []
        seen = set()
        for source in ["LearnedQA"] + list(base.get("sources") or []):
            s = str(source or "").strip()
            if s and s not in seen:
                seen.add(s)
                sources.append(s)

        return {
            "context": merged_context,
            "has_relevant_data": True,
            "confidence": max(float(base.get("confidence") or 0.0), 1.0),
            "sources": sources,
        }

    async def index_mongodb_collections(self) -> int:
        db = db_engine_async.db
        if db is None:
            print("[AsyncRAG] MongoDB not connected, skipping collection indexing")
            return 0

        mongo_texts, mongo_metadata = await self._collect_mongo_index_payload()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: _sync_rag_engine.rebuild_index_with_mongo_entries(mongo_texts, mongo_metadata),
        )

    async def add_document(self, text: str, source: str) -> bool:
        """
        Backward-compatible helper for old callers that push raw text.
        Writes a temporary text file, then ingests through the sync engine.
        """
        import os
        import tempfile

        filename = os.path.basename(str(source or "uploaded.txt"))
        suffix = os.path.splitext(filename)[1].lower() or ".txt"
        if suffix not in {".txt", ".csv"}:
            suffix = ".txt"

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=suffix, delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(str(text or ""))
                tmp_path = tmp.name
            return await self.ingest_file(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    async def ingest_file(self, file_path: str) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            _sync_rag_engine.ingest_file,
            file_path,
        )

    async def get_index_info(self) -> Dict[str, Any]:
        """Get information about the current FAISS index."""
        if not _sync_rag_engine.enabled:
            return {
                "enabled": False,
                "document_count": 0,
                "index_loaded": False,
                "model_name": None,
                "dimension": 0,
            }

        index = _sync_rag_engine.index
        metadata = _sync_rag_engine.metadata

        # Count documents by source type
        source_counts: Dict[str, int] = {}
        for item in metadata:
            if isinstance(item, dict):
                source = str(item.get("source") or "unknown")
                if source.startswith("MongoDB:"):
                    key = "mongodb"
                elif source.endswith(".pdf"):
                    key = "pdf"
                elif source.endswith(".txt"):
                    key = "txt"
                elif source.endswith(".csv"):
                    key = "csv"
                else:
                    key = "other"
                source_counts[key] = source_counts.get(key, 0) + 1

        return {
            "enabled": True,
            "document_count": len(metadata),
            "index_loaded": index is not None,
            "index_size": getattr(index, "ntotal", 0) if index else 0,
            "model_name": _sync_rag_engine.model_name,
            "dimension": _sync_rag_engine.dimension,
            "source_breakdown": source_counts,
        }


# Singleton
rag_engine_async = AsyncRAGEngine()
