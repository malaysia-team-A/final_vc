"""
RAG Engine (sync core used by async wrapper)

This module intentionally keeps CPU-bound FAISS + embedding work synchronous.
`rag_engine_async` runs these methods inside a thread pool so FastAPI endpoints
stay non-blocking.
"""

from __future__ import annotations

import csv
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Suppress noisy transformer logs.
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

try:
    import faiss
    import PyPDF2
    from sentence_transformers import SentenceTransformer

    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    faiss = None  # type: ignore[assignment]
    PyPDF2 = None  # type: ignore[assignment]
    SentenceTransformer = None  # type: ignore[assignment]
    print("Warning: RAG dependencies missing. Install faiss-cpu sentence-transformers PyPDF2")

KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")
INDEX_FILE = KNOWLEDGE_BASE_DIR / "faiss_index.bin"
METADATA_FILE = KNOWLEDGE_BASE_DIR / "faiss_metadata.pkl"
DEFAULT_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
FICTIONAL_OR_INVALID_KEYWORDS: Set[str] = {
    "ufo",
    "jedi",
    "spiderman",
    "moon campus",
    "mars campus",
    "teleport pad",
    "time machine",
    "alien faculty",
}

DOMAIN_HINTS: Dict[str, Set[str]] = {
    "CampusBlocks": {
        "block",
        "building",
        "address",
        "location",
        "campus",
        "map",
        "where is",
    },
    "HostelFAQ": {"hostel faq", "refund", "installment", "policy", "guaranteed", "accommodation policy"},
    "Hostel": {"hostel", "dorm", "accommodation", "rent", "deposit", "room"},
    "Facility": {"facility", "library", "gym", "cafeteria", "pool", "laundry", "print", "printer", "prayer"},
    "Schedule": {"schedule", "calendar", "intake", "deadline", "event", "semester"},
    "Programme": {
        "programme",
        "program",
        "major",
        "course",
        "tuition",
        "fee",
        "fees",
        "scholarship",
        "foundation",
        "diploma",
        "degree",
        "master",
        "phd",
    },
    "Staff": {"staff", "lecturer", "professor", "dean", "advisor", "teacher", "faculty", "head"},
}

DOMAIN_RELATIONS: Dict[str, Set[str]] = {
    "HostelFAQ": {"Hostel"},
    "Hostel": {"HostelFAQ"},
    "CampusBlocks": {"Facility"},
    "Facility": {"CampusBlocks"},
    "Programme": {"Staff"},
    "Staff": {"Programme"},
    "Schedule": {"Programme"},
}

COLLECTION_TO_LABEL: Dict[str, str] = {
    "hostel": "Hostel",
    "ucsi_facility": "Facility",
    "usci_schedual": "Schedule",
    "ucsi_ major": "Programme",
    "ucsi_staff": "Staff",
    "ucsi_hostel_faq": "HostelFAQ",
    "ucsi_university_blocks_data": "CampusBlocks",
}

ALIAS_MAP: Dict[str, List[str]] = {
    "hostel": ["accommodation", "dorm"],
    "accommodation": ["hostel", "dorm"],
    "tuition": ["fee", "fees", "cost"],
    "fee": ["tuition", "cost", "price"],
    "programme": ["program", "major", "course"],
    "program": ["programme", "major", "course"],
    "lecturer": ["staff", "professor"],
    "professor": ["staff", "lecturer"],
    "library": ["facility", "opening hours"],
    "bus": ["shuttle", "route", "transport"],
    "route": ["bus", "shuttle", "transport"],
    "block": ["building", "address", "location"],
}


def _tokenize(text: str) -> Set[str]:
    return set(TOKEN_RE.findall(str(text or "").lower()))


class RAGEngine:
    def __init__(self) -> None:
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.model = None
        self.enabled = HAS_DEPENDENCIES
        self.model_name = DEFAULT_EMBEDDING_MODEL
        self.dimension = 384
        self.metric_type = None

        if not self.enabled:
            return

        try:
            self.model = SentenceTransformer(self.model_name)
            self.dimension = int(self.model.get_sentence_embedding_dimension() or 384)
            self.metric_type = faiss.METRIC_INNER_PRODUCT
            self._load_index()
        except Exception as exc:
            print(f"RAG init error: {exc}")
            self.enabled = False

    def _create_new_index(self) -> None:
        if not self.enabled:
            return
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []

    def _load_index(self) -> None:
        if not self.enabled:
            return

        if INDEX_FILE.exists() and METADATA_FILE.exists():
            try:
                loaded = faiss.read_index(str(INDEX_FILE))
                if getattr(loaded, "d", None) != self.dimension:
                    print(
                        f"[RAG] Index dimension mismatch (index={getattr(loaded, 'd', None)}, "
                        f"model={self.dimension}); rebuilding."
                    )
                    self._create_new_index()
                    return
                if getattr(loaded, "metric_type", None) != self.metric_type:
                    print(
                        f"[RAG] Index metric mismatch (index={getattr(loaded, 'metric_type', None)}, "
                        f"expected={self.metric_type}); rebuilding."
                    )
                    self._create_new_index()
                    return

                with METADATA_FILE.open("rb") as handle:
                    metadata = pickle.load(handle)
                    if isinstance(metadata, list):
                        self.metadata = metadata
                    else:
                        self.metadata = []

                self.index = loaded
                return
            except Exception as exc:
                print(f"[RAG] Failed to load persisted index: {exc}")

        self._create_new_index()

    def _save_index(self) -> None:
        if not self.enabled or self.index is None:
            return
        KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_FILE))
        with METADATA_FILE.open("wb") as handle:
            pickle.dump(self.metadata, handle)

    def _infer_preferred_labels(self, query: str) -> Set[str]:
        q = str(query or "").lower()
        preferred: Set[str] = set()
        for label, hints in DOMAIN_HINTS.items():
            if any(h in q for h in hints):
                preferred.add(label)
        return preferred

    def _expand_query_variants(self, query: str) -> List[str]:
        q = str(query or "").strip()
        if not q:
            return []

        ql = q.lower()
        alias_terms: List[str] = []
        for trigger, aliases in ALIAS_MAP.items():
            if trigger in ql:
                alias_terms.extend(aliases)

        out: List[str] = [q]
        if alias_terms:
            seen_alias: Set[str] = set()
            normalized_aliases: List[str] = []
            for alias in alias_terms:
                a = str(alias).strip().lower()
                if not a or a in seen_alias or a in ql:
                    continue
                seen_alias.add(a)
                normalized_aliases.append(a)

            if normalized_aliases:
                out.append(f"{q} {' '.join(normalized_aliases[:4])}")
                out.extend(normalized_aliases[:6])

        deduped: List[str] = []
        seen: Set[str] = set()
        for candidate in out:
            key = str(candidate).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(str(candidate).strip())
            if len(deduped) >= 7:
                break
        return deduped

    def _source_to_label(self, source: str, text: str = "") -> Optional[str]:
        source_lower = str(source or "").lower()
        for coll_name, label in COLLECTION_TO_LABEL.items():
            if coll_name in source_lower:
                return label

        m = re.search(r"\[([A-Za-z]+)\]", str(text or ""))
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
            return token_map.get(token)
        return None

    def _apply_domain_boost(self, score: float, label: Optional[str], preferred_labels: Set[str]) -> float:
        if not preferred_labels:
            return score

        if not label:
            return score * 0.60
        if label in preferred_labels:
            return min(score * 1.40, 1.20)
        if any(label in DOMAIN_RELATIONS.get(target, set()) for target in preferred_labels):
            return score
        return score * 0.60

    def _expand_domain_scope(self, preferred_labels: Set[str], query: str = "") -> Set[str]:
        if not preferred_labels:
            return set()

        expanded = set(preferred_labels)
        q = str(query or "").lower()
        specific_facility_intent = any(
            token in q for token in {"library", "gym", "cafeteria", "pool", "laundry", "print", "printer", "prayer"}
        )
        for label in list(preferred_labels):
            if label == "Facility" and specific_facility_intent:
                continue
            expanded.update(DOMAIN_RELATIONS.get(label, set()))
        return expanded

    def _is_forced_no_data_query(self, query: str) -> bool:
        q = str(query or "").lower()
        return any(token in q for token in FICTIONAL_OR_INVALID_KEYWORDS)

    def _is_route_specific_query(self, query: str) -> bool:
        q = str(query or "").lower()
        return "route" in q and any(token in q for token in {"bus", "shuttle"})

    def _is_route_relevant_text(self, text: str) -> bool:
        t = str(text or "").lower()
        return any(token in t for token in {"route", "bus", "shuttle", "transport"})

    def _iter_campus_block_entries(self, doc: dict) -> List[dict]:
        entries: List[dict] = []
        campus_root = (doc or {}).get("campus")
        if not isinstance(campus_root, dict):
            return entries

        def _coerce_entry(campus_name: str, payload: dict) -> Optional[dict]:
            if not isinstance(payload, dict):
                return None
            name = str(payload.get("Name") or "").strip()
            address = str(payload.get("Address") or "").strip()
            map_url = str(payload.get("MAP") or "").strip()
            image_url = str(payload.get("BUILDING_IMAGE") or "").strip()
            latitude = payload.get("Latitude")
            longitude = payload.get("Longitude")
            if not any([name, address, map_url, image_url, latitude, longitude]):
                return None
            return {
                "campus_name": str(campus_name or "").strip(),
                "name": name,
                "address": address,
                "map_url": map_url,
                "image_url": image_url,
                "latitude": latitude,
                "longitude": longitude,
            }

        for campus_name, payload in campus_root.items():
            if not isinstance(payload, dict):
                continue
            blocks = payload.get("blocks")
            if isinstance(blocks, list):
                for block in blocks:
                    entry = _coerce_entry(str(campus_name), block)
                    if entry:
                        entries.append(entry)
            else:
                entry = _coerce_entry(str(campus_name), payload)
                if entry:
                    entries.append(entry)
        return entries

    def _build_campus_block_text(self, entry: dict) -> str:
        parts = [f"campus: {entry.get('campus_name')}", f"name: {entry.get('name')}"]
        if entry.get("address"):
            parts.append(f"address: {entry.get('address')}")
        if entry.get("map_url"):
            parts.append(f"map: {entry.get('map_url')}")
        if entry.get("image_url"):
            parts.append(f"building_image: {entry.get('image_url')}")
        if entry.get("latitude") is not None and entry.get("longitude") is not None:
            parts.append(f"coordinates: {entry.get('latitude')}, {entry.get('longitude')}")
        return f"[CampusBlocks] {' | '.join(parts)}"

    def ingest_file(self, file_path: str) -> bool:
        if not self.enabled or self.index is None or self.model is None:
            return False

        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return False

        try:
            ext = path.suffix.lower()
            text = ""
            if ext == ".pdf":
                with path.open("rb") as handle:
                    reader = PyPDF2.PdfReader(handle)
                    for page in reader.pages:
                        extracted = page.extract_text() or ""
                        if extracted.strip():
                            text += extracted + "\n"
            elif ext == ".txt":
                text = path.read_text(encoding="utf-8", errors="ignore")
            elif ext == ".csv":
                rows: List[str] = []
                with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
                    reader = csv.reader(handle)
                    for row in reader:
                        rows.append(",".join(str(cell) for cell in row))
                text = "\n".join(rows)
            else:
                return False

            if not text.strip():
                return False

            chunk_size = 600
            overlap = 80
            step = max(64, chunk_size - overlap)
            chunks: List[str] = []
            for i in range(0, len(text), step):
                chunk = text[i : i + chunk_size].strip()
                if len(chunk) >= 60:
                    chunks.append(chunk)
            if not chunks:
                return False

            embeddings = np.array(self.model.encode(chunks)).astype("float32")
            if getattr(self.index, "metric_type", None) == faiss.METRIC_INNER_PRODUCT:
                faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            for chunk in chunks:
                self.metadata.append({"text": chunk, "source": path.name})
            self._save_index()
            return True
        except Exception as exc:
            print(f"[RAG] ingest error for {file_path}: {exc}")
            return False

    def index_mongodb_collections(self) -> int:
        print("[RAG] Sync Mongo indexing is deprecated. Use rag_engine_async.index_mongodb_collections().")
        return 0

    def rebuild_index_with_mongo_entries(
        self,
        mongo_texts: List[str],
        mongo_metadata_entries: List[Dict[str, Any]],
    ) -> int:
        if not self.enabled or self.model is None:
            return 0

        try:
            safe_texts = [str(t) for t in (mongo_texts or []) if str(t).strip()]
            safe_meta: List[Dict[str, Any]] = []
            for i, text in enumerate(safe_texts):
                meta = None
                if i < len(mongo_metadata_entries):
                    candidate = mongo_metadata_entries[i]
                    if isinstance(candidate, dict):
                        meta = dict(candidate)
                if not meta:
                    meta = {"text": text, "source": "MongoDB:unknown", "type": "collection"}
                meta["text"] = text
                meta.setdefault("source", "MongoDB:unknown")
                meta.setdefault("type", "collection")
                safe_meta.append(meta)

            preserved = [
                item
                for item in self.metadata
                if isinstance(item, dict)
                and str(item.get("text") or "").strip()
                and not str(item.get("source") or "").startswith("MongoDB:")
            ]

            rebuilt_texts = [str(item["text"]) for item in preserved] + safe_texts
            rebuilt_meta = preserved + safe_meta

            self._create_new_index()
            if self.index is None:
                return 0

            if rebuilt_texts:
                embeddings = np.array(self.model.encode(rebuilt_texts)).astype("float32")
                if getattr(self.index, "metric_type", None) == faiss.METRIC_INNER_PRODUCT:
                    faiss.normalize_L2(embeddings)
                self.index.add(embeddings)

            self.metadata = rebuilt_meta
            self._save_index()
            print(f"[RAG] Rebuilt Mongo entries: {len(safe_texts)} (total index size: {len(self.metadata)})")
            return len(safe_texts)
        except Exception as exc:
            print(f"[RAG] Mongo rebuild error: {exc}")
            return 0

    def search(
        self,
        query: str,
        n_results: int = 3,
        preferred_labels: Optional[List[str]] = None,
    ) -> dict:
        """
        Return:
        {
          "context": str,
          "has_relevant_data": bool,
          "confidence": float,
          "sources": list[str]
        }
        """
        confidence_threshold = 0.35
        q = str(query or "").strip()
        if not q:
            return {
                "context": "[NO_RELEVANT_DATA_FOUND]",
                "has_relevant_data": False,
                "confidence": 0.0,
                "sources": [],
            }

        if self._is_forced_no_data_query(q):
            return {
                "context": "[NO_RELEVANT_DATA_FOUND] Query appears fictional or out-of-scope.",
                "has_relevant_data": False,
                "confidence": 0.0,
                "sources": [],
            }

        query_variants = self._expand_query_variants(q) or [q]
        effective_preferred = set(preferred_labels or [])
        effective_preferred.update(self._infer_preferred_labels(q))
        scoped_labels = self._expand_domain_scope(effective_preferred, q)
        route_specific = self._is_route_specific_query(q)

        results: List[Tuple[str, float, str, Optional[str]]] = []
        if self.enabled and self.index is not None and self.model is not None and self.index.ntotal > 0:
            try:
                vector_queries = query_variants[:4]
                query_vectors = np.array(self.model.encode(vector_queries)).astype("float32")
                if getattr(self.index, "metric_type", None) == faiss.METRIC_INNER_PRODUCT:
                    faiss.normalize_L2(query_vectors)
                faiss_k = max(8, int(n_results) * 3)
                distances_all, indices_all = self.index.search(query_vectors, k=faiss_k)

                for q_idx, variant in enumerate(vector_queries):
                    variant_bonus = 1.0 if q_idx == 0 else 0.96
                    variant_tokens = _tokenize(variant)

                    for item_idx, metadata_idx in enumerate(indices_all[q_idx]):
                        if metadata_idx == -1 or metadata_idx >= len(self.metadata):
                            continue

                        meta = self.metadata[metadata_idx] or {}
                        text_payload = str(meta.get("text") or "").strip()
                        if not text_payload:
                            continue

                        raw_distance = float(distances_all[q_idx][item_idx])
                        if getattr(self.index, "metric_type", None) == faiss.METRIC_INNER_PRODUCT:
                            base_score = ((raw_distance + 1.0) / 2.0) * variant_bonus
                        else:
                            base_score = (1.0 / (1.0 + max(0.0, raw_distance))) * variant_bonus

                        text_tokens = _tokenize(text_payload)
                        if variant_tokens and text_tokens:
                            overlap = len(variant_tokens.intersection(text_tokens)) / max(
                                1, min(len(variant_tokens), len(text_tokens))
                            )
                            base_score += min(0.12, overlap * 0.12)

                        source_name = str(meta.get("source") or "doc")
                        label = self._source_to_label(source_name, text_payload)
                        if scoped_labels and label and label not in scoped_labels:
                            continue
                        if route_specific and label in {"Schedule", "Facility"} and not self._is_route_relevant_text(
                            text_payload
                        ):
                            continue

                        boosted_score = self._apply_domain_boost(base_score, label, effective_preferred)
                        if boosted_score < 0.24:
                            continue
                        results.append((text_payload, boosted_score, source_name, label))
            except Exception as exc:
                print(f"[RAG] search error: {exc}")

        if not results:
            return {
                "context": "[NO_RELEVANT_DATA_FOUND]",
                "has_relevant_data": False,
                "confidence": 0.0,
                "sources": [],
            }

        deduped: Dict[Tuple[str, str], Tuple[str, float, str, Optional[str]]] = {}
        for text, score, source, label in results:
            key = (text, source)
            prev = deduped.get(key)
            if prev is None or score > prev[1]:
                deduped[key] = (text, score, source, label)

        ranked = sorted(deduped.values(), key=lambda item: item[1], reverse=True)
        max_confidence = float(ranked[0][1])
        has_relevant = max_confidence >= confidence_threshold

        source_list: List[str] = []
        seen_sources: Set[str] = set()
        for _, _, source, _ in ranked[:8]:
            s = str(source).strip()
            if s and s not in seen_sources:
                seen_sources.add(s)
                source_list.append(s)

        if not has_relevant:
            return {
                "context": (
                    "[NO_RELEVANT_DATA_FOUND] "
                    f"Best match confidence: {max_confidence:.2f}"
                ),
                "has_relevant_data": False,
                "confidence": max_confidence,
                "sources": source_list,
            }

        context_parts: List[str] = []
        for text, score, source, label in ranked[:5]:
            if score < (confidence_threshold * 0.75):
                continue
            prefix = f"[{label}]" if label else "[Document]"
            context_parts.append(f"{prefix} {text} [conf:{score:.2f}]")

        if not context_parts:
            context_parts = [f"[Document] {ranked[0][0]} [conf:{ranked[0][1]:.2f}]"]

        return {
            "context": "\n\n".join(context_parts),
            "has_relevant_data": True,
            "confidence": max_confidence,
            "sources": source_list,
        }


# Singleton
rag_engine = RAGEngine()

