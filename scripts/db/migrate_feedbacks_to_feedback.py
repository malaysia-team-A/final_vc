import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pymongo import MongoClient

CANONICAL_COLLECTION = "Feedback"
LEGACY_COLLECTION = "feedbacks"
RATING_ALLOWED = {"positive", "negative"}
TOKEN_RE = re.compile(r"[a-z0-9가-힣]{2,}")
POLICY_TAG_RULES = {
    "no_hallucination": ["halluc", "fabricat", "made up", "invent"],
    "grounded_to_db": ["db", "database", "source", "citation"],
    "verify_numbers": ["wrong number", "incorrect number", "price", "fee", "tuition"],
    "more_specific": ["generic", "too broad", "vague", "too generic"],
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_env() -> str:
    root = _project_root()
    load_dotenv(dotenv_path=root / ".env")
    uri = str(os.getenv("MONGO_URI") or "").strip()
    if not uri:
        raise RuntimeError("MONGO_URI not found in .env")
    return uri


def _db_name_from_uri(uri: str) -> str:
    return uri.split("/")[-1].split("?")[0] or "UCSI_DB"


def _normalize_text(text: Any) -> str:
    value = str(text or "").strip().lower()
    value = value.replace("_", " ")
    value = re.sub(r"[\t\r\n]+", " ", value)
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _extract_tokens(norm: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for tok in TOKEN_RE.findall(str(norm or "")):
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out[:24]


def _derive_policy_tags(comment: Any, ai_response: Any) -> List[str]:
    text = f"{str(comment or '')} {str(ai_response or '')}".lower()
    tags: List[str] = []
    for tag, keywords in POLICY_TAG_RULES.items():
        if any(k in text for k in keywords):
            tags.append(tag)
    return tags


def _fingerprint(doc: Dict[str, Any]) -> str:
    payload = "|".join(
        [
            _normalize_text(doc.get("user_message")),
            _normalize_text(doc.get("ai_response")),
            str(doc.get("rating") or "").strip().lower(),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _normalize_legacy_doc(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    user_message = str(raw.get("user_message") or raw.get("query") or "").strip()
    ai_response = str(raw.get("ai_response") or raw.get("response") or "").strip()
    rating = str(raw.get("rating") or "").strip().lower()
    if not user_message or not ai_response or rating not in RATING_ALLOWED:
        return None

    query_norm = _normalize_text(user_message)
    doc = {
        "user_message": user_message,
        "ai_response": ai_response,
        "rating": rating,
        "reward": 1.0 if rating == "positive" else -1.0,
        "comment": None if raw.get("comment") is None else str(raw.get("comment")),
        "session_id": str(raw.get("session_id") or "legacy_feedbacks_import"),
        "timestamp": str(raw.get("timestamp") or datetime.now(timezone.utc).isoformat()),
        "query_norm": query_norm,
        "query_tokens": _extract_tokens(query_norm),
        "policy_tags": _derive_policy_tags(raw.get("comment"), ai_response),
        "migrated_from": LEGACY_COLLECTION,
        "migrated_at": datetime.now(timezone.utc).isoformat(),
    }
    if raw.get("id") is not None:
        doc["legacy_id"] = raw.get("id")
    return doc


def _ensure_indexes(coll) -> None:
    coll.create_index(
        [("query_norm", 1), ("rating", 1), ("timestamp", -1)],
        name="feedback_query_rating_ts",
    )
    coll.create_index([("timestamp", -1)], name="feedback_timestamp_desc")


def _write_backup(rows: List[Dict[str, Any]]) -> Path:
    backup_dir = _project_root() / "data" / "reference"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"feedbacks_backup_{stamp}.json"
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, default=str)
    return backup_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safely migrate legacy 'feedbacks' docs into canonical 'Feedback'."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (default is dry-run)",
    )
    parser.add_argument(
        "--drop-source",
        action="store_true",
        help="Drop legacy 'feedbacks' collection after successful apply",
    )
    args = parser.parse_args()

    uri = _load_env()
    client = MongoClient(uri, serverSelectionTimeoutMS=7000)
    client.admin.command("ping")
    db = client[_db_name_from_uri(uri)]

    collections = set(db.list_collection_names())
    if LEGACY_COLLECTION not in collections:
        print(f"[INFO] '{LEGACY_COLLECTION}' collection not found. Nothing to migrate.")
        return 0

    legacy_coll = db[LEGACY_COLLECTION]
    target_coll = db[CANONICAL_COLLECTION]

    existing_fingerprints = set()
    for row in target_coll.find({}, {"_id": 0, "user_message": 1, "ai_response": 1, "rating": 1}):
        existing_fingerprints.add(_fingerprint(row))

    source_rows = list(legacy_coll.find({}, {"_id": 0}))
    normalized_rows = []
    skipped_invalid = 0
    skipped_duplicate = 0
    for raw in source_rows:
        normalized = _normalize_legacy_doc(raw)
        if not normalized:
            skipped_invalid += 1
            continue
        fp = _fingerprint(normalized)
        if fp in existing_fingerprints:
            skipped_duplicate += 1
            continue
        existing_fingerprints.add(fp)
        normalized_rows.append(normalized)

    print(f"[INFO] Source rows: {len(source_rows)}")
    print(f"[INFO] Insert candidates: {len(normalized_rows)}")
    print(f"[INFO] Skipped invalid: {skipped_invalid}")
    print(f"[INFO] Skipped duplicates: {skipped_duplicate}")

    if not args.apply:
        print("[DRY-RUN] No changes applied. Re-run with --apply to migrate.")
        return 0

    backup_path = _write_backup(source_rows)
    print(f"[APPLY] Backup written: {backup_path}")

    inserted = 0
    if normalized_rows:
        result = target_coll.insert_many(normalized_rows, ordered=False)
        inserted = len(result.inserted_ids)
    _ensure_indexes(target_coll)
    print(f"[APPLY] Inserted into '{CANONICAL_COLLECTION}': {inserted}")

    if args.drop_source:
        legacy_coll.drop()
        print(f"[APPLY] Dropped legacy collection '{LEGACY_COLLECTION}'.")
    else:
        print(f"[APPLY] Legacy collection '{LEGACY_COLLECTION}' kept (use --drop-source to remove).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
