import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from pymongo import MongoClient

COLLECTION = "Feedback"
TOKEN_RE = re.compile(r"[a-z0-9가-힣]{2,}")
RATING_ALLOWED = {"positive", "negative"}
POLICY_TAG_RULES = {
    "no_hallucination": ["halluc", "fabricat", "made up", "invent"],
    "grounded_to_db": ["db", "database", "source", "citation"],
    "verify_numbers": ["wrong number", "incorrect number", "price", "fee", "tuition"],
    "more_specific": ["generic", "too broad", "vague", "too generic"],
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def _merge_tags(existing: Any, derived: List[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    if isinstance(existing, list):
        for tag in existing:
            t = str(tag).strip()
            if t and t not in seen:
                seen.add(t)
                merged.append(t)
    for tag in derived:
        t = str(tag).strip()
        if t and t not in seen:
            seen.add(t)
            merged.append(t)
    return merged


def _load_db():
    root = _project_root()
    load_dotenv(dotenv_path=root / ".env")
    uri = str(os.getenv("MONGO_URI") or "").strip()
    if not uri:
        raise RuntimeError("MONGO_URI not found in .env")
    client = MongoClient(uri, serverSelectionTimeoutMS=7000)
    client.admin.command("ping")
    db_name = uri.split("/")[-1].split("?")[0] or "UCSI_DB"
    return client[db_name]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill and normalize Feedback collection fields for RLHF runtime."
    )
    parser.add_argument("--apply", action="store_true", help="Apply updates (default dry-run)")
    args = parser.parse_args()

    db = _load_db()
    coll = db[COLLECTION]
    rows = list(
        coll.find(
            {},
            {
                "_id": 1,
                "user_message": 1,
                "ai_response": 1,
                "rating": 1,
                "reward": 1,
                "comment": 1,
                "session_id": 1,
                "query_norm": 1,
                "query_tokens": 1,
                "policy_tags": 1,
            },
        )
    )

    updates = []
    skipped_missing_required = 0
    for row in rows:
        user_message = str(row.get("user_message") or "").strip()
        ai_response = str(row.get("ai_response") or "").strip()
        if not user_message or not ai_response:
            skipped_missing_required += 1
            continue

        rating = str(row.get("rating") or "").strip().lower()
        reward = 1.0 if rating == "positive" else (-1.0 if rating == "negative" else 0.0)
        query_norm = _normalize_text(user_message)
        query_tokens = _extract_tokens(query_norm)
        comment = row.get("comment")
        comment_norm = None if comment is None else str(comment).strip() or None
        policy_tags = _merge_tags(row.get("policy_tags"), _derive_policy_tags(comment_norm, ai_response))
        session_id = str(row.get("session_id") or "guest_session").strip() or "guest_session"

        patch: Dict[str, Any] = {}
        if row.get("query_norm") != query_norm:
            patch["query_norm"] = query_norm
        if row.get("query_tokens") != query_tokens:
            patch["query_tokens"] = query_tokens
        if row.get("session_id") != session_id:
            patch["session_id"] = session_id
        if row.get("reward") != reward:
            patch["reward"] = reward
        if row.get("comment") != comment_norm:
            patch["comment"] = comment_norm
        if row.get("policy_tags") != policy_tags:
            patch["policy_tags"] = policy_tags
        if patch:
            updates.append((row["_id"], patch))

    print(f"[INFO] Total docs scanned: {len(rows)}")
    print(f"[INFO] Docs needing updates: {len(updates)}")
    print(f"[INFO] Skipped missing required fields: {skipped_missing_required}")

    if not args.apply:
        print("[DRY-RUN] No changes applied. Re-run with --apply to normalize.")
        return 0

    applied = 0
    for _id, patch in updates:
        coll.update_one({"_id": _id}, {"$set": patch})
        applied += 1

    coll.create_index(
        [("query_norm", 1), ("rating", 1), ("timestamp", -1)],
        name="feedback_query_rating_ts",
    )
    coll.create_index([("timestamp", -1)], name="feedback_timestamp_desc")

    print(f"[APPLY] Updated docs: {applied}")
    print("[APPLY] Ensured indexes: feedback_query_rating_ts, feedback_timestamp_desc")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
