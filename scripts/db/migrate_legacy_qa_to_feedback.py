import os
import re
from datetime import datetime
from typing import List

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

FEEDBACK_COLLECTION = "Feedback"
LEARNED_QA_COLLECTION = "LearnedQA"
BAD_QA_COLLECTION = "BadQA"


def _normalize_query_text(query: str) -> str:
    text = str(query or "").strip().lower()
    if not text:
        return ""
    text = text.replace("_", " ")
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_query_tokens(query_norm: str) -> List[str]:
    if not query_norm:
        return []
    toks = re.findall(r"[a-z0-9]{2,}", query_norm)
    out = []
    seen = set()
    for tok in toks:
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out[:24]


def _coerce_timestamp(value):
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value or "").strip()
    return text or datetime.now().isoformat()


def main() -> int:
    load_dotenv(dotenv_path=".env")
    uri = os.getenv("MONGO_URI")
    if not uri:
        print("MONGO_URI is missing.")
        return 1

    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    db_name = uri.split("/")[-1].split("?")[0] or "UCSI_DB"
    db = client[db_name]

    feedback_coll = db[FEEDBACK_COLLECTION]
    learned_coll = db[LEARNED_QA_COLLECTION]
    bad_coll = db[BAD_QA_COLLECTION]

    learned_docs = list(
        learned_coll.find(
            {},
            {"_id": 0, "query": 1, "query_norm": 1, "query_tokens": 1, "answer": 1, "timestamp": 1, "source": 1},
        )
    )
    bad_docs = list(
        bad_coll.find(
            {},
            {
                "_id": 0,
                "query": 1,
                "query_norm": 1,
                "query_tokens": 1,
                "bad_answer": 1,
                "reason": 1,
                "timestamp": 1,
                "count": 1,
            },
        )
    )

    ops = []
    for doc in learned_docs:
        query_raw = str(doc.get("query") or "").strip()
        query_norm = str(doc.get("query_norm") or "").strip().lower() or _normalize_query_text(query_raw)
        if not query_norm:
            continue
        answer = str(doc.get("answer") or "").strip()
        if not answer:
            continue
        query_tokens = doc.get("query_tokens")
        if not isinstance(query_tokens, list):
            query_tokens = _extract_query_tokens(query_norm)
        timestamp = _coerce_timestamp(doc.get("timestamp"))
        source = str(doc.get("source") or "legacy_learnedqa").strip() or "legacy_learnedqa"

        ops.append(
            UpdateOne(
                {"query_norm": query_norm, "rating": "positive"},
                {
                    "$set": {
                        "user_message": query_raw or query_norm,
                        "ai_response": answer,
                        "rating": "positive",
                        "reward": 1.0,
                        "query_norm": query_norm,
                        "query_tokens": query_tokens,
                        "session_id": "legacy_qa_migration",
                        "timestamp": timestamp,
                        "comment": f"Migrated from {LEARNED_QA_COLLECTION}",
                        "policy_tags": ["legacy_learnedqa"],
                        "memory_source": source,
                    }
                },
                upsert=True,
            )
        )

    for doc in bad_docs:
        query_raw = str(doc.get("query") or "").strip()
        query_norm = str(doc.get("query_norm") or "").strip().lower() or _normalize_query_text(query_raw)
        if not query_norm:
            continue
        bad_answer = str(doc.get("bad_answer") or "").strip()
        if not bad_answer:
            continue
        reason = str(doc.get("reason") or "").strip() or "Migrated from legacy bad QA memory"
        query_tokens = doc.get("query_tokens")
        if not isinstance(query_tokens, list):
            query_tokens = _extract_query_tokens(query_norm)
        timestamp = _coerce_timestamp(doc.get("timestamp"))

        ops.append(
            UpdateOne(
                {"query_norm": query_norm, "rating": "negative"},
                {
                    "$set": {
                        "user_message": query_raw or query_norm,
                        "ai_response": bad_answer,
                        "rating": "negative",
                        "reward": -1.0,
                        "query_norm": query_norm,
                        "query_tokens": query_tokens,
                        "session_id": "legacy_qa_migration",
                        "timestamp": timestamp,
                        "comment": reason,
                        "policy_tags": ["legacy_badqa"],
                        "memory_source": "legacy_badqa",
                    }
                },
                upsert=True,
            )
        )

    if not ops:
        print("No legacy QA documents found to migrate.")
        return 0

    result = feedback_coll.bulk_write(ops, ordered=False)
    print(
        "Migration complete:",
        f"matched={result.matched_count}",
        f"modified={result.modified_count}",
        f"upserted={len(result.upserted_ids or {})}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

