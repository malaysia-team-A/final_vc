import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - runtime dependency guard
    SentenceTransformer = None


SEMANTIC_COLLECTION = "semantic_intents"
UNANSWERED_COLLECTION = "unanswered"
FEEDBACK_COLLECTION = "Feedback"
THRESHOLD_KEY = "SEMANTIC_ROUTER_MIN_CONFIDENCE"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _normalize_key(text: Any) -> str:
    s = _normalize_space(text).lower()
    s = re.sub(r"[\t\r\n]+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    return any(n in text for n in needles)


def _label_query(query: str) -> Optional[str]:
    """
    Heuristic intent label for tuning set construction.
    Domain-first policy: if a query has UCSI/domain hints, route to UCSI intent
    even for potentially nonexistent entities, so downstream RAG can return no-data.
    """
    q = _normalize_space(query).lower()
    if not q:
        return None

    if _contains_any(q, ["neg_test", "neg_direct"]):
        return "unknown"

    if _contains_any(q, ["who are you", "what can you do", "can you dance", "can you sing", "test q"]):
        return "capability_smalltalk"

    if _contains_any(q, ["hostel", "accommodation", "deposit", "dorm", "room"]):
        return "ucsi_hostel"

    if _contains_any(q, ["schedule", "semester", "intake", "calendar", "route ", "exam"]):
        return "ucsi_schedule"

    if _contains_any(
        q,
        [
            "library",
            "print",
            "printer",
            "cafeteria",
            "gym",
            "facility",
            "facilities",
            "block",
            "building",
            "campus map",
            "where is block",
        ],
    ):
        return "ucsi_facility"

    if _contains_any(q, ["professor", "lecturer", "dean", "staff", "advisor"]):
        return "ucsi_staff"

    if _contains_any(q, ["tuition", "fee", "fees", "installment", "programme", "program", "course", "major"]):
        return "ucsi_programme"

    if q.startswith("who is ") or q.startswith("tell me about ") or q.startswith("what do you know about "):
        return "general_person"

    if _contains_any(q, ["capital of", "what is machine learning", "explain python"]):
        return "general_world"

    return None


def _load_db() -> Tuple[Any, str]:
    root = _project_root()
    load_dotenv(dotenv_path=root / ".env")
    uri = str(os.getenv("MONGO_URI") or "").strip()
    if not uri:
        raise RuntimeError("MONGO_URI not found in .env")
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    db_name = uri.split("/")[-1].split("?")[0] or "UCSI_DB"
    return client[db_name], db_name


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _aggregate_score(sims: List[float]) -> float:
    if not sims:
        return -1.0
    mx = max(sims)
    avg = sum(sims) / float(len(sims))
    mixed = (mx * 0.75) + (avg * 0.25)
    return float(max(min(mixed, 1.0), -1.0))


def _score_query(
    query_vec: np.ndarray,
    vectors_by_intent: Dict[str, List[np.ndarray]],
) -> Tuple[str, float]:
    scores: Dict[str, float] = {}
    for intent, vecs in vectors_by_intent.items():
        sims = [max(min(_cosine(query_vec, v), 1.0), -1.0) for v in vecs]
        if not sims:
            continue
        scores[intent] = _aggregate_score(sims)
    if not scores:
        return "unknown", 0.0
    best_intent, best_score = max(scores.items(), key=lambda kv: kv[1])
    return best_intent, float(best_score)


def _load_seed_vectors(db: Any, collection_name: str) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, set], int]:
    rows = list(db[collection_name].find({}, {"_id": 0, "intent": 1, "example": 1, "embedding": 1}))
    vectors_by_intent: Dict[str, List[np.ndarray]] = defaultdict(list)
    examples_by_intent: Dict[str, set] = defaultdict(set)
    valid_rows = 0
    for row in rows:
        intent = str(row.get("intent") or "").strip()
        emb = row.get("embedding")
        example = _normalize_space(row.get("example"))
        if not intent:
            continue
        if example:
            examples_by_intent[intent].add(_normalize_key(example))
        if isinstance(emb, list) and emb:
            vectors_by_intent[intent].append(np.asarray(emb, dtype=np.float32))
            valid_rows += 1
    return vectors_by_intent, examples_by_intent, valid_rows


def _collect_queries(db: Any, unanswered_limit: int, feedback_limit: int) -> Dict[str, Dict[str, Any]]:
    bucket: Dict[str, Dict[str, Any]] = {}

    unanswered = list(
        db[UNANSWERED_COLLECTION]
        .find({}, {"_id": 0, "question": 1, "search_query": 1, "timestamp": 1})
        .sort("timestamp", -1)
        .limit(max(1, int(unanswered_limit)))
    )
    for row in unanswered:
        text = _normalize_space(row.get("question") or row.get("search_query"))
        if not text:
            continue
        key = _normalize_key(text)
        if not key:
            continue
        slot = bucket.setdefault(
            key,
            {
                "query": text,
                "count": 0,
                "sources": set(),
            },
        )
        slot["count"] += 1
        slot["sources"].add("unanswered")

    feedback = list(
        db[FEEDBACK_COLLECTION]
        .find({}, {"_id": 0, "user_message": 1, "query": 1, "timestamp": 1, "rating": 1})
        .sort("timestamp", -1)
        .limit(max(1, int(feedback_limit)))
    )
    for row in feedback:
        text = _normalize_space(row.get("user_message") or row.get("query"))
        if not text:
            continue
        key = _normalize_key(text)
        if not key:
            continue
        slot = bucket.setdefault(
            key,
            {
                "query": text,
                "count": 0,
                "sources": set(),
            },
        )
        slot["count"] += 1
        slot["sources"].add("feedback")

    return bucket


def _evaluate_threshold(
    records: List[Dict[str, Any]],
    threshold: float,
) -> Dict[str, float]:
    # known = labels except unknown
    tp = fp = fn = tn = 0
    exact_correct = 0
    for rec in records:
        truth = str(rec["label"])
        pred_intent = str(rec["pred_intent"])
        score = float(rec["pred_score"])
        pred = "unknown" if score < threshold else pred_intent

        if pred == truth:
            exact_correct += 1

        truth_known = truth != "unknown"
        pred_known = pred != "unknown"
        if truth_known and pred_known and pred == truth:
            tp += 1
        elif (not truth_known) and (not pred_known):
            tn += 1
        elif (not truth_known) and pred_known:
            fp += 1
        elif truth_known and (not pred_known):
            fn += 1
        else:
            # wrong known intent
            fp += 1
            fn += 1

    total = max(1, len(records))
    accuracy = exact_correct / float(total)
    precision = tp / float(max(1, tp + fp))
    recall = tp / float(max(1, tp + fn))
    f1 = 0.0 if (precision + recall) <= 0 else (2.0 * precision * recall / (precision + recall))

    return {
        "accuracy": round(accuracy, 6),
        "known_precision": round(precision, 6),
        "known_recall": round(recall, 6),
        "known_f1": round(f1, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total,
    }


def _suggest_threshold(
    records: List[Dict[str, Any]],
    threshold_min: float,
    threshold_max: float,
    step: float,
) -> Tuple[float, Dict[str, float], List[Dict[str, Any]]]:
    if not records:
        return 0.53, {}, []

    cur = threshold_min
    rows: List[Dict[str, Any]] = []
    while cur <= threshold_max + 1e-9:
        th = round(float(cur), 4)
        m = _evaluate_threshold(records, th)
        rows.append({"threshold": th, **m})
        cur += step

    best = max(rows, key=lambda r: (r["known_f1"], r["accuracy"], -abs(r["threshold"] - 0.5)))
    return float(best["threshold"]), best, rows


def _build_seed_candidates(
    records: List[Dict[str, Any]],
    existing_examples: Dict[str, set],
    current_threshold: float,
    suggested_threshold: float,
    min_frequency: int,
    max_per_intent: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for rec in records:
        label = str(rec["label"])
        if not label or label == "unknown":
            continue
        freq = int(rec.get("count") or 1)
        if freq < min_frequency:
            continue

        query = _normalize_space(rec.get("query"))
        query_key = _normalize_key(query)
        if not query_key:
            continue
        if query_key in existing_examples.get(label, set()):
            continue

        pred_intent = str(rec.get("pred_intent") or "")
        score = float(rec.get("pred_score") or 0.0)

        needs_seed = (
            pred_intent != label
            or score < float(current_threshold)
            or score < float(suggested_threshold)
        )
        if not needs_seed:
            continue

        grouped[label].append(
            {
                "intent": label,
                "example": query,
                "query_key": query_key,
                "count": freq,
                "pred_intent": pred_intent,
                "pred_score": round(score, 6),
            }
        )

    selected: List[Dict[str, Any]] = []
    for intent, rows in grouped.items():
        rows.sort(key=lambda r: (-int(r["count"]), float(r["pred_score"]), r["example"]))
        picked = rows[: max(1, int(max_per_intent))]
        selected.extend(picked)
    return selected


def _upsert_seed_candidates(
    db: Any,
    collection_name: str,
    model: Any,
    candidates: List[Dict[str, Any]],
) -> Dict[str, int]:
    if not candidates:
        return {"inserted": 0, "skipped": 0}

    coll = db[collection_name]
    stamp = datetime.now(timezone.utc).isoformat()
    source_tag = f"log_tune_{datetime.now(timezone.utc).strftime('%Y%m%d')}"

    inserted = 0
    skipped = 0
    for cand in candidates:
        intent = str(cand["intent"]).strip()
        example = _normalize_space(cand["example"])
        if not intent or not example:
            skipped += 1
            continue

        exists = coll.find_one({"intent": intent, "example": example}, {"_id": 1})
        if exists:
            skipped += 1
            continue

        vec = model.encode([example])[0]
        coll.insert_one(
            {
                "intent": intent,
                "example": example,
                "embedding": np.asarray(vec, dtype=np.float32).tolist(),
                "source": source_tag,
                "created_at": stamp,
            }
        )
        inserted += 1

    return {"inserted": inserted, "skipped": skipped}


def _update_env_threshold(env_file: Path, threshold: float) -> bool:
    if not env_file.exists():
        return False
    text = env_file.read_text(encoding="utf-8")
    line = f"{THRESHOLD_KEY}={threshold:.2f}"
    if re.search(rf"^{THRESHOLD_KEY}\s*=\s*.*$", text, flags=re.MULTILINE):
        new_text = re.sub(rf"^{THRESHOLD_KEY}\s*=\s*.*$", line, text, flags=re.MULTILINE)
    else:
        suffix = "" if text.endswith("\n") else "\n"
        new_text = f"{text}{suffix}{line}\n"
    if new_text == text:
        return False
    env_file.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune semantic router threshold and seed examples from Feedback/unanswered logs."
    )
    parser.add_argument("--unanswered-limit", type=int, default=400)
    parser.add_argument("--feedback-limit", type=int, default=400)
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument("--max-candidates-per-intent", type=int, default=4)
    parser.add_argument("--threshold-min", type=float, default=0.35)
    parser.add_argument("--threshold-max", type=float, default=0.75)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--apply-seeds", action="store_true", help="Insert suggested seed examples into semantic_intents")
    parser.add_argument("--apply-threshold", type=float, default=None, help="Set explicit threshold value into env files")
    parser.add_argument(
        "--env-files",
        nargs="*",
        default=[".env.example"],
        help="Env files to update when --apply-threshold is used",
    )
    args = parser.parse_args()

    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is required for semantic tuning")

    root = _project_root()
    db, db_name = _load_db()

    model_name = os.getenv("RAG_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    collection_name = os.getenv("SEMANTIC_ROUTER_COLLECTION", SEMANTIC_COLLECTION)
    current_threshold = float(os.getenv(THRESHOLD_KEY, "0.53"))
    model = SentenceTransformer(model_name)

    vectors_by_intent, existing_examples, valid_seed_rows = _load_seed_vectors(db, collection_name)
    if not vectors_by_intent:
        raise RuntimeError(f"No valid embeddings found in collection '{collection_name}'")

    query_pool = _collect_queries(db, unanswered_limit=args.unanswered_limit, feedback_limit=args.feedback_limit)
    labeled_queries: List[Dict[str, Any]] = []
    for key, row in query_pool.items():
        label = _label_query(row["query"])
        if not label:
            continue
        labeled_queries.append(
            {
                "query_key": key,
                "query": row["query"],
                "count": int(row["count"]),
                "sources": sorted(list(row.get("sources") or [])),
                "label": label,
            }
        )

    query_texts = [r["query"] for r in labeled_queries]
    embeddings = model.encode(query_texts) if query_texts else []

    records: List[Dict[str, Any]] = []
    for row, emb in zip(labeled_queries, embeddings):
        vec = np.asarray(emb, dtype=np.float32)
        pred_intent, pred_score = _score_query(vec, vectors_by_intent)
        records.append(
            {
                **row,
                "pred_intent": pred_intent,
                "pred_score": round(float(pred_score), 6),
            }
        )

    suggested_threshold, suggested_metrics, threshold_rows = _suggest_threshold(
        records,
        threshold_min=float(args.threshold_min),
        threshold_max=float(args.threshold_max),
        step=float(args.threshold_step),
    )

    current_metrics = _evaluate_threshold(records, current_threshold)

    candidates = _build_seed_candidates(
        records,
        existing_examples=existing_examples,
        current_threshold=current_threshold,
        suggested_threshold=suggested_threshold,
        min_frequency=max(1, int(args.min_frequency)),
        max_per_intent=max(1, int(args.max_candidates_per_intent)),
    )

    upsert_result = {"inserted": 0, "skipped": 0}
    if args.apply_seeds and candidates:
        upsert_result = _upsert_seed_candidates(
            db,
            collection_name=collection_name,
            model=model,
            candidates=candidates,
        )

    env_updates: List[str] = []
    if args.apply_threshold is not None:
        threshold_to_apply = float(args.apply_threshold)
        for rel in args.env_files:
            target = (root / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
            changed = _update_env_threshold(target, threshold_to_apply)
            if changed:
                env_updates.append(str(target))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_dir = root / "data" / "reference"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"semantic_router_tuning_{stamp}.json"

    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": db_name,
        "collection": collection_name,
        "model": model_name,
        "current_threshold": round(current_threshold, 6),
        "suggested_threshold": round(suggested_threshold, 6),
        "current_metrics": current_metrics,
        "suggested_metrics": suggested_metrics,
        "records_count": len(records),
        "seed_rows_valid": valid_seed_rows,
        "candidates_count": len(candidates),
        "candidates": candidates,
        "upsert": upsert_result,
        "env_updates": env_updates,
        "threshold_sweep": threshold_rows,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] semantic seed rows (valid): {valid_seed_rows}")
    print(f"[INFO] labeled records: {len(records)}")
    print(f"[INFO] current {THRESHOLD_KEY}: {current_threshold:.2f}")
    print(
        "[INFO] current metrics:",
        f"acc={current_metrics['accuracy']:.4f}",
        f"known_f1={current_metrics['known_f1']:.4f}",
    )
    print(
        "[INFO] suggested metrics:",
        f"threshold={suggested_threshold:.2f}",
        f"acc={suggested_metrics.get('accuracy', 0.0):.4f}",
        f"known_f1={suggested_metrics.get('known_f1', 0.0):.4f}",
    )
    print(f"[INFO] candidate seeds: {len(candidates)}")
    if candidates:
        for row in candidates[:20]:
            print(
                "  -",
                f"{row['intent']} | {row['example']}",
                f"(count={row['count']}, pred={row['pred_intent']}, score={row['pred_score']:.3f})",
            )
    if args.apply_seeds:
        print(
            "[APPLY] seed upsert:",
            f"inserted={upsert_result['inserted']}",
            f"skipped={upsert_result['skipped']}",
        )
    if env_updates:
        print(f"[APPLY] threshold updated in {len(env_updates)} file(s):")
        for p in env_updates:
            print(f"  - {p}")
    print(f"[INFO] report written: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

