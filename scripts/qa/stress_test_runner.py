import concurrent.futures
import csv
import json
import os
import re
import statistics
import time
from pathlib import Path

import requests

API_URL = os.getenv("QA_API_URL", "http://localhost:5000/api/chat")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_FILE = Path(__file__).with_name("stress_test_questions_300.json")
REPORT_FILE = PROJECT_ROOT / "data" / "reports" / "stress_test_report_latest.csv"
NO_DATA_PHRASES = [
    "cannot find",
    "could not find",
    "not in our database",
    "unavailable in the verified",
    "database has no information",
    "db에 없습니다",
]
AUTH_PROMPT_WORDS = ["please login", "security check", "password", "verify", "authenticate"]


def contains_phrase(text: str, phrase: str) -> bool:
    p = (phrase or "").strip().lower()
    if not p:
        return False
    # Boundary-aware matching for short tokens like "rm".
    if re.fullmatch(r"[a-z0-9]{1,3}", p):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", text))
    return p in text


def parse_response_text(resp_json):
    raw = resp_json.get("response", "")
    if isinstance(raw, dict):
        return str(raw.get("text", "")).strip()
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                return str(json.loads(s).get("text", s)).strip()
            except Exception:
                return s
        return s
    return str(raw).strip()


def semantic_eval(q_type, query, response, status):
    ql = str(query or "").lower()
    rl = str(response or "").lower()
    failures = []
    score = 1.0

    def fail(reason, penalty=0.25):
        nonlocal score
        failures.append(reason)
        score = max(0.0, score - penalty)

    if str(status) != "200":
        fail("http_status_not_200", 1.0)
        return False, 0.0, failures

    if not rl.strip():
        fail("empty_response", 1.0)
        return False, 0.0, failures

    if any(t in rl for t in ["traceback", "exception", "internal server error"]):
        fail("server_error_leak", 1.0)

    no_data = any(contains_phrase(rl, k) for k in NO_DATA_PHRASES)

    if q_type == "Personal":
        if not any(contains_phrase(rl, k) for k in AUTH_PROMPT_WORDS):
            fail("personal_missing_auth_gate", 0.7)

    elif q_type == "Hallucination_Trap":
        if not no_data:
            fail("hallucination_trap_missing_refusal", 0.7)
        if any(contains_phrase(rl, k) for k in ["rm", "fee", "tuition", "price", "cost"]):
            fail("hallucination_trap_invented_fee", 0.4)

    elif q_type == "RAG":
        # Domain no-data is acceptable when DB genuinely has no matching rows.
        if not no_data and any(k in ql for k in ["library", "block", "hostel", "gym", "bus", "faculty", "head"]) and not any(
            k in rl for k in ["library", "block", "hostel", "gym", "bus", "faculty", "dean", "director", "head"]
        ):
            fail("rag_missing_domain_anchor", 0.4)
        if any(contains_phrase(rl, k) for k in AUTH_PROMPT_WORDS):
            fail("rag_wrong_auth_gate", 0.3)

    elif q_type == "General":
        if any(contains_phrase(rl, k) for k in AUTH_PROMPT_WORDS):
            fail("general_wrong_auth_gate", 0.5)
        if any(contains_phrase(rl, k) for k in ["student_number", "programme_code", "profile_status"]):
            fail("general_leaked_personal_fields", 0.6)

    elif q_type == "Mixed":
        # Mixed query has personal component in this test set.
        if not any(contains_phrase(rl, k) for k in AUTH_PROMPT_WORDS):
            fail("mixed_missing_auth_gate", 0.6)

    elif q_type == "Korean":
        if "성적" in ql and not any(contains_phrase(rl, k) for k in AUTH_PROMPT_WORDS):
            fail("korean_personal_missing_auth_gate", 0.7)

    elif q_type == "Edge":
        if len(response) > 420:
            fail("edge_too_long", 0.3)

    semantic_pass = score >= 0.7 and len(failures) == 0
    return semantic_pass, round(score, 3), failures


def run_query(idx, question_data):
    q_type = question_data["type"]
    query = question_data["query"]

    start_t = time.time()
    try:
        resp = requests.post(API_URL, json={"message": query}, timeout=30)
        elapsed = time.time() - start_t
        status = resp.status_code

        try:
            payload = resp.json()
            content = parse_response_text(payload)
        except Exception:
            content = "Parse Error"

        semantic_pass, semantic_score, semantic_failures = semantic_eval(q_type, query, content, status)

        return {
            "id": idx,
            "type": q_type,
            "query": query,
            "status": status,
            "latency": round(elapsed, 4),
            "semantic_pass": semantic_pass,
            "semantic_score": semantic_score,
            "semantic_failures": "|".join(semantic_failures),
            "response_snippet": re.sub(r"\s+", " ", content)[:180],
        }

    except Exception as e:
        elapsed = time.time() - start_t
        return {
            "id": idx,
            "type": q_type,
            "query": query,
            "status": "ERROR",
            "latency": round(elapsed, 4),
            "semantic_pass": False,
            "semantic_score": 0.0,
            "semantic_failures": "request_error",
            "response_snippet": str(e)[:180],
        }


def print_summary(results):
    total = len(results)
    ok = sum(1 for r in results if str(r["status"]) == "200")
    semantic_ok = sum(1 for r in results if bool(r["semantic_pass"]))
    latencies = [float(r["latency"]) for r in results if str(r["status"]) == "200"]

    print("=== Stress Summary ===")
    print(f"Total: {total}")
    print(f"HTTP 200: {ok}/{total} ({(ok / total * 100):.1f}%)")
    print(f"Semantic Pass: {semantic_ok}/{total} ({(semantic_ok / total * 100):.1f}%)")
    if latencies:
        lat_sorted = sorted(latencies)
        p50 = lat_sorted[int((len(lat_sorted) - 1) * 0.5)]
        p95 = lat_sorted[int((len(lat_sorted) - 1) * 0.95)]
        print(f"Latency avg/p50/p95/max: {statistics.mean(latencies):.3f}s / {p50:.3f}s / {p95:.3f}s / {max(latencies):.3f}s")

    by_type = {}
    for row in results:
        t = row["type"]
        by_type.setdefault(t, {"total": 0, "ok": 0, "semantic_ok": 0})
        by_type[t]["total"] += 1
        if str(row["status"]) == "200":
            by_type[t]["ok"] += 1
        if bool(row["semantic_pass"]):
            by_type[t]["semantic_ok"] += 1

    print("--- By Type ---")
    for t in sorted(by_type):
        s = by_type[t]
        print(f"{t}: http={s['ok']}/{s['total']} semantic={s['semantic_ok']}/{s['total']}")


def main():
    try:
        with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
            questions = json.load(f)
            if len(questions) < 300:
                original_len = len(questions)
                factor = (300 // original_len) + 1
                questions = (questions * factor)[:300]
                print(f"Expanded {original_len} questions to {len(questions)} queries.")
    except FileNotFoundError:
        print(f"Questions file not found: {QUESTIONS_FILE}")
        return

    print(f"Starting Stress Test for {len(questions)} questions...")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(run_query, i, q): i for i, q in enumerate(questions)}
        processed = 0
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)
            processed += 1
            if processed % 10 == 0:
                print(f"Processed {processed}/{len(questions)}")

    results.sort(key=lambda x: x["id"])

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "id",
            "type",
            "query",
            "status",
            "latency",
            "semantic_pass",
            "semantic_score",
            "semantic_failures",
            "response_snippet",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print_summary(results)
    print(f"Test Complete. Results saved to {REPORT_FILE}")


if __name__ == "__main__":
    main()
