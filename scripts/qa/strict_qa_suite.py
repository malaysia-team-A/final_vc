"""
Strict QA Suite for chatbot intent alignment and security behavior.

Focus:
1) Intent alignment (general vs RAG vs personal)
2) Security gates (guest/login/2FA-required)
3) Grounded response hygiene (no leakage/hallucination markers)
4) UX constraints (concise/plain output)
"""

import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Tuple

import requests


API_BASE = os.getenv("QA_API_BASE", "http://localhost:8000")
CHAT_API = f"{API_BASE}/api/chat"
LOGIN_API = f"{API_BASE}/api/login"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEST_CASES_DIR = os.path.join(PROJECT_ROOT, "test_cases")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "reports")


def _parse_response_payload(api_json: Dict) -> Tuple[str, Dict, str]:
    """
    Returns:
      - text: final user-visible response text
      - nested: inner response payload if JSON-string exists
      - outer_type: top-level API type (message/login_hint/password_prompt)
    """
    outer_type = str(api_json.get("type", "message"))
    raw = api_json.get("response", "")
    nested = {}
    text = ""

    if isinstance(raw, dict):
        nested = raw
        text = str(raw.get("text", ""))
    elif isinstance(raw, str):
        candidate = raw.strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            try:
                nested = json.loads(candidate)
                text = str(nested.get("text", candidate))
            except Exception:
                text = raw
        else:
            text = raw
    else:
        text = str(raw)

    return text.strip(), nested, outer_type


class StrictQASuite:
    def __init__(
        self,
        mode: str = "full",
        workers: int = 4,
        timeout: int = 30,
        limit: int = None,
        verbose: bool = False,
        student_number: str = None,
        student_name: str = None,
    ):
        self.mode = mode
        self.workers = workers
        self.timeout = timeout
        self.limit = limit
        self.verbose = verbose
        self.student_number = student_number or os.getenv("QA_TEST_STUDENT_NUMBER", "5004273609")
        self.student_name = student_name or os.getenv("QA_TEST_NAME", "Gomana Wern Rou")

        self.session = requests.Session()
        self.tests: List[Dict] = []
        self.results: List[Dict] = []
        self.auth_token = None

        self.stats = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "error": 0,
            "by_scope": defaultdict(lambda: {"passed": 0, "failed": 0}),
            "by_type": defaultdict(lambda: {"passed": 0, "failed": 0}),
        }

    def load_tests(self) -> List[Dict]:
        tests = []

        include_rag = self.mode in ("full", "base", "rag")
        include_general = self.mode in ("full", "base", "general")
        include_auth = self.mode in ("full", "auth")

        if include_rag:
            path = os.path.join(TEST_CASES_DIR, "rag_accuracy_tests.json")
            if os.path.exists(path):
                rag = json.loads(open(path, "r", encoding="utf-8").read())
                for t in rag:
                    t = dict(t)
                    t["suite"] = "base"
                    t["scope"] = "guest"
                    t["source"] = "rag_accuracy"
                    tests.append(t)
                print(f"[LOADED] {len(rag)} strict RAG base tests")

            path_real = os.path.join(TEST_CASES_DIR, "rag_real_world_tests.json")
            if os.path.exists(path_real):
                rag_real = json.loads(open(path_real, "r", encoding="utf-8").read())
                for t in rag_real:
                    t = dict(t)
                    t["suite"] = "base"
                    t["scope"] = "guest"
                    t["source"] = "rag_real_world"
                    tests.append(t)
                print(f"[LOADED] {len(rag_real)} strict RAG real-world tests")

        if include_general:
            path = os.path.join(TEST_CASES_DIR, "general_knowledge_tests.json")
            if os.path.exists(path):
                gk = json.loads(open(path, "r", encoding="utf-8").read())
                for t in gk:
                    t = dict(t)
                    t["suite"] = "base"
                    t["scope"] = "guest"
                    t["source"] = "general_knowledge"
                    tests.append(t)
                print(f"[LOADED] {len(gk)} strict general base tests")

        if include_auth:
            path = os.path.join(TEST_CASES_DIR, "strict_auth_intent_tests.json")
            if os.path.exists(path):
                auth = json.loads(open(path, "r", encoding="utf-8").read())
                for t in auth:
                    t = dict(t)
                    t["suite"] = "auth"
                    t["source"] = "strict_auth_intent"
                    tests.append(t)
                print(f"[LOADED] {len(auth)} strict auth/intent tests")

        if self.limit:
            tests = tests[: self.limit]
            print(f"[LIMIT] Using first {len(tests)} tests")

        self.tests = tests
        return tests

    def login_for_auth(self) -> bool:
        if self.mode not in ("full", "auth"):
            return True
        payload = {"student_number": self.student_number, "name": self.student_name}
        try:
            resp = self.session.post(LOGIN_API, json=payload, timeout=self.timeout)
            if resp.status_code != 200:
                print(f"[AUTH] Login failed: HTTP {resp.status_code}")
                return False
            body = resp.json()
            if not body.get("success") or not body.get("token"):
                print(f"[AUTH] Login failed: {body}")
                return False
            self.auth_token = body["token"]
            print("[AUTH] Login succeeded for strict auth tests")
            return True
        except Exception as e:
            print(f"[AUTH] Login error: {e}")
            return False

    def _chat(self, query: str, token: str = None) -> Dict:
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        start = time.time()
        try:
            r = self.session.post(CHAT_API, json={"message": query}, headers=headers, timeout=self.timeout)
            latency = round(time.time() - start, 3)
            if r.status_code != 200:
                return {
                    "ok": False,
                    "status": f"HTTP_{r.status_code}",
                    "latency": latency,
                    "text": r.text[:300],
                    "outer_type": "error",
                    "raw_json": {},
                }
            data = r.json()
            text, nested, outer_type = _parse_response_payload(data)
            return {
                "ok": True,
                "status": "OK",
                "latency": latency,
                "text": text,
                "nested": nested,
                "outer_type": outer_type,
                "raw_json": data,
            }
        except requests.Timeout:
            return {
                "ok": False,
                "status": "TIMEOUT",
                "latency": round(time.time() - start, 3),
                "text": "timeout",
                "outer_type": "error",
                "raw_json": {},
            }
        except Exception as e:
            return {
                "ok": False,
                "status": "ERROR",
                "latency": round(time.time() - start, 3),
                "text": str(e),
                "outer_type": "error",
                "raw_json": {},
            }

    def _evaluate_base(self, test: Dict, response_text: str, outer_type: str) -> Tuple[bool, List[str]]:
        q = str(test.get("query", ""))
        ql = q.lower()
        r = str(response_text or "")
        rl = r.lower()
        t = str(test.get("type", ""))

        failures = []
        expected = [str(x).lower() for x in test.get("expected_keywords", [])]
        forbidden = [str(x).lower() for x in test.get("must_not_contain", [])]
        allow_no_data = bool(test.get("allow_no_data", False))

        def fail_if(cond: bool, reason: str):
            if cond:
                failures.append(reason)

        # Core hygiene
        fail_if(not r.strip(), "empty_response")
        fail_if(any(m in rl for m in ["[document]", "[no_relevant_data_found]", "mongodb:"]), "internal_marker_leak")
        fail_if(bool(re.search(r"(^|\s)(\*\*|__|`|##)", r)), "markdown_format_leak")
        fail_if(outer_type not in {"message", "login_hint", "password_prompt"}, f"unexpected_type:{outer_type}")

        def contains_phrase(text: str, phrase: str) -> bool:
            """Use boundary-aware matching for short alnum tokens (e.g., RM)."""
            p = (phrase or "").strip().lower()
            if not p:
                return False
            if re.fullmatch(r"[a-z0-9]{1,3}", p):
                return bool(re.search(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", text))
            return p in text

        no_data_phrases = [
            "cannot find", "could not find", "not in our database", "i don't have that information",
            "unavailable in the verified", "database has no information", "db에 없습니다", "데이터베이스에 없습니다"
        ]
        no_data_hit = any(contains_phrase(rl, p) for p in no_data_phrases)
        busy_phrases = [
            "temporarily busy", "try again in", "handling many requests", "having trouble processing"
        ]
        auth_prompt_words = ["please login", "security check", "password", "authenticate", "verify"]

        # Expected / forbidden keywords
        if expected:
            if not any(contains_phrase(rl, k) for k in expected):
                failures.append(f"missing_expected_any:{expected}")
        if forbidden:
            found_forbidden = [k for k in forbidden if contains_phrase(rl, k)]
            if found_forbidden:
                failures.append(f"contains_forbidden:{found_forbidden}")

        if t == "rag_exact":
            fail_if(no_data_hit and not allow_no_data, "rag_exact_returned_no_data")
            fail_if(any(p in rl for p in busy_phrases), "rag_exact_service_busy")
            fail_if(len(r) > 420, f"rag_exact_too_long:{len(r)}")

        if t in {"rag_negative", "safeguard_hallucination"}:
            fail_if(not no_data_hit, "negative_test_missing_refusal")
            # If refusal/no-data policy was satisfied, keyword style differences are acceptable.
            if no_data_hit:
                failures = [f for f in failures if not f.startswith("missing_expected_any")]
            fail_if(len(r) > 260, f"negative_response_too_long:{len(r)}")

        if t in {"factual", "mathematical", "creative", "practical"}:
            # General knowledge should not ask auth or leak university DB framing.
            fail_if(any(w in rl for w in auth_prompt_words), "general_wrong_auth_gate")
            if not any(k in ql for k in ["ucsi", "campus", "hostel", "programme", "program", "faculty", "lecturer", "student"]):
                leak_words = ["ucsi", "hostel", "block ", "faculty", "student_number", "programme_code"]
                fail_if(any(w in rl for w in leak_words), "general_domain_leak")
            fail_if(len(r) > 380, f"general_too_long:{len(r)}")

        return (len(failures) == 0), failures

    def _evaluate_auth(self, test: Dict, response_text: str, outer_type: str) -> Tuple[bool, List[str]]:
        r = str(response_text or "")
        rl = r.lower()
        failures = []

        expect_type = test.get("expect_type")
        require_any = [str(x).lower() for x in test.get("require_any", [])]
        require_all = [str(x).lower() for x in test.get("require_all", [])]
        forbid_any = [str(x).lower() for x in test.get("forbid_any", [])]
        max_chars = int(test.get("max_chars", 0) or 0)

        if expect_type and outer_type != expect_type:
            failures.append(f"type_mismatch:expected={expect_type},actual={outer_type}")

        if require_any and not any(k in rl for k in require_any):
            failures.append(f"missing_require_any:{require_any}")

        if require_all:
            missing = [k for k in require_all if k not in rl]
            if missing:
                failures.append(f"missing_require_all:{missing}")

        if forbid_any:
            found = [k for k in forbid_any if k in rl]
            if found:
                failures.append(f"contains_forbidden:{found}")

        if max_chars and len(r) > max_chars:
            failures.append(f"too_long:{len(r)}>{max_chars}")

        if test.get("type") == "auth_guard":
            # Guard responses must not leak concrete personal fields.
            leaks = ["student_number:", "nationality:", "gpa:", "cgpa:", "programme_code:"]
            found = [k for k in leaks if k in rl]
            if found:
                failures.append(f"guard_leak:{found}")

        return (len(failures) == 0), failures

    def _run_one(self, test: Dict) -> Dict:
        scope = test.get("scope", "guest")
        token = self.auth_token if scope in ("auth", "auth2fa") else None
        query = str(test.get("query", ""))

        if scope in ("auth", "auth2fa") and not token:
            return {
                "id": test.get("id"),
                "scope": scope,
                "type": test.get("type", "unknown"),
                "source": test.get("source", "unknown"),
                "query": query,
                "status": "ERROR",
                "latency": 0.0,
                "outer_type": "error",
                "passed": False,
                "reason": "missing_auth_token",
                "failed_checks": "missing_auth_token",
                "response": "",
            }

        chat = self._chat(query, token=token)
        if not chat["ok"]:
            return {
                "id": test.get("id"),
                "scope": scope,
                "type": test.get("type", "unknown"),
                "source": test.get("source", "unknown"),
                "query": query,
                "status": chat["status"],
                "latency": chat["latency"],
                "outer_type": chat["outer_type"],
                "passed": False,
                "reason": "request_error",
                "failed_checks": chat["status"],
                "response": chat["text"][:220],
            }

        if test.get("suite") == "auth":
            passed, failed_checks = self._evaluate_auth(test, chat["text"], chat["outer_type"])
        else:
            passed, failed_checks = self._evaluate_base(test, chat["text"], chat["outer_type"])

        reason = "ok" if passed else "; ".join(failed_checks)
        return {
            "id": test.get("id"),
            "scope": scope,
            "type": test.get("type", "unknown"),
            "source": test.get("source", "unknown"),
            "query": query,
            "status": "OK",
            "latency": chat["latency"],
            "outer_type": chat["outer_type"],
            "passed": passed,
            "reason": reason,
            "failed_checks": "|".join(failed_checks),
            "response": chat["text"][:220],
        }

    def run(self) -> List[Dict]:
        if not self.tests:
            self.load_tests()
        if not self.tests:
            print("[ERROR] No tests loaded.")
            return []

        if self.mode in ("full", "auth"):
            self.login_for_auth()

        print("\n============================================================")
        print("Starting Strict QA Suite")
        print(f"Mode: {self.mode}")
        print(f"Total tests: {len(self.tests)}")
        print(f"Workers: {self.workers}")
        print("Checks: intent route + auth gate + grounding hygiene + UX length")
        print("============================================================\n")

        auth_tests = [t for t in self.tests if t.get("suite") == "auth"]
        base_tests = [t for t in self.tests if t.get("suite") != "auth"]

        completed = 0
        total = len(self.tests)
        results = []

        # Run base tests concurrently.
        if base_tests:
            with ThreadPoolExecutor(max_workers=max(1, self.workers)) as executor:
                futures = {executor.submit(self._run_one, t): t for t in base_tests}
                for f in as_completed(futures):
                    res = f.result()
                    results.append(res)
                    completed += 1
                    if completed % 5 == 0 or completed == total:
                        print(f"[{completed}/{total}] progress")
                    if self.verbose and not res["passed"]:
                        print(f"  FAIL {res['id']}: {res['reason']}")

        # Run auth tests sequentially to preserve state clarity.
        for t in auth_tests:
            res = self._run_one(t)
            results.append(res)
            completed += 1
            if completed % 5 == 0 or completed == total:
                print(f"[{completed}/{total}] progress")
            if self.verbose and not res["passed"]:
                print(f"  FAIL {res['id']}: {res['reason']}")

        self.results = results
        self._finalize_stats()
        return results

    def _finalize_stats(self):
        for r in self.results:
            self.stats["total"] += 1
            scope = r["scope"]
            typ = r["type"]
            if r["status"] not in ("OK",):
                self.stats["error"] += 1
            if r["passed"]:
                self.stats["passed"] += 1
                self.stats["by_scope"][scope]["passed"] += 1
                self.stats["by_type"][typ]["passed"] += 1
            else:
                self.stats["failed"] += 1
                self.stats["by_scope"][scope]["failed"] += 1
                self.stats["by_type"][typ]["failed"] += 1

    def save_report(self) -> str:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = os.path.join(RESULTS_DIR, f"strict_qa_report_{ts}.csv")

        fields = [
            "id", "scope", "type", "source", "query", "status",
            "latency", "outer_type", "passed", "reason", "failed_checks", "response"
        ]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(self.results)

        pass_rate = (self.stats["passed"] / self.stats["total"] * 100) if self.stats["total"] else 0.0
        print("\n============================================================")
        print("STRICT QA REPORT")
        print("============================================================")
        print(f"Total:  {self.stats['total']}")
        print(f"Passed: {self.stats['passed']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Error:  {self.stats['error']}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        print("\nBy Scope:")
        for k in sorted(self.stats["by_scope"].keys()):
            p = self.stats["by_scope"][k]["passed"]
            f = self.stats["by_scope"][k]["failed"]
            tot = p + f
            rate = (p / tot * 100) if tot else 0.0
            print(f"  - {k}: {p}/{tot} ({rate:.1f}%)")
        print("\nBy Type:")
        for k in sorted(self.stats["by_type"].keys()):
            p = self.stats["by_type"][k]["passed"]
            f = self.stats["by_type"][k]["failed"]
            tot = p + f
            rate = (p / tot * 100) if tot else 0.0
            print(f"  - {k}: {p}/{tot} ({rate:.1f}%)")

        failed = [r for r in self.results if not r["passed"]][:12]
        print("\nTop Failures:")
        for r in failed:
            print(f"  - {r['id']} [{r['scope']}/{r['type']}]: {r['reason']}")

        print(f"\nSaved: {out_csv}")
        print("============================================================")
        return out_csv


def main():
    parser = argparse.ArgumentParser(description="Strict QA Suite")
    parser.add_argument("--mode", choices=["full", "base", "rag", "general", "auth"], default="full")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=35)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--student-number", default=None)
    parser.add_argument("--student-name", default=None)
    args = parser.parse_args()

    suite = StrictQASuite(
        mode=args.mode,
        workers=args.workers,
        timeout=args.timeout,
        limit=args.limit,
        verbose=args.verbose,
        student_number=args.student_number,
        student_name=args.student_name,
    )
    suite.load_tests()
    suite.run()
    suite.save_report()


if __name__ == "__main__":
    main()
