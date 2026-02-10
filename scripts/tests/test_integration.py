"""
Integration Test Suite for UCSI Buddy Chatbot

Tests:
1. MongoDB Connection & Data Retrieval
2. LLM (Gemini) Integration
3. Intent Classification (Hybrid)
4. RAG Pipeline
5. End-to-End Chat Flow

Usage:
    python scripts/tests/test_integration.py
    python scripts/tests/test_integration.py --verbose
    python scripts/tests/test_integration.py --test db
    python scripts/tests/test_integration.py --test llm
    python scripts/tests/test_integration.py --test all
"""

import asyncio
import argparse
import sys
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()


class TestResult:
    def __init__(self, name: str, passed: bool, message: str, details: Any = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details
        self.timestamp = datetime.now().isoformat()

    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} | {self.name}: {self.message}"


class IntegrationTester:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[TestResult] = []

    def log(self, message: str):
        if self.verbose:
            print(f"  [DEBUG] {message}")

    async def test_mongodb_connection(self) -> TestResult:
        """Test MongoDB connection and basic operations."""
        try:
            from app.engines.db_engine_async import db_engine_async

            # Connect
            await db_engine_async.connect()

            if db_engine_async.db is None:
                return TestResult(
                    "MongoDB Connection",
                    False,
                    "Failed to connect to MongoDB",
                )

            # Ping
            await db_engine_async.client.admin.command('ping')

            # List collections
            collections = await db_engine_async.db.list_collection_names()
            self.log(f"Found collections: {collections}")

            return TestResult(
                "MongoDB Connection",
                True,
                f"Connected successfully. Found {len(collections)} collections.",
                {"collections": collections},
            )
        except Exception as e:
            return TestResult("MongoDB Connection", False, str(e))

    async def test_student_data_retrieval(self) -> TestResult:
        """Test student data retrieval from MongoDB."""
        try:
            from app.engines.db_engine_async import db_engine_async

            if db_engine_async.db is None:
                await db_engine_async.connect()

            # Get a sample student (first one available)
            sample = await db_engine_async.student_coll.find_one({})

            if not sample:
                return TestResult(
                    "Student Data Retrieval",
                    False,
                    "No student data found in collection",
                )

            student_number = sample.get("STUDENT_NUMBER")
            self.log(f"Found sample student: {student_number}")

            # Test get_student_by_number
            fetched = await db_engine_async.get_student_by_number(str(student_number))

            if not fetched:
                return TestResult(
                    "Student Data Retrieval",
                    False,
                    f"Could not fetch student by number: {student_number}",
                )

            return TestResult(
                "Student Data Retrieval",
                True,
                f"Successfully retrieved student data",
                {"student_number": student_number, "fields": list(fetched.keys())},
            )
        except Exception as e:
            return TestResult("Student Data Retrieval", False, str(e))

    async def test_staff_data_retrieval(self) -> TestResult:
        """Test staff data retrieval from MongoDB."""
        try:
            from app.engines.db_engine_async import db_engine_async

            if db_engine_async.db is None:
                await db_engine_async.connect()

            # Check if UCSI_STAFF collection exists
            collections = await db_engine_async.db.list_collection_names()
            if "UCSI_STAFF" not in collections:
                return TestResult(
                    "Staff Data Retrieval",
                    False,
                    "UCSI_STAFF collection not found",
                )

            # Get a sample staff
            sample = await db_engine_async.db["UCSI_STAFF"].find_one({})

            if not sample:
                return TestResult(
                    "Staff Data Retrieval",
                    False,
                    "No staff data found",
                )

            # Try to find a staff member
            staff_members = sample.get("staff_members", [])
            if staff_members and len(staff_members) > 0:
                name = staff_members[0].get("name", "")
                if name:
                    found = await db_engine_async.find_staff_by_name(name)
                    if found:
                        return TestResult(
                            "Staff Data Retrieval",
                            True,
                            f"Successfully found staff: {name}",
                            found,
                        )

            return TestResult(
                "Staff Data Retrieval",
                True,
                "Staff collection exists but no searchable members found",
                {"major": sample.get("major")},
            )
        except Exception as e:
            return TestResult("Staff Data Retrieval", False, str(e))

    async def test_llm_connection(self) -> TestResult:
        """Test LLM (Gemini) API connection."""
        try:
            from app.engines.ai_engine_async import ai_engine_async

            if not ai_engine_async.client:
                return TestResult(
                    "LLM Connection",
                    False,
                    "Gemini client not initialized. Check GEMINI_API_KEY.",
                )

            # Test simple generation
            response = await ai_engine_async.client.aio.models.generate_content(
                model=ai_engine_async.model_name,
                contents="Say 'Hello' in one word.",
            )

            if response and response.text:
                return TestResult(
                    "LLM Connection",
                    True,
                    f"Gemini API working. Model: {ai_engine_async.model_name}",
                    {"response_preview": response.text[:100]},
                )

            return TestResult(
                "LLM Connection",
                False,
                "No response from Gemini API",
            )
        except Exception as e:
            return TestResult("LLM Connection", False, str(e))

    async def test_intent_classification(self) -> TestResult:
        """Test intent classification with various queries."""
        try:
            from app.engines.intent_classifier import intent_classifier

            test_cases = [
                ("hostel fee 얼마야?", "ucsi_domain"),
                ("my GPA", "personal"),
                ("who is Taylor Swift?", "general_knowledge"),
                ("물구나무 서봐", "capability"),
            ]

            results = []
            for query, expected_category in test_cases:
                classification = await intent_classifier.classify(
                    user_message=query,
                    language="auto",
                )
                actual = classification.get("category", "unknown")
                passed = actual == expected_category
                results.append({
                    "query": query,
                    "expected": expected_category,
                    "actual": actual,
                    "passed": passed,
                    "confidence": classification.get("confidence", 0),
                    "source": classification.get("source", "unknown"),
                })
                self.log(f"Query: {query} -> {actual} (expected: {expected_category})")

            passed_count = sum(1 for r in results if r["passed"])
            total = len(results)

            return TestResult(
                "Intent Classification",
                passed_count == total,
                f"Passed {passed_count}/{total} test cases",
                {"test_cases": results},
            )
        except Exception as e:
            return TestResult("Intent Classification", False, str(e))

    async def test_semantic_router(self) -> TestResult:
        """Test semantic router (vector search)."""
        try:
            from app.engines.semantic_router_async import semantic_router_async

            # Initialize
            ok = await semantic_router_async.initialize()
            if not ok:
                return TestResult(
                    "Semantic Router",
                    False,
                    "Failed to initialize semantic router",
                )

            # Test classification
            result = await semantic_router_async.classify(
                user_message="기숙사 비용 알려줘",
                language="ko",
            )

            if not result:
                return TestResult(
                    "Semantic Router",
                    False,
                    "No result from semantic router",
                )

            return TestResult(
                "Semantic Router",
                True,
                f"Intent: {result.get('intent')}, Confidence: {result.get('confidence'):.2f}",
                result,
            )
        except Exception as e:
            return TestResult("Semantic Router", False, str(e))

    async def test_rag_pipeline(self) -> TestResult:
        """Test RAG search pipeline."""
        try:
            from app.engines.rag_engine_async import rag_engine_async
            from app.engines.db_engine_async import db_engine_async

            if db_engine_async.db is None:
                await db_engine_async.connect()

            # Initialize RAG
            await rag_engine_async.initialize()

            # Test search
            result = await rag_engine_async.search_context(
                query="hostel fee",
                top_k=3,
            )

            if not result:
                return TestResult(
                    "RAG Pipeline",
                    False,
                    "No result from RAG search",
                )

            has_data = result.get("has_relevant_data", False)
            confidence = result.get("confidence", 0)
            sources = result.get("sources", [])

            return TestResult(
                "RAG Pipeline",
                has_data or confidence > 0,
                f"Has data: {has_data}, Confidence: {confidence:.2f}, Sources: {len(sources)}",
                {
                    "has_relevant_data": has_data,
                    "confidence": confidence,
                    "sources": sources,
                    "context_preview": (result.get("context") or "")[:200],
                },
            )
        except Exception as e:
            return TestResult("RAG Pipeline", False, str(e))

    async def test_feedback_system(self) -> TestResult:
        """Test feedback save and retrieval."""
        try:
            from app.engines.db_engine_async import db_engine_async

            if db_engine_async.db is None:
                await db_engine_async.connect()

            # Test saving feedback
            test_feedback = {
                "user_message": "test question for integration test",
                "ai_response": "test answer for integration test",
                "rating": "positive",
                "comment": "Integration test feedback",
                "session_id": "test_session",
            }

            saved = await db_engine_async.save_feedback(test_feedback)

            if not saved:
                return TestResult(
                    "Feedback System",
                    False,
                    "Failed to save feedback",
                )

            # Test learned response retrieval
            learned = await db_engine_async.search_learned_response(
                "test question for integration test"
            )

            return TestResult(
                "Feedback System",
                True,
                f"Feedback saved. Learned response found: {bool(learned)}",
                {"saved": saved, "learned_found": bool(learned)},
            )
        except Exception as e:
            return TestResult("Feedback System", False, str(e))

    async def test_end_to_end_chat(self) -> TestResult:
        """Test end-to-end chat flow."""
        try:
            from app.engines.intent_classifier import intent_classifier
            from app.engines.ai_engine_async import ai_engine_async
            from app.engines.db_engine_async import db_engine_async

            if db_engine_async.db is None:
                await db_engine_async.connect()

            # Simulate chat flow
            test_message = "Tell me about UCSI hostel fees"

            # 1. Classify intent
            classification = await intent_classifier.classify(
                user_message=test_message,
                language="en",
            )
            self.log(f"Classification: {classification}")

            # 2. Generate response
            ai_result = await ai_engine_async.process_message(
                user_message=test_message,
                data_context="",
                language="en",
            )

            response_text = ai_result.get("response", "")

            if not response_text:
                return TestResult(
                    "End-to-End Chat",
                    False,
                    "No response generated",
                )

            return TestResult(
                "End-to-End Chat",
                True,
                f"Chat flow completed. Response length: {len(response_text)}",
                {
                    "classification": classification.get("category"),
                    "response_preview": response_text[:200],
                },
            )
        except Exception as e:
            return TestResult("End-to-End Chat", False, str(e))

    async def run_all_tests(self) -> List[TestResult]:
        """Run all integration tests."""
        print("\n" + "=" * 60)
        print("UCSI Buddy Integration Test Suite")
        print("=" * 60 + "\n")

        tests = [
            ("MongoDB Connection", self.test_mongodb_connection),
            ("Student Data Retrieval", self.test_student_data_retrieval),
            ("Staff Data Retrieval", self.test_staff_data_retrieval),
            ("LLM Connection", self.test_llm_connection),
            ("Semantic Router", self.test_semantic_router),
            ("Intent Classification", self.test_intent_classification),
            ("RAG Pipeline", self.test_rag_pipeline),
            ("Feedback System", self.test_feedback_system),
            ("End-to-End Chat", self.test_end_to_end_chat),
        ]

        for name, test_func in tests:
            print(f"Running: {name}...")
            try:
                result = await test_func()
                self.results.append(result)
                print(f"  {result}")
                if self.verbose and result.details:
                    print(f"  Details: {json.dumps(result.details, indent=2, default=str)[:500]}")
            except Exception as e:
                result = TestResult(name, False, f"Unexpected error: {e}")
                self.results.append(result)
                print(f"  {result}")

        return self.results

    async def run_specific_test(self, test_name: str) -> List[TestResult]:
        """Run a specific test."""
        test_map = {
            "db": [self.test_mongodb_connection, self.test_student_data_retrieval, self.test_staff_data_retrieval],
            "llm": [self.test_llm_connection],
            "intent": [self.test_semantic_router, self.test_intent_classification],
            "rag": [self.test_rag_pipeline],
            "feedback": [self.test_feedback_system],
            "e2e": [self.test_end_to_end_chat],
        }

        tests = test_map.get(test_name.lower(), [])
        if not tests:
            print(f"Unknown test: {test_name}")
            print(f"Available tests: {', '.join(test_map.keys())}")
            return []

        print(f"\nRunning {test_name} tests...\n")

        for test_func in tests:
            result = await test_func()
            self.results.append(result)
            print(f"  {result}")
            if self.verbose and result.details:
                print(f"  Details: {json.dumps(result.details, indent=2, default=str)[:500]}")

        return self.results

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        for result in self.results:
            status = "✅" if result.passed else "❌"
            print(f"  {status} {result.name}")

        print(f"\nTotal: {passed}/{total} passed")

        if passed == total:
            print("\n🎉 All tests passed!")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed")

        return passed == total


async def main():
    parser = argparse.ArgumentParser(description="UCSI Buddy Integration Tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--test", "-t", type=str, default="all", help="Specific test to run (db, llm, intent, rag, feedback, e2e, all)")
    args = parser.parse_args()

    tester = IntegrationTester(verbose=args.verbose)

    if args.test.lower() == "all":
        await tester.run_all_tests()
    else:
        await tester.run_specific_test(args.test)

    success = tester.print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
