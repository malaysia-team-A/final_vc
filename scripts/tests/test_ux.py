"""
UX Test Suite for UCSI Buddy Chatbot

Tests:
1. Greeting Messages
2. Error Messages
3. Response Formatting
4. Clarification Messages

Usage:
    python scripts/tests/test_ux.py
    python scripts/tests/test_ux.py --verbose
"""

import asyncio
import argparse
import sys
import os
from typing import List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()


class TestResult:
    def __init__(self, name: str, passed: bool, message: str, details=None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


class UXTester:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[TestResult] = []

    def log(self, msg: str):
        if self.verbose:
            print(f"    {msg}")

    def test_greeting_messages(self) -> TestResult:
        """Test greeting message generation."""
        from app.engines.ux_engine import get_greeting

        test_cases = [
            ("en", False),
            ("en", True),
            ("ko", False),
            ("ko", True),
        ]

        passed = 0

        for language, is_returning in test_cases:
            greeting = get_greeting(language, is_returning)

            if greeting and len(greeting) > 5:
                passed += 1
                self.log(f"OK: {language}, returning={is_returning} -> '{greeting[:50]}...'")
            else:
                self.log(f"FAIL: No greeting for {language}, returning={is_returning}")

        total = len(test_cases)
        return TestResult(
            "Greeting Messages",
            passed == total,
            f"{passed}/{total} greetings generated",
        )

    def test_error_messages(self) -> TestResult:
        """Test error message generation."""
        from app.engines.ux_engine import get_error_message

        error_types = [
            "generic",
            "no_data",
            "login_required",
            "password_required",
            "rate_limited",
            "timeout",
            "invalid_input",
            "ambiguous",
        ]

        languages = ["en", "ko"]
        passed = 0
        total = 0

        for error_type in error_types:
            for lang in languages:
                total += 1
                message = get_error_message(error_type, lang)

                if message and len(message) > 5:
                    passed += 1
                    self.log(f"OK: {error_type}/{lang} -> '{message[:40]}...'")
                else:
                    self.log(f"FAIL: No message for {error_type}/{lang}")

        return TestResult(
            "Error Messages",
            passed == total,
            f"{passed}/{total} error messages generated",
        )

    def test_response_formatting(self) -> TestResult:
        """Test response formatting functions."""
        from app.engines.ux_engine import (
            format_list_response,
            format_key_value_response,
            truncate_response,
        )

        # Test list formatting
        items = ["Item 1", "Item 2", "Item 3"]
        formatted_list = format_list_response(items, "Test List")
        list_ok = "1. Item 1" in formatted_list and "Test List" in formatted_list

        # Test key-value formatting
        data = {"name": "Test", "value": 100}
        formatted_kv = format_key_value_response(data)
        kv_ok = "Name" in formatted_kv and "100" in formatted_kv

        # Test truncation
        long_text = "A" * 2000
        truncated = truncate_response(long_text, max_length=100)
        truncate_ok = len(truncated) < len(long_text)

        passed = sum([list_ok, kv_ok, truncate_ok])
        total = 3

        self.log(f"List formatting: {'OK' if list_ok else 'FAIL'}")
        self.log(f"Key-value formatting: {'OK' if kv_ok else 'FAIL'}")
        self.log(f"Truncation: {'OK' if truncate_ok else 'FAIL'}")

        return TestResult(
            "Response Formatting",
            passed == total,
            f"{passed}/{total} formatting functions work",
        )

    def test_clarification_messages(self) -> TestResult:
        """Test clarification message creation."""
        from app.engines.ux_engine import create_clarification_message

        test_cases = [
            ("ambiguous_person", "en", {"name": "John", "options": "1. John A\n2. John B"}),
            ("ambiguous_topic", "en", {"options": "1. Fees\n2. Schedule"}),
            ("need_more_info", "ko", {"suggestions": "- 학과\n- 학년"}),
            ("confirm_intent", "ko", {"topic": "기숙사 비용"}),
        ]

        passed = 0

        for clarification_type, lang, kwargs in test_cases:
            message = create_clarification_message(clarification_type, lang, **kwargs)

            if message and len(message) > 5:
                passed += 1
                self.log(f"OK: {clarification_type}/{lang} -> '{message[:50]}...'")
            else:
                self.log(f"FAIL: No clarification for {clarification_type}/{lang}")

        total = len(test_cases)
        return TestResult(
            "Clarification Messages",
            passed == total,
            f"{passed}/{total} clarification messages created",
        )

    def test_follow_up_suggestions(self) -> TestResult:
        """Test follow-up suggestion generation."""
        from app.engines.ux_engine import create_follow_up_suggestions

        topics = ["hostel", "programme", "staff", "personal", "default"]
        languages = ["en", "ko"]

        passed = 0
        total = 0

        for topic in topics:
            for lang in languages:
                total += 1
                suggestions = create_follow_up_suggestions(topic, lang)

                if isinstance(suggestions, list) and len(suggestions) > 0:
                    passed += 1
                    self.log(f"OK: {topic}/{lang} -> {len(suggestions)} suggestions")
                else:
                    self.log(f"FAIL: No suggestions for {topic}/{lang}")

        return TestResult(
            "Follow-up Suggestions",
            passed == total,
            f"{passed}/{total} suggestion sets generated",
        )

    def test_ux_engine_enhance(self) -> TestResult:
        """Test UX engine response enhancement."""
        from app.engines.ux_engine import ux_engine

        test_response = "Here is information about the hostel."

        enhanced = ux_engine.enhance_response(
            response_text=test_response,
            language="en",
            category="ucsi_hostel",
            sources=["MongoDB:Hostel"],
            is_first_message=False,
            session_id="test123",
        )

        has_text = "text" in enhanced and enhanced["text"]
        has_suggestions = "suggestions" in enhanced and isinstance(enhanced["suggestions"], list)
        has_metadata = "metadata" in enhanced

        passed = sum([has_text, has_suggestions, has_metadata])
        total = 3

        self.log(f"Has text: {'OK' if has_text else 'FAIL'}")
        self.log(f"Has suggestions: {'OK' if has_suggestions else 'FAIL'}")
        self.log(f"Has metadata: {'OK' if has_metadata else 'FAIL'}")

        return TestResult(
            "UX Engine Enhancement",
            passed == total,
            f"{passed}/{total} enhancement components present",
        )

    async def run_all_tests(self) -> List[TestResult]:
        """Run all UX tests."""
        print("\n" + "=" * 60)
        print("UCSI Buddy UX Test Suite")
        print("=" * 60 + "\n")

        tests = [
            self.test_greeting_messages,
            self.test_error_messages,
            self.test_response_formatting,
            self.test_clarification_messages,
            self.test_follow_up_suggestions,
            self.test_ux_engine_enhance,
        ]

        for test_func in tests:
            print(f"Running: {test_func.__name__}...")
            try:
                result = test_func()
                self.results.append(result)
                print(f"  {result}")
            except Exception as e:
                result = TestResult(test_func.__name__, False, f"Error: {e}")
                self.results.append(result)
                print(f"  {result}")

        return self.results

    def print_summary(self) -> bool:
        """Print test summary."""
        print("\n" + "=" * 60)
        print("UX Test Summary")
        print("=" * 60)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] {result.name}")

        print(f"\nTotal: {passed}/{total} passed")

        if passed == total:
            print("\nAll UX tests passed!")
            return True
        else:
            print(f"\n{total - passed} test(s) failed")
            return False


async def main():
    parser = argparse.ArgumentParser(description="UCSI Buddy UX Tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    tester = UXTester(verbose=args.verbose)
    await tester.run_all_tests()
    success = tester.print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
