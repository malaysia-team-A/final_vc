"""
Test Runner for UCSI Buddy Chatbot

Runs all test suites:
1. Integration Tests
2. Security Tests
3. UX Tests

Usage:
    python scripts/tests/run_all_tests.py
    python scripts/tests/run_all_tests.py --verbose
    python scripts/tests/run_all_tests.py --suite security
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()


async def run_integration_tests(verbose: bool = False):
    """Run integration tests."""
    from scripts.tests.test_integration import IntegrationTester

    print("\n" + "=" * 70)
    print("INTEGRATION TESTS")
    print("=" * 70)

    tester = IntegrationTester(verbose=verbose)
    await tester.run_all_tests()
    return tester.results


async def run_security_tests(verbose: bool = False):
    """Run security tests."""
    from scripts.tests.test_security import SecurityTester

    print("\n" + "=" * 70)
    print("SECURITY TESTS")
    print("=" * 70)

    tester = SecurityTester(verbose=verbose)
    await tester.run_all_tests()
    return tester.results


async def run_ux_tests(verbose: bool = False):
    """Run UX tests."""
    from scripts.tests.test_ux import UXTester

    print("\n" + "=" * 70)
    print("UX TESTS")
    print("=" * 70)

    tester = UXTester(verbose=verbose)
    await tester.run_all_tests()
    return tester.results


async def main():
    parser = argparse.ArgumentParser(description="UCSI Buddy Test Runner")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--suite", "-s",
        type=str,
        default="all",
        choices=["all", "integration", "security", "ux"],
        help="Which test suite to run",
    )
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("UCSI Buddy Test Runner")
    print(f"Started: {datetime.now().isoformat()}")
    print("#" * 70)

    all_results = []

    if args.suite in ("all", "integration"):
        results = await run_integration_tests(args.verbose)
        all_results.extend(results)

    if args.suite in ("all", "security"):
        results = await run_security_tests(args.verbose)
        all_results.extend(results)

    if args.suite in ("all", "ux"):
        results = await run_ux_tests(args.verbose)
        all_results.extend(results)

    # Final Summary
    print("\n" + "#" * 70)
    print("FINAL TEST SUMMARY")
    print("#" * 70)

    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)

    # Group by test type
    integration_results = [r for r in all_results if "MongoDB" in r.name or "LLM" in r.name or "RAG" in r.name or "E2E" in r.name or "Intent" in r.name or "Feedback" in r.name or "Staff" in r.name or "Student" in r.name or "Semantic" in r.name]
    security_results = [r for r in all_results if "Injection" in r.name or "Sanitiz" in r.name or "Validation" in r.name or "Safe" in r.name]
    ux_results = [r for r in all_results if "Greeting" in r.name or "Error" in r.name or "Formatting" in r.name or "Clarification" in r.name or "Suggestion" in r.name or "UX" in r.name]

    def print_category(name, results):
        if not results:
            return
        cat_passed = sum(1 for r in results if r.passed)
        print(f"\n{name}: {cat_passed}/{len(results)}")
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.name}")

    print_category("Integration Tests", integration_results)
    print_category("Security Tests", security_results)
    print_category("UX Tests", ux_results)

    print(f"\n{'=' * 40}")
    print(f"TOTAL: {passed}/{total} tests passed")
    print(f"{'=' * 40}")

    if passed == total:
        print("\nALL TESTS PASSED!")
        sys.exit(0)
    else:
        print(f"\n{total - passed} test(s) FAILED")

        # Show failed tests
        failed = [r for r in all_results if not r.passed]
        if failed:
            print("\nFailed tests:")
            for r in failed:
                print(f"  - {r.name}: {r.message}")

        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
