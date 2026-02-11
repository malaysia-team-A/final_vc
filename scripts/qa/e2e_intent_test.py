
import requests
import json
import time

BASE_URL = "http://localhost:8000/api/chat"

def check_intent(query, expected_route=None, expected_text=None, auth_token=None):
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    payload = {
        "message": query,
        "session_id": f"test_e2e_{int(time.time())}"
    }

    print(f"\n[TESTING] Query: '{query}'")
    try:
        response = requests.post(BASE_URL, json=payload, headers=headers)
        if response.status_code != 200:
            print(f"FAILED: Status {response.status_code}")
            return False

        data = response.json()
        text = data.get("text", "")
        retrieval = data.get("retrieval", {})
        used_route = retrieval.get("route", "unknown")
        
        print(f"  -> Route: {used_route}")
        print(f"  -> Text: {text[:100]}...")
        
        # Verification Logic
        success = True
        if expected_route and expected_route not in used_route:
            print(f"  [FAIL] Expected route '{expected_route}', got '{used_route}'")
            success = False
        
        if expected_text and expected_text.lower() not in text.lower():
            print(f"  [FAIL] Expected text '{expected_text}' not found.")
            success = False
            
        if success:
            print("  [PASS] All checks passed.")
        return success

    except Exception as e:
        print(f"ERROR: {e}")
        return False

def run_tests():
    with open("scripts/qa/e2e_results.txt", "w", encoding="utf-8") as f:
        f.write("=== STARTING E2E INTENT VERIFICATION ===\n")
        
        def log(msg):
            print(msg)
            f.write(msg + "\n")
            f.flush()

        # 1. Hostel Check-in
        log("\n[TEST 1] Hostel Check-in")
        res1 = check_intent("What is the hostel check-in time?", expected_route="rag")
        log(f"RESULT: {'PASS' if res1 else 'FAIL'}")

        # 2. Pet Policy
        log("\n[TEST 2] Pet Policy")
        res2 = check_intent("Can I bring my cat?", expected_route="rag", expected_text="no")
        log(f"RESULT: {'PASS' if res2 else 'FAIL'}")

        # 3. Student Info (Unauthenticated)
        log("\n[TEST 3] Student Info (Unauth)")
        res3 = check_intent("Who is Vicky Yiran?", expected_text="login")
        log(f"RESULT: {'PASS' if res3 else 'FAIL'}")
        
        # 4. GPA (Unauthenticated)
        log("\n[TEST 4] GPA (Unauth)")
        res4 = check_intent("My GPA", expected_text="login")
        log(f"RESULT: {'PASS' if res4 else 'FAIL'}")

        # 5. General Knowledge (BTS)
        log("\n[TEST 5] General Knowledge (BTS)")
        res5 = check_intent("Tell me about BTS.", expected_route="general_ai")
        log(f"RESULT: {'PASS' if res5 else 'FAIL'}")

        log("\n=== E2E TESTS COMPLETED ===")

if __name__ == "__main__":
    # Wait for user to start server
    print("Make sure the server is running on http://localhost:8000")
    run_tests()
