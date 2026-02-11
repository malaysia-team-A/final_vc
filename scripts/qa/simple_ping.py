
import requests
import sys

try:
    print("Checking specific endpoints...", flush=True)
    
    # 1. Health/Root check
    try:
        r = requests.get("http://localhost:8000/", timeout=2)
        print(f"ROOT: {r.status_code}", flush=True)
    except Exception as e:
        print(f"ROOT FAIL: {e}", flush=True)

    # 2. Chat endpoint (OPTIONS/POST check)
    try:
        r = requests.post("http://localhost:8000/api/chat", json={"message": "ping", "session_id": "ping"}, timeout=5)
        print(f"CHAT: {r.status_code} - {r.text[:50]}", flush=True)
    except Exception as e:
        print(f"CHAT FAIL: {e}", flush=True)

except Exception as e:
    print(f"GLOBAL FAIL: {e}", flush=True)
