import os

import requests


API_BASE = os.getenv("QA_API_BASE", "http://localhost:5000").rstrip("/")

try:
    print("Checking server...")
    resp = requests.get(f"{API_BASE}/", timeout=5)
    print(f"Server check: {resp.status_code}")
except Exception as e:
    print(f"Server check failed: {e}")

try:
    print("Checking chat API...")
    resp = requests.post(f"{API_BASE}/api/chat", json={"message": "hi"}, timeout=10)
    print(f"Chat API check: {resp.status_code}")
    print(f"Response: {resp.text[:100]}")
except Exception as e:
    print(f"Chat API failed: {e}")
