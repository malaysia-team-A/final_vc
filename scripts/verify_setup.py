import sys
import os

print("--- Checking Imports ---")
try:
    import fastapi
    print("[OK] fastapi")
except ImportError as e:
    print(f"[FAIL] fastapi: {e}")

try:
    import uvicorn
    print("[OK] uvicorn")
except ImportError as e:
    print(f"[FAIL] uvicorn: {e}")

try:
    import pymongo
    print("[OK] pymongo")
except ImportError as e:
    print(f"[FAIL] pymongo: {e}")

try:
    import dotenv
    print("[OK] python-dotenv")
except ImportError as e:
    print(f"[FAIL] python-dotenv: {e}")

try:
    from google import genai
    print("[OK] google-genai (New SDK)")
except ImportError as e:
    print(f"[FAIL] google-genai: {e}")

try:
    import motor
    print("[OK] motor")
except ImportError as e:
    print(f"[FAIL] motor: {e}")

try:
    import sentence_transformers
    print("[OK] sentence_transformers")
except ImportError as e:
    print(f"[FAIL] sentence_transformers: {e}")

try:
    import faiss
    print("[OK] faiss-cpu")
except ImportError as e:
    print(f"[FAIL] faiss-cpu: {e}")

try:
    import pandas
    print("[OK] pandas")
except ImportError as e:
    print(f"[FAIL] pandas: {e}")

try:
    import jwt
    print("[OK] pyjwt")
except ImportError as e:
    print(f"[FAIL] pyjwt: {e}")

print("\n--- Checking Project Modules ---")
sys.path.append(os.getcwd())

try:
    from app.config import Config
    print("[OK] app.config")
except Exception as e:
    print(f"[FAIL] app.config: {e}")

try:
    import app.api.auth
    import app.api.chat
    import app.api.admin
    print("[OK] app.api.* routers")
except Exception as e:
    print(f"[FAIL] app.api routers: {e}")

try:
    from app.engines.ai_engine import AIEngine
    print("[OK] app.engines.ai_engine")
except Exception as e:
    print(f"[FAIL] app.engines.ai_engine: {e}")

try:
    from app.engines.db_engine import DatabaseEngine
    print("[OK] app.engines.db_engine")
except Exception as e:
    print(f"[FAIL] app.engines.db_engine: {e}")

print("\n--- Verification Complete ---")
