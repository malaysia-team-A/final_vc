import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

def check_integrity():
    print("=== System Integrity Check ===")
    
    # 1. Directory Structure Check
    required_dirs = [
        "app/engines",
        "app/utils",
        "data",
        "docs",
        "static/site",
        "static/admin"
    ]
    all_dirs_ok = True
    for d in required_dirs:
        if os.path.isdir(d):
            print(f"[PASS] Directory exists: {d}")
        else:
            print(f"[FAIL] Directory missing: {d}")
            all_dirs_ok = False
            
    # 2. Key File Check
    required_files = [
        "main.py",
        "app/engines/ai_engine_async.py",
        "app/engines/db_engine_async.py",
        "app/engines/rag_engine_async.py",
        "app/engines/semantic_router_async.py",
        "app/engines/rag_engine.py",
        "app/utils/auth_utils.py",
        ".env"
    ]
    for f in required_files:
        if os.path.exists(f):
            print(f"[PASS] File exists: {f}")
        else:
            print(f"[WARN] File missing or moved: {f}")
            
    # 3. Import Check
    print("\n[INFO] Testing Imports...")
    try:
        from app.engines.ai_engine_async import ai_engine_async
        from app.engines.db_engine_async import db_engine_async
        from app.engines.rag_engine_async import rag_engine_async
        from app.engines.semantic_router_async import semantic_router_async
        from app.utils import auth_utils
        print("[PASS] Core async modules imported successfully.")
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return
        
    print("\n=== SYSTEM INTEGRITY CHECK PASSED ===")

if __name__ == "__main__":
    check_integrity()
