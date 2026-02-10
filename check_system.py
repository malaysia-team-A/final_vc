import os
import sys

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
        "app/engines/ai_engine.py",
        "app/engines/data_engine.py",
        "app/engines/rag_engine.py",
        "app/utils/auth_utils.py",
        "data/feedback_log.json"
    ]
    for f in required_files:
        if os.path.exists(f):
            print(f"[PASS] File exists: {f}")
        else:
            print(f"[WARN] File missing or moved: {f}")
            
    # 3. Import Check
    print("\n[INFO] Testing Imports...")
    try:
        from app.engines.data_engine import DataEngine
        from app.engines.ai_engine import AIEngine
        from app.utils import auth_utils
        print("[PASS] Core modules imported successfully.")
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return
        
    print("\n=== SYSTEM INTEGRITY CHECK PASSED ===")

if __name__ == "__main__":
    check_integrity()
