import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

try:
    from app.engines.db_engine import db_engine
    print("Import Successful: db_engine is valid.")
except Exception as e:
    print(f"Import Failed: {e}")
except SyntaxError as e:
    print(f"Syntax Error: {e}")
