import ast
import sys

files = [
    "app/api/chat.py",
    "app/engines/rag_engine_async.py",
    "app/engines/rag_engine.py",
    "app/engines/ai_engine_async.py",
]

ok = True
for f in files:
    try:
        with open(f, encoding="utf-8") as fh:
            ast.parse(fh.read())
        print(f"OK: {f}")
    except SyntaxError as e:
        print(f"FAIL: {f} -> {e}")
        ok = False

sys.exit(0 if ok else 1)
