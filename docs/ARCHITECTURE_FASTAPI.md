# FastAPI Runtime Architecture

This project now runs on a single FastAPI-first runtime path.

## Entry Point
- `main.py`

## Active Engine Layer
- `app/engines/ai_engine_async.py`
- `app/engines/db_engine_async.py`
- `app/engines/rag_engine_async.py`
- `app/engines/semantic_router_async.py`
- `app/engines/language_engine.py`
- `app/engines/rag_engine.py` (CPU/FAISS core called by async wrapper)

## API Layer
- `app/api/auth.py`
- `app/api/chat.py`
- `app/api/admin.py`

## Removed Legacy Sync Engines
- `app/engines/ai_engine.py`
- `app/engines/data_engine.py`
- `app/engines/feedback_engine.py`

## Notes
- Mongo I/O for runtime is handled through `db_engine_async` (Motor).
- RAG indexing/search is exposed via `rag_engine_async`.
- Validation scripts in `scripts/checks/` and `scripts/verify_setup.py` were updated to async imports.

## Mongo Collections (Runtime)
- Canonical:
  - `Feedback` (feedback, learned positive memory, bad-answer guard signal, RLHF policy signal)
  - `unanswered` (RAG misses / no-data logs)
  - `semantic_intents` (semantic router intent vectors)
  - `UCSI` or auto-detected student collection (`students`, `Students`, `UCSI_STUDENTS`)
  - RAG domain data: `Hostel`, `UCSI_FACILITY`, `UCSI_ MAJOR`, `USCI_SCHEDUAL`, `UCSI_STAFF`, `UCSI_HOSTEL_FAQ`, `UCSI_University_Blocks_Data`
- Optional legacy fallback (disabled by default):
  - `feedbacks` via `USE_LEGACY_FEEDBACK_COLLECTION=true`
  - `LearnedQA` / `BadQA` via `USE_LEGACY_QA_COLLECTIONS=true`
