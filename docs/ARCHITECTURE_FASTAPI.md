# FastAPI Runtime Architecture

This project runs on a single FastAPI-first runtime path with a unified chat module.

**Version:** 3.1.0
**Updated:** 2026-02-10

## Entry Point
- `main.py`

## Active Engine Layer

### Core Engines
- `app/engines/ai_engine_async.py` - LLM 엔진 (Gemini, 대화 요약 포함)
- `app/engines/db_engine_async.py` - DB 엔진 (MongoDB Motor)
- `app/engines/rag_engine_async.py` - RAG 엔진 (비동기 래퍼, 인덱스 정보 조회)
- `app/engines/rag_engine.py` - RAG 코어 (FAISS + MongoDB 인덱싱)
- `app/engines/semantic_router_async.py` - 시맨틱 라우터 (SentenceTransformer)
- `app/engines/language_engine.py` - 언어 감지

### v3.1.0 New Engines
- `app/engines/intent_classifier.py` - 하이브리드 인텐트 분류기 (Keyword Guard → Vector Search → LLM Fallback)
- `app/engines/response_validator.py` - 응답 검증 및 할루시네이션 방지
- `app/engines/ux_engine.py` - UX 엔진 (인사말, 에러 메시지, 응답 포맷팅)
- `app/engines/index_manager.py` - 인덱스 관리 (자동 재인덱싱, 변경 감지)
- `app/engines/monitoring.py` - 성능 모니터링 (응답 시간, RAG hit rate, LLM 사용량)
- `app/engines/unanswered_analyzer.py` - 미답변 질문 분석 및 리포트 생성

## API Layer
- `app/api/auth.py` - 인증 API
- `app/api/chat.py` - 통합 채팅 API (하이브리드 분류기, 보안, 모니터링 통합)
- `app/api/admin.py` - 관리자 API (인덱스 관리, 모니터링, 미답변 분석)

### Backup
- `app/api/chat_legacy.py` - 이전 chat.py 백업

## Removed Legacy Sync Engines
- `app/engines/ai_engine.py`
- `app/engines/data_engine.py`
- `app/engines/feedback_engine.py`
- `app/engines/db_engine.py`

## Notes
- Mongo I/O for runtime is handled through `db_engine_async` (Motor).
- RAG indexing/search is exposed via `rag_engine_async`.
- Validation scripts in `scripts/checks/` and `scripts/verify_setup.py` were updated to async imports.
- chat.py now includes prompt injection detection, input sanitization, and response validation.

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

## Test Suites
- `scripts/tests/test_integration.py` - 통합 테스트 (DB, LLM, RAG, E2E)
- `scripts/tests/test_security.py` - 보안 테스트 (Prompt Injection, Sanitization)
- `scripts/tests/test_ux.py` - UX 테스트 (Greeting, Error Messages)
- `scripts/tests/run_all_tests.py` - 전체 테스트 러너

## Intent Classification Flow (Hybrid)
```
User Message
    │
    ▼
[1] Keyword Guard (빠른 분류)
    ├─ 개인정보 키워드 → personal
    ├─ 능력 질문 → capability
    ├─ UCSI 키워드 → ucsi_domain (force RAG)
    │
    ▼ (매칭 없음)
[2] Vector Search (SentenceTransformer)
    ├─ confidence >= 0.65 → 분류 결과 사용
    │
    ▼ (낮은 confidence)
[3] LLM Fallback (Gemini)
    └─ 최종 분류 결정
```

## Response Validation Flow
```
LLM Response
    │
    ▼
Response Validator
    ├─ 숫자 검증 (Context와 일치 여부)
    ├─ 소스 기반 검증
    ├─ Grounding 검사
    │
    ├─ Valid → 응답 반환
    └─ Invalid → Safe Response 생성 + 로깅
```
