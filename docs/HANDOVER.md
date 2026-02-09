# UCSI 버디 인수인계 문서 (최신)

- 작성일: 2026-02-09
- 대상: 다음 개발 세션 (백엔드/프런트엔드/QA)

## 1. 현재 구조 핵심
- 서버: `main.py` (FastAPI, 포트 8000)
- 인증: `app/api/auth.py`
- 채팅: `app/api/chat.py`
- 관리자: `app/api/admin.py`
- 비동기 엔진: `ai_engine_async.py`, `db_engine_async.py`, `rag_engine_async.py`
- RAG 코어: `rag_engine.py` (FAISS + MongoDB 인덱싱)

## 2. 반드시 알고 시작할 사항
1. 서버 시작 시 RAG 재색인이 자동 수행됨
2. 개인 정보 질의는 JWT 로그인 필요
3. GPA/성적 질의는 비밀번호 재검증 필요
4. 피드백은 `LearnedQA`/`BadQA`에 반영됨

## 3. 주요 API
- 인증
  - `POST /api/login`
  - `POST /api/verify_password`
  - `POST /api/logout`
- 채팅
  - `POST /api/chat`
  - `POST /api/feedback`
  - `GET /api/export_chat`
- 관리자
  - `GET /api/admin/stats`
  - `POST /api/admin/upload`
  - `POST /api/admin/reindex`
  - `GET /api/admin/files`
  - `DELETE /api/admin/files`

## 4. 최근 검증 결과
- Strict QA: `58/58` 통과
- Stress Test: `300/300` HTTP 성공, `300/300` semantic 통과
- 리포트 경로
  - `data/reports/strict_qa_report_latest.csv`
  - `data/reports/stress_test_report_latest.csv`

## 5. 남은 이슈 (우선순위)
1. P0: 관리자 API 인증 미적용
2. P1: 코드/프런트 문자열 인코딩 깨짐 정리
3. P1: 레거시 동기 엔진 정리
4. P1: QA 자동화 파이프라인 정착

## 6. 다음 세션 권장 순서
1. 관리자 인증 미들웨어/의존성 우선 적용
2. 문자열 인코딩 정리 및 사용자 문구 검수
3. strict/stress QA 재실행
4. 결과를 `docs/PROJECT_STATUS_ANALYSIS.md`에 반영
