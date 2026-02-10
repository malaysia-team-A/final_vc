# UCSI 버디 인수인계 문서 (최신)

- 작성일: 2026-02-10
- 버전: 3.2.0
- 대상: 다음 개발 세션 (백엔드/프런트엔드/QA)

## 1. 현재 구조 핵심
- 서버: `main.py` (FastAPI, 포트 8000)
- 인증: `app/api/auth.py` (JWT, 2FA)
- 채팅: `app/api/chat.py` (하이브리드 인텐트 분류기 통합)
- 관리자: `app/api/admin.py` (JWT + admin role 체크 적용 완료)
- 비동기 엔진: `ai_engine_async.py`, `db_engine_async.py`, `rag_engine_async.py`
- RAG 코어: `rag_engine.py` (FAISS + MongoDB 인덱싱)
- 인텐트 분류: `intent_classifier.py` (Keyword Guard → Vector → LLM 3단계)
- 응답 검증: `response_validator.py` (할루시네이션 방지, 숫자 검증)
- 모니터링: `monitoring.py` (응답 시간, RAG hit rate, 분류 통계)

## 2. 반드시 알고 시작할 사항
1. 서버 시작 시 RAG 재색인이 자동 수행됨 (`index_mongodb_collections`)
2. 개인 정보 질의는 JWT 로그인 필요 (`POST /api/login`)
3. GPA/성적 질의는 비밀번호 재검증 필요 (`POST /api/verify_password`, 10분 유효)
4. 피드백은 `Feedback` 컬렉션에 저장되며 RLHF 정책 신호로 활용됨
   - 긍정 피드백 → 학습 응답 (learned response)
   - 부정 피드백 → 가드 레일 (bad response guard)
5. 관리자 API는 `require_admin` 의존성으로 JWT + admin 역할을 체크함
6. 인텐트 분류는 3단계 하이브리드: Keyword Guard(즉시) → Vector Search(0.65+) → LLM Fallback
7. UCSI 도메인 응답에 `rich_content` (링크/이미지)가 포함될 수 있음 → 프론트엔드에서 자동 렌더링

## 3. 주요 API
- 인증
  - `POST /api/login` - 학생 로그인 (학번 + 이름)
  - `POST /api/verify_password` - 비밀번호 2FA (10분)
  - `POST /api/logout`
- 채팅
  - `POST /api/chat` - 메인 채팅
  - `POST /api/feedback` - 피드백 (positive/negative)
  - `GET /api/export_chat` - 대화 내보내기
- 관리자 (JWT + admin role 필요)
  - `POST /api/admin/login` / `POST /api/admin/logout`
  - `GET /api/admin/stats` - 통계
  - `POST /api/admin/upload` - 문서 업로드 (PDF/TXT/CSV)
  - `POST /api/admin/reindex` - 재인덱싱
  - `GET /api/admin/files` / `DELETE /api/admin/files`
  - `GET /api/admin/index/status` / `POST /api/admin/index/reindex`
  - `GET /api/admin/index/history` / `GET /api/admin/index/changes`
  - `GET /api/admin/monitoring/dashboard`
  - `GET /api/admin/unanswered/analysis` / `GET /api/admin/unanswered/report`
  - `GET /api/admin/health` (Public - 인증 불필요)

## 4. 최근 검증 결과
- Strict QA: `58/58` 통과
- Stress Test: `300/300` HTTP 성공, `300/300` semantic 통과
- 리포트 경로
  - `data/reports/strict_qa_report_latest.csv`
  - `data/reports/stress_test_report_latest.csv`

## 5. 완료된 사항 (v3.1.0 ~ v3.2.0)
1. 관리자 API 인증 적용 (`require_admin` 의존성)
2. 하이브리드 인텐트 분류기 도입 (`intent_classifier.py`)
3. 응답 검증기 도입 (`response_validator.py`)
4. 성능 모니터링 시스템 구축 (`monitoring.py`)
5. 인덱스 관리 자동화 (`index_manager.py`)
6. 미답변 질문 분석기 (`unanswered_analyzer.py`)
7. UX 엔진 (`ux_engine.py`)
8. **[v3.2.0]** Rich Content 지원 - Staff 프로필 링크, 건물 이미지, 프로그램/지도 링크
9. **[v3.2.0]** Staff 인덱싱 개선 - 구조화된 key-value 포맷 (`rag_engine_async.py`)
10. **[v3.2.0]** 응답 텍스트 URL 자동 링크화 (`app.js`)

## 6. 남은 이슈 (우선순위)
1. **P0**: 세션/대화기록 영속성 - 현재 인메모리, 서버 재시작 시 유실 → MongoDB TTL 컬렉션 도입 필요
2. **P1**: RAG 인덱스 디스크 캐싱 - 매 시작 시 전체 DB 재로드 → FAISS 디스크 캐싱 도입
3. **P1**: 코드/프런트 문자열 인코딩 깨짐 정리 (`test_integration.py` 한글 테스트 케이스)
4. **P1**: `chat_legacy.py` 정리 여부 결정 (현재 백업용으로 보존 중)
5. **P2**: QA 자동화 파이프라인 정착 (CI/CD 연동)
6. **P2**: 구조적 로깅 시스템 표준화

## 7. 다음 세션 권장 순서
1. 세션 영속성 구현 (MongoDB TTL 컬렉션)
2. FAISS 인덱스 디스크 캐싱 도입
3. 인코딩 깨짐 정리 및 사용자 문구 검수
4. strict/stress QA 재실행 → 결과 반영
5. `docs/PROJECT_STATUS_ANALYSIS.md`에 최신 결과 반영
