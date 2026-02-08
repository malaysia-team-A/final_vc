# UCSI 버디 인수인계 문서

- 작성일: 2026-02-08
- 대상: 다음 개발 세션(백엔드 + 프론트엔드 + QA)

## 1. 이번 세션까지 반영된 핵심 사항

1. RAG 중심 라우팅 구조 유지 및 고도화
2. 일반 질문/도메인 질문/개인정보 질문 경로 분리
3. 로그인 + 2차 인증(비밀번호) 보안 게이트 분리
4. `NO_DATA` 정책 적용(근거 없을 때 사실 단정 금지)
5. 추천질문(normalize) 및 피드백 저장 루프 연결
6. 이름 응답 규칙 반영 (`Buddy`, 서비스 표기 `UCSI buddy`)

## 2. 현재 아키텍처 요약

- API 진입점: `main.py`
- AI 추론: `app/engines/ai_engine.py`
- 검색/RAG: `app/engines/rag_engine.py`
- DB 연동: `app/engines/db_engine.py`, `app/engines/data_engine.py`
- 보조 엔진: `app/engines/query_rewriter.py`, `app/engines/reranker.py`, `app/engines/semantic_cache_engine.py`
- UI: `static/site/code_hompage.html`

## 3. 요청 처리 워크플로우(핵심)

1. 입력 전처리
- 이름 질문, 물리 행동형 스몰톡, 노이즈/무의미 입력은 즉시 응답

2. 라우팅 결정
- `should_force_rag()`가 참이면 도메인 질문으로 강제 RAG
- 개인질문이면 인증 경로로 이동

3. 개인질문 보안 경로
- 로그인 없음: 로그인 유도
- 로그인 있음 + 민감정보 요청: 비밀번호(2차 인증) 요구
- 2차 인증 완료: GPA/CGPA 포함 응답 허용

4. 비개인 질문 검색 경로
- 안전한 경우에만 semantic cache 사용
- 이후 RAG 검색(빠른 경로)
- 신뢰도 낮으면 query rewrite + rerank(무거운 경로)

5. 응답 후처리
- NO_DATA 시 "정보 없음" 응답 강제
- 환각성 숫자/날짜 단정 표현 점검
- 추천질문 문구를 사용자 질의형으로 정규화

## 4. 주요 API 엔드포인트

- `POST /api/login`: 1차 로그인
- `POST /api/verify_password`: 2차 인증
- `POST /api/chat`: 일반 채팅
- `POST /api/chat/stream`: 스트리밍 채팅
- `POST /api/feedback`: 답변 피드백 저장
- `POST /api/logout`: 로그아웃
- `GET /api/export_chat`: 대화 내보내기
- `GET /api/admin/stats`: 관리자 통계
- `POST /api/admin/upload`: 문서 업로드/인제스트

## 5. DB 점검 결과

- DB: `UCSI_DB`
- 접근 가능한 컬렉션 수: `11`
- 주요 컬렉션: `UCSI`, `Hostel`, `UCSI_ MAJOR`, `USCI_SCHEDUAL`, `UCSI_FACILITY`, `UCSI_STAFF` 등

운영 원칙:

- MongoDB 원본 데이터를 임의 변경하지 않는다.
- 데이터 부재 시 모델 추측 대신 명시적 NO_DATA 응답을 우선한다.

## 6. 최신 검증 스냅샷

- Strict QA: `58/58` 통과 (`data/reports/strict_qa_report_latest.csv`)
- Stress Test: `300/300` 성공 (`data/reports/stress_test_report_latest.csv`)
- 평균 지연 `3.306s`, P95 `8.698s`

## 7. 미해결 이슈(다음 세션 우선)

1. 챗봇 열기 버튼 클릭 불가 이슈
- 사용자 체감 치명 이슈, P0 처리 필요

2. UI 표기/문구 일관성
- `UCSI buddy` 표기와 실제 화면 텍스트 통일 필요

3. UI 문자열 깨짐 및 가독성 저하
- 특수문자/인코딩 정리 필요

4. 회귀 자동화 부족
- 코드 변경 후 기본 QA 자동 실행 파이프라인 필요

## 8. 다음 세션 실행 순서(권장)

1. 프론트엔드 클릭 버그 재현 테스트 후 즉시 수정
2. 이름/브랜딩/문구 일괄 정리
3. QA 재실행(`scripts/qa/strict_qa_suite.py`, `scripts/qa/stress_test_runner.py`)
4. 결과를 `docs/PROJECT_STATUS_ANALYSIS.md`에 반영
5. 사용자 시나리오(로그인/2차 인증/NO_DATA) 수동 검증
