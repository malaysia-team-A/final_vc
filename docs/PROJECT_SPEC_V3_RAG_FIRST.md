# UCSI AI Chatbot Project Spec v3.0 (RAG-First)

- Date: 2026-02-08
- Base: `project_plan.pdf` (2주/40시간 목표, UCSI DB 통합, 자기개선 루프)
- Current Runtime: `main.py` + Flask + MongoDB Atlas + FAISS + Gemini

## 1) 제품 목표

이 프로젝트의 핵심은 **RAG 기반 대학 도메인 챗봇**이다. 일반 상식 응답도 제공하지만, 제품 신뢰성의 우선순위는 다음과 같다.

1. UCSI 도메인 질문은 반드시 RAG/DB 근거를 우선한다.
2. 개인정보/성적 질문은 인증 단계를 정확히 분리한다.
3. 답변은 짧고 명확하게 유지한다(UX 가독성).

## 2) 범위 정의 (project_plan 반영)

- General Query: 인증 없이 학교 일반 정보 응답
- Personal Query: 로그인 후 개인 프로필 정보 응답
- Sensitive Query: 성적/GPA는 2차 인증 후에만 응답
- Self-learning (프로젝트 범위 내):
  - 미응답/저신뢰 질문 로깅
  - 피드백 기반 개선
  - 관리자 문서 업로드 후 벡터화

## 3) 현재 아키텍처

- API Layer: `main.py`
- Intent/Safety/LLM: `app/engines/ai_engine.py`
- Retrieval Core: `app/engines/rag_engine.py`
- DB Access: `app/engines/db_engine.py`, `app/engines/data_engine.py`
- Cache/Rewriter/Reranker: `semantic_cache_engine.py`, `query_rewriter.py`, `reranker.py`
- Frontend: `static/site/code_hompage.html`

### RAG 워크플로우

1. 캐시 조회 (semantic/FAQ)
2. 의도 라우팅 (general vs domain vs personal)
3. 보안 게이트 (guest/login/2FA)
4. RAG 검색
- FAISS + MongoDB 컬렉션 검색
- 쿼리 리라이트 + 리랭크(조건부)
5. grounded 응답 생성/후처리
- 환각성 표현 차단
- UX 길이 압축

## 4) DB 연결 및 스키마 현황 (실측)

`UCSI_DB` 접속 기준, 컬렉션 9개 확인:

- `UCSI` (500)
- `Hostel` (7)
- `UCSI_ MAJOR` (117)
- `USCI_SCHEDUAL` (86)
- `UCSI_FACILITY` (5)
- `UCSI_HOSTEL_FAQ` (4)
- `UCSI_STAFF` (17)
- `UCSI_University_Blocks_Data` (8)
- `Feedback` (1)

확인된 데이터 갭:

- `UCSI_FACILITY`에 `library` 항목 없음
- `UCSI_STAFF`에 `Vice Chancellor` role 항목 없음

## 5) 성공 기준 (QA Scoring Rubric)

단순 200 응답률이 아니라, 아래를 동시에 만족해야 성공으로 본다.

1. Intent Correctness
- 질문 의도와 라우팅이 일치하는가
- 예: 일반상식이 DB 경로로 과잉 라우팅되지 않는가

2. Security Correctness
- guest 상태에서 personal/sensitive 정보 차단
- login + no-2FA 상태에서 GPA 차단

3. Grounded Correctness
- 도메인 답변이 근거 문맥에 기반하는가
- 허구 질문에 대해 일관된 거절을 하는가

4. UX Quality
- 답변이 과도하게 길지 않은가
- 사용자가 즉시 행동 가능한 형태인가

## 6) 현재 검증 결과 (2026-02-08, latest)

- Strict QA: `50/50 (100%)`
- Report: `data/reports/strict_qa_report_latest.csv`

- Stress Test (5 workers, 300 queries):
- HTTP Success: `300/300 (100%)`
- Semantic Pass: `300/300 (100%)`
- Avg latency: `3.505s`
- P50 latency: `2.325s`
- P95 latency: `7.651s`
- Max latency: `16.634s`
- Report: `data/reports/stress_test_report_latest.csv`

### DB Integrity Policy

- MongoDB 원본 데이터를 그대로 사용하며, 런타임/검증 과정에서 신규 seed 삽입은 수행하지 않는다.
- 현재 확인된 컬렉션 크기:
  - `UCSI_FACILITY`: 5
  - `UCSI_STAFF`: 17

## 7) 핵심 문제점과 개선 우선순위

P0 (즉시)

1. 데이터 커버리지 보강
- library 운영시간, Vice Chancellor 정보를 원천 DB에 추가
- 현재는 "정보 부재"를 정확히 알리도록 처리되어 있음

2. 스키마 표준화
- 컬렉션 명 오탈자/공백 정리 (`USCI_SCHEDUAL`, `UCSI_ MAJOR`)
- alias 매핑 계층 도입

P1 (품질 상향)

1. Retrieval 품질 고도화
- Hybrid retrieval (sparse + dense)
- entity-aware 검색(직책/직함 정규화)

2. 평가 자동화 강화
- CI에서 strict QA + 회귀 테스트 자동 실행
- semantic correctness 평가셋 확장

P2 (UX/운영)

1. UX 개선
- 응답 길이 adaptive 정책(질문 타입별)
- 근거 출처 요약 노출(사용자 신뢰성 향상)

2. 관측성 개선
- query trace id + retrieval trace 저장
- 실패 유형 대시보드화

## 8) 외부 레퍼런스 기반 설계 포인트

아래 자료의 공통 패턴을 본 프로젝트에 반영한다.

- RAG 기본 구조/정합성: Lewis et al., 2020
- https://arxiv.org/abs/2005.11401

- Self-reflection 기반 retrieval 개선: Self-RAG, 2023
- https://arxiv.org/abs/2310.11511

- Retrieval 오류 교정 전략: CRAG, 2024
- https://arxiv.org/abs/2401.15884

- 실무형 retrieval 구현 예시 (OpenAI Cookbook)
- https://cookbook.openai.com/examples/question_answering_using_embeddings

- Contextual Retrieval (Anthropic)
- https://www.anthropic.com/engineering/contextual-retrieval

- 운영 평가 프레임워크
- RAGAS: https://docs.ragas.io/
- LangSmith Evaluation: https://docs.langchain.com/langsmith/evaluation-concepts

## 9) 다음 개발 단계 (즉시 실행 계획)

1. DB 정비
- library + leadership seed 데이터 추가
- 스키마 alias 테이블 도입

2. 검색 고도화
- 직책 질의(`Vice Chancellor`, `Dean`) 전용 retriever 분기 추가
- 시설 운영시간 질의 전용 retriever 분기 추가

3. QA 고도화
- strict QA에 "의도-근거 일치" 수동 라벨셋 추가
- stress test에 semantic pass/fail 판정 추가
