# UCSI 버디 프로젝트 문서 허브

- 문서 기준일: 2026-02-08
- 서비스 표기명: `UCSI buddy`
- 챗봇 응답명: `Buddy`

## 1. 프로젝트 요약
이 프로젝트는 **UCSI 대학 도메인 RAG 챗봇**을 목표로 하며, 다음 2가지를 동시에 만족하도록 설계되어 있습니다.

1. 대학/학생 관련 질문은 DB·RAG 근거를 우선 사용한다.
2. 일반 대화/일반 상식 질문은 짧고 자연스럽게 답한다.

핵심 스택은 `Flask + Gemini + MongoDB Atlas + FAISS`입니다.

## 2. 핵심 원칙

- `RAG 우선`: UCSI 도메인 질문은 강제 RAG 라우팅.
- `DB 원본 유지`: MongoDB 원본 데이터를 그대로 사용(무단 seed 삽입 없음).
- `인증 분리`: 일반 로그인과 2차 인증(성적/GPA)을 분리.
- `NO_DATA 정책`: 도메인 질문인데 근거가 없으면 "DB에 정보가 없습니다" 계열 응답.
- `응답 UX`: 장황한 답변 대신 짧고 의도 중심으로 응답.

## 3. 문서 인덱스

- `docs/PROJECT_SPEC_V3_RAG_FIRST.md`: RAG 중심 제품 명세(v3)
- `docs/SYSTEM_WORKFLOW.md`: 실제 쿼리 처리 워크플로우와 알고리즘
- `docs/PROJECT_STATUS_ANALYSIS.md`: 최신 진행률/검증 결과/리스크
- `docs/HANDOVER.md`: 다음 세션용 인수인계 문서
- `docs/project_plan_extracted.txt`: `project_plan.pdf` 추출 텍스트(원문 보존용)

## 4. 빠른 실행

```bash
pip install -r requirements.txt
python main.py
```

- 사용자 UI: `http://localhost:5000/site/code_hompage.html`
- 관리자 UI: `http://localhost:5000/admin`

## 5. 현재 확인된 핵심 상태

- 백엔드 API 라우팅/인증/RAG 파이프라인은 동작함.
- MongoDB `UCSI_DB` 연결 및 컬렉션 조회 가능.
- 자동 QA/스트레스 테스트 스크립트와 리포트 파일 존재.
- 프론트엔드는 **클릭/문구/가독성 관련 잔여 버그**가 있어 추가 안정화 필요.

## 6. 오늘 기준 우선 과제

1. 챗봇 열기 버튼 클릭 불가 이슈 재현 및 수정
2. UI 문구/표기 통일 (`UCSI buddy`, `Buddy`)
3. 깨진 UI 문자열/가독성 개선
4. 회귀 테스트 재실행 후 결과 문서 갱신
