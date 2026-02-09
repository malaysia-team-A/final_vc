# UCSI 버디 챗봇 문서 허브

- 문서 기준일: 2026-02-09
- 기준 코드: `main.py`, `app/api/*.py`, `app/engines/*_async.py`
- 기본 실행 포트: `8000`

## 1. 프로젝트 개요
이 프로젝트는 UCSI 대학 도메인 질문을 우선 처리하는 RAG 기반 챗봇입니다.  
현재 서버는 FastAPI 비동기 구조로 동작하며 인증/개인정보 질의/일반 질의/관리자 기능이 API로 분리되어 있습니다.

## 2. 현재 아키텍처
- 백엔드: FastAPI + Uvicorn
- 인증: JWT (`/api/login`, `/api/verify_password`, `/api/logout`)
- 데이터 계층: MongoDB Atlas + Motor(비동기)
- RAG: FAISS 인덱스 + MongoDB 컬렉션 인덱싱
- LLM: Google GenAI SDK (`google-genai`)
- 프런트엔드: `static/site/code_hompage.html` + `static/site/js/app.js`

## 3. API 요약
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

## 4. 실행 방법
1. 가상환경 생성 및 활성화
```bash
python -m venv .venv
.venv\Scripts\activate
```
2. 의존성 설치
```bash
pip install -r requirements.txt
```
3. 환경 변수 설정 (`.env`)
- 필수: `SECRET_KEY`, `MONGO_URI`, `GEMINI_API_KEY`(또는 `GOOGLE_API_KEY`)
- 선택: `GEMINI_MODEL`, `RAG_*` 튜닝 변수
4. 서버 실행
```bash
python main.py
```
5. 접속
- 메인 UI: `http://localhost:8000/`
- 관리자 UI: `http://localhost:8000/admin`
- Swagger: `http://localhost:8000/docs`

## 5. QA/검증 스크립트
- 환경 검증: `python scripts/verify_setup.py`
- 런타임 검증: `python scripts/checks/check_python.py`
- 서버 체크: `python scripts/checks/check_server.py`
- 정합성 QA: `python scripts/qa/strict_qa_suite.py --mode full`
- 부하 테스트: `python scripts/qa/stress_test_runner.py`

## 6. 운영 주의사항
- 서버 시작 시 `index_mongodb_collections()`가 실행되어 MongoDB 기반 RAG 인덱스를 재구축합니다.
- 관리자 업로드(`POST /api/admin/upload`)는 저장 후 즉시 인덱싱을 수행합니다.
- 현재 코드 기준으로 관리자 API에 별도 인증 가드가 없습니다. 운영 환경에서는 인증/권한 제어를 우선 적용해야 합니다.
- `main_fastapi.py`는 호환용 진입점이며 실제 기준 실행은 `main.py`입니다.

## 7. 문서 목록
- `docs/SYSTEM_WORKFLOW.md`: 요청 처리 워크플로우
- `docs/PROJECT_STATUS_ANALYSIS.md`: 최신 상태 및 지표
- `docs/PROJECT_SPEC_V3_RAG_FIRST.md`: 제품/품질 스펙
- `docs/Project_SPEC.md`: 시스템 기술 명세 요약
- `docs/IMPLEMENTATION_PLAN_V2.md`: 구현 로드맵
- `docs/HANDOVER.md`: 인수인계 포인트
- `docs/Guideline.md`: 협업 가이드라인
- `docs/FunctionTech_Plan.md`: 기능 확장 계획
- `docs/project_plan_extracted.txt`: 기획안 추출 정리본
- `docs/assets/project_plan.pdf`: 원본 기획안 PDF
