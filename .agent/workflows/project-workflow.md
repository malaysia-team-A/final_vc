---
description: UCSI 챗봇 프로젝트 개발/운영 기본 워크플로우
---

# UCSI 프로젝트 워크플로우

## 1. Setup
1. 가상환경 준비
```bash
python -m venv .venv
.venv\Scripts\activate
```
2. 의존성 설치
```bash
pip install -r requirements.txt
```
3. `.env` 구성
- `SECRET_KEY`
- `MONGO_URI`
- `GEMINI_API_KEY` (또는 `GOOGLE_API_KEY`)

## 2. Develop
1. 서버 실행
```bash
python main.py
```
2. 핵심 수정 위치
- API: `app/api/`
- 엔진: `app/engines/`
- UI: `static/site/`
- 관리자 UI: `static/admin/`

## 3. Test
```bash
python scripts/checks/check_server.py
python scripts/qa/strict_qa_suite.py --mode full
python scripts/qa/stress_test_runner.py
```

## 4. Deploy (로컬 기준)
- 기본 실행: `python main.py`
- 빠른 실행: `start_fastapi.bat` 또는 `start_chatbot.bat`
- 접속
  - `http://localhost:8000/`
  - `http://localhost:8000/admin`

## 5. Maintain
- 피드백/미응답 로그 확인: 관리자 통계 API
- 지식 파일 반영: `/api/admin/upload`
- 전체 재색인: `/api/admin/reindex`

## 일일 체크리스트
- 서버 정상 기동
- 인증/개인정보 게이트 동작 확인
- QA 기본 케이스 점검
- 문서 동기화 여부 확인
