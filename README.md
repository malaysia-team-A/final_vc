# UCSI Buddy Chatbot

FastAPI + MongoDB + RAG(FAISS) 기반 UCSI 대학 도메인 챗봇 프로젝트입니다.

## 빠른 시작
1. 가상환경 생성/활성화
```bash
python -m venv .venv
.venv\Scripts\activate
```

2. 의존성 설치
```bash
pip install -r requirements.txt
```

3. `.env` 작성
- `SECRET_KEY`
- `MONGO_URI`
- `GEMINI_API_KEY` (또는 `GOOGLE_API_KEY`)

4. 실행
```bash
python main.py
```

## 실행 스크립트
- `scripts/run/start_fastapi.bat`: 기본 실행
- `scripts/run/start_chatbot.bat`: 기본 실행 별칭
- `scripts/run/start_standard.bat`: `py -3.13` 경로 실행

루트 `start_*.bat`, `fix_dependencies.bat`는 호환용 래퍼입니다.

## 주요 경로
- API: `app/api/`
- 엔진: `app/engines/`
- 정적 UI: `static/site/`
- 관리자 UI: `static/admin/`
- 점검/QA 스크립트: `scripts/`
- 프로젝트 문서: `docs/`

## 점검 명령
```bash
python scripts/verify_setup.py
python scripts/checks/check_server.py
python scripts/qa/strict_qa_suite.py --mode full
python scripts/qa/stress_test_runner.py
```

## 라우팅(분기) 방식
- 기본 정책: `semantic router` 우선 + `rule` 최소 폴백
- Semantic router는 MongoDB 컬렉션(`semantic_intents`)의 임베딩 예시를 기준으로 질의 의도를 분류합니다.
- 고신뢰 의도는 분기에 바로 반영하고, 저신뢰(`SEMANTIC_ROUTER_MIN_CONFIDENCE`)는 기존 룰/LLM 흐름으로 폴백합니다.
- 보안 경로(로그인 필요, GPA/성적 접근)는 룰 검증을 유지합니다.

