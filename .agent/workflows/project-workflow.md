---
description: UCSI 챗봇 프로젝트 전체 개발/운영 워크플로우
---

# UCSI Chatbot 프로젝트 워크플로우

## 프로젝트 라이프사이클

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. SETUP          2. DEVELOP        3. TEST         4. DEPLOY         │
│  ──────────────────────────────────────────────────────────────────────│
│  환경 설정    ──▶  기능 개발    ──▶  검증      ──▶  운영        ──▶    │
│  의존성 설치       코드 수정         테스트 실행     서버 시작          │
│  DB 연결           RAG 구축          버그 수정       모니터링           │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        5. MAINTAIN
                        ─────────────
                        피드백 수집
                        FAQ 업데이트
                        성능 개선
```

---

## 1️⃣ SETUP - 환경 설정

// turbo-all

### 1.1 프로젝트 클론/이동
```bash
cd c:\Users\leejb\Desktop\final
```

### 1.2 가상환경 생성 (권장)
```bash
python -m venv venv
venv\Scripts\activate
```

### 1.3 의존성 설치
```bash
pip install -r requirements.txt
```

### 1.4 환경변수 설정 (.env)
```
MONGODB_URI=mongodb+srv://...
GOOGLE_API_KEY=AIza...
ADMIN_PASSWORD=...
```

### 1.5 MongoDB 연결 확인
```bash
python -c "from app.engines.db_engine import db_engine; print(f'Connected: {db_engine.connected}')"
```

### 1.6 텍스트 인덱스 생성 (최초 1회)
```bash
python create_text_index.py
```

---

## 2️⃣ DEVELOP - 개발

### 2.1 서버 개발 모드 시작
```bash
python main.py
```

### 2.2 주요 파일 구조
```
final/
├── main.py                    # API 서버 (수정 시 자동 재시작)
├── app/engines/
│   ├── ai_engine.py           # AI 로직 수정
│   ├── rag_engine.py          # 검색 로직 수정  
│   ├── db_engine.py           # DB 쿼리 수정
│   ├── semantic_cache_engine.py
│   ├── query_rewriter.py      # 쿼리 확장 규칙
│   └── reranker.py
├── data/knowledge_base/       # RAG 문서 추가
└── static/site/               # 프론트엔드 수정
```

### 2.3 새 문서 추가 (RAG)
1. PDF/TXT 파일을 `data/knowledge_base/`에 복사
2. 관리자 API로 인제스트: `POST /api/admin/upload`
3. 또는 수동: `rag_engine.ingest_file("path/to/file.pdf")`

### 2.4 FAQ 추가
```python
from app.engines.faq_cache_engine import faq_cache
faq_cache.add_faq("질문", "답변", ["추천1", "추천2"])
```

---

## 3️⃣ TEST - 검증

### 3.1 빠른 테스트
```bash
python test_start.py
```

### 3.2 100개 질문 테스트
```bash
python test_chatbot_100.py
```

### 3.3 300개 스트레스 테스트
```bash
python stress_test_runner.py
```

### 3.4 특정 기능 테스트
```bash
# RAG 검색 테스트
python -c "from app.engines.rag_engine import rag_engine; print(rag_engine.search('Block A'))"

# Semantic Cache 테스트
python app/engines/semantic_cache_engine.py

# Query Rewriter 테스트
python app/engines/query_rewriter.py
```

### 3.5 로그 확인
```bash
type server.log
```

---

## 4️⃣ DEPLOY - 배포

### 4.1 프로덕션 서버 시작
```bash
start_chatbot.bat
```

### 4.2 수동 시작
```bash
python main.py
```

### 4.3 접속 확인
- API: http://localhost:5000
- 챗봇 UI: http://localhost:5000/site/code_hompage.html
- 관리자: http://localhost:5000/admin

---

## 5️⃣ MAINTAIN - 유지보수

### 5.1 피드백 확인
```bash
# 부정 피드백 조회
python -c "from app.engines.faq_cache_engine import unanswered_manager; print(unanswered_manager.get_unresolved())"
```

### 5.2 학습된 응답 확인
관리자 대시보드: `/admin` → 통계/피드백 탭

### 5.3 캐시 정리
```python
from app.engines.semantic_cache_engine import semantic_cache
semantic_cache.clear()
```

### 5.4 인덱스 재구축
```bash
# FAISS 인덱스 삭제 후 재시작
del data\knowledge_base\faiss_index.bin
del data\knowledge_base\faiss_metadata.pkl
python main.py
```

---

## 📋 일일 운영 체크리스트

- [ ] 서버 정상 작동 확인
- [ ] 로그에 에러 없는지 확인
- [ ] 부정 피드백 검토 및 대응
- [ ] API 응답 시간 모니터링

---

## 🔧 문제 해결

| 증상 | 원인 | 해결 |
|------|------|------|
| 서버 시작 안됨 | 포트 사용 중 | `netstat -ano \| findstr :5000` |
| DB 연결 실패 | .env 설정 오류 | MONGODB_URI 확인 |
| 검색 결과 없음 | 인덱스 미생성 | `python create_text_index.py` |
| AI 응답 없음 | API 키 문제 | GOOGLE_API_KEY 확인 |
| 환각 발생 | 신뢰도 임계값 | CONFIDENCE_THRESHOLD 조정 |
