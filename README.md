# UCSI Buddy Chatbot

FastAPI + MongoDB + RAG(FAISS) + Google Gemini 기반 UCSI 대학 도메인 챗봇 프로젝트입니다.

**Version:** 3.2.0 | **Updated:** 2026-02-10

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

3. `.env` 작성 (`.env.example` 참고)
- 필수: `SECRET_KEY`, `MONGO_URI`, `GEMINI_API_KEY` (또는 `GOOGLE_API_KEY`), `ADMIN_PASSWORD`
- 선택: `GEMINI_MODEL`, `RAG_FAST_CONFIDENCE`, `RAG_REWRITE_TRIGGER`, `RAG_RERANK_TRIGGER` 등

4. 실행
```bash
python main.py
```

5. 접속
- 메인 UI: `http://localhost:8000/`
- 관리자 UI: `http://localhost:8000/admin`
- API 문서 (Swagger): `http://localhost:8000/docs`

## 프로젝트 구조
```
final/
├── main.py                    # FastAPI 메인 엔트리포인트
├── main_fastapi.py            # 호환용 진입점
├── requirements.txt           # Python 의존성
├── .env.example               # 환경변수 템플릿
│
├── app/                       # 애플리케이션 코어
│   ├── config.py              # 설정 관리 (환경변수 로딩)
│   ├── schemas.py             # Pydantic 모델 (Request/Response)
│   ├── extensions.py          # 인메모리 세션 저장소
│   │
│   ├── api/                   # FastAPI 라우터
│   │   ├── auth.py            # 인증 (login, verify_password, logout)
│   │   ├── chat.py            # 통합 채팅 API (하이브리드 분류기)
│   │   ├── chat_legacy.py     # 이전 chat.py 백업
│   │   ├── chat_helpers.py    # 채팅 헬퍼 함수
│   │   ├── admin.py           # 관리자 API (인덱스/모니터링/파일)
│   │   └── dependencies.py    # FastAPI 의존성 (JWT 검증)
│   │
│   ├── engines/               # 비즈니스 로직 엔진
│   │   ├── ai_engine_async.py       # LLM 엔진 (Gemini, 회로차단기, 속도제한)
│   │   ├── db_engine_async.py       # MongoDB 비동기 엔진 (Motor)
│   │   ├── rag_engine.py            # RAG 코어 (FAISS 인덱싱/검색)
│   │   ├── rag_engine_async.py      # RAG 비동기 래퍼
│   │   ├── intent_classifier.py     # 하이브리드 인텐트 분류기
│   │   ├── semantic_router_async.py # 벡터 기반 의도 라우팅
│   │   ├── response_validator.py    # 응답 검증/할루시네이션 방지
│   │   ├── index_manager.py         # 인덱스 수명주기 관리
│   │   ├── monitoring.py            # 성능 메트릭 수집
│   │   ├── query_rewriter.py        # 쿼리 최적화/확장
│   │   ├── reranker.py              # Cross-Encoder 재순위화
│   │   ├── language_engine.py       # 다국어 감지 (en/ko/zh)
│   │   ├── ux_engine.py             # UX 개선 (인사말, 에러, 제안)
│   │   ├── semantic_cache_engine.py # 시맨틱 응답 캐시
│   │   ├── faq_cache_engine.py      # FAQ 캐시 (빈도 기반)
│   │   └── unanswered_analyzer.py   # 미답변 질문 분석
│   │
│   ├── core/
│   │   └── session.py         # 세션 관리 (대화기록, 보안세션)
│   │
│   └── utils/
│       ├── auth_utils.py      # JWT/비밀번호 유틸리티
│       └── logging_utils.py   # 감사 로깅 (PII 마스킹)
│
├── static/                    # 프론트엔드
│   ├── site/
│   │   ├── code_hompage.html  # 메인 챗봇 UI
│   │   ├── css/styles.css     # 커스텀 스타일
│   │   └── js/app.js          # 클라이언트 로직
│   └── admin/
│       └── admin.html         # 관리자 대시보드
│
├── scripts/                   # 스크립트 모음
│   ├── tests/                 # 테스트 스위트
│   │   ├── run_all_tests.py   # 전체 테스트 러너
│   │   ├── test_integration.py
│   │   ├── test_security.py
│   │   └── test_ux.py
│   ├── qa/                    # QA 스위트
│   │   ├── strict_qa_suite.py
│   │   └── stress_test_runner.py
│   ├── checks/                # 환경 점검
│   ├── db/                    # DB 마이그레이션/점검
│   ├── debug/                 # 디버그 도구
│   └── run/                   # 실행 스크립트 (.bat)
│
├── test_cases/                # 테스트 데이터
│   ├── e2e_rag_regression.py
│   ├── rag_accuracy_tests.json
│   ├── rag_real_world_tests.json
│   ├── general_knowledge_tests.json
│   └── strict_auth_intent_tests.json
│
├── data/                      # 런타임 데이터
│   ├── knowledge_base/        # FAISS 인덱스 (faiss_index.bin, faiss_metadata.pkl)
│   ├── reference/             # 백업 및 참조 데이터
│   └── reports/               # QA/테스트 리포트 (CSV)
│
└── docs/                      # 프로젝트 문서
    ├── README.md              # 문서 허브
    ├── ARCHITECTURE_FASTAPI.md
    ├── SYSTEM_WORKFLOW.md
    ├── PROJECT_STATUS_ANALYSIS.md
    ├── HANDOVER.md
    └── ...
```

## 핵심 아키텍처

### 인텐트 분류 (3단계 하이브리드)
```
사용자 메시지
    │
    ▼
[1] Keyword Guard (즉시 분류)
    ├─ 개인정보 패턴 → personal
    ├─ 능력 질문 → capability
    ├─ UCSI 키워드 → ucsi_domain (RAG 강제)
    │
    ▼ (매칭 없음)
[2] Vector Search (SentenceTransformer, confidence ≥ 0.65)
    │
    ▼ (낮은 confidence)
[3] LLM Fallback (Gemini)
    └─ 최종 분류 결정
```

### 카테고리별 처리
| 카테고리 | 처리 방식 | 인증 필요 |
|---------|----------|----------|
| personal | MongoDB 학생 데이터 직접 조회 | JWT 필수, GPA는 2FA |
| ucsi_domain | RAG 검색 → LLM 응답 생성 | 불필요 |
| general_knowledge | LLM 직접 응답 (컨텍스트 없음) | 불필요 |
| capability | 하드코딩된 응답 | 불필요 |

### 안전장치
- **Prompt Injection 탐지**: 12개 패턴 탐지
- **입력 Sanitization**: HTML/스크립트 태그 제거, 길이 제한
- **응답 검증**: 숫자 검증, 할루시네이션 패턴 탐지, 소스 기반 grounding
- **RLHF 피드백 루프**: 긍정 피드백 → 학습 응답, 부정 피드백 → 가드 레일

### Rich Content (v3.2.0)
- **Staff 프로필 링크**: `profile_url` 클릭 시 해당 Staff 프로필 페이지로 이동
- **건물 이미지**: `BUILDING_IMAGE` (Google Drive) 채팅창 내 인라인 표시
- **프로그램 정보 링크**: `Url` 필드 기반 "More Information" 버튼
- **지도 링크**: `MAP` 필드 기반 "View on Map" 버튼
- **URL 자동 링크화**: 응답 텍스트 내 URL을 클릭 가능한 링크로 자동 변환

## API 요약

### 인증 (3)
| Method | Endpoint | 설명 |
|--------|---------|------|
| POST | `/api/login` | 학생 로그인 (학번 + 이름) |
| POST | `/api/verify_password` | 비밀번호 검증 (2FA, 10분) |
| POST | `/api/logout` | 로그아웃 |

### 채팅 (3)
| Method | Endpoint | 설명 |
|--------|---------|------|
| POST | `/api/chat` | 메인 채팅 |
| POST | `/api/feedback` | 피드백 제출 (positive/negative) |
| GET | `/api/export_chat` | 대화 내보내기 |

### 관리자 (12)
| Method | Endpoint | 설명 |
|--------|---------|------|
| POST | `/api/admin/login` | 관리자 로그인 |
| POST | `/api/admin/logout` | 관리자 로그아웃 |
| GET | `/api/admin/stats` | 통계 조회 |
| POST | `/api/admin/upload` | 문서 업로드 (PDF/TXT/CSV) |
| POST | `/api/admin/reindex` | MongoDB 재인덱싱 |
| GET | `/api/admin/files` | 파일 목록 |
| DELETE | `/api/admin/files` | 파일 삭제 |
| GET | `/api/admin/index/status` | 인덱스 상태 |
| POST | `/api/admin/index/reindex` | 수동 재인덱싱 |
| GET | `/api/admin/index/history` | 인덱싱 히스토리 |
| GET | `/api/admin/index/changes` | 데이터 변경 감지 |
| GET | `/api/admin/monitoring/dashboard` | 모니터링 대시보드 |
| GET | `/api/admin/unanswered/analysis` | 미답변 질문 분석 |
| GET | `/api/admin/unanswered/report` | 미답변 리포트 |
| GET | `/api/admin/health` | 헬스체크 (Public) |

## 실행 스크립트
- `scripts/run/start_fastapi.bat`: 기본 실행
- `scripts/run/start_chatbot.bat`: 기본 실행 별칭
- `scripts/run/start_standard.bat`: `py -3.13` 경로 실행

루트 `start_*.bat`, `fix_dependencies.bat`는 호환용 래퍼입니다.

## 점검/테스트 명령
```bash
# 환경 점검
python scripts/verify_setup.py
python scripts/checks/check_server.py

# 테스트 스위트
python scripts/tests/run_all_tests.py --verbose
python scripts/tests/test_integration.py --test all
python scripts/tests/test_security.py
python scripts/tests/test_ux.py

# QA 스위트
python scripts/qa/strict_qa_suite.py --mode full
python scripts/qa/stress_test_runner.py

# E2E 회귀 테스트
python test_cases/e2e_rag_regression.py
```

## 환경 변수 (.env)

### 필수
| 변수 | 설명 |
|-----|------|
| `SECRET_KEY` | JWT 서명 키 |
| `MONGO_URI` | MongoDB Atlas 연결 문자열 |
| `GEMINI_API_KEY` | Google Gemini API 키 |
| `ADMIN_PASSWORD` | 관리자 비밀번호 |

### 선택 (RAG 튜닝)
| 변수 | 기본값 | 설명 |
|-----|-------|------|
| `GEMINI_MODEL` | `gemma-3-27b-it` | LLM 모델명 |
| `RAG_FAST_CONFIDENCE` | `0.72` | RAG 고신뢰 임계값 |
| `RAG_REWRITE_TRIGGER` | `0.68` | 쿼리 재작성 임계값 |
| `RAG_RERANK_TRIGGER` | `0.82` | 재순위화 임계값 |
| `RAG_MAX_TOTAL_QUERIES` | `3` | 최대 검색 쿼리 수 |
| `SEMANTIC_ROUTER_ENABLED` | `true` | 시맨틱 라우터 활성화 |
| `AUTO_REINDEX_ENABLED` | `false` | 자동 재인덱싱 |
| `REINDEX_INTERVAL_HOURS` | `6` | 재인덱싱 간격 (시간) |

## 프로젝트 문서
상세 문서는 `docs/` 디렉토리를 참조하세요:
- `docs/README.md`: 문서 허브 (전체 목록)
- `docs/ARCHITECTURE_FASTAPI.md`: 아키텍처 상세
- `docs/SYSTEM_WORKFLOW.md`: 워크플로우
- `docs/PROJECT_STATUS_ANALYSIS.md`: 현재 상태/리스크
- `docs/HANDOVER.md`: 인수인계 문서
- `docs/PROJECT_SPEC_V3_RAG_FIRST.md`: 제품/품질 스펙
