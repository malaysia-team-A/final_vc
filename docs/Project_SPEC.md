# 📋 프로젝트 명세서: UCSI AI 챗봇 (Kai) v2.4

## 1. 소개
본 문서는 UCSI 대학교 학생들을 위한 AI 챗봇 **"Kai"**의 기술 명세서입니다.  
버전 2.4에서는 **멀티소스 RAG** 시스템과 **관리자 대시보드**, **대화 내보내기** 기능을 통합하여 사용자 편의성과 관리 효율성을 극대화했습니다.

---

## 2. 시스템 아키텍처

### 2.1 기술 스택
| 구분 | 기술 |
|------|------|
| 언어 | Python 3.10+ |
| 웹 프레임워크 | Flask (RESTful API) |
| LLM 엔진 | Google Gemini (gemma-3-27b-it) |
| 벡터 DB | FAISS (CPU 최적화) |
| 데이터베이스 | MongoDB Atlas |
| 임베딩 | HuggingFace (all-MiniLM-L6-v2) |
| 프론트엔드 | HTML5, TailwindCSS, Vanilla JS |

### 2.2 핵심 모듈 구조 (`app/engines/`)

**AI Engine (`ai_engine.py`)**
- 사용자 의도 분류 (일반 대화 vs 개인 데이터 조회)
- Google GenAI SDK 직접 통합
- 스마트 추천 질문 생성 알고리즘

**RAG Engine (`rag_engine.py`)**
- FAISS 문서 벡터 검색 (Faiss Index)
- MongoDB 6개 컬렉션 실시간 통합 검색:
  - 기숙사(Hostel), 시설(Facility), 기숙사 FAQ
  - 학사 일정(Schedule), 전공(Major), 교직원(Staff)

**Data Engine (`data_engine.py` & `db_engine.py`)**
- MongoDB 연결 풀링 및 쿼리 최적화
- 대소문자 무시 검색(Aggregation Pipeline)

**Security Module (`app/utils/auth_utils.py`)**
- 이중 인증(Dual Auth): 성적 접근 시 비밀번호 재확인
- JWT 기반 세션 관리 (Stateless)

---

## 3. 기능 요구사항

### 3.1 챗봇 페르소나 (Kai)
- **이름**: Kai
- **역할**: 대학교 생활을 돕는 친근하고 에너지 넘치는 조교
- **행동 수칙**:
  - 학교 정보는 반드시 DB/RAG 근거 데이터 활용
  - 일상 대화는 자연스럽게 대응하되 짧고 간결하게
  - 모르는 정보는 정중히 모른다고 답변

### 3.2 주요 기능 명세
| 기능 | 설명 | 인증 수준 |
|------|------|------|
| **일반 문의** | 장학금, 수강신청, 시설 위치, 학사 일정 확인 | 불필요 |
| **개인 정보** | 본인 학번, 학과, 담당 교수 조회 | 로그인 필요 |
| **민감 정보** | GPA, CGPA, 상세 성적표 조회 | **이중 인증** 필요 |
| **피드백** | 답변에 대한 좋아요/싫어요 평가 및 의견 | 불필요 |
| **내보내기** | 현재 대화 세션 기록을 파일로 다운로드 | 로그인 필요 |

---

## 4. 데이터베이스 스키마 (MongoDB)

### 4.1 UCSI_DB 컬렉션
| 컬렉션명 | 용도 |
|--------|------|
| `UCSI` | 학생 기본 정보 (학번, 이름, GPA, 비밀번호 등) |
| `Hostel` | 기숙사 정보 (가격, 건물명, 시설 옵션) |
| `UCSI_FACILITY` | 캠퍼스 시설 안내 및 운영 시간 |
| `UCSI_HOSTEL_FAQ` | 기숙사 관련 자주 묻는 질문 |
| `USCI_SCHEDUAL` | 학사 일정 및 주요 행사 |
| `UCSI_ MAJOR` | 전공 소개 및 학비 정보 |
| `UCSI_STAFF` | 교직원 연락처 및 소속 |
| `Feedback` | 사용자 피드백 로그 |
| `UCSI_University_Blocks_Data` | 캠퍼스 지도 및 위치 데이터 |

---

## 5. 배포 및 테스트
- **운영 체제**: Windows (로컬 개발) / Linux (서버 배포 권장)
- **API 키 관리**: `.env` 파일을 통한 보안 관리
- **실행**: `python main.py` 실행 후 `check_system.py`로 무결성 검증

---

**버전 히스토리**
- v2.0: 초기 MVP 출시
- v2.1: Gemma 3 모델 도입
- v2.2: Gemini Flash 전환 및 최적화
- v2.3: 멀티소스 RAG, UI 개선, 스마트 추천
- v2.4: **FAQ 캐싱, 다국어 지원, 관리자 대시보드, 대화 내보내기, 파일 구조 재편** (현재)
