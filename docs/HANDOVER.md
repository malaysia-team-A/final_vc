# 프로젝트 인수인계서 (Handover Document)

**작성일**: 2026-02-06
**프로젝트**: UCSI University AI Chatbot
**버전**: 2.4
**상태**: 최종 MVP 완성 (파일 구조 개선 및 한글화 완료)

---

## 1. 프로젝트 개요
본 프로젝트는 **Flask** 백엔드와 **Google Gemini** 모델을 기반으로 한 대학 학사 행정 챗봇 시스템입니다.

### 핵심 기술
- **AI 엔진**: Google GenAI SDK (Gemini 3 / Gemma 3)
- **RAG**: FAISS 벡터 검색 + MongoDB 컬렉션 검색 통합
- **인증**: JWT + 이중 비밀번호 인증
- **프론트엔드**: TailwindCSS, Vanilla JS

---

## 2. 최근 개선 사항 (v2.4 완료)

### 핵심 백엔드 고도화 (Phase 1)
- ✅ **FAQ 캐싱 엔진**: 자주 묻는 질문 자동 캐싱 (즉시 응답, 비용 절감)
- ✅ **다국어 지원**: 한/영/중 자동 감지 및 맞춤형 응답
- ✅ **미응답 질문 관리**: 답변 못한 질문 로깅 → 관리자 대시보드와 연동
- ✅ **피드백 학습**: 👍 피드백 시 FAQ 자동 등록

### 데이터 조회 확장 (Phase 2)
- ✅ **학기/입학 정보 조회**: 학생 프로필에 Semester, Intake 정보 추가 제공
- ✅ **교직원 검색**: "Professor", "Staff" 등 키워드로 MongoDB 교직원 DB 검색

### 관리자 기능 (Phase 3)
- ✅ **Admin 대시보드**: `/admin` 경로에 Glassmorphism UI 제공
- ✅ **통계 시각화**: 피드백 만족도, 미응답 질문 수 실시간 조회
- ✅ **문서 업로드**: 드래그 앤 드롭으로 RAG 지식 베이스 업데이트

### 사용자 편의 (Phase 4)
- ✅ **대화 내보내기**: `/api/export_chat` 엔드포인트로 대화 기록 다운로드

### 코드 정리 및 최적화 (Phase 5)
- ✅ **파일 구조 재편**: `app/engines`, `data`, `static` 등으로 구조화
- ✅ **코드 리팩토링**: 엑셀 의존성 제거, 불필요 파일(`learning_engine.py`) 삭제
- ✅ **문서 한글화**: 모든 기술 문서 한국어 번역 완료

---

## 3. 실행 및 테스트

### 필수 요구사항
- Python 3.10+
- MongoDB Atlas 계정
- Google API Key

### 설치 및 실행
```bash
pip install -r requirements.txt
python main.py
```
또는 `start_chatbot.bat` 실행

### 테스트 시나리오
1. **일반 대화**: "안녕", "너 누구야?" → 즉시 응답
2. **학생 정보**: 로그인 → "내 정보 알려줘" → 개인 정보 확인
3. **GPA 조회**: 로그인 → UNLOCK → "내 GPA 알려줘" → GPA 값 표시
4. **기숙사 정보**: "기숙사 가격 알려줘" → MongoDB에서 실시간 조회

---

## 4. 파일 구조 (v2.4 기준)

| 경로 | 역할 |
|------|------|
| `main.py` | Flask 서버 진입점 |
| `app/engines/` | 핵심 엔진 (`ai_engine.py`, `rag_engine.py` 등) |
| `app/utils/` | 유틸리티 (`auth_utils.py` 등) |
| `data/` | 데이터 파일 (JSON 설정, RAG 인덱스) |
| `static/site/` | 사용자 웹 페이지 HTML/JS |
| `static/admin/` | 관리자 대시보드 HTML |
| `docs/` | 프로젝트 문서 (한글) |

---

## 5. 트러블슈팅

| 증상 | 해결 방법 |
|------|-----------|
| GPA 조회 안됨 | UNLOCK 버튼으로 2차 인증 필요 |
| 피드백 버튼 안 눌림 | 이벤트 위임 방식으로 수정됨 |
| 마크다운 표시됨 | 백엔드에서 자동 제거 처리됨 |
| 추천 질문이 generic | 스마트 추천 로직 추가됨 |

---

**Note**: 이 문서는 v2.4 (최종 리팩토링 및 한글화) 기준으로 작성되었습니다.
