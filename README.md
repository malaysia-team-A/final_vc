# UCSI University AI 챗봇 프로젝트

## 개요
**Flask** 기반의 대학 AI 챗봇 시스템입니다. **RAG (검색 증강 생성)** 기술을 활용하여 정확한 학사 정보를 제공하며, **이중 인증** 시스템을 통해 성적과 같은 민감한 개인 정보를 안전하게 보호합니다.

## 주요 기능
- **AI 채팅**: Google Gemini 연동 + 단일 호출 최적화, 다국어(한/영/중) 자동 지원
- **스마트 FAQ**: 자주 묻는 질문 캐싱 및 사용자 피드백 기반 자동 학습
- **멀티소스 RAG**: PDF 문서 + MongoDB(학생/교수/학기 정보) 통합 검색
- **보안 시스템**: JWT 기반 인증 + 2차 비밀번호(성적 조회 시), 대화 기록 내보내기 기능
- **관리자 기능**: 대시보드 UI, 미응답 질문 관리, PDF 문서 업로드, 피드백 통계 시각화

## 빠른 시작 (Quick Start)

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 설정 (.env)
프로젝트 루트에 `.env` 파일을 생성하고 다음 정보를 입력하세요:
- `GOOGLE_API_KEY`: Google AI API 키
- `MONGO_URI`: MongoDB Atlas 연결 문자열
- `SECRET_KEY`: JWT 암호화 키

### 3. 서버 실행
```bash
python main.py
```
또는 `start_chatbot.bat` 파일을 실행하세요.

- **사용자 접속**: `http://localhost:5000/site/code_hompage.html`
- **관리자 접속**: `http://localhost:5000/admin`

## 파일 구조 (File Structure)
| 경로 | 설명 |
|------|------|
| `main.py` | Flask 메인 서버 및 API 엔드포인트 |
| `app/engines/` | 핵심 엔진 모듈 (AI, DB, RAG, 피드백 등) |
| `app/utils/` | 유틸리티 모듈 (인증, 로깅) |
| `data/` | 데이터 파일 (DB, Config, RAG 저장소) |
| `docs/` | 프로젝트 문서 |
| `static/` | 정적 리소스 (웹페이지, 관리자 페이지) |

## 문서 (Documents)
- **상세 인수인계 (Handover)**: [HANDOVER.md](HANDOVER.md)
- **기술 명세서 (Spec)**: [Project_SPEC.md](Project_SPEC.md)
- **구현 계획 (Plan)**: [Implementation_Plan.md](../../../.gemini/antigravity/brain/d06d4b6d-a518-4376-afb7-6c3aad018f90/implementation_plan.md)
