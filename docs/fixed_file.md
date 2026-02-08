# 🛠️ 프로젝트 변경 사항 정리 (Project Change Log)

> **작성일**: 2026-02-08  
> **내용**: RLHF Lite(피드백 기반 자가 학습) 기능 구현 및 DB 컬렉션 통일 작업

---

## 1. `app/engines/db_engine.py`

- **변경된 내용**: MongoDB 컬렉션 명칭 통일 (`feedbacks` → `Feedback`)
    - **위치**: `save_feedback` 메서드 (Line 291)
    - **내용**: `self.db.feedbacks.insert_one` → `self.db.Feedback.insert_one`
    - **위치**: `get_feedback_stats` 메서드 (Line 299-301)
    - **내용**: `self.db.feedbacks.count_documents` → `self.db.Feedback.count_documents`

- **변경된 내용**: 키워드 기반 피드백 검색 메서드 추가
    - **위치**: 파일 하단 (Line 318-344)
    - **내용**: `search_feedback_by_keywords(self, keywords, limit)` 메서드 구현. 입력된 키워드로 `Feedback` 컬렉션에서 유사한 질문/답변을 검색.

## 2. `app/engines/feedback_engine.py`

- **변경된 내용**: 정규식 모듈 Import 추가
    - **위치**: 파일 상단 (Line 7)
    - **내용**: `import re`

- **변경된 내용**: 유사 피드백 예시 조회 메서드 추가
    - **위치**: 파일 하단 (Line 120-153)
    - **내용**: `get_related_examples(self, current_query)` 메서드 구현. 사용자 질문에서 키워드를 추출하고, `db_engine`을 통해 검색된 결과를 Good/Bad 예시로 분류하여 반환.

## 3. `app/engines/ai_engine.py`

- **변경된 내용**: Feedback Engine 모듈 Import 추가
    - **위치**: 파일 상단 (Line 6)
    - **내용**: `from .feedback_engine import feedback_engine`

- **변경된 내용**: 프롬프트 템플릿에 피드백 Context 플레이스홀더 추가
    - **위치**: `qa_template` 정의 부분 (Line 43)
    - **내용**: `{feedback_context}` 추가

- **변경된 내용**: 답변 생성 시 피드백 반영 로직 추가
    - **위치**: `process_message` 메서드 내부 (Line 90-108)
    - **내용**: 
        1. `feedback_engine.get_related_examples()` 호출하여 유사 피드백 조회
        2. 조회된 Good/Bad 예시를 문자열로 포맷팅
        3. `qa_template.format()` 호출 시 `feedback_context` 파라미터에 주입
