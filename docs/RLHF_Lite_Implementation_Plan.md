# 📋 RLHF Lite 피드백 적용 기능 구현 계획서

> **작성일**: 2026-02-06  
> **목적**: 사용자 피드백을 AI 답변 생성에 반영하여 지속적인 성능 개선 구현

---

## 1. 📊 현황 분석

### A. 현재 시스템 상태

| 구성요소 | 파일 | 상태 |
|----------|------|------|
| 피드백 저장 | `feedback_engine.py` | ✅ 구현 완료 |
| MongoDB 저장 | `db_engine.py` | ✅ 구현 완료 (Feedback 컬렉션) |
| 유사 질문 검색 | - | ❌ **미구현** |
| 프롬프트 주입 | `ai_engine.py` | ❌ **미구현** |

### B. 현재 피드백 데이터 구조 (MongoDB `Feedback` 컬렉션)

```json
{
    "id": 1,
    "timestamp": "2026-02-06T12:00:00",
    "session_id": "abc123",
    "user_message": "학비가 얼마야?",
    "ai_response": "컴퓨터공학과 학비는...",
    "rating": "positive",  // 또는 "negative"
    "comment": "정확해요!"
}
```

### C. 기존 FunctionTech_Plan.md 가이드라인 요약

1. **저장**: 👍/👎 피드백을 질문-답변-점수와 함께 저장 ✅ (완료)
2. **검색**: 새 질문과 **유사한 과거 질문**을 검색 ❌ (미구현)
3. **주입**: 검색된 좋은/나쁜 예시를 **System Prompt에 Context로 주입** ❌ (미구현)

---

## 2. 🎯 구현 목표

사용자가 질문을 할 때:
1. DB에서 **유사한 과거 질문-답변 쌍**을 검색
2. **Positive 피드백** 받은 답변 → "좋은 예시"로 프롬프트에 추가
3. **Negative 피드백** 받은 답변 → "피해야 할 예시"로 프롬프트에 추가

---

## 3. 🔧 구현 계획

### Step 1: `feedback_engine.py` 확장 - 유사 질문 검색 메서드 추가

**추가할 메서드**: `get_related_examples(current_query: str) -> Dict`

```python
def get_related_examples(self, current_query: str, limit: int = 3) -> Dict[str, List[Dict]]:
    """
    현재 질문과 유사한 과거 피드백 예시를 검색
    
    Returns:
        {"good": [...], "bad": [...]}
    """
    # 1. 로컬 JSON 또는 MongoDB에서 피드백 로드
    # 2. 키워드 기반 유사도 매칭 (초기 버전: 단순 키워드 매칭)
    # 3. positive/negative 분류하여 반환
```

**유사도 검색 방식 옵션**:
| 방식 | 난이도 | 정확도 | 권장 |
|------|--------|--------|------|
| 키워드 매칭 | 낮음 | 보통 | ✅ 1차 구현 |
| TF-IDF | 중간 | 좋음 | 2차 개선 |
| 벡터 임베딩 | 높음 | 최상 | 3차 고도화 |

### Step 2: `ai_engine.py` 수정 - 피드백 Context 주입

**수정할 메서드**: `process_message()`

**변경 내용**:
1. `feedback_engine`에서 유사 예시 조회
2. `qa_template` 프롬프트에 피드백 컨텍스트 추가

```python
# 추가할 프롬프트 섹션
feedback_context = """
📌 Reference Examples from Past Feedback:
✅ Good answers to similar questions:
{good_examples}

❌ Avoid these mistakes from past answers:
{bad_examples}
"""
```

### Step 3: 프라이버시 필터 추가 (권장)

`save_feedback()` 호출 시 **개인정보(학번, 전화번호 등) 포함 여부 체크**

```python
import re

def has_pii(text: str) -> bool:
    patterns = [
        r'\b\d{10,}\b',           # 학번/전화번호 (10자리 이상 숫자)
        r'\b\d{3}-\d{4}-\d{4}\b'  # 전화번호 형식
    ]
    return any(re.search(p, text) for p in patterns)
```

---

## 4. 📁 수정 파일 목록

| 파일 | 작업 | 우선순위 |
|------|------|----------|
| `app/engines/feedback_engine.py` | `get_related_examples()` 메서드 추가 | 🔴 높음 |
| `app/engines/ai_engine.py` | 프롬프트에 피드백 Context 주입 로직 추가 | 🔴 높음 |
| `app/engines/db_engine.py` | MongoDB에서 피드백 검색 메서드 추가 (선택) | 🟡 중간 |

---

## 5. ⚠️ 주의사항 (가이드라인 준수)

1. **수정 표기**: 모든 변경에 `# [MODIFIED]` 주석과 변경 이유 명시
2. **추가 원칙**: 새 함수는 파일 마지막(`if __name__` 직전)에 추가
3. **포맷팅 제한**: 수정하지 않는 영역은 건드리지 않음
4. **레거시 보존**: 기존 로직 삭제 시 주석 처리 후 진행

---

## 6. 📅 구현 순서

```
[Phase 1] feedback_engine.py - get_related_examples() 추가
    ↓
[Phase 2] ai_engine.py - process_message()에 피드백 조회 및 주입 로직 추가
    ↓
[Phase 3] 테스트 및 검증
    ↓
[Phase 4] (선택) PII 필터, MongoDB 기반 검색으로 고도화
```

---

## 7. ✅ 예상 결과

**Before (현재)**:
```
User: 등록금이 얼마야?
AI: (피드백 무관하게 일반 답변 생성)
```

**After (구현 후)**:
```
User: 등록금이 얼마야?
AI: (과거 유사 질문의 좋은 답변 참조 + 나쁜 답변 회피하여 답변 생성)
```

---

## 8. 🤔 결정 필요 사항

구현 전 확인이 필요한 사항:

1. **유사도 검색 방식**: 초기에는 키워드 매칭으로 시작? 아니면 바로 고급 방식?
2. **데이터 소스**: 로컬 JSON 우선? MongoDB 우선?
3. **피드백 최소 개수**: 몇 개 이상의 피드백이 쌓여야 적용할지?

---

**작성 완료. 구현을 진행할까요?**
