# UCSI Buddy: 링크 자동 제공 기능(Link Injection) 통합 가이드

이 문서는 교수진(Staff) 및 전공(Major) 질문 시 관련 상세 보기 링크를 답변에 자동으로 포함하는 기능을 팀 프로젝트에 통합하기 위한 기술 명세서입니다.

## 1. 개요
- **기술명**: 컨텍스트 기반 링크 주입 (Context-based Link Injection)
- **목적**: RAG(Retrieval-Augmented Generation)를 통해 조회된 메타데이터 중 URL 정보를 AI가 인식하여 답변 끝에 HTML/Markdown 링크로 자동 변환하여 제공함.
- **대상 데이터**: 
  - `UCSI_STAFF` (교수진): `profile_url` 필드
  - `UCSI_ MAJOR` (전공): `Url` 필드

---

## 2. 구현 방식

### [A] RAG 엔진: 메타데이터 추출 및 제한 완화
RAG 엔진은 검색 결과에서 단순 텍스트뿐만 아니라 **URL 필드**를 명시적으로 추출하여 AI에게 전달되는 'Context' 문자열에 포함해야 합니다.

*   **주요 수정 파일**: `app/engines/rag_engine.py`
*   **수정 내용**:
    1.  **데이터 필드 추가**: `UCSI_MAJOR` 검색 필드에 `Url`을 추가하고, `UCSI_STAFF` 포맷팅 로직에 `profile_url`을 포함함.
    2.  **검색 범위 확대**: 다수의 교수/전공 노출을 위해 `limit` 설정을 상향(3 -> 15)하고, AI에게 전달되는 최종 결과 수를 확대(5 -> 20)함.

### [B] AI 엔진: 조건부 링크 생성 지침 (Prompt Engineering)
AI는 전달받은 Context 내에 특정 키워드(`Url:`, `profile_url:`)가 포착될 경우에만 링크를 생성하도록 지침을 받습니다.

*   **주요 수정 파일**: `app/engines/ai_engine_async.py` (또는 `ai_engine.py`)
*   **프롬프트 지침 예시**:
    ```text
    5. CLICKABLE LINKS: 
       - If Context contains "Url:" or "profile_url:", you MUST include the link in Markdown format [상세 보기](URL) at the end of YOUR response.
       - If the response is in Korean, use exactly "[상세 보기](URL)".
       - ABSOLUTELY NO PLACEHOLDERS: NEVER use [상세 보기](URL) if no URL is provided in Context.
    ```

---

## 3. 핵심 코드 수정 내역 (Diff 요약)

### RAG Engine (`rag_engine.py`)
```python
# 전공 검색 필드 보강
domains = [
    ('UCSI_MAJOR', ['Programme', 'Fields of Study', ..., 'Url'], 'Programme'),
]

# 교수진 프로필 URL 추출
if member.get("profile_url"):
    details_parts.append(f"profile_url: {member.get('profile_url')}")

# 글로벌 결과 제한 확대
for text, score, source in results_with_scores[:20]: # 5에서 20으로 확대
    ...
```

### AI Engine (`ai_engine_async.py`)
```python
prompt = """
...
- If Context contains "Url:" or "profile_url:", you MUST include the link in Markdown format [상세 보기](URL) at the end of YOUR response.
...
"""
```

---

## 4. 통합 시 주의사항
1.  **데이터 정합성**: MongoDB의 각 도큐먼트에 `Url` 또는 `profile_url` 필드가 존재해야 링크가 생성됩니다.
2.  **토큰 관리**: 검색 제한을 20개로 확대했으므로, AI 모델의 Context Window(토큰 제한)를 확인해야 합니다. (Gemma-3 또는 Gemini 1.5 계열에서는 충분함)
3.  **다국어 지원**: 한국어 답변 시 명확하게 `[상세 보기]` 텍스트를 사용하도록 지침을 명시해야 UX가 일관됩니다.
