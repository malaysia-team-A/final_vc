# UCSI Buddy: 링크 자동 제공 기능(Link Injection) 통합 가이드

이 문서는 교수진(Staff) 및 전공(Major) 질문 시 관련 상세 보기 링크를 답변에 자동으로 포함하는 기능을 팀 프로젝트에 통합하기 위한 상세 기술 명세서입니다.

## 1. 개요
- **기술명**: 컨텍스트 기반 링크 주입 (Context-based Link Injection)
- **목표**: RAG 엔진이 DB에서 조회한 URL 정보를 AI 엔진에게 전달하고, AI가 이를 인식하여 답변 끝에 Markdown 형식의 링크를 자동 생성함.
- **핵심 수정 파일**:
  1. `app/engines/rag_engine.py`: 데이터 추출 및 검색 제한 해제 로직
  2. `app/engines/ai_engine_async.py`: AI 링크 생성 프롬프트 지침

---

## 2. RAG 엔진 수정 상세 (`rag_engine.py`)

기존에 검색 결과 수와 리스트 출력 수를 제한하던 로직을 확대하고, 링크 필드를 명시적으로 포함하도록 수정되었습니다.

### [A] 검색 및 출력 제한 확대
더 많은 교수진과 전공 정보를 수용하기 위해 모든 `limit` 설정을 상향했습니다.

```python
# 1. 특정 역할(교수 등) 직접 검색 제한 확대 (L415)
def _search_staff_role_direct(self, db, query: str, limit: int = 15) -> list: # 3 -> 15

# 2. 컬렉션 스마트 검색 결과 제한 확대 (L885)
def smart_search_collection(..., limit: int = 15) -> list: # 3 -> 15

# 3. 최종 AI 전달 컨텍스트 결과 수 확대 (L1152)
for text, score, source in results_with_scores[:20]: # 5 -> 20
```

### [B] 링크(URL) 데이터 포함 로직
AI가 링크를 생성할 수 있도록 검색 대상 필드와 포맷팅 로직에 URL을 추가했습니다.

```python
# 1. 전공(Major) 검색 필드에 'Url' 추가 (L1049)
domains = [
    ('UCSI_MAJOR', ['Programme', 'Fields of Study', ..., 'Url'], 'Programme'),
]

# 2. 교수진(Staff) 데이터 추출 시 'profile_url' 추가 (L1088)
elif k == "staff_members" and isinstance(v, list):
    for member in v: # v[:3] 제한 제거
        ...
        m_profile_url = member.get("profile_url")
        if m_profile_url:
            details_parts.append(f"profile_url: {m_profile_url}")
```

---

## 3. AI 엔진 수정 상세 (`ai_engine_async.py`)

AI가 Context 내의 URL 정보를 감지했을 때의 행동 지침을 프롬프트에 추가했습니다.

### [A] 링크 생성 프롬프트 규칙 (L185 부근)
```text
5. CLICKABLE LINKS: 
   - If Context contains "Url:" or "profile_url:", you MUST include the link in Markdown format [상세 보기](URL) at the end of YOUR response.
   - If the response is in Korean, use exactly "[상세 보기](URL)".
   - If multiple URLs are present, list them clearly with descriptions or at the end of relevant points.
   - Example: "...정보는 여기서 확인할 수 있습니다. [상세 보기](https://example.com)"
6. ABSOLUTELY NO PLACEHOLDERS: NEVER use [상세 보기](URL) if no URL is provided in Context.
```

---

## 4. 통합 시 체크리스트
1.  **DB 필드명**: MongoDB 도큐먼트 내 필드명이 `Url` 및 `profile_url`과 일치하는지 확인하십시오.
2.  **데이터 무결성**: 링크가 없는 데이터에 대해 AI가 가짜 링크를 생성하지 않도록 프롬프트 6번 규칙을 반드시 준수해야 합니다.
3.  **검색 성능**: 결과 수를 20개로 늘렸으나 대규모 리스트 조회 시 약간의 지연이 발생할 수 있으므로, AI 모델의 처리 속도를 모니터링하십시오.
