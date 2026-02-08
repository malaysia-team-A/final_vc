# UCSI 버디 시스템 워크플로우

- 기준일: 2026-02-08
- 기준 코드: `main.py`, `app/engines/*.py`

## 1. 전체 흐름

```text
사용자 질문 입력
  -> 즉시 처리 규칙 검사(이름/스몰톡/노이즈)
  -> 라우팅 결정(force RAG / 일반 경로 / 개인정보 경로)
  -> (개인정보) 인증 게이트 확인
  -> (비개인) RAG/DB 검색 + 캐시/리라이트/리랭크
  -> 답변 생성 + 후처리(NO_DATA, 환각 방지, 길이 조정)
  -> 추천질문 정규화 + 피드백 버튼 노출
  -> 대화 로그 및 통계 기록
```

## 2. 단계별 알고리즘

### 2.1 입력 즉시 처리

- 이름 질문: 언어 감지 후 짧게 응답
- 물리 행동형 질문: 재치 있는 고정 응답
- 노이즈/무의미 입력: 재질문 유도 응답

이 단계는 불필요한 LLM 호출을 줄이고 UX 반응성을 높이기 위한 빠른 경로입니다.

### 2.2 라우팅

핵심 함수:

- `should_force_rag(user_message)`
- `check_personal_intent(user_message, search_term)`

규칙:

1. 대학 도메인 키워드가 감지되면 강제 RAG
2. 개인정보 의도면 인증 경로
3. 그 외는 일반 경로(필요 시 LLM 단독 응답)

### 2.3 개인정보/민감정보 게이트

핵심 정책:

- 로그인 전: 개인정보 응답 금지
- 로그인 후: 기본 개인정보 일부 허용
- 성적/GPA/CGPA: 2차 인증 완료 시에만 허용

관련 함수:

- `build_student_context(..., include_sensitive=False)`
- `/api/login`
- `/api/verify_password`

### 2.4 비개인 RAG 경로

핵심 함수:

- `retrieve_non_personal_context(user_message)`
- `rag_engine.search(query, preferred_labels=...)`

실행 순서:

1. Staff/Faculty 직접 조회
2. 학사 일정/요약 정보 조회
3. RAG 검색(Fast path)
4. 신뢰도 낮으면 Heavy path 실행
- query rewrite
- 확장 질의 재검색
- rerank
5. 필요 시 hostel 전용 DB fallback

### 2.5 응답 생성 및 안전 후처리

핵심 함수:

- `postprocess_grounded_response(...)`
- `check_hallucination_rules(...)`
- `enforce_general_keyword_coverage(...)`

정책:

- 컨텍스트가 `NO_DATA`면 추정/단정 답변 금지
- 가격/날짜 등 사실값은 근거가 있을 때만 제시
- 과도한 장문 대신 짧은 의도 맞춤 응답 유지

### 2.6 추천질문/피드백 루프

- 추천질문은 `_normalize_suggestions()`로 사용자 질문형 문장으로 정규화
- 피드백은 `/api/feedback`로 저장
- 긍정 피드백은 FAQ 캐시 학습에 활용
- 부정 피드백은 개선 대상 로그로 적재

## 3. NO_DATA 정책(중요)

도메인 질문인데 DB/RAG 근거가 없으면 다음 원칙을 지킵니다.

1. "정보가 없습니다"를 명확히 표시
2. 사실을 추정하지 않음
3. 필요하면 질문 범위를 좁혀 재질문하도록 유도

## 4. 성능/품질 관찰 지표

- 정합성: 의도 라우팅 정확도
- 보안성: 인증 단계별 차단/허용 정확도
- 근거성: 도메인 응답의 DB/RAG 근거 일치율
- UX: 응답 길이, 버튼 클릭 성공률, 재질문 없이 해결되는 비율

## 5. 현재 병목

1. 프론트엔드 클릭/문구 안정성 이슈
2. 컬렉션 명명 불일치로 인한 유지보수 비용
3. 일부 질문에서 일반 대화와 도메인 라우팅 경계가 모호한 케이스

## 6. 다음 개선 방향

1. UI 이벤트 안정화(클릭/레이어/버튼 상태)
2. RAG 신뢰도 임계값 구간별 응답 템플릿 개선
3. 도메인별 평가셋 확장으로 회귀 검증 강화
