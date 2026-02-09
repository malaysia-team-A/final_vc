---
description: RAG 검색 파이프라인 운영 워크플로우
---

# RAG 파이프라인 워크플로우

## 1. 목표
- UCSI 도메인 질문에 대해 근거 기반 응답을 우선 제공
- 무근거 추측 대신 "정보 없음"으로 안전하게 처리

## 2. 처리 흐름
```text
질문 입력
  -> (선택) 캐시/피드백 기반 우선 응답
  -> 질의 분류(개인/도메인/일반)
  -> 도메인/필요 시 RAG 검색
  -> 신뢰도/관련성 점검
  -> AI 응답 생성 또는 NO_DATA 응답
  -> unanswered/feedback 로그 반영
```

## 3. RAG 인덱스 운영
- 시작 시 재색인: `rag_engine_async.index_mongodb_collections()`
- 파일 업로드 인제스트: `/api/admin/upload`
- 수동 재색인: `/api/admin/reindex`

## 4. 품질 게이트
- `has_relevant_data == false`: 보수 응답
- 저신뢰 문맥: 과도한 단정 표현 억제
- 내부 마커/시스템 텍스트 사용자 노출 금지

## 5. 점검 명령
```bash
python scripts/checks/check_server.py
python scripts/qa/strict_qa_suite.py --mode rag
python scripts/qa/stress_test_runner.py
```

## 6. 자주 발생하는 문제
1. 검색 누락
- 컬렉션/필드명 불일치
- 인덱스 재생성 필요

2. 과도한 NO_DATA
- 질의 키워드 정규화 부족
- preferred label 추론 튜닝 필요

3. 응답 일관성 저하
- 피드백 루프 데이터 품질 확인 필요
- BadQA/LearnedQA 누적 상태 점검 필요
