# 스크립트 정리 가이드

프로젝트 루트 정리를 위해 실행/점검 스크립트를 아래처럼 분리했습니다.

- `scripts/checks/`: 시스템/환경 점검 스크립트
- `scripts/db/`: MongoDB 구조/인덱스/데이터 점검 스크립트
- `scripts/qa/`: QA, 스트레스 테스트, 검증 스크립트
- `scripts/debug/`: 디버그/재시작용 배치 파일 및 보조 스크립트

주요 실행 예시:

```bash
python scripts/checks/check_system.py
python scripts/qa/strict_qa_suite.py --mode full
python scripts/qa/stress_test_runner.py
```
