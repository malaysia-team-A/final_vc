# 스크립트 가이드

`scripts/` 폴더는 실행, 점검, DB 보조작업, QA, 디버그 도구를 분리 관리합니다.

## 폴더 구조
- `scripts/run/`: 실행 런처
- `scripts/checks/`: 환경/서버/구성 점검
- `scripts/db/`: DB 인덱스/데이터 점검
- `scripts/qa/`: 정합성 QA/부하 테스트/벤치마크
- `scripts/debug/`: 로컬 디버깅 보조

## 자주 쓰는 명령
```bash
python scripts/verify_setup.py
python scripts/checks/check_python.py
python scripts/checks/check_server.py
python scripts/qa/strict_qa_suite.py --mode full
python scripts/qa/stress_test_runner.py
```

## 실행 배치
- `scripts/run/start_fastapi.bat`
- `scripts/run/start_chatbot.bat`
- `scripts/run/start_standard.bat`

## 출력 위치
- QA 리포트: `data/reports/`
  - `strict_qa_report_latest.csv`
  - `stress_test_report_latest.csv`

## 주의사항
- 일부 스크립트는 로컬 `.env`와 실행 중 서버(`http://localhost:8000`)를 전제로 동작합니다.
- 생성 산출물은 `.gitignore`에서 제외되며 필요 시 별도 백업 후 정리하세요.
