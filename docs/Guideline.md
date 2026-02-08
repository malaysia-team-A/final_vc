# 코드 협업 가이드라인 (Code Collaboration Guidelines)

모든 AI 에이전트 및 팀원은 코드 충돌 방지 및 원활한 버전 관리를 위해 아래 규칙을 **절대적으로 준수**해야 합니다.

## 1. 수정 표기 (Modification Marking)
- 기존 코드를 수정할 때, 단순한 마킹보다는 **변경 이유**를 주석으로 명확히 남기십시오.
- **Good Example**: 
  ```python
  # [MODIFIED] 몽고DB 연결 타임아웃 해결을 위해 리트라이 로직 추가
  connect(retry_writes=True)
  ```
- **Bad Example**:
  ```python
  # 변경됨
  connect()
  ```

## 2. 추가 원칙 (Append Strategy)
- **충돌 방지**를 위해 새로운 독립 함수나 클래스는 **파일의 맨 마지막**에 추가하는 것을 원칙으로 합니다.
- **예외 사항 (Exceptions)**:
  - 새로운 라이브러리 `import` 구문은 반드시 **파일 최상단**에 작성하십시오.
  - Python 스크립트의 경우, 메인 실행 블록(`if __name__ == "__main__":`)은 항상 **파일 최하단**에 유지되어야 하므로, 그 바로 윗부분에 새 함수를 추가하십시오.

## 3. 포맷팅 제한 (No Global Formatting)
- 수정하지 않는 영역의 코드는 절대 건드리지 마십시오. (전체 파일 자동 포맷팅 금지)
- 변경 사항은 오직 수정이 필요한 로직 부분에만 국소적으로 적용되어야 합니다. 이는 `git diff`를 최소화하여 리뷰와 병합을 용이하게 하기 위함입니다.

## 4. 레거시 보존 (Legacy Preservation)
- 삭제 여부가 확실하지 않은 코드는 바로 지우지 말고 주석 처리(`comment out`) 하십시오.
