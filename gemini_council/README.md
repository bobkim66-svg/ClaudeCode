# gemini_council — 멀티 AI 구조화 토론 오케스트레이터

역할이 다른 AI 4명(제안자 → 비평가 → 대안 설계자 → 사회자)이 라운드 단위로 토론하고,
사회자가 마지막 줄에 `[DISCUSSION_COMPLETE]` 를 내거나 최대 라운드·사용량 한도·오류가 나면
`logs/discussion_YYYYMMDD_HHMMSS.md` 회의록을 남기고 종료한다.

현재 **1단계**: 순수 로직 + mock 백엔드 + CLI (외부 연결 없음, 표준 라이브러리만 사용).

## 실행 (Ubuntu 22.04, bash)

```bash
cd gemini_council
python3 -m pip install -r requirements-dev.txt
python3 -m pytest -q
python3 run_council.py --topic "AUV 군집의 통신 두절 시 재집결 전략" --rounds 5 --mock-end-round 3
```

## 실행 (Windows 11, PowerShell)

```powershell
cd gemini_council
python -m pip install -r requirements-dev.txt
python -m pytest -q
python run_council.py --topic "AUV 군집의 통신 두절 시 재집결 전략" --rounds 5 --mock-end-round 3
```

## 구조

| 파일 | 역할 |
|---|---|
| `council/config.py` | 실행 파라미터 dataclass |
| `council/roles.py` | 역할 정의, seed 프롬프트 (종료 토큰은 사회자에게만 알려줌) |
| `council/protocol.py` | 순수 로직: 델타 버퍼, 턴 프롬프트 조립, 종료 토큰 판정 |
| `council/backends/base.py` | 백엔드 인터페이스 (`stateful` 여부, `send()`, 한도/타임아웃 예외) |
| `council/backends/mock.py` | 시드 고정 가짜 응답기 (종료·한도·타임아웃·오류 재현) |
| `council/orchestrator.py` | 메인 루프, 일시정지/중단, 모든 종료 경로에서 회의록 저장 |
| `council/transcript.py` | 회의록 마크다운 생성·저장 (UTF-8, 덮어쓰기 방지) |

## 맥락 전달 방식 (입력 길이 일정 유지)

- stateful 백엔드(웹 탭 등 대화를 스스로 기억): 그 역할이 마지막으로 말한 뒤의 **새 발언만** 보낸다.
- stateless 백엔드(API/CLI 단발 호출): **seed + 사회자 최신 요약 + 그 이후 발언**만 보낸다.

두 방식 모두 라운드가 늘어도 턴당 입력 길이가 일정하다 (`tests/test_protocol.py::test_prompt_length_bounded`).
