# gemini_council — 멀티 AI 구조화 토론 오케스트레이터

역할이 다른 AI 4명(제안자 → 비평가 → 대안 설계자 → 사회자)이 라운드 단위로 토론하고,
사회자가 마지막 줄에 `[DISCUSSION_COMPLETE]` 를 내거나 최대 라운드·사용량 한도·오류가 나면
`logs/discussion_YYYYMMDD_HHMMSS.md` 회의록(사용량 표 포함)을 남기고 종료한다.

- 1단계: 순수 로직 + mock 백엔드 + CLI
- 2단계: **Gemini CLI 백엔드** (Google 공식 도구를 헤드리스로 호출, 브라우저 자동화 없음)

Python 코드는 표준 라이브러리만 사용한다. Gemini CLI 는 Node.js 20 이상이 필요하다.

## Gemini CLI 준비 (최초 1회, Ubuntu 22.04)

Ubuntu 22.04 기본 apt 의 nodejs 는 v12 라서 쓸 수 없다. nvm 등으로 Node 20+ 를 설치한다.

```bash
node --version
```

```bash
npm install -g @google/gemini-cli
```

```bash
gemini
```

마지막 명령은 대화형으로 열린다. 'Login with Google' 을 골라 브라우저에서 로그인한 뒤 `/quit` 로 나온다.
로그인 정보는 `~/.gemini/` 에 저장되어 이후 헤드리스 호출에 그대로 쓰인다.

연결 확인 (fast/thinking 모델에 1번씩 짧은 질문):

```bash
cd gemini_council && python3 run_council.py --backend gemini-cli --check
```

토론 실행:

```bash
python3 run_council.py --backend gemini-cli --topic "AUV 군집의 통신 두절 시 재집결 전략" --rounds 3
```

Windows 11 (PowerShell) 도 같은 명령이며 `python3` 대신 `python` 을 쓴다.
`gemini` 를 못 찾으면 `--gemini-cmd "C:\Users\<사용자>\AppData\Roaming\npm\gemini.cmd"` 처럼 경로를 준다.

### 안전장치

- `--yolo` 를 쓰지 않는다. 헤드리스 모드에서는 승인이 필요한 도구(셸 실행, 파일 쓰기/수정)가 CLI 정책상 자동 거부된다.
- 매 호출을 빈 임시 폴더에서 실행해 프로젝트 파일과 `GEMINI.md` 가 맥락에 섞이지 않게 한다. 단, 사용자 전역 `~/.gemini/GEMINI.md` 는 CLI 가 읽는다.
- 호출 사이 최소 간격 `--min-interval`(기본 2 s)을 둔다.
- 한도 초과(`TerminalQuotaError`/`RetryableQuotaError`)를 감지하면 회의록을 저장하고 종료한다(종료 코드 2).
- `pro` 를 요청했는데 pro 모델이 쓰이지 않으면 '강등 가능성' 경고를 남긴다.

### 모델

기본값은 CLI 별칭이다: 비평가 `pro`, 나머지 `flash`. CLI 가 계정 권한에 맞는 실제 모델로 바꿔 준다.
`--model-fast`, `--model-thinking` 으로 바꿀 수 있고, 빈 문자열(`""`)을 주면 CLI 기본 모델을 쓴다.

## mock 으로 실행 / 테스트 (Ubuntu 22.04, bash)

```bash
cd gemini_council
python3 -m pip install -r requirements-dev.txt
python3 -m pytest -q
python3 run_council.py --topic "AUV 군집의 통신 두절 시 재집결 전략" --rounds 5 --mock-end-round 3
```

## mock 으로 실행 / 테스트 (Windows 11, PowerShell)

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
| `council/backends/gemini_cli.py` | Gemini CLI 헤드리스 호출, 출력 파싱, 한도/인증/타임아웃 분류, 사용량 표 |
| `tests/fake_gemini.py` | 실제 CLI 의 입출력 규약을 흉내 낸 가짜 CLI (로그인 없이 테스트) |
| `council/orchestrator.py` | 메인 루프, 일시정지/중단, 모든 종료 경로에서 회의록 저장 |
| `council/transcript.py` | 회의록 마크다운 생성·저장 (UTF-8, 덮어쓰기 방지) |

## 맥락 전달 방식 (입력 길이 일정 유지)

- stateful 백엔드(웹 탭 등 대화를 스스로 기억): 그 역할이 마지막으로 말한 뒤의 **새 발언만** 보낸다.
- stateless 백엔드(API/CLI 단발 호출): **seed + 사회자 최신 요약 + 그 이후 발언**만 보낸다.

두 방식 모두 라운드가 늘어도 턴당 입력 길이가 일정하다 (`tests/test_protocol.py::test_prompt_length_bounded`).
