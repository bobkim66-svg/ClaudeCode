# gemini_council — 멀티 AI 구조화 토론 오케스트레이터

역할이 다른 AI 4명(제안자 → 비평가 → 대안 설계자 → 사회자)이 라운드 단위로 토론하고,
사회자가 마지막 줄에 `[DISCUSSION_COMPLETE]` 를 내거나 최대 라운드에 도달하면
`logs/discussion_YYYYMMDD_HHMMSS.md` 회의록(사용량 표 포함)을 남기고 종료한다.

| 모드 | 무엇을 쓰나 | 사람 | 비고 |
|---|---|---|---|
| **manual (기본, 반자동)** | 내 계정의 Gemini 웹 탭 4개 (학교 Gemini Pro 그대로) | 매 차례 복사·붙여넣기 | 웹 화면 자동 조작 없음 |
| gemini-cli (무인) | Google 공식 Gemini CLI | 불필요 | 개인 계정 필요 (학교 Workspace 계정 불가) |
| mock | 가짜 응답 | 불필요 | 오프라인 시험용 |

작동 원리(Gemini 를 어떻게 쓰는지, 그림 포함): [docs/PIPELINE.md](docs/PIPELINE.md)

## 반자동 모드 사용법 (Windows 11)

1. `gemini_council.exe` 를 받는다: GitHub **Actions → gemini-council → 최근 성공 실행 → Artifacts → gemini-council-windows-exe**
   (처음 실행 시 "Windows의 PC 보호" 가 뜨면 **추가 정보 → 실행**. 서명하지 않은 exe 라서 뜨는 경고)
2. Chrome 에서 학교 계정으로 Gemini **새 채팅 탭 4개**를 열고 순서대로 제안자 / 비평가 / 대안 설계자 / 사회자로 정한다.
   비평가 탭은 Pro(사고 모드) 권장.
3. `gemini_council.exe` 를 **더블클릭** → 토론 주제, 토론 유형(1: 자유 토론, 2: 연구 주제 검토), 최대 라운드(기본 3),
   차례 사이 대기 시간(기본 5분) 입력. 3라운드 = 12차례라 대기만 약 55분 (+ 답변 시간).
   대기 중에는 다음 메시지를 클립보드에 넣지 않으며 1분마다 남은 시간을 보여 준다.
4. 매 차례 화면 안내대로:
   - `▶ [탭 2 · 비평가] 에 붙여 넣으세요` → 그 탭에서 **Ctrl+V, Enter**
   - 답변이 끝나면 답변 맨 아래 아이콘 줄(좋아요·싫어요·다시 생성·**복사**·더보기)의 4번째 **복사 아이콘** 클릭 → 프로그램이 감지해 다음 차례로 넘어감 (프로그램 창 조작 불필요)
5. 명령: `r` + Enter = 보낼 메시지 다시 복사, `n` + Enter = 대기 건너뛰기, `q` + Enter = 지금까지 저장하고 종료.
   회의록은 exe 가 있는 폴더의 `logs\` 에 저장된다.

주의: 기다리는 동안 다른 글을 복사하면 그 글이 답변으로 처리된다. 보낸 메시지나 직전 답변을 다시 복사한 경우는
감지해서 경고만 하고 계속 기다린다.

명령줄로 실행할 때 (PowerShell):

```powershell
.\gemini_council.exe --topic "AUV 군집의 통신 두절 시 재집결 전략" --rounds 3 --interval 5 --agenda topic-review
```

## 무인 모드 (gemini-cli): 개인 Google 계정이 있을 때

### 준비 (최초 1회, Windows 11 PowerShell)

1. Node.js LTS 설치 (Gemini CLI 는 Node 20 이상 필요). 설치 후 **PowerShell 을 새로 연다.**

```powershell
winget install OpenJS.NodeJS.LTS
```

```powershell
node --version
```

2. Gemini CLI 설치

```powershell
npm install -g @google/gemini-cli
```

3. 로그인: 아래를 실행해 **"Sign in with Google"** 을 고르고 브라우저에서 로그인한 뒤 `/quit` 로 나온다.
   (`gemini` 만 치면 "스크립트를 실행할 수 없으므로…" 오류가 날 수 있어 `gemini.cmd` 로 실행한다.)

```powershell
gemini.cmd
```

4. 연결 확인 (fast/thinking 모델에 짧은 질문 1번씩). `확인(fast) 응답`, `확인(thinking) 응답` 이 나오면 성공.

```powershell
.\gemini_council.exe --backend gemini-cli --check
```

### 무인 토론 실행 (Windows)

```powershell
.\gemini_council.exe --backend gemini-cli --topic "AUV 군집의 통신 두절 시 재집결 전략" --rounds 3
```

회의록은 실행한 폴더의 `logs\discussion_YYYYMMDD_HHMMSS.md` 에 저장된다.
exe 대신 소스로 실행하려면 `gemini_council` 폴더에서 `python run_council.py ...` (같은 옵션).

### exe 받기 / 만들기

- 받기: GitHub 저장소 **Actions → gemini-council → 최근 성공한 실행 → Artifacts → gemini-council-windows-exe**
- 직접 빌드 (Python 3.10+ 필요, 결과 `dist\gemini_council.exe`):

```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

### Ubuntu 22.04 에서 쓸 때

apt 의 nodejs 는 v12 라 쓸 수 없다. nvm 으로 Node 22 를 설치한 뒤 `npm install -g @google/gemini-cli`, `gemini` 로 로그인한다.
실행은 `python3 run_council.py --backend gemini-cli ...`. 공용 PC 라면 로그인 토큰(`~/.gemini/oauth_creds.json`)을
다른 사용자가 쓸 수 있으므로 본인 전용 리눅스 계정에서 쓰거나, 사용 후 그 파일을 지운다.

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
python3 run_council.py --backend mock --topic "AUV 군집의 통신 두절 시 재집결 전략" --rounds 5 --mock-end-round 3
```

## mock 으로 실행 / 테스트 (Windows 11, PowerShell)

```powershell
cd gemini_council
python -m pip install -r requirements-dev.txt
python -m pytest -q
python run_council.py --backend mock --topic "AUV 군집의 통신 두절 시 재집결 전략" --rounds 5 --mock-end-round 3
```

## 구조

| 파일 | 역할 |
|---|---|
| `council/config.py` | 실행 파라미터 dataclass |
| `council/roles.py` | 역할 정의, seed 프롬프트 (종료 토큰은 사회자에게만 알려줌) |
| `council/protocol.py` | 순수 로직: 델타 버퍼, 턴 프롬프트 조립, 종료 토큰 판정 |
| `council/backends/base.py` | 백엔드 인터페이스 (`stateful` 여부, `send()`, 한도/타임아웃 예외) |
| `council/backends/manual.py` | 반자동 중계: 클립보드에 프롬프트 복사, 답변 복사 감지, 오복사 경고 |
| `council/clipboard.py` | OS 클립보드 (Windows: Win32 API 직접, Linux: xclip 등) |
| `council/console.py` | 클립보드 감시 중 r/q 명령 입력 |
| `council/backends/mock.py` | 시드 고정 가짜 응답기 (종료·한도·타임아웃·오류 재현) |
| `council/backends/gemini_cli.py` | Gemini CLI 헤드리스 호출, 출력 파싱, 한도/인증/타임아웃 분류, 사용량 표 |
| `tests/fake_gemini.py` | 실제 CLI 의 입출력 규약을 흉내 낸 가짜 CLI (로그인 없이 테스트) |
| `tests/test_platform.py` | npm 식 래퍼(gemini.cmd) 경유 호출, 타임아웃 시 손자 프로세스 종료 |
| `build.ps1` | Windows 단일 exe 빌드 + 단독 실행 점검 |
| `council/orchestrator.py` | 메인 루프, 일시정지/중단, 모든 종료 경로에서 회의록 저장 |
| `council/transcript.py` | 회의록 마크다운 생성·저장 (UTF-8, 덮어쓰기 방지) |

## 맥락 전달 방식 (입력 길이 일정 유지)

- stateful 백엔드(웹 탭 등 대화를 스스로 기억): 그 역할이 마지막으로 말한 뒤의 **새 발언만** 보낸다.
- stateless 백엔드(API/CLI 단발 호출): **seed + 사회자 최신 요약 + 그 이후 발언**만 보낸다.

두 방식 모두 라운드가 늘어도 턴당 입력 길이가 일정하다 (`tests/test_protocol.py::test_prompt_length_bounded`).
