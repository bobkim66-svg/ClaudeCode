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

## Deep Research 대기열 (`gemini_research.exe`)

주제 여러 개를 Gemini **Deep Research** 로 차례로 돌리고 보고서를 `research\날짜_시각_주제.md` 로 저장한다.
토론 자동 모드와 같은 전용 Chrome 프로필을 쓰므로 로그인은 한 번이면 된다.

1. `gemini_research.exe` 더블클릭 → 기본 모드(1: 일반, 2: 사고) → 주제 사이 간격(Enter = 12분 고정).
2. 주제 목록 `research_topics.txt` (exe 폴더) 가 없으면 콘솔에서 한 줄씩 입력받아 만든다. 형식:
   ```
   사고 | IR·LiDAR 융합 지각 신뢰도 기반 로컬 플래너 전환 선행연구
   일반 | 열화상 카메라 깊이 추정 방법 비교
   ```
   모드를 생략하면 기본 모드. **Pro 는 Deep Research 를 지원하지 않으므로 거부**한다. 메모장으로 고쳐도 된다.
3. 주제마다: 새 채팅 → 모델 메뉴에서 모드 선택 → `+` → `도구 더보기` → `Deep Research` → 주제 전송
   → 연구 계획이 나오면 **`연구 시작` 클릭** → 보고서가 완성될 때까지 대기(최대 180분, `--max-minutes`) → `.md` 저장.
   기본으로 주제 뒤에 선행연구 대조표·출처 링크·'확인 필요' 표시 요구를 붙인다 (`--raw` 면 주제만).
4. 진행 상황은 `research_queue.json`. 중간에 멈춰도 다시 실행하면 **안 끝난 주제부터** 이어 한다
   (완료한 주제는 건너뛰고, 실패한 주제는 다시 시도).
5. 멈추는 경우: CAPTCHA·로그아웃·Deep Research 한도 → 저장 후 종료(다음 실행에 이어 함).
   계획이 5분 안에 안 나오거나 메뉴를 못 찾으면 그 주제만 실패로 기록하고 다음 주제로 간다.

첫 사용 전 화면 구조 확인 (실제 Gemini 화면으로 검증되지 않았음):

```powershell
.\gemini_research.exe --probe
```

메시지를 보내지 않고 모델 메뉴·`+` 메뉴를 열어 실제 항목 이름을 `research\probe_시각\probe_report.json` 에 기록한다.
끝까지 확인하려면 `--probe-run` (짧은 주제로 Deep Research 1회 실제 진행, 사용량 1회 차감).
메뉴 이름이 다르면 `research_selectors.json` 으로 맞춘다 (exe 옆에 두면 재빌드 불필요).

하지 않는 것: 자동화 흔적 숨기기, 무작위 지연·마우스 흉내 같은 사람 흉내, CAPTCHA 풀기. 간격은 고정값이다.
학교 계정의 웹 Gemini 를 프로그램이 조작하는 것이므로 약관·계정 제한 위험은 사용자 판단이다.

## 자동 모드 (퇴근 후 실행용)

exe 를 더블클릭 → `실행 방식` 에서 **2: 자동** 을 고른다. 프로그램이 전용 Chrome 을 띄워 Gemini 탭 4개를 직접 조작한다
(마우스 매크로가 아니라 페이지 구조를 읽고 쓰므로 팝업·스크롤에 덜 민감하고, 클립보드를 쓰지 않는다).

1. **최초 1회:** 전용 Chrome 창이 뜨면 경북대 계정으로 로그인 → 콘솔에서 Enter. (평소 Chrome 과 별개 프로필)
2. **채팅 4개 등록 (첫 실행):** 이미 만들어 둔 채팅(예: 토론① 제안자 ~ 토론④ 사회자)의 주소를 하나씩 묻는다.
   전용 Chrome 창에서 그 채팅을 열고 주소창의 `https://gemini.google.com/app/...` 를 붙여 넣는다.
   Enter 만 누르면 그 역할은 새 채팅을 만들고, 첫 답변 뒤 주소를 저장한다. 등록 내용은 exe 폴더의 `tabs.json`.
   - 나중에 바꾸기: `.\gemini_council.exe --set-tabs` (기존 `tabs.json` 은 백업)
   - 전부 새 채팅으로: `--new-chats`
3. **간격:** 메시지 사이 **16분 고정** (`--interval` 로 변경). 3라운드 = 12차례 + 초록 → 약 3시간 반.
4. **멈추는 경우:** CAPTCHA·비정상 트래픽 화면, 로그아웃, 사용량 한도, 답변 10분 초과(`--answer-timeout`)
   → 즉시 저장하고 멈춘다. 다음에 exe 를 실행하면 이어서 할지 묻는다.
5. PC 는 켜 두고 **절전을 끈다**. Chrome 창·탭은 닫지 않는다 (최소화는 가능할 것으로 보이나 실기 확인 필요).

하지 않는 것: 자동화 흔적 숨기기, 무작위 지연 등 사람 흉내, CAPTCHA 풀기, 모델 자동 변경
(비평가 탭의 Pro 선택은 직접 해 둔다). 학교 계정의 웹 Gemini 를 프로그램이 조작하는 것이므로
약관·계정 제한 위험은 사용자 판단이다.

### 첫 사용 전 화면 구조 확인 (필수)

Gemini 의 실제 화면 구조는 개발 환경에서 확인할 수 없었다. 처음 한 번 아래를 실행해 결과를 확인한다.

```powershell
.\gemini_council.exe --web-probe
```

새 채팅에 시험 메시지 1개를 보내고 `logs\probe_시각\` 에 보고서(probe_report.json)와 화면 캡처를 남긴다.
"진단 성공" 이 아니면 그 폴더를 보내면 `web_selectors.json` 을 맞춰 준다 (exe 옆에 두면 재빌드 없이 적용).

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

### 초록 (토론 전체 요약)

토론이 정상 종료되거나 최대 라운드에 도달하면 **사회자 탭에 한 번 더** 초록 요청이 나간다
(`▶ [탭 4 · 사회자] 에 붙여 넣으세요` 가 한 번 더 뜸). 받은 초록은 회의록 맨 위 `## 초록` 에 들어간다.
형식: 배경·주제 / 핵심 쟁점 / 합의된 결론 / 미해결·검증 필요 / 다음 단계 (600자 이내).
초록 단계에서 `q` 를 눌러도 토론 본문은 완결된 상태로 저장된다. 끄려면 `--no-abstract`.

### 탭 재활용

같은 탭 4개를 여러 토론에 계속 써도 된다. 매 실행에서 각 탭의 첫 메시지 앞에
`[새 토론 시작 — 이전 대화 무시]` 문구가 붙어 이전 토론을 잊으라고 지시한다.
단, 지시에 의한 리셋이라 이전 내용은 Gemini 맥락에 남아 있다. 대화가 길어져 느려지거나
이전 토론이 섞여 보이면 새 채팅 4개로 바꾼다.

### 중단돼도 이어 하기

발언 하나가 끝날 때마다 `logs\discussion_*.md`(회의록)와 같은 이름의 `.json`(진행 상태)이 자동 저장된다.
`q`, 창 닫기, 오류, PC 절전 등으로 멈춰도 마지막으로 끝난 발언까지 남는다.

- exe 를 다시 더블클릭하면 "중단된 토론이 있습니다 … 이어서 할까요?" 를 묻는다 → `y`.
- 명령줄: `.\gemini_council.exe --resume logs\discussion_YYYYMMDD_HHMMSS.md`
- 이어 하면 각 탭의 첫 메시지에 규칙 + 사회자 최신 요약 + 그 뒤 발언 + 그 탭의 직전 발언이 들어가
  탭이 대화를 잃었어도 맥락이 복구된다 (같은 탭이든 새 채팅이든 상관없음).
- 최대 라운드까지 끝난 토론을 더 하려면 `--resume … --rounds 5` 처럼 라운드를 늘린다.

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
| `council/backends/gemini_web.py` | 자동: Playwright 로 Gemini 탭 제어, 완료 감지, CAPTCHA·한도 감지 |
| `council/chrome.py` | 전용 프로필 Chrome 실행 (디버깅 포트) |
| `council/web_tools.py` | `--web-login`, `--web-probe` |
| `tests/fake_gemini_page.html` | 가짜 Gemini 페이지 (스트리밍·중지 버튼·CAPTCHA·한도 재현) |
| `council/research.py`, `research_cli.py` | Deep Research 대기열 (`gemini_research.exe`) |
| `tests/fake_research_page.html` | 가짜 Deep Research 페이지 (모델 메뉴, 도구 메뉴, 연구 계획, '연구 시작') |
| `council/clipboard.py` | OS 클립보드 (Windows: Win32 API 직접, Linux: xclip 등) |
| `council/console.py` | 클립보드 감시 중 r/q 명령 입력 |
| `council/backends/mock.py` | 시드 고정 가짜 응답기 (종료·한도·타임아웃·오류 재현) |
| `council/backends/gemini_cli.py` | Gemini CLI 헤드리스 호출, 출력 파싱, 한도/인증/타임아웃 분류, 사용량 표 |
| `tests/fake_gemini.py` | 실제 CLI 의 입출력 규약을 흉내 낸 가짜 CLI (로그인 없이 테스트) |
| `tests/test_platform.py` | npm 식 래퍼(gemini.cmd) 경유 호출, 타임아웃 시 손자 프로세스 종료 |
| `build.ps1` | Windows 단일 exe 빌드 + 단독 실행 점검 |
| `council/orchestrator.py` | 메인 루프, 일시정지/중단, 모든 종료 경로에서 회의록 저장 |
| `council/session.py` | 발언마다 자동 저장(원자적 쓰기), 이어 하기용 상태 파일 |
| `council/transcript.py` | 회의록 마크다운 생성·저장 (UTF-8, 덮어쓰기 방지) |

## 맥락 전달 방식 (입력 길이 일정 유지)

- stateful 백엔드(웹 탭 등 대화를 스스로 기억): 그 역할이 마지막으로 말한 뒤의 **새 발언만** 보낸다.
- stateless 백엔드(API/CLI 단발 호출): **seed + 사회자 최신 요약 + 그 이후 발언**만 보낸다.

두 방식 모두 라운드가 늘어도 턴당 입력 길이가 일정하다 (`tests/test_protocol.py::test_prompt_length_bounded`).
