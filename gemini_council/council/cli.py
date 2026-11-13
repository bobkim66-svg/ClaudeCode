"""명령줄 실행기. 백엔드: mock (오프라인 검증) / gemini-cli (Google 공식 Gemini CLI)."""
from __future__ import annotations

import argparse
import shlex
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

from .backends import BackendError, ChatBackend, MockBackend
from .backends.gemini_cli import GeminiCliBackend, GeminiCliConfig
from .backends.gemini_web import GeminiWebBackend, WebConfig
from .backends.manual import ManualConfig, ManualRelayBackend
from .clipboard import system_clipboard
from .console import ConsoleCommands
from .config import CouncilConfig
from .orchestrator import Orchestrator
from .session import SessionError, find_unfinished, load_state
from .web_tools import register_tabs, web_login, web_probe
from .roles import AGENDAS, Role

_GC = GeminiCliConfig()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="멀티 AI 구조화 토론")
    p.add_argument("--topic", help="토론 주제 (--check 가 아니면 필수)")
    p.add_argument("--rounds", type=int, default=None,
                   help=f"최대 라운드 수 (기본 {CouncilConfig.max_rounds}, 이어 하기면 저장된 값)")
    p.add_argument("--resume", type=Path, default=None,
                   help="중단된 토론 이어 하기: 회의록(.md) 또는 상태(.json) 경로")
    p.add_argument("--char-limit", type=int, default=CouncilConfig.char_limit, help="발언 권장 상한 [자]")
    p.add_argument("--log-dir", type=Path, default=Path("logs"), help="회의록 저장 폴더")
    p.add_argument("--backend", choices=["manual", "web", "gemini-cli", "mock"], default="manual",
                   help="manual=반자동(복사·붙여넣기, 기본), web=자동(Chrome 의 Gemini 탭 직접 제어), "
                        "gemini-cli=무인(개인 계정 CLI), mock=오프라인 시험")
    p.add_argument("--clipboard-test", action="store_true", help="클립보드 읽기/쓰기만 점검하고 종료")
    p.add_argument("--interval", type=float, default=None,
                   help="차례 사이 최소 간격 [분, 고정값] (기본: 반자동 5, 자동 16, 0 = 대기 없음)")
    p.add_argument("--no-abstract", action="store_true", help="토론 종료 후 초록(전체 요약) 요청 생략")
    p.add_argument("--agenda", choices=sorted(AGENDAS), default="general",
                   help="general=자유 토론, topic-review=연구 주제 검토(새로운 점·실현 가능성·위험·대안 점검)")

    g = p.add_argument_group("gemini-cli 백엔드")
    g.add_argument("--gemini-cmd", default="gemini",
                   help='Gemini CLI 실행 명령 (예: "gemini", "npx @google/gemini-cli")')
    g.add_argument("--model-fast", default=_GC.models["fast"], help="제안자/대안/사회자 모델 (빈 문자열=CLI 기본)")
    g.add_argument("--model-thinking", default=_GC.models["thinking"], help="비평가 모델")
    g.add_argument("--timeout", type=float, default=_GC.timeout_s, help="호출 1회 제한 시간 [s]")
    g.add_argument("--min-interval", type=float, default=_GC.min_interval_s, help="호출 간 최소 간격 [s]")
    g.add_argument("--check", action="store_true", help="로그인/모델 동작만 1회 호출로 확인하고 종료")

    w = p.add_argument_group("web 백엔드 (자동 모드)")
    w.add_argument("--web-login", action="store_true", help="전용 Chrome 프로필로 Gemini 로그인 (최초 1회)")
    w.add_argument("--web-probe", action="store_true", help="새 채팅에 시험 메시지 1개를 보내 화면 구조 진단")
    w.add_argument("--new-chats", action="store_true", help="저장된 채팅 4개 대신 새 채팅 4개로 시작 (tabs.json 백업)")
    w.add_argument("--set-tabs", action="store_true", help="사용할 기존 채팅 4개의 주소를 다시 등록 (tabs.json 백업)")
    w.add_argument("--answer-timeout", type=float, default=10.0, help="답변 1개를 기다리는 최대 시간 [분]")
    w.add_argument("--port", type=int, default=9222, help="Chrome 원격 디버깅 포트")
    w.add_argument("--profile-dir", type=Path, default=None, help="전용 Chrome 프로필 폴더 (기본: 사용자 앱 데이터)")
    w.add_argument("--web-url", default=None, help="(시험용) 새 채팅 주소 대신 열 주소")
    w.add_argument("--chrome-arg", action="append", default=[], help="(시험용) Chrome 추가 옵션, 여러 번 지정 가능")

    m = p.add_argument_group("mock 백엔드")
    m.add_argument("--stateless", action="store_true", help="mock 을 stateless 모드로")
    m.add_argument("--mock-end-round", type=int, default=None, help="mock 사회자가 이 라운드에 종료 선언")
    m.add_argument("--seed", type=int, default=0, help="mock 응답 난수 시드")
    return p


def split_command(s: str, windows: bool = sys.platform == "win32") -> tuple[str, ...]:
    """'--gemini-cmd' 문자열을 argv 로 나눈다.

    존재하는 파일 경로면 공백이 있어도 통째로 쓴다. Windows 에서는 posix=False 로 나눠
    백슬래시 경로(C:\\Users\\...\\gemini.cmd)가 깨지지 않게 하고 감싼 따옴표만 벗긴다.
    """
    if Path(s).exists():
        return (s,)
    parts = shlex.split(s, posix=not windows)
    return tuple(x[1:-1] if len(x) >= 2 and x[0] == x[-1] == '"' else x for x in parts)


def _log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


DEFAULT_INTERVAL_MIN = {"manual": 5.0, "web": 16.0}


def _interval_min(args: argparse.Namespace) -> float:
    return args.interval if args.interval is not None else DEFAULT_INTERVAL_MIN.get(args.backend, 0.0)


def web_config(args: argparse.Namespace) -> WebConfig:
    cfg = WebConfig(port=args.port, turn_interval_s=_interval_min(args) * 60.0,
                    response_timeout_s=args.answer_timeout * 60.0, extra_args=tuple(args.chrome_arg))
    if args.profile_dir is not None:
        cfg.profile_dir = args.profile_dir
    if args.web_url:
        cfg.new_chat_url = args.web_url
    return cfg


def make_backend(args: argparse.Namespace, cfg: CouncilConfig,
                 console: ConsoleCommands | None = None) -> ChatBackend:
    if args.backend == "manual":
        return ManualRelayBackend(commands=(console.poll if console else (lambda: None)),
                                  cfg=ManualConfig(turn_interval_s=_interval_min(args) * 60.0))
    if args.backend == "web":
        return GeminiWebBackend(web_config(args), log=_log, commands=(console.poll if console else (lambda: None)))
    if args.backend == "mock":
        return MockBackend(stateful=not args.stateless, seed=args.seed,
                           end_at_round=args.mock_end_round, end_token=cfg.end_token)
    gcfg = GeminiCliConfig(
        command=split_command(args.gemini_cmd),
        models={"fast": args.model_fast, "thinking": args.model_thinking},
        timeout_s=args.timeout, min_interval_s=args.min_interval)
    return GeminiCliBackend(gcfg, log=_log)


def run_check(backend: ChatBackend) -> int:
    """fast/thinking 모델에 짧은 질문 1번씩 보내 로그인·모델 접근을 확인한다."""
    probes = (Role("check_fast", "확인(fast)", "-", "fast"), Role("check_thinking", "확인(thinking)", "-", "thinking"))
    backend.open(probes)
    try:
        for r in probes:
            reply = backend.send(r, "연결 확인입니다. 'OK' 두 글자만 답하세요.")
            _log(f"{r.name} 응답: {reply[:80]!r}")
    except BackendError as e:
        _log(f"확인 실패: {e}")
        return 1
    finally:
        backend.close()
    print(backend.usage_report())
    return 0


def _force_utf8_output() -> None:
    """Windows 에서 출력을 파일/파이프로 보내면 cp949 등으로 인코딩되어 '⚠' 같은 문자에서 죽을 수 있다."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


INTERACTIVE_DEFAULT_ROUNDS = 3  # 반자동은 라운드당 복사·붙여넣기 4회 → 기본 3라운드(12회)


def _ask_mode(read: Callable[[str], str]) -> str:
    mode = ""
    while mode not in ("1", "2"):
        mode = read("실행 방식 — 1: 반자동(복사·붙여넣기), 2: 자동(Chrome 탭 직접 제어, 퇴근 후용) [Enter = 1]: ").strip() or "1"
    if mode == "2":
        print("  ※ 자동 모드는 학교 계정의 웹 Gemini 를 프로그램이 조작합니다 (약관·계정 제한 위험은 사용자 판단).")
        print("    CAPTCHA·한도·로그아웃이 보이면 저장 후 멈추며, 다음에 exe 를 실행하면 이어 할 수 있습니다.")
    return "web" if mode == "2" else "manual"


def _ask_interval(read: Callable[[str], str], default: float) -> float:
    interval = default
    while True:
        raw = read(f"차례 사이 대기 시간(분, 고정) [Enter = {default:g}]: ").strip()
        if not raw:
            return interval
        try:
            interval = float(raw)
        except ValueError:
            interval = -1.0
        if interval >= 0:
            return interval
        print("  0 이상의 숫자를 입력하세요.")


def ask_interactive(read: Callable[[str], str] = input, log_dir: Path = Path("logs")) -> list[str]:
    """exe 를 더블클릭(인자 없음)했을 때 질문으로 argv 를 만든다. 중단된 토론이 있으면 이어 할지 먼저 묻는다."""
    print("=" * 64)
    print(" 멀티 AI 구조화 토론")
    print("=" * 64)
    pending = find_unfinished(log_dir)
    if pending is not None:
        data, _ = load_state(pending)
        n = len(data["utterances"])
        print(f" 중단된 토론이 있습니다: \"{data['topic']}\"")
        print(f"   발언 {n}개 진행, 중단 사유: {data.get('stop_reason', '?')}, 파일: {pending.with_suffix('.md').name}")
        if read("이어서 할까요? (y = 이어 하기, Enter = 새 토론): ").strip().lower() in ("y", "yes", "ㅛ"):
            backend = _ask_mode(read)
            interval = _ask_interval(read, DEFAULT_INTERVAL_MIN[backend])
            return ["--backend", backend, "--resume", str(pending), "--interval", f"{interval:g}"]
    backend = _ask_mode(read)
    topic = ""
    while not topic:
        topic = read("토론 주제를 입력하세요: ").strip()
    kind = ""
    while kind not in ("1", "2"):
        kind = read("토론 유형 — 1: 자유 토론, 2: 연구 주제 검토 [Enter = 1]: ").strip() or "1"
    rounds = INTERACTIVE_DEFAULT_ROUNDS
    while True:
        raw = read(f"최대 라운드 수 [Enter = {INTERACTIVE_DEFAULT_ROUNDS}]: ").strip()
        if not raw:
            break
        if raw.isdigit() and int(raw) >= 1:
            rounds = int(raw)
            break
        print("  1 이상의 숫자를 입력하세요.")
    interval = _ask_interval(read, DEFAULT_INTERVAL_MIN[backend])
    turns = 4 * rounds
    total_min = interval * (turns - 1)
    print(f"  → 최대 {rounds}라운드 = {turns}차례, 대기만 최대 약 {total_min:g}분 (+ Gemini 답변 시간)")
    agenda = "topic-review" if kind == "2" else "general"
    return ["--backend", backend, "--topic", topic, "--rounds", str(rounds), "--agenda", agenda,
            "--interval", f"{interval:g}"]


def clipboard_selftest() -> int:
    """클립보드에 한글/이모지를 쓰고 다시 읽어 같은지 확인. 끝나면 원래 내용을 되돌린다."""
    clip = system_clipboard()
    before = clip.get_text()
    sample = "클립보드 점검 ✓ 😀 \"따옴표\" & | < >\n두 번째 줄"
    t0 = clip.change_token()
    clip.set_text(sample)
    ok = clip.get_text() == sample and clip.change_token() != t0
    clip.set_text(before)
    print(f"클립보드 점검: {'성공' if ok else '실패'}")
    return 0 if ok else 1


def run(argv: list[str], console: ConsoleCommands | None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.clipboard_test:
        return clipboard_selftest()
    if args.set_tabs:
        args.backend = "web"  # 채팅 주소 등록은 자동 모드 설정
    if args.web_login:
        return web_login(web_config(args))
    if args.web_probe:
        return web_probe(web_config(args), args.log_dir)
    resume = resume_json = None
    if args.resume is not None:
        try:
            resume, resume_json = load_state(args.resume)
        except SessionError as e:
            print(f"이어 하기 실패: {e}")
            return 1
        # 주제·점검 항목·길이 상한은 저장된 값을 쓴다 (같은 토론이므로). 라운드는 --rounds 로 늘릴 수 있다.
        args.topic = resume["topic"]
        cfg = CouncilConfig(max_rounds=args.rounds or int(resume["max_rounds"]),
                            char_limit=int(resume.get("char_limit", args.char_limit)),
                            log_dir=resume_json.parent, agenda=tuple(resume.get("agenda", ())))
    else:
        cfg = CouncilConfig(max_rounds=args.rounds or CouncilConfig.max_rounds, char_limit=args.char_limit,
                            log_dir=args.log_dir, agenda=AGENDAS[args.agenda])
    if args.backend in ("manual", "web") and console is None:
        console = ConsoleCommands()
    if args.backend == "web":
        wcfg = web_config(args)
        if not wcfg.profile_dir.exists():  # 최초 실행: 먼저 로그인
            print("자동 모드를 처음 실행합니다. 먼저 전용 Chrome 에서 로그인합니다.")
            read = (lambda q: (print(q, end="", flush=True), console.wait_line())[1] or "") if console else input
            if web_login(wcfg, read=read) != 0:
                return 1
        if (args.new_chats or args.set_tabs) and wcfg.tabs_file.exists():
            backup = wcfg.tabs_file.with_name(f"tabs_{datetime.now():%Y%m%d_%H%M%S}.bak.json")
            wcfg.tabs_file.rename(backup)
            print(f"저장된 채팅 주소를 {backup.name} 로 백업했습니다.")
        # 채팅 주소가 아직 없으면(첫 실행) 기존 채팅을 쓸지 먼저 묻는다. 사람이 없는 실행(파이프 입력)은 새 채팅.
        if not args.new_chats and not wcfg.tabs_file.exists() and console is not None and sys.stdin.isatty():
            read = lambda q: (print(q, end="", flush=True), console.wait_line())[1] or ""  # noqa: E731
            register_tabs(wcfg, read=read)
        if args.set_tabs and not args.topic:
            return 0
    backend = make_backend(args, cfg, console)
    if args.check:
        return run_check(backend)
    if not args.topic:
        parser.error("--topic 이 필요함")

    try:
        orch = Orchestrator(backend, args.topic, cfg, on_event=_log, resume=resume, resume_json=resume_json,
                            make_abstract=not args.no_abstract)
    except SessionError as e:
        print(f"이어 하기 실패: {e}")
        return 1
    result = orch.run()
    print(f"결과: {result.reason.value}, 완료 라운드 {result.rounds_done}, 발언 {len(result.utterances)}개")
    if result.detail:
        print(f"상세: {result.detail}")
    print(f"회의록: {result.log_path}")
    return 0 if result.reason.name in ("COMPLETED", "MAX_ROUNDS") else 2


def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()
    interactive = argv is None and len(sys.argv) == 1 and sys.stdin is not None and sys.stdin.isatty()
    if not interactive:
        return run(sys.argv[1:] if argv is None else argv, None)

    # exe 더블클릭: 질문으로 인자를 받고, 끝나면 창이 바로 닫히지 않게 Enter 를 기다린다.
    console: ConsoleCommands | None = None
    try:
        console_argv = ask_interactive()
        console = ConsoleCommands()  # 이후 모든 입력은 이 객체로 받는다 (input() 과 경합 방지)
        return run(console_argv, console)
    except BaseException:
        traceback.print_exc()  # 오류를 숨기지 않고 보여 준 뒤 다시 던진다
        raise
    finally:
        print("\nEnter 를 누르면 창이 닫힙니다.", flush=True)
        if console is not None:
            console.wait_line()
        else:
            try:
                input()
            except EOFError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
