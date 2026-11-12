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
from .backends.manual import ManualRelayBackend
from .clipboard import system_clipboard
from .console import ConsoleCommands
from .config import CouncilConfig
from .orchestrator import Orchestrator
from .roles import Role

_GC = GeminiCliConfig()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="멀티 AI 구조화 토론")
    p.add_argument("--topic", help="토론 주제 (--check 가 아니면 필수)")
    p.add_argument("--rounds", type=int, default=CouncilConfig.max_rounds, help="최대 라운드 수")
    p.add_argument("--char-limit", type=int, default=CouncilConfig.char_limit, help="발언 권장 상한 [자]")
    p.add_argument("--log-dir", type=Path, default=Path("logs"), help="회의록 저장 폴더")
    p.add_argument("--backend", choices=["manual", "gemini-cli", "mock"], default="manual",
                   help="manual=반자동(웹 탭에 직접 복사·붙여넣기, 기본), gemini-cli=무인, mock=오프라인 시험")
    p.add_argument("--clipboard-test", action="store_true", help="클립보드 읽기/쓰기만 점검하고 종료")

    g = p.add_argument_group("gemini-cli 백엔드")
    g.add_argument("--gemini-cmd", default="gemini",
                   help='Gemini CLI 실행 명령 (예: "gemini", "npx @google/gemini-cli")')
    g.add_argument("--model-fast", default=_GC.models["fast"], help="제안자/대안/사회자 모델 (빈 문자열=CLI 기본)")
    g.add_argument("--model-thinking", default=_GC.models["thinking"], help="비평가 모델")
    g.add_argument("--timeout", type=float, default=_GC.timeout_s, help="호출 1회 제한 시간 [s]")
    g.add_argument("--min-interval", type=float, default=_GC.min_interval_s, help="호출 간 최소 간격 [s]")
    g.add_argument("--check", action="store_true", help="로그인/모델 동작만 1회 호출로 확인하고 종료")

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


def make_backend(args: argparse.Namespace, cfg: CouncilConfig,
                 console: ConsoleCommands | None = None) -> ChatBackend:
    if args.backend == "manual":
        return ManualRelayBackend(commands=(console.poll if console else (lambda: None)))
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


def ask_interactive(read: Callable[[str], str] = input) -> list[str]:
    """exe 를 더블클릭(인자 없음)했을 때 주제와 라운드를 물어 argv 를 만든다."""
    print("=" * 64)
    print(" 멀티 AI 구조화 토론 (반자동 모드)")
    print("=" * 64)
    topic = ""
    while not topic:
        topic = read("토론 주제를 입력하세요: ").strip()
    rounds = INTERACTIVE_DEFAULT_ROUNDS
    while True:
        raw = read(f"최대 라운드 수 [Enter = {INTERACTIVE_DEFAULT_ROUNDS}]: ").strip()
        if not raw:
            break
        if raw.isdigit() and int(raw) >= 1:
            rounds = int(raw)
            break
        print("  1 이상의 숫자를 입력하세요.")
    return ["--backend", "manual", "--topic", topic, "--rounds", str(rounds)]


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
    cfg = CouncilConfig(max_rounds=args.rounds, char_limit=args.char_limit, log_dir=args.log_dir)
    if args.backend == "manual" and console is None:
        console = ConsoleCommands()
    backend = make_backend(args, cfg, console)
    if args.check:
        return run_check(backend)
    if not args.topic:
        parser.error("--topic 이 필요함")

    result = Orchestrator(backend, args.topic, cfg, on_event=_log).run()
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
