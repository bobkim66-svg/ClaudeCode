"""명령줄 실행기. 백엔드: mock (오프라인 검증) / gemini-cli (Google 공식 Gemini CLI)."""
from __future__ import annotations

import argparse
import shlex
import sys
from datetime import datetime
from pathlib import Path

from .backends import BackendError, ChatBackend, MockBackend
from .backends.gemini_cli import GeminiCliBackend, GeminiCliConfig
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
    p.add_argument("--backend", choices=["mock", "gemini-cli"], default="mock")

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


def make_backend(args: argparse.Namespace, cfg: CouncilConfig) -> ChatBackend:
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


def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = CouncilConfig(max_rounds=args.rounds, char_limit=args.char_limit, log_dir=args.log_dir)
    backend = make_backend(args, cfg)
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


if __name__ == "__main__":
    raise SystemExit(main())
