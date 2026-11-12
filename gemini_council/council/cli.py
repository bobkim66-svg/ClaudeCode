"""명령줄 실행기. 1단계에서는 mock 백엔드만 지원한다."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .backends import MockBackend
from .config import CouncilConfig
from .orchestrator import Orchestrator


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="멀티 AI 구조화 토론 (1단계: mock 백엔드)")
    p.add_argument("--topic", required=True, help="토론 주제")
    p.add_argument("--rounds", type=int, default=CouncilConfig.max_rounds, help="최대 라운드 수")
    p.add_argument("--char-limit", type=int, default=CouncilConfig.char_limit, help="발언 권장 상한 [자]")
    p.add_argument("--log-dir", type=Path, default=Path("logs"), help="회의록 저장 폴더")
    p.add_argument("--backend", choices=["mock"], default="mock")
    p.add_argument("--stateless", action="store_true", help="mock 을 stateless 모드로 (API/CLI 백엔드 흉내)")
    p.add_argument("--mock-end-round", type=int, default=None, help="mock 사회자가 이 라운드에 종료 선언")
    p.add_argument("--seed", type=int, default=0, help="mock 응답 난수 시드")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = CouncilConfig(max_rounds=args.rounds, char_limit=args.char_limit, log_dir=args.log_dir)
    backend = MockBackend(stateful=not args.stateless, seed=args.seed,
                          end_at_round=args.mock_end_round, end_token=cfg.end_token)

    def log(msg: str) -> None:
        print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)

    result = Orchestrator(backend, args.topic, cfg, on_event=log).run()
    print(f"결과: {result.reason.value}, 완료 라운드 {result.rounds_done}, 발언 {len(result.utterances)}개")
    print(f"회의록: {result.log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
