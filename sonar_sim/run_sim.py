"""SSS 가상 시뮬레이션 실행기.

예 (PowerShell, sonar_sim 폴더에서):
  python run_sim.py                                   # 기본 장면 1장 (scenes/default.json)
  python run_sim.py --scene scenes/wreck_site.json    # 다른 장면
  python run_sim.py --random 20 --no-figure           # 무작위 장면 20장 (데이터셋)
  python run_sim.py --scene scenes/default.json --seed 7 --cmap gray
출력: results/<타임스탬프>/ 아래 npz (영상+라벨+항법), *_noisy.png, *_clean.png, *.json, *_quicklook.png
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from sss_sim.config import SceneConfig
from sss_sim.display import save_record
from sss_sim.randomize import random_scene
from sss_sim.sonar import simulate

HERE = Path(__file__).resolve().parent


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, default=HERE / "scenes" / "default.json", help="장면 JSON")
    ap.add_argument("--seed", type=int, default=None, help="장면 파일의 seed 덮어쓰기")
    ap.add_argument("--random", type=int, default=0, help="무작위 장면 N 개 생성 (--scene 무시)")
    ap.add_argument("--random-seed0", type=int, default=0, help="무작위 장면의 첫 seed")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--cmap", default="sonar", help="sonar | gray | 기타 matplotlib 컬러맵")
    ap.add_argument("--no-figure", action="store_true", help="확인용 4면 그림 생략 (대량 생성 시 빠름)")
    ap.add_argument("--save-iq", action="store_true", help="복소 IQ 저장 (용량 큼)")
    args = ap.parse_args(argv)

    out = args.out / datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.random > 0:
        jobs = [random_scene(args.random_seed0 + k) for k in range(args.random)]
    else:
        cfg = SceneConfig.load(args.scene)
        jobs = [cfg if args.seed is None else replace(cfg, seed=args.seed)]

    t_all = time.perf_counter()
    for k, cfg in enumerate(jobs, start=1):
        t0 = time.perf_counter()
        rec = simulate(cfg, keep_iq=args.save_iq)
        stem = f"{cfg.name}_seed{cfg.seed}"
        save_record(rec, out, stem, cmap=args.cmap, figure=not args.no_figure)
        elapsed = time.perf_counter() - t_all
        eta = elapsed / k * (len(jobs) - k)
        print(f"[{k}/{len(jobs)}] {stem}: {rec.intensity.shape[0]} pings x {rec.intensity.shape[1]} bins, "
              f"objects {len(cfg.objects)}, {time.perf_counter() - t0:.1f} s  (남은 시간 ~{eta / 60:.1f} 분)", flush=True)
    print(f"완료: {len(jobs)} 장, {time.perf_counter() - t_all:.1f} s -> {out}")


if __name__ == "__main__":
    main()
