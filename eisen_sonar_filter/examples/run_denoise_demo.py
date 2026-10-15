"""종단간 벤치마크: 합성 SSS 영상 -> 필터 7종 -> 지표 표, 통계 검정, 그림.

공정성 절차:
  1) 튜닝 시드(1000번대)에서 각 방법의 하이퍼파라미터를 평균 PSNR 최대로 고른다.
  2) 겹치지 않는 평가 시드(0..n-1)에서 고정된 설정으로 평가한다.
  3) 시나리오 두 가지: "default" (고정 물리, 장면 무작위) / "randomized" (도메인 랜덤화, sim-to-real 강건성).
  4) 제안법 B 와 각 방법의 쌍체 Wilcoxon 부호순위 검정 (지표별 Holm 보정).

실행 (PowerShell, eisen_sonar_filter 폴더에서):
  python examples/run_denoise_demo.py              # 본 실행 (약 1~2분)
  python examples/run_denoise_demo.py --smoke      # 빠른 점검
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eisen_sonar.filters.baseline_filters import frost_filter, kuan_filter, lee_filter  # noqa: E402
from eisen_sonar.filters.gauss_filter import EisensteinGaussFilter  # noqa: E402
from eisen_sonar.utils import metrics as M  # noqa: E402
from eisen_sonar.utils.data_loader import SonarSimConfig, make_synthetic_sonar, randomized_config  # noqa: E402


@dataclass
class BenchConfig:
    n_seeds: int = 20
    n_tune: int = 5
    tune_seed0: int = 1000
    n_looks: int = 1  # 기준선 필터에 주는 L (시뮬레이터 실제값과 같음 = oracle)
    timing_repeats: int = 5
    dpi: int = 160
    out_root: Path = ROOT / "results"


# 방법 이름 -> (설정 dict -> 필터 함수), 튜닝 격자
def _eg(**kw) -> Callable[[np.ndarray], np.ndarray]:
    return EisensteinGaussFilter(**kw).denoise


def make_methods(n_looks: int):
    return {
        "Noisy": (lambda p: (lambda I: I), [{}]),
        "Lee": (lambda p: (lambda I: lee_filter(I, p["window"], n_looks)), [{"window": w} for w in (3, 5, 7, 9)]),
        "Kuan": (lambda p: (lambda I: kuan_filter(I, p["window"], n_looks)), [{"window": w} for w in (3, 5, 7, 9)]),
        "Frost": (
            lambda p: (lambda I: frost_filter(I, p["window"], p["damping"])),
            [{"window": w, "damping": d} for w, d in itertools.product((5, 7), (0.1, 0.3, 1.0, 2.0))],
        ),
        "HexResample": (lambda p: _eg(mode="resample_only"), [{}]),
        "A_EisLinear": (lambda p: _eg(mode="linear", window_size=p["window"]), [{"window": w} for w in (3, 5, 7)]),
        "B_EisAdaptive": (
            lambda p: _eg(mode="adaptive", window_size=p["window"], p_false_alarm=p["pfa"], detector_scale=p["scale"]),
            [{"window": w, "pfa": a, "scale": s} for w, a, s in itertools.product((5, 7, 9), (0.01, 0.05, 0.2), (1, 2))],
        ),
    }


METRICS = ["ENL", "PSNR", "SSIM", "EPI", "CNR_target", "CNR_shadow", "ms"]
PROPOSED = "B_EisAdaptive"


def sample_for(scenario: str, seed: int):
    if scenario == "default":
        return make_synthetic_sonar(SonarSimConfig(), seed=seed)
    cfg = randomized_config(np.random.default_rng(10_000 + seed))
    return make_synthetic_sonar(cfg, seed=seed)


def score(sample, out: np.ndarray) -> Dict[str, float]:
    m = sample.masks
    row = {
        "ENL": M.enl(out, m["homogeneous"]),
        "PSNR": M.psnr(sample.clean, out),
        "SSIM": M.ssim(sample.clean, out),
        "EPI": M.epi(sample.clean, out),
    }
    for key, mk in (("CNR_target", "target"), ("CNR_shadow", "shadow")):
        try:
            row[key] = M.cnr(out, m[mk], m["background"])
        except ValueError:  # 마스크 화소 부족 (작은/먼 표적) -> 해당 지표만 결측, 개수는 보고
            row[key] = float("nan")
    return row


def tune(methods, bc: BenchConfig, log) -> Dict[str, dict]:
    samples = [sample_for("default", bc.tune_seed0 + i) for i in range(bc.n_tune)]
    best = {}
    for name, (factory, grid) in methods.items():
        scores = []
        for p in grid:
            f = factory(p)
            scores.append(np.mean([M.psnr(s.clean, f(s.intensity)) for s in samples]))
        i = int(np.argmax(scores))
        best[name] = grid[i]
        log(f"  tune {name:14s} -> {grid[i]}  (PSNR {scores[i]:.2f} dB, grid {len(grid)})")
    return best


def evaluate(scenario, methods, params, bc: BenchConfig):
    rows, keep = [], {}
    fns = {n: methods[n][0](params[n]) for n in methods}
    for seed in range(bc.n_seeds):
        s = sample_for(scenario, seed)
        for name, f in fns.items():
            t0 = time.perf_counter()
            out = f(s.intensity)
            dt = (time.perf_counter() - t0) * 1e3
            assert out.shape == s.intensity.shape and np.all(np.isfinite(out)), name
            rows.append({"scenario": scenario, "seed": seed, "method": name, **score(s, out), "ms": dt})
            if seed == 0:
                keep[name] = out
        if seed == 0:
            keep["_sample"] = s
    return rows, keep


def holm(pvals: List[float]) -> List[float]:
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return list(adj)


def summarize(rows, methods, log):
    names = list(methods)
    log(f"\n{'method':14s} " + " ".join(f"{k:>16s}" for k in METRICS))
    table = {}
    for n in names:
        vals = {k: np.array([r[k] for r in rows if r["method"] == n], float) for k in METRICS}
        table[n] = vals
        cells = []
        for k in METRICS:
            v = vals[k][np.isfinite(vals[k])]
            cells.append(f"{v.mean():8.3f}±{v.std(ddof=1):6.3f}" if len(v) > 1 else f"{'n/a':>16s}")
        log(f"{n:14s} " + " ".join(f"{c:>16s}" for c in cells))
    nan_counts = {k: int(np.sum(~np.isfinite(table[PROPOSED][k]))) for k in METRICS}
    log(f"(seed 당 결측 CNR 수 {nan_counts['CNR_target']}/{nan_counts['CNR_shadow']} : target/shadow 마스크 < 20 px)")

    log(f"\nWilcoxon signed-rank (paired), {PROPOSED} - other, 지표별 Holm 보정. 중앙값 차이 [p_holm]")
    others = [n for n in names if n != PROPOSED]
    for k in METRICS[:-1]:
        ps, meds = [], []
        for o in others:
            a, b = table[PROPOSED][k], table[o][k]
            ok = np.isfinite(a) & np.isfinite(b)
            d = a[ok] - b[ok]
            meds.append(np.median(d) if len(d) else np.nan)
            ps.append(wilcoxon(a[ok], b[ok]).pvalue if len(d) >= 6 and np.any(d != 0) else np.nan)
        finite = [p for p in ps if np.isfinite(p)]
        adj_f = iter(holm(finite)) if finite else iter([])
        adj = [next(adj_f) if np.isfinite(p) else np.nan for p in ps]
        log(f"  {k:10s} " + "  ".join(f"{o}:{m:+.3f}[{p:.1e}]" for o, m, p in zip(others, meds, adj)))
    return table


# ---------------------------------------------------------------- 그림
GRAY, BLUE, ORANGE = "#8c8c8c", "#2a78d6", "#eb6834"
METHOD_COLOR = {"A_EisLinear": ORANGE, PROPOSED: BLUE}


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": "#444444", "text.color": "#222222", "axes.labelcolor": "#222222",
        "xtick.color": "#333333", "ytick.color": "#333333", "axes.grid": True, "grid.color": "#e3e3e3",
        "grid.linewidth": 0.6, "axes.unicode_minus": False, "font.size": 9,
    })


def to_db(img, ref):
    return 10 * np.log10(np.maximum(img, 1e-12) / ref)


def plot_images(keep, rows, path: Path, scenario: str, dpi: int):
    s = keep["_sample"]
    ref = np.median(s.clean[s.masks["homogeneous"]])
    names = ["Clean"] + [n for n in keep if n != "_sample"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6), constrained_layout=True)
    ext = [s.range_m[0], s.range_m[-1], s.along_m[-1], s.along_m[0]]
    for ax, n in zip(axes.flat, names):
        img = s.clean if n == "Clean" else keep[n]
        im = ax.imshow(to_db(img, ref), cmap="gray", vmin=-15, vmax=10, extent=ext, aspect="auto")
        ax.grid(False)
        if n == "Clean":
            ax.set_title("Clean (expected intensity)")
        else:
            r = next(r for r in rows if r["method"] == n and r["seed"] == 0)
            ax.set_title(f"{n}\nENL {r['ENL']:.1f} | PSNR {r['PSNR']:.1f} dB | SSIM {r['SSIM']:.2f} | EPI {r['EPI']:.2f}",
                         color=METHOD_COLOR.get(n, "#222222"))
        ax.set_xlabel("slant range [m]")
        ax.set_ylabel("along-track [m]")
    fig.colorbar(im, ax=axes, shrink=0.8, label="intensity [dB re. flat-sand mean]")
    fig.suptitle(f"Synthetic side-scan sonar, single-look speckle - scenario '{scenario}', seed 0 (n = 1 image shown)")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_boxes(table, path: Path, scenario: str, n: int, dpi: int):
    names = list(table)
    keys = ["ENL", "PSNR", "SSIM", "EPI", "CNR_target", "CNR_shadow"]
    units = {"ENL": "ENL [-]", "PSNR": "PSNR [dB]", "SSIM": "SSIM [-]", "EPI": "EPI [-]",
             "CNR_target": "CNR target [-]", "CNR_shadow": "CNR shadow [-]"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
    for ax, k in zip(axes.flat, keys):
        data = [table[m][k][np.isfinite(table[m][k])] for m in names]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops={"color": "#222222", "linewidth": 1.5},
                        flierprops={"markersize": 3})
        for patch, m in zip(bp["boxes"], names):
            patch.set_facecolor(METHOD_COLOR.get(m, "#d4d4d4"))
            patch.set_edgecolor("#444444")
        ax.set_xticks(range(1, len(names) + 1), names, rotation=35, ha="right")
        ax.set_ylabel(units[k])
        ax.set_title(f"{units[k]}  (n = {len(data[0])} images per method)")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="#444444") for c in ("#d4d4d4", ORANGE, BLUE)]
    fig.legend(handles, ["baselines / controls", "A: Eisenstein linear (Gauss-sum kernel)",
                         "B: Eisenstein adaptive (character sums)"], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.suptitle(f"Scenario '{scenario}': metric distributions over {n} seeds", y=1.1)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_profile(keep, path: Path, dpi: int):
    s = keep["_sample"]
    tm = s.masks["target"]
    row = int(np.argmax(tm.sum(axis=1)))  # 표적 화소가 가장 많은 핑
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ref = np.median(s.clean[s.masks["homogeneous"]])
    lines = [("Noisy", "#c8c8c8", 0.8), ("Lee", "#6b6b6b", 1.4), ("A_EisLinear", ORANGE, 1.4),
             (PROPOSED, BLUE, 2.0)]
    for n, c, lw in lines:
        ax.plot(s.range_m, to_db(keep[n][row], ref), color=c, lw=lw, label=n)
    ax.plot(s.range_m, to_db(s.clean[row], ref), color="#111111", lw=1.2, ls="--", label="Clean")
    ax.set_ylim(-25, 15)
    ax.set_xlabel("slant range [m]")
    ax.set_ylabel("intensity [dB re. flat-sand mean]")
    ax.set_title(f"Range profile through target, ping {row} (along {s.along_m[row]:.2f} m), n = 1 ping", pad=28)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def timing(methods, params, bc: BenchConfig, log):
    s = make_synthetic_sonar(SonarSimConfig(), seed=0)
    log(f"\n처리 시간 (영상 {s.intensity.shape[0]} ping x {s.intensity.shape[1]} bin, 중앙값 {bc.timing_repeats}회)")
    for n, (factory, _) in methods.items():
        f = factory(params[n])
        ts = []
        for _ in range(bc.timing_repeats):
            t0 = time.perf_counter()
            f(s.intensity)
            ts.append(time.perf_counter() - t0)
        t = float(np.median(ts))
        log(f"  {n:14s} {t * 1e3:8.1f} ms  -> {s.intensity.shape[0] / t:9.0f} pings/s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-seeds", type=int, default=BenchConfig.n_seeds)
    ap.add_argument("--n-tune", type=int, default=BenchConfig.n_tune)
    ap.add_argument("--smoke", action="store_true", help="시드 3개/튜닝 1개로 빠른 점검")
    args = ap.parse_args(argv)
    bc = BenchConfig(n_seeds=3 if args.smoke else args.n_seeds, n_tune=1 if args.smoke else args.n_tune)

    out = bc.out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg=""):
        print(msg, flush=True)
        log_lines.append(msg)

    _style()
    methods = make_methods(bc.n_looks)
    log(f"[1] 하이퍼파라미터 튜닝 (default 시나리오, 튜닝 시드 {bc.n_tune}개, 기준: 평균 PSNR)")
    params = tune(methods, bc, log)

    all_rows = []
    for scenario in ("default", "randomized"):
        log(f"\n[2] 평가: scenario={scenario}, 평가 시드 {bc.n_seeds}개 (튜닝 시드와 분리)")
        rows, keep = evaluate(scenario, methods, params, bc)
        all_rows += rows
        table = summarize(rows, methods, log)
        plot_images(keep, rows, out / f"denoise_benchmark_{scenario}.png", scenario, bc.dpi)
        plot_boxes(table, out / f"metrics_boxplot_{scenario}.png", scenario, bc.n_seeds, bc.dpi)
        if scenario == "default":
            plot_profile(keep, out / "range_profile_default.png", bc.dpi)

    timing(methods, params, bc, log)

    with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["scenario", "seed", "method"] + METRICS)
        w.writeheader()
        w.writerows(all_rows)
    (out / "summary.txt").write_text("\n".join(log_lines) + f"\nparams: {params}\n", encoding="utf-8")
    log(f"\n결과 폴더: {out}")


if __name__ == "__main__":
    main()
