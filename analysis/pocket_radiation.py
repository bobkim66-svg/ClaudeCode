"""AUV 품(삼면 박스) 방사: 품 밖 입자 궤적 편광이 거리에 따라 어떻게 변하는가 (엔진 B, 2D).

실행 (PowerShell):  python -m analysis.pocket_radiation            # 그림 포함
                    python -m analysis.pocket_radiation --no-plot  # 표만
출력: 표준출력 표 + results/pocket_<시각>/ 에 그림 (거리별 타원도, 지표 맵).

비교: '3면 방사' (뒷벽 0°, 윗판·아랫판 90°) 와 '뒷벽만 방사' (순수 종파 피스톤) 를 주파수별로.
지표는 입구 중심 (L, 0) 앞쪽 반평면에서 거리 구간별 에너지 가중 타원도. 계산 영역이 반고리를 다 덮지 못하는
구간 (coverage < 0.95) 은 모서리 방향만 표본이 되어 비교할 수 없으므로 표에 표시하고 그림에서 뺀다.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from polarsim.export import timestamped_dir
from polarsim.fdtd import PocketSpec, pocket_config, radial_ellipticity, solve_harmonic, wave_metrics


@dataclass(frozen=True)
class Args:
    freqs: tuple[float, ...] = (2400.0, 2700.0, 3000.0, 3300.0)
    domain: tuple[float, float, float, float] = (-1.0, 10.5, -6.0, 6.0)  # 입구 앞 반고리를 ~16λ 까지 덮도록
    edges_lambda: tuple[float, ...] = (0, 1, 2, 4, 8, 12, 16)
    full_coverage: float = 0.95


def run_cases(args: Args) -> list[dict]:
    base = PocketSpec()
    off = (False, base.back[1], 0.0)
    cases = [("3면 방사", base), ("뒷벽만 방사", replace(base, top=off, bottom=off))]
    out = []
    for f in args.freqs:
        for label, spec in cases:
            cfg = pocket_config(spec, frequency=f, domain=args.domain)
            res = solve_harmonic(cfg)
            met = wave_metrics(res, cfg.medium)
            X, Y = np.meshgrid(res.xc, res.yc, indexing="ij")
            inside = (X > 0) & (X < spec.length) & (np.abs(Y) < spec.gap / 2)
            cut = spec.cutoff_frequencies(cfg.medium.c, 2 * f)
            out.append(dict(
                f=f, label=label, lam=cfg.wavelength, near_cutoff=min(cut, key=lambda fc: abs(fc - f)),
                curl=res.curl_ratio, e_inside=float(np.nanmedian(met["e"][inside])),
                p_inside=float(np.abs(res.P[inside]).max()), res=res, met=met,
                rows=radial_ellipticity(res, met, (spec.length, 0.0), cfg.wavelength, args.edges_lambda,
                                        half_plane_x=spec.length)))
    return out


def print_table(summary, args: Args) -> None:
    for s in summary:
        print(f"\nf = {s['f']:.0f} Hz (λ = {s['lam']:.3f} m, 가장 가까운 횡모드 차단 {s['near_cutoff']:.0f} Hz) · "
              f"{s['label']} · curl 비 {s['curl']:.1e}")
        print(f"  품 안: e 중앙값 {s['e_inside']:.2f}, 최대 |p| {s['p_inside']:.3g} Pa")
        print("  품 밖 거리 r/λ   포함률   e(에너지가중)  e 중앙값  e>0.3   1/(kr)")
        for r in s["rows"]:
            flag = "" if r["coverage"] >= args.full_coverage else "  ← 영역 잘림, 비교 불가"
            print(f"    {r['lo']:4.0f}–{r['hi']:<4.0f}  {r['coverage']:6.0%}   {r['e_energy']:8.3f}     "
                  f"{r['e_median']:7.3f}  {r['frac_e03']:5.0%}  {1 / (np.pi * (r['lo'] + r['hi'])):6.3f}{flag}")


def plot(summary, args: Args, out_root: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import ScalarFormatter

    names = {f.name for f in font_manager.fontManager.ttflist}
    for fam in ("Malgun Gothic", "Noto Sans CJK KR", "NanumGothic"):
        if fam in names:
            plt.rcParams["font.family"] = fam
            break
    plt.rcParams.update({"axes.unicode_minus": False, "figure.facecolor": "white", "axes.facecolor": "white",
                         "axes.grid": True, "grid.color": "#dddddd", "text.color": "#222222"})
    out = timestamped_dir(out_root, "pocket")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for s in summary:
        rows = [r for r in s["rows"] if r["coverage"] >= args.full_coverage and r["lo"] > 0]
        mids = np.array([(r["lo"] + r["hi"]) / 2 for r in rows])
        ax.plot(mids, [r["e_energy"] for r in rows], marker="o", ls="-" if s["label"] == "3면 방사" else "--",
                label=f"{s['f']:.0f} Hz · {s['label']}")
    m = np.array([1.5, 14.0])
    ax.plot(m, 1 / (2 * np.pi * m), color="k", lw=1, ls=":", label="1/(kr) (원거리 이론)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([1.5, 3, 6, 10, 14])
    ax.set_yticks([0.01, 0.02, 0.05, 0.1, 0.2])
    ax.set_xlabel("입구 중심으로부터 거리 r / λ (구간 중앙)")
    ax.set_ylabel("에너지 가중 타원도 e (입구 앞 반평면)")
    ax.set_title(f"품 밖 입자 궤적 타원도 vs 거리 · 2D, 물, 셀 = λ/20 · 곡선 {len(summary)}개")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "ellipticity_vs_distance.png", dpi=150)
    plt.close(fig)

    for s in summary:
        if s["label"] != "3면 방사":
            continue
        res, met = s["res"], s["met"]
        ext = [res.xc[0], res.xc[-1], res.yc[0], res.yc[-1]]
        fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
        for a, key, title in ((axs[0], "rms", "RMS 변위 [m]"), (axs[1], "e", "타원도 e (회색 = 강체)")):
            cmap = plt.get_cmap("viridis").copy()
            cmap.set_bad("#bbbbbb")
            vals = met[key]
            vmax = float(np.nanpercentile(vals, 99)) if key == "rms" else 1.0
            im = a.imshow(vals.T, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
            fig.colorbar(im, ax=a, shrink=0.8)
            a.set_xlabel("x [m]")
            a.set_ylabel("y [m]")
            a.set_title(f"{s['f']:.0f} Hz · {title}", fontsize=10)
        fig.suptitle(f"AUV 품 방사 (뒷벽 0°, 윗판·아랫판 90°) · 셀 {res.P.size}개 · curl 비 {res.curl_ratio:.0e}",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out / f"maps_{s['f']:.0f}Hz.png", dpi=150)
        plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    args = Args()
    summary = run_cases(args)
    print_table(summary, args)
    if not a.no_plot:
        print(f"\n그림 저장: {plot(summary, args, Path('results')).resolve()}")


if __name__ == "__main__":
    main()
