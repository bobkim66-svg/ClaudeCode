"""소나 영상에 필터를 적용하고 비교 그림/지표를 저장한다.

입력 (파일 또는 폴더):
  - sonar_sim 결과 .npz  (권장: 선형 강도 'intensity' + 정답 'clean' 이 있어 정량 평가 가능)
  - .npy  (2D 선형 강도)
  - 폴더를 주면 그 안의 *.npz 를 모두 처리
  * PNG 는 dB 로 잘리고 컬러맵이 입혀진 표시용 영상이라 입력으로 받지 않는다 (원래 강도로 되돌릴 수 없음).

실행 (PowerShell, eisen_sonar_filter 폴더에서, 가상환경 켠 상태):
  python filter_sonar.py "C:\\...\\sonar_sim\\results\\20261016_163449"
  python filter_sonar.py 파일.npz --methods B Lee --window 7

출력: results/filtered_<타임스탬프>/
  <이름>_compare.png     원본 / 정답 / 각 필터 결과 비교 (dB, 같은 색 범위)
  <이름>_<방법>.png      필터 결과 영상 (화소 1:1, sonar_sim 과 같은 표시 방식)
  <이름>_filtered.npz    필터 결과 선형 강도 (float32)
  metrics.csv           방법별 PSNR/SSIM/EPI (정답이 있을 때) + 처리 시간
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.ndimage import laplace  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio, structural_similarity  # noqa: E402

from eisen_sonar.filters.baseline_filters import frost_filter, kuan_filter, lee_filter  # noqa: E402
from eisen_sonar.filters.gauss_filter import EisensteinGaussFilter  # noqa: E402

HERE = Path(__file__).resolve().parent
SONAR_CMAP = LinearSegmentedColormap.from_list(
    "sonar", ["#000000", "#2b1505", "#6e3a0e", "#b5701f", "#e3a547", "#fff2c9"]
)  # sonar_sim 과 같은 표시 컬러맵


@dataclass
class FilterSettings:
    """필터 파라미터 (기본값 = 벤치마크에서 튜닝 시드로 고른 값)."""

    window_b: int = 7  # B (적응형) 육각 창: 7 -> 37점
    pfa_b: float = 0.2  # B 오경보 확률
    window_a: int = 5  # A (원안 선형)
    window_lee: int = 5
    window_frost: int = 5
    frost_damping: float = 0.3
    n_looks: float = 1.0  # Lee/Kuan 의 스페클 look 수 (sonar_sim json 에 있으면 그 값 사용)
    db_range: tuple = (-18.0, 10.0)  # 표시/평가 dB 범위 (sonar_sim 기본과 동일)


def make_methods(s: FilterSettings) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    return {
        "B": EisensteinGaussFilter(mode="adaptive", window_size=s.window_b, p_false_alarm=s.pfa_b).denoise,
        "A": EisensteinGaussFilter(mode="linear", window_size=s.window_a).denoise,
        "Lee": lambda I: lee_filter(I, s.window_lee, s.n_looks),
        "Kuan": lambda I: kuan_filter(I, s.window_lee, s.n_looks),
        "Frost": lambda I: frost_filter(I, s.window_frost, s.frost_damping),
    }


def load_input(path: Path):
    """-> (강도 2D, 정답 or None, 축 dict or None, n_looks or None)."""
    if path.suffix.lower() == ".npz":
        with np.load(path) as z:
            if "intensity" not in z.files:
                raise KeyError(f"{path.name}: 'intensity' 배열이 없다 (sonar_sim 결과 npz 가 아님?)")
            I = z["intensity"].astype(float)
            C = z["clean"].astype(float) if "clean" in z.files else None
            axes = {"across": z["across_m"], "along": z["along_m"]} if "across_m" in z.files else None
        n_looks = None
        meta = path.with_suffix(".json")
        if meta.exists():
            n_looks = json.loads(meta.read_text(encoding="utf-8"))["config"]["noise"]["n_looks"]
        return I, C, axes, n_looks
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(float), None, None, None
    raise ValueError(f"지원하지 않는 형식: {path.suffix} (.npz 또는 .npy). PNG 는 표시용이라 강도로 되돌릴 수 없다.")


def to_unit(img: np.ndarray, db_range) -> np.ndarray:
    """선형 강도 -> dB 표시값을 [0, 1] 로 (PNG 에서 보는 것과 같은 스케일)."""
    lo, hi = db_range
    return np.clip((10 * np.log10(np.maximum(img, 1e-10)) - lo) / (hi - lo), 0.0, 1.0)


def evaluate(clean: np.ndarray, img: np.ndarray, db_range) -> Dict[str, float]:
    """표시 영역(dB 범위 -> [0,1]) 기준 PSNR/SSIM, 표시 영역 라플라시안 상관 EPI."""
    a, b = to_unit(clean, db_range), to_unit(img, db_range)
    la, lb = laplace(a), laplace(b)
    la, lb = la - la.mean(), lb - lb.mean()
    return {
        "PSNR_dB": float(peak_signal_noise_ratio(a, b, data_range=1.0)),
        "SSIM": float(structural_similarity(a, b, data_range=1.0, gaussian_weights=True, sigma=1.5)),
        "EPI": float(np.sum(la * lb) / np.sqrt(np.sum(la * la) * np.sum(lb * lb))),
    }


def save_png(img: np.ndarray, path: Path, db_range) -> None:
    plt.imsave(path, to_unit(img, db_range), cmap=SONAR_CMAP, vmin=0, vmax=1)


def compare_figure(panels, axes, title: str, path: Path, db_range) -> None:
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.unicode_minus": False,
                         "font.size": 9, "text.color": "#222222", "axes.labelcolor": "#222222"})
    n = len(panels)
    cols = 3 if n > 4 else 2
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.6 * rows), constrained_layout=True, squeeze=False)
    ext = None
    if axes is not None:
        ext = [axes["across"][0], axes["across"][-1], axes["along"][-1], axes["along"][0]]
    for ax, (name, img) in zip(axs.flat, panels):
        im = ax.imshow(10 * np.log10(np.maximum(img, 1e-10)), cmap=SONAR_CMAP, vmin=db_range[0], vmax=db_range[1],
                       extent=ext, aspect="auto", interpolation="nearest")
        ax.set_title(name)
        ax.set_xlabel("slant range [m]  (port < 0 < starboard)" if ext else "bin")
        ax.set_ylabel("along-track [m]" if ext else "ping")
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    fig.colorbar(im, ax=axs, shrink=0.8, label="intensity [dB re. sand mean]")
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def process(path: Path, out: Path, methods: List[str], s: FilterSettings) -> List[dict]:
    I, C, axes, n_looks = load_input(path)
    if I.ndim != 2:
        raise ValueError(f"{path.name}: 2D 영상이 아니다 {I.shape}")
    if n_looks is not None:
        s = FilterSettings(**{**s.__dict__, "n_looks": float(n_looks)})
    fns = make_methods(s)
    stem = path.stem
    panels = [("Noisy (input)", I)] + ([("Clean (ground truth)", C)] if C is not None else [])
    results, rows = {}, []
    base = {"file": path.name, "method": "Noisy", "ms": 0.0}
    if C is not None:
        base.update(evaluate(C, I, s.db_range))
    rows.append(base)
    for m in methods:
        t0 = time.perf_counter()
        y = fns[m](I)
        dt = (time.perf_counter() - t0) * 1e3
        if not np.all(np.isfinite(y)):
            raise FloatingPointError(f"{m} 결과에 NaN/inf")
        results[m] = y.astype(np.float32)
        save_png(y, out / f"{stem}_{m}.png", s.db_range)
        row = {"file": path.name, "method": m, "ms": dt}
        label = m
        if C is not None:
            row.update(evaluate(C, y, s.db_range))
            label += f"   PSNR {row['PSNR_dB']:.1f} dB | SSIM {row['SSIM']:.3f} | EPI {row['EPI']:.3f}"
        panels.append((label, y))
        rows.append(row)
    np.savez_compressed(out / f"{stem}_filtered.npz", **results)
    title = f"{stem}: {I.shape[0]} pings x {I.shape[1]} bins (n = 1 image)"
    if C is not None:
        title += f"  |  noisy PSNR {base['PSNR_dB']:.1f} dB, SSIM {base['SSIM']:.3f}"
    compare_figure(panels, axes, title, out / f"{stem}_compare.png", s.db_range)
    return rows


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="sonar_sim 결과 폴더, .npz 또는 .npy 파일")
    ap.add_argument("--methods", nargs="+", default=["Lee", "Frost", "A", "B"],
                    choices=["B", "A", "Lee", "Kuan", "Frost"], help="적용할 필터")
    ap.add_argument("--window", type=int, default=FilterSettings.window_b, help="B 필터 창 크기 (3,5,7,9)")
    ap.add_argument("--pfa", type=float, default=FilterSettings.pfa_b, help="B 필터 오경보 확률 (작을수록 더 평활)")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args(argv)

    src = args.input
    if not src.exists():
        raise FileNotFoundError(f"입력이 없다: {src}")
    files = sorted(src.glob("*.npz")) if src.is_dir() else [src]
    files = [f for f in files if not f.stem.endswith("_filtered")]
    if not files:
        raise FileNotFoundError(f"{src} 에 .npz 파일이 없다")
    out = args.out / ("filtered_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    s = FilterSettings(window_b=args.window, pfa_b=args.pfa)

    all_rows = []
    for k, f in enumerate(files, start=1):
        rows = process(f, out, args.methods, s)
        all_rows += rows
        print(f"[{k}/{len(files)}] {f.name}")
        for r in rows:
            q = f"PSNR {r['PSNR_dB']:6.2f} dB  SSIM {r['SSIM']:.3f}  EPI {r['EPI']:.3f}" if "PSNR_dB" in r else "(정답 없음)"
            print(f"    {r['method']:6s} {q}  {r['ms']:7.1f} ms")
    keys = ["file", "method", "PSNR_dB", "SSIM", "EPI", "ms"]
    with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"결과 폴더: {out}")


if __name__ == "__main__":
    main()
