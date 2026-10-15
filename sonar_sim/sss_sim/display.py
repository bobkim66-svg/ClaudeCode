"""표시와 저장: dB 변환, 소나 컬러맵, PNG/NPZ/JSON 출력, 확인용 그림."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# 실제 SSS 표시 화면에서 흔한 호박/갈색 계열 (검정 -> 갈색 -> 호박 -> 밝은 노랑)
SONAR_CMAP = LinearSegmentedColormap.from_list(
    "sonar", ["#000000", "#2b1505", "#6e3a0e", "#b5701f", "#e3a547", "#fff2c9"]
)
DEFAULT_DB_RANGE: Tuple[float, float] = (-18.0, 10.0)  # 모래 평균 = 0 dB


def cmap_by_name(name: str):
    return SONAR_CMAP if name == "sonar" else plt.get_cmap(name)


def to_db(img: np.ndarray) -> np.ndarray:
    """선형 강도 -> dB (모래 평균 1 = 0 dB). 0 은 -inf 대신 -100 dB."""
    return 10 * np.log10(np.maximum(img, 1e-10))


def to_uint8(img: np.ndarray, db_range=DEFAULT_DB_RANGE) -> np.ndarray:
    """선형 강도 -> 8-bit 표시값 (dB 를 db_range 로 잘라 선형 매핑)."""
    lo, hi = db_range
    return np.round(np.clip((to_db(img) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def save_png(img: np.ndarray, path: Path, cmap: str = "sonar", db_range=DEFAULT_DB_RANGE) -> None:
    """화소 1:1 영상 저장 (축/여백 없음). 학습 데이터용."""
    plt.imsave(path, to_uint8(img, db_range), cmap=cmap_by_name(cmap), vmin=0, vmax=255)


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "text.color": "#222222", "axes.labelcolor": "#222222", "axes.edgecolor": "#444444",
        "xtick.color": "#333333", "ytick.color": "#333333", "axes.unicode_minus": False, "font.size": 9,
    })


def quicklook(rec, path: Path, cmap: str = "sonar", db_range=DEFAULT_DB_RANGE, dpi: int = 150) -> None:
    """가상 공간(위에서 본 해저 + 항적 + 물체) / 노이즈 영상 / 정답 / 라벨 4면 그림."""
    _style()
    w, nav, cfg = rec.world, rec.nav, rec.config
    fig = plt.figure(figsize=(16, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1, 1])
    ext_img = [rec.across_m[0], rec.across_m[-1], rec.along_m[-1], rec.along_m[0]]
    n_pings = rec.intensity.shape[0]

    ax = fig.add_subplot(gs[:, 0])
    ny, nx = w.shape
    ext_w = [w.x0, w.x0 + (nx - 1) * w.res, w.y0, w.y0 + (ny - 1) * w.res]
    step = max(1, int(max(nx, ny) / 1500))  # 그림 크기 제한용 축소
    hm = ax.imshow(w.height[::step, ::step], origin="lower", extent=ext_w, cmap="gist_earth")
    fig.colorbar(hm, ax=ax, shrink=0.6, label="seabed height [m]")
    ax.plot(nav.x, nav.y, color="#d62728", lw=1.5, label="track")
    ax.annotate("", xy=(nav.x[-1], nav.y[-1]), xytext=(nav.x[-8], nav.y[-8]),
                arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 1.5})
    sx, sy = nav.starboard()
    R = cfg.sonar.max_range_m
    for side, ls in ((1, "--"), (-1, ":")):
        ax.plot(nav.x + side * R * sx, nav.y + side * R * sy, color="#444444", ls=ls, lw=0.8,
                label="starboard swath edge" if side == 1 else "port swath edge")
    for k, o in enumerate(cfg.objects, start=1):
        ax.plot(o.x, o.y, "o", ms=9, mfc="none", mec="#ffffff", mew=1.5)
        ax.text(o.x, o.y, f" {k}:{o.name or o.kind}", color="white", fontsize=8, va="bottom")
    ax.set_xlabel("East x [m]")
    ax.set_ylabel("North y [m]")
    ax.set_title(f"Virtual seabed (top view), scene '{cfg.name}', seed {cfg.seed}")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.8)
    ax.set_aspect("equal")

    panels = [
        (gs[0, 1], rec.intensity, f"Simulated SSS (noisy), n = {n_pings} pings"),
        (gs[0, 2], rec.clean, "Clean expected intensity (same PSF)"),
    ]
    for spec, img, title in panels:
        a = fig.add_subplot(spec)
        im = a.imshow(to_db(img), cmap=cmap_by_name(cmap), vmin=db_range[0], vmax=db_range[1], extent=ext_img, aspect="auto")
        a.set_title(title)
        a.set_xlabel("slant range [m]  (port < 0 < starboard)")
        a.set_ylabel("along-track [m]")
    fig.colorbar(im, ax=fig.axes[-2:], shrink=0.8, label="intensity [dB re. sand mean]")

    a = fig.add_subplot(gs[1, 1])
    a.imshow(to_db(rec.intensity), cmap="gray", vmin=db_range[0], vmax=db_range[1], extent=ext_img, aspect="auto")
    lab = np.zeros(rec.intensity.shape + (4,))
    lab[rec.labels["shadow"]] = (0.12, 0.47, 0.71, 0.45)
    lab[rec.labels["object_shadow_id"] > 0] = (0.12, 0.47, 0.71, 0.85)
    lab[rec.labels["object_id"] > 0] = (1.0, 0.5, 0.05, 0.9)
    a.imshow(lab, extent=ext_img, aspect="auto")
    a.set_title("Labels: orange = object, blue = object shadow (light = any shadow)")
    a.set_xlabel("slant range [m]")
    a.set_ylabel("along-track [m]")

    a = fig.add_subplot(gs[1, 2])
    a.plot(rec.along_m, np.degrees(nav.roll), label="roll [deg]", color="#2a78d6", lw=1.2)
    a.plot(rec.along_m, np.degrees(nav.pitch), label="pitch [deg]", color="#eb6834", lw=1.2)
    a.plot(rec.along_m, np.degrees(nav.heading - np.median(nav.heading)), label="yaw dev. [deg]", color="#1baf7a", lw=1.2)
    a.plot(rec.along_m, (nav.z - cfg.trajectory.altitude_m) * 10, label="heave x10 [m]", color="#6b6b6b", lw=1.2)
    a.set_xlabel("along-track [m]")
    a.set_ylabel("attitude [deg] / heave x10 [m]")
    a.set_title(f"Vehicle motion (n = {n_pings} pings)")
    a.grid(True, color="#e3e3e3")
    a.legend(fontsize=7, ncol=2)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_record(rec, out_dir: Path, stem: str, cmap: str = "sonar", figure: bool = True,
                db_range=DEFAULT_DB_RANGE) -> Path:
    """npz (float32 영상 + 라벨 + 항법) + PNG (노이즈/정답) + JSON (설정, 메타) + 확인용 그림."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "intensity": rec.intensity.astype(np.float32),
        "clean": rec.clean.astype(np.float32),
        "across_m": rec.across_m.astype(np.float32),
        "along_m": rec.along_m.astype(np.float32),
        **{f"label_{k}": v for k, v in rec.labels.items()},
        **{f"nav_{k}": getattr(rec.nav, k).astype(np.float32) for k in ("t", "x", "y", "z", "heading", "roll", "pitch")},
    }
    if rec.iq is not None:
        arrays["iq"] = rec.iq
    np.savez_compressed(out_dir / f"{stem}.npz", **arrays)
    save_png(rec.intensity, out_dir / f"{stem}_noisy.png", cmap, db_range)
    save_png(rec.clean, out_dir / f"{stem}_clean.png", cmap, db_range)
    meta = {
        "config": rec.config.to_dict(),
        "shape": list(rec.intensity.shape),
        "columns": "port far..near | starboard near..far",
        "units": "linear intensity, sand mean ~ 1; PNG = dB clipped to db_range",
        "db_range": list(db_range),
        "objects_visible": sorted(int(v) for v in np.unique(rec.labels["object_id"]) if v > 0),
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if figure:
        quicklook(rec, out_dir / f"{stem}_quicklook.png", cmap, db_range)
    return out_dir / f"{stem}.npz"


def load_record_arrays(path: Path) -> dict:
    """save_record 로 저장한 npz 를 dict 로 읽는다."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def save_figure_only(rec, path: Path, cmap: Optional[str] = None) -> None:
    quicklook(rec, path, cmap or "sonar")
