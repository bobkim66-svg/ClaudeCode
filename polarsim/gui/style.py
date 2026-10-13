"""그림 스타일 (흰 배경, 어두운 글씨, 연회색 grid) 과 색 지도."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

BACKGROUND = "w"
FOREGROUND = "#222222"
GRID_ALPHA = 0.25  # 연한 회색 grid
NAN_RGBA = (190, 190, 190, 255)  # 정의 불가(NaN) = 회색
EQUILIBRIUM_COLOR = (170, 170, 170)
SEGMENT_COLOR = (60, 60, 60, 160)
SELECT_COLOR = (220, 30, 30)
ARROW_COLOR = (20, 20, 20, 220)


def apply_global_style() -> None:
    pg.setConfigOptions(background=BACKGROUND, foreground=FOREGROUND, antialias=True, imageAxisOrder="col-major")


def style_plot(plot: pg.PlotItem, x_label: str, y_label: str, title: str | None = None, units: str = "m") -> None:
    """units 를 주면 pyqtgraph 가 SI 접두어를 붙여 '이름 (mm)' 처럼 표시한다 (눈금 값과 단위가 일치)."""
    plot.showGrid(x=True, y=True, alpha=GRID_ALPHA)
    plot.setLabel("bottom", x_label, units=units)
    plot.setLabel("left", y_label, units=units)
    if title:
        plot.setTitle(title, color=FOREGROUND, size="10pt")


def sequential_lut(n: int = 256) -> np.ndarray:
    """0~1 지표용 순차 색 지도 (viridis). (n,4) uint8."""
    return pg.colormap.get("viridis").getLookupTable(0.0, 1.0, n, alpha=True)


def diverging_lut(n: int = 256) -> np.ndarray:
    """평면 밖 변위 부호용 발산 색 지도 (파랑 − 회색 − 빨강). (n,4) uint8."""
    cmap = pg.ColorMap([0.0, 0.5, 1.0], [(40, 90, 200), (110, 110, 110), (200, 40, 40)])
    return cmap.getLookupTable(0.0, 1.0, n, alpha=True)


def to_rgba(values: np.ndarray, vmin: float, vmax: float, lut: np.ndarray) -> np.ndarray:
    """스칼라 맵 (nx,ny) → RGBA (nx,ny,4) uint8. NaN 은 회색으로 칠한다 (임의 값으로 채우지 않음)."""
    span = vmax - vmin if vmax > vmin else 1.0
    norm = np.clip((values - vmin) / span, 0.0, 1.0)
    idx = np.nan_to_num(norm * (len(lut) - 1), nan=0.0).astype(int)
    rgba = lut[idx].copy()
    rgba[np.isnan(values)] = NAN_RGBA
    return rgba
