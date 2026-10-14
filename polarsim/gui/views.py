"""표시 위젯: 관측 평면 애니메이션, 지표 맵, 선택 입자 궤적.

좌표: 모든 그림은 관측 평면 좌표 (s_u, s_v) [m] 와 평면 기저 (e_u, e_v, n) 성분으로 그린다.
변위 표시는 시각 배율 gain 을 곱한 값이다 (실제 변위는 격자 간격보다 수만 배 작다).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .style import (ARROW_COLOR, EQUILIBRIUM_COLOR, SEGMENT_COLOR, SELECT_COLOR, diverging_lut, sequential_lut,
                    style_plot, to_rgba)

_N_COLORS = 64


def _nearest_index(su: np.ndarray, sv: np.ndarray, x: float, y: float, mask=None) -> tuple[int, int]:
    d2 = (su - x) ** 2 + (sv - y) ** 2
    if mask is not None:
        d2 = np.where(mask, d2, np.inf)
    i, j = np.unravel_index(int(np.argmin(d2)), su.shape)
    return int(i), int(j)


def _pairs(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """선분 (x0→x1) 묶음을 connect='pairs' 용 1D 배열로."""
    out = np.empty(2 * x0.size)
    out[0::2] = x0.ravel()
    out[1::2] = x1.ravel()
    return out


class _ClickablePlot(pg.GraphicsLayoutWidget):
    particleClicked = QtCore.Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot = self.addPlot(row=0, col=0)
        self.plot.setAspectLocked(True)
        self._su = self._sv = None
        self._mask = None  # 클릭/표시 가능한 입자 (None = 전부)
        self._overlay: list = []
        self.scene().sigMouseClicked.connect(self._on_click)

    def _on_click(self, ev) -> None:
        if self._su is None or not self.plot.sceneBoundingRect().contains(ev.scenePos()):
            return
        p = self.plot.vb.mapSceneToView(ev.scenePos())
        self.particleClicked.emit(*_nearest_index(self._su, self._sv, p.x(), p.y(), self._mask))

    def set_overlay(self, rects=(), lines=()) -> None:
        """강체 직사각형 (x0,x1,y0,y1) [m] 은 진회색으로, 방사면 선분 ((x0,y0),(x1,y1)) 은 주황 굵은 선으로."""
        for it in self._overlay:
            self.plot.removeItem(it)
        self._overlay = []
        for x0, x1, y0, y1 in rects:
            r = QtWidgets.QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
            r.setBrush(pg.mkBrush(90, 90, 90, 255))
            r.setPen(pg.mkPen(None))
            r.setZValue(20)
            self.plot.addItem(r)
            self._overlay.append(r)
        for (xa, ya), (xb, yb) in lines:
            ln = pg.PlotDataItem([xa, xb], [ya, yb], pen=pg.mkPen((240, 140, 20), width=4))
            ln.setZValue(21)
            self.plot.addItem(ln)
            self._overlay.append(ln)


class PlaneView(_ClickablePlot):
    """평면 위 입자 변위 애니메이션. 점 = 현재 위치(배율 적용), 선분 = 평형점→현재 위치, 색 = 평면 밖 성분."""

    def __init__(self, parent=None):
        super().__init__(parent)
        style_plot(self.plot, "평면 u축 s_u", "평면 v축 s_v")
        self.eq = pg.ScatterPlotItem(size=3, pen=None, brush=pg.mkBrush(*EQUILIBRIUM_COLOR))
        self.seg = pg.PlotDataItem(connect="pairs", pen=pg.mkPen(SEGMENT_COLOR, width=1))
        self.dots = pg.ScatterPlotItem(size=7, pen=pg.mkPen((40, 40, 40, 140), width=0.5))
        self.sel = pg.ScatterPlotItem(size=18, pen=pg.mkPen(SELECT_COLOR, width=2), brush=None)
        for it in (self.eq, self.seg, self.dots, self.sel):
            self.plot.addItem(it)
        self._brushes = [pg.mkBrush(*c) for c in diverging_lut(_N_COLORS)]
        self._sel = (0, 0)

    def set_grid(self, su: np.ndarray, sv: np.ndarray, mask: np.ndarray | None = None) -> None:
        """mask (nx,ny) bool: 표시할 입자 (강체 안 입자 제외용). None = 전부."""
        self._su, self._sv, self._mask = su, sv, mask
        m = np.ones(su.shape, bool) if mask is None else mask
        self.eq.setData(su[m], sv[m])
        self.plot.enableAutoRange()

    def set_selected(self, i: int, j: int) -> None:
        self._sel = (i, j)

    def set_frame(self, du, dv, dn, gain: float, n_scale: float, n_particles: int,
                  color_label: str = "평면 밖 변위", color_unit: str = "m") -> None:
        """du, dv (nx,ny) [m]: 평면 기저 성분의 실제 변위. gain: 시각 배율.
        dn (nx,ny): 점 색으로 표시할 양 (기본 평면 밖 변위 [m]), n_scale: 색 범위 (±)."""
        m = np.ones(du.shape, bool) if self._mask is None else self._mask
        x = self._su + gain * du
        y = self._sv + gain * dv
        self.seg.setData(_pairs(self._su[m], x[m]), _pairs(self._sv[m], y[m]))
        c = dn[m]
        level = np.clip((c / n_scale + 1.0) / 2.0, 0.0, 1.0) if n_scale > 0 else np.full(c.size, 0.5)
        idx = (level * (_N_COLORS - 1)).astype(int)
        self.dots.setData(x[m], y[m], brush=[self._brushes[k] for k in idx])
        i, j = self._sel
        self.sel.setData([x[i, j]], [y[i, j]])
        self.plot.setTitle(
            f"입자 변위 (시각 배율 ×{gain:.3g}) · n = {n_particles}<br>"
            f"점 색 = {color_label}: 파랑 −, 회색 0, 빨강 + (±{n_scale:.2g} {color_unit})", size="9pt")


class MetricMap(_ClickablePlot):
    """입자별 지표 맵 + 색 막대 + (선택) 평면 내 세기 흐름 화살표. NaN = 회색."""

    def __init__(self, parent=None):
        super().__init__(parent)
        style_plot(self.plot, "평면 u축 s_u", "평면 v축 s_v")
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)
        self.arrows = pg.PlotDataItem(connect="pairs", pen=pg.mkPen(ARROW_COLOR, width=1.2))
        self.heads = pg.ScatterPlotItem(size=4, pen=None, brush=pg.mkBrush(*ARROW_COLOR[:3]))
        self.sel = pg.ScatterPlotItem(size=16, pen=pg.mkPen(SELECT_COLOR, width=2), brush=None, symbol="s")
        for it in (self.arrows, self.heads, self.sel):
            self.plot.addItem(it)
        self.cbar = self.addPlot(row=0, col=1)
        self.cbar.hideAxis("bottom")
        self.cbar.hideAxis("left")
        self.cbar.showAxis("right")
        self.cbar.setMouseEnabled(False, False)
        self.cbar_img = pg.ImageItem()
        self.cbar.addItem(self.cbar_img)
        self.ci.layout.setColumnMaximumWidth(1, 70)
        self.ci.layout.setColumnStretchFactor(0, 10)

    def set_data(self, values, su, sv, spacing, vmin, vmax, title, flow_uv=None) -> None:
        """values (nx,ny); su, sv (nx,ny) [m]; flow_uv (nx,ny,2) 평면 내 세기 성분 또는 None."""
        self._su, self._sv = su, sv
        lut = sequential_lut()
        self.img.setImage(to_rgba(values, vmin, vmax, lut), autoLevels=False)
        nx, ny = values.shape
        rect = QtCore.QRectF(su[0, 0] - spacing / 2, sv[0, 0] - spacing / 2, nx * spacing, ny * spacing)
        self.img.setRect(rect)
        bar = lut[None, :, :]  # (1,256,4): 세로 막대
        self.cbar_img.setImage(bar, autoLevels=False)
        self.cbar_img.setRect(QtCore.QRectF(0, vmin, 1, (vmax - vmin) or 1.0))
        self.cbar.setYRange(vmin, vmax if vmax > vmin else vmin + 1.0, padding=0)
        self.plot.setTitle(title, size="9pt")
        self._set_flow(flow_uv, su, sv, spacing)
        self.plot.setRange(rect, padding=0.02)

    def _set_flow(self, flow_uv, su, sv, spacing) -> None:
        if flow_uv is None:
            self.arrows.setData([], [])
            self.heads.setData([], [])
            return
        stride = max(1, max(su.shape) // 16)
        f = flow_uv[::stride, ::stride]
        x0, y0 = su[::stride, ::stride], sv[::stride, ::stride]
        mag = np.linalg.norm(f, axis=-1)
        scale = 0.9 * stride * spacing / mag.max() if mag.max() > 0 else 0.0
        x1, y1 = x0 + scale * f[..., 0], y0 + scale * f[..., 1]
        self.arrows.setData(_pairs(x0, x1), _pairs(y0, y1))
        self.heads.setData(x1.ravel(), y1.ravel())

    def set_selected(self, i: int, j: int) -> None:
        if self._su is not None:
            self.sel.setData([self._su[i, j]], [self._sv[i, j]])


class TrajectoryView(QtWidgets.QWidget):
    """선택 입자의 실제 변위 궤적 (배율 없음) 을 평면 기저 성분 세 투영으로 표시."""

    def __init__(self, parent=None, axis_names: tuple[str, str, str] = ("u", "v", "n")):
        super().__init__(parent)
        a0, a1, a2 = axis_names
        self.PROJECTIONS = ((0, 1, a0, a1), (0, 2, a0, a2), (1, 2, a1, a2))
        self.glw = pg.GraphicsLayoutWidget()
        self.curves, self.dots, self.plots = [], [], []
        for col, (a, b, la, lb) in enumerate(self.PROJECTIONS):
            p = self.glw.addPlot(row=0, col=col)
            style_plot(p, f"변위 {la} 성분", f"변위 {lb} 성분", f"{la}–{lb} 투영")
            p.setAspectLocked(True)
            c = p.plot(pen=pg.mkPen((30, 80, 180), width=1.5))
            d = pg.ScatterPlotItem(size=9, pen=None, brush=pg.mkBrush(*SELECT_COLOR))
            p.addItem(d)
            self.plots.append(p)
            self.curves.append(c)
            self.dots.append(d)
        self.info = QtWidgets.QLabel()
        self.info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.info.setWordWrap(True)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.glw, 1)
        lay.addWidget(self.info)
        self.traj = None

    def set_trajectory(self, traj: np.ndarray, info_html: str) -> None:
        """traj (Nt,3) [m] 평면 기저 성분 (u, v, n)."""
        self.traj = traj
        closed = np.vstack([traj, traj[:1]])
        lim = float(np.max(np.abs(traj))) * 1.1 or 1.0
        for (a, b, *_), c, p in zip(self.PROJECTIONS, self.curves, self.plots):
            c.setData(closed[:, a], closed[:, b])
            p.setXRange(-lim, lim, padding=0)
            p.setYRange(-lim, lim, padding=0)
        self.info.setText(info_html)

    def set_current(self, vec: np.ndarray) -> None:
        for (a, b, *_), d in zip(self.PROJECTIONS, self.dots):
            d.setData([vec[a]], [vec[b]])


def open_3d_view(parent, traj: np.ndarray, title: str) -> QtWidgets.QWidget:
    """OpenGL 3D 궤적 창. OpenGL 초기화 실패 시 예외를 그대로 올린다 (호출자가 로그+안내)."""
    import pyqtgraph.opengl as gl

    w = gl.GLViewWidget(parent)
    w.setWindowFlag(QtCore.Qt.WindowType.Window)
    w.setWindowTitle(title)
    w.setBackgroundColor("w")
    scale = float(np.max(np.linalg.norm(traj, axis=1))) or 1.0
    pts = np.vstack([traj, traj[:1]]) / scale
    w.addItem(gl.GLLinePlotItem(pos=pts, color=(0.1, 0.3, 0.7, 1.0), width=2, antialias=True))
    for k, col in enumerate([(0.8, 0.2, 0.2, 1), (0.2, 0.6, 0.2, 1), (0.2, 0.2, 0.8, 1)]):
        axis = np.zeros((2, 3))
        axis[1, k] = 1.2
        w.addItem(gl.GLLinePlotItem(pos=axis, color=col, width=2))
    grid = gl.GLGridItem()
    grid.setSize(2.4, 2.4)
    grid.setSpacing(0.2, 0.2)
    grid.setColor((150, 150, 150, 120))
    w.addItem(grid)
    w.setCameraPosition(distance=4.0)
    w.resize(600, 500)
    w.show()
    return w
