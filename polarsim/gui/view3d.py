"""3D 뷰 (CAD 처럼 돌려 보기): 형상 음영, 압력 단면, 입자 점 구름, 축·격자·영역 상자, 색 막대.

마우스: 왼쪽 드래그 = 회전, 가운데 드래그(또는 Ctrl+왼쪽) = 이동, 휠 = 확대/축소 (pyqtgraph GLViewWidget 기본).
좌표: z 위쪽. 단면 이미지는 셀 하나 = 픽셀 하나로 그 평면 위치에 놓는다.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6 import QtCore, QtGui, QtWidgets

BG = (247, 248, 250)
BODY_RGBA = (0.62, 0.64, 0.68, 1.0)
TRANSDUCER_RGBA = (0.95, 0.55, 0.10, 1.0)
GRID_RGBA = (0, 0, 0, 40)
BOX_RGBA = (0.35, 0.35, 0.40, 1.0)


def _register_cad_shader() -> str:
    """CAD 식 조명 셰이더 (카메라 쪽 헤드라이트 + 위쪽 보조광, 양면). pyqtgraph 'shaded' 는 빛이 화면 안쪽
    (1,−1,−1) 에서 와서 카메라를 향한 면이 거의 검게 보이므로 따로 등록한다. 법선 방향이 뒤집힌 STL 도 같은 밝기."""
    from pyqtgraph.opengl import shaders as sh

    name = "polarsim_cad"
    if name not in sh.ShaderProgram.names:
        sh.ShaderProgram(name, [
            sh.VertexShader("""
                uniform mat4 u_mvp;
                uniform mat3 u_normal;
                attribute vec4 a_position;
                attribute vec3 a_normal;
                attribute vec4 a_color;
                varying vec4 v_color;
                varying vec3 v_normal;
                void main() {
                    v_normal = normalize(u_normal * a_normal);
                    v_color = a_color;
                    gl_Position = u_mvp * a_position;
                }
            """),
            sh.FragmentShader("""
                #ifdef GL_ES
                precision mediump float;
                #endif
                varying vec4 v_color;
                varying vec3 v_normal;
                void main() {
                    vec3 n = normalize(v_normal);
                    float head = abs(dot(n, normalize(vec3(0.25, 0.35, 1.0))));
                    float top = abs(dot(n, normalize(vec3(-0.4, 1.0, 0.3))));
                    vec3 rgb = v_color.rgb * (0.30 + 0.55 * head + 0.25 * top);
                    gl_FragColor = vec4(min(rgb, vec3(1.0)), v_color.a);
                }
            """),
        ])
    return name


def pressure_lut(n: int = 256) -> np.ndarray:
    """음압 부호 색 지도: 파랑(−) – 연회색(0) – 빨강(+). (n,4) uint8."""
    cmap = pg.ColorMap([0.0, 0.5, 1.0], [(30, 80, 200), (225, 225, 228), (205, 30, 30)])
    return cmap.getLookupTable(0.0, 1.0, n, alpha=True)


def amplitude_lut(n: int = 256) -> np.ndarray:
    """진폭(0 이상) 색 지도 (inferno 계열이 흰 배경에서 0 근처가 어두워 보여 viridis 사용)."""
    return pg.colormap.get("viridis").getLookupTable(0.0, 1.0, n, alpha=True)


def colorize(values: np.ndarray, vmin: float, vmax: float, lut: np.ndarray, alpha_by_magnitude: bool) -> np.ndarray:
    """값 → RGBA uint8. NaN (강체) = 완전 투명. alpha_by_magnitude: |값| 이 작을수록 반투명 (단면 너머가 보이게)."""
    span = (vmax - vmin) or 1.0
    norm = np.clip((values - vmin) / span, 0.0, 1.0)
    idx = np.nan_to_num(norm * (len(lut) - 1)).astype(int)
    rgba = lut[idx].copy()
    if alpha_by_magnitude:
        mid = 0.5 if vmin < 0 < vmax else 0.0
        mag = np.clip(np.abs(np.nan_to_num(norm) - mid) / (1 - mid), 0, 1)
        rgba[..., 3] = (70 + 185 * mag).astype(np.uint8)
    else:
        rgba[..., 3] = 230
    rgba[np.isnan(values)] = 0
    return rgba


def _plane_matrix(axis: int, coord: float, origin_a: float, origin_b: float, da: float, db: float) -> QtGui.QMatrix4x4:
    """GLImageItem 국소 (u, v, w) → 전역. 축 axis 에 수직인 평면, 나머지 두 축 (a, b) = 오른손 순서 아닌 오름차순."""
    a, b = [k for k in range(3) if k != axis]
    m = np.zeros((4, 4))
    m[a, 0] = da
    m[b, 1] = db
    m[axis, 2] = 1.0
    m[axis, 3] = coord
    m[a, 3] = origin_a
    m[b, 3] = origin_b
    m[3, 3] = 1.0
    return QtGui.QMatrix4x4(*m.ravel().tolist())


class ColorBar(QtWidgets.QWidget):
    """세로 색 막대 + 최대/0/최소 값 + 단위."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(92)
        self._lut = pressure_lut()
        self._vmin, self._vmax, self._label = -1.0, 1.0, ""

    def set_range(self, vmin: float, vmax: float, label: str, lut: np.ndarray) -> None:
        self._vmin, self._vmax, self._label, self._lut = vmin, vmax, label, lut
        self.update()

    def paintEvent(self, ev) -> None:
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(*BG))
        top, bottom, x0, w = 34, self.height() - 14, 8, 18
        h = max(bottom - top, 10)
        for k in range(h):
            c = self._lut[int((1 - k / (h - 1)) * (len(self._lut) - 1))]
            p.setPen(QtGui.QColor(int(c[0]), int(c[1]), int(c[2])))
            p.drawLine(x0, top + k, x0 + w, top + k)
        p.setPen(QtGui.QColor(40, 40, 40))
        p.drawRect(x0, top, w, h)
        f = p.font()
        f.setPointSize(8)
        p.setFont(f)
        p.drawText(QtCore.QRect(2, 0, self.width() - 4, 30), QtCore.Qt.AlignmentFlag.AlignLeft
                   | QtCore.Qt.TextFlag.TextWordWrap, self._label)
        ticks = [(self._vmax, top), (self._vmin, bottom)]
        if self._vmin < 0 < self._vmax:
            ticks.append((0.0, top + h * self._vmax / (self._vmax - self._vmin)))
        for v, y in ticks:
            p.drawText(x0 + w + 4, int(y) + 4, f"{v:.3g}")
        p.end()


class View3D(QtWidgets.QWidget):
    """GLViewWidget + 시점 단추 + 색 막대."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(BG)
        self.view.setMinimumSize(400, 300)
        self.cbar = ColorBar()
        bar = QtWidgets.QHBoxLayout()
        for text, fn in (("등각", self.view_iso), ("정면 (−y 에서)", self.view_front), ("윗면 (+z 에서)", self.view_top),
                         ("측면 (+x 에서)", self.view_side), ("화면 맞춤", self.fit)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addStretch(1)
        hint = QtWidgets.QLabel("<small>왼쪽 드래그 = 회전 · 휠 = 확대 · 가운데 드래그(Ctrl+왼쪽) = 이동</small>")
        bar.addWidget(hint)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.view, 1)
        row.addWidget(self.cbar)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(bar)
        lay.addLayout(row, 1)

        self._scene: list = []  # 형상·축·격자 (장면 바꿀 때 지움)
        self._slices: dict[int, gl.GLImageItem] = {}
        self.particles = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=(0, 0, 0, 0), size=5, pxMode=True)
        self.particles.setGLOptions("opaque")  # 기본 'additive' 는 흰 배경에서 하얗게 사라짐
        self.particles.setVisible(False)
        self.view.addItem(self.particles)
        self.domain = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)

    # ------------------------------------------------------------------ 장면
    def set_scene(self, domain, bodies, transducers, arrows=()) -> None:
        """domain (x0,x1,y0,y1,z0,z1) [m]; bodies/transducers: [(verts (V,3), faces (F,3))];
        arrows: [(시작 (3,), 벡터 (3,))] 송신 방향 표시."""
        for it in self._scene:
            self.view.removeItem(it)
        self._scene = []
        self.domain = tuple(float(v) for v in domain)
        x0, x1, y0, y1, z0, z1 = self.domain

        def add(item):
            self.view.addItem(item)
            self._scene.append(item)

        # 바닥 격자 (z0 평면, 0.1 m 간격 또는 영역 크기에 맞춤)
        span = max(x1 - x0, y1 - y0)
        step = 10 ** np.floor(np.log10(span / 4)) if span > 0 else 0.1
        grid = gl.GLGridItem(color=GRID_RGBA)
        grid.setSize(x1 - x0, y1 - y0)
        grid.setSpacing(step, step)
        grid.translate((x0 + x1) / 2, (y0 + y1) / 2, z0)
        add(grid)
        # 영역 상자 (모서리 12개)
        c = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
        edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
        add(gl.GLLinePlotItem(pos=np.array([c[k] for e in edges for k in e]), mode="lines", color=BOX_RGBA,
                              width=1.0, antialias=True))
        # 좌표축 (원점 대신 영역 모서리에서, 길이 = 영역의 15%)
        L = 0.15 * span
        o = np.array([x0, y0, z0])
        for k, (name, col) in enumerate((("x", (0.85, 0.15, 0.15, 1)), ("y", (0.10, 0.60, 0.15, 1)),
                                         ("z", (0.15, 0.30, 0.85, 1)))):
            e = np.zeros(3)
            e[k] = L
            add(gl.GLLinePlotItem(pos=np.array([o, o + e]), color=col, width=3, antialias=True))
            add(gl.GLTextItem(pos=o + 1.12 * e, text=name, color=tuple(int(255 * v) for v in col)))
        shader = _register_cad_shader()
        for verts, faces in bodies:
            add(gl.GLMeshItem(vertexes=verts, faces=faces, color=BODY_RGBA, shader=shader, smooth=False,
                              drawEdges=False, glOptions="opaque"))
        for verts, faces in transducers:
            add(gl.GLMeshItem(vertexes=verts, faces=faces, color=TRANSDUCER_RGBA, shader=shader, smooth=False,
                              glOptions="opaque"))
        for start, vec in arrows:
            start, vec = np.asarray(start, float), np.asarray(vec, float)
            add(gl.GLLinePlotItem(pos=np.array([start, start + vec]), color=TRANSDUCER_RGBA, width=4, antialias=True))
            add(gl.GLScatterPlotItem(pos=(start + vec)[None], color=TRANSDUCER_RGBA, size=10, pxMode=True,
                                     glOptions="opaque"))

    # ------------------------------------------------------------------ 단면·입자
    def set_slice(self, axis: int, coord: float, rgba: np.ndarray, origin_a: float, origin_b: float, d: float) -> None:
        """축 axis 에 수직인 평면 coord [m] 에 rgba (na, nb, 4) uint8 단면. origin = 첫 셀의 아래쪽 모서리 [m]."""
        item = self._slices.get(axis)
        if item is None:
            item = gl.GLImageItem(rgba, smooth=False, glOptions="translucent")
            self.view.addItem(item)
            self._slices[axis] = item
        else:
            item.setData(rgba)
        item.setTransform(_plane_matrix(axis, coord, origin_a, origin_b, d, d))
        item.setVisible(True)

    def hide_slice(self, axis: int) -> None:
        if axis in self._slices:
            self._slices[axis].setVisible(False)

    def set_particles(self, pos: np.ndarray | None, colors: np.ndarray | None = None, size: float = 5.0) -> None:
        """pos (N,3) [m] (배율 적용된 화면 위치), colors (N,4) float 0~1. None 이면 숨김."""
        if pos is None or len(pos) == 0:
            self.particles.setVisible(False)
            return
        self.particles.setData(pos=pos, color=colors, size=size)
        self.particles.setVisible(True)

    def set_colorbar(self, vmin: float, vmax: float, label: str, lut: np.ndarray) -> None:
        self.cbar.set_range(vmin, vmax, label, lut)

    # ------------------------------------------------------------------ 시점
    def _center_and_size(self):
        x0, x1, y0, y1, z0, z1 = self.domain
        return QtGui.QVector3D((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), max(x1 - x0, y1 - y0, z1 - z0)

    def _set_camera(self, elevation: float, azimuth: float) -> None:
        c, s = self._center_and_size()
        self.view.setCameraPosition(pos=c, distance=1.6 * s, elevation=elevation, azimuth=azimuth)

    def view_iso(self) -> None:
        self._set_camera(25, -60)

    def view_front(self) -> None:
        self._set_camera(0, -90)

    def view_top(self) -> None:
        self._set_camera(90, -90)

    def view_side(self) -> None:
        self._set_camera(0, 0)

    def fit(self) -> None:
        c, s = self._center_and_size()
        self.view.setCameraPosition(pos=c, distance=1.6 * s)

    def grab_image(self) -> QtGui.QImage:
        """현재 3D 화면 (색 막대 제외). OpenGL 컨텍스트가 없으면 빈 이미지."""
        return self.view.grabFramebuffer()
