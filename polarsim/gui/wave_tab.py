"""엔진 B 탭: 로봇 프레임(막대 여러 개, 임의 각도) 방사, 2D 주파수 영역 정상상태. 벽 반사·회절 포함.

엔진 A (무한 평면파) 와 달리 소스가 막대의 방사 면에만 있으므로, 프레임 밖에서 입자 궤적이 어떻게 바뀌는지 볼 수 있다.
"""

from __future__ import annotations

import html
import json
import logging
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PySide6 import QtCore, QtWidgets

from ..config import ConfigError, Medium
from ..export import timestamped_dir
from ..metrics import SMALL_AMPLITUDE_KA
from ..fdtd import (Bar, FramePreset, WaveConfig, WaveResult, emitted_power, frame_config, parallel_gaps,
                    radial_ellipticity, solve_harmonic, wave_metrics)
from ..display import CROSSING_AREA, area_ratio, emphasis_gain, gradient_scale
from .frame_editor import FrameEditor
from .views import MetricMap, PlaneView, TrajectoryView

log = logging.getLogger(__name__)

FRAME_MS = 33
MAX_PARTICLES_PER_AXIS = 70  # 애니메이션 입자 수 상한 (축당)
DEFAULT_EMPHASIS_PCT = 60  # 자동 배율 목표 압축률 (display.py)
CROSSING_WARN_FRAC = 0.01  # 겹쳐 보이는 입자가 이 비율을 넘으면 경고 (벽 모서리 몇 점은 허용)
RADIAL_EDGES = (0, 1, 2, 4, 8, 16)  # 거리 구간 [λ]
FULL_COVERAGE = 0.95  # 이 이상이면 거리 구간을 '완전히 덮음' 으로 본다
RESONANCE_BAND = 0.15  # 차단 주파수 간격의 이 비율 안이면 공진 경고
CAVITATION_PA = 1e5  # 수면 근처 정수압 ~1 bar. 이를 넘는 음압 진폭은 캐비테이션 위험 (수심에 따라 다름)
MAPS = {
    "e": ("타원도 e = √(λ2/λ1)", (0.0, 1.0)),
    "L": ("직선성 L = 1 − λ2/λ1", (0.0, 1.0)),
    "T": ("횡성분율 T", (0.0, 1.0)),
    "rms": ("RMS 변위 [m]", None),
    "pabs": ("음압 진폭 |p| [Pa]", None),
    "Imag": ("세기 크기 |I| [W/m²]", None),
}
ARROW_LEN_LAMBDA = 0.35  # 방사 방향 화살표 길이 [λ]
SLANT_E_NOTE = 0.05  # 기울어진 막대 계단 근사로 생길 수 있는 가짜 타원도 크기 (λ/20, 측정값)


def frame_overlay(bars, wavelength: float):
    """막대 → (다각형, 방사면 선분, 방사 방향 화살표). 화살표는 각 방사 면 가운데에서 바깥 법선 방향."""
    polys, lines, arrows = [], [], []
    for b in bars:
        polys.append(b.corners())
        if not b.emits:
            continue
        c = b.corners()
        mid = (np.array([b.x1, b.y1]) + np.array([b.x2, b.y2])) / 2
        half = b.normal * b.thickness / 2
        for side, sign, edge in (("left", 1, (c[0], c[1])), ("right", -1, (c[3], c[2]))):
            if b.emit in (side, "both"):
                lines.append((tuple(edge[0]), tuple(edge[1])))
                arrows.append((tuple(mid + sign * half), tuple(sign * b.normal * ARROW_LEN_LAMBDA * wavelength)))
    return polys, lines, arrows


@dataclass
class WaveBundle:
    cfg: WaveConfig
    ref: tuple[float, float]  # 거리 통계 기준점 [m]
    forward_deg: float  # 기준점에서 '앞쪽' 방향 [°]
    res: WaveResult
    met: dict
    radial: list[dict]
    power_in: float
    power_out: float | None
    solve_s: float


def _flux_out(res: WaveResult, I: np.ndarray, margin: int = 3) -> float | None:
    """관측 영역 안쪽 margin 셀에 놓은 직사각형 경로의 바깥 방향 세기 플럭스 [W/m]. 경로가 강체를 지나면 None."""
    nx, ny = res.solid.shape
    i0, i1, j0, j1 = margin, nx - 1 - margin, margin, ny - 1 - margin
    ring = np.zeros_like(res.solid)
    ring[[i0, i1], j0:j1 + 1] = True
    ring[i0:i1 + 1, [j0, j1]] = True
    if (ring & res.solid).any():
        return None
    d = res.dx
    return float((I[i1, j0:j1 + 1, 0].sum() - I[i0, j0:j1 + 1, 0].sum()
                  + I[i0:i1 + 1, j1, 1].sum() - I[i0:i1 + 1, j0, 1].sum()) * d)


def compute(cfg: WaveConfig, ref: tuple[float, float], forward_deg: float) -> WaveBundle:
    t = time.perf_counter()
    res = solve_harmonic(cfg)
    dt = time.perf_counter() - t
    met = wave_metrics(res, cfg.medium)
    met["pabs"] = np.where(res.solid, np.nan, np.abs(res.P))
    met["Imag"] = np.linalg.norm(met["I"], axis=-1)
    radial = radial_ellipticity(res, met, ref, cfg.wavelength, RADIAL_EDGES, forward_deg=forward_deg)
    return WaveBundle(cfg, ref, forward_deg, res, met, radial, emitted_power(res, cfg), _flux_out(res, met["I"]), dt)


def _line(text: str) -> QtWidgets.QLineEdit:
    return QtWidgets.QLineEdit(text)


def _num(edit: QtWidgets.QLineEdit, name: str) -> float:
    try:
        return float(edit.text())
    except ValueError:
        raise ConfigError(f"{name}: 숫자가 아닙니다 ({edit.text()!r}).") from None


class WaveTab(QtWidgets.QWidget):
    def __init__(self, log_path: Path | None = None, parent=None):
        super().__init__(parent)
        self.log_path = log_path
        self.bundle: WaveBundle | None = None
        self.sel = (0, 0)  # 전체 격자 셀 인덱스
        self.t_sim = 0.0
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Future | None = None

        self._build_controls()
        self.plane_view = PlaneView()
        self.map_view = MetricMap()
        for v in (self.plane_view, self.map_view):
            v.plot.setLabel("bottom", "x", units="m")
            v.plot.setLabel("left", "y", units="m")
        self.traj_view = TrajectoryView(axis_names=("x", "y", "z"))
        self.info = QtWidgets.QTextBrowser()
        self.plane_view.particleClicked.connect(self._on_plane_click)
        self.map_view.particleClicked.connect(self.select_cell)

        top = QtWidgets.QSplitter()
        top.addWidget(self.plane_view)
        top.addWidget(self.map_view)
        bottom = QtWidgets.QSplitter()
        bottom.addWidget(self.traj_view)
        bottom.addWidget(self.info)
        bottom.setSizes([650, 450])
        right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        right.addWidget(top)
        right.addWidget(bottom)
        right.setSizes([600, 380])
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.controls)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(640)  # 막대 표 11열이 가로 스크롤 없이 보이도록
        split = QtWidgets.QSplitter()
        split.addWidget(scroll)
        split.addWidget(right)
        split.setSizes([700, 1000])
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(FRAME_MS)
        self.timer.timeout.connect(self._tick)
        self._poll = QtCore.QTimer(self)
        self._poll.setInterval(50)
        self._poll.timeout.connect(self._check_future)
        self.info.setHtml(
            "<b>엔진 B: 로봇 프레임 방사 (2D, 벽 반사·회절 포함)</b><br>"
            "왼쪽 '로봇 프레임' 표에서 막대를 편집하면 그림에 바로 미리보기가 나옵니다 (회색 = 막대, 주황 = 방사 면과 방향). "
            "다 되면 [계산 실행] 을 누르세요.<br>"
            "<small>주파수 영역 정상상태 해 (희소 직접법). z 방향으로 무한히 긴 막대를 가정한 2D 모델입니다.</small>")
        self.editor.changed.connect(self._preview)
        self.editor.selectionChanged.connect(lambda _r: self._preview())
        self.editor.presetLoaded.connect(self._apply_preset)
        self.editor.window_extra = self._extra_for_file
        for e in self.dom + [self.ref_x, self.ref_y, self.fwd_edit, self.c_edit, self.f_edit]:
            e.textChanged.connect(self._preview)
        self.editor.load_preset("box")

    # ------------------------------------------------------------------ 입력
    def _build_controls(self) -> None:
        self.controls = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.controls)

        g = QtWidgets.QGroupBox("매질 · 주파수")
        fl = QtWidgets.QFormLayout(g)
        self.c_edit, self.rho_edit, self.f_edit = _line("1500"), _line("1000"), _line("2700")
        fl.addRow("음속 c [m/s] (물)", self.c_edit)
        fl.addRow("밀도 ρ [kg/m³] (물)", self.rho_edit)
        fl.addRow("주파수 f [Hz]", self.f_edit)
        self.lam_label = QtWidgets.QLabel()
        fl.addRow(self.lam_label)
        for e in (self.c_edit, self.f_edit):
            e.textChanged.connect(self._update_lambda)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox("로봇 프레임 (막대, 최대 12개)")
        gl = QtWidgets.QVBoxLayout(g)
        self.editor = FrameEditor()
        gl.addWidget(self.editor)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox("거리 통계 기준점 (보통 입구 중심)")
        fl = QtWidgets.QFormLayout(g)
        self.ref_x, self.ref_y, self.fwd_edit = _line("1"), _line("0"), _line("0")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.ref_x)
        row.addWidget(self.ref_y)
        fl.addRow("기준점 x, y [m]", row)
        fl.addRow("앞쪽 방향 [°] (x 축에서 반시계)", self.fwd_edit)
        tip = QtWidgets.QLabel("<small>기준점에서 앞쪽 반평면만 거리별 타원도 표에 넣습니다. 그림의 빨간 + 와 화살표.</small>")
        tip.setWordWrap(True)
        fl.addRow(tip)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox("계산 영역 [m] (바깥에 흡수층 PML 자동 추가)")
        fl = QtWidgets.QFormLayout(g)
        self.dom = [_line(v) for v in ("-1", "5", "-3", "3")]
        row = QtWidgets.QHBoxLayout()
        for e, ph in zip(self.dom, ("x0", "x1", "y0", "y1")):
            e.setPlaceholderText(ph)
            row.addWidget(e)
        fl.addRow("x0, x1, y0, y1", row)
        self.cpw = QtWidgets.QSpinBox()
        self.cpw.setRange(10, 40)
        self.cpw.setValue(20)
        fl.addRow("파장당 셀 수", self.cpw)
        lay.addWidget(g)

        self.run_btn = QtWidgets.QPushButton("계산 실행")
        self.run_btn.setMinimumHeight(34)
        f = self.run_btn.font()
        f.setBold(True)
        self.run_btn.setFont(f)
        self.run_btn.clicked.connect(self.run)
        lay.addWidget(self.run_btn)

        g = QtWidgets.QGroupBox("표시")
        fl = QtWidgets.QFormLayout(g)
        self.map_box = QtWidgets.QComboBox()
        for key, (label, _) in MAPS.items():
            self.map_box.addItem(label, key)
        self.map_box.currentIndexChanged.connect(self._update_map)
        self.flow = QtWidgets.QCheckBox("세기(에너지 흐름) 화살표")
        self.flow.setChecked(True)
        self.flow.toggled.connect(self._update_map)
        self.auto_gain = QtWidgets.QCheckBox("자동")
        self.auto_gain.setChecked(True)
        self.emphasis = QtWidgets.QSpinBox()
        self.emphasis.setRange(5, 150)
        self.emphasis.setValue(DEFAULT_EMPHASIS_PCT)
        self.emphasis.setSuffix(" % 뭉침 강도")
        self.emphasis.setToolTip("자동 배율에서 화면 면적이 최대 몇 % 변할지 (뭉침·흩어짐 정도). 100% 근처부터 입자가 겹쳐 보입니다.")
        self.emphasis.valueChanged.connect(self._refresh_frame)
        self.gain_edit = _line("")
        self.gain_edit.setEnabled(False)
        self.auto_gain.toggled.connect(lambda on: (self.gain_edit.setEnabled(not on), self.emphasis.setEnabled(on),
                                                   self._refresh_frame()))
        self.gain_edit.editingFinished.connect(self._refresh_frame)
        gr = QtWidgets.QHBoxLayout()
        gr.addWidget(self.auto_gain)
        gr.addWidget(self.emphasis)
        gr.addWidget(self.gain_edit, 1)
        self.gain_note = QtWidgets.QLabel()
        self.gain_note.setWordWrap(True)
        self.play_btn = QtWidgets.QPushButton("일시정지")
        self.play_btn.clicked.connect(self.toggle_play)
        self.speed = QtWidgets.QDoubleSpinBox()
        self.speed.setRange(0.01, 20.0)
        self.speed.setValue(0.3)
        self.speed.setSuffix(" 주기/초")
        save = QtWidgets.QPushButton("결과 저장 (PNG, CSV, 설정, 요약)")
        save.clicked.connect(self.export_dialog)
        fl.addRow("지표 맵", self.map_box)
        fl.addRow(self.flow)
        fl.addRow("시각 배율", gr)
        fl.addRow(self.gain_note)
        fl.addRow("재생", self.play_btn)
        fl.addRow("속도", self.speed)
        fl.addRow(save)
        lay.addWidget(g)
        lay.addStretch(1)
        self._update_lambda()

    def _update_lambda(self) -> None:
        try:
            lam = float(self.c_edit.text()) / float(self.f_edit.text())
            self.lam_label.setText(f"파장 λ = {lam:.4g} m")
        except (ValueError, ZeroDivisionError):
            self.lam_label.setText("파장: 입력 확인")

    def build(self) -> tuple[WaveConfig, tuple[float, float], float]:
        medium = Medium(c=_num(self.c_edit, "음속"), rho=_num(self.rho_edit, "밀도"))
        dom = tuple(_num(e, "계산 영역") for e in self.dom)
        cfg = frame_config(self.editor.get_bars(), medium=medium, frequency=_num(self.f_edit, "주파수"), domain=dom,
                           cells_per_wavelength=float(self.cpw.value()))
        cfg.validate()
        ref = (_num(self.ref_x, "기준점 x"), _num(self.ref_y, "기준점 y"))
        return cfg, ref, _num(self.fwd_edit, "앞쪽 방향")

    # ------------------------------------------------------------------ 프레임 미리보기·예시
    def _apply_preset(self, pr) -> None:
        """예시(FramePreset) 또는 파일에서 읽은 dict 의 영역·기준점을 입력칸에 반영."""
        if isinstance(pr, FramePreset):
            dom, ref, fwd = pr.domain, pr.reference, pr.forward_deg
        else:
            dom, ref, fwd = pr.get("domain"), pr.get("reference"), pr.get("forward_deg")
        if dom:
            for e, v in zip(self.dom, dom):
                e.setText(f"{v:g}")
        if ref:
            self.ref_x.setText(f"{ref[0]:g}")
            self.ref_y.setText(f"{ref[1]:g}")
        if fwd is not None:
            self.fwd_edit.setText(f"{fwd:g}")
        self._preview()

    def _extra_for_file(self) -> dict:
        try:
            return {"domain": [float(e.text()) for e in self.dom],
                    "reference": [float(self.ref_x.text()), float(self.ref_y.text())],
                    "forward_deg": float(self.fwd_edit.text())}
        except ValueError:
            return {}

    def _preview(self) -> None:
        """표를 고칠 때마다 막대·방사 방향·기준점을 바로 그린다 (계산 전에도)."""
        bars = self.editor.try_bars()
        try:
            lam = float(self.c_edit.text()) / float(self.f_edit.text())
        except (ValueError, ZeroDivisionError):
            lam = 0.5
        polys, lines, arrows = frame_overlay(bars, lam)
        try:
            rx, ry, fwd = float(self.ref_x.text()), float(self.ref_y.text()), math.radians(float(self.fwd_edit.text()))
            arrows = arrows + [((rx, ry), (0.5 * lam * math.cos(fwd), 0.5 * lam * math.sin(fwd)))]
        except ValueError:
            rx = ry = None
        r = self.editor.current_row()
        hl = None
        if 0 <= r < self.editor.table.rowCount():
            try:
                hl = self.editor._read_row(r).corners()
            except ConfigError:
                hl = None
        for v in (self.plane_view, self.map_view):
            v.set_overlay((), lines, polys, arrows, hl)
            if rx is not None:
                marker = pg.ScatterPlotItem([rx], [ry], size=14, symbol="+", pen=pg.mkPen((200, 30, 30), width=2))
                marker.setZValue(24)
                v.plot.addItem(marker)
                v._overlay.append(marker)
        if self.bundle is None:
            try:
                x0, x1, y0, y1 = (float(e.text()) for e in self.dom)
                for v in (self.plane_view, self.map_view):
                    v.plot.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0.02)
            except ValueError:
                pass
        else:
            self._mark_stale()

    def _mark_stale(self) -> None:
        """계산 뒤에 입력이 바뀌면 그림의 장(field) 은 옛 결과임을 알린다."""
        try:
            cur = self.build()
        except ConfigError:
            cur = None
        b = self.bundle
        stale = cur is None or (cur[0], cur[1], cur[2]) != (b.cfg, b.ref, b.forward_deg)
        self.run_btn.setText("계산 실행  (입력이 바뀜 — 다시 계산하세요)" if stale else "계산 실행")

    # ------------------------------------------------------------------ 계산
    def run(self) -> None:
        try:
            cfg, ref, fwd = self.build()
        except ConfigError as e:
            QtWidgets.QMessageBox.warning(self, "입력 오류", str(e))
            return
        if self._future is not None and not self._future.done():
            return
        self.run_btn.setEnabled(False)
        self.run_btn.setText("계산 중… (보통 수 초)")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self._future = self._executor.submit(compute, cfg, ref, fwd)
        self._poll.start()

    def run_sync(self) -> None:
        self._apply(compute(*self.build()))

    def _check_future(self) -> None:
        if self._future is None or not self._future.done():
            return
        self._poll.stop()
        QtWidgets.QApplication.restoreOverrideCursor()
        self.run_btn.setEnabled(True)
        self.run_btn.setText("계산 실행")
        fut, self._future = self._future, None
        try:
            b = fut.result()
        except ConfigError as e:
            QtWidgets.QMessageBox.warning(self, "입력 오류", str(e))
            return
        except Exception:
            log.exception("wave solve failed")
            QtWidgets.QMessageBox.critical(self, "계산 오류", f"계산 중 오류가 발생했습니다.\n로그 파일: {self.log_path}")
            return
        self._apply(b)

    def _apply(self, b: WaveBundle) -> None:
        self.bundle = b
        res = b.res
        nx, ny = res.P.shape
        X, Y = np.meshgrid(res.xc, res.yc, indexing="ij")
        self._X, self._Y = X, Y
        stride = max(1, math.ceil(max(nx, ny) / MAX_PARTICLES_PER_AXIS))
        off = stride // 2
        self._sub = (slice(off, None, stride), slice(off, None, stride))
        self._stride, self._off = stride, off
        fluid_sub = ~res.solid[self._sub]
        self.plane_view.set_grid(X[self._sub], Y[self._sub], fluid_sub)
        self._preview()
        self._U_sub = res.U[self._sub]
        self._P_sub = res.P[self._sub]
        u_abs = np.linalg.norm(np.abs(self._U_sub), axis=-1)[fluid_sub]
        p_abs = np.abs(self._P_sub)[fluid_sub]
        self._u_scale = float(np.percentile(u_abs, 99)) if u_abs.size else 1.0  # 극단값 하나에 끌려가지 않게
        self._p_scale = float(np.percentile(p_abs, 99)) if p_abs.size else 1.0
        self._spacing = stride * res.dx
        # 평면 내 변위 기울기 페이저 (중앙차분). 강체 셀과 그에 붙은 셀은 기울기가 의미 없으므로 제외.
        G2 = np.empty(res.U.shape[:2] + (2, 2), complex)
        for a in range(2):
            gx, gy = np.gradient(res.U[..., a], res.dx)
            G2[..., a, 0], G2[..., a, 1] = gx, gy
        near_solid = res.solid.copy()
        near_solid[1:] |= res.solid[:-1]
        near_solid[:-1] |= res.solid[1:]
        near_solid[:, 1:] |= res.solid[:, :-1]
        near_solid[:, :-1] |= res.solid[:, 1:]
        self._G2_sub = G2[self._sub]
        self._grad_ok_sub = ~near_solid[self._sub]
        # 99 백분위: 판 모서리의 회절 특이점 몇 개에 배율이 끌려가지 않게
        self._grad_scale = gradient_scale([G2], ~near_solid, percentile=99)
        self.t_sim = 0.0
        fw = math.radians(b.forward_deg)
        tx, ty = b.ref[0] + 2 * b.cfg.wavelength * math.cos(fw), b.ref[1] + 2 * b.cfg.wavelength * math.sin(fw)
        ic, jc = int(np.argmin(np.abs(res.xc - tx))), int(np.argmin(np.abs(res.yc - ty)))
        if res.solid[ic, jc]:  # 기준점 앞 2λ 가 막대 속이면 가장 가까운 유체 셀
            d2 = np.where(res.solid, np.inf, (X - tx) ** 2 + (Y - ty) ** 2)
            ic, jc = np.unravel_index(int(np.argmin(d2)), d2.shape)
        self.select_cell(int(ic), int(jc))  # 기본 선택: 기준점 앞 2λ
        self.run_btn.setText("계산 실행")
        self._update_map()
        self._update_info()
        self._refresh_frame()
        if self.play_btn.text() == "일시정지":
            self.timer.start()

    # ------------------------------------------------------------------ 표시
    def gain(self) -> float:
        if self._grad_scale > 0:
            auto = emphasis_gain(self._grad_scale, self.emphasis.value() / 100)
        else:
            auto = 0.4 * self._spacing / self._u_scale if self._u_scale > 0 else 1.0
        if self.auto_gain.isChecked():
            self.gain_edit.setText(f"{auto:.4g}")
            return auto
        try:
            return float(self.gain_edit.text())
        except ValueError:
            self.gain_edit.setText(f"{auto:.4g}")
            return auto

    def _refresh_frame(self) -> None:
        b = self.bundle
        if b is None:
            return
        g = self.gain()
        rot = np.exp(-1j * b.res.omega * self.t_sim)
        u = np.real(self._U_sub * rot)
        p = np.real(self._P_sub * rot)
        n = int((~b.res.solid[self._sub]).sum())
        self.plane_view.set_frame(u[..., 0], u[..., 1], p, g, self._p_scale, n, "음압", "Pa")
        i, j = self.sel
        self.traj_view.set_current(np.r_[np.real(b.res.U[i, j] * rot), 0.0])
        area = area_ratio(np.real(self._G2_sub * rot), g)[self._grad_ok_sub]
        note = (f"화면 ×{g:.3g} 확대 · 지금 화면 면적비 {np.percentile(area, 1):.0%} (뭉침) ~ "
                f"{np.percentile(area, 99):.0%} (흩어짐), 1~99 백분위. 점 색 = 음압 (빨강 압축, 파랑 팽창). "
                "입자는 평형점 주위로만 진동합니다.")
        crossed = float(np.mean(area <= CROSSING_AREA))
        if crossed > CROSSING_WARN_FRAC:
            note = (f"<span style='color:#c0392b'><b>주의:</b> 입자 {crossed:.0%} 가 이웃과 겹치거나 지나쳐 보입니다 "
                    "(착시). 뭉침 강도를 낮추세요.</span><br>" + note)
        self.gain_note.setText(note)

    def _tick(self) -> None:
        if self.bundle is None:
            return
        self.t_sim += self.speed.value() / self.bundle.cfg.frequency * FRAME_MS / 1000.0
        self._refresh_frame()

    def toggle_play(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("재생")
        else:
            self.play_btn.setText("일시정지")
            if self.bundle is not None:
                self.timer.start()

    def _update_map(self) -> None:
        b = self.bundle
        if b is None:
            return
        key = self.map_box.currentData()
        label, fixed = MAPS[key]
        vals = b.met[key]
        if fixed is None:
            finite = vals[np.isfinite(vals)]
            vmin, vmax = 0.0, float(np.percentile(finite, 99)) if finite.size else 1.0
            label += " (99 백분위에서 포화)"
        else:
            vmin, vmax = fixed
        n_fluid = int((~b.res.solid).sum())
        title = f"{label} · n = {n_fluid} 셀"
        flow = b.met["I"] if self.flow.isChecked() else None
        if flow is not None:  # 화살표 길이를 로그 압축 (품 안의 큰 세기에 밖이 묻히지 않게), 방향은 그대로
            mag = np.linalg.norm(np.nan_to_num(flow), axis=-1)
            ref = np.percentile(mag[mag > 0], 50) if (mag > 0).any() else 1.0
            flow = np.nan_to_num(flow) / np.where(mag > 0, mag, 1.0)[..., None] * np.log1p(mag / ref)[..., None]
            title += " · 화살표: 세기 방향 (길이 로그 압축)"
        self.map_view.set_data(vals, self._X, self._Y, b.res.dx, vmin, vmax, title, flow)
        self.map_view.set_selected(*self.sel)

    def _on_plane_click(self, i: int, j: int) -> None:
        self.select_cell(self._off + i * self._stride, self._off + j * self._stride)

    def select_cell(self, i: int, j: int) -> None:
        b = self.bundle
        if b is None:
            return
        if b.res.solid[i, j]:
            return
        self.sel = (i, j)
        self.map_view.set_selected(i, j)
        si = min(max(round((i - self._off) / self._stride), 0), self._U_sub.shape[0] - 1)
        sj = min(max(round((j - self._off) / self._stride), 0), self._U_sub.shape[1] - 1)
        self.plane_view.set_selected(si, sj)
        U = b.res.U[i, j]
        t = np.arange(64) / 64 * 2 * np.pi
        traj = np.stack([np.real(U[0] * np.exp(-1j * t)), np.real(U[1] * np.exp(-1j * t)), 0 * t], -1)
        m = b.met
        I = m["I"][i, j]
        In = float(np.linalg.norm(I))
        lam = b.cfg.wavelength
        r = math.hypot(b.res.xc[i] - b.ref[0], b.res.yc[j] - b.ref[1]) / lam

        def fmt(v):
            return "정의 불가 (NaN)" if not np.isfinite(v) else f"{v:.3f}"

        idir = "정의 불가" if not np.isfinite(m["T"][i, j]) else f"({I[0] / In:+.2f}, {I[1] / In:+.2f})"
        html_ = (f"<b>선택 입자</b> ({b.res.xc[i]:.3f}, {b.res.yc[j]:.3f}) m · 기준점에서 {r:.2f} λ · "
                 f"RMS 변위 {m['rms'][i, j]:.3g} m · |p| {abs(b.res.P[i, j]):.3g} Pa<br>"
                 f"직선성 L = {fmt(m['L'][i, j])} · 타원도 e = {fmt(m['e'][i, j])} · 횡성분율 T = {fmt(m['T'][i, j])} · "
                 f"세기 방향 Î = {idir}<br><small>궤적은 시각 배율 없는 실제 변위 (2D 이므로 z 성분 0)</small>")
        self.traj_view.set_trajectory(traj, html_)
        self._refresh_frame()

    def _update_info(self) -> None:
        b = self.bundle
        res, cfg = b.res, b.cfg
        lam = cfg.wavelength
        ok = res.curl_ratio < 1e-10
        color = "#1e7d32" if ok else "#c0392b"
        L = [f"<b>엔진 B (2D, 주파수 영역 정상상태)</b> · 격자 {res.P.shape[0]}×{res.P.shape[1]} "
             f"(셀 {res.dx * 1000:.1f} mm = λ/{lam / res.dx:.0f}) + PML {res.pml_cells} 셀 · 풀이 {b.solve_s:.1f} s",
             f"<b>회전(∇×u) 검증</b>: <span style='color:{color}'><b>{res.curl_ratio:.1e}</b></span> "
             "<small>(엇갈림 격자에서 구조적으로 0. 유체에는 횡파가 생길 수 없음)</small>"]
        if b.power_out is not None:
            ratio = b.power_out / b.power_in if b.power_in else float("nan")
            col = "#1e7d32" if abs(ratio - 1) < 0.03 else "#c0392b"
            L.append(f"<b>에너지 보존</b>: 방사 일률 {b.power_in:.3g} W/m, 경계 밖으로 나간 세기 {b.power_out:.3g} W/m "
                     f"(<span style='color:{col}'>비 {ratio:.3f}</span>)")
        for na, nb, gap in parallel_gaps(cfg.bars):  # 마주 보는 평행 막대 사이 횡모드 공진
            spacing = cfg.medium.c / (2 * gap)  # 차단 주파수 간격 c/2D [Hz]
            m = round(cfg.frequency / spacing)
            if m >= 1 and abs(cfg.frequency - m * spacing) < RESONANCE_BAND * spacing:
                mid = [(k + 0.5) * spacing for k in (m - 1, m)]
                L.append(f"<b style='color:#b9770e'>공진 주의</b>: '{html.escape(na)}' 와 '{html.escape(nb)}' 사이 (간격 "
                         f"{gap:.3g} m) 횡모드 차단 주파수 {m * spacing:.0f} Hz (= {m}·c/2D) 근처입니다. 압력이 크게 증폭될 수 "
                         f"있고 (실제 구조에서는 손실이 제한), 피하려면 {mid[0]:.0f} 또는 {mid[1]:.0f} Hz 를 쓰세요.")
        ka = 2 * math.pi / lam * b.met["a_ref"]
        if ka > SMALL_AMPLITUDE_KA:
            L.append(f"<b style='color:#c0392b'>소진폭 가정 위반</b>: 최대 변위 {b.met['a_ref']:.3g} m, k·u = {ka:.3g} "
                     f"(> {SMALL_AMPLITUDE_KA}). 선형 음향 결과가 물리적으로 의미가 없습니다. 화면 움직임을 키우려면 진폭이 아니라 "
                     "'뭉침 강도' 를 올리세요.")
        pmax = float(np.nanmax(b.met["pabs"]))
        if pmax > CAVITATION_PA:
            L.append(f"<b style='color:#b9770e'>음압 크기</b>: 최대 {pmax:.3g} Pa ({pmax / 1e5:.1f} bar). 선형 모델 계산은 "
                     "유효하지만, 실제 물에서는 수심에 따른 정수압보다 크면 캐비테이션·비선형이 생깁니다. "
                     "진폭을 줄여도 모든 지표(L, e, T)는 그대로입니다 (선형).")
        def row(r):
            cut = r["coverage"] < FULL_COVERAGE
            style = " style='color:#999999'" if cut else ""
            note = f"{r['coverage']:.0%} (영역 잘림)" if cut else f"{r['coverage']:.0%}"
            return (f"<tr{style}><td>{r['lo']:g}–{r['hi']:g} λ</td><td align=right>{note}</td>"
                    f"<td align=right>{r['e_energy']:.3f}</td><td align=right>{r['e_median']:.3f}</td>"
                    f"<td align=right>{r['frac_e03']:.0%}</td>"
                    f"<td align=right>{1 / (math.pi * (r['lo'] + r['hi'])):.3f}</td></tr>")

        rows = "".join(row(r) for r in b.radial if r["n"] > 0)
        X, Y = self._X, self._Y
        fw = math.radians(b.forward_deg)
        along = (X - b.ref[0]) * math.cos(fw) + (Y - b.ref[1]) * math.sin(fw)
        behind = (along < 0) & (np.hypot(X - b.ref[0], Y - b.ref[1]) < cfg.wavelength) & ~res.solid
        e_behind = f"{np.nanmedian(b.met['e'][behind]):.3f}" if behind.any() else "–"
        slanted = any(abs(bb.angle_deg / 90 - round(bb.angle_deg / 90)) > 1e-9 for bb in cfg.bars)
        slant_note = (f" <small>(기울어진 막대 근처의 e &lt; {SLANT_E_NOTE} 는 계단 근사 오차일 수 있음 — 파장당 셀 수를 "
                      "늘려 같은지 확인)</small>") if slanted else ""
        L.append(f"<b>기준점 뒤쪽 1λ 이내 (보통 품 안)</b>: 타원도 e 중앙값 {e_behind}{slant_note}<br>"
                 "<b>기준점 앞쪽 반평면, 기준점으로부터 거리별</b>"
                 "<table cellspacing=4><tr><th>거리</th><th>포함률</th><th>e (에너지 가중)</th><th>e 중앙값</th>"
                 f"<th>e&gt;0.3</th><th>1/(kr)</th></tr>{rows}</table>"
                 "<small>회색 행 = 계산 영역이 그 거리의 반고리를 다 덮지 못함 → 모서리 방향만 표본이라 비교 불가. "
                 "더 먼 곳을 보려면 계산 영역을 넓히세요.<br>"
                 "에너지 가중 = 세기 |I| 로 가중한 평균 (에너지가 실제로 흐르는 곳의 궤적 모양). "
                 "지향성 영점 방향의 밝은 선은 진폭이 거의 0 인 곳이라 에너지 가중값에는 거의 기여하지 않습니다.</small>")
        if res.warnings:
            L.append("<b style='color:#b9770e'>경고</b><ul>"
                     + "".join(f"<li>{html.escape(w)}</li>" for w in res.warnings) + "</ul>")
        self.info.setHtml("<br>".join(L))

    # ------------------------------------------------------------------ 저장
    def export_dialog(self) -> None:
        if self.bundle is None:
            QtWidgets.QMessageBox.information(self, "결과 저장", "먼저 계산을 실행하세요.")
            return
        default = Path(QtCore.QStandardPaths.writableLocation(
            QtCore.QStandardPaths.StandardLocation.DocumentsLocation)) / "polarsim_results"
        default.mkdir(parents=True, exist_ok=True)
        base = QtWidgets.QFileDialog.getExistingDirectory(self, "저장할 상위 폴더", str(default))
        if not base:
            return
        try:
            out = self.export_to(Path(base))
        except PermissionError as e:
            QtWidgets.QMessageBox.warning(self, "저장 실패", f"쓰기 권한이 없거나 파일이 열려 있습니다:\n{e.filename}")
            return
        QtWidgets.QMessageBox.information(self, "저장 완료", f"저장 위치:\n{out}")

    def export_to(self, base: Path) -> Path:
        b = self.bundle
        out = timestamped_dir(base, "frame")
        res, m = b.res, b.met
        X, Y = self._X, self._Y
        cols = np.stack([X.ravel(), Y.ravel(), res.solid.ravel().astype(float), m["L"].ravel(), m["e"].ravel(),
                         m["T"].ravel(), m["rms"].ravel(), m["pabs"].ravel(), m["I"][..., 0].ravel(),
                         m["I"][..., 1].ravel()], -1)
        header = "x [m],y [m],solid,L,e,T,rms [m],|p| [Pa],Ix [W/m2],Iy [W/m2]"
        np.savetxt(out / "field_metrics.csv", cols, delimiter=",", header=header, comments="", fmt="%.8g",
                   encoding="utf-8")
        (out / "config.json").write_text(json.dumps(
            {"wave": asdict(b.cfg), "reference": list(b.ref), "forward_deg": b.forward_deg},
            ensure_ascii=False, indent=2), encoding="utf-8")
        self.editor.save_to(out / "frame.json", {"domain": list(b.cfg.domain), "reference": list(b.ref),
                                                 "forward_deg": b.forward_deg})
        (out / "summary.html").write_text(self.info.toHtml(), encoding="utf-8")
        key = self.map_box.currentData()
        for name, widget in (("particles", self.plane_view), (f"map_{key}", self.map_view),
                             ("trajectory", self.traj_view.glw)):
            exp = ImageExporter(widget.scene())
            exp.parameters()["width"] = 1800
            exp.export(str(out / f"{name}.png"))
        return out

    def shutdown(self) -> None:
        self.timer.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)
