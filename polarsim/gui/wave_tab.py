"""엔진 B 탭: AUV 품(삼면 박스) 방사, 2D 주파수 영역 정상상태. 벽 반사·회절 포함.

엔진 A (무한 평면파) 와 달리 소스가 박스 내부 면에만 있으므로, 품 밖에서 입자 궤적이 어떻게 바뀌는지 볼 수 있다.
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
from pyqtgraph.exporters import ImageExporter
from PySide6 import QtCore, QtWidgets

from ..config import ConfigError, Medium
from ..export import timestamped_dir
from ..fdtd import (PocketSpec, WaveConfig, WaveResult, emitted_power, pocket_config, radial_ellipticity,
                    solve_harmonic, wave_metrics)
from .views import MetricMap, PlaneView, TrajectoryView

log = logging.getLogger(__name__)

FRAME_MS = 33
MAX_PARTICLES_PER_AXIS = 70  # 애니메이션 입자 수 상한 (축당)
AUTO_GAIN_FRAC = 0.4
RADIAL_EDGES = (0, 1, 2, 4, 8, 16)  # 거리 구간 [λ]
FULL_COVERAGE = 0.95  # 이 이상이면 거리 구간을 '완전히 덮음' 으로 본다
CAVITATION_PA = 1e5  # 수면 근처 정수압 ~1 bar. 이를 넘는 음압 진폭은 캐비테이션 위험 (수심에 따라 다름)
MAPS = {
    "e": ("타원도 e = √(λ2/λ1)", (0.0, 1.0)),
    "L": ("직선성 L = 1 − λ2/λ1", (0.0, 1.0)),
    "T": ("횡성분율 T", (0.0, 1.0)),
    "rms": ("RMS 변위 [m]", None),
    "pabs": ("음압 진폭 |p| [Pa]", None),
    "Imag": ("세기 크기 |I| [W/m²]", None),
}
WALLS = (("back", "뒷벽 (+x 방사)", 0.0), ("top", "윗판 (−y 방사)", 90.0), ("bottom", "아랫판 (+y 방사)", 90.0))


@dataclass
class WaveBundle:
    cfg: WaveConfig
    spec: PocketSpec
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


def compute(cfg: WaveConfig, spec: PocketSpec) -> WaveBundle:
    t = time.perf_counter()
    res = solve_harmonic(cfg)
    dt = time.perf_counter() - t
    met = wave_metrics(res, cfg.medium)
    met["pabs"] = np.where(res.solid, np.nan, np.abs(res.P))
    met["Imag"] = np.linalg.norm(met["I"], axis=-1)
    radial = radial_ellipticity(res, met, (spec.length, 0.0), cfg.wavelength, RADIAL_EDGES, half_plane_x=spec.length)
    return WaveBundle(cfg, spec, res, met, radial, emitted_power(res, cfg), _flux_out(res, met["I"]), dt)


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
        scroll.setMinimumWidth(400)
        split = QtWidgets.QSplitter()
        split.addWidget(scroll)
        split.addWidget(right)
        split.setSizes([420, 1180])
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
            "<b>엔진 B: AUV 품 방사 (2D, 벽 반사·회절 포함)</b><br>"
            "삼면이 막힌 박스의 안쪽 면이 피스톤처럼 진동합니다. [계산 실행] 을 누르세요.<br>"
            "<small>주파수 영역 정상상태 해 (희소 직접법). z 방향으로 무한히 긴 판을 가정한 2D 모델입니다.</small>")

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

        g = QtWidgets.QGroupBox("품 (삼면 박스) 형상 [m]")
        fl = QtWidgets.QFormLayout(g)
        self.L_edit, self.D_edit, self.t_edit, self.body_edit = _line("1.0"), _line("1.0"), _line("0.1"), _line("0.6")
        fl.addRow("깊이 L (뒷벽→입구)", self.L_edit)
        fl.addRow("판 사이 간격 D", self.D_edit)
        fl.addRow("판 두께", self.t_edit)
        fl.addRow("뒷벽 뒤 AUV 몸체 길이", self.body_edit)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox("방사면 (면 변위 ξ0 cos(ωt − φ))")
        gl = QtWidgets.QGridLayout(g)
        for col, h in enumerate(("면", "켜기", "진폭 ξ0 [m]", "위상 φ [°]")):
            gl.addWidget(QtWidgets.QLabel(f"<b>{h}</b>"), 0, col)
        self.wall_widgets = {}
        for row, (key, label, ph) in enumerate(WALLS, start=1):
            on = QtWidgets.QCheckBox()
            on.setChecked(True)
            amp, phase = _line("1e-5"), _line(f"{ph:g}")
            gl.addWidget(QtWidgets.QLabel(label), row, 0)
            gl.addWidget(on, row, 1)
            gl.addWidget(amp, row, 2)
            gl.addWidget(phase, row, 3)
            self.wall_widgets[key] = (on, amp, phase)
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
        self.gain_edit = _line("")
        self.gain_edit.setEnabled(False)
        self.auto_gain.toggled.connect(lambda on: (self.gain_edit.setEnabled(not on), self._refresh_frame()))
        self.gain_edit.editingFinished.connect(self._refresh_frame)
        gr = QtWidgets.QHBoxLayout()
        gr.addWidget(self.auto_gain)
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

    def build(self) -> tuple[WaveConfig, PocketSpec]:
        medium = Medium(c=_num(self.c_edit, "음속"), rho=_num(self.rho_edit, "밀도"))
        walls = {}
        for key, label, _ in WALLS:
            on, amp, ph = self.wall_widgets[key]
            a = _num(amp, f"{label} 진폭")
            if a < 0 or not math.isfinite(a):
                raise ConfigError(f"{label} 진폭: 0 이상이어야 합니다.")
            walls[key] = (on.isChecked(), a, math.radians(_num(ph, f"{label} 위상")))
        geo = [_num(e, n) for e, n in ((self.L_edit, "깊이 L"), (self.D_edit, "간격 D"), (self.t_edit, "판 두께"),
                                       (self.body_edit, "몸체 길이"))]
        if min(geo) <= 0:
            raise ConfigError("품 치수는 모두 양수여야 합니다.")
        spec = PocketSpec(length=geo[0], gap=geo[1], thickness=geo[2], body_depth=geo[3], **walls)
        dom = tuple(_num(e, "계산 영역") for e in self.dom)
        cfg = pocket_config(spec, medium=medium, frequency=_num(self.f_edit, "주파수"), domain=dom,
                            cells_per_wavelength=float(self.cpw.value()))
        cfg.validate()
        x0, x1, y0, y1 = dom
        h = spec.gap / 2 + spec.thickness
        if not (x0 < -spec.body_depth and x1 > spec.length and y0 < -h and y1 > h):
            raise ConfigError("계산 영역이 품과 몸체를 모두 포함해야 합니다 (x0 < −몸체 길이, x1 > L, |y| > D/2 + 두께).")
        return cfg, spec

    # ------------------------------------------------------------------ 계산
    def run(self) -> None:
        try:
            cfg, spec = self.build()
        except ConfigError as e:
            QtWidgets.QMessageBox.warning(self, "입력 오류", str(e))
            return
        if self._future is not None and not self._future.done():
            return
        self.run_btn.setEnabled(False)
        self.run_btn.setText("계산 중… (보통 수 초)")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self._future = self._executor.submit(compute, cfg, spec)
        self._poll.start()

    def run_sync(self) -> None:
        cfg, spec = self.build()
        self._apply(compute(cfg, spec))

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
        rects = [(r.x0, r.x1, r.y0, r.y1) for r in b.cfg.solids]
        lines = []
        for e in b.cfg.emitters:
            if e.enabled and e.amplitude > 0:
                lines.append(((e.position, e.span[0]), (e.position, e.span[1])) if e.axis == "x"
                             else ((e.span[0], e.position), (e.span[1], e.position)))
        self.plane_view.set_overlay(rects, lines)
        self.map_view.set_overlay(rects, lines)
        self._U_sub = res.U[self._sub]
        self._P_sub = res.P[self._sub]
        u_abs = np.linalg.norm(np.abs(self._U_sub), axis=-1)[fluid_sub]
        p_abs = np.abs(self._P_sub)[fluid_sub]
        self._u_scale = float(np.percentile(u_abs, 99)) if u_abs.size else 1.0  # 극단값 하나에 끌려가지 않게
        self._p_scale = float(np.percentile(p_abs, 99)) if p_abs.size else 1.0
        self._spacing = stride * res.dx
        self.t_sim = 0.0
        ic = int(np.argmin(np.abs(res.xc - (b.spec.length + 2 * b.cfg.wavelength))))
        jc = int(np.argmin(np.abs(res.yc)))
        self.select_cell(ic, jc)  # 기본 선택: 입구 앞 2λ, 축 위
        self._update_map()
        self._update_info()
        self._refresh_frame()
        if self.play_btn.text() == "일시정지":
            self.timer.start()

    # ------------------------------------------------------------------ 표시
    def gain(self) -> float:
        auto = AUTO_GAIN_FRAC * self._spacing / self._u_scale if self._u_scale > 0 else 1.0
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
        shown = g * self._u_scale / self._spacing
        note = (f"화면 ×{g:.3g} 확대 (99% 입자의 실제 변위 ≤ {self._u_scale:.2g} m). 점 색 = 음압 (빨강 압축, 파랑 팽창).")
        if shown > 0.5:
            note = ("<span style='color:#c0392b'><b>주의:</b> 표시 변위가 입자 간격의 "
                    f"{shown:.0%} — 입자가 겹쳐 보일 수 있습니다 (착시).</span><br>" + note)
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
        r = math.hypot(b.res.xc[i] - b.spec.length, b.res.yc[j]) / lam

        def fmt(v):
            return "정의 불가 (NaN)" if not np.isfinite(v) else f"{v:.3f}"

        idir = "정의 불가" if not np.isfinite(m["T"][i, j]) else f"({I[0] / In:+.2f}, {I[1] / In:+.2f})"
        html_ = (f"<b>선택 입자</b> ({b.res.xc[i]:.3f}, {b.res.yc[j]:.3f}) m · 입구 중심에서 {r:.2f} λ · "
                 f"RMS 변위 {m['rms'][i, j]:.3g} m · |p| {abs(b.res.P[i, j]):.3g} Pa<br>"
                 f"직선성 L = {fmt(m['L'][i, j])} · 타원도 e = {fmt(m['e'][i, j])} · 횡성분율 T = {fmt(m['T'][i, j])} · "
                 f"세기 방향 Î = {idir}<br><small>궤적은 시각 배율 없는 실제 변위 (2D 이므로 z 성분 0)</small>")
        self.traj_view.set_trajectory(traj, html_)
        self._refresh_frame()

    def _update_info(self) -> None:
        b = self.bundle
        res, cfg, spec = b.res, b.cfg, b.spec
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
        cuts = spec.cutoff_frequencies(cfg.medium.c, cfg.frequency * 1.2)
        near = [fc for fc in cuts if abs(fc - cfg.frequency) / fc < 0.05]
        if near:
            L.append(f"<b style='color:#b9770e'>공진 주의</b>: 판 사이 횡모드 차단 주파수 {near[0]:.0f} Hz "
                     f"(= m·c/2D) 근처입니다. 품 안 압력이 크게 증폭될 수 있고, 실제 구조에서는 손실이 이를 제한합니다.")
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
        inside = (X > 0) & (X < spec.length) & (np.abs(Y) < spec.gap / 2)
        L.append(f"<b>품 안</b>: 타원도 e 중앙값 {np.nanmedian(b.met['e'][inside]):.3f}<br>"
                 "<b>품 밖 (입구 앞 반평면, 입구 중심으로부터 거리별)</b>"
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
        out = timestamped_dir(base, "pocket")
        res, m = b.res, b.met
        X, Y = self._X, self._Y
        cols = np.stack([X.ravel(), Y.ravel(), res.solid.ravel().astype(float), m["L"].ravel(), m["e"].ravel(),
                         m["T"].ravel(), m["rms"].ravel(), m["pabs"].ravel(), m["I"][..., 0].ravel(),
                         m["I"][..., 1].ravel()], -1)
        header = "x [m],y [m],solid,L,e,T,rms [m],|p| [Pa],Ix [W/m2],Iy [W/m2]"
        np.savetxt(out / "field_metrics.csv", cols, delimiter=",", header=header, comments="", fmt="%.8g",
                   encoding="utf-8")
        (out / "config.json").write_text(json.dumps({"wave": asdict(b.cfg), "pocket": asdict(b.spec)},
                                                    ensure_ascii=False, indent=2), encoding="utf-8")
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
