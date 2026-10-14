"""메인 창: 입력 패널 + 평면 애니메이션 + 지표 맵 + 궤적 + 도움말."""

from __future__ import annotations

import html
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyqtgraph.exporters import ImageExporter
from PySide6 import QtCore, QtGui, QtWidgets

from ..config import ConfigError, Medium, ObservationPlane, SimConfig, TimeWindow
from ..export import timestamped_dir, write_config_json, write_metrics_csv, write_summary
from ..metrics import SimulationResult, simulate
from ..physics import AnalyticField, Phasors, plane_basis
from ..presets import orthogonal_quadrature, standing_wave, three_wave_hex
from .source_table import SourceTable
from .views import MetricMap, PlaneView, TrajectoryView, open_3d_view
from .wave_tab import WaveTab

log = logging.getLogger(__name__)

FRAME_MS = 33  # 애니메이션 갱신 간격 (~30 fps)
AUTO_GAIN_FRAC = 0.4  # 자동 배율: 최대 변위가 격자 간격의 40% 로 보이게
OVERLAP_FRAC = 0.5  # 표시 변위가 격자 간격의 50% 를 넘으면 착시 경고
CURL_PASS = 1e-4  # 회전 검증 통과 기준 (차분 절단오차 ~7e-6 에 여유)
EXPORT_WIDTH_PX = 1800  # PNG 저장 폭

PRESET_FACTORIES = {
    "standing": standing_wave,
    "circular": orthogonal_quadrature,
    "hex": three_wave_hex,
}
MAPS = {
    # key: (표시 이름, 고정 범위 또는 None=자동)
    "L": ("직선성 L = 1 − λ2/λ1", (0.0, 1.0)),
    "e": ("타원도 e = √(λ2/λ1)", (0.0, 1.0)),
    "planarity": ("평면성 1 − λ3/λ2", (0.0, 1.0)),
    "T": ("횡성분율 T", (0.0, 1.0)),
    "rms": ("RMS 변위 [m]", None),
    "Imag": ("세기 크기 |I| [W/m²]", None),
}


@dataclass
class Bundle:
    """한 번의 계산 결과 + 애니메이션용 캐시."""

    config: SimConfig
    result: SimulationResult
    field: AnalyticField
    phasors: list[Phasors]
    basis: np.ndarray  # (3,3) 행 = e_u, e_v, n
    su: np.ndarray  # (nx,ny) [m]
    sv: np.ndarray
    f_ref: float  # 애니메이션 기준 주파수 [Hz] (최저 활성 주파수)


def compute_bundle(config: SimConfig) -> Bundle:
    """GUI 와 무관한 계산 (작업 스레드에서 실행)."""
    result = simulate(config)
    field = AnalyticField(config.sources, config.medium)
    phasors = field.phasors(result.points.reshape(-1, 3))
    basis = np.stack(plane_basis(config.plane))
    rel = result.points - np.asarray(config.plane.origin, float)
    f_ref = min(s.frequency for s in config.active_sources())
    return Bundle(config, result, field, phasors, basis, rel @ basis[0], rel @ basis[1], f_ref)


def _fmt(v: float, nd: int = 3) -> str:
    return "정의 불가 (NaN)" if not np.isfinite(v) else f"{v:.{nd}f}"


class VecEdit(QtWidgets.QWidget):
    """3성분 벡터 입력."""

    def __init__(self, values, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.edits = [QtWidgets.QLineEdit(f"{v:g}") for v in values]
        for e, name in zip(self.edits, "xyz"):
            e.setPlaceholderText(name)
            lay.addWidget(e)

    def value(self, name: str) -> tuple[float, float, float]:
        try:
            return tuple(float(e.text()) for e in self.edits)
        except ValueError:
            raise ConfigError(f"{name}: 세 성분 모두 숫자여야 합니다.") from None

    def set_value(self, values) -> None:
        for e, v in zip(self.edits, values):
            e.setText(f"{v:g}")


def _float(edit: QtWidgets.QLineEdit, name: str) -> float:
    try:
        return float(edit.text())
    except ValueError:
        raise ConfigError(f"{name}: 숫자가 아닙니다 ({edit.text()!r}).") from None


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, guide_path: Path | None = None, log_path: Path | None = None):
        super().__init__()
        self.setWindowTitle("음파 중첩 입자 궤적 편광 시뮬레이터 (엔진 A: 평면파 해석해)")
        self.log_path = log_path
        self.bundle: Bundle | None = None
        self.sel = (0, 0)
        self.t_sim = 0.0
        self.preset_text = ""
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Future | None = None
        self._windows_3d: list[QtWidgets.QWidget] = []

        self._build_controls()
        self.plane_view = PlaneView()
        self.map_view = MetricMap()
        self.traj_view = TrajectoryView()
        self.info = QtWidgets.QTextBrowser()
        self.info.setOpenExternalLinks(False)
        for v in (self.plane_view, self.map_view):
            v.particleClicked.connect(self.select_particle)

        sim_page = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        top = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        top.addWidget(self.plane_view)
        top.addWidget(self.map_view)
        bottom = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        bottom.addWidget(self.traj_view)
        bottom.addWidget(self.info)
        bottom.setSizes([700, 400])
        sim_page.addWidget(top)
        sim_page.addWidget(bottom)
        sim_page.setSizes([600, 380])

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.controls)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(460)
        engine_a = QtWidgets.QSplitter()
        engine_a.addWidget(scroll)
        engine_a.addWidget(sim_page)
        engine_a.setSizes([500, 1100])

        self.wave_tab = WaveTab(log_path)
        self.help = QtWidgets.QTextBrowser()
        self.help.setOpenExternalLinks(True)
        self._load_guide(guide_path)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(engine_a, "엔진 A: 평면파 중첩")
        self.tabs.addTab(self.wave_tab, "엔진 B: AUV 품 방사 (벽·회절)")
        self.tabs.addTab(self.help, "도움말")
        self.setCentralWidget(self.tabs)
        self.statusBar().showMessage("프리셋을 불러오거나 소스를 편집한 뒤 [계산 실행] 을 누르세요.")

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(FRAME_MS)
        self.timer.timeout.connect(self._tick)
        self._poll = QtCore.QTimer(self)
        self._poll.setInterval(50)
        self._poll.timeout.connect(self._check_future)

    # ------------------------------------------------------------------ 입력 패널
    def _build_controls(self) -> None:
        self.controls = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.controls)

        g = QtWidgets.QGroupBox("프리셋")
        gl = QtWidgets.QVBoxLayout(g)
        row = QtWidgets.QHBoxLayout()
        self.preset_box = QtWidgets.QComboBox()
        for key, fac in PRESET_FACTORIES.items():
            self.preset_box.addItem(fac().title, key)
        load = QtWidgets.QPushButton("불러와서 실행")
        load.clicked.connect(lambda: self.load_preset(self.preset_box.currentData(), run=True))
        row.addWidget(self.preset_box, 1)
        row.addWidget(load)
        gl.addLayout(row)
        self.preset_label = QtWidgets.QLabel()
        self.preset_label.setWordWrap(True)
        gl.addWidget(self.preset_label)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox("소스 (최대 8개)")
        gl = QtWidgets.QVBoxLayout(g)
        self.sources = SourceTable()
        self.sources.setMinimumHeight(170)
        gl.addWidget(self.sources)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox("매질")
        fl = QtWidgets.QFormLayout(g)
        self.dimless = QtWidgets.QCheckBox("무차원 모드 (c = 1, ρ = 1)")
        self.dimless.toggled.connect(self._on_dimless)
        self.c_edit = QtWidgets.QLineEdit("343")
        self.rho_edit = QtWidgets.QLineEdit("1.2")
        fl.addRow(self.dimless)
        fl.addRow("음속 c [m/s]", self.c_edit)
        fl.addRow("밀도 ρ [kg/m³]", self.rho_edit)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox("관측 평면")
        fl = QtWidgets.QFormLayout(g)
        self.origin = VecEdit((0, 0, 0))
        self.normal = VecEdit((0, 0, 1))
        self.nx = QtWidgets.QSpinBox()
        self.ny = QtWidgets.QSpinBox()
        for sb in (self.nx, self.ny):
            sb.setRange(1, 200)
            sb.setValue(40)
        self.auto_spacing = QtWidgets.QCheckBox("자동 (최소 파장 / 10)")
        self.auto_spacing.setChecked(True)
        self.spacing = QtWidgets.QLineEdit("0.0343")
        self.auto_spacing.toggled.connect(lambda on: self.spacing.setEnabled(not on))
        self.spacing.setEnabled(False)
        fl.addRow("원점 [m]", self.origin)
        fl.addRow("법선", self.normal)
        fl.addRow("격자 Nx", self.nx)
        fl.addRow("격자 Ny", self.ny)
        fl.addRow("격자 간격", self.auto_spacing)
        fl.addRow("  간격 [m]", self.spacing)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox("시간창 (지표 평균 구간)")
        fl = QtWidgets.QFormLayout(g)
        self.win_mode = QtWidgets.QComboBox()
        self.win_mode.addItem("자동 (공통 주기)", "auto")
        self.win_mode.addItem("수동 (길이 지정)", "manual")
        self.win_len = QtWidgets.QLineEdit("")
        self.win_len.setPlaceholderText("비우면 최저 주파수 20주기")
        self.spp = QtWidgets.QSpinBox()
        self.spp.setRange(8, 1024)
        self.spp.setValue(64)
        fl.addRow("모드", self.win_mode)
        fl.addRow("길이 [s]", self.win_len)
        fl.addRow("주기당 표본", self.spp)
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
        self.map_box.setCurrentIndex(list(MAPS).index("e"))
        self.map_box.currentIndexChanged.connect(self._update_map)
        self.flow = QtWidgets.QCheckBox("세기(에너지 흐름) 화살표")
        self.flow.toggled.connect(self._update_map)
        self.auto_gain = QtWidgets.QCheckBox("자동")
        self.auto_gain.setChecked(True)
        self.gain_edit = QtWidgets.QLineEdit("")
        self.gain_edit.setEnabled(False)
        self.auto_gain.toggled.connect(lambda on: (self.gain_edit.setEnabled(not on), self._refresh_frame()))
        self.gain_edit.editingFinished.connect(self._refresh_frame)
        gain_row = QtWidgets.QHBoxLayout()
        gain_row.addWidget(self.auto_gain)
        gain_row.addWidget(self.gain_edit, 1)
        self.gain_note = QtWidgets.QLabel()
        self.gain_note.setWordWrap(True)
        self.play_btn = QtWidgets.QPushButton("일시정지")
        self.play_btn.clicked.connect(self.toggle_play)
        self.speed = QtWidgets.QDoubleSpinBox()
        self.speed.setRange(0.01, 20.0)
        self.speed.setValue(0.5)
        self.speed.setSingleStep(0.1)
        self.speed.setSuffix(" 주기/초")
        self.time_label = QtWidgets.QLabel("t = 0")
        b3d = QtWidgets.QPushButton("선택 입자 3D 궤적 창")
        b3d.clicked.connect(self.open_3d)
        save = QtWidgets.QPushButton("결과 저장 (PNG, CSV, 설정)")
        save.clicked.connect(self.export_dialog)
        fl.addRow("지표 맵", self.map_box)
        fl.addRow(self.flow)
        fl.addRow("시각 배율", gain_row)
        fl.addRow(self.gain_note)
        fl.addRow("재생", self.play_btn)
        fl.addRow("속도", self.speed)
        fl.addRow(self.time_label)
        fl.addRow(b3d)
        fl.addRow(save)
        lay.addWidget(g)
        lay.addStretch(1)

    def _load_guide(self, path: Path | None) -> None:
        if path is not None and Path(path).is_file():
            self.help.setMarkdown(Path(path).read_text(encoding="utf-8"))
        else:
            self.help.setPlainText(f"사용자 가이드 파일을 찾을 수 없습니다: {path}")
            log.warning("guide not found: %s", path)

    def _on_dimless(self, on: bool) -> None:
        self.c_edit.setEnabled(not on)
        self.rho_edit.setEnabled(not on)
        if on:
            self.c_edit.setText("1")
            self.rho_edit.setText("1")
        self.sources.set_units(on)

    # ------------------------------------------------------------------ 설정 ↔ 위젯
    def build_config(self) -> SimConfig:
        """위젯 값 → SimConfig (검증 포함). 실패 시 ConfigError."""
        medium = Medium(c=_float(self.c_edit, "음속 c"), rho=_float(self.rho_edit, "밀도 ρ"))
        spacing = None if self.auto_spacing.isChecked() else _float(self.spacing, "격자 간격")
        plane = ObservationPlane(origin=self.origin.value("관측 평면 원점"), normal=self.normal.value("관측 평면 법선"),
                                 nx=self.nx.value(), ny=self.ny.value(), spacing=spacing)
        mode = self.win_mode.currentData()
        dur = self.win_len.text().strip()
        window = TimeWindow(mode=mode, duration=_float(self.win_len, "시간창 길이") if dur else None,
                            samples_per_period=self.spp.value())
        cfg = SimConfig(medium=medium, sources=self.sources.get_sources(), plane=plane, window=window)
        cfg.validate()
        return cfg

    def load_preset(self, key: str, run: bool = False) -> None:
        dimless = self.dimless.isChecked()
        preset = PRESET_FACTORIES[key](Medium.dimensionless() if dimless else None, 1.0 if dimless else 1000.0)
        cfg = preset.config
        self.preset_box.setCurrentIndex(self.preset_box.findData(key))
        self.sources.set_sources(cfg.sources)
        self.origin.set_value(cfg.plane.origin)
        self.normal.set_value(cfg.plane.normal)
        self.nx.setValue(cfg.plane.nx)
        self.ny.setValue(cfg.plane.ny)
        self.auto_spacing.setChecked(True)
        self.win_mode.setCurrentIndex(0)
        self.win_len.setText("")
        self.preset_text = f"<b>{html.escape(preset.title)}</b><br>예상: {html.escape(preset.expected)}"
        self.preset_label.setText(self.preset_text)
        if run:
            self.run()

    # ------------------------------------------------------------------ 계산
    def run(self) -> None:
        try:
            cfg = self.build_config()
        except ConfigError as e:
            QtWidgets.QMessageBox.warning(self, "입력 오류", str(e))
            return
        if self._future is not None and not self._future.done():
            return
        self.run_btn.setEnabled(False)
        self.statusBar().showMessage("계산 중…")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self._future = self._executor.submit(compute_bundle, cfg)
        self._poll.start()

    def run_sync(self, cfg: SimConfig) -> None:
        """테스트/자체진단용 동기 실행."""
        self._apply_bundle(compute_bundle(cfg))

    def _check_future(self) -> None:
        if self._future is None or not self._future.done():
            return
        self._poll.stop()
        QtWidgets.QApplication.restoreOverrideCursor()
        self.run_btn.setEnabled(True)
        fut, self._future = self._future, None
        try:
            bundle = fut.result()
        except ConfigError as e:
            QtWidgets.QMessageBox.warning(self, "입력 오류", str(e))
            self.statusBar().showMessage("입력 오류")
            return
        except Exception:
            log.exception("simulation failed")
            QtWidgets.QMessageBox.critical(
                self, "계산 오류", f"계산 중 예기치 못한 오류가 발생했습니다.\n로그 파일: {self.log_path}")
            self.statusBar().showMessage("계산 오류 (로그 확인)")
            return
        self._apply_bundle(bundle)

    def _apply_bundle(self, b: Bundle) -> None:
        self.bundle = b
        r = b.result
        nx, ny = r.L.shape
        self.plane_view.set_grid(b.su, b.sv)
        self.t_sim = 0.0
        self.select_particle(nx // 2, ny // 2)
        self._update_map()
        self._update_info()
        self._refresh_frame()
        if self.play_btn.text() == "일시정지":
            self.timer.start()
        self.statusBar().showMessage(
            f"계산 완료: 입자 {r.L.size}개, 시간 표본 {len(r.times)}개, 경고 {len(r.warnings)}건")

    # ------------------------------------------------------------------ 표시 갱신
    def gain(self) -> float:
        b = self.bundle
        if b is None:
            return 1.0
        if self.auto_gain.isChecked():
            g = AUTO_GAIN_FRAC * b.result.spacing / b.result.a_ref
            self.gain_edit.setText(f"{g:.4g}")
            return g
        try:
            g = float(self.gain_edit.text())
        except ValueError:
            g = AUTO_GAIN_FRAC * b.result.spacing / b.result.a_ref
            self.gain_edit.setText(f"{g:.4g}")
        return g

    def _refresh_frame(self) -> None:
        b = self.bundle
        if b is None:
            return
        g = self.gain()
        r = b.result
        u = AnalyticField.frame(b.phasors, self.t_sim).reshape(r.L.shape + (3,))
        comp = u @ b.basis.T  # (nx,ny,3) 평면 기저 성분
        self.plane_view.set_frame(comp[..., 0], comp[..., 1], comp[..., 2], g, r.a_ref, r.L.size)
        i, j = self.sel
        self.traj_view.set_current(comp[i, j])
        ratio = r.a_ref / r.spacing
        shown = g * r.a_ref / r.spacing
        note = (f"실제 최대 변위 ≤ {r.a_ref:.3g} m = 격자 간격의 {ratio:.2e} 배. 화면은 ×{g:.3g} 확대. "
                "입자는 평형점 주위로만 진동하며 이동하지 않습니다.")
        if shown > OVERLAP_FRAC:
            note = (f"<span style='color:#c0392b'><b>주의:</b> 표시 변위가 격자 간격의 {shown:.0%} 로 입자가 "
                    "겹치거나 이동하는 것처럼 보일 수 있습니다 (착시).</span><br>" + note)
        self.gain_note.setText(note)
        periods = self.t_sim * b.f_ref
        self.time_label.setText(f"t = {self.t_sim:.4g} s ({periods:.2f} 주기, 기준 {b.f_ref:g} Hz)")

    def _tick(self) -> None:
        if self.bundle is None:
            return
        self.t_sim += self.speed.value() / self.bundle.f_ref * FRAME_MS / 1000.0
        self._refresh_frame()

    def toggle_play(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("재생")
        else:
            self.play_btn.setText("일시정지")
            if self.bundle is not None:
                self.timer.start()

    def map_values(self, key: str) -> np.ndarray:
        r = self.bundle.result
        if key == "Imag":
            return np.linalg.norm(r.intensity, axis=-1)
        return getattr(r, key)

    def _update_map(self) -> None:
        b = self.bundle
        if b is None:
            return
        key = self.map_box.currentData()
        label, fixed = MAPS[key]
        vals = self.map_values(key)
        if fixed is None:
            finite = vals[np.isfinite(vals)]
            vmin, vmax = 0.0, float(finite.max()) if finite.size else 1.0
        else:
            vmin, vmax = fixed
        n_nan = int(np.isnan(vals).sum())
        title = f"{label} · n = {vals.size}" + (f" · 회색 = 정의 불가 {n_nan}개" if n_nan else "")
        flow = None
        if self.flow.isChecked():
            flow = np.stack([b.result.intensity @ b.basis[0], b.result.intensity @ b.basis[1]], -1)
        self.map_view.set_data(vals, b.su, b.sv, b.result.spacing, vmin, vmax, title, flow)
        self.map_view.set_selected(*self.sel)

    def select_particle(self, i: int, j: int) -> None:
        self.sel = (i, j)
        b = self.bundle
        if b is None:
            return
        self.plane_view.set_selected(i, j)
        self.map_view.set_selected(i, j)
        r = b.result
        u = b.field.displacement(r.points[i, j][None, :], r.times)[:, 0, :] @ b.basis.T  # (Nt,3) u,v,n 성분
        I = r.intensity[i, j] @ b.basis.T
        In = np.linalg.norm(I)
        idir = "정의 불가 (|I|≈0)" if not np.isfinite(r.T[i, j]) or In == 0 else (
            f"({I[0] / In:+.3f}, {I[1] / In:+.3f}, {I[2] / In:+.3f})")
        html = (f"<b>선택 입자 ({i}, {j})</b> · 평면 좌표 ({b.su[i, j]:.4g}, {b.sv[i, j]:.4g}) m · "
                f"RMS 변위 {r.rms[i, j]:.3g} m<br>"
                f"직선성 L = {_fmt(r.L[i, j])} · 타원도 e = {_fmt(r.e[i, j])} · 평면성 = {_fmt(r.planarity[i, j])} · "
                f"횡성분율 T = {_fmt(r.T[i, j])}<br>"
                f"세기 방향 Î (u, v, n) = {idir} · 궤적은 시각 배율 없는 실제 변위")
        self.traj_view.set_trajectory(u, html)
        self._refresh_frame()

    def _update_info(self) -> None:
        r = self.bundle.result
        ok = r.curl_ratio < CURL_PASS
        color = "#1e7d32" if ok else "#c0392b"
        verdict = "통과" if ok else "실패 — 구현 오류 의심, 결과를 신뢰하지 마세요"
        lines = [
            f"<b>회전(∇×u) 검증</b>: max|∇×u| / max|∇u| = <span style='color:{color}'><b>{r.curl_ratio:.2e}</b> "
            f"({verdict})</span><br>"
            "<small>변위장은 비회전이어야 합니다. 엔진 A 에서는 해석적으로 0 이므로 이 값은 구현 버그 검출용이며, "
            "관찰 대상은 '횡파'가 아니라 국소 입자 궤적의 편광입니다.</small>",
            f"<b>시간창</b>: {r.window.duration:.4g} s, 표본 {r.window.n_samples}개"
            + (f", 공통 주기 {r.window.common_period:.4g} s" if r.window.common_period else ", 공통 주기 없음"),
            f"<b>격자</b>: {r.L.shape[0]}×{r.L.shape[1]}, 간격 {r.spacing:.4g} m "
            f"(최소 파장 / {self.bundle.config.min_wavelength() / r.spacing:.1f})",
        ]
        if r.warnings:
            lines.append("<b style='color:#b9770e'>경고</b><ul>" + "".join(f"<li>{html.escape(w)}</li>" for w in r.warnings) + "</ul>")
        if self.preset_text:
            lines.append(self.preset_text)
        self.info.setHtml("<br>".join(lines))

    # ------------------------------------------------------------------ 기타 동작
    def open_3d(self) -> None:
        if self.bundle is None or self.traj_view.traj is None:
            return
        try:
            w = open_3d_view(self, self.traj_view.traj,
                             f"입자 {self.sel} 3D 궤적 (빨강 u, 초록 v, 파랑 n · 최대 변위로 정규화)")
            self._windows_3d.append(w)
        except Exception:
            log.exception("OpenGL 3D view failed")
            QtWidgets.QMessageBox.warning(
                self, "3D 보기 불가",
                "OpenGL 3D 창을 열 수 없습니다 (그래픽 드라이버/원격 데스크톱 환경일 수 있음).\n"
                f"2D 투영 궤적은 정상 사용 가능합니다. 자세한 내용: {self.log_path}")

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
        """base 아래 타임스탬프 폴더에 CSV, 설정 JSON, 요약, PNG 3장 저장."""
        b = self.bundle
        out = timestamped_dir(base)
        write_metrics_csv(b.result, b.config.plane, out / "metrics.csv")
        write_config_json(b.config, out / "config.json")
        write_summary(b.result, out / "summary.txt")
        key = self.map_box.currentData()
        for name, widget in (("plane", self.plane_view), (f"map_{key}", self.map_view),
                             ("trajectory", self.traj_view.glw)):
            exp = ImageExporter(widget.scene())
            exp.parameters()["width"] = EXPORT_WIDTH_PX
            exp.export(str(out / f"{name}.png"))
        return out

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        self.timer.stop()
        self.wave_tab.shutdown()
        self._executor.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(ev)
