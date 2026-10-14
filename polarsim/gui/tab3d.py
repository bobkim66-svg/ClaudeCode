"""3D 소나 탭: 물체·송신기 편집 → 3D FDTD 계산 → CAD 처럼 돌려 보는 3D 뷰에서 물 입자 움직임과 음압 단면.

계산은 백그라운드 스레드 (화면이 멈추지 않음, 진행률·취소). 표를 고치면 형상이 바로 3D 뷰에 미리보기로 나온다.
"""

from __future__ import annotations

import html
import json
import logging
import math
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..config import ConfigError, Medium
from ..engine3d.config3d import (BODY_KIND_LABELS, MAX_BODIES, MAX_TRANSDUCERS, Body, Signal, Sim3DConfig,
                                 Transducer, from_dict, presets, to_dict)
from ..engine3d.fdtd3d import Cancelled, Result3D, estimate, run
from ..engine3d.geometry3d import body_mesh, transducer_mesh
from ..export import timestamped_dir
from ..metrics import SMALL_AMPLITUDE_KA
from .param_table import Column, ParamTable
from .view3d import View3D, amplitude_lut, colorize, pressure_lut

log = logging.getLogger(__name__)

FRAME_MS = 40
DEFAULT_SPEED = {"cw": 0.5, "ping": 2.0}  # 화면 재생 [주기/초]
DEFAULT_EMPHASIS_PCT = 40  # 화면 최대 변위 = 입자 간격의 몇 %
DEFAULT_NS_PER_CELL_STEP = 25.0  # 실행 시간 견적 (첫 계산 뒤 실측값으로 바뀜)
CAVITATION_PA = 1e5
DB_RANGE = 40.0  # dB 눈금 표시 범위
ARROW_LEN_LAMBDA = 0.6

BODY_COLUMNS = [
    Column("name", "이름", "text", "물체", width=120),
    Column("kind", "종류", "combo", "box", tuple(BODY_KIND_LABELS.items()), width=78),
    Column("x", "x", default=0.0, tip="중심 x [m] (STL 은 이동량)"),
    Column("y", "y", default=0.0, tip="중심 y [m]"),
    Column("z", "z", default=0.0, tip="중심 z [m] (위쪽 +)"),
    Column("s1", "크기1", default=0.5, tip="직육면체 = 길이(국소 x), 원기둥 = 길이, 구 = 지름, STL = 배율 (mm 파일이면 0.001) [m]"),
    Column("s2", "크기2", default=0.3, tip="직육면체 = 폭(국소 y), 원기둥 = 지름 [m]"),
    Column("s3", "크기3", default=0.3, tip="직육면체 = 높이(국소 z) [m]"),
    Column("az", "방위각°", default=0.0, tip="국소 x 축 방향: x 축에서 반시계 [°]"),
    Column("el", "고각°", default=0.0, tip="국소 x 축 방향: 수평면에서 위로 [°]"),
    Column("path", "STL 파일", "text", "", width=140, tip="종류가 STL 일 때 파일 경로"),
]
TR_COLUMNS = [
    Column("name", "이름", "text", "송신기", width=110),
    Column("enabled", "켜기", "check", True, width=40),
    Column("x", "x", default=0.0, tip="원판 중심 [m]"),
    Column("y", "y", default=0.0),
    Column("z", "z", default=0.0),
    Column("az", "방위각°", default=0.0, tip="앞면 법선 방향: x 축에서 반시계 [°]"),
    Column("el", "고각°", default=0.0, tip="앞면 법선 방향: 수평면에서 위로 [°] (아래로 쏘려면 음수)"),
    Column("diameter", "지름", default=0.2, tip="원판 지름 [m]"),
    Column("thickness", "두께", default=0.06, tip="원판 두께 [m] (뒷면은 강체)"),
    Column("amplitude", "진폭 ξ0", default=1e-6, tip="앞면 변위 진폭 [m]. 선형이라 모양에는 영향 없음 (크기만 비례)"),
    Column("phase", "위상°", default=0.0, tip="연속파 위상 φ: 변위 ξ0 cos(ωt − φ). 클수록 늦게 진동"),
    Column("delay_ms", "지연 ms", default=0.0, tip="이 송신기만 늦게 시작 [ms] (핑 시차)"),
]
SLICE_MODES = {"inst": "순간 음압 p(t)", "amp": "음압 진폭 |p| (핑: 지금까지 최대)"}
PARTICLE_MODES = {"slices": "단면 위 입자만", "all": "전체 3D 격자", "none": "숨김"}


def _line(text: str) -> QtWidgets.QLineEdit:
    return QtWidgets.QLineEdit(text)


def _num(edit: QtWidgets.QLineEdit, name: str) -> float:
    try:
        return float(edit.text())
    except ValueError:
        raise ConfigError(f"{name}: 숫자가 아닙니다 ({edit.text()!r}).") from None


def body_to_row(b: Body) -> dict:
    return dict(name=b.name, kind=b.kind, x=b.center[0], y=b.center[1], z=b.center[2], s1=b.size[0], s2=b.size[1],
                s3=b.size[2], az=b.az_deg, el=b.el_deg, path=b.path)


def row_to_body(v: dict) -> Body:
    return Body(kind=v["kind"], center=(v["x"], v["y"], v["z"]), size=(v["s1"], v["s2"], v["s3"]), az_deg=v["az"],
                el_deg=v["el"], name=v["name"], path=v["path"])


def tr_to_row(t: Transducer) -> dict:
    return dict(name=t.name, enabled=t.enabled, x=t.center[0], y=t.center[1], z=t.center[2], az=t.az_deg,
                el=t.el_deg, diameter=t.diameter, thickness=t.thickness, amplitude=t.amplitude, phase=t.phase_deg,
                delay_ms=t.delay * 1e3)


def row_to_tr(v: dict) -> Transducer:
    return Transducer(center=(v["x"], v["y"], v["z"]), az_deg=v["az"], el_deg=v["el"], diameter=v["diameter"],
                      thickness=v["thickness"], amplitude=v["amplitude"], phase_deg=v["phase"],
                      delay=v["delay_ms"] * 1e-3, enabled=v["enabled"], name=v["name"])


class Tab3D(QtWidgets.QWidget):
    def __init__(self, log_path: Path | None = None, parent=None):
        super().__init__(parent)
        self.log_path = log_path
        self.res: Result3D | None = None
        self.preset_key: str | None = None
        self.t_sim = 0.0
        self.frame = 0
        self.ns_per = DEFAULT_NS_PER_CELL_STEP
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Future | None = None
        self._progress = 0.0
        self._stop = threading.Event()
        self._file_dir: Path | None = None  # 마지막으로 열거나 저장한 설정 파일 폴더 (STL 상대경로 기준)

        self._build_controls()
        self.view = View3D()
        self.info = QtWidgets.QTextBrowser()
        right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        right.addWidget(self.view)
        right.addWidget(self.info)
        right.setSizes([760, 220])
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.controls)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(660)
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
        self._poll.setInterval(100)
        self._poll.timeout.connect(self._check_future)
        for w in (self.bodies, self.trs):
            w.changed.connect(self._preview)
        for e in self.dom + [self.c_edit, self.f_edit, self.cycles_edit]:
            e.textChanged.connect(self._preview)
        for w in self.pml_checks:
            w.toggled.connect(self._preview)
        self.cpw.valueChanged.connect(self._preview)
        self.signal_box.currentIndexChanged.connect(self._preview)
        self.info.setHtml(
            "<b>3D 소나 (수중, 3D 음향 FDTD)</b><br>왼쪽 표에서 물체(회색)와 송신기(주황)를 편집하면 3D 뷰에 바로 나타납니다. "
            "[계산 실행] 뒤 재생하면 물 입자(점)가 평형 위치 주위로 진동하고, 단면 색은 음압입니다.<br>"
            "<small>강체 물체·송신기 원판, 바깥은 흡수층(PML, 끄면 강체 벽). 선형 음향·이상 유체.</small>")
        self.load_preset("auv_cw")

    # ------------------------------------------------------------------ 입력 화면
    def _build_controls(self) -> None:
        self.controls = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.controls)

        g = QtWidgets.QGroupBox("예시 · 파일")
        gl_ = QtWidgets.QGridLayout(g)
        self.preset_box = QtWidgets.QComboBox()
        for key, p in presets().items():
            self.preset_box.addItem(p.title, key)
        load_p = QtWidgets.QPushButton("예시 불러오기")
        load_p.clicked.connect(lambda: self.load_preset(self.preset_box.currentData()))
        open_b = QtWidgets.QPushButton("설정 열기 (JSON)")
        open_b.clicked.connect(self.open_dialog)
        save_b = QtWidgets.QPushButton("설정 저장 (JSON)")
        save_b.clicked.connect(self.save_dialog)
        self.preset_note = QtWidgets.QLabel()
        self.preset_note.setWordWrap(True)
        gl_.addWidget(self.preset_box, 0, 0, 1, 2)
        gl_.addWidget(load_p, 0, 2)
        gl_.addWidget(open_b, 1, 0)
        gl_.addWidget(save_b, 1, 1)
        gl_.addWidget(self.preset_note, 2, 0, 1, 3)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox("매질 · 신호")
        fl = QtWidgets.QFormLayout(g)
        self.c_edit, self.rho_edit, self.f_edit = _line("1500"), _line("1000"), _line("5000")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("c [m/s]"))
        row.addWidget(self.c_edit)
        row.addWidget(QtWidgets.QLabel("ρ [kg/m³]"))
        row.addWidget(self.rho_edit)
        fl.addRow("물", row)
        self.signal_box = QtWidgets.QComboBox()
        self.signal_box.addItem("연속파 (정상상태)", "cw")
        self.signal_box.addItem("핑 (펄스)", "ping")
        self.cycles_edit = _line("4")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.f_edit)
        row.addWidget(self.signal_box)
        row.addWidget(QtWidgets.QLabel("핑 주기 수"))
        row.addWidget(self.cycles_edit)
        fl.addRow("주파수 f [Hz]", row)
        self.lam_label = QtWidgets.QLabel()
        fl.addRow(self.lam_label)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox(f"물체 (강체, 최대 {MAX_BODIES}개)")
        v = QtWidgets.QVBoxLayout(g)
        self.bodies = ParamTable(BODY_COLUMNS, "물체", MAX_BODIES)
        stl_b = QtWidgets.QPushButton("STL 불러오기…")
        stl_b.clicked.connect(self.add_stl_dialog)
        self.bodies.buttons.insertWidget(3, stl_b)
        tip = QtWidgets.QLabel("<small>크기: 직육면체 = 길이·폭·높이, 원기둥 = 길이·지름, 구 = 지름, STL = 배율 "
                               "(mm 로 그린 파일이면 0.001). 방위각·고각 = 물체 국소 x 축(원기둥 축) 방향. "
                               "STL 은 닫힌 솔리드여야 합니다.</small>")
        tip.setWordWrap(True)
        v.addWidget(self.bodies)
        v.addWidget(tip)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox(f"송신기 (원판, 앞면이 진동, 최대 {MAX_TRANSDUCERS}개)")
        v = QtWidgets.QVBoxLayout(g)
        self.trs = ParamTable(TR_COLUMNS, "송신기", MAX_TRANSDUCERS)
        tip = QtWidgets.QLabel("<small>방위각·고각 = 앞면이 향하는 방향 (주황 화살표). 위상차를 주면 빔이 꺾입니다 "
                               "(배열 예시 참고). 원판 두께는 셀 크기보다 커야 합니다.</small>")
        tip.setWordWrap(True)
        v.addWidget(self.trs)
        v.addWidget(tip)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox("계산 영역 [m] · 격자")
        fl = QtWidgets.QFormLayout(g)
        self.dom = [_line(t) for t in ("-1", "1", "-1", "1", "-1", "1")]
        row = QtWidgets.QHBoxLayout()
        for e, ph in zip(self.dom, ("x0", "x1", "y0", "y1", "z0", "z1")):
            e.setPlaceholderText(ph)
            e.setMaximumWidth(60)
            row.addWidget(e)
        fl.addRow("x0 x1 y0 y1 z0 z1", row)
        self.pml_checks = []
        row = QtWidgets.QHBoxLayout()
        for name in ("x−", "x+", "y−", "y+", "z− (바닥)", "z+ (수면쪽)"):
            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(True)
            self.pml_checks.append(cb)
            row.addWidget(cb)
        fl.addRow("흡수 경계", row)
        tip = QtWidgets.QLabel("<small>체크 = 파가 빠져나감 (넓은 바다), 끄면 그 면이 강체 벽 (예: 해저 바닥).</small>")
        tip.setWordWrap(True)
        fl.addRow(tip)
        self.cpw = QtWidgets.QSpinBox()
        self.cpw.setRange(6, 30)
        self.cpw.setValue(10)
        self.cpw.setToolTip("파장당 셀 수. 10 = 화면 확인용 (위상 오차 약 1.6%), 16 이상 = 정밀 (시간 약 4배 이상)")
        fl.addRow("파장당 셀 수", self.cpw)
        self.est_label = QtWidgets.QLabel()
        self.est_label.setWordWrap(True)
        fl.addRow(self.est_label)
        lay.addWidget(g)

        row = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("계산 실행")
        self.run_btn.setMinimumHeight(34)
        f = self.run_btn.font()
        f.setBold(True)
        self.run_btn.setFont(f)
        self.run_btn.clicked.connect(self.run)
        self.cancel_btn = QtWidgets.QPushButton("취소")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._stop.set)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        row.addWidget(self.run_btn, 2)
        row.addWidget(self.cancel_btn)
        row.addWidget(self.progress, 2)
        lay.addLayout(row)

        g = QtWidgets.QGroupBox("표시")
        fl = QtWidgets.QFormLayout(g)
        self.slice_checks, self.slice_sliders = [], []
        for k, name in enumerate(("x", "y", "z")):
            row = QtWidgets.QHBoxLayout()
            cb = QtWidgets.QCheckBox(f"{name} 단면")
            cb.setChecked(k != 0)
            sl = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            sl.setRange(0, 1000)
            sl.setValue(500)
            cb.toggled.connect(self._refresh)
            sl.valueChanged.connect(self._refresh)
            row.addWidget(cb)
            row.addWidget(sl, 1)
            self.slice_checks.append(cb)
            self.slice_sliders.append(sl)
            fl.addRow(row)
        self.slice_label = QtWidgets.QLabel()
        self.slice_label.setWordWrap(True)
        fl.addRow(self.slice_label)
        self.mode_box = QtWidgets.QComboBox()
        for k, label in SLICE_MODES.items():
            self.mode_box.addItem(label, k)
        self.mode_box.currentIndexChanged.connect(self._refresh)
        self.db_check = QtWidgets.QCheckBox(f"dB 눈금 (−{DB_RANGE:.0f} ~ 0 dB)")
        self.db_check.setChecked(True)
        self.db_check.toggled.connect(self._refresh)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.mode_box, 1)
        row.addWidget(self.db_check)
        fl.addRow("단면 색", row)
        self.particle_box = QtWidgets.QComboBox()
        for k, label in PARTICLE_MODES.items():
            self.particle_box.addItem(label, k)
        self.particle_box.currentIndexChanged.connect(self._refresh)
        self.emphasis = QtWidgets.QSpinBox()
        self.emphasis.setRange(5, 200)
        self.emphasis.setValue(DEFAULT_EMPHASIS_PCT)
        self.emphasis.setSuffix(" % 확대")
        self.emphasis.setToolTip("화면에서 가장 크게 움직이는 입자(상위 1% 제외)의 변위 = 입자 간격의 몇 %")
        self.emphasis.valueChanged.connect(self._refresh)
        self.psize = QtWidgets.QSpinBox()
        self.psize.setRange(1, 15)
        self.psize.setValue(5)
        self.psize.setSuffix(" px")
        self.psize.valueChanged.connect(self._refresh)
        self.auto_range = QtWidgets.QCheckBox("핑: 프레임마다 색 범위·확대 자동 (멀리 퍼져 약해진 펄스도 보이게)")
        self.auto_range.setChecked(True)
        self.auto_range.setToolTip("끄면 전체 시간 최대 기준 고정 범위 (펄스가 퍼지며 약해지는 것이 그대로 보임)")
        self.auto_range.toggled.connect(self._refresh)
        fl.addRow(self.auto_range)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.particle_box, 1)
        row.addWidget(self.emphasis)
        row.addWidget(self.psize)
        fl.addRow("물 입자", row)
        self.play_btn = QtWidgets.QPushButton("재생")
        self.play_btn.clicked.connect(self.toggle_play)
        self.speed = QtWidgets.QDoubleSpinBox()
        self.speed.setRange(0.02, 20.0)
        self.speed.setValue(0.5)
        self.speed.setSuffix(" 주기/초")
        self.speed.setToolTip("화면 재생 속도: 실제 1초에 파동 몇 주기를 보여 줄지 (실제 5 kHz 는 너무 빨라 느리게 재생)")
        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.time_slider.valueChanged.connect(self._on_time_slider)
        self.time_label = QtWidgets.QLabel("t = 0")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.play_btn)
        row.addWidget(self.speed)
        row.addWidget(self.time_slider, 1)
        row.addWidget(self.time_label)
        fl.addRow("재생", row)
        save = QtWidgets.QPushButton("결과 저장 (3D 화면 PNG, 설정, 요약, 입자 CSV)")
        save.clicked.connect(self.export_dialog)
        fl.addRow(save)
        lay.addWidget(g)
        lay.addStretch(1)

    # ------------------------------------------------------------------ 설정 ↔ 화면
    def build(self) -> Sim3DConfig:
        medium = Medium(c=_num(self.c_edit, "음속"), rho=_num(self.rho_edit, "밀도"))
        sig = Signal(self.signal_box.currentData(), _num(self.f_edit, "주파수"),
                     cycles=_num(self.cycles_edit, "핑 주기 수"))
        cfg = Sim3DConfig(medium=medium, signal=sig, domain=tuple(_num(e, "계산 영역") for e in self.dom),
                          cells_per_wavelength=float(self.cpw.value()),
                          pml_sides=tuple(cb.isChecked() for cb in self.pml_checks),
                          bodies=tuple(row_to_body(v) for v in self.bodies.all_values()),
                          transducers=tuple(row_to_tr(v) for v in self.trs.all_values()))
        cfg.validate()
        return cfg

    def set_config(self, cfg: Sim3DConfig) -> None:
        widgets = [self.c_edit, self.rho_edit, self.f_edit, self.cycles_edit, *self.dom, *self.pml_checks,
                   self.cpw, self.signal_box]
        for w in widgets:
            w.blockSignals(True)
        try:
            self.c_edit.setText(f"{cfg.medium.c:g}")
            self.rho_edit.setText(f"{cfg.medium.rho:g}")
            self.f_edit.setText(f"{cfg.signal.frequency:g}")
            self.cycles_edit.setText(f"{cfg.signal.cycles:g}")
            self.signal_box.setCurrentIndex(max(0, self.signal_box.findData(cfg.signal.kind)))
            for e, val in zip(self.dom, cfg.domain):
                e.setText(f"{val:g}")
            for cb, on in zip(self.pml_checks, cfg.pml_sides):
                cb.setChecked(on)
            self.cpw.setValue(int(round(cfg.cells_per_wavelength)))
        finally:
            for w in widgets:
                w.blockSignals(False)
        self.bodies.set_rows([body_to_row(b) for b in cfg.bodies])
        self.trs.set_rows([tr_to_row(t) for t in cfg.transducers])
        self._preview()
        self.view.view_iso()

    def load_preset(self, key: str) -> None:
        p = presets()[key]
        self.preset_key = key
        self.preset_box.setCurrentIndex(max(0, self.preset_box.findData(key)))
        self.preset_note.setText(f"<small>{html.escape(p.note)}</small>")
        self.set_config(p.config)

    def _preview(self) -> None:
        """표·입력이 바뀔 때마다 형상을 3D 뷰에 그린다. 잘못된 입력은 안내만 (마지막 형상 유지)."""
        try:
            lam = float(self.c_edit.text()) / float(self.f_edit.text())
            self.lam_label.setText(f"파장 λ = {lam:.4g} m")
        except (ValueError, ZeroDivisionError):
            lam = 0.3
            self.lam_label.setText("파장: 입력 확인")
        try:
            cfg = self.build()
        except ConfigError as e:
            self.est_label.setText(f"<span style='color:#c0392b'>입력 확인: {html.escape(str(e))}</span>")
            self._mark_stale(True)
            return
        try:
            bodies = [body_mesh(b) for b in cfg.bodies]
        except ConfigError as e:
            self.est_label.setText(f"<span style='color:#c0392b'>{html.escape(str(e))}</span>")
            return
        trs = [transducer_mesh(t) for t in cfg.transducers]
        arrows = [(np.asarray(t.center) + t.normal * t.thickness / 2, t.normal * ARROW_LEN_LAMBDA * lam)
                  for t in cfg.transducers if t.enabled]
        self.view.set_scene(cfg.domain, bodies, trs, arrows)
        est = estimate(cfg)
        secs = est["cells"] * est["steps"] * self.ns_per * 1e-9
        nx, ny, nz = est["shape"]
        color = "#c0392b" if secs > 300 else "#333333"
        self.est_label.setText(
            f"<span style='color:{color}'>격자 {nx}×{ny}×{nz} = {est['cells'] / 1e6:.2f}M 셀 (흡수층 포함), "
            f"{est['steps']} 스텝, 메모리 약 {est['mem_mb']:.0f} MB, 예상 {secs:.0f} 초</span>"
            + (" <small>(5분 넘음: 영역을 줄이거나 셀 수를 낮추세요)</small>" if secs > 300 else ""))
        self._mark_stale(self.res is not None and self.res.cfg != cfg)

    def _mark_stale(self, stale: bool) -> None:
        if self._future is None:
            self.run_btn.setText("계산 실행  (입력이 바뀜 — 다시 계산하세요)" if stale and self.res else "계산 실행")

    # ------------------------------------------------------------------ 파일
    def open_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "3D 설정 열기", str(self._file_dir or Path.home()),
                                                        "JSON (*.json)")
        if path:
            try:
                self.open_file(Path(path))
            except ConfigError as e:
                QtWidgets.QMessageBox.warning(self, "열기 실패", str(e))

    def open_file(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise ConfigError(f"파일을 읽을 수 없습니다: {e}") from None
        self._file_dir = path.parent
        self.preset_key = None
        self.preset_note.setText(f"<small>파일: {html.escape(path.name)}</small>")
        self.set_config(from_dict(data, base_dir=path.parent))

    def save_dialog(self) -> None:
        try:
            cfg = self.build()
        except ConfigError as e:
            QtWidgets.QMessageBox.warning(self, "입력 오류", str(e))
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "3D 설정 저장", str((self._file_dir or Path.home())
                                                                              / "sonar3d.json"), "JSON (*.json)")
        if path:
            self.save_file(cfg, Path(path))

    def save_file(self, cfg: Sim3DConfig, path: Path) -> None:
        path.write_text(json.dumps(to_dict(cfg, base_dir=path.parent), ensure_ascii=False, indent=2), encoding="utf-8")
        self._file_dir = path.parent

    def add_stl_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "STL 불러오기", str(self._file_dir or Path.home()),
                                                        "STL (*.stl *.STL)")
        if not path:
            return
        scale, ok = QtWidgets.QInputDialog.getDouble(
            self, "STL 단위", "배율 (STL 단위 → m). mm 로 그린 파일이면 0.001, m 이면 1:", 0.001, 1e-6, 1e3, 6)
        if ok:
            self.add_stl(Path(path), scale)

    def add_stl(self, path: Path, scale: float) -> None:
        self.bodies.add_row(dict(name=path.stem, kind="stl", x=0.0, y=0.0, z=0.0, s1=scale, s2=0.0, s3=0.0,
                                 az=0.0, el=0.0, path=str(path)))

    # ------------------------------------------------------------------ 계산
    def run(self) -> None:
        try:
            cfg = self.build()
        except ConfigError as e:
            QtWidgets.QMessageBox.warning(self, "입력 오류", str(e))
            return
        if self._future is not None:
            return
        self._stop.clear()
        self._progress = 0.0
        self.run_btn.setEnabled(False)
        self.run_btn.setText("계산 중…")
        self.cancel_btn.setEnabled(True)
        self.timer.stop()
        self._future = self._executor.submit(self._compute, cfg, self._slice_coords(cfg))
        self._poll.start()

    def _compute(self, cfg, slice_center):
        return run(cfg, progress=lambda f: setattr(self, "_progress", f), should_stop=self._stop.is_set,
                   slice_center=slice_center)

    def run_sync(self) -> None:
        cfg = self.build()
        self._apply(run(cfg, slice_center=self._slice_coords(cfg)))

    def _check_future(self) -> None:
        self.progress.setValue(int(1000 * self._progress))
        if self._future is None or not self._future.done():
            return
        self._poll.stop()
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.run_btn.setText("계산 실행")
        fut, self._future = self._future, None
        try:
            res = fut.result()
        except Cancelled:
            self.progress.setValue(0)
            return
        except ConfigError as e:
            QtWidgets.QMessageBox.warning(self, "입력 오류", str(e))
            return
        except MemoryError:
            QtWidgets.QMessageBox.warning(self, "메모리 부족", "메모리가 부족합니다. 영역을 줄이거나 파장당 셀 수를 낮추세요.")
            return
        except Exception:
            log.exception("3D solve failed")
            QtWidgets.QMessageBox.critical(self, "계산 오류", f"계산 중 오류가 발생했습니다.\n로그 파일: {self.log_path}")
            return
        self._apply(res)

    def _apply(self, res: Result3D) -> None:
        if self.res is None or self.res.kind != res.kind:  # 종류가 바뀌면 재생 속도 기본값 (핑은 한 바퀴 약 3초)
            self.speed.setValue(DEFAULT_SPEED[res.kind])
        self.res = res
        self.ns_per = res.solve_s / max(res.n_cells_total * res.n_steps, 1) * 1e9
        self.t_sim = 0.0
        self.frame = 0
        self._fframe = 0.0
        if res.kind == "ping":
            self.time_slider.setRange(0, len(res.times) - 1)
        else:
            self.time_slider.setRange(0, 63)
        self.time_slider.setValue(0)
        lat = res.lat_idx
        self._P_lat = res.P[lat[:, 0], lat[:, 1], lat[:, 2]] if res.kind == "cw" else None
        if res.kind == "cw":
            amp_u = np.linalg.norm(np.abs(res.U_lat), axis=-1)
            fluid_p = np.abs(res.P[~res.solid])
        else:
            amp_u = np.linalg.norm(res.u_frames, axis=-1).max(axis=0)
            fluid_p = res.p_max[~res.solid]
        self._u_scale = float(np.percentile(amp_u, 99)) if amp_u.size else 1.0
        self._p_scale = float(np.percentile(fluid_p, 99.5)) if fluid_p.size else 1.0
        self._u_max, self._p_max_all = float(amp_u.max(initial=0)), float(fluid_p.max(initial=0))
        self._update_info()
        self._preview()
        self._refresh()
        self.play_btn.setText("일시정지")
        self.timer.start()

    # ------------------------------------------------------------------ 표시
    def _slice_coords(self, cfg: Sim3DConfig) -> tuple[float, float, float]:
        x0, x1, y0, y1, z0, z1 = cfg.domain
        return tuple(lo + (hi - lo) * s.value() / 1000 for (lo, hi), s in
                     zip(((x0, x1), (y0, y1), (z0, z1)), self.slice_sliders))

    def _frame_scales(self) -> tuple[float, float]:
        """(압력 색 범위 [Pa], 변위 기준 [m]). 핑 + 자동이면 현재 프레임의 99 백분위, 아니면 전체 기준."""
        r = self.res
        if r.kind == "ping" and self.auto_range.isChecked():
            p = float(np.percentile(np.abs(r.p_frames[self.frame]), 99))
            u = float(np.percentile(np.linalg.norm(r.u_frames[self.frame], axis=-1), 99))
            if p > 1e-3 * self._p_scale and u > 0:  # 아직 파가 거의 없는 프레임은 고정 범위
                return p, u
        return self._p_scale, self._u_scale

    def gain(self) -> float:
        r = self.res
        u_scale = self._frame_scales()[1]
        return self.emphasis.value() / 100 * r.lat_spacing / u_scale if u_scale > 0 else 1.0

    def _slice_values(self, axis: int, idx: int) -> tuple[np.ndarray, float, float, str, np.ndarray, bool]:
        """(값 (na,nb), vmin, vmax, 색 막대 이름, lut, 투명도 사용)."""
        r = self.res
        mode = self.mode_box.currentData()
        if mode == "inst":
            if r.kind == "cw":
                vals = r.p_slice(axis, idx, t=self.t_sim)
            else:
                vals = r.p_slice(axis, idx, frame=self.frame)
            s = self._frame_scales()[0]
            auto = r.kind == "ping" and self.auto_range.isChecked()
            return vals, -s, s, "순간 음압 p [Pa]" + (" (프레임 자동)" if auto else ""), pressure_lut(), True
        sl = [slice(None)] * 3
        sl[axis] = idx
        sl = tuple(sl)
        if r.kind == "cw":
            vals = np.where(r.solid[sl], np.nan, np.abs(r.P[sl]))
        else:
            vals = r.p_max[sl]
        if self.db_check.isChecked():
            ref = self._p_max_all or 1.0
            with np.errstate(divide="ignore", invalid="ignore"):
                vals = np.maximum(20 * np.log10(vals / ref), -DB_RANGE)
            return vals, -DB_RANGE, 0.0, "|p| [dB, 최대 기준]", amplitude_lut(), False
        return vals, 0.0, self._p_scale, "|p| [Pa]", amplitude_lut(), False

    def _refresh(self) -> None:
        r = self.res
        if r is None:
            return
        cfg = r.cfg
        coords = self._slice_coords(cfg)
        axes = (r.xc, r.yc, r.zc)
        d = r.dx
        notes = []
        lut = None
        layer_masks = []
        for axis in range(3):
            if not self.slice_checks[axis].isChecked():
                self.view.hide_slice(axis)
                continue
            idx = int(np.argmin(np.abs(axes[axis] - coords[axis])))
            if r.kind == "ping" and self.mode_box.currentData() == "inst":
                if idx != r.slice_index[axis]:
                    notes.append(f"{'xyz'[axis]} 단면: 핑의 순간 음압은 계산할 때의 위치에서만 기록됩니다 → 그 위치로 표시 "
                                 "(다시 계산하면 지금 위치 반영). '음압 진폭(최대)' 은 아무 위치나 됩니다.")
                idx = r.slice_index[axis]
            vals, vmin, vmax, label, lut, alpha = self._slice_values(axis, idx)
            a, b = [k for k in range(3) if k != axis]
            self.view.set_slice(axis, axes[axis][idx], colorize(vals, vmin, vmax, lut, alpha),
                                axes[a][0] - d / 2, axes[b][0] - d / 2, d)
            layer = r.lat_idx[:, axis]
            levels = np.unique(layer)
            layer_masks.append(layer == levels[np.argmin(np.abs(levels - idx))])
        if lut is not None:
            self.view.set_colorbar(vmin, vmax, label, lut)
        self.slice_label.setText("<small>" + "<br>".join(notes) + "</small>" if notes else "")
        # 입자
        mode = self.particle_box.currentData()
        if mode == "none":
            self.view.set_particles(None)
        else:
            if r.kind == "cw":
                rot = np.exp(-1j * r.omega * self.t_sim)
                u = np.real(r.U_lat * rot)
                p = np.real(self._P_lat * rot)
            else:
                u = r.u_frames[self.frame]
                p = r.p_frames[self.frame]
            sel = np.ones(len(u), bool) if mode == "all" else (np.any(layer_masks, axis=0) if layer_masks
                                                               else np.zeros(len(u), bool))
            pos = r.lat_pos[sel] + self.gain() * u[sel]
            s = self._frame_scales()[0] or 1.0
            level = np.clip((p[sel] / s + 1) / 2, 0, 1)
            colors = pressure_lut()[(level * 255).astype(int)].astype(float) / 255
            colors[:, :3] *= 0.85  # 단면 색보다 조금 진하게 (점이 묻히지 않게)
            colors[:, 3] = 1.0
            self.view.set_particles(pos, colors, self.psize.value())
        if r.kind == "cw":
            self.time_label.setText(f"t = {self.t_sim * cfg.signal.frequency % 1:.2f} 주기")
        else:
            self.time_label.setText(f"t = {r.times[self.frame] * 1e3:.3f} ms")

    def _tick(self) -> None:
        r = self.res
        if r is None:
            return
        if r.kind == "cw":
            self.t_sim += self.speed.value() / r.cfg.signal.frequency * FRAME_MS / 1000
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(int(self.t_sim * r.cfg.signal.frequency % 1 * 64))
            self.time_slider.blockSignals(False)
        else:
            # 핑: '주기/초' 를 프레임 속도로 환산 (주기당 프레임 = 주기당 스텝 / 기록 간격). 소수 프레임 누적.
            fpp = r.steps_per_period / self._ping_every()
            self._fframe = (self._fframe + self.speed.value() * fpp * FRAME_MS / 1000) % len(r.times)
            self.frame = int(self._fframe)
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(self.frame)
            self.time_slider.blockSignals(False)
        self._refresh()

    def _ping_every(self) -> int:
        r = self.res
        return max(1, round((r.times[1] - r.times[0]) / r.dt)) if r is not None and len(r.times) > 1 else 1

    def _on_time_slider(self, v: int) -> None:
        r = self.res
        if r is None:
            return
        if r.kind == "cw":
            self.t_sim = v / 64 / r.cfg.signal.frequency
        else:
            self.frame = int(v)
            self._fframe = float(v)
        self._refresh()

    def toggle_play(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("재생")
        else:
            self.play_btn.setText("일시정지")
            if self.res is not None:
                self.timer.start()

    # ------------------------------------------------------------------ 요약
    def ping_energy_ratio(self) -> float:
        """송신이 끝난 직후 (영역 안 음향 에너지) / (송신기가 한 일). 흡수층에 닿기 전이면 1 이어야 한다."""
        r = self.res
        e = r.energy
        max_delay = max((t.delay for t in r.cfg.transducers if t.enabled), default=0.0)
        k = int(np.argmin(np.abs(e["t"] - (max_delay + r.cfg.signal.duration + 2 * r.dt))))
        return float(e["E"][k] / e["W"][k]) if e["W"][k] > 0 else float("nan")

    def _update_info(self) -> None:
        r = self.res
        cfg = r.cfg
        nx, ny, nz = r.solid.shape
        ok = r.curl_ratio < 1e-4
        col = "#1e7d32" if ok else "#c0392b"
        L = [f"<b>3D FDTD</b> · 관측 격자 {nx}×{ny}×{nz} (셀 {r.dx * 1000:.0f} mm = λ/{cfg.cells_per_wavelength:g}), "
             f"전체 {r.n_cells_total / 1e6:.2f}M 셀 · {r.n_steps} 스텝 · {r.solve_s:.1f} s "
             f"({self.ns_per:.0f} ns/셀·스텝) · 입자 {len(r.lat_pos)}개 (간격 {r.lat_spacing * 100:.1f} cm)",
             f"<b>회전(∇×v) 검증</b>: <span style='color:{col}'><b>{r.curl_ratio:.1e}</b></span> "
             "<small>(엇갈림 격자에서 구조적으로 0 → 물에는 종파만. float32 반올림 수준이면 정상)</small>"]
        if r.kind == "cw":
            c2 = "#1e7d32" if r.convergence < 0.02 else "#b9770e"
            L.append(f"<b>정상상태</b>: 마지막 두 평균 창 차이 <span style='color:{c2}'>{r.convergence:.1e}</span>")
        else:
            ratio = self.ping_energy_ratio()
            c2 = "#1e7d32" if abs(ratio - 1) < 0.05 else "#b9770e"
            L.append(f"<b>에너지 보존</b>: 송신 직후 (영역 안 음향 에너지) / (송신기가 한 일) = "
                     f"<span style='color:{c2}'>{ratio:.3f}</span> <small>(펄스가 흡수층·강체벽 근처에 이미 닿았으면 1 과 "
                     "다를 수 있음)</small>")
        if self.preset_key == "piston_check" and cfg == presets()["piston_check"].config and r.kind == "cw":
            from ..engine3d.analytic3d import piston_on_axis

            t = cfg.transducers[0]
            i, j = int(np.argmin(np.abs(r.xc - t.center[0]))), int(np.argmin(np.abs(r.yc - t.center[1])))
            sel = r.zc > 0.3
            num = np.abs(r.P[i, j, sel])
            ana = piston_on_axis(r.zc[sel], t.diameter / 2, r.omega / cfg.medium.c, cfg.medium.rho, cfg.medium.c,
                                 r.omega * t.amplitude)
            err = np.sqrt(np.mean((num - ana) ** 2)) / ana.mean()
            L.append(f"<b>해석해 비교 (배플 피스톤 축 위, z &gt; 0.3 m)</b>: RMS 상대오차 {err:.1%} "
                     "<small>(파장당 셀 10/16/24 → 5.9/2.2/0.4%, 원판 계단 근사가 주원인)</small>")
        ka = 2 * math.pi / cfg.wavelength * self._u_max
        if ka > SMALL_AMPLITUDE_KA:
            L.append(f"<b style='color:#c0392b'>소진폭 가정 위반</b>: 최대 변위 {self._u_max:.3g} m, k·u = {ka:.3g}. "
                     "선형 결과가 물리적으로 의미 없습니다. 화면 움직임은 진폭 대신 '확대' 를 올리세요.")
        if self._p_max_all > CAVITATION_PA:
            L.append(f"<b style='color:#b9770e'>음압 크기</b>: 최대 {self._p_max_all:.3g} Pa. 수심 정수압보다 크면 실제 물에서는 "
                     "캐비테이션·비선형이 생깁니다 (선형이므로 진폭을 줄이면 모양은 같고 크기만 비례).")
        L.append(f"최대 변위 {self._u_max:.3g} m · 최대 음압 {self._p_max_all:.3g} Pa · 화면 확대 ×{self.gain():.3g} "
                 "<small>(점 = 물 입자, 평형 위치 + 변위×확대. 점 색 = 순간 음압: 빨강 압축, 파랑 팽창)</small>")
        if r.warnings:
            L.append("<b style='color:#b9770e'>경고</b><ul>" + "".join(f"<li>{html.escape(w)}</li>" for w in r.warnings)
                     + "</ul>")
        self.info.setHtml("<br>".join(L))

    # ------------------------------------------------------------------ 저장
    def export_dialog(self) -> None:
        if self.res is None:
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
        r = self.res
        out = timestamped_dir(base, "sonar3d")
        img = self.view.grab_image()
        if img.isNull() or not img.save(str(out / "view3d.png")):  # OpenGL 없는 환경 (원격 데스크톱 등)
            log.warning("3D 화면 캡처 실패 (OpenGL 컨텍스트 없음)")
        self.save_file(r.cfg, out / "config.json")
        (out / "summary.html").write_text(self.info.toHtml(), encoding="utf-8")
        if r.kind == "cw":
            u_amp = np.linalg.norm(np.abs(r.U_lat), axis=-1)
            p_amp = np.abs(self._P_lat)
            head = "x [m],y [m],z [m],|u| amplitude [m],|p| amplitude [Pa]"
        else:
            u_amp = np.linalg.norm(r.u_frames, axis=-1).max(axis=0)
            p_amp = np.abs(r.p_frames).max(axis=0)
            head = "x [m],y [m],z [m],max |u| [m],max |p| [Pa]"
        np.savetxt(out / "particles.csv", np.column_stack([r.lat_pos, u_amp, p_amp]), delimiter=",", header=head,
                   comments="", fmt="%.8g", encoding="utf-8")
        return out

    def shutdown(self) -> None:
        self.timer.stop()
        self._stop.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
