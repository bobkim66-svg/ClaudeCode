"""결과 저장(Qt 불필요) 과 GUI 동작(헤드리스, PySide6 없으면 건너뜀)."""

import csv
import json
import math
import os

import numpy as np
import pytest

from conftest import A, LAM, one_period
from polarsim.config import ConfigError, ObservationPlane, SimConfig, Source
from polarsim.export import (config_from_dict, config_to_dict, timestamped_dir, write_config_json,
                             write_metrics_csv, write_summary)
from polarsim.metrics import simulate
from polarsim.physics import AnalyticField
from polarsim.presets import PRESETS


# ---- physics: 애니메이션 프레임 --------------------------------------------------
def test_frame_matches_displacement(rng, medium):
    srcs = (Source(direction=(1, 2, 0), frequency=1000.0, phase=0.3),
            Source(direction=(0, 1, 1), frequency=1300.0, kind="gaussian", waist=2 * LAM))
    fld = AnalyticField(srcs, medium)
    pts = rng.normal(size=(30, 3)) * LAM
    ph = fld.phasors(pts)
    for t in (0.0, 1.234e-3, 7e-3):
        assert np.allclose(AnalyticField.frame(ph, t), fld.displacement(pts, np.array([t]))[0], atol=1e-14 * A)


# ---- export -------------------------------------------------------------------
def test_config_json_roundtrip(tmp_path):
    cfg = SimConfig(sources=(Source(direction=(1, 2, 3), phase=0.5),
                             Source(kind="gaussian", waist=0.7, ref_point=(0.1, 0, 0), enabled=False)),
                    plane=ObservationPlane(u_axis=(1, 0, 0), spacing=0.01))
    path = write_config_json(cfg, tmp_path / "config.json")
    assert config_from_dict(json.loads(path.read_text(encoding="utf-8"))) == cfg


def test_timestamped_dir_never_overwrites(tmp_path):
    a, b = timestamped_dir(tmp_path), timestamped_dir(tmp_path)
    assert a != b and a.is_dir() and b.is_dir()


def test_metrics_csv_and_summary(tmp_path):
    cfg = PRESETS["standing"].config
    cfg = SimConfig(medium=cfg.medium, sources=cfg.sources, plane=ObservationPlane(nx=4, ny=3))
    r = simulate(cfg)
    path = write_metrics_csv(r, cfg.plane, tmp_path / "m.csv")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0][7:11] == ["L", "e", "planarity", "T"]
    assert len(rows) == 1 + 12
    assert all(row[10] == "nan" for row in rows[1:])  # 정재파: T 정의 불가 → nan 그대로
    assert float(rows[1][7]) == pytest.approx(r.L[0, 0])
    text = write_summary(r, tmp_path / "s.txt").read_text(encoding="utf-8")
    assert "회전 검증" in text and "T: 범위 전부 NaN" in text


# ---- GUI (헤드리스) ------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from polarsim.app import create_app

    return create_app(["pytest"])


@pytest.fixture
def window(qapp):
    from polarsim.gui.main_window import MainWindow

    w = MainWindow(None, None)
    w.timer.stop()
    yield w
    w.close()


def test_gui_preset_roundtrip_matches_physics(window):
    """프리셋 → 위젯 → build_config 가 원래 설정과 같은 물리를 만든다 (도↔rad 변환 포함)."""
    window.load_preset("circular")
    cfg = window.build_config()
    assert cfg.sources[1].phase == pytest.approx(math.pi / 2)
    assert cfg.sources == PRESETS["circular"].config.sources
    window.run_sync(cfg)
    r = window.bundle.result
    assert r.L.shape == (40, 40) and r.curl_ratio < 1e-4


def test_gui_all_maps_and_selection(window):
    window.load_preset("hex")
    window.run_sync(window.build_config())
    window.flow.setChecked(True)
    for k in range(window.map_box.count()):
        window.map_box.setCurrentIndex(k)  # 모든 지표 맵 그리기 (NaN 포함)
    window.select_particle(3, 7)
    assert window.sel == (3, 7) and window.traj_view.traj.shape == (len(window.bundle.result.times), 3)
    for _ in range(3):
        window._tick()
    assert window.t_sim > 0


def test_gui_bad_input_is_config_error(window):
    window.load_preset("standing")
    window.sources.table.item(0, 5).setText("abc")  # 진폭 칸
    with pytest.raises(ConfigError, match="숫자가 아닙니다"):
        window.build_config()
    window.sources.table.item(0, 5).setText("1e-6")
    window.normal.edits[2].setText("0")
    with pytest.raises(ConfigError, match="영벡터"):
        window.build_config()


def test_gui_emphasis_controls_bunching(window):
    """자동 배율 = 목표 뭉침 강도. 정재파는 한 방향 압축이라 최소 면적비 = 1 − 강도 (정확)."""
    from polarsim.display import area_ratio, frame_real

    window.load_preset("standing")
    window.run_sync(window.build_config())
    b = window.bundle
    for pct in (30, 60, 90):
        window.emphasis.setValue(pct)
        g = window.gain()
        amin = min(area_ratio(frame_real(b.G2, t), g).min() for t in np.arange(64) / 64e3)
        assert amin == pytest.approx(1 - pct / 100, abs=0.02)
    window.emphasis.setValue(60)
    window._refresh_frame()
    assert "주의" not in window.gain_note.text()
    window.emphasis.setValue(150)  # 면적비 < 0 → 입자가 지나쳐 보임
    window.t_sim = 0.25e-3  # sin(ωt) = 1 인 순간 (정재파 최대 변형)
    window._refresh_frame()
    assert "착시" in window.gain_note.text()


def test_gui_gain_independent_of_amplitude(window):
    """진폭을 10배 키워도 자동 배율이 1/10 이 되어 화면 움직임은 같다."""
    window.load_preset("circular")
    window.run_sync(window.build_config())
    g1 = window.gain()
    for r in range(window.sources.table.rowCount()):
        window.sources.table.item(r, 5).setText("1e-5")
    window.run_sync(window.build_config())
    assert window.gain() == pytest.approx(g1 / 10, rel=1e-9)


def test_large_amplitude_warning():
    from polarsim.metrics import simulate

    r = simulate(SimConfig(sources=(Source(amplitude=10.0),), plane=ObservationPlane(nx=3, ny=3)))
    assert any("소진폭" in w for w in r.warnings)
    r = simulate(SimConfig(sources=(Source(amplitude=1e-6),), plane=ObservationPlane(nx=3, ny=3)))
    assert not any("소진폭" in w for w in r.warnings)


def test_gui_dimensionless_mode(window):
    window.dimless.setChecked(True)
    window.load_preset("hex")
    cfg = window.build_config()
    assert cfg.medium.c == 1.0 and cfg.sources[0].frequency == 1.0 and cfg.min_wavelength() == 1.0
    window.run_sync(cfg)
    assert window.bundle.result.curl_ratio < 1e-4


def test_gui_export(window, tmp_path):
    window.load_preset("standing")
    window.run_sync(window.build_config())
    out = window.export_to(tmp_path / "한글 경로")  # 한글·공백 경로
    names = sorted(p.name for p in out.iterdir())
    assert names == ["config.json", "map_e.png", "metrics.csv", "plane.png", "summary.txt", "trajectory.png"]
    assert all((out / n).stat().st_size > 0 for n in names)


# ---- 엔진 B 프레임 편집기 ---------------------------------------------------------
def test_frame_editor_endpoint_angle_sync(window):
    from polarsim.gui.frame_editor import C_ANG, C_LEN, C_X2, C_Y2

    ed = window.wave_tab.editor
    ed.load_preset("single")
    ed.table.item(0, C_ANG).setText("90")  # 각도를 바꾸면 끝점 2 가 다시 계산
    ed.table.item(0, C_LEN).setText("2")
    b = ed.get_bars()[0]
    assert (b.x2, b.y2) == pytest.approx((b.x1, b.y1 + 2.0))
    ed.table.item(0, C_X2).setText(str(b.x1 + 1.0))  # 끝점을 바꾸면 각도·길이 갱신
    ed.table.item(0, C_Y2).setText(str(b.y1))
    assert float(ed.table.item(0, C_ANG).text()) == pytest.approx(0.0)
    assert float(ed.table.item(0, C_LEN).text()) == pytest.approx(1.0)


def test_frame_editor_flip_keeps_physical_face(window):
    ed = window.wave_tab.editor
    ed.load_preset("single")
    ed.table.selectRow(0)
    before = ed.get_bars()[0]
    ed.flip()
    after = ed.get_bars()[0]
    assert (after.x1, after.y1, after.emit) == (before.x2, before.y2, "left")
    assert np.allclose(after.normal, -before.normal)  # 왼쪽 법선이 뒤집혀 같은 면(오른쪽→왼쪽) 을 가리킴


def test_frame_editor_buttons_and_file_roundtrip(window, tmp_path):
    import json

    from polarsim.gui.frame_editor import FrameEditor

    ed = window.wave_tab.editor
    ed.load_preset("box")
    ed.table.selectRow(1)
    ed.duplicate()
    ed.add_bar()
    assert ed.table.rowCount() == 5
    ed.table.selectRow(4)
    ed.remove()
    ed.table.selectRow(3)
    ed.move(-1)
    bars = ed.get_bars()
    ed.save_to(tmp_path / "f.json", {"domain": [-1, 5, -3, 3]})
    data = json.loads((tmp_path / "f.json").read_text(encoding="utf-8"))
    loaded = FrameEditor.bars_from_json(data)
    num = lambda b: [b.x1, b.y1, b.x2, b.y2, b.thickness, b.amplitude, b.phase]  # noqa: E731
    assert [(b.name, b.emit) for b in loaded] == [(b.name, b.emit) for b in bars]
    assert np.allclose([num(b) for b in loaded], [num(b) for b in bars], rtol=1e-12, atol=1e-15)
    assert data["domain"] == [-1, 5, -3, 3]


def test_engine_b_tilted_preset_runs(window, tmp_path):
    wt = window.wave_tab
    wt.editor.load_preset("v")
    wt.cpw.setValue(12)  # 테스트 시간 단축
    wt.run_sync()
    b = wt.bundle
    assert b.res.curl_ratio < 1e-12
    assert b.power_out / b.power_in == pytest.approx(1.0, abs=0.03)
    assert "다시 계산" not in wt.run_btn.text()
    wt.editor.table.item(0, 7).setText("0.5")  # 두께를 바꾸면 '다시 계산' 안내
    assert "다시 계산" in wt.run_btn.text()
    out = wt.export_to(tmp_path)
    assert (out / "frame.json").exists()


def test_engine_b_bad_frame_message(window):
    wt = window.wave_tab
    wt.editor.load_preset("single")
    wt.editor.table.item(0, 7).setText("0.001")  # 너무 얇음
    with pytest.raises(ConfigError, match="너무 얇아"):
        wt.build()


def test_frame_editor_no_slot_exceptions_when_switching(window, monkeypatch):
    """예시를 연달아 바꾸고 행을 선택·수정해도 Qt 슬롯 안에서 예외가 나지 않는다 (실제 앱에서는 오류 창이 뜸)."""
    import sys

    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: errors.append(a))
    wt = window.wave_tab
    for key in ("v", "fins", "single", "box", "fins"):
        wt.editor.table.selectRow(wt.editor.table.rowCount() - 1)
        wt.editor.load_preset(key)
        wt.editor.table.selectRow(1 if wt.editor.table.rowCount() > 1 else 0)
        wt.editor.table.item(0, 5).setText("12")
        wt.editor.add_bar()
        wt.editor.table.item(wt.editor.current_row(), 1).setText("abc")  # 입력 중 잘못된 값
    assert errors == []


# ---- 3D 소나 탭 -------------------------------------------------------------------
def _small_3d(tab, kind="cw"):
    """빠른 테스트용: 0.9 m 정육면체, 파장당 8 셀, 구 하나 + 송신기 하나."""
    from polarsim.engine3d.config3d import Body, Signal, Sim3DConfig, Transducer

    cfg = Sim3DConfig(signal=Signal(kind, 5000.0, cycles=3), domain=(-0.45, 0.45, -0.45, 0.45, -0.45, 0.45),
                      cells_per_wavelength=8, bodies=(Body("sphere", (0.2, 0, 0), (0.25, 0, 0), name="공"),),
                      transducers=(Transducer(center=(-0.25, 0, 0), diameter=0.2, thickness=0.08, name="소나"),))
    tab.set_config(cfg)
    return cfg


@pytest.mark.parametrize("kind", ["cw", "ping"])
def test_tab3d_run_and_all_display_modes(window, kind, monkeypatch):
    import sys

    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: errors.append(a))
    t3 = window.tab3d
    t3.timer.stop()
    cfg = _small_3d(t3, kind)
    assert t3.build() == cfg  # 표 ↔ 설정 왕복 (도 단위, ms 지연 변환 포함)
    t3.run_sync()
    t3.timer.stop()
    assert t3.res.kind == kind and t3.res.curl_ratio < 1e-4
    for mode in range(t3.mode_box.count()):
        t3.mode_box.setCurrentIndex(mode)
        for pm in range(t3.particle_box.count()):
            t3.particle_box.setCurrentIndex(pm)
            for db in (True, False):
                t3.db_check.setChecked(db)
                t3._tick()
    t3.slice_sliders[2].setValue(100)
    t3.slice_checks[0].setChecked(True)
    t3.time_slider.setValue(t3.time_slider.maximum())
    assert "다시 계산" not in t3.run_btn.text()
    t3.trs.table.item(0, 7).setText("0.25")  # 지름 변경 → 다시 계산 안내
    assert "다시 계산" in t3.run_btn.text()
    assert errors == []


def test_tab3d_gain_and_auto_range(window):
    t3 = window.tab3d
    t3.timer.stop()
    _small_3d(t3, "ping")
    t3.run_sync()
    t3.timer.stop()
    t3.auto_range.setChecked(False)
    g_fixed = t3.gain()
    t3.emphasis.setValue(80)
    assert t3.gain() == pytest.approx(2 * g_fixed)  # 확대 % 에 비례
    t3.time_slider.setValue(int(0.9 * t3.time_slider.maximum()))  # 펄스가 거의 빠져나간 뒤
    t3.auto_range.setChecked(True)
    assert t3._frame_scales()[0] < t3._p_scale  # 약한 프레임은 자동 범위가 더 좁음


def test_tab3d_bad_input_and_files(window, tmp_path):
    from polarsim.engine3d.geometry3d import box_mesh, write_stl_ascii

    t3 = window.tab3d
    t3.timer.stop()
    t3.load_preset("auv_cw")
    t3.bodies.table.item(0, 2).setText("abc")
    with pytest.raises(ConfigError, match="물체 1행 'x'"):
        t3.build()
    t3.load_preset("auv_cw")
    v, f = box_mesh((200.0, 100.0, 100.0))  # mm
    write_stl_ascii(tmp_path / "pod.stl", v[f])
    t3.add_stl(tmp_path / "pod.stl", 0.001)
    cfg = t3.build()
    assert cfg.bodies[-1].kind == "stl" and cfg.bodies[-1].size[0] == 0.001
    t3.save_file(cfg, tmp_path / "scene.json")
    t3.load_preset("piston_check")
    t3.open_file(tmp_path / "scene.json")
    assert t3.build() == cfg


def test_tab3d_export(window, tmp_path):
    t3 = window.tab3d
    t3.timer.stop()
    _small_3d(t3, "cw")
    t3.run_sync()
    t3.timer.stop()
    out = t3.export_to(tmp_path / "한글 경로")
    names = {p.name for p in out.iterdir()}
    assert {"config.json", "summary.html", "particles.csv"} <= names  # view3d.png 는 OpenGL 있을 때만
    rows = (out / "particles.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + len(t3.res.lat_pos)


def test_view3d_survives_opengl_failure(qapp, tmp_path, monkeypatch):
    """OpenGL 이 없는 PC: 그리기 오류가 예외(오류 창)로 번지지 않고, 안내 문구 + 다음 실행용 소프트웨어 GL 표시."""
    import sys

    import pyqtgraph.opengl as gl

    from polarsim.gui import view3d

    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: errors.append(a))
    monkeypatch.setattr(view3d, "software_gl_flag", lambda: tmp_path / "use_software_gl")
    monkeypatch.delenv("QT_OPENGL", raising=False)

    def broken(self, *a, **kw):
        raise RuntimeError("GLError 1282 invalid operation (모의)")

    monkeypatch.setattr(gl.GLViewWidget, "paintGL", broken)
    v = view3d.View3D()
    v.view.paintGL()
    v.view.paintGL()  # 두 번째부터는 그리지 않음 (매 프레임 예외 없음)
    qapp.processEvents()
    assert v.view.gl_error.startswith("RuntimeError") and errors == []
    assert v.stack.currentWidget() is v.gl_message and "다시 시작" in v.gl_message.text()
    assert (tmp_path / "use_software_gl").exists()
    assert v.grab_image().isNull()
