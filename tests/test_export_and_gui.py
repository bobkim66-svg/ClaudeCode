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
