"""지표 함수 단위 테스트, 시간창, 경계 입력 (요구 테스트 8), 가우시안 빔, 프리셋 예상 결과."""

import dataclasses

import numpy as np
import pytest

from conftest import A, C, F, LAM
from polarsim.config import (ConfigError, MetricSettings, ObservationPlane, SimConfig, Source, TimeWindow)
from polarsim.metrics import choose_window, common_period, simulate, simulate_points, trajectory_metrics
from polarsim.physics import gaussian_replica_period
from polarsim.presets import PRESETS

ST = MetricSettings()


# ---- 지표 함수 (합성 궤적) ---------------------------------------------------------
def _traj(fn, n=64):
    t = np.arange(n) / n * 2 * np.pi
    return np.stack(fn(t), -1)[:, None, :]  # (Nt,1,3)


def test_metrics_circle_line_ellipse_3d():
    circ = trajectory_metrics(_traj(lambda t: (np.cos(t), np.sin(t), 0 * t)), 1.0, ST)
    assert circ["e"][0] == pytest.approx(1) and circ["L"][0] == pytest.approx(0) and circ["planarity"][0] == pytest.approx(1)
    line = trajectory_metrics(_traj(lambda t: (np.cos(t), 2 * np.cos(t), 0 * t)), 1.0, ST)
    assert line["L"][0] == pytest.approx(1) and np.isnan(line["planarity"][0])
    ell = trajectory_metrics(_traj(lambda t: (2 * np.cos(t), np.sin(t), 0 * t)), 1.0, ST)
    assert ell["e"][0] == pytest.approx(0.5)  # 반축비 1:2 → √(λ2/λ1) = 1/2
    knot = trajectory_metrics(_traj(lambda t: (np.cos(t), np.sin(t), np.cos(2 * t))), 1.0, ST)
    assert knot["planarity"][0] < 0.5  # 평면을 벗어난 3D 궤적
    zero = trajectory_metrics(np.zeros((64, 1, 3)), 1.0, ST)
    assert np.isnan(zero["L"][0]) and np.isnan(zero["e"][0]) and zero["node"][0]


# ---- 시간창 ---------------------------------------------------------------------
def test_common_period():
    assert common_period([1000.0], 50) == pytest.approx(1e-3)
    assert common_period([1000.0, 1100.0], 50) == pytest.approx(1e-2)  # 맥놀이 100 Hz
    assert common_period([1000.0, 1000.0 * np.sqrt(2)], 50) is None  # 무리수 비
    assert common_period([1000.0, 1001.0], 50) is None  # 공통 주기 1 s = 1000 주기 > 50


def test_window_warnings_and_sampling():
    cfg = SimConfig(sources=(Source(frequency=1000.0), Source(direction=(0, 1, 0), frequency=1100.0)))
    t, info = choose_window(cfg)
    assert info.duration == pytest.approx(1e-2) and len(t) == int(np.ceil(1e-2 * 1100 * 64))
    assert t[-1] < info.duration  # 끝점 제외
    assert any("맥놀이" in w for w in info.warnings)
    cfg2 = dataclasses.replace(cfg, sources=(Source(frequency=1000.0), Source(frequency=1000.0 * np.sqrt(2))))
    _, info2 = choose_window(cfg2)
    assert info2.common_period is None and any("공통 주기" in w for w in info2.warnings)
    cfg3 = dataclasses.replace(cfg, window=TimeWindow(mode="manual", duration=10.0))
    with pytest.raises(ConfigError, match="앨리어싱"):
        choose_window(cfg3)


def test_beat_metrics_need_full_window():
    """맥놀이: 공통 주기 전체 평균은 짧은 창(한 주기) 평균과 다르다 → 경고가 필요한 이유."""
    src = (Source(frequency=1000.0), Source(direction=(0, 1, 0), frequency=1100.0))
    pts = np.zeros((1, 3))
    full = simulate_points(SimConfig(sources=src), pts)
    short = simulate_points(SimConfig(sources=src, window=TimeWindow(mode="manual", duration=1e-3)), pts)
    assert full.e[0] == pytest.approx(1.0, abs=1e-6)  # 장시간 평균: x, y 진동이 무상관 → 원과 같은 공분산
    assert abs(short.e[0] - full.e[0]) > 0.05


# ---- 요구 테스트 8: 경계 입력 ------------------------------------------------------
@pytest.mark.parametrize("kwargs, msg", [
    (dict(direction=(0, 0, 0)), "영벡터"),
    (dict(frequency=0.0), "주파수"),
    (dict(frequency=-5.0), "주파수"),
    (dict(frequency=float("nan")), "주파수"),
    (dict(amplitude=-1.0), "진폭"),
    (dict(direction=(1, float("inf"), 0)), "NaN/inf"),
    (dict(kind="gaussian"), "허리 반경"),
    (dict(kind="gaussian", waist=0.0), "허리 반경"),
    (dict(kind="spherical"), "종류"),
])
def test_bad_source_inputs(kwargs, msg):
    with pytest.raises(ConfigError, match=msg):
        simulate(SimConfig(sources=(Source(**kwargs),)))


@pytest.mark.parametrize("cfg, msg", [
    (SimConfig(sources=(Source(enabled=False),)), "활성화된 소스"),
    (SimConfig(sources=(Source(amplitude=0.0),)), "진폭이 모두 0"),
    (SimConfig(sources=tuple(Source() for _ in range(9))), "최대 8"),
    (SimConfig(plane=ObservationPlane(nx=0)), "nx"),
    (SimConfig(plane=ObservationPlane(ny=2.5)), "ny"),
    (SimConfig(plane=ObservationPlane(normal=(0, 0, 0))), "영벡터"),
    (SimConfig(plane=ObservationPlane(normal=(0, 0, 1), u_axis=(0, 0, 3))), "평행"),
    (SimConfig(plane=ObservationPlane(spacing=-1.0)), "격자 간격"),
    (SimConfig(window=TimeWindow(samples_per_period=4)), "표본"),
])
def test_bad_config_inputs(cfg, msg):
    with pytest.raises(ConfigError, match=msg):
        simulate(cfg)


def test_single_particle_grid():
    """N=1 격자는 에러가 아니라 입자 1개로 정상 동작 (회전 검증은 3D 오프셋 점을 쓰므로 가능)."""
    r = simulate(SimConfig(plane=ObservationPlane(nx=1, ny=1)))
    assert r.L.shape == (1, 1) and r.L[0, 0] == pytest.approx(1.0)
    assert np.isfinite(r.curl_ratio)


def test_disabled_and_zero_sources_ignored():
    base = SimConfig(sources=(Source(),))
    extra = SimConfig(sources=(Source(), Source(direction=(0, 1, 0), enabled=False),
                               Source(direction=(0, 0, 1), amplitude=0.0)))
    pts = np.random.default_rng(1).normal(size=(5, 3)) * LAM
    assert np.allclose(simulate_points(base, pts).intensity, simulate_points(extra, pts).intensity)


def test_coarse_grid_warning():
    r = simulate(SimConfig(plane=ObservationPlane(nx=3, ny=3, spacing=LAM / 4)))
    assert any("앨리어싱" in w for w in r.warnings)
    r = simulate(SimConfig(plane=ObservationPlane(nx=3, ny=3, spacing=LAM / 10)))
    assert not any("앨리어싱" in w for w in r.warnings)


# ---- 가우시안 빔 ---------------------------------------------------------------
def test_gaussian_beam_profile():
    w0 = 3 * LAM
    src = Source(direction=(0, 0, 1), amplitude=A, kind="gaussian", waist=w0)
    # 초점 평면 (z=0) 의 x 축 위 점들: 축 위, ρ = w0, ρ = 3 w0
    pts = np.array([[0, 0, 0], [w0, 0, 0], [3 * w0, 0, 0]], float)
    r = simulate_points(SimConfig(sources=(src,)), pts)
    amp = r.rms * np.sqrt(2)
    assert amp[0] == pytest.approx(A, rel=0.01)  # Σ a_j cosθ_j ≈ A (근축)
    assert amp[1] / amp[0] == pytest.approx(np.exp(-1), rel=0.03)  # exp(−ρ²/w0²)
    assert amp[2] / amp[0] < 0.01
    assert r.L[0] == pytest.approx(1.0, abs=1e-6) and r.T[0] == pytest.approx(0.0, abs=1e-6)
    assert r.intensity[0, 2] > 0 and np.allclose(r.intensity[0, :2], 0, atol=1e-9 * r.i_ref)


def test_gaussian_beam_crossing_is_local():
    """두 빔이 교차하는 영역에서만 타원 궤적, 한 빔만 있는 곳은 직선 (=평면파 중첩과의 차이)."""
    w0 = 2 * LAM
    srcs = (Source(direction=(1, 0, 0), amplitude=A, kind="gaussian", waist=w0),
            Source(direction=(0, 1, 0), amplitude=A, kind="gaussian", waist=w0, phase=np.pi / 2))
    pts = np.array([[0, 0, 0], [0, 3 * w0, 0]], float)  # 교차점, 빔1 위 (빔2 축에서 멂)
    r = simulate_points(SimConfig(sources=srcs), pts)
    assert r.e[0] > 0.95  # 교차점: 원에 가까움
    assert r.L[1] > 0.99  # 빔 하나만: 직선


def test_gaussian_warnings():
    narrow = Source(direction=(1, 0, 0), kind="gaussian", waist=0.5 * LAM)
    r = simulate(SimConfig(sources=(narrow,), plane=ObservationPlane(nx=3, ny=3)))
    assert any("근축" in w for w in r.warnings)
    src = Source(direction=(1, 0, 0), kind="gaussian", waist=LAM, n_angular=7)
    period = gaussian_replica_period(src)
    big = ObservationPlane(nx=41, ny=41, spacing=period / 40, normal=(1, 0, 0))
    r = simulate(SimConfig(sources=(src,), plane=big))
    assert any("복제" in w for w in r.warnings)


# ---- 프리셋 예상 결과 -----------------------------------------------------------
def test_hex_preset_structure():
    cfg = PRESETS["hex"].config
    y0 = 2 * LAM / (3 * np.sqrt(3))
    pts = np.array([[0, 0, 0], [0, y0, 0], [0, -y0, 0], [2 * LAM / 3, 0, 0]], float)
    r = simulate_points(cfg, pts)
    # 원점과 (2λ/3, 0): 변위 마디
    assert np.isnan(r.L[0]) and np.isnan(r.L[3]) and r.rms[0] < 1e-12 * A
    # 소용돌이 중심: 정확한 원 궤적, |u| = 1.5 A, I = 0 → T NaN
    assert np.allclose(r.e[1:3], 1.0, atol=1e-9)
    assert np.allclose(r.rms[1:3], 1.5 * A, rtol=1e-9)
    assert np.all(np.isnan(r.T[1:3]))
    # 소용돌이 주변의 I 순환: 작은 원을 따라 접선 성분 부호가 일정하고, 두 중심의 회전 방향은 반대
    th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    circ = []
    for yc in (y0, -y0):
        ring = np.stack([0.03 * LAM * np.cos(th), yc + 0.03 * LAM * np.sin(th), 0 * th], -1)
        I = simulate_points(cfg, ring).intensity
        tang = np.stack([-np.sin(th), np.cos(th), 0 * th], -1)
        it = np.sum(I * tang, axis=1)
        assert np.all(np.sign(it) == np.sign(it[0]))
        circ.append(np.sign(it[0]))
    assert circ[0] == -circ[1]


def test_standing_preset_all_T_nan():
    r = simulate(PRESETS["standing"].config)
    assert np.all(np.isnan(r.T))
    assert np.allclose(r.L[~np.isnan(r.L)], 1.0, atol=1e-9)
