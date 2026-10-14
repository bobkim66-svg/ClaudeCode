"""엔진 B (2D 주파수 영역 / FDTD) 검증: 해석해, 상호 교차검증, 에너지 보존, 대칭, 경계 입력."""

import dataclasses
import math

import numpy as np
import pytest

from polarsim.config import ConfigError, MetricSettings, SimConfig, Source
from polarsim.fdtd import (Emitter, PocketSpec, Rect, WaveConfig, emitted_power, fdtd_time_step, pocket_config,
                           radial_ellipticity, run_fdtd, solve_harmonic, wave_metrics)
from polarsim.metrics import phasor_metrics, simulate_points, trajectory_metrics
from polarsim.physics import AnalyticField

LAM = 0.5  # 물, 3 kHz
XI = 1e-5


def channel(phi=0.4, **kw):
    """폭 전체를 덮는 피스톤 + 강체 옆벽 = 정확한 평면파 (오른쪽 끝만 PML)."""
    return WaveConfig(domain=(-0.2, 4.0, -0.25, 0.25), pml_sides=(False, True, False, False),
                      solids=(Rect(-1, 0, -1, 1),), emitters=(Emitter("x", 0.0, (-1, 1), +1, XI, phi, True, "p"),),
                      **kw)


def _plane_wave_checks(r, phi=0.4):
    k = 2 * np.pi / LAM
    u = r.U[:, r.U.shape[1] // 2, 0]
    m = (r.xc > 0.2) & (r.xc < 3.5)
    x, u = r.xc[m], u[m]
    amp = np.abs(u)
    assert amp.mean() / XI == pytest.approx(math.cos(math.pi / 20), rel=1e-3)  # 면→셀 평균 오차만 남음
    assert np.polyfit(x, np.unwrap(np.angle(u)), 1)[0] / k == pytest.approx(1.0, abs=0.006)  # 수치 분산 < 0.6%
    assert (amp.max() - amp.min()) / (amp.max() + amp.min()) < 1e-4  # PML 반사율
    assert abs(np.angle(u[0] * np.exp(-1j * (k * x[0] + phi)))) < 0.02  # 위상 규약 = 엔진 A
    assert np.abs(r.U[..., 1]).max() < 1e-12 * XI  # 횡 성분 없음
    assert r.curl_ratio < 1e-12


def test_harmonic_plane_wave_channel():
    _plane_wave_checks(solve_harmonic(channel()))


def test_fdtd_plane_wave_channel():
    r = run_fdtd(channel())
    assert r.convergence < 0.01
    _plane_wave_checks(r)


def test_fdtd_equals_harmonic_with_leapfrog_frequency():
    """같은 이산 연산자: FDTD 정상상태 = 주파수 영역 해 (ω̃ = 2 sin(ωdt/2)/dt). 서로 독립된 풀이의 교차검증."""
    cfg = WaveConfig(domain=(-0.5, 2.5, -1.5, 1.5), solids=(Rect(-0.2, 0, -0.5, 0.5),),
                     emitters=(Emitter("x", 0.0, (-0.5, 0.5), +1, XI, 0.3, True, "plate"),))
    dt, _ = fdtd_time_step(cfg)
    w = 2 * math.pi * cfg.frequency
    f = run_fdtd(cfg)
    h = solve_harmonic(cfg, omega_eff=2 * math.sin(w * dt / 2) / dt)
    m = ~f.solid
    assert np.abs(f.P - h.P)[m].max() / np.abs(h.P[m]).max() < 1e-3
    assert np.abs(f.V - h.V)[m].max() / np.abs(h.V[m]).max() < 1e-3


@pytest.fixture(scope="module")
def pocket():
    cfg = pocket_config(PocketSpec(), frequency=2700.0, domain=(-1, 3, -2, 2))
    return cfg, solve_harmonic(cfg)


def test_energy_conservation(pocket):
    """방사면이 넣은 일률 = 품을 둘러싼 닫힌 경로 밖으로 나가는 세기 플럭스 (손실 없는 매질)."""
    cfg, r = pocket
    I = wave_metrics(r, cfg.medium)["I"]
    ix = lambda x: int(np.argmin(abs(r.xc - x)))  # noqa: E731
    iy = lambda y: int(np.argmin(abs(r.yc - y)))  # noqa: E731
    i0, i1, j0, j1 = ix(-0.9), ix(2.8), iy(-1.5), iy(1.5)
    d = r.dx
    flux = (I[i1, j0:j1 + 1, 0].sum() - I[i0, j0:j1 + 1, 0].sum()
            + I[i0:i1 + 1, j1, 1].sum() - I[i0:i1 + 1, j0, 1].sum()) * d
    assert flux / emitted_power(r, cfg) == pytest.approx(1.0, abs=0.02)


def test_pocket_symmetry_and_curl(pocket):
    """윗판·아랫판 같은 위상 + 뒷벽: 형상과 구동이 y 대칭 → p(x,y) = p(x,−y), v_y 반대칭."""
    _, r = pocket
    assert np.allclose(r.P, r.P[:, ::-1], atol=1e-9 * np.abs(r.P).max())
    assert np.allclose(r.V[..., 1], -r.V[:, ::-1, 1], atol=1e-9 * np.abs(r.V).max())
    assert r.curl_ratio < 1e-12
    n = round(1.0 / r.dx)  # L = D = 1 m 를 덮는 면 수 (2700 Hz: 36)
    assert r.driven_faces == {"뒷벽 (+x 방사)": n, "윗판 (−y 방사)": n, "아랫판 (+y 방사)": n}
    assert any("격자" in w for w in r.warnings)  # 판 두께 0.1 m = 3.6 셀 → 스냅 경고


def test_no_snap_warning_when_aligned():
    r = solve_harmonic(pocket_config(PocketSpec(), frequency=3000.0, domain=(-1, 2, -1.5, 1.5)))
    assert not any("격자" in w for w in r.warnings)  # 3 kHz: 셀 25 mm, 모든 치수가 배수


def test_superposition():
    """선형성: 각 면만 켠 해의 합 = 모두 켠 해."""
    base = PocketSpec()
    off = (False, XI, 0.0)
    kw = dict(frequency=2700.0, domain=(-1, 2.5, -1.5, 1.5))
    parts = [dataclasses.replace(base, top=off, bottom=off), dataclasses.replace(base, back=off, bottom=off),
             dataclasses.replace(base, back=off, top=off)]
    total = solve_harmonic(pocket_config(base, **kw)).P
    summed = sum(solve_harmonic(pocket_config(s, **kw)).P for s in parts)
    assert np.allclose(total, summed, atol=1e-9 * np.abs(total).max())


def test_far_field_ellipticity_decays(pocket):
    """품 밖으로 멀어질수록 에너지 가중 타원도가 줄어든다 (원거리 이론: ~1/(kr))."""
    cfg, r = pocket
    met = wave_metrics(r, cfg.medium)
    rows = radial_ellipticity(r, met, (1.0, 0.0), cfg.wavelength, (1, 2, 3.5), half_plane_x=1.0)
    assert rows[1]["e_energy"] < rows[0]["e_energy"]
    inside = (r.xc[:, None] > 0) & (r.xc[:, None] < 1) & (np.abs(r.yc[None, :]) < 0.5)
    assert np.nanmedian(met["e"][inside]) > 2 * rows[1]["e_energy"]  # 품 안이 훨씬 타원적


def test_radial_coverage_flags_clipped_bins(pocket):
    """영역에 잘린 거리 구간은 coverage < 1 로 표시된다 (모서리 방향만 남는 표본 편향 방지)."""
    cfg, r = pocket  # 영역 x ≤ 3, |y| ≤ 2, 입구 (1, 0) → 앞쪽 반고리는 2 m (≈3.6λ) 까지만 온전
    met = wave_metrics(r, cfg.medium)
    rows = radial_ellipticity(r, met, (1.0, 0.0), cfg.wavelength, (1, 2, 3, 5), half_plane_x=1.0)
    assert rows[0]["coverage"] > 0.95 and rows[1]["coverage"] > 0.95
    assert rows[2]["coverage"] < 0.8


def test_phasor_metrics_match_time_series(rng):
    """단일 주파수: 페이저 공식 (½Re(UU^H), ½Re(PV*)) = 시간 표본 평균 (엔진 A 로 비교)."""
    srcs = tuple(Source(direction=tuple(rng.normal(size=3)), amplitude=1e-6 * rng.uniform(0.3, 1),
                        phase=float(rng.uniform(-3, 3))) for _ in range(4))
    cfg = SimConfig(sources=srcs)
    pts = rng.normal(size=(25, 3)) * 0.3
    ts = simulate_points(cfg, pts)
    fld = AnalyticField(srcs, cfg.medium)
    ph = fld.phasors(pts)[0]
    P = -cfg.medium.rho * cfg.medium.c**2 * ph.D
    m = phasor_metrics(ph.U, P, -1j * ph.omega * ph.U, ts.a_ref, ts.i_ref, MetricSettings())
    for k in ("L", "e", "T"):
        assert np.allclose(m[k], getattr(ts, k), atol=1e-9, equal_nan=True), k
    assert np.allclose(m["intensity"], ts.intensity, rtol=1e-9, atol=1e-12 * ts.i_ref)


@pytest.mark.parametrize("kw, msg", [
    (dict(cfl=0.8), "CFL"),
    (dict(cells_per_wavelength=6), "파장당 셀"),
    (dict(frequency=0.0), "주파수"),
    (dict(emitters=()), "방사면이 없습니다"),
    (dict(emitters=(Emitter("x", 2.0, (-1, 1), +1),)), "강체 표면과 맞지 않습니다"),
    (dict(domain=(1, 0, 0, 1)), "계산 영역"),
])
def test_bad_wave_inputs(kw, msg):
    with pytest.raises(ConfigError, match=msg):
        solve_harmonic(dataclasses.replace(channel(), **kw))


def test_cutoff_frequencies():
    assert PocketSpec(gap=1.0).cutoff_frequencies(1500.0, 3200.0) == [750.0, 1500.0, 2250.0, 3000.0]


# ---- 막대 프레임 (임의 방향) ------------------------------------------------------
from polarsim.fdtd import Bar, frame_config, frame_presets, parallel_gaps  # noqa: E402


def test_bars_reproduce_pocket_exactly():
    """같은 형상을 막대 3개로 표현하면 기존 품 모델과 비트 단위로 같은 해."""
    spec = PocketSpec()
    for f in (2700.0, 3000.0):
        a = solve_harmonic(pocket_config(spec, frequency=f, domain=(-1, 2.5, -1.5, 1.5)))
        b = solve_harmonic(frame_config(spec.bars(), frequency=f, domain=(-1, 2.5, -1.5, 1.5)))
        assert np.array_equal(a.P, b.P) and np.array_equal(a.solid, b.solid)


def test_bar_polar_and_endpoint_forms_agree():
    b = Bar.from_polar(0.2, -0.1, 30.0, 0.8, 0.05)
    assert (b.x2, b.y2) == pytest.approx((0.2 + 0.8 * math.cos(math.pi / 6), -0.1 + 0.4))
    assert b.length == pytest.approx(0.8) and b.angle_deg == pytest.approx(30.0)
    assert np.allclose(b.normal, [-0.5, math.sqrt(3) / 2])  # 왼쪽 법선


@pytest.mark.parametrize("angle", np.arange(0.0, 91.0, 7.0))
def test_thin_bar_has_no_staircase_gap(angle):
    """최소 두께 규칙(1.05×) 을 넘는 막대는 계단 근사에서 양쪽 유체가 막대 구간 안에서 연결되지 않는다."""
    from scipy.ndimage import label

    dx = 0.025
    b = Bar.from_polar(-0.6, -0.4, angle, 1.0, 1.06 * Bar.from_polar(0, 0, angle, 1, 1).min_thickness(dx))
    xc = (np.arange(-60, 60) + 0.5) * dx
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    s_, d = b.local(X, Y)
    strip = (s_ >= 0) & (s_ <= b.length) & (np.abs(d) < b.thickness / 2 + 3 * dx)
    lab, _ = label(strip & ~b.mask(X, Y))  # 4-연결
    left = set(np.unique(lab[strip & (d > 0)])) - {0}
    right = set(np.unique(lab[strip & (d < 0) & ~b.mask(X, Y)])) - {0}
    assert not (left & right)


def test_tilted_piston_matches_plane_wave():
    """30° 기울인 긴 피스톤: 중심선 위에서 진폭 ≈ ξ0, 에너지 흐름 방향 ≈ 법선, 가짜 타원도 작음 (계단 오차)."""
    a = math.radians(30.0)
    t = np.array([math.cos(a), math.sin(a)])
    L = 8 * LAM
    p1 = -t * L / 2
    bar = Bar(p1[0], p1[1], p1[0] + t[0] * L, p1[1] + t[1] * L, 0.1, "left", XI, 0.3, "piston")
    r = solve_harmonic(frame_config((bar,), frequency=3000.0, domain=(-2.5, 2.5, -2.5, 2.5)))
    m = wave_metrics(r, WaveConfig().medium)
    X, Y = np.meshgrid(r.xc, r.yc, indexing="ij")
    s_, d = bar.local(X, Y)
    sel = (np.abs(s_ - L / 2) < 0.6 * r.dx) & (d > 0.5 * LAM) & (d < 2.0 * LAM)
    amp = np.linalg.norm(np.abs(r.U[sel]), axis=1) / XI
    assert amp.mean() == pytest.approx(1.0, abs=0.03)
    I = m["I"][sel]
    err = np.degrees(np.arccos(np.clip(I @ bar.normal / np.linalg.norm(I, axis=1), -1, 1)))
    assert err.max() < 4.0  # λ/20 에서 ≤ 3° (λ/40 에서 ≤ 1.1°, 셀 크기에 비례)
    assert np.nanmax(m["e"][sel]) < 0.07
    assert r.curl_ratio < 1e-12


def test_tilted_frame_energy_conservation():
    pr = frame_presets()["v"]
    cfg = frame_config(pr.bars, frequency=2700.0, domain=(-1, 2.5, -1.8, 1.8))
    r = solve_harmonic(cfg)
    I = wave_metrics(r, cfg.medium)["I"]
    m = 3
    i0, i1, j0, j1 = m, r.P.shape[0] - 1 - m, m, r.P.shape[1] - 1 - m
    flux = (I[i1, j0:j1 + 1, 0].sum() - I[i0, j0:j1 + 1, 0].sum()
            + I[i0:i1 + 1, j1, 1].sum() - I[i0:i1 + 1, j0, 1].sum()) * r.dx
    assert flux / emitted_power(r, cfg) == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("bar, msg", [
    (Bar(0, 0, 0, 0, 0.1, "left"), "길이가 0"),
    (Bar(0, 0, 1, 0, 0.0, "left"), "두께는 양수"),
    (Bar(0, 0, 1, 0, 0.1, "up"), "방사면은"),
    (Bar(0, 0, 9, 0, 0.1, "left"), "계산 영역 밖"),
    (Bar.from_polar(0, 0, 30, 1, 0.01, emit="left"), "너무 얇아"),
    (Bar(0, 0, 1, 0, 0.1, "none"), "켜진 방사면이 없습니다"),
])
def test_bad_bar_inputs(bar, msg):
    with pytest.raises(ConfigError, match=msg):
        solve_harmonic(frame_config((bar,), domain=(-1, 2, -1, 1)))


def test_parallel_gaps_and_forward_half_plane(pocket):
    gaps = parallel_gaps(PocketSpec().bars())
    assert gaps == [("윗판", "아랫판", pytest.approx(1.0))]
    assert parallel_gaps(frame_presets()["v"].bars) == []  # ±20° 판은 평행 아님
    cfg, r = pocket
    met = wave_metrics(r, cfg.medium)
    a = radial_ellipticity(r, met, (1.0, 0.0), cfg.wavelength, (1, 2, 3), half_plane_x=1.0)
    b = radial_ellipticity(r, met, (1.0, 0.0), cfg.wavelength, (1, 2, 3), forward_deg=0.0)
    assert [x["e_energy"] for x in a] == pytest.approx([x["e_energy"] for x in b])


def test_presets_solve():
    for key, pr in frame_presets().items():
        r = solve_harmonic(frame_config(pr.bars, frequency=2700.0, domain=pr.domain, cells_per_wavelength=12))
        assert r.curl_ratio < 1e-12, key
