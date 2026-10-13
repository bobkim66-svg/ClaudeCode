"""회전 검증, 좌표 변환 불변성, 시간 반전/위상 이동 대칭, 재현성 (요구 테스트 5~7)."""

import dataclasses

import numpy as np
import pytest

from conftest import A, LAM, one_period, random_rotation
from polarsim.config import ObservationPlane, SimConfig, Source
from polarsim.metrics import simulate, simulate_points
from polarsim.physics import AnalyticField, numerical_curl_ratio
from polarsim.presets import PRESETS

CURL_TOL = 1e-4  # 차분 간격 h = λ/1000 의 절단오차 (kh)²/6 ≈ 7e-6 에 여유


def random_config(rng, n=5, freqs=(1000.0,)):
    srcs = tuple(
        Source(direction=tuple(rng.normal(size=3)), amplitude=A * rng.uniform(0.3, 1.0),
               frequency=float(rng.choice(freqs)), phase=float(rng.uniform(-np.pi, np.pi)),
               ref_point=tuple(rng.normal(size=3) * 0.05))
        for _ in range(n)
    )
    plane = ObservationPlane(origin=tuple(rng.normal(size=3) * 0.1), normal=tuple(rng.normal(size=3)), nx=9, ny=7)
    return SimConfig(sources=srcs, plane=plane)


def beam_config():
    return SimConfig(sources=(
        Source(direction=(1, 0, 0), amplitude=A, kind="gaussian", waist=2 * LAM),
        Source(direction=(0, 1, 0.2), amplitude=A, kind="gaussian", waist=1.5 * LAM, phase=1.0),
    ), plane=ObservationPlane(nx=12, ny=12))


# ---- 요구 테스트 5: 모든 케이스에서 ∇×u ≈ 0 ---------------------------------------
@pytest.mark.parametrize("name", ["standing", "circular", "hex", "random", "random_multifreq", "beams"])
def test_curl_free(name, rng):
    cfg = {
        "random": lambda: random_config(rng),
        "random_multifreq": lambda: random_config(rng, n=6, freqs=(1000.0, 1250.0, 800.0)),
        "beams": beam_config,
    }.get(name, lambda: PRESETS[name].config)()
    r = simulate(cfg)
    assert r.curl_ratio < CURL_TOL


def test_curl_check_detects_transverse_bug():
    """음성 대조: 변위를 k̂ 에 수직으로 잘못 설정하면 (=횡파 버그) 검사가 O(1) 값을 내야 한다."""
    fld = AnalyticField([Source(direction=(1, 0, 0), amplitude=A)], PRESETS["standing"].config.medium)
    fld.waves = dataclasses.replace(fld.waves, khat=np.array([[0.0, 1.0, 0.0]]))  # kvec 은 x 그대로
    pts = np.random.default_rng(0).normal(size=(20, 3)) * LAM
    assert numerical_curl_ratio(fld, pts, one_period(n=8), LAM * 1e-3) > 0.5


# ---- 요구 테스트 6: 좌표 변환 불변성 --------------------------------------------
def _rotate_config(cfg, R, shift=np.zeros(3)):
    srcs = tuple(dataclasses.replace(s, direction=tuple(R @ np.asarray(s.direction)),
                                     ref_point=tuple(R @ np.asarray(s.ref_point) + shift)) for s in cfg.sources)
    return srcs


METRIC_KEYS = ("L", "e", "planarity", "T")


def _assert_same_metrics(r1, r2):
    for k in METRIC_KEYS:
        a, b = getattr(r1, k), getattr(r2, k)
        assert np.array_equal(np.isnan(a), np.isnan(b)), k
        assert np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], atol=1e-8), k


def test_rotation_and_translation_invariance(rng):
    cfg = random_config(rng)
    pts = rng.normal(size=(40, 3)) * LAM
    r0 = simulate_points(cfg, pts)
    R = random_rotation(rng)
    shift = rng.normal(size=3) * 0.3
    cfg_rt = dataclasses.replace(cfg, sources=_rotate_config(cfg, R, shift))
    r1 = simulate_points(cfg_rt, pts @ R.T + shift)
    _assert_same_metrics(r0, r1)
    assert np.allclose(r1.intensity, r0.intensity @ R.T, atol=1e-10 * r0.i_ref)  # 세기는 벡터로 회전
    assert r1.curl_ratio < CURL_TOL


def test_plane_reparameterization(rng):
    """같은 물리 평면을 다른 평면 내 축으로 기술해도 같은 입자에서 같은 지표 (격자는 90° 회전)."""
    cfg = random_config(rng)
    n = 11
    pa = ObservationPlane(origin=(0.1, 0, 0), normal=(0, 0, 1), nx=n, ny=n, spacing=LAM / 10, u_axis=(1, 0, 0))
    pb = dataclasses.replace(pa, u_axis=(0, 1, 0))
    ra = simulate(dataclasses.replace(cfg, plane=pa))
    rb = simulate(dataclasses.replace(cfg, plane=pb))
    # B[i,j] 의 물리 위치 = A[n−1−j, i]  →  A 를 시계 방향 대응으로 재배열
    remap = lambda a: np.transpose(a[::-1], (1, 0) + tuple(range(2, a.ndim)))
    assert np.allclose(rb.points, remap(ra.points), atol=1e-15)
    for k in METRIC_KEYS:
        assert np.allclose(getattr(rb, k), remap(getattr(ra, k)), atol=1e-8, equal_nan=True)


def test_plane_basis_is_right_handed():
    from polarsim.physics import plane_basis

    for normal in [(0, 0, 1), (1, 2, 3), (0, -1, 0), (1e-9, 0, -1)]:
        e_u, e_v, n = plane_basis(ObservationPlane(normal=normal))
        assert np.allclose([e_u @ e_u, e_v @ e_v, e_u @ e_v, e_u @ n], [1, 1, 0, 0], atol=1e-12)
        assert np.allclose(np.cross(e_u, e_v), n, atol=1e-12)


# ---- 요구 테스트 7: 시간 반전, 위상 이동, 재현성 ------------------------------------
def test_time_reversal_symmetry(rng):
    """u(r, −t; d, φ) = −u(r, t; −d, −φ)  (cos 이 짝함수이고 변위 방향이 k̂ 이므로)."""
    cfg = random_config(rng)
    rev = tuple(dataclasses.replace(s, direction=tuple(-np.asarray(s.direction)), phase=-s.phase) for s in cfg.sources)
    pts = rng.normal(size=(20, 3)) * LAM
    t = rng.uniform(0, 0.005, size=11)
    u_fwd = AnalyticField(cfg.sources, cfg.medium).displacement(pts, -t)
    u_rev = AnalyticField(rev, cfg.medium).displacement(pts, t)
    assert np.allclose(u_fwd, -u_rev, atol=1e-13 * A)
    r0 = simulate_points(cfg, pts)
    r1 = simulate_points(dataclasses.replace(cfg, sources=rev), pts)
    _assert_same_metrics(r0, r1)  # 궤적 모양은 그대로
    assert np.allclose(r1.intensity, -r0.intensity, atol=1e-10 * r0.i_ref)  # 에너지 흐름만 반대


def test_global_phase_shift_is_time_shift(rng):
    """단일 주파수에서 모든 φ += ωτ 는 시간 이동 t → t − τ 와 같고, 한 주기 평균 지표는 불변."""
    cfg = random_config(rng)
    w = 2 * np.pi * 1000.0
    tau = 0.37e-3
    shifted = tuple(dataclasses.replace(s, phase=s.phase + w * tau) for s in cfg.sources)
    pts = rng.normal(size=(20, 3)) * LAM
    t = rng.uniform(0, 0.005, size=11)
    u0 = AnalyticField(cfg.sources, cfg.medium).displacement(pts, t - tau)
    u1 = AnalyticField(shifted, cfg.medium).displacement(pts, t)
    assert np.allclose(u0, u1, atol=1e-13 * A)
    _assert_same_metrics(simulate_points(cfg, pts), simulate_points(dataclasses.replace(cfg, sources=shifted), pts))


def test_reproducible(rng):
    cfg = random_config(rng, freqs=(1000.0, 1500.0))
    r1, r2 = simulate(cfg), simulate(cfg)
    for k in METRIC_KEYS + ("intensity", "eigvals", "rms"):
        assert np.array_equal(getattr(r1, k), getattr(r2, k), equal_nan=True)
    assert r1.curl_ratio == r2.curl_ratio
