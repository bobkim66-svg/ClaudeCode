"""엔진 A 의 장 계산 검증: 무작정 공식과 비교, 해석해 케이스 (요구 테스트 1~4)."""

import numpy as np
import pytest

from conftest import A, C, F, LAM, one_period, random_unit
from polarsim.config import Medium, ObservationPlane, SimConfig, Source
from polarsim.metrics import simulate_points
from polarsim.physics import AnalyticField


def brute_force_u(sources, medium, pts, t):
    """u = Σ A k̂ cos(k·(r−r0) − ωt + φ) 를 루프로 직접 계산 (페이저 구현과 독립). (Nt,P,3)."""
    u = np.zeros((len(t), len(pts), 3))
    for s in sources:
        if not s.enabled:
            continue
        kh = np.asarray(s.direction, float) / np.linalg.norm(s.direction)
        k = 2 * np.pi * s.frequency / medium.c
        ph = k * (pts - np.asarray(s.ref_point)) @ kh
        arg = ph[None, :] - 2 * np.pi * s.frequency * t[:, None] + s.phase
        u += s.amplitude * np.cos(arg)[..., None] * kh
    return u


def test_phasor_matches_brute_force_multi_freq(rng, medium):
    sources = [
        Source(direction=tuple(rng.normal(size=3) * 3), amplitude=A * rng.uniform(0.2, 1),
               frequency=f, phase=rng.uniform(-np.pi, np.pi), ref_point=tuple(rng.normal(size=3) * 0.1))
        for f in (F, F, 1300.0, 777.0)
    ] + [Source(direction=(0, 0, 1), amplitude=A, frequency=F, enabled=False)]
    pts = rng.normal(size=(50, 3)) * LAM
    t = rng.uniform(0, 0.01, size=17)
    fld = AnalyticField(sources, medium)
    ev = fld.evaluate(pts, t, with_grad=True)
    ref = brute_force_u(sources, medium, pts, t)
    assert np.allclose(ev["u"], ref, rtol=0, atol=1e-12 * A)

    # v = ∂u/∂t, ∇u, p = −ρc²∇·u 를 유한차분과 비교 (해석식의 부호/인수 버그 탐지)
    dt = 1e-8
    v_fd = (brute_force_u(sources, medium, pts, t + dt) - brute_force_u(sources, medium, pts, t - dt)) / (2 * dt)
    vscale = 2 * np.pi * 1300 * A
    assert np.allclose(ev["v"], v_fd, atol=1e-5 * vscale)
    h = 1e-6 * LAM
    grad_fd = np.empty_like(ev["grad"])
    for j in range(3):
        d = np.zeros(3)
        d[j] = h
        grad_fd[..., j] = (brute_force_u(sources, medium, pts + d, t) - brute_force_u(sources, medium, pts - d, t)) / (2 * h)
    gscale = 2 * np.pi / LAM * 1.3 * A
    assert np.allclose(ev["grad"], grad_fd, atol=1e-5 * gscale)
    p_fd = -medium.rho * C**2 * np.trace(grad_fd, axis1=-2, axis2=-1)
    assert np.allclose(ev["p"], p_fd, atol=1e-5 * medium.rho * C**2 * gscale)


# ---- 요구 테스트 1: 단일 평면파 -------------------------------------------------
def test_single_plane_wave(rng, medium):
    khat = random_unit(rng)
    src = Source(direction=tuple(5 * khat), amplitude=A, frequency=F, phase=0.7)  # 단위화 안 된 입력
    pts = rng.normal(size=(30, 3)) * LAM
    cfg = SimConfig(sources=(src,))
    r = simulate_points(cfg, pts)

    u = AnalyticField([src], medium).displacement(pts, one_period())
    assert np.max(np.linalg.norm(np.cross(u, khat), axis=-1)) < 1e-12 * A  # u ∥ k̂ (종파)
    assert np.allclose(r.L, 1.0, atol=1e-12)
    assert np.allclose(r.T, 0.0, atol=1e-12)
    assert np.all(np.isnan(r.planarity))  # 직선 궤적 → 평면성 정의 불가
    # 세기 = ½ ρ c ω² A² k̂
    I_exact = 0.5 * medium.rho * C * (2 * np.pi * F) ** 2 * A**2
    assert np.allclose(r.intensity, I_exact * khat, rtol=1e-12, atol=1e-12 * I_exact)


def test_single_plane_wave_phase_velocity(rng, medium):
    """k̂ 방향으로 Δ 떨어진 두 점의 위상차에서 측정한 위상속도 = c."""
    khat = random_unit(rng)
    src = Source(direction=tuple(khat), amplitude=A, frequency=F)
    fld = AnalyticField([src], medium)
    t = one_period(n=256)
    r0 = rng.normal(size=3) * LAM
    delta = 0.137 * LAM
    s = fld.displacement(np.stack([r0, r0 + delta * khat]), t) @ khat  # (Nt,2)
    # ⟨cos(k·r − ωt) e^{iωt}⟩ = ½ e^{i k·r} → 추출 위상 = k·r (하류 점이 kΔ 만큼 큼)
    ph = np.angle(np.mean(s * np.exp(1j * 2 * np.pi * F * t)[:, None], axis=0))
    dphi = np.mod(ph[1] - ph[0], 2 * np.pi)
    c_measured = 2 * np.pi * F * delta / dphi
    assert c_measured == pytest.approx(C, rel=1e-10)


# ---- 요구 테스트 2: 대향 2파 → 정재파 -------------------------------------------
def test_standing_wave_nodes(medium):
    src = (Source(direction=(1, 0, 0), amplitude=A), Source(direction=(-1, 0, 0), amplitude=A))
    fld = AnalyticField(src, medium)
    dx = LAM / 400
    x = np.arange(0, 2 * LAM + dx / 2, dx)
    pts = np.stack([x, 0 * x, 0 * x], -1)
    amp = np.sqrt(2 * np.mean(np.sum(fld.displacement(pts, one_period()) ** 2, -1), axis=0))
    assert np.allclose(amp, 2 * A * np.abs(np.sin(2 * np.pi / LAM * x)), atol=1e-12 * A)
    # 국소 최솟값 = 마디. 간격 λ/2, 진폭 0
    inner = np.arange(1, len(x) - 1)
    mins = inner[(amp[inner] <= amp[inner - 1]) & (amp[inner] <= amp[inner + 1])]
    assert np.allclose(np.diff(x[mins]), LAM / 2, atol=dx)
    assert np.all(amp[mins] < 1e-10 * A)
    assert amp.max() == pytest.approx(2 * A, rel=1e-6)

    cfg = SimConfig(sources=src)
    r = simulate_points(cfg, pts)
    assert np.all(np.isnan(r.T))  # I = 0 → T 정의 불가
    assert np.all(np.isnan(r.L[mins]))  # 마디 → L 정의 불가
    ok = ~np.isnan(r.L)
    assert np.allclose(r.L[ok], 1.0, atol=1e-9)


# ---- 요구 테스트 3: 같은 방향 2파 ----------------------------------------------
@pytest.mark.parametrize("phi", [0.0, np.pi / 3, np.pi / 2, np.pi, 4.0])
def test_codirectional_superposition(rng, medium, phi):
    a1, a2 = A, 0.6 * A
    khat = random_unit(rng)
    src = (Source(direction=tuple(khat), amplitude=a1), Source(direction=tuple(khat), amplitude=a2, phase=phi))
    pts = rng.normal(size=(10, 3)) * LAM
    u = AnalyticField(src, medium).displacement(pts, one_period())
    amp = np.sqrt(2 * np.mean(np.sum(u**2, -1), axis=0))
    assert np.allclose(amp, abs(a1 + a2 * np.exp(1j * phi)), rtol=1e-10)
    if phi == 0.0:
        assert np.allclose(amp, a1 + a2)  # 보강
    if phi == np.pi:
        assert np.allclose(amp, a1 - a2)  # 상쇄


# ---- 요구 테스트 4: 직교 2파 + π/2 위상차 ----------------------------------------
def test_orthogonal_quadrature_circle():
    src = (Source(direction=(1, 0, 0), amplitude=A),
           Source(direction=(0, 1, 0), amplitude=A, phase=np.pi / 2))
    cfg = SimConfig(sources=src)
    # 원점, δ=0 직선(I 와 평행), δ=π 직선(I=0) 지점
    pts = np.array([[0, 0, 0], [LAM / 4, 0, 0], [-LAM / 4, 0, 0]], float)
    r = simulate_points(cfg, pts)
    assert r.e[0] == pytest.approx(1.0, abs=1e-9)
    assert r.L[0] == pytest.approx(0.0, abs=1e-9)
    assert r.T[0] == pytest.approx(0.5, abs=1e-9)
    assert r.planarity[0] == pytest.approx(1.0, abs=1e-9)  # xy 평면에 갇힘
    assert np.allclose(r.axis[1] * np.sign(r.axis[1][0]), np.array([1, 1, 0]) / np.sqrt(2), atol=1e-6)
    assert r.L[1] == pytest.approx(1.0, abs=1e-9) and r.T[1] == pytest.approx(0.0, abs=1e-9)
    assert r.L[2] == pytest.approx(1.0, abs=1e-9) and np.isnan(r.T[2])
