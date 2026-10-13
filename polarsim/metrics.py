"""입자별 궤적 편광 지표와 시뮬레이션 실행.

지표 (입자마다, 시간창 평균 ⟨·⟩):
    C = ⟨(u − ū)(u − ū)ᵀ⟩  (3×3, [m²]),  고유값 λ1 ≥ λ2 ≥ λ3
    직선성  L = 1 − λ2/λ1        (1: 직선 진동 = 종파적, 0: 원 궤적)
    타원도  e = √(λ2/λ1)         (1: 원 궤적)
    평면성  P = 1 − λ3/λ2        (1: 궤적이 한 평면에 갇힘)
    세기    I = ⟨p v⟩            ([W/m²])
    횡성분율 T = ⟨|u − (u·Î)Î|²⟩ / ⟨|u|²⟩,  Î = I/|I|

정의 불가 처리 (임의 값으로 채우지 않고 NaN):
    λ1 ≈ 0 (변위 마디)       → L, e, P, T 모두 NaN
    λ2 ≈ 0 (직선 궤적)        → P 만 NaN (0/0)
    |I| ≈ 0 (정재파, 소용돌이 중심) → T NaN
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from .config import ConfigError, MetricSettings, SimConfig
from .physics import (
    AnalyticField,
    gaussian_replica_period,
    numerical_curl_ratio,
    plane_points,
)

# 공통 주기 탐색: 주파수비를 분모 ≤ _MAX_DENOM 인 분수로 근사하고 상대오차 ≤ _RATIO_TOL 이면 채택.
_MAX_DENOM = 1000
_RATIO_TOL = 1e-9
# 격자 간격 권장 상한 = 최소 파장 / 8 (공간 Nyquist 여유).
SPACING_WARN_FRAC = 1.0 / 8.0
# 지표 계산 시 한 번에 처리하는 (시간 × 점) 원소 수 상한 (메모리 제한).
_SERIES_CHUNK = 1_500_000


# ---------------------------------------------------------------------------
# 시간창
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WindowInfo:
    duration: float  # [s]
    n_samples: int
    common_period: float | None  # [s], 없으면 None
    warnings: tuple[str, ...]


def common_period(freqs: list[float], max_periods: int) -> float | None:
    """모든 주파수의 공통 주기 [s]. 최저 주파수 주기의 max_periods 배를 넘거나 비가 무리수면 None.

    f_i = f_min · p_i/q_i (기약분수) 일 때 f_min·T 가 모든 q_i 의 배수여야 f_i·T 가 정수가 된다.
    → 최소 공통 주기 T = lcm(q_i) / f_min.
    """
    fmin = min(freqs)
    denoms = []
    for f in freqs:
        r = f / fmin
        frac = Fraction(r).limit_denominator(_MAX_DENOM)
        if abs(float(frac) - r) > _RATIO_TOL * r:
            return None
        denoms.append(frac.denominator)
    n_periods = math.lcm(*denoms)
    if n_periods > max_periods:
        return None
    return n_periods / fmin


def choose_window(config: SimConfig) -> tuple[np.ndarray, WindowInfo]:
    """시간 표본 (Nt,) [s] 과 창 정보. 끝점은 제외 (주기 신호의 이산 평균이 정확해지도록)."""
    win = config.window
    freqs = sorted({float(s.frequency) for s in config.active_sources()})
    fmin, fmax = freqs[0], freqs[-1]
    warns: list[str] = []
    period = common_period(freqs, win.max_auto_periods)

    if win.mode == "auto" and period is not None:
        duration = period
    else:
        duration = win.duration if (win.mode == "manual" and win.duration is not None) else win.manual_periods / fmin
        if win.mode == "auto":
            warns.append(
                "주파수들의 공통 주기가 없거나 너무 깁니다. 지표는 시간창 길이에 의존하므로 "
                f"창을 길게 잡으세요 (현재 {duration:.4g} s)."
            )
    if len(freqs) > 1:
        warns.append(
            "서로 다른 주파수가 섞여 맥놀이가 생깁니다. 지표는 맥놀이 주기 전체를 덮는 긴 시간창의 평균이며, "
            "순간적인 궤적 모양은 시간에 따라 변합니다."
        )

    n = int(math.ceil(duration * fmax * win.samples_per_period))
    if n > win.max_samples:
        n = win.max_samples
        per_period = n / (duration * fmax)
        warns.append(f"표본 수를 {n} 개로 제한했습니다 (최고 주파수 주기당 {per_period:.1f} 개).")
        if per_period < 8:
            raise ConfigError(
                "시간창이 너무 길어 최고 주파수 주기당 표본이 8 개 미만입니다 (앨리어싱). 창을 줄이세요."
            )
    times = np.arange(n) * (duration / n)
    return times, WindowInfo(duration, n, period, tuple(warns))


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------
def trajectory_metrics(u: np.ndarray, a_ref: float, settings: MetricSettings) -> dict:
    """u (Nt,P,3) [m] → 공분산 기반 지표.

    반환: eigvals (P,3) [m²] 내림차순, axis (P,3) 주축 단위벡터, L, e, planarity (P,), node (P,) bool.
    """
    du = u - u.mean(axis=0, keepdims=True)
    cov = np.einsum("tpi,tpj->pij", du, du) / u.shape[0]
    w, vec = np.linalg.eigh(cov)  # 오름차순
    w = np.clip(w[:, ::-1], 0.0, None)  # 반올림 오차로 생긴 −0 근처 값 제거
    vec = vec[:, :, ::-1]
    l1, l2, l3 = w[:, 0], w[:, 1], w[:, 2]

    node = l1 < settings.node_tol * a_ref**2
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio21 = np.where(node, np.nan, l2 / l1)
        lin = l2 < settings.linear_tol * l1
        planarity = np.where(node | lin, np.nan, 1.0 - l3 / l2)
    return {
        "eigvals": w,
        "axis": vec[:, :, 0],
        "L": 1.0 - ratio21,
        "e": np.sqrt(ratio21),
        "planarity": planarity,
        "node": node,
    }


def intensity(p: np.ndarray, v: np.ndarray) -> np.ndarray:
    """시간평균 음향 세기 I = ⟨p v⟩. p (Nt,P) [Pa], v (Nt,P,3) [m/s] → (P,3) [W/m²]."""
    return np.mean(p[..., None] * v, axis=0)


def transverse_fraction(u: np.ndarray, I: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """T = ⟨|u − (u·Î)Î|²⟩ / ⟨|u|²⟩. u (Nt,P,3), I (P,3), valid (P,) bool → (P,), 무효 점은 NaN."""
    norm = np.linalg.norm(I, axis=1)
    ihat = np.divide(I, norm[:, None], out=np.zeros_like(I), where=norm[:, None] > 0)
    u2 = np.sum(u**2, axis=-1).mean(axis=0)
    upar2 = (np.einsum("tpi,pi->tp", u, ihat) ** 2).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        T = (u2 - upar2) / u2
    return np.where(valid, np.clip(T, 0.0, 1.0), np.nan)


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
@dataclass
class SimulationResult:
    """shape 는 입력 점 배열의 앞쪽 shape (평면이면 (nx, ny)) 를 따른다."""

    points: np.ndarray  # (...,3) [m]
    times: np.ndarray  # (Nt,) [s]
    L: np.ndarray
    e: np.ndarray
    planarity: np.ndarray
    T: np.ndarray
    intensity: np.ndarray  # (...,3) [W/m²]
    eigvals: np.ndarray  # (...,3) [m²]
    axis: np.ndarray  # (...,3)
    rms: np.ndarray  # (...) [m], √⟨|u|²⟩
    curl_ratio: float
    a_ref: float  # [m]
    i_ref: float  # [W/m²]
    window: WindowInfo
    spacing: float | None = None
    warnings: list[str] = field(default_factory=list)


def reference_scales(config: SimConfig) -> tuple[float, float]:
    """A_ref = 활성 진폭 합 [m], I_ref = ½ ρ c ω_max² A_ref² [W/m²]."""
    act = config.active_sources()
    a_ref = sum(float(s.amplitude) for s in act)
    w_max = 2.0 * np.pi * max(float(s.frequency) for s in act)
    return a_ref, 0.5 * config.medium.rho * config.medium.c * w_max**2 * a_ref**2


def config_warnings(config: SimConfig, points: np.ndarray, spacing: float | None) -> list[str]:
    """해석을 왜곡할 수 있는 설정에 대한 경고 (에러는 아님)."""
    warns = []
    lam = config.min_wavelength()
    if spacing is not None and spacing > SPACING_WARN_FRAC * lam:
        warns.append(
            f"격자 간격 {spacing:.4g} m 가 최소 파장/8 ({lam / 8:.4g} m) 보다 큽니다. "
            "공간 표본이 거칠어 가짜 무늬(앨리어싱)가 보일 수 있습니다."
        )
    pts = points.reshape(-1, 3)
    for i, s in enumerate(config.sources):
        if not s.enabled or s.kind != "gaussian":
            continue
        lam_s = config.medium.c / s.frequency
        if s.waist < lam_s:
            warns.append(
                f"소스 {i + 1}: 허리 반경이 파장보다 작아 근축(paraxial) 근사가 깨지고 "
                "각스펙트럼이 전파 가능 범위에서 잘립니다. 빔 모양이 가우시안과 다를 수 있습니다."
            )
        khat = np.asarray(s.direction, float) / np.linalg.norm(s.direction)
        rel = pts - np.asarray(s.ref_point, float)
        dist = np.linalg.norm(rel - (rel @ khat)[:, None] * khat, axis=1).max()
        half = 0.5 * gaussian_replica_period(s)
        if dist > 0.8 * half:
            warns.append(
                f"소스 {i + 1}: 관측점이 빔 축에서 최대 {dist:.4g} m 떨어져 있어, 이산 각스펙트럼의 "
                f"가짜 빔 복제(주기 {2 * half:.4g} m)에 닿을 수 있습니다. 각스펙트럼 표본 수를 늘리세요."
            )
    return warns


def simulate_points(config: SimConfig, points: np.ndarray, spacing: float | None = None) -> SimulationResult:
    """임의 점 배열 (...,3) [m] 의 입자들에 대해 지표 계산."""
    config.validate()
    pts = np.asarray(points, dtype=float)
    lead = pts.shape[:-1]
    flat = pts.reshape(-1, 3)
    P = flat.shape[0]

    times, winfo = choose_window(config)
    fld = AnalyticField(config.sources, config.medium)
    a_ref, i_ref = reference_scales(config)
    st = config.metrics

    out = {k: np.empty(P) for k in ("L", "e", "planarity", "T", "rms")}
    out["intensity"] = np.empty((P, 3))
    out["eigvals"] = np.empty((P, 3))
    out["axis"] = np.empty((P, 3))
    step = max(1, _SERIES_CHUNK // len(times))
    for s in range(0, P, step):
        sl = slice(s, s + step)
        ev = fld.evaluate(flat[sl], times)
        tm = trajectory_metrics(ev["u"], a_ref, st)
        I = intensity(ev["p"], ev["v"])
        valid = (~tm["node"]) & (np.linalg.norm(I, axis=1) >= st.intensity_tol * i_ref)
        out["T"][sl] = transverse_fraction(ev["u"], I, valid)
        out["intensity"][sl] = I
        out["rms"][sl] = np.sqrt(np.sum(ev["u"] ** 2, axis=-1).mean(axis=0))
        for k in ("L", "e", "planarity", "eigvals", "axis"):
            out[k][sl] = tm[k]

    t_check = times[np.linspace(0, len(times) - 1, min(st.curl_n_times, len(times))).astype(int)]
    curl = numerical_curl_ratio(fld, flat, t_check, st.curl_h_frac * config.min_wavelength())

    def shape(a):
        return a.reshape(lead + a.shape[1:])

    return SimulationResult(
        points=pts,
        times=times,
        curl_ratio=curl,
        a_ref=a_ref,
        i_ref=i_ref,
        window=winfo,
        spacing=spacing,
        warnings=list(winfo.warnings) + config_warnings(config, pts, spacing),
        **{k: shape(v) for k, v in out.items()},
    )


def simulate(config: SimConfig) -> SimulationResult:
    """관측 평면 격자 전체에 대해 지표 계산. 결과 배열 shape (nx, ny[,3])."""
    config.validate()
    spacing = config.plane.spacing or config.min_wavelength() / 10.0
    pts = plane_points(config.plane, spacing)
    return simulate_points(config, pts, spacing)
