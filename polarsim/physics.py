"""엔진 A: 평면파 해석적 중첩 (선형 음향, 소진폭, 이상 유체).

변위장 (평면파 m 에 대해 합):
    u(r,t) = Σ_m a_m k̂_m cos(k_m·(r − r_m) − ω_m t + φ_m),   k_m = (ω_m / c) k̂_m

변위가 k̂_m 와 평행(종파)이므로 각 항은 스칼라 퍼텐셜의 기울기이고 ∇×u = 0 이 해석적으로 성립한다.
따라서 엔진 A 의 수치 회전 검증은 '물리 증거'가 아니라 '구현 버그 검출기' 다
(방향 단위화 누락, 변위 방향 ≠ k̂ 같은 실수를 잡는다).

계산 방식 — 주파수별 복소 페이저:
    u(r,t) = Σ_ω Re[ U_ω(r) e^{−iωt} ],   U_ω(r) = Σ_{m: ω_m=ω} a_m k̂_m e^{i(k_m·(r−r_m)+φ_m)}
    공간 의존 부분 U_ω 를 한 번만 계산하면 시간 표본은 곱셈 한 번으로 얻는다 (애니메이션 성능).
    ∂_j u_i  → Re[ G_ω,ij e^{−iωt} ],  G_ω,ij = Σ i k_m,j a_m k̂_m,i e^{i(...)}
    ∇·u      → Re[ D_ω e^{−iωt} ],     D_ω    = Σ i |k_m| a_m e^{i(...)}   (k̂·k = |k|)
    v = ∂u/∂t → Re[ −iω U_ω e^{−iωt} ]
    p = −ρ c² ∇·u

가우시안 빔: 허리 반경 w0 의 스칼라 프로파일 exp(−ρ²/w0²) 의 각스펙트럼 ∝ exp(−k_t² w0²/4) 를
    정사각 격자로 표본화한 평면파 묶음으로 만든다. 각 성분이 정확한 평면파이므로 파동방정식과
    ∇×u = 0 을 정확히 만족한다. 이산 표본화 때문에 빔 단면이 주기 2π/Δk_t 로 반복(가짜 복제)되므로
    관측 영역은 이 주기의 절반 안에 두어야 한다 (gaussian_replica_period 참고).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ConfigError, Medium, ObservationPlane, Source, unit_vector

# 가우시안 각스펙트럼을 자르는 범위 (표준편차의 배수). exp(−9) ≈ 1.2e-4 에서 자른다.
GAUSS_SPECTRUM_SIGMAS = 3.0
# 전파 가능한 성분만 사용: k_t < PROPAGATING_FRAC · k (스치는 각도/소멸파 제외).
PROPAGATING_FRAC = 0.99
# 페이저 행렬 (점 × 평면파) 한 덩어리의 최대 원소 수 (메모리 상한 ~ 32 MB complex128).
_CHUNK_ELEMS = 2_000_000


# ---------------------------------------------------------------------------
# 기하
# ---------------------------------------------------------------------------
def perpendicular_basis(n: np.ndarray, hint: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """단위벡터 n 에 수직인 정규직교 (e1, e2), e1 × e2 = n (오른손). 각 shape (3,)."""
    if hint is not None:
        e1 = np.asarray(hint, dtype=float) - np.dot(hint, n) * n
        if np.linalg.norm(e1) < 1e-9:
            raise ConfigError("평면 축 힌트가 법선과 평행합니다.")
    else:
        # n 과 가장 덜 평행한 좌표축에서 출발 (수치적으로 안정)
        axis = np.eye(3)[int(np.argmin(np.abs(n)))]
        e1 = axis - np.dot(axis, n) * n
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2


def plane_basis(plane: ObservationPlane) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """관측 평면의 (e_u, e_v, n). 각 shape (3,), e_u × e_v = n."""
    n = unit_vector(plane.normal, "관측 평면 법선")
    hint = None if plane.u_axis is None else np.asarray(plane.u_axis, dtype=float)
    e_u, e_v = perpendicular_basis(n, hint)
    return e_u, e_v, n


def plane_points(plane: ObservationPlane, spacing: float) -> np.ndarray:
    """관측 격자 입자의 평형 위치 [m]. shape (nx, ny, 3). 격자 중심 = plane.origin."""
    e_u, e_v, _ = plane_basis(plane)
    iu = (np.arange(plane.nx) - (plane.nx - 1) / 2.0) * spacing
    iv = (np.arange(plane.ny) - (plane.ny - 1) / 2.0) * spacing
    origin = np.asarray(plane.origin, dtype=float)
    return origin + iu[:, None, None] * e_u + iv[None, :, None] * e_v


# ---------------------------------------------------------------------------
# 소스 → 평면파 묶음
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlaneWaveSet:
    """평면파 M 개. amp (M,) [m], khat (M,3), kvec (M,3) [rad/m], omega (M,) [rad/s],
    phase (M,) [rad], ref (M,3) [m], source_index (M,) 원래 소스 번호."""

    amp: np.ndarray
    khat: np.ndarray
    kvec: np.ndarray
    omega: np.ndarray
    phase: np.ndarray
    ref: np.ndarray
    source_index: np.ndarray

    def __len__(self) -> int:
        return int(self.amp.shape[0])


def gaussian_spectrum_step(src: Source) -> float:
    """가우시안 빔 각스펙트럼 격자 간격 Δk_t [rad/m]."""
    sigma_k = np.sqrt(2.0) / float(src.waist)
    kt_max = GAUSS_SPECTRUM_SIGMAS * sigma_k
    return 2.0 * kt_max / (int(src.n_angular) - 1)


def gaussian_replica_period(src: Source) -> float:
    """이산 각스펙트럼 때문에 빔이 횡방향으로 반복되는 주기 [m] = 2π / Δk_t."""
    return 2.0 * np.pi / gaussian_spectrum_step(src)


def _gaussian_components(src: Source, k0: float):
    """가우시안 빔 → (가중치 합 1 로 정규화된 진폭 비율 (M,), 단위 전파방향 (M,3)).

    각스펙트럼 가중치 g = exp(−k_t² w0² / 4) (스칼라 프로파일 exp(−ρ²/w0²) 의 2D 푸리에 변환).
    """
    khat0 = unit_vector(src.direction, "빔 방향")
    e1, e2 = perpendicular_basis(khat0)
    n = int(src.n_angular)
    kt_max = GAUSS_SPECTRUM_SIGMAS * np.sqrt(2.0) / float(src.waist)
    kt = np.linspace(-kt_max, kt_max, n)
    k1, k2 = np.meshgrid(kt, kt, indexing="ij")
    kt2 = k1**2 + k2**2
    mask = kt2 < (PROPAGATING_FRAC * k0) ** 2
    k1, k2, kt2 = k1[mask], k2[mask], kt2[mask]
    weight = np.exp(-kt2 * float(src.waist) ** 2 / 4.0)
    weight = weight / weight.sum()
    kz = np.sqrt(k0**2 - kt2)
    dirs = (k1[:, None] * e1 + k2[:, None] * e2 + kz[:, None] * khat0) / k0
    return weight, dirs


def expand_sources(sources: tuple[Source, ...] | list[Source], medium: Medium) -> PlaneWaveSet:
    """활성 소스를 평면파 묶음으로 전개. 꺼진 소스와 진폭 0 소스는 제외."""
    amp, khat, omega, phase, ref, idx = [], [], [], [], [], []
    for i, src in enumerate(sources):
        if not src.enabled or float(src.amplitude) == 0.0:
            continue
        w = 2.0 * np.pi * float(src.frequency)
        k0 = w / medium.c
        if src.kind == "plane":
            weights = np.ones(1)
            dirs = unit_vector(src.direction, f"소스 {i + 1} 방향 벡터")[None, :]
        else:
            weights, dirs = _gaussian_components(src, k0)
        m = len(weights)
        amp.append(float(src.amplitude) * weights)
        khat.append(dirs)
        omega.append(np.full(m, w))
        phase.append(np.full(m, float(src.phase)))
        ref.append(np.tile(np.asarray(src.ref_point, dtype=float), (m, 1)))
        idx.append(np.full(m, i))
    if not amp:
        raise ValueError("활성 평면파가 없습니다.")
    khat_a = np.concatenate(khat)
    omega_a = np.concatenate(omega)
    return PlaneWaveSet(
        amp=np.concatenate(amp),
        khat=khat_a,
        kvec=khat_a * (omega_a / medium.c)[:, None],
        omega=omega_a,
        phase=np.concatenate(phase),
        ref=np.concatenate(ref),
        source_index=np.concatenate(idx),
    )


# ---------------------------------------------------------------------------
# 장 계산
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Phasors:
    """한 각주파수 ω 의 공간 페이저. P = 점 개수.
    U (P,3) 변위, G (P,3,3) 변위 기울기 G[p,i,j] = ∂_j u_i, D (P,) 발산."""

    omega: float
    U: np.ndarray
    G: np.ndarray | None
    D: np.ndarray


class AnalyticField:
    """엔진 A. 시간 독립 페이저를 만들고 임의 시각의 u, v, p, ∇u 를 합성한다."""

    def __init__(self, sources, medium: Medium):
        self.medium = medium
        self.waves = expand_sources(sources, medium)
        # 정확히 같은 주파수끼리 묶는다 (사용자가 입력한 float 값 기준).
        self._omegas = np.unique(self.waves.omega)

    @property
    def omegas(self) -> np.ndarray:
        return self._omegas

    def phasors(self, points: np.ndarray, with_grad: bool = False) -> list[Phasors]:
        """points (P,3) [m] → 주파수별 Phasors 목록."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        wv = self.waves
        out = []
        for w in self._omegas:
            sel = wv.omega == w
            kvec, khat, amp = wv.kvec[sel], wv.khat[sel], wv.amp[sel]
            # 기준점과 초기위상을 상수 위상으로 합침: k·(r − r_m) + φ = k·r + (φ − k·r_m)
            const = wv.phase[sel] - np.einsum("mj,mj->m", kvec, wv.ref[sel])
            kmag = np.linalg.norm(kvec, axis=1)
            avec = amp[:, None] * khat  # (M,3) 변위 벡터 진폭
            P = pts.shape[0]
            U = np.empty((P, 3), complex)
            D = np.empty(P, complex)
            G = np.empty((P, 3, 3), complex) if with_grad else None
            step = max(1, _CHUNK_ELEMS // max(1, len(amp)))
            for s in range(0, P, step):
                E = np.exp(1j * (pts[s : s + step] @ kvec.T + const))  # (p, M)
                U[s : s + step] = E @ avec
                D[s : s + step] = 1j * (E @ (amp * kmag))
                if with_grad:
                    # G_ij = Σ_m E_m · i a_m k̂_i k_j
                    tensor = avec[:, :, None] * kvec[:, None, :]  # (M,3,3)
                    G[s : s + step] = 1j * np.einsum("pm,mij->pij", E, tensor)
            out.append(Phasors(float(w), U, G, D))
        return out

    def evaluate(self, points: np.ndarray, times: np.ndarray, with_grad: bool = False) -> dict:
        """points (P,3) [m], times (Nt,) [s] →
        {"u": (Nt,P,3) [m], "v": (Nt,P,3) [m/s], "p": (Nt,P) [Pa], "grad": (Nt,P,3,3) 또는 None}."""
        t = np.atleast_1d(np.asarray(times, dtype=float))
        ph = self.phasors(points, with_grad=with_grad)
        P = ph[0].U.shape[0]
        u = np.zeros((t.size, P, 3))
        v = np.zeros((t.size, P, 3))
        div = np.zeros((t.size, P))
        grad = np.zeros((t.size, P, 3, 3)) if with_grad else None
        for f in ph:
            rot = np.exp(-1j * f.omega * t)  # (Nt,)
            u += np.real(rot[:, None, None] * f.U[None])
            v += np.real((-1j * f.omega) * rot[:, None, None] * f.U[None])
            div += np.real(rot[:, None] * f.D[None])
            if with_grad:
                grad += np.real(rot[:, None, None, None] * f.G[None])
        p = -self.medium.rho * self.medium.c**2 * div
        return {"u": u, "v": v, "p": p, "grad": grad}

    def displacement(self, points: np.ndarray, times: np.ndarray) -> np.ndarray:
        """변위만 필요할 때. (Nt,P,3) [m]."""
        return self.evaluate(points, times)["u"]


def numerical_curl_ratio(field: AnalyticField, points: np.ndarray, times: np.ndarray, h: float) -> float:
    """max|∇×u| / max|∇u| 를 3D 중앙차분 (간격 h [m]) 으로 계산. 무차원.

    관측 평면의 이웃 점만으로는 법선 방향 미분이 없어 회전을 계산할 수 없으므로, 각 점에서 ±h e_j
    (j = x,y,z) 오프셋 6 점을 따로 평가한다. 해석적 ∇u 를 쓰지 않는 이유: 해석식은 구조상 대칭이라
    변위 계산 자체의 버그(방향 오류 등)를 잡지 못한다. 차분 오차는 O((k h)²) 수준이다.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    grad = np.empty((len(times), pts.shape[0], 3, 3))  # [t,p,i,j] = ∂_j u_i
    for j in range(3):
        dh = np.zeros(3)
        dh[j] = h
        up = field.displacement(pts + dh, times)
        dn = field.displacement(pts - dh, times)
        grad[..., j] = (up - dn) / (2.0 * h)
    curl = np.stack(
        [
            grad[..., 2, 1] - grad[..., 1, 2],
            grad[..., 0, 2] - grad[..., 2, 0],
            grad[..., 1, 0] - grad[..., 0, 1],
        ],
        axis=-1,
    )
    gmax = float(np.max(np.linalg.norm(grad, axis=(-2, -1))))
    if gmax == 0.0:
        return float("nan")
    return float(np.max(np.linalg.norm(curl, axis=-1)) / gmax)
