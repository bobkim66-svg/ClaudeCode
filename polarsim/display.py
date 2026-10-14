"""화면 표시용 계산 (Qt 없음 → pytest 로 검사).

화면의 입자 위치는 평형 위치 x 에 시각 배율 g 를 곱한 변위를 더한 X = x + g·u(x) 이다.
이 사상의 국소 면적비 (야코비안 행렬식)

    A(x,t) = det(I + g ∇u_∥),   ∇u_∥ = 관측 평면 안의 2×2 변위 기울기

가 화면에서 입자가 '뭉치는'(A < 1) / '흩어지는'(A > 1) 정도다. A ≤ 0 이면 이웃 입자가 서로를 지나쳐
보인다 (실제로는 일어나지 않는 착시). 선형 음향에서 ∇·u = −p/(ρc²) 이므로 뭉치는 곳 = 압력이 높은 곳이다.

자동 배율은 '목표 압축률' c* 로 정한다: g = c* / max‖∇u_∥‖_F (기울기 진폭의 최댓값).
한 방향 압축(평면파, 정재파) 이면 최대 면적 변화가 정확히 c* 이다. 그래서 진폭·격자 간격·주파수를 바꿔도
화면에서 보이는 뭉침 정도가 같다 (진폭을 키운다고 더 잘 보이지 않는 이유).
"""

from __future__ import annotations

import numpy as np

CROSSING_AREA = 0.1  # 면적비가 이보다 작으면 '입자가 겹쳐/지나쳐 보임' 경고


def inplane_gradient(G: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """3D 변위 기울기 G (...,3,3) [G_ij = ∂u_i/∂x_j] → 관측 평면 성분 (...,2,2) = E G Eᵀ, E = basis[:2] (2,3)."""
    E = basis[:2]
    return np.einsum("ai,...ij,bj->...ab", E, G, E)


def gradient_scale(G2_phasors: list[np.ndarray], mask: np.ndarray | None = None, percentile: float = 100.0) -> float:
    """주파수별 평면 내 기울기 페이저 목록 [(P,2,2) complex] → 시간에 대한 ‖∇u_∥‖_F 상한의 최댓값 (무차원).

    percentile < 100 이면 그 백분위 값 (벽 모서리처럼 기울기가 특이한 몇 점에 끌려가지 않게, 엔진 B 용)."""
    amp = sum(np.sqrt(np.sum(np.abs(G2) ** 2, axis=(-2, -1))) for G2 in G2_phasors)
    if mask is not None:
        amp = amp[mask]
    amp = amp[np.isfinite(amp)]
    if amp.size == 0:
        return 0.0
    return float(np.percentile(amp, percentile))


def emphasis_gain(scale: float, target_compression: float) -> float:
    """목표 압축률 (예: 0.6 = 화면 면적 최대 60% 변화) 에 맞는 시각 배율. scale 은 gradient_scale 값."""
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("변위 기울기가 0 이라 배율을 정할 수 없습니다.")
    return target_compression / scale


def area_ratio(G2_t: np.ndarray, gain: float) -> np.ndarray:
    """시각 t 의 실수 평면 내 기울기 (...,2,2) → 화면 면적비 det(I + g G) (...)."""
    a = 1 + gain * G2_t[..., 0, 0]
    d = 1 + gain * G2_t[..., 1, 1]
    return a * d - gain**2 * G2_t[..., 0, 1] * G2_t[..., 1, 0]


def frame_real(phasors: list[tuple[float, np.ndarray]], t: float) -> np.ndarray:
    """[(ω, Q complex)] → Σ Re(Q e^{−iωt})."""
    return sum(np.real(Q * np.exp(-1j * w * t)) for w, Q in phasors)
