"""검증용 해석해 (엔진 3D).

배플 원형 피스톤 (무한 강체 벽에 박힌 반지름 a 원판, 법선 속도 v0 e^{−iωt}) 의 축 위 음압 진폭:
    |p(z)| = 2 ρ c v0 |sin( (k/2) (√(z² + a²) − z) )|
    (Rayleigh 적분을 축 위에서 닫힌 꼴로 적분한 결과. Kinsler et al., Fundamentals of Acoustics, 7.4 절 — 판 번호 확인 필요)
관 속 평면파 (단면 전체가 피스톤, 옆벽 강체): |p| = ρ c v0 (파면이 평면이므로 임피던스 ρc).
"""

from __future__ import annotations

import numpy as np


def piston_on_axis(z: np.ndarray, radius: float, k: float, rho: float, c: float, v0: float) -> np.ndarray:
    """z [m] (피스톤 면에서 축 방향 거리, z ≥ 0) → |p| [Pa]."""
    z = np.asarray(z, float)
    return 2 * rho * c * v0 * np.abs(np.sin(0.5 * k * (np.sqrt(z**2 + radius**2) - z)))


def duct_plane_wave(rho: float, c: float, v0: float) -> float:
    return rho * c * v0
