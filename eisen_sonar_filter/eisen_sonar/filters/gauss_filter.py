"""아이젠슈타인-가우스 합 스페클 필터.

세 가지 모드 (모두 육각 격자 Z[w] 위에서 동작):

- "resample_only": 직교 -> 육각 -> 직교 왕복만. 보간 자체의 평활 효과를 분리하는 대조군.
- "linear"  (A, 원안): 고정 복소 커널 w(z) = chi(z mod pi) psi(z mod pi) / sum, 출력 = Re(sum_z w(z) I(p+z)).
  가중치는 영상과 무관한 상수라서 선형 FIR 필터와 같다 (적응성 없음).
- "adaptive" (B): 로그 강도 영역에서 단위 고리(1-링) 위 곱셈 지표 합
      S_m(p) = sum_{j=0..5} chi_m(zeta^j mod pi) * L(p + s zeta^j)
  을 구한다. mu_6 -> F_q^* 가 단사이므로 chi_m 은 각도 조화성분이고,
  지표 직교성 sum_j chi_m(zeta^j) = 0 덕분에 국소 평균(동질 영역의 DC)은 정확히 상쇄된다.
  곱셈 스페클은 로그에서 덧셈 잡음이 되므로, 동질 영역에서는 |S_m|^2 가 근사적으로
  지수분포(카이제곱 2자유도/2)를 따른다. 이를 귀무분포로 보정한 구조 통계량 T 로
  Lee 필터처럼 "육각 원판 평균"과 "경계/선 방향 선평균"을 섞는다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import gamma

from ..core.eisenstein import (
    UNIT_OFFSETS,
    EisensteinInt,
    hex_disk,
    hex_shift,
    hex_to_cartesian,
    resample_to_hex_grid,
)
from ..core.gauss_sum import GaussSumKernel

MODES = ("adaptive", "linear", "resample_only")


def hex_weighted_sum(hex_values: np.ndarray, offsets: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """sum_z w(z) H(p + z). offsets shape (n,2), weights shape (n,). 출력 shape = hex_values.shape."""
    out = np.zeros(hex_values.shape, dtype=np.result_type(hex_values, weights))
    for (da, db), w in zip(offsets, weights):
        out += w * hex_shift(hex_values, int(da), int(db))
    return out


@dataclass
class EisensteinGaussFilter:
    """2D 소나 강도 영상용 아이젠슈타인-가우스 합 필터.

    Parameters
    ----------
    prime_a, prime_b : 아이젠슈타인 소수 pi = a + b w. N(pi) 는 유리소수여야 한다.
        adaptive 모드는 q = N(pi) = 1 (mod 3), q >= 7 이 필요 (예: 3+w -> 7, 4+w -> 13).
        주의: adaptive 결과는 그런 소수 선택과 무관하다 (모두 같은 각도 조화성분을 준다).
    window_size : 홀수 3, 5, 7, ... -> 육각 원판 반경 R = (window_size-1)//2 (7, 19, 37 점).
    mode : "adaptive" | "linear" | "resample_only".
    hex_spacing : 육각 격자 간격 [pixel].
    detector_scale : 구조 검출 고리 반경 s [격자칸] (s * zeta^j 이웃 사용).
    presmooth : 검출 전 로그영상에 7점 육각 평균을 적용할지 (단일 look 스페클의 분산 감소).
    p_false_alarm : 동질 영역에서 구조로 오판할 확률. 임계값 tau = Gamma(M,1) 의 상위 분위수.
    harmonics : 사용할 각도 조화 차수 (1: 경계/쌍극자, 2: 선/사중극자).
    eps_rel : log(0) 방지용 상대 오프셋 (영상 평균 대비).
    """

    prime_a: int = 3
    prime_b: int = 1
    window_size: int = 5
    mode: str = "adaptive"
    hex_spacing: float = 1.0
    detector_scale: int = 1
    presmooth: bool = True
    p_false_alarm: float = 0.05
    harmonics: tuple = (1, 2)
    eps_rel: float = 1e-6

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode 는 {MODES} 중 하나")
        if self.window_size < 3 or self.window_size % 2 == 0:
            raise ValueError("window_size 는 3 이상의 홀수")
        if not 0 < self.p_false_alarm < 1:
            raise ValueError("0 < p_false_alarm < 1")
        self.radius = (self.window_size - 1) // 2
        self.kernel = GaussSumKernel(EisensteinInt(self.prime_a, self.prime_b))
        self.disk = hex_disk(self.radius)
        if self.mode == "linear":
            self.weights = self.kernel.kernel_weights(self.disk)
        if self.mode == "adaptive":
            if 1 not in self.harmonics:
                raise ValueError("방향 추정에 1차 조화성분이 필요하다")
            self.ring_chars = self.kernel.ring_characters(self.harmonics)
            self.tau = float(gamma.isf(self.p_false_alarm, a=len(self.harmonics)))
        self.last_diagnostics: dict = {}

    # ------------------------------------------------------------------
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """스페클 영상 (H, W) 강도 [선형, >= 0] -> 필터 영상 (H, W) 강도."""
        image = np.asarray(image, dtype=float)
        if image.ndim != 2 or min(image.shape) < 3:
            raise ValueError(f"(H>=3, W>=3) 2D 영상이어야 한다: shape={image.shape}")
        if not np.all(np.isfinite(image)):
            raise ValueError("영상에 NaN/inf 가 있다")
        if image.min() < 0:
            raise ValueError("강도 영상은 음수가 없어야 한다 (진폭이면 제곱해서 넣을 것)")

        hexv, grid = resample_to_hex_grid(image, self.hex_spacing)
        if self.mode == "resample_only":
            out = hexv
        elif self.mode == "linear":
            out = np.maximum(hex_weighted_sum(hexv, self.disk, self.weights).real, 0.0)
        else:
            out = self._adaptive(hexv, grid)
        return hex_to_cartesian(out, grid)

    # ------------------------------------------------------------------
    def _adaptive(self, hexv: np.ndarray, grid) -> np.ndarray:
        n = len(self.disk)
        disk_mean = hex_weighted_sum(hexv, self.disk, np.full(n, 1.0 / n))

        eps = self.eps_rel * max(float(hexv.mean()), np.finfo(float).tiny)
        logv = np.log(hexv + eps)
        if self.presmooth:
            d1 = hex_disk(1)
            logv = hex_weighted_sum(logv, d1, np.full(len(d1), 1.0 / len(d1)))

        s = self.detector_scale
        ring = np.stack([hex_shift(logv, s * da, s * db) for da, db in UNIT_OFFSETS])  # (6, nb, na)

        # 영상 안쪽 격자점만으로 귀무분포 척도를 추정 (margin 의 복제값 제외)
        x, y = grid.lattice_xy()
        h, w = grid.image_shape
        inside = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)

        T = np.zeros(hexv.shape)
        S = {}
        for m, c in self.ring_chars.items():
            S[m] = np.tensordot(c, ring, axes=1)  # sum_j c_j L(p + s zeta^j)
            e = np.abs(S[m]) ** 2
            scale = np.median(e[inside]) / np.log(2.0)  # Exp(1) 의 중앙값 = ln 2
            T += e / max(scale, np.finfo(float).tiny)
        k = np.clip((T - self.tau) / self.tau, 0.0, 1.0)

        # 방향: 1차(경계) -> 기울기 방향 phi = arg S1, 접선 = phi + 90 deg
        #       2차(선)   -> 선 방향 = arg S2 / 2
        # c_1j = exp(i theta_j) 이므로 S1 = gx + i gy (unit_harmonic 에서 m=1 로 선택됨)
        tangent = np.angle(S[1]) + np.pi / 2
        if 2 in S:
            use_line = np.abs(S[2]) ** 2 > np.abs(S[1]) ** 2
            tangent = np.where(use_line, 0.5 * np.angle(S[2]), tangent)
        axis = np.rint(np.mod(tangent, np.pi) / (np.pi / 3)).astype(int) % 3  # 0, 60, 120 deg 축

        line_means = []
        for j in range(3):
            da, db = UNIT_OFFSETS[j]
            offs = np.array([(t * da, t * db) for t in range(-self.radius, self.radius + 1)])
            line_means.append(hex_weighted_sum(hexv, offs, np.full(len(offs), 1.0 / len(offs))))
        line_mean = np.choose(axis, line_means)

        self.last_diagnostics = {"T": T, "k": k, "axis": axis, "inside": inside, "grid": grid}
        return (1.0 - k) * disk_mean + k * line_mean
