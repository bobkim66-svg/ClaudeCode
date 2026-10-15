"""잉여체 F_q = Z[w]/(pi) 위의 지표(character)와 가우스 합.

- 곱셈 지표 chi_k(g^j) = exp(2 pi i k j / (q-1)),  g = 원시근,  chi_k(0) = 0
- 덧셈 지표 psi(x)    = exp(2 pi i x / q)
- 가우스 합 G(chi, psi) = sum_{x in F_q^*} chi(x) psi(x)
  * chi 가 비자명이면 |G| = sqrt(q),  자명(k=0)이면 G = -1
  * 이차 지표의 고전값: q = 1 (mod 4) -> sqrt(q),  q = 3 (mod 4) -> i sqrt(q)
  * 곱셈/덧셈 푸리에 변환을 잇는 항등식:  sum_t chi(t) psi(t x) = conj(chi(x)) G  (x != 0)

필터 설계에 쓰는 구조적 사실 (eisenstein.py 의 UNIT_OFFSETS 참조):
  q 가 분해소수(q = 1 mod 3, q >= 7)이면 단위군 mu_6 = {zeta^j} 는 F_q^* 에 단사로 들어간다
  (서로 다른 두 단위의 차의 노름은 <= 4 < q 이라 pi 로 나누어지지 않음).
  특히 q = 7 이면 mu_6 -> F_7^* 는 동형이고, 육각 7점 클러스터가 Z[w]/(pi) 의 완전잉여계다.
  그러므로 chi 를 단위 고리(1-링 이웃)로 제한하면 회전군 mu_6 의 지표, 즉
  각도 조화성분 exp(i m theta_j) (theta_j = j * 60 deg) 이 된다.
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from .eisenstein import UNIT_OFFSETS, EisensteinInt, omega_residue


def primitive_root(q: int) -> int:
    """소수 q 의 최소 원시근."""
    if q == 2:
        return 1
    factors = {p for p in range(2, q) if (q - 1) % p == 0 and all(p % d for d in range(2, int(p ** 0.5) + 1))}
    for g in range(2, q):
        if all(pow(g, (q - 1) // p, q) != 1 for p in factors):
            return g
    raise RuntimeError(f"원시근을 찾지 못함: q={q}")


class GaussSumKernel:
    """아이젠슈타인 소수 pi 로 만든 잉여체 F_q 의 지표, 가우스 합, 커널 가중치.

    Parameters
    ----------
    prime : EisensteinInt
        N(pi) 가 유리소수인 아이젠슈타인 소수 (예: 3+w -> q=7).
    chi_index : int
        기본 곱셈 지표 번호 k (0 <= k < q-1). k=0 은 자명 지표.
    """

    def __init__(self, prime: EisensteinInt, chi_index: int = 1) -> None:
        self.prime = prime
        self.omega_res, self.q = omega_residue(prime)
        q = self.q
        if not 0 <= chi_index < q - 1:
            raise ValueError(f"chi_index 는 0..{q - 2} 범위여야 한다")
        self.chi_index = chi_index
        self.g = primitive_root(q)
        # 이산로그 표: dlog[g^j mod q] = j,  dlog[0] 은 사용 안 함(-1)
        self._dlog = np.full(q, -1, dtype=int)
        for j in range(q - 1):
            self._dlog[pow(self.g, j, q)] = j

    # ---- 잉여 사상 ---------------------------------------------------------
    def residue(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """a + b w (mod pi) -> Z/q 의 대표원 (0..q-1). 입력 정수 배열, shape 유지."""
        return (np.asarray(a) + np.asarray(b) * self.omega_res) % self.q

    # ---- 지표 --------------------------------------------------------------
    def chi(self, x: np.ndarray, k: int | None = None) -> np.ndarray:
        """곱셈 지표 chi_k(x), x in Z/q (정수 배열). chi(0) = 0. 출력 complex, shape 유지."""
        k = self.chi_index if k is None else k
        x = np.asarray(x) % self.q
        out = np.exp(2j * np.pi * k * self._dlog[x] / (self.q - 1))
        return np.where(x == 0, 0.0, out)

    def psi(self, x: np.ndarray, t: int = 1) -> np.ndarray:
        """덧셈 지표 psi_t(x) = exp(2 pi i t x / q). 출력 complex, shape 유지."""
        return np.exp(2j * np.pi * t * (np.asarray(x) % self.q) / self.q)

    def gauss_sum(self, k: int | None = None) -> complex:
        """G(chi_k, psi) 를 정의대로 직접 합산 (q-1 항)."""
        x = np.arange(1, self.q)
        return complex(np.sum(self.chi(x, k) * self.psi(x)))

    # ---- (A) 원안: 고정 복소 커널 -------------------------------------------
    def kernel_weights(self, offsets: np.ndarray) -> np.ndarray:
        """격자 오프셋 z = (da, db) 의 가중치 w(z) = chi(z mod pi) psi(z mod pi) / sum_z(...).

        offsets: shape (n, 2) int. 출력 shape (n,) complex, sum = 1 (상수 영상 보존).
        7점 클러스터(q=7)이면 정규화 상수가 정확히 G(chi, psi) 이다 (완전잉여계, chi(0)=0).
        """
        res = self.residue(offsets[:, 0], offsets[:, 1])
        w = self.chi(res) * self.psi(res)
        s = w.sum()
        if abs(s) < 1e-9:
            raise ValueError("커널 가중치 합이 0 이라 정규화할 수 없다 (prime/window 조합 변경 필요)")
        return w / s

    # ---- (B) 적응형: 단위 고리 위 지표 = 각도 조화성분 -----------------------
    def unit_harmonic(self, k: int) -> int:
        """chi_k 를 mu_6 로 제한했을 때의 각도 조화 차수 m (0..5): chi_k(zeta^j) = exp(2 pi i m j/6)."""
        if self.q < 7 or self.q % 3 != 1:
            raise ValueError(
                f"q={self.q}: 단위군 mu_6 가 F_q^* 에 단사로 들어가려면 q = 1 (mod 3), q >= 7 이어야 한다"
            )
        za, zb = UNIT_OFFSETS[1]  # zeta = 1 + w
        val = complex(self.chi(self.residue(za, zb), k))
        m = int(round(np.angle(val) / (np.pi / 3))) % 6
        if not np.isclose(val, np.exp(1j * np.pi * m / 3), atol=1e-9):
            raise AssertionError("제한된 지표가 6차 단위근이 아니다 (구현 오류)")
        return m

    def ring_characters(self, harmonics: Iterable[int] = (1, 2)) -> Dict[int, np.ndarray]:
        """각 조화 차수 m 에 대해 단위 고리 계수 c_j = chi_k(res(zeta^j)), j=0..5 를 돌려준다.

        k 는 F_q^* 의 지표 중 mu_6 제한이 m 이 되는 것을 찾는다. 출력 {m: shape (6,) complex}.
        지표 직교성 sum_j c_j = 0 (m != 0) 이 성립하므로 고리의 평균(DC) 성분은 정확히 상쇄된다.
        """
        table = {self.unit_harmonic(k): k for k in range(self.q - 1)}
        out = {}
        for m in harmonics:
            if m not in table:
                raise ValueError(f"조화 차수 {m} 를 주는 지표가 없다 (q={self.q})")
            res = np.array([self.residue(da, db) for da, db in UNIT_OFFSETS])
            out[m] = self.chi(res, table[m])
        return out
