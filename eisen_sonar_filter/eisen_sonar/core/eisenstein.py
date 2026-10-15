"""아이젠슈타인 정수 Z[w] 산술과 육각 격자 매핑.

w = exp(i 2pi/3) = -1/2 + i sqrt(3)/2,  w^2 = -1 - w.
a + b w  <->  직교좌표 (x, y) = (a - b/2, (sqrt(3)/2) b)   (격자 간격 1 기준)

좌표 약속: 영상 화소 (row, col) 를 (y, x) = (row, col) 로 본다 (y 가 아래 방향).
이 반사(reflection)는 격자 구조와 거리에는 영향이 없고, 각도 부호만 바뀐다.
한 모듈 안에서 일관되게 쓰므로 문제 없다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.ndimage import map_coordinates

SQRT3_2 = np.sqrt(3.0) / 2.0

# 단위원 6개 (단위군 mu_6) 를 각도 순서로: zeta^j, zeta = 1 + w = exp(i pi/3)
# 축(axial) 좌표 (da, db):  1, 1+w, w, -1, w^2=-1-w, -w
UNIT_OFFSETS: Tuple[Tuple[int, int], ...] = ((1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1))


@dataclass(frozen=True)
class EisensteinInt:
    """아이젠슈타인 정수 a + b w (a, b 는 정수)."""

    a: int
    b: int

    def __post_init__(self) -> None:
        if not (isinstance(self.a, (int, np.integer)) and isinstance(self.b, (int, np.integer))):
            raise TypeError(f"계수는 정수여야 한다: ({self.a!r}, {self.b!r})")
        object.__setattr__(self, "a", int(self.a))
        object.__setattr__(self, "b", int(self.b))

    def __add__(self, o: "EisensteinInt") -> "EisensteinInt":
        return EisensteinInt(self.a + o.a, self.b + o.b)

    def __sub__(self, o: "EisensteinInt") -> "EisensteinInt":
        return EisensteinInt(self.a - o.a, self.b - o.b)

    def __neg__(self) -> "EisensteinInt":
        return EisensteinInt(-self.a, -self.b)

    def __mul__(self, o: "EisensteinInt") -> "EisensteinInt":
        # (a + b w)(c + d w) = ac + (ad + bc) w + bd w^2,  w^2 = -1 - w
        a, b, c, d = self.a, self.b, o.a, o.b
        return EisensteinInt(a * c - b * d, a * d + b * c - b * d)

    def conj(self) -> "EisensteinInt":
        """복소켤레: conj(w) = w^2 = -1 - w  ->  (a - b) - b w."""
        return EisensteinInt(self.a - self.b, -self.b)

    def norm(self) -> int:
        """N(a + b w) = a^2 - ab + b^2 = |z|^2 (항상 0 이상의 정수)."""
        return self.a * self.a - self.a * self.b + self.b * self.b

    def to_complex(self) -> complex:
        return complex(self.a - 0.5 * self.b, SQRT3_2 * self.b)

    def __divmod__(self, o: "EisensteinInt") -> Tuple["EisensteinInt", "EisensteinInt"]:
        """유클리드 나눗셈: self = q*o + r,  N(r) < N(o).

        z/o = z*conj(o)/N(o) 의 두 계수를 각각 반올림한다. 평행사변형 기본영역에서
        반올림 잔차의 최대 노름은 3/4 < 1 이므로 N(r) < N(o) 가 보장된다.
        """
        n = o.norm()
        if n == 0:
            raise ZeroDivisionError("0 으로 나눌 수 없다")
        num = self * o.conj()
        q = EisensteinInt(int(round(num.a / n)), int(round(num.b / n)))
        r = self - q * o
        return q, r


def is_rational_prime(n: int) -> bool:
    if n < 2:
        return False
    return all(n % p for p in range(2, int(n ** 0.5) + 1))


def omega_residue(prime: EisensteinInt) -> Tuple[int, int]:
    """환 준동형 Z[w] -> Z[w]/(pi) ~= Z/q 에서 w 의 상 r 과 q = N(pi) 를 돌려준다.

    pi = a + b w 이면 몫에서 a + b w = 0, 즉 w = -a b^{-1} (mod q).
    N(pi) 가 유리소수일 때만 몫이 소수체 F_q 이다 (분해소수 q=1 mod 3, 또는 q=3).
    관성소수 p=2 mod 3 은 몫이 F_{p^2} 라서 이 구현에서는 거부한다.

    Returns: (r, q),  r^2 + r + 1 = 0 (mod q).
    """
    q = prime.norm()
    if not is_rational_prime(q):
        raise ValueError(
            f"N(pi) = {q} 가 유리소수가 아니다. 몫환이 소수체 F_q 가 되려면 N(pi) 가 소수여야 한다 "
            "(예: 1+2w -> 3, 3+w -> 7, 4+w -> 13)."
        )
    # q | N(pi) 이고 q 가 소수이면 b != 0 (mod q) (아니면 q^2 | N(pi) 로 모순)
    r = (-prime.a * pow(prime.b, -1, q)) % q
    assert (r * r + r + 1) % q == 0, "w 의 상이 w^2+w+1=0 을 만족해야 한다"
    return r, q


def hex_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """축 좌표 (a, b) 의 원점까지 육각 거리 (이웃 1칸 = 1). 이웃: (+-1,0),(0,+-1),(1,1),(-1,-1)."""
    a = np.asarray(a)
    b = np.asarray(b)
    return np.maximum(np.maximum(np.abs(a), np.abs(b)), np.abs(a - b))


def hex_disk(radius: int) -> np.ndarray:
    """육각 거리 <= radius 인 격자점 오프셋. 출력 shape (n, 2) int, n = 3R(R+1)+1."""
    if radius < 0:
        raise ValueError("radius >= 0")
    rng = np.arange(-radius, radius + 1)
    aa, bb = np.meshgrid(rng, rng, indexing="ij")
    keep = hex_distance(aa, bb) <= radius
    return np.stack([aa[keep], bb[keep]], axis=1)


def cartesian_to_eisenstein(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """직교 (x, y) -> 연속 아이젠슈타인 좌표 (a, b). 격자 간격 1.

    역변환 x = a - b/2, y = (sqrt3/2) b 를 풀면 b = y / (sqrt3/2), a = x + b/2.
    입력 shape 임의(브로드캐스트), 출력 같은 shape 의 float.
    """
    b = np.asarray(y, dtype=float) / SQRT3_2
    a = np.asarray(x, dtype=float) + 0.5 * b
    return a, b


def eisenstein_to_cartesian(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """아이젠슈타인 (a, b) -> 직교 (x, y) = (a - b/2, (sqrt3/2) b). 격자 간격 1."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a - 0.5 * b, SQRT3_2 * b


@dataclass(frozen=True)
class HexGrid:
    """직교 영상을 덮는 육각 격자의 기하 정보.

    육각 배열 H[j, i] 는 격자점 a = a0 + i, b = b0 + j 에 해당한다.
    점의 화소 좌표: x(col) = s (a - b/2), y(row) = s (sqrt3/2) b,  s = spacing [pixel].
    축(axial) 배열 저장이라 직사각 영상을 덮을 때 약 1.5배 공간을 쓰지만(평행사변형),
    대신 모든 이웃 연산이 배열 평행이동 하나로 끝나서 벡터화가 단순하다.
    """

    a0: int
    b0: int
    na: int
    nb: int
    spacing: float
    image_shape: Tuple[int, int]

    def lattice_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        """모든 격자점의 화소 좌표 (x, y). 출력 shape 각각 (nb, na), 단위 pixel."""
        a = self.a0 + np.arange(self.na)[None, :]
        b = self.b0 + np.arange(self.nb)[:, None]
        x, y = np.broadcast_arrays(*eisenstein_to_cartesian(a, b))
        return x * self.spacing, y * self.spacing


def make_hex_grid(image_shape: Tuple[int, int], spacing: float = 1.0, margin: int = 2) -> HexGrid:
    """영상 전체(+margin 격자칸)를 덮는 육각 격자를 만든다.

    spacing 이 1 pixel 이면 격자점 밀도 = 2/sqrt(3) ~ 1.155 점/pixel (약간 과표본화).
    """
    if spacing <= 0:
        raise ValueError("spacing > 0")
    h, w = image_shape
    b_min = -margin
    b_max = int(np.ceil((h - 1) / (spacing * SQRT3_2))) + margin
    a_min = int(np.floor(0.5 * b_min)) - margin
    a_max = int(np.ceil((w - 1) / spacing + 0.5 * b_max)) + margin
    return HexGrid(a_min, b_min, a_max - a_min + 1, b_max - b_min + 1, float(spacing), (h, w))


def resample_to_hex_grid(image: np.ndarray, spacing: float = 1.0) -> Tuple[np.ndarray, HexGrid]:
    """직교 영상 -> 육각 격자 (쌍선형 보간, 경계 밖은 최근접값).

    image: shape (H, W) float. Returns (hex_values shape (nb, na), HexGrid).
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"2D 영상이어야 한다: shape={image.shape}")
    grid = make_hex_grid(image.shape, spacing)
    x, y = grid.lattice_xy()
    vals = map_coordinates(image, [y.ravel(), x.ravel()], order=1, mode="nearest")
    return vals.reshape(grid.nb, grid.na), grid


def hex_to_cartesian(hex_values: np.ndarray, grid: HexGrid) -> np.ndarray:
    """육각 격자 -> 직교 영상 (정삼각형 무게중심 선형 보간).

    축 좌표 칸 [a0, a0+1] x [b0, b0+1] 은 짧은 대각선 (0) - (1+w) 로 두 정삼각형
    {0, 1, 1+w}, {0, w, 1+w} 로 나뉜다 (|1+w| = 1, |1-w| = sqrt3).
    소수부 (u, v) 에서 u >= v 면 첫 삼각형: 가중치 (1-u, u-v, v),
    아니면 둘째 삼각형: (1-v, v-u, u). 선형 함수는 정확히 복원된다.

    hex_values: shape (grid.nb, grid.na). 출력 shape grid.image_shape.
    """
    if hex_values.shape != (grid.nb, grid.na):
        raise ValueError(f"shape 불일치: {hex_values.shape} vs {(grid.nb, grid.na)}")
    h, w = grid.image_shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    a, b = cartesian_to_eisenstein(xx / grid.spacing, yy / grid.spacing)
    fi = a - grid.a0
    fj = b - grid.b0
    i0 = np.floor(fi).astype(int)
    j0 = np.floor(fj).astype(int)
    u = fi - i0
    v = fj - j0
    if i0.min() < 0 or j0.min() < 0 or i0.max() + 1 >= grid.na or j0.max() + 1 >= grid.nb:
        raise RuntimeError("격자 margin 부족: 보간 꼭짓점이 배열 밖이다")
    f00 = hex_values[j0, i0]
    f11 = hex_values[j0 + 1, i0 + 1]
    f10 = hex_values[j0, i0 + 1]  # (a+1, b)
    f01 = hex_values[j0 + 1, i0]  # (a, b+1)
    upper = u >= v
    return np.where(
        upper,
        (1 - u) * f00 + (u - v) * f10 + v * f11,
        (1 - v) * f00 + (v - u) * f01 + u * f11,
    )


def hex_shift(hex_values: np.ndarray, da: int, db: int) -> np.ndarray:
    """G[j, i] = H[j + db, i + da] (경계는 가장자리값 복제). 이웃값 모으기용. shape 유지."""
    pad = max(abs(da), abs(db))
    if pad == 0:
        return hex_values
    p = np.pad(hex_values, pad, mode="edge")
    nb, na = hex_values.shape
    return p[pad + db: pad + db + nb, pad + da: pad + da + na]
