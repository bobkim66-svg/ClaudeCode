"""고전 적응형 스페클 필터 (비교 기준선): Lee, Kuan, Frost.

공통 기호 (window 안 국소 통계, 강도 영역):
    m = 국소 평균, v = 국소 분산, Ci^2 = v / m^2  (국소 변동계수 제곱)
    Cu^2 = 1 / L  (L-look 감마 스페클의 변동계수 제곱, 동질 영역 기준)

- Lee  (1980):  out = m + W (I - m),  W = clip(1 - Cu^2/Ci^2, 0, 1)
- Kuan (1985):  W = clip((1 - Cu^2/Ci^2) / (1 + Cu^2), 0, 1)
- Frost(1982):  out = sum_t K(t) I(p+t) / sum_t K(t),  K(t) = exp(-D Ci^2(p) |t|)
(정확한 원 논문 표기와 상수는 확인 필요. 위 식은 널리 쓰이는 단순화 형태다.)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def _local_stats(image: np.ndarray, window: int):
    if window < 3 or window % 2 == 0:
        raise ValueError("window 는 3 이상의 홀수")
    img = np.asarray(image, dtype=float)
    m = uniform_filter(img, window, mode="reflect")
    m2 = uniform_filter(img * img, window, mode="reflect")
    v = np.maximum(m2 - m * m, 0.0)
    ci2 = v / np.maximum(m * m, np.finfo(float).tiny)
    return img, m, ci2


def lee_filter(image: np.ndarray, window: int = 5, n_looks: float = 1.0) -> np.ndarray:
    """Lee 필터. image (H, W) 강도, n_looks = 스페클 look 수 L. 출력 (H, W)."""
    img, m, ci2 = _local_stats(image, window)
    cu2 = 1.0 / n_looks
    W = np.clip(1.0 - cu2 / np.maximum(ci2, 1e-12), 0.0, 1.0)
    return m + W * (img - m)


def kuan_filter(image: np.ndarray, window: int = 5, n_looks: float = 1.0) -> np.ndarray:
    """Kuan 필터. 입력/출력은 lee_filter 와 같다."""
    img, m, ci2 = _local_stats(image, window)
    cu2 = 1.0 / n_looks
    W = np.clip((1.0 - cu2 / np.maximum(ci2, 1e-12)) / (1.0 + cu2), 0.0, 1.0)
    return m + W * (img - m)


def frost_filter(image: np.ndarray, window: int = 5, damping: float = 2.0) -> np.ndarray:
    """Frost 필터. damping D 가 클수록 고변동(경계) 영역에서 덜 평활. 출력 (H, W)."""
    img, m, ci2 = _local_stats(image, window)
    r = window // 2
    p = np.pad(img, r, mode="reflect")
    h, w = img.shape
    num = np.zeros_like(img)
    den = np.zeros_like(img)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            k = np.exp(-damping * ci2 * np.hypot(dx, dy))
            num += k * p[r + dy: r + dy + h, r + dx: r + dx + w]
            den += k
    return num / den
