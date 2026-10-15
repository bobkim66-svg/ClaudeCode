"""스페클 필터 평가 지표.

- ENL  = mu^2 / sigma^2  (동질 영역, 강도 영역). L-look 감마 스페클의 원영상은 ENL ~ L.
- PSNR, SSIM: 표시(진폭) 영역 sqrt(I) / sqrt(max clean) 를 [0,1] 로 자른 값으로 비교.
  (강도 영역은 스페클 꼬리 때문에 MSE 가 극단값에 지배되므로 표시 영역을 사용 - 가정)
- CNR  = |mu_t - mu_b| / sqrt(sigma_t^2 + sigma_b^2)  (강도 영역)
- EPI  = 라플라시안 상관 (Sattar et al. 1997 형태, 확인 필요): 1 이면 경계 완전 보존.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import laplace
from skimage.metrics import structural_similarity


def _check_mask(mask: np.ndarray, name: str, min_px: int = 20) -> None:
    if mask.dtype != bool:
        raise TypeError(f"{name} 는 bool 마스크")
    if mask.sum() < min_px:
        raise ValueError(f"{name} 화소 수 {mask.sum()} < {min_px}: 통계가 불안정하다")


def enl(image: np.ndarray, mask: np.ndarray) -> float:
    """등가 look 수. image (H,W) 강도, mask (H,W) bool (동질 영역)."""
    _check_mask(mask, "mask")
    v = image[mask]
    var = v.var()
    return float(v.mean() ** 2 / var) if var > 0 else float("inf")


def to_display(image: np.ndarray, ref_max: float) -> np.ndarray:
    """강도 -> 진폭 정규화 [0,1]."""
    return np.clip(np.sqrt(np.maximum(image, 0.0) / ref_max), 0.0, 1.0)


def psnr(clean: np.ndarray, image: np.ndarray) -> float:
    """PSNR [dB], 표시 영역, data_range = 1."""
    ref_max = float(clean.max())
    mse = np.mean((to_display(clean, ref_max) - to_display(image, ref_max)) ** 2)
    return float(10 * np.log10(1.0 / mse)) if mse > 0 else float("inf")


def ssim(clean: np.ndarray, image: np.ndarray) -> float:
    """SSIM (skimage 구현, 가우시안 창 sigma=1.5), 표시 영역."""
    ref_max = float(clean.max())
    return float(
        structural_similarity(
            to_display(clean, ref_max), to_display(image, ref_max), data_range=1.0, gaussian_weights=True, sigma=1.5
        )
    )


def cnr(image: np.ndarray, target_mask: np.ndarray, background_mask: np.ndarray) -> float:
    """대비-잡음비. 두 마스크 (H,W) bool."""
    _check_mask(target_mask, "target_mask")
    _check_mask(background_mask, "background_mask")
    t, b = image[target_mask], image[background_mask]
    return float(abs(t.mean() - b.mean()) / np.sqrt(t.var() + b.var()))


def epi(clean: np.ndarray, image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """경계 보존 지수: 정답/필터 영상 라플라시안 (표시 영역) 의 상관계수."""
    ref_max = float(clean.max())
    a = laplace(to_display(clean, ref_max))
    b = laplace(to_display(image, ref_max))
    if mask is not None:
        a, b = a[mask], b[mask]
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / den) if den > 0 else 0.0
