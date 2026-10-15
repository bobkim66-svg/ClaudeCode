"""해상도(PSF), 스페클, 열잡음, 아티팩트.

복소 신호 모델: Z = (sqrt(I0) w) (*) g + n,  w ~ CN(0,1) 백색,  n ~ CN(0, N0(r))
  g = 거리방향 가우시안 (sigma_r = 거리해상도/2.355) x 진행방향 가우시안 (sigma_a(r) = r theta_h / 2.355),
  각 커널은 sum g^2 = 1 로 정규화 -> E|Z|^2 = I0 (*) g^2 + N0  (정답 = I0 (*) g^2)
진행방향 해상도가 거리에 따라 달라지므로 sigma_a 를 0.1 핑 단위로 묶어서 열(bin) 묶음별로 컨볼루션한다.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve1d

from .config import NoiseConfig, SonarConfig

SIGMA_QUANT = 0.1  # 진행방향 sigma 묶음 단위 [핑]  (근사 오차와 속도의 절충)


def gaussian_kernel(sigma: float, power: int = 1) -> np.ndarray:
    """sum g^2 = 1 인 가우시안 g (power=1) 또는 g^2 (power=2, sum = 1)."""
    if sigma <= 1e-6:
        return np.array([1.0])
    t = np.arange(-int(np.ceil(4 * sigma)), int(np.ceil(4 * sigma)) + 1)
    k = np.exp(-0.5 * (t / sigma) ** 2)
    k = k / np.sqrt(np.sum(k ** 2))
    return k ** power


def along_sigma_pings(sonar: SonarConfig, ping_spacing_m: float) -> np.ndarray:
    """bin 별 진행방향 PSF 표준편차 [핑]. shape (n_bins,)."""
    r = (np.arange(sonar.n_bins) + 0.5) * sonar.bin_size_m
    fwhm = r * np.radians(sonar.horizontal_beamwidth_deg)
    return np.maximum(fwhm / 2.355 / ping_spacing_m, sonar.min_along_sigma_pings)


def apply_psf(img: np.ndarray, sig_along: np.ndarray, sig_range: float, power: int) -> np.ndarray:
    """거리 의존 PSF 적용. img (n_pings, n_bins) real 또는 complex. power=1: 복소장, power=2: 강도."""
    out = convolve1d(img.real, gaussian_kernel(sig_range, power), axis=1, mode="reflect")
    if np.iscomplexobj(img):
        out = out + 1j * convolve1d(img.imag, gaussian_kernel(sig_range, power), axis=1, mode="reflect")
    res = np.empty_like(out)
    q = np.round(sig_along / SIGMA_QUANT) * SIGMA_QUANT
    for s in np.unique(q):
        cols = np.flatnonzero(q == s)
        k = gaussian_kernel(float(s), power)
        blk = out[:, cols]
        if np.iscomplexobj(blk):
            res[:, cols] = convolve1d(blk.real, k, axis=0, mode="reflect") + 1j * convolve1d(blk.imag, k, axis=0, mode="reflect")
        else:
            res[:, cols] = convolve1d(blk, k, axis=0, mode="reflect")
    return res


def complex_normal(rng: np.random.Generator, shape) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def thermal_floor(sonar: SonarConfig, snr_far_db: float) -> np.ndarray:
    """TVG 후 열잡음 평균 강도 (모래 평균 = 1 기준). 수신기 잡음은 TVG 로 거리와 함께 증폭된다. shape (n_bins,)."""
    r = (np.arange(sonar.n_bins) + 0.5) * sonar.bin_size_m
    rm = r[-1]
    rel = (r / rm) ** (sonar.tvg_exponent / 10) * 10 ** (2 * sonar.absorption_db_per_m * (r - rm) / 10)
    return 10 ** (-snr_far_db / 10) * rel


def speckle_and_noise(I0: np.ndarray, sonar: SonarConfig, noise: NoiseConfig, ping_spacing_m: float,
                      rng: np.random.Generator):
    """기대 강도 I0 (한 현, n_pings x n_bins) -> (노이즈 강도, 정답, 첫 look 복소신호)."""
    sa = along_sigma_pings(sonar, ping_spacing_m)
    sr = sonar.range_resolution_m / 2.355 / sonar.bin_size_m
    clean = apply_psf(I0, sa, sr, power=2)
    n0 = thermal_floor(sonar, noise.snr_far_db) if noise.snr_far_db is not None else np.zeros(sonar.n_bins)
    if not noise.speckle:
        return clean + n0[None, :], clean, None
    looks, iq = [], None
    for _ in range(noise.n_looks):
        z = apply_psf(np.sqrt(np.maximum(I0, 0.0)) * complex_normal(rng, I0.shape), sa, sr, power=1)
        z = z + np.sqrt(n0)[None, :] * complex_normal(rng, I0.shape)
        iq = z if iq is None else iq
        looks.append(np.abs(z) ** 2)
    return np.mean(looks, axis=0), clean, iq


def add_interference(img: np.ndarray, noise: NoiseConfig, rng: np.random.Generator) -> np.ndarray:
    """다른 PRF 의 소나 간섭: 연속된 핑 구간에서 거리가 일정하게 미끄러지는 점선(사선 줄무늬). img 는 수정하지 않음."""
    if noise.interference_prob <= 0:
        return img
    out = img.copy()
    n, nb = img.shape
    active = rng.random(n) < noise.interference_prob
    # 활성 핑을 구간으로 늘림 (간섭원은 일정 시간 지속)
    active = np.convolve(active.astype(float), np.ones(15), mode="same") > 0
    r0, drift = rng.uniform(0, nb), rng.uniform(-6, 6)
    amp = 10 ** (noise.interference_db / 10)
    for i in np.flatnonzero(active):
        center = (r0 + drift * i) % nb
        for rep in range(3):  # 간섭 펄스열
            c = int((center + rep * nb / 3) % nb)
            w = slice(max(c - 2, 0), min(c + 3, nb))
            out[i, w] += amp * rng.exponential(size=w.stop - w.start)
    return out


def add_dropouts(img: np.ndarray, noise: NoiseConfig, rng: np.random.Generator) -> np.ndarray:
    """핑 누락: 해당 행을 0 으로 (실제 장비의 검은 줄)."""
    if noise.dropout_prob <= 0:
        return img
    out = img.copy()
    out[rng.random(img.shape[0]) < noise.dropout_prob] = 0.0
    return out
