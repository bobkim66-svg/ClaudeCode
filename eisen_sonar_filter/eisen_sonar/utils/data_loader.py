"""물리 기반(간이) 사이드스캔 소나(SSS) 영상 생성기 + 실제 영상 로더.

실측 노이즈 영상이 없을 때 sim-to-real 실험용 샘플을 만든다. 모델:

1. 해저 장면 (ground-range 좌표, 핑 x 지상거리):
   높이맵 h (모래 물결, 암석, 원통/상자 표적) + 재질 반사율 sigma0 (모래/진흙/암석/표적)
   + 감마 텍스처 (K-분포형 무거운 꼬리 클러터, 장면의 일부로 간주 -> 정답에 포함)
2. 기하/음향:
   소나 위치 (x_i, 0, altitude). 지상점 (x, y, h) 까지 경사거리 r = sqrt(y^2 + (H-h)^2)
   - 람베르트 산란: sigma = sigma0 cos^2(theta_inc),  theta_inc = 표면 법선과 입사방향 사이 각
   - 그림자: 고각 e(y) = atan2(h - H, y) 가 앞쪽(가까운) 점들의 최대 고각보다 작으면 가려짐
   - 수직 빔패턴 (가우시안, -3 dB 반폭), 왕복 확산손실 r^-4, 흡수 10^(-2 alpha r / 10)
   - 경사거리 bin 으로 에너지 누적 (layover/압축 자연 발생, r < H 는 수주(water column))
   - TVG 보상 (적용 지수를 실제와 다르게 해서 보상 오차 모사 가능) + 빔패턴 정규화
3. 수신 신호 (복소 IQ):
   Z = (sqrt(I0) * w) (*) g,  w ~ CN(0,1) 백색,  g = 거리/진행방향 PSF (sum g^2 = 1)
   -> E|Z|^2 = I0 (*) g^2  (정답 영상),  |Z|^2 은 상관된 레일리/지수 스페클
   + 열잡음 CN(0, N0(r)), N0 는 TVG 로 거리에 따라 증가
   L-look 은 독립 실현 L 개의 강도 평균 (감마 분포, ENL = L).

단순화(한계): 단일 산란, 수평 빔 이상적, 다중경로/수면반사/소나 자세 흔들림 없음,
흡수계수/빔폭 기본값은 400 kHz 대역 대략값 (확인 필요).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, convolve1d, gaussian_filter

# 재질 코드
SAND, MUD, ROCK, TARGET = 0, 1, 2, 3
DEFAULT_REFLECTIVITY = {SAND: 1.0, MUD: 0.35, ROCK: 2.5, TARGET: 6.0}


@dataclass
class TargetSpec:
    """해저 표적. kind = "cylinder" (축이 진행방향) | "box". 단위 m."""

    kind: str
    along_m: float  # 진행방향 중심
    ground_m: float  # 지상거리 중심
    length_m: float
    width_m: float
    height_m: float


@dataclass
class SonarSimConfig:
    """SSS 시뮬레이션 파라미터 (모든 길이 m, 각도 deg)."""

    n_pings: int = 256
    n_bins: int = 512
    altitude_m: float = 8.0
    max_range_m: float = 40.0
    ping_spacing_m: float = 0.08
    ground_oversample: int = 4  # 거리 bin 당 지상 표본 수
    absorption_db_per_m: float = 0.1  # 400 kHz 대 근사 (확인 필요)
    beam_tilt_deg: float = 15.0  # 빔 중심 내림각
    beam_halfwidth_deg: float = 25.0  # -3 dB 반폭 (수직)
    tvg_exponent_true: float = 40.0  # 실제 확산손실 (dB/decade 계수: 40 log r)
    tvg_exponent_applied: float = 40.0  # 수신기가 보상하는 값 (다르면 거리 의존 밝기 오차)
    normalize_beam: bool = True  # 평탄 해저 응답으로 나누는 경험적 빔패턴 보정
    psf_sigma_range_bins: float = 1.0
    psf_sigma_pings: float = 0.8
    n_looks: int = 1
    snr_far_db: float = 15.0  # 최대 거리에서 모래 평균강도 대비 열잡음
    # 장면
    ripple_amp_m: float = 0.04
    ripple_wavelength_m: float = 0.9
    texture_shape: Optional[float] = 6.0  # 감마 텍스처 형상 nu (None 이면 없음), 동질 패치 제외
    texture_corr_m: float = 0.6
    n_rocks: int = 6
    targets: List[TargetSpec] = field(default_factory=lambda: [
        TargetSpec("cylinder", 6.0, 18.0, 2.5, 0.6, 0.6),
        TargetSpec("box", 13.0, 28.0, 1.5, 1.5, 0.9),
    ])
    mud_patch: Tuple[float, float, float, float] = (14.0, 20.0, 8.0, 16.0)  # along0, along1, ground0, ground1
    homogeneous_patch: Tuple[float, float, float, float] = (1.0, 5.0, 26.0, 36.0)  # 평탄 모래 (ENL 측정용)
    reflectivity: Dict[int, float] = field(default_factory=lambda: dict(DEFAULT_REFLECTIVITY))


@dataclass
class SonarSample:
    """생성 결과. 영상 shape 모두 (n_pings, n_bins), 강도는 선형 스케일."""

    intensity: np.ndarray  # 노이즈 영상 (L-look 평균 강도)
    iq: np.ndarray  # 첫 look 의 복소 신호 (complex)
    clean: np.ndarray  # 정답: 스페클/열잡음 없는 기대 강도
    masks: Dict[str, np.ndarray]  # homogeneous, target, shadow, background, seafloor
    range_m: np.ndarray  # (n_bins,) 경사거리
    along_m: np.ndarray  # (n_pings,)
    config: SonarSimConfig


def _scene(cfg: SonarSimConfig, rng: np.random.Generator, flat: bool = False):
    """지상 좌표 장면. 출력 shape (n_pings, n_ground): along, ground, height, sigma0, material, hom."""
    dr = cfg.max_range_m / cfg.n_bins
    ymax = np.sqrt(max(cfg.max_range_m ** 2 - cfg.altitude_m ** 2, 1.0))
    yg = np.arange(1, int(ymax / dr * cfg.ground_oversample) + 1) * (dr / cfg.ground_oversample)
    xa = np.arange(cfg.n_pings) * cfg.ping_spacing_m
    X, Y = np.meshgrid(xa, yg, indexing="ij")
    h = np.zeros_like(X)
    mat = np.full(X.shape, SAND, dtype=int)
    hom = np.zeros(X.shape, dtype=bool)
    if flat:
        return X, Y, h, np.full(X.shape, cfg.reflectivity[SAND]), mat, hom

    a0, a1, g0, g1 = cfg.homogeneous_patch
    hom = (X >= a0) & (X <= a1) & (Y >= g0) & (Y <= g1)

    # 모래 물결 (진행방향과 비스듬한 마루선) + 위상 흔들림
    k = 2 * np.pi / cfg.ripple_wavelength_m
    phase = gaussian_filter(rng.standard_normal(X.shape), 20) * 30
    h += cfg.ripple_amp_m * np.sin(k * (0.8 * Y + 0.6 * X) + phase)
    # 진흙 패치 (평탄, 낮은 반사)
    m0, m1, mg0, mg1 = cfg.mud_patch
    mud = (X >= m0) & (X <= m1) & (Y >= mg0) & (Y <= mg1)
    h[mud] *= 0.2
    mat[mud] = MUD
    # 암석 (가우시안 언덕)
    for _ in range(cfg.n_rocks):
        cx, cy = rng.uniform(xa[0], xa[-1]), rng.uniform(8.0, yg[-1] - 2.0)
        rad, ht = rng.uniform(0.3, 0.8), rng.uniform(0.2, 0.6)
        bump = ht * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * rad ** 2))
        bump[hom] = 0.0
        h += bump
        mat[bump > 0.3 * ht] = ROCK
    # 표적
    for t in cfg.targets:
        inx = np.abs(X - t.along_m) <= t.length_m / 2
        if t.kind == "cylinder":
            rr = t.width_m / 2
            dy = Y - t.ground_m
            prof = np.sqrt(np.maximum(rr ** 2 - dy ** 2, 0.0)) + (t.height_m - rr)
            sel = inx & (np.abs(dy) <= rr)
        elif t.kind == "box":
            sel = inx & (np.abs(Y - t.ground_m) <= t.width_m / 2)
            prof = np.full(X.shape, t.height_m)
        else:
            raise ValueError(f"알 수 없는 표적 종류: {t.kind}")
        h = np.where(sel, np.maximum(h, prof), h)
        mat[sel] = TARGET
    h[hom] = 0.0  # 동질 패치는 완전 평탄 모래

    sigma0 = np.vectorize(cfg.reflectivity.get)(mat).astype(float)
    if cfg.texture_shape is not None:
        corr = cfg.texture_corr_m / (dr / cfg.ground_oversample)
        gfield = gaussian_filter(rng.standard_normal(X.shape), (cfg.texture_corr_m / cfg.ping_spacing_m, corr))
        gfield /= gfield.std()
        # 가우시안 -> 감마(nu, 1/nu) 분위수 변환 (평균 1, 공간 상관 유지)
        from scipy.stats import gamma, norm

        tex = gamma.ppf(norm.cdf(gfield), a=cfg.texture_shape, scale=1.0 / cfg.texture_shape)
        tex[hom | (mat == TARGET)] = 1.0
        sigma0 *= tex
    return X, Y, h, sigma0, mat, hom


def _render_mean_intensity(cfg: SonarSimConfig, X, Y, h, sigma0, labels: Dict[str, np.ndarray]):
    """지상 장면 -> 경사거리 영상의 기대 강도 I0 (n_pings, n_bins) 와 라벨별 에너지 비율."""
    H = cfg.altitude_m
    dr = cfg.max_range_m / cfg.n_bins
    dy = Y[0, 1] - Y[0, 0]
    # 표면 법선과 입사방향 (소나 쪽을 향하는 단위벡터)
    hx = np.gradient(h, cfg.ping_spacing_m, axis=0)
    hy = np.gradient(h, dy, axis=1)
    nrm = np.sqrt(hx ** 2 + hy ** 2 + 1.0)
    r = np.sqrt(Y ** 2 + (H - h) ** 2)
    dirx, diry, dirz = 0.0, -Y / r, (H - h) / r
    cos_inc = (-hx * dirx - hy * diry + dirz) / nrm
    sigma = sigma0 * np.clip(cos_inc, 0.0, None) ** 2
    # 그림자: 고각 e 가 앞쪽 점들의 누적 최대보다 작으면 가려짐
    e = np.arctan2(h - H, Y)
    visible = e >= np.maximum.accumulate(e, axis=1) - 1e-12
    # 빔패턴 (왕복), 확산, 흡수
    dep = np.degrees(np.arctan2(H - h, Y))
    beam = (10 ** (-0.3 * ((dep - cfg.beam_tilt_deg) / cfg.beam_halfwidth_deg) ** 2)) ** 2  # 편도 -3 dB @ 반폭, 왕복 제곱
    loss = r ** (-cfg.tvg_exponent_true / 10.0) * 10 ** (-2 * cfg.absorption_db_per_m * r / 10)
    energy = sigma * beam * loss * visible * dy * cfg.ping_spacing_m

    bins = np.minimum((r / dr).astype(int), cfg.n_bins - 1)
    flat_idx = (np.arange(cfg.n_pings)[:, None] * cfg.n_bins + bins).ravel()
    size = cfg.n_pings * cfg.n_bins

    def acc(w):
        return np.bincount(flat_idx, weights=np.broadcast_to(w, X.shape).ravel(), minlength=size).reshape(
            cfg.n_pings, cfg.n_bins
        )

    I0 = acc(energy)
    r_bin = (np.arange(cfg.n_bins) + 0.5) * dr
    tvg = r_bin ** (cfg.tvg_exponent_applied / 10.0) * 10 ** (2 * cfg.absorption_db_per_m * r_bin / 10)
    I0 = I0 * tvg[None, :]
    frac = {}
    count = acc(np.ones_like(energy))
    for name, lab in labels.items():
        frac[name] = acc(lab.astype(float)) / np.maximum(count, 1.0)
    frac["_count"] = count
    return I0, frac


def _psf_kernels(cfg: SonarSimConfig):
    def g(sig):
        if sig <= 0:
            return np.array([1.0])
        t = np.arange(-int(np.ceil(4 * sig)), int(np.ceil(4 * sig)) + 1)
        k = np.exp(-0.5 * (t / sig) ** 2)
        return k / np.sqrt(np.sum(k ** 2))  # sum g^2 = 1 -> 스페클 평균 강도 보존

    return g(cfg.psf_sigma_pings), g(cfg.psf_sigma_range_bins)


def _apply_sep(img: np.ndarray, ka: np.ndarray, kr: np.ndarray) -> np.ndarray:
    return convolve1d(convolve1d(img, ka, axis=0, mode="reflect"), kr, axis=1, mode="reflect")


def make_synthetic_sonar(cfg: Optional[SonarSimConfig] = None, seed: int = 0) -> SonarSample:
    """합성 SSS 영상 하나를 만든다 (재현 가능: 같은 seed -> 같은 영상).

    Returns SonarSample. 모든 영상 shape (cfg.n_pings, cfg.n_bins).
    """
    cfg = cfg or SonarSimConfig()
    if cfg.n_looks < 1:
        raise ValueError("n_looks >= 1")
    if cfg.altitude_m >= cfg.max_range_m:
        raise ValueError("altitude_m < max_range_m 이어야 한다")
    rng = np.random.default_rng(seed)
    X, Y, h, sigma0, mat, hom = _scene(cfg, rng)

    # 그림자 라벨: 지상점이 가려졌는지 (표적/암석 뒤)
    H = cfg.altitude_m
    e = np.arctan2(h - H, Y)
    shadowed = e < np.maximum.accumulate(e, axis=1) - 1e-12
    labels = {"hom": hom, "target": mat == TARGET, "shadow": shadowed, "sand": mat == SAND}
    I0, frac = _render_mean_intensity(cfg, X, Y, h, sigma0, labels)

    if cfg.normalize_beam:
        # 평탄 모래 해저의 기대 응답으로 나눔 (실제 처리 체인의 경험적 빔패턴/TVG 보정에 해당)
        Xf, Yf, hf, sf, _, _ = _scene(cfg, rng, flat=True)
        ref, _ = _render_mean_intensity(cfg, Xf, Yf, hf, sf, {})
        curve = ref.mean(axis=0)
        good = curve > 1e-6 * curve.max()
        gain = np.where(good, 1.0 / np.where(good, curve, 1.0), 0.0)
        I0 = I0 * gain[None, :]
    else:
        I0 = I0 / np.median(I0[I0 > 0])

    ka, kr = _psf_kernels(cfg)
    g2a, g2r = ka ** 2, kr ** 2  # sum = 1
    clean = convolve1d(convolve1d(I0, g2a, axis=0, mode="reflect"), g2r, axis=1, mode="reflect")

    dr = cfg.max_range_m / cfg.n_bins
    r_bin = (np.arange(cfg.n_bins) + 0.5) * dr
    tvg_rel = (r_bin / r_bin[-1]) ** (cfg.tvg_exponent_applied / 10.0) * 10 ** (
        2 * cfg.absorption_db_per_m * (r_bin - r_bin[-1]) / 10
    )
    sand_far = np.median(clean[:, -cfg.n_bins // 10:])
    n0 = sand_far / 10 ** (cfg.snr_far_db / 10) * tvg_rel  # (n_bins,)

    def cn(shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)

    looks = []
    iq0 = None
    for _ in range(cfg.n_looks):
        w = np.sqrt(I0) * cn(I0.shape)
        z = _apply_sep(w.real, ka, kr) + 1j * _apply_sep(w.imag, ka, kr)
        z = z + np.sqrt(n0)[None, :] * cn(I0.shape)
        if iq0 is None:
            iq0 = z
        looks.append(np.abs(z) ** 2)
    intensity = np.mean(looks, axis=0)

    # 마스크 (영상 좌표)
    seafloor = (r_bin > H * 1.05)[None, :] & np.ones((cfg.n_pings, 1), bool)
    margin = int(np.ceil(3 * max(cfg.psf_sigma_range_bins, cfg.psf_sigma_pings))) + 1
    homm = binary_erosion((frac["hom"] > 0.999) & (frac["shadow"] < 1e-3), iterations=margin) & seafloor
    target = (frac["target"] > 0.5) & seafloor
    shadow = binary_erosion((frac["shadow"] > 0.95) & seafloor, iterations=1)
    near = binary_dilation(target | shadow, iterations=12)
    background = near & ~binary_dilation(target | shadow, iterations=4) & (frac["sand"] > 0.95) & seafloor
    masks = {"homogeneous": homm, "target": target, "shadow": shadow, "background": background, "seafloor": seafloor}
    return SonarSample(intensity, iq0, clean, masks, r_bin, np.arange(cfg.n_pings) * cfg.ping_spacing_m, cfg)


def randomized_config(rng: np.random.Generator, base: Optional[SonarSimConfig] = None) -> SonarSimConfig:
    """도메인 랜덤화: sim-to-real 용으로 물리 파라미터와 표적 배치를 무작위로 바꾼 설정."""
    base = base or SonarSimConfig()
    along_max = base.n_pings * base.ping_spacing_m
    targets = [
        TargetSpec(
            str(rng.choice(["cylinder", "box"])),
            float(rng.uniform(7.5, along_max - 1.5)),  # 동질 패치(진행방향 1~5 m)와 겹치지 않게
            float(rng.uniform(12.0, 34.0)),
            float(rng.uniform(1.0, 3.0)),
            float(rng.uniform(0.4, 1.5)),
            float(rng.uniform(0.3, 1.0)),
        )
        for _ in range(int(rng.integers(1, 4)))
    ]
    return replace(
        base,
        altitude_m=float(rng.uniform(5.0, 12.0)),
        beam_tilt_deg=float(rng.uniform(10.0, 25.0)),
        tvg_exponent_applied=float(rng.uniform(34.0, 42.0)),
        snr_far_db=float(rng.uniform(5.0, 25.0)),
        texture_shape=float(rng.uniform(2.0, 20.0)),
        psf_sigma_range_bins=float(rng.uniform(0.6, 1.6)),
        n_rocks=int(rng.integers(0, 12)),
        targets=targets,
    )


def load_sonar_image(path: str | Path, amplitude: bool = True) -> np.ndarray:
    """실측 영상 로더 (.npy 또는 png/jpg/tif). 출력: 선형 강도 (H, W) float.

    amplitude=True 면 파일 값을 진폭으로 보고 제곱한다. dB 로 저장된 영상은 먼저 선형으로 바꿔서 넣을 것.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        img = np.load(path).astype(float)
    else:
        import matplotlib.image as mpimg

        img = mpimg.imread(path).astype(float)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)
    if img.ndim != 2:
        raise ValueError(f"2D 영상이 아니다: {img.shape}")
    img = np.abs(img)
    return img ** 2 if amplitude else img
