"""가상 해저 공간 (2.5D 격자): 높이, 반사율, 재질, 물체 id.

격자 W[j, i] 는 월드 점 (x0 + i*d, y0 + j*d) 에 해당한다 (행 = y, 열 = x).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm

from .config import ObjectSpec, SeabedConfig

# 재질 코드 (라벨 맵에 그대로 쓰임)
SAND, MUD, GRAVEL, ROCK, OBJECT = 0, 1, 2, 3, 4
MATERIAL_NAMES = {SAND: "sand", MUD: "mud", GRAVEL: "gravel", ROCK: "rock", OBJECT: "object"}


@dataclass
class World:
    """해저 격자. 모든 배열 shape (ny, nx)."""

    x0: float
    y0: float
    res: float
    height: np.ndarray  # [m], 해저 기준면 위
    reflectivity: np.ndarray  # 확산 반사율 (선형, 상대값)
    specular: np.ndarray  # 정반사 강도 (물체 표면만 > 0)
    material: np.ndarray  # int 재질 코드
    object_id: np.ndarray  # int, 0 = 없음, k = objects[k-1]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.height.shape

    def to_index(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """월드 좌표 -> 연속 격자 인덱스 (행, 열)."""
        return (np.asarray(y) - self.y0) / self.res, (np.asarray(x) - self.x0) / self.res


def _smooth_field(rng: np.random.Generator, shape, corr_cells: float) -> np.ndarray:
    """평균 0, 표준편차 1 의 공간 상관 가우시안 장 (가우시안 저역통과, FFT 로 O(N log N)).

    공간영역 gaussian_filter(sigma) 와 같은 전달함수 exp(-2 pi^2 sigma^2 k^2) 를 주파수영역에서 곱한다 (주기 경계).
    """
    sig = max(corr_cells, 0.5)
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.rfftfreq(shape[1])[None, :]
    H = np.exp(-2 * np.pi ** 2 * sig ** 2 * (ky ** 2 + kx ** 2))
    f = np.fft.irfft2(np.fft.rfft2(rng.standard_normal(shape)) * H, s=shape)
    return f / max(f.std(), 1e-12)


def _gaussian_to_gamma(g: np.ndarray, shape_nu: float) -> np.ndarray:
    """표준정규 값 -> 감마(nu, 1/nu) 분위수 변환 (평균 1). ppf 를 4096점 표로 보간 (정확 계산보다 수십 배 빠름)."""
    z = np.linspace(-6, 6, 4096)
    table = gamma_dist.ppf(norm.cdf(z), a=shape_nu, scale=1.0 / shape_nu)
    return np.interp(g, z, table)


def power_law_surface(rng: np.random.Generator, shape, res: float, rms: float, beta: float, cutoff_m: float) -> np.ndarray:
    """멱법칙 스펙트럼 P(k) ~ k^-beta 의 랜덤 지형 (FFT 합성). 출력 shape, RMS = rms [m]."""
    if rms <= 0:
        return np.zeros(shape)
    ky = np.fft.fftfreq(shape[0], d=res)
    kx = np.fft.rfftfreq(shape[1], d=res)
    k = np.hypot(ky[:, None], kx[None, :])
    k_min = 1.0 / cutoff_m
    amp = np.where(k >= k_min, np.maximum(k, k_min) ** (-beta / 2.0), 0.0)
    spec = amp * (rng.standard_normal(amp.shape) + 1j * rng.standard_normal(amp.shape))
    surf = np.fft.irfft2(spec, s=shape)
    return surf * (rms / max(surf.std(), 1e-15))


def _rasterize_object(o: ObjectSpec, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """물체 높이 프로파일 [m] (해저 기준면 위) 과 덮는 영역 bool. X, Y 는 부분 격자."""
    c, s = np.cos(np.radians(o.heading_deg)), np.sin(np.radians(o.heading_deg))
    u = (X - o.x) * c + (Y - o.y) * s  # 물체 축 방향
    v = -(X - o.x) * s + (Y - o.y) * c  # 축에 수직
    if o.kind == "cylinder":
        r = o.width / 2
        prof = np.sqrt(np.maximum(r * r - v * v, 0.0)) + r
        sel = (np.abs(u) <= o.length / 2) & (np.abs(v) < r)
    elif o.kind == "box":
        prof = np.full(X.shape, o.height)
        sel = (np.abs(u) <= o.length / 2) & (np.abs(v) <= o.width / 2)
    elif o.kind == "sphere":
        r = o.width / 2
        d2 = u * u + v * v
        prof = np.sqrt(np.maximum(r * r - d2, 0.0)) + r
        sel = d2 < r * r
    else:
        raise ValueError(o.kind)
    prof = prof - o.burial_m
    return prof, sel & (prof > 0)


def build_world(cfg: SeabedConfig, objects: List[ObjectSpec], bounds: Tuple[float, float, float, float],
                rng: np.random.Generator) -> World:
    """bounds = (xmin, xmax, ymin, ymax) [m] 를 덮는 해저를 만든다."""
    xmin, xmax, ymin, ymax = bounds
    d = cfg.grid_resolution_m
    nx = int(np.ceil((xmax - xmin) / d)) + 1
    ny = int(np.ceil((ymax - ymin) / d)) + 1
    if nx * ny > 40_000_000:
        raise MemoryError(f"해저 격자 {ny}x{nx} 가 너무 크다: grid_resolution_m 을 키울 것")
    xs = xmin + np.arange(nx) * d
    ys = ymin + np.arange(ny) * d
    X, Y = np.meshgrid(xs, ys)

    # 1) 지형: 멱법칙 거칠기 + 국지적 물결 + 지역 경사
    h = power_law_surface(rng, (ny, nx), d, cfg.roughness_rms_m, cfg.roughness_beta, cfg.roughness_cutoff_m)
    if cfg.ripple_amp_m > 0 and cfg.ripple_coverage > 0:
        th = np.radians(cfg.ripple_direction_deg)
        phase = 2 * np.pi / cfg.ripple_wavelength_m * (X * np.cos(th) + Y * np.sin(th))
        # 마루선 휘어짐: 큰 규모 위상 왜곡 (수 rad, 상관길이 ~6 m) + 작은 규모 끊김/분기.
        # 주의: 방향각 자체를 장(field)으로 흔들면 절대좌표가 곱해져 동심원 무늬가 생긴다 (이전 버그).
        phase += 6.0 * _smooth_field(rng, (ny, nx), 6.0 / d) + 1.2 * _smooth_field(rng, (ny, nx), 1.5 / d)
        env_field = _smooth_field(rng, (ny, nx), 5.0 / d)
        thr = norm.ppf(1 - cfg.ripple_coverage)
        envelope = np.clip((env_field - thr) / 0.5, 0.0, 1.0)  # 물결 영역 경계를 부드럽게
        # 비대칭 물결 (실제 물결은 사인파보다 마루가 뾰족): |sin|^1.5 형태
        ripple = np.sign(np.sin(phase)) * np.abs(np.sin(phase)) ** 1.5
        h += cfg.ripple_amp_m * envelope * ripple
    sx, sy = np.tan(np.radians(cfg.regional_slope_deg[0])), np.tan(np.radians(cfg.regional_slope_deg[1]))
    h += sx * (X - X.mean()) + sy * (Y - Y.mean())

    # 2) 퇴적물: 매끄러운 장의 분위수로 나눔
    sed = _smooth_field(rng, (ny, nx), cfg.sediment_corr_m / d)
    q_mud = np.quantile(sed, cfg.mud_fraction) if cfg.mud_fraction > 0 else -np.inf
    q_grav = np.quantile(sed, 1 - cfg.gravel_fraction) if cfg.gravel_fraction > 0 else np.inf
    material = np.full((ny, nx), SAND, dtype=np.int16)
    material[sed <= q_mud] = MUD
    material[sed >= q_grav] = GRAVEL
    # 반사율은 경계에서 부드럽게 (실제 퇴적물 경계는 점진적)
    w_mud = np.clip((q_mud + 0.3 - sed) / 0.6, 0, 1) if np.isfinite(q_mud) else 0.0
    w_grav = np.clip((sed - q_grav + 0.3) / 0.6, 0, 1) if np.isfinite(q_grav) else 0.0
    refl = cfg.sand_reflectivity * (1 - w_mud - w_grav) + cfg.mud_reflectivity * w_mud + cfg.gravel_reflectivity * w_grav
    refl = np.broadcast_to(refl, (ny, nx)).astype(float)  # 퇴적물을 끈 경우 스칼라가 될 수 있음 -> 배열로
    h *= 1.0 - 0.6 * w_mud  # 진흙은 매끄러움 (가중치로 부드럽게: 경계에서 높이 계단이 생기면 가짜 선이 보임)
    if cfg.texture_shape is not None:
        g = _smooth_field(rng, (ny, nx), cfg.texture_corr_m / d)
        refl = refl * _gaussian_to_gamma(g, cfg.texture_shape)

    # 3) 암석
    for _ in range(cfg.n_rocks):
        cx, cy = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)
        rad = rng.uniform(*cfg.rock_radius_m)
        n_lobes = int(rng.integers(1, 4))  # 여러 덩어리로 된 불규칙 암석
        for _ in range(n_lobes):
            ox, oy = rng.normal(0, rad * 0.5, 2)
            rr = rad * rng.uniform(0.5, 1.0)
            ht = rr * rng.uniform(0.5, 1.2)
            i0, i1 = np.searchsorted(xs, [cx + ox - 3 * rr, cx + ox + 3 * rr])
            j0, j1 = np.searchsorted(ys, [cy + oy - 3 * rr, cy + oy + 3 * rr])
            if i1 <= i0 or j1 <= j0:
                continue
            sl = (slice(j0, j1), slice(i0, i1))  # 국소 창만 계산
            bump = ht * np.exp(-((X[sl] - cx - ox) ** 2 + (Y[sl] - cy - oy) ** 2) / (2 * (0.6 * rr) ** 2))
            h[sl] += bump
            rock = bump > 0.15 * ht
            material[sl][rock] = ROCK
            refl[sl][rock] *= cfg.rock_reflectivity / max(cfg.sand_reflectivity, 1e-9)

    # 4) 검증용 평탄 패치
    if cfg.flat_patch is not None:
        x0, x1, y0, y1 = cfg.flat_patch
        fp = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        h[fp] = 0.0
        material[fp] = SAND
        refl[fp] = cfg.sand_reflectivity

    # 5) 물체 (지형 위에 얹음: 바닥 높이 = 물체 중심의 지형 높이)
    spec = np.zeros((ny, nx))
    oid = np.zeros((ny, nx), dtype=np.int32)
    for k, o in enumerate(objects, start=1):
        reach = max(o.length, o.width) / 2 + 2 * d
        i0, i1 = np.searchsorted(xs, [o.x - reach, o.x + reach])
        j0, j1 = np.searchsorted(ys, [o.y - reach, o.y + reach])
        if i1 <= i0 or j1 <= j0:
            continue  # 해저 영역 밖
        sl = (slice(j0, j1), slice(i0, i1))
        base = float(np.median(h[sl]))
        prof, sel = _rasterize_object(o, X[sl], Y[sl])
        top = base + prof
        sub_h = h[sl]
        put = sel & (top > sub_h)
        sub_h[put] = top[put]
        material[sl][put] = OBJECT
        refl[sl][put] = o.reflectivity
        spec[sl][put] = o.specular
        oid[sl][put] = k

    return World(xmin, ymin, d, h, refl, spec, material, oid)
