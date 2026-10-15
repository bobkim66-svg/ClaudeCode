"""시뮬레이션 진입점: SceneConfig -> SonarRecord (양현 워터폴 영상 + 정답 + 라벨)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional

import numpy as np

from .acoustics import render_side
from .config import SceneConfig
from .noise import add_dropouts, add_interference, speckle_and_noise
from .seabed import World, build_world
from .trajectory import Navigation, make_navigation

SIDES = (-1, 1)  # 좌현, 우현


@dataclass
class SonarRecord:
    """시뮬레이션 결과. 영상 shape (n_pings, 2 * n_bins), 열 = [좌현 먼->가까운 | 우현 가까운->먼]."""

    intensity: np.ndarray  # 노이즈 영상 (선형 강도, 모래 평균 ~ 1)
    clean: np.ndarray  # 정답: 스페클/잡음/아티팩트 없는 기대 강도 (같은 PSF)
    iq: Optional[np.ndarray]  # 첫 look 복소 신호 (complex64) 또는 None
    labels: Dict[str, np.ndarray]  # object_id, object_shadow_id, shadow, seafloor
    across_m: np.ndarray  # (2 n_bins,) 부호 있는 경사거리, 좌현 < 0
    along_m: np.ndarray  # (n_pings,) 항적 누적 거리
    nav: Navigation
    world: World
    config: SceneConfig


def _flat_world(bounds) -> World:
    """평탄 모래 (반사율 1) 월드: 빔패턴/TVG 보정 곡선 계산용. 2x2 격자 + 최근접 외삽 = 무한 평면."""
    xmin, xmax, ymin, ymax = bounds
    z = np.zeros((2, 2))
    return World(xmin, ymin, max(xmax - xmin, ymax - ymin), z, np.ones((2, 2)), z.copy(),
                 np.zeros((2, 2), np.int16), np.zeros((2, 2), np.int32))


def _correction_gain(curve: np.ndarray, enable: bool) -> np.ndarray:
    """평탄 해저 기대 곡선 -> 보정 이득. 첫 해저 반사 전 (수주) 은 첫 유효 bin 의 이득을 그대로 씀."""
    valid = curve > 1e-6 * curve.max()
    if not valid.any():
        raise RuntimeError("평탄 해저 기준 곡선이 0: 고도/거리 설정 확인")
    if not enable:
        return np.full_like(curve, 1.0 / np.median(curve[valid]))
    i0 = int(np.argmax(valid))
    c = curve.copy()
    c[:i0] = curve[i0]
    c[~valid & (np.arange(len(c)) >= i0)] = np.inf  # 범위 밖 (거의 없음)
    return 1.0 / c


def simulate(cfg: SceneConfig, keep_iq: bool = False) -> SonarRecord:
    """장면 설정으로 SSS 영상 한 장(항적 하나)을 만든다. 같은 cfg (seed 포함) -> 같은 결과."""
    cfg.validate()
    rng = np.random.default_rng(cfg.seed)
    s, t, nz = cfg.sonar, cfg.trajectory, cfg.noise
    nav = make_navigation(t, s, rng)
    margin = s.max_range_m + 2.0
    bounds = (nav.x.min() - margin, nav.x.max() + margin, nav.y.min() - margin, nav.y.max() + margin)
    world = build_world(cfg.seabed, cfg.objects, bounds, rng)

    # 평탄 해저 + 운동 없음 = 기대 응답 곡선 (실제 처리의 경험적 빔패턴 보정에 해당)
    nominal = replace(nav, z=np.full(nav.n, t.altitude_m), roll=np.zeros(nav.n), pitch=np.zeros(nav.n))
    flat = _flat_world(bounds)
    spacing = t.speed_mps / s.ping_rate_hz
    r_bin = (np.arange(s.n_bins) + 0.5) * s.bin_size_m
    surf = None if nz.surface_return_db is None else 10 ** (nz.surface_return_db / 10)
    mp = None if nz.multipath_db is None else 10 ** (nz.multipath_db / 10)

    halves = {}
    for side in SIDES:
        ref = render_side(flat, nominal, s, side, t.water_depth_m, cfg.seabed.object_specular_width_deg)
        gain = _correction_gain(ref.intensity.mean(axis=0), s.beam_correction)
        rend = render_side(world, nav, s, side, t.water_depth_m, cfg.seabed.object_specular_width_deg, surf, mp)
        I0 = rend.intensity * gain[None, :]
        if nz.volume_scatter_db is not None:
            I0 += 10 ** (nz.volume_scatter_db / 10)  # 상수 근사 (실제는 TVG 법칙/수층 구조에 따라 변함)
        I0 = _add_fish(I0, nz, s, rng)
        noisy, clean, iq = speckle_and_noise(I0, s, nz, spacing, rng)
        halves[side] = (noisy, clean, iq, rend.labels)

    def join(get):
        a, b = get(halves[-1]), get(halves[1])
        return np.concatenate([a[:, ::-1], b], axis=1)

    intensity = join(lambda h: h[0])
    clean = join(lambda h: h[1])
    iq = join(lambda h: h[2]).astype(np.complex64) if keep_iq and halves[1][2] is not None else None
    labels = {k: join(lambda h, k=k: h[3][k]) for k in halves[1][3]}
    intensity = add_dropouts(add_interference(intensity, nz, rng), nz, rng)
    across = np.concatenate([-r_bin[::-1], r_bin])
    along = t.speed_mps * nav.t
    assert intensity.shape == clean.shape == (nav.n, 2 * s.n_bins)
    assert np.all(np.isfinite(intensity)) and np.all(intensity >= 0)
    return SonarRecord(intensity, clean, iq, labels, across, along, nav, world, cfg)


def _add_fish(I0: np.ndarray, nz, s, rng: np.random.Generator) -> np.ndarray:
    """수주 점 산란체 무리. 위치/밝기 무작위 (진행방향 퍼짐은 이후 PSF 가 만든다)."""
    if nz.n_fish <= 0:
        return I0
    out = I0.copy()
    n, nb = I0.shape
    for _ in range(nz.n_fish):
        i_c, b_c = rng.integers(0, n), rng.integers(int(0.05 * nb), int(0.7 * nb))
        m = int(rng.integers(3, 25))
        ii = np.clip(i_c + rng.normal(0, 4, m).astype(int), 0, n - 1)
        bb = np.clip(b_c + rng.normal(0, 6, m).astype(int), 0, nb - 1)
        np.add.at(out, (ii, bb), 10 ** (rng.uniform(nz.fish_level_db - 15, nz.fish_level_db, m) / 10))
    return out
