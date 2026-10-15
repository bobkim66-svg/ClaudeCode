"""음향 렌더링: 해저 월드 + 항법 -> 한쪽 현(side)의 기대 강도 (스페클 없음, TVG 적용 후).

한 핑, 한 현에서 빔 평면 위 지상 표본 g_k (수평거리) 마다:
  위치   P = S + g s_side + a f,  a = (z_s - h) tan(pitch)  (기울어진 빔 평면, 1회 고정점 반복)
  거리   r = |P - S|
  산란   sigma = refl cos^2(theta_inc) + spec exp(-(theta_inc/w)^2)   (람베르트 + 물체 정반사)
  그림자 e_k = atan2(h_k - z_s, rho_k) 가 앞쪽 누적 최대보다 작으면 가려짐
  빔     G(delta) = max(10^(-0.3 ((delta - c)/hw)^2), sidelobe),  c = tilt +- roll (현에 따라)
  손실   r^-4 10^(-2 alpha r / 10)  ->  TVG  r^(k/10) 10^(2 alpha r/10)
  에너지 sigma G^2 loss visible dg  를 거리 bin floor(r / dr) 에 누적
수면 반사 경로 (다중경로): 수면에 대한 거울상 소나 (z_m = 2D - z_s). 한 번 반사 경로의 등가 거리 (r + r_m)/2.
수면 산란: 수면 위 점 (거리 r_s = sqrt(g^2 + (D - z_s)^2)), cos(입사) = (D - z_s)/r_s.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import map_coordinates

from .config import SonarConfig
from .seabed import World
from .trajectory import Navigation


@dataclass
class SideRender:
    """한 현의 렌더 결과. 배열 shape (n_pings, n_bins)."""

    intensity: np.ndarray  # 기대 강도 (TVG 후, 물리 단위의 상대값)
    labels: Dict[str, np.ndarray]


def _beam_gain(delta: np.ndarray, center: np.ndarray, sonar: SonarConfig) -> np.ndarray:
    """편도 빔 이득 (선형 강도). delta, center [rad]."""
    hw = np.radians(sonar.vertical_halfwidth_deg)
    g = 10 ** (-0.3 * ((delta - center) / hw) ** 2)
    return np.maximum(g, 10 ** (sonar.sidelobe_db / 10))


def tvg_curve(sonar: SonarConfig) -> np.ndarray:
    """bin 중심 거리의 TVG 이득. shape (n_bins,)."""
    r = (np.arange(sonar.n_bins) + 0.5) * sonar.bin_size_m
    return r ** (sonar.tvg_exponent / 10) * 10 ** (2 * sonar.absorption_db_per_m * r / 10)


def _sample(world: World, field: np.ndarray, rows, cols, order: int) -> np.ndarray:
    return map_coordinates(field, [rows.ravel(), cols.ravel()], order=order, mode="nearest").reshape(rows.shape)


def render_side(world: World, nav: Navigation, sonar: SonarConfig, side: int, water_depth: float,
                specular_width_deg: float, surface_ratio: Optional[float] = None,
                multipath_ratio: Optional[float] = None) -> SideRender:
    """side = +1 (우현) / -1 (좌현).

    surface_ratio: 수면 산란계수 / 모래 반사율 (선형). None 이면 수면 산란 없음.
    multipath_ratio: 수면 반사 경로 이득 (선형). None 이면 없음.
    """
    if side not in (1, -1):
        raise ValueError("side 는 +1 또는 -1")
    nb, dr = sonar.n_bins, sonar.bin_size_m
    dg = dr / sonar.ground_oversample
    g = (np.arange(int(np.ceil(sonar.max_range_m / dg))) + 0.5) * dg  # (K,)
    hx, hy = np.gradient(world.height, world.res)  # 행(y), 열(x) 순서
    hy, hx = hx, hy
    w_spec = np.radians(specular_width_deg)
    tilt = np.radians(sonar.tilt_deg)

    out = np.zeros((nav.n, nb))
    acc_names = ["count", "shadow", "obj_e", "obj_id_e", "objshadow", "objshadow_id", "ground"]
    lab = {k: np.zeros((nav.n, nb)) for k in acc_names}

    fx_all, fy_all = nav.forward()
    sx_all, sy_all = nav.starboard()
    for c0 in range(0, nav.n, sonar.render_chunk_pings):
        sl = slice(c0, min(c0 + sonar.render_chunk_pings, nav.n))
        n = sl.stop - sl.start
        Sx, Sy, Sz = nav.x[sl, None], nav.y[sl, None], nav.z[sl, None]
        fx, fy = fx_all[sl, None], fy_all[sl, None]
        ux, uy = side * sx_all[sl, None], side * sy_all[sl, None]
        tp = np.tan(nav.pitch[sl, None])

        # 빔 평면과 지형의 교선 (1회 고정점 반복)
        Px, Py = Sx + g * ux, Sy + g * uy
        rows, cols = world.to_index(Px, Py)
        h = _sample(world, world.height, rows, cols, 1)
        a = (Sz - h) * tp
        Px, Py = Px + a * fx, Py + a * fy
        rows, cols = world.to_index(Px, Py)
        h = _sample(world, world.height, rows, cols, 1)
        gx = _sample(world, hx, rows, cols, 1)
        gy = _sample(world, hy, rows, cols, 1)
        refl = _sample(world, world.reflectivity, rows, cols, 1)
        spec = _sample(world, world.specular, rows, cols, 0)
        oid = _sample(world, world.object_id.astype(float), rows, cols, 0).astype(int)

        rho = np.hypot(g, a)  # 수평거리
        dz = Sz - h
        r = np.sqrt(rho ** 2 + dz ** 2)
        # 소나 쪽 단위벡터와 표면 법선
        dx, dy = (Sx - Px) / r, (Sy - Py) / r
        cos_inc = (-gx * dx - gy * dy + dz / r) / np.sqrt(gx ** 2 + gy ** 2 + 1.0)
        cos_inc = np.clip(cos_inc, 0.0, 1.0)
        th = np.arccos(cos_inc)
        sigma = refl * cos_inc ** 2 + spec * np.exp(-((th / w_spec) ** 2)) * (cos_inc > 0)

        # 그림자 + 가린 물체 (가장 최근 누적 최대 지점)
        e = np.arctan2(h - Sz, rho)
        cm = np.maximum.accumulate(e, axis=1)
        visible = e >= cm - 1e-9
        k_idx = np.broadcast_to(np.arange(len(g)), e.shape)
        caster = np.maximum.accumulate(np.where(visible, k_idx, 0), axis=1)
        caster_oid = np.take_along_axis(oid, caster, axis=1)

        center = tilt + side * nav.roll[sl, None]
        delta = np.arctan2(dz, rho)
        G = _beam_gain(delta, center, sonar)
        alpha = sonar.absorption_db_per_m
        loss = r ** -4.0 * 10 ** (-2 * alpha * r / 10)
        energy = sigma * G ** 2 * loss * visible * dg

        def acc(rr, w):
            b = (rr / dr).astype(int)
            ok = b < nb
            flat = (np.arange(n)[:, None] * nb + b)[ok]
            return np.bincount(flat, weights=np.broadcast_to(w, rr.shape)[ok], minlength=n * nb).reshape(n, nb)

        out[sl] += acc(r, energy)
        ones = np.ones_like(r)
        lab["count"][sl] += acc(r, ones)
        lab["shadow"][sl] += acc(r, ~visible)
        lab["obj_e"][sl] += acc(r, energy * (oid > 0))
        lab["obj_id_e"][sl] += acc(r, energy * oid)
        sh_obj = (~visible) & (caster_oid > 0)
        lab["objshadow"][sl] += acc(r, sh_obj)
        lab["objshadow_id"][sl] += acc(r, sh_obj * caster_oid)
        lab["ground"][sl] += acc(r, energy)

        if multipath_ratio is not None:
            zm = 2 * water_depth - Sz  # 거울상 소나
            rm = np.sqrt(rho ** 2 + (zm - h) ** 2)
            delta_m = np.arctan2(Sz - (2 * water_depth - h), rho)  # 수면을 향하는 다리 (위쪽, 음수)
            Gm = _beam_gain(delta_m, center, sonar)
            e_mp = 2 * multipath_ratio * sigma * G * Gm * (r * rm) ** -2.0 * 10 ** (-alpha * (r + rm) / 10) * visible * dg
            out[sl] += acc(0.5 * (r + rm), e_mp)

        if surface_ratio is not None:
            dzs = water_depth - Sz  # 수면까지 (양수)
            rs = np.sqrt(g ** 2 + dzs ** 2)
            Gs = _beam_gain(-np.arctan2(dzs, g), center, sonar)
            e_s = surface_ratio * (dzs / rs) ** 2 * Gs ** 2 * rs ** -4.0 * 10 ** (-2 * alpha * rs / 10) * dg
            out[sl] += acc(rs, e_s)

    tvg = tvg_curve(sonar)
    out *= tvg[None, :]
    cnt = np.maximum(lab["count"], 1.0)
    labels = {
        "shadow": (lab["shadow"] / cnt > 0.9) & (lab["count"] > 0),
        "object_id": np.where(lab["obj_e"] > 0.5 * np.maximum(lab["ground"], 1e-300),
                              np.rint(lab["obj_id_e"] / np.maximum(lab["obj_e"], 1e-300)), 0).astype(np.int32),
        "object_shadow_id": np.where(lab["objshadow"] / cnt > 0.5,
                                     np.rint(lab["objshadow_id"] / np.maximum(lab["objshadow"], 1e-300)), 0).astype(np.int32),
        "seafloor": lab["count"] > 0,
    }
    return SideRender(out, labels)
