"""엔진 3D: 3D 음향 FDTD (엇갈림 Yee 격자, 압력-속도, 선형 음향, 이상 유체).

지배 방정식 (전단 응력 없음):
    ∂v/∂t = −(1/ρ) ∇p,      ∂p/∂t = −ρ c² ∇·v,      u = ∫ v dt

격자: p 는 셀 중심 (nx,ny,nz), vx 는 x-면 (nx+1,ny,nz), vy (nx,ny+1,nz), vz (nx,ny,nz+1). 배열 축 0,1,2 = x,y,z.
도약법: v 는 t_{n+½}, p 는 t_n. 안정 조건 c dt/dx ≤ 1/√3.
    2D 엔진 B 와 같은 이산 연산자를 3D 로 확장한 것이라 이산 curl∘grad ≡ 0 이 그대로 성립한다 (유체에 횡파 없음).

경계:
    강체 (물체·송신기 원판·격자 바깥 끝): 면 법선 속도 0 (미끄럼 허용).
    송신기 앞면: 원판 셀과 그 앞쪽 유체 셀 사이 계단 면마다 면 속도 = V(t) (n·e_축). 계단을 따라 합한 부피 유량이
        실제 원판 면과 같다 (2D 기울어진 막대와 같은 방식, 2D 테스트로 검증됨).
    흡수층 PML: 분리장(split-field) 방식. p = px + py + pz, (∂t + σ_a) p_a = −ρc² ∂_a v_a, (∂t + σ_a) v_a = −(1/ρ) ∂_a p.

신호:
    연속파 (cw): 켜짐 구간 뒤 영역을 몇 번 횡단할 때까지 진행하고, 마지막 정수 주기들에서 이산 푸리에 평균으로
        정상상태 페이저 P (영역 전체), U (입자 격자) 를 얻는다 → 어떤 시각이든 즉시 재생.
    핑 (ping): 프레임마다 입자 격자 변위·압력, 직교 단면 3장의 압력, 셀별 최대 |p| 를 기록한다.

정밀도: float32 (메모리·속도 절반). 상대 반올림 오차 ~1e-7 이라 결과 해석에 영향 없음 (회전 비 검사로 확인).
알려진 이산화 오차 (파장당 셀 N): 3D 위상속도 오차 ≈ (π/N)²/6 (N=10 에서 약 1.6%, 축 방향 최악).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..config import ConfigError
from .config3d import Sim3DConfig
from .geometry3d import body_mask, transducer_cells

F32 = np.float32


class Cancelled(Exception):
    """사용자가 계산을 취소함."""


@dataclass
class Grid3D:
    nx: int
    ny: int
    nz: int
    dx: float
    xc: np.ndarray
    yc: np.ndarray
    zc: np.ndarray
    solid: np.ndarray  # (nx,ny,nz) bool
    obs: tuple[slice, slice, slice]  # 관측 영역 (PML 제외)
    n_pml: int
    drive: list[dict]  # 축별 구동 면: {"axis", "flat" (면 배열 평면 인덱스), "factor", "tid", "fluid_flat", "sign"}
    counts: dict[str, int]
    warnings: list[str]


def _sigma(n_cells: int, n_pml: int, lo: bool, hi: bool, smax: float, order: int, half: bool) -> np.ndarray:
    idx = np.arange(n_cells + 1) if half else np.arange(n_cells) + 0.5
    s = np.zeros(idx.shape)
    if n_pml:
        if lo:
            s = np.maximum(s, smax * np.clip((n_pml - idx) / n_pml, 0, 1) ** order)
        if hi:
            s = np.maximum(s, smax * np.clip((idx - (n_cells - n_pml)) / n_pml, 0, 1) ** order)
    return s


def setup_grid(cfg: Sim3DConfig) -> Grid3D:
    cfg.validate()
    dx = cfg.dx
    n_pml = math.ceil(cfg.pml_wavelengths * cfg.cells_per_wavelength)
    x0, x1, y0, y1, z0, z1 = cfg.domain
    lo = [n_pml if cfg.pml_sides[2 * a] else 0 for a in range(3)]
    hi = [n_pml if cfg.pml_sides[2 * a + 1] else 0 for a in range(3)]
    axes, obs, n = [], [], []
    for a, (u0, u1) in enumerate(((x0, x1), (y0, y1), (z0, z1))):
        i0 = math.floor(u0 / dx + 1e-9) - lo[a]  # 면 좌표 = dx 정수배 (축 정렬 경계가 면에 정확히 놓이게)
        m = math.ceil(u1 / dx - 1e-9) + hi[a] - i0
        axes.append((i0 + np.arange(m) + 0.5) * dx)
        obs.append(slice(lo[a], m - hi[a]))
        n.append(m)
    nx, ny, nz = n
    xc, yc, zc = axes
    warns: list[str] = []
    solid = np.zeros((nx, ny, nz), bool)
    counts: dict[str, int] = {}
    for i, b in enumerate(cfg.bodies):
        name = b.name or f"물체 {i + 1}"
        cnt, odd = body_mask(b, xc, yc, zc, solid)
        if cnt == 0:
            raise ConfigError(f"{name}: 격자 셀을 하나도 덮지 않습니다 (너무 작거나 영역 밖).")
        if odd:
            warns.append(f"{name}: STL 메시가 닫혀 있지 않습니다 (광선 {odd}개가 면을 홀수 번 지남). "
                         "구멍이 있으면 내부 채우기가 틀릴 수 있습니다. CAD 에서 '닫힌 솔리드' 로 내보내세요.")
        counts[name] = cnt
    tcells = []
    for i, t in enumerate(cfg.transducers):
        r = transducer_cells(t, xc, yc, zc)
        if r is None or not r[1].any():
            raise ConfigError(f"{t.name or f'송신기 {i + 1}'}: 격자 셀을 덮지 않습니다 (지름을 셀 크기보다 크게).")
        solid[r[0]] |= r[1]
        tcells.append(r)

    drive: list[dict] = []
    taken = [set(), set(), set()]
    for ti, (t, (sl, tm)) in enumerate(zip(cfg.transducers, tcells)):
        name = t.name or f"송신기 {ti + 1}"
        if not (t.enabled and t.amplitude > 0):
            continue
        nrm = t.normal
        c = np.asarray(t.center)
        a_rad = t.diameter / 2
        # 원판 블록을 한 칸 넓혀 앞쪽 유체 셀까지 포함
        ex = tuple(slice(max(s.start - 1, 0), min(s.stop + 1, m)) for s, m in zip(sl, (nx, ny, nz)))
        T = np.zeros(tuple(s.stop - s.start for s in ex), bool)
        T[tuple(slice(s.start - e.start, s.stop - e.start) for s, e in zip(sl, ex))] = tm
        S = solid[ex]
        X = xc[ex[0]][:, None, None] - c[0]
        Y = yc[ex[1]][None, :, None] - c[1]
        Z = zc[ex[2]][None, None, :] - c[2]
        d = X * nrm[0] + Y * nrm[1] + Z * nrm[2]
        rad2 = (X - d * nrm[0]) ** 2 + (Y - d * nrm[1]) ** 2 + (Z - d * nrm[2]) ** 2
        front = (d > 0) & (rad2 <= a_rad**2 * 1.0001)  # 원판 앞쪽, 반지름 안 (유체 셀 기준)
        count = 0
        for axis in range(3):
            lo_s = [slice(None)] * 3
            hi_s = [slice(None)] * 3
            lo_s[axis], hi_s[axis] = slice(None, -1), slice(1, None)
            lo_s, hi_s = tuple(lo_s), tuple(hi_s)
            for t_s, f_s, sign in ((lo_s, hi_s, +1), (hi_s, lo_s, -1)):  # (원판 셀, 유체 셀, 유체 쪽 방향)
                cand = T[t_s] & ~S[f_s] & front[f_s]
                idx = np.argwhere(cand)
                if idx.size == 0:
                    continue
                cell_t = idx + np.array([e.start for e in ex])  # 원판 셀 전역 인덱스 (cand 는 lo/hi 잘린 좌표)
                if sign < 0:
                    cell_t[:, axis] += 1  # hi 쪽 잘림 보정: t_s = hi_s 이면 원판 셀은 한 칸 위
                cell_f = cell_t.copy()
                cell_f[:, axis] += sign
                face = cell_t.copy()  # 면 인덱스 = 두 셀 중 큰 인덱스 (면 i 는 셀 i−1 과 i 사이)
                face[:, axis] = np.maximum(cell_t[:, axis], cell_f[:, axis])
                shape = [nx, ny, nz]
                shape[axis] += 1
                flat = np.ravel_multi_index(face.T, shape)
                keep = np.array([f not in taken[axis] for f in flat], bool)
                flat, cell_f = flat[keep], cell_f[keep]
                taken[axis].update(flat.tolist())
                if flat.size == 0:
                    continue
                drive.append(dict(axis=axis, flat=flat, factor=F32(nrm[axis]), tid=ti, sign=sign,
                                  fluid_flat=np.ravel_multi_index(cell_f.T, (nx, ny, nz))))
                count += flat.size
        counts[name] = count
        if count == 0:
            raise ConfigError(f"{name}: 앞면이 유체에 닿지 않습니다 (물체 속에 묻혔는지 확인).")
    if any(abs(v) > 1e-9 and abs(abs(v) - 1) > 1e-9 for t in cfg.transducers for v in t.normal):
        warns.append(f"기울어진 송신기 면은 격자(셀 {dx * 1000:.0f} mm) 위에서 계단 모양으로 근사됩니다.")
    return Grid3D(nx, ny, nz, dx, xc, yc, zc, solid, tuple(obs), n_pml, drive, counts, warns)


@dataclass
class Result3D:
    cfg: Sim3DConfig
    kind: str  # "cw" | "ping"
    xc: np.ndarray  # 관측 영역 셀 중심 [m]
    yc: np.ndarray
    zc: np.ndarray
    solid: np.ndarray  # (nx,ny,nz) bool (관측 영역)
    dx: float
    dt: float
    n_steps: int
    steps_per_period: int
    lat_idx: np.ndarray  # (L,3) 입자 격자 셀 인덱스 (관측 영역 기준, 유체만)
    lat_pos: np.ndarray  # (L,3) 평형 위치 [m]
    lat_spacing: float  # 입자 간격 [m]
    curl_ratio: float
    solve_s: float
    driven_faces: dict[str, int]
    n_cells_total: int
    # 연속파: p(t) = Re(P e^{−iωt})
    P: np.ndarray | None = None  # (nx,ny,nz) complex64 [Pa]
    U_lat: np.ndarray | None = None  # (L,3) complex64 [m]
    convergence: float = float("nan")  # 마지막 두 평균 창 사이 입자 격자 |P| 변화 (최대값 대비)
    # 핑
    times: np.ndarray | None = None  # (F,) [s]
    u_frames: np.ndarray | None = None  # (F,L,3) float32 [m]
    p_frames: np.ndarray | None = None  # (F,L) float32 [Pa]
    slice_index: tuple[int, int, int] = (0, 0, 0)  # 단면 위치 (관측 영역 셀 인덱스)
    slice_frames: dict = field(default_factory=dict)  # axis → (F, a, b) float32 [Pa]
    p_max: np.ndarray | None = None  # (nx,ny,nz) float32 셀별 최대 |p| (프레임 표본) [Pa]
    energy: dict = field(default_factory=dict)  # {"t", "E" 영역 내 음향 에너지 [J], "W" 송신기가 한 일 [J]}
    warnings: list[str] = field(default_factory=list)

    @property
    def omega(self) -> float:
        return self.cfg.signal.omega

    def p_slice(self, axis: int, index: int, t: float | None = None, frame: int | None = None) -> np.ndarray:
        """축 axis 에 수직인 단면의 압력 [Pa] (강체 = NaN). 연속파는 시각 t, 핑은 기록된 단면(frame)."""
        if self.kind == "cw":
            sl = [slice(None)] * 3
            sl[axis] = index
            sl = tuple(sl)
            vals = np.real(self.P[sl] * np.exp(-1j * self.omega * (t or 0.0)))
        else:
            vals = self.slice_frames[axis][frame or 0]
            sl = [slice(None)] * 3
            sl[axis] = self.slice_index[axis]
            sl = tuple(sl)
        return np.where(self.solid[sl], np.nan, vals)


def time_step(cfg: Sim3DConfig) -> tuple[float, int]:
    """(dt [s], 주기당 스텝 수). 한 주기 = 정수 스텝 → 연속파 이산 푸리에 평균이 정확."""
    spp = math.ceil(cfg.cells_per_wavelength / cfg.cfl)
    return 1.0 / cfg.signal.frequency / spp, spp


def estimate(cfg: Sim3DConfig) -> dict:
    """계산 전 견적: 셀 수, 스텝 수, 메모리 [MB] (실행 시간은 기계마다 달라 GUI 에서 실측 속도로 환산)."""
    nx, ny, nz = cfg.grid_shape()
    dt, spp = time_step(cfg)
    n_steps = _n_steps(cfg, dt, spp)
    cells = nx * ny * nz
    return dict(shape=(nx, ny, nz), cells=cells, steps=n_steps, mem_mb=cells * 4 * 16 / 1e6)


def _diag(cfg: Sim3DConfig) -> float:
    x0, x1, y0, y1, z0, z1 = cfg.domain
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)


def _n_steps(cfg: Sim3DConfig, dt: float, spp: int) -> int:
    c = cfg.medium.c
    sig = cfg.signal
    max_delay = max((t.delay for t in cfg.transducers if t.enabled), default=0.0)
    if sig.kind == "cw":
        t_lead = max_delay + sig.ramp_cycles / sig.frequency + cfg.transits * _diag(cfg) / c
        n_lead = math.ceil(t_lead / dt / spp) * spp
        return n_lead + 2 * cfg.avg_periods * spp  # 평균 창 2개 (수렴 확인용)
    return math.ceil((max_delay + sig.duration + cfg.ping_transits * _diag(cfg) / c) / dt)


def run(cfg: Sim3DConfig, progress=None, should_stop=None, slice_center=None) -> Result3D:
    """FDTD 실행. progress(fraction), should_stop() → True 면 Cancelled.
    slice_center [m]: 핑 단면 위치 (기본 = 첫 송신기 중심)."""
    t_start = time.perf_counter()
    g = setup_grid(cfg)
    rho, c = cfg.medium.rho, cfg.medium.c
    sig = cfg.signal
    nx, ny, nz, dx = g.nx, g.ny, g.nz, g.dx
    dt, spp = time_step(cfg)
    assert c * dt / dx <= 1 / math.sqrt(3) + 1e-12, "CFL 안정 조건 위반"
    n_total = _n_steps(cfg, dt, spp)
    smax = (cfg.pml_order + 1) * c * math.log(1 / cfg.pml_reflection) / (2 * max(g.n_pml, 1) * dx)
    shapes = ((nx, ny, nz),) * 3
    sides = cfg.pml_sides

    # ---- 계수 (PML 은 축별 1D, 강체 면은 B = 0) ----
    pad = np.pad(g.solid, 1, constant_values=True)
    fixed = (pad[:-1, 1:-1, 1:-1] | pad[1:, 1:-1, 1:-1],
             pad[1:-1, :-1, 1:-1] | pad[1:-1, 1:, 1:-1],
             pad[1:-1, 1:-1, :-1] | pad[1:-1, 1:-1, 1:])
    A_v, B_v, A_p, B_p = [], [], [], []
    for a, n_a in enumerate((nx, ny, nz)):
        sf = _sigma(n_a, g.n_pml, sides[2 * a], sides[2 * a + 1], smax, cfg.pml_order, True) * dt / 2
        sc = _sigma(n_a, g.n_pml, sides[2 * a], sides[2 * a + 1], smax, cfg.pml_order, False) * dt / 2
        bshape = [1, 1, 1]
        bshape[a] = -1
        A_v.append(((1 - sf) / (1 + sf)).astype(F32).reshape(bshape))
        B_v.append((~fixed[a]) * ((dt / (rho * dx)) / (1 + sf)).astype(F32).reshape(bshape))
        A_p.append(((1 - sc) / (1 + sc)).astype(F32).reshape(bshape))
        B_p.append(((rho * c**2 * dt / dx) / (1 + sc)).astype(F32).reshape(bshape))
    B_v = [b.astype(F32) for b in B_v]
    v = [np.zeros((nx + 1, ny, nz), F32), np.zeros((nx, ny + 1, nz), F32), np.zeros((nx, ny, nz + 1), F32)]
    pa = [np.zeros(shapes[0], F32) for _ in range(3)]
    p = np.zeros(shapes[0], F32)
    buf = np.empty(shapes[0], F32)

    # 구동 면 묶기
    amps = [t.amplitude for t in cfg.transducers]
    phases = [math.radians(t.phase_deg) for t in cfg.transducers]
    delays = [t.delay for t in cfg.transducers]

    # ---- 관측 영역과 입자 격자 ----
    ox, oy, oz = g.obs
    xo, yo, zo = g.xc[ox], g.yc[oy], g.zc[oz]
    solid_o = g.solid[ox, oy, oz]
    no = solid_o.shape
    stride = max(1, math.ceil(max(no) / cfg.particles_per_axis))
    grids = [np.arange(stride // 2, m, stride) for m in no]
    I, J, K = np.meshgrid(*grids, indexing="ij")
    lat = np.stack([I.ravel(), J.ravel(), K.ravel()], -1)
    lat = lat[~solid_o[lat[:, 0], lat[:, 1], lat[:, 2]]]
    lat_full = lat + np.array([ox.start, oy.start, oz.start])
    lat_pos = np.stack([xo[lat[:, 0]], yo[lat[:, 1]], zo[lat[:, 2]]], -1)
    p_flat = np.ravel_multi_index(lat_full.T, (nx, ny, nz))
    v_lo, v_hi = [], []  # 셀 중심 속도 = 양쪽 면 평균
    for a in range(3):
        shp = [nx, ny, nz]
        shp[a] += 1
        hi_idx = lat_full.copy()
        hi_idx[:, a] += 1
        v_lo.append(np.ravel_multi_index(lat_full.T, shp))
        v_hi.append(np.ravel_multi_index(hi_idx.T, shp))

    def v_lattice() -> np.ndarray:
        return np.stack([0.5 * (v[a].ravel()[v_lo[a]] + v[a].ravel()[v_hi[a]]) for a in range(3)], -1)

    res = Result3D(cfg=cfg, kind=sig.kind, xc=xo, yc=yo, zc=zo, solid=solid_o, dx=dx, dt=dt, n_steps=n_total,
                   steps_per_period=spp, lat_idx=lat, lat_pos=lat_pos, lat_spacing=stride * dx, curl_ratio=float("nan"),
                   solve_s=0.0, driven_faces=g.counts, n_cells_total=nx * ny * nz, warnings=list(g.warnings))

    # ---- 기록 준비 ----
    omega = sig.omega
    if sig.kind == "cw":
        win = cfg.avg_periods * spp
        n_avg0 = n_total - 2 * win  # 창 1: [n_avg0, n_avg0+win), 창 2: 마지막 win 스텝
        P_acc = np.zeros(no, np.complex64)
        Plat_acc = [np.zeros(len(lat), complex), np.zeros(len(lat), complex)]
        V_acc = np.zeros((len(lat), 3), complex)
    else:
        every = max(1, round(spp / cfg.frames_per_period))
        n_frames = n_total // every + 1
        if n_frames > cfg.max_frames:
            every = math.ceil(n_total / (cfg.max_frames - 1))
            n_frames = n_total // every + 1
        if slice_center is None:
            tr = next(t for t in cfg.transducers if t.enabled)
            slice_center = tr.center
        sidx = tuple(int(np.clip(np.searchsorted(ax_, s_), 0, len(ax_) - 1))
                     for ax_, s_ in zip((xo, yo, zo), slice_center))
        res.slice_index = sidx
        times, uf, pf = [], [], []
        sf = {0: [], 1: [], 2: []}
        u_lat = np.zeros((len(lat), 3))
        p_max = np.zeros(no, F32)
        fluid_o = ~solid_o
        e_t, e_E, e_W = [], [], []
        W = 0.0
        # 회전 검사 시점: 펄스가 다 나오고 영역 안에 머무는 동안 (끝에는 장이 거의 다 빠져나가 반올림 오차만 남음)
        max_delay = max((t.delay for t in cfg.transducers if t.enabled), default=0.0)
        n_curl = min(n_total - 1, int((max_delay + sig.duration + 0.15 * _diag(cfg) / c) / dt))
        p_drv = [(d, d["fluid_flat"]) for d in g.drive]

    def p_obs():
        return p[ox, oy, oz]

    # ---- 시간 진행 ----
    for n in range(n_total):
        tv = (n + 0.5) * dt  # v^{n+½}
        for a in range(3):
            va = v[a]
            sl_in = [slice(None)] * 3
            sl_in[a] = slice(1, -1)
            sl_in = tuple(sl_in)
            lo_ = [slice(None)] * 3
            hi_ = [slice(None)] * 3
            lo_[a], hi_[a] = slice(None, -1), slice(1, None)
            d = buf[tuple(slice(0, s - (1 if k == a else 0)) for k, s in enumerate(shapes[0]))]
            np.subtract(p[tuple(hi_)], p[tuple(lo_)], out=d)
            d *= B_v[a][sl_in]
            va_in = va[sl_in]
            va_in *= A_v[a][sl_in]
            va_in -= d
        # 송신기 면 (해석적 속도 파형)
        tvals = {}
        for dd in g.drive:
            ti = dd["tid"]
            if ti not in tvals:
                tvals[ti] = float(amps[ti] * sig.velocity(tv - delays[ti], phases[ti]))
            v[dd["axis"]].ravel()[dd["flat"]] = dd["factor"] * tvals[ti]
        if sig.kind == "ping":
            # 송신기가 유체에 한 일 dW = Σ p v_n dA dt (p 는 유체 쪽 셀, t_n 과 t_{n+1} 평균은 아래에서)
            p_before = [p.ravel()[ff] for _, ff in p_drv]
        for a in range(3):
            lo_ = [slice(None)] * 3
            hi_ = [slice(None)] * 3
            lo_[a], hi_[a] = slice(None, -1), slice(1, None)
            np.subtract(v[a][tuple(hi_)], v[a][tuple(lo_)], out=buf)
            buf *= B_p[a]
            pa[a] *= A_p[a]
            pa[a] -= buf
        np.add(pa[0], pa[1], out=p)
        p += pa[2]
        # 이제 p = p^{n+1}, v = v^{n+½}
        if sig.kind == "cw":
            if n + 1 > n_avg0:
                k = 0 if n + 1 <= n_avg0 + win else 1
                ep = np.exp(1j * omega * (n + 1) * dt)
                Plat_acc[k] += p.ravel()[p_flat] * ep
                if k == 1:
                    P_acc += p_obs() * np.complex64(ep)
                    V_acc += v_lattice() * np.exp(1j * omega * tv)
        else:
            u_lat += v_lattice() * dt
            for (dd, ff), pb in zip(p_drv, p_before):
                pm = 0.5 * (pb + p.ravel()[ff])  # p 를 v 시각 t_{n+½} 로 보간
                W += float(np.sum(pm * v[dd["axis"]].ravel()[dd["flat"]] * dd["sign"])) * dx * dx * dt
            if n == n_curl:
                res.curl_ratio = _curl_ratio(v, fixed, g)
            if (n + 1) % every == 0 or n == n_total - 1:
                po = p_obs()
                times.append((n + 1) * dt)
                uf.append(u_lat.astype(F32))
                pf.append(p.ravel()[p_flat].astype(F32))
                for a in range(3):
                    s_ = [slice(None)] * 3
                    s_[a] = sidx[a]
                    sf[a].append(po[tuple(s_)].copy())
                np.maximum(p_max, np.abs(po), out=p_max)
                vc = [0.5 * (v[a][tuple(slice(1, None) if k == a else slice(None) for k in range(3))]
                             + v[a][tuple(slice(None, -1) if k == a else slice(None) for k in range(3))])
                      for a in range(3)]
                ke = sum(vv[ox, oy, oz].astype(float) ** 2 for vv in vc)
                E = float(np.sum(np.where(fluid_o, po.astype(float) ** 2 / (2 * rho * c**2) + 0.5 * rho * ke, 0.0)))
                e_t.append((n + 1) * dt)
                e_E.append(E * dx**3)
                e_W.append(W)
        if progress is not None and n % 10 == 0:
            progress(n / n_total)
        if should_stop is not None and n % 10 == 0 and should_stop():
            raise Cancelled()

    # ---- 결과 ----
    if sig.kind == "cw":
        res.curl_ratio = _curl_ratio(v, fixed, g)
        scale = 2.0 / win
        res.P = np.where(solid_o, 0, P_acc * scale).astype(np.complex64)
        V = V_acc * scale
        res.U_lat = (1j * V / omega).astype(np.complex64)
        P1, P2 = Plat_acc[0] * scale, Plat_acc[1] * scale
        ref = np.abs(P2).max()
        res.convergence = float(np.abs(P2 - P1).max() / ref) if ref > 0 else float("nan")
        if res.convergence > 0.05:
            res.warnings.append(
                f"정상상태 미도달 가능성: 마지막 두 평균 창의 압력 차이 {res.convergence:.0%}. 물체 사이 공진이 오래 "
                "울리거나 영역이 커서 파가 아직 다 퍼지지 않았을 수 있습니다 (설정의 '횡단 횟수' 를 늘려 확인).")
    else:
        res.times = np.asarray(times)
        res.u_frames = np.asarray(uf)
        res.p_frames = np.asarray(pf)
        res.slice_frames = {a: np.asarray(sf[a]) for a in range(3)}
        res.p_max = np.where(solid_o, np.nan, p_max).astype(F32)
        res.energy = dict(t=np.asarray(e_t), E=np.asarray(e_E), W=np.asarray(e_W))
    res.solve_s = time.perf_counter() - t_start
    if progress is not None:
        progress(1.0)
    return res


def _take(arr: np.ndarray, axis: int, sl: slice) -> np.ndarray:
    idx = [slice(None)] * 3
    idx[axis] = sl
    return arr[tuple(idx)]


def _diff(arr: np.ndarray, axis: int) -> np.ndarray:
    return _take(arr, axis, slice(1, None)) - _take(arr, axis, slice(None, -1))


def _both(mask: np.ndarray, axis: int) -> np.ndarray:
    return _take(mask, axis, slice(1, None)) & _take(mask, axis, slice(None, -1))


def _curl_ratio(v, fixed, g: Grid3D) -> float:
    """max|∇×v| / max|∂_a v_a| (모서리를 둘러싼 4 면이 모두 자유이고 관측 영역 안인 모서리).

    모서리 축 e, 나머지 축 a=(e+1)%3, b=(e+2)%3: (∇×v)_e = ∂_a v_b − ∂_b v_a.
    v_b 의 b 축 안쪽 면 [1:-1] 을 a 축으로 차분, v_a 의 a 축 안쪽 면을 b 축으로 차분 → 둘 다 (n_e, n_a−1, n_b−1) 꼴.
    엇갈림 격자에서 자유 면 v = −Σ B ∇p 이므로 구조적으로 0 (float32 반올림 수준만 남음).
    """
    vv = [a.astype(float) for a in v]
    free = [~f for f in fixed]
    worst = 0.0
    for e in range(3):
        a, b = (e + 1) % 3, (e + 2) % 3
        curl = _diff(_take(vv[b], b, slice(1, -1)), a) - _diff(_take(vv[a], a, slice(1, -1)), b)
        ok = _both(_take(free[b], b, slice(1, -1)), a) & _both(_take(free[a], a, slice(1, -1)), b)
        roi = np.zeros(ok.shape, bool)
        r = [slice(None)] * 3
        r[e] = g.obs[e]
        r[a] = slice(g.obs[a].start, g.obs[a].stop - 1)  # 모서리 양쪽 셀이 모두 관측 영역 안
        r[b] = slice(g.obs[b].start, g.obs[b].stop - 1)
        roi[tuple(r)] = True
        ok &= roi
        if ok.any():
            worst = max(worst, float(np.abs(curl[ok]).max()))
    scale = max(float(np.abs(_diff(vv[a], a)).max()) for a in range(3))
    return worst / scale if scale > 0 else float("nan")
