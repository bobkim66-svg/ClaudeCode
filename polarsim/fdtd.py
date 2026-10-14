"""엔진 B: 2D 유한차분 음향 (엇갈림 Yee 격자, 압력-속도, 선형 음향, 이상 유체). 벽·반사·회절 포함.

지배 방정식 (전단 응력 없음):
    ∂p/∂t = −ρ c² ∇·v,      ∂v/∂t = −(1/ρ) ∇p,      u = ∫ v dt

격자: p 는 셀 중심 (nx, ny), vx 는 x-면 (nx+1, ny), vy 는 y-면 (nx, ny+1). 배열 축 0 = x, 축 1 = y.
z 방향으로는 무한히 균일하다고 가정 (2D).

두 가지 풀이 (같은 격자·같은 이산 연산자):
    solve_harmonic(): 주파수 영역. e^{−iωt} 정상상태를 희소 선형계 한 번으로 정확히 구한다 (기본).
        ∂t → −iω,  PML 은 복소 좌표 늘이기 (σ − iω) 로 표현.
    run_fdtd(): 시간 영역 FDTD (도약법). 과도 응답을 보거나 주파수 영역 해를 독립적으로 교차 검증할 때.
        이상 유체 + 강체 벽은 손실이 거의 없어, 판 사이 횡모드가 차단 주파수 근처(f ≈ m c / 2D) 에 있으면
        공진이 오래 울려 정상상태 도달이 매우 느리다 (그래서 정상상태 지표는 주파수 영역 풀이를 쓴다).

비회전 성질: 엇갈림 격자에서 이산 curl∘grad ≡ 0 이 정확히 성립한다. 자유 유체 노드(주변 4 면이 압력
    기울기로만 결정되고 PML 밖) 에서 ∇×v 는 반올림 오차 수준이며 result.curl_ratio 로 보고한다.
    즉 이 엔진에서 '횡파' 는 구조적으로 생길 수 없다.

경계와 소스:
    - 강체(solid) 셀: 인접 면의 법선 속도 = 0 (미끄럼 허용, 이상 유체).
    - 방사면(Emitter): 강체 표면의 일부가 피스톤처럼 진동. 면 변위 ξ0 cos(ωt − φ) (유체 쪽 법선 방향).
      → 벽에서 나가는 평면파는 엔진 A 의 cos(k·(r−r0) − ωt + φ) 와 같은 위상 규약.
    - 바깥 경계: PML (흡수층). 반사율은 테스트에서 측정한다.

알려진 이산화 오차 (파장당 셀 N): 위상속도 오차 ≈ (π/N)²/6 (N=20 에서 약 0.4%, FDTD 는 CFL 에 따라 조금 작음),
    셀 중심 속도 = 양쪽 면 평균이라 진폭이 cos(π/N) 배 (N=20 에서 −1.2%).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .config import ConfigError, Medium

WATER = Medium(c=1500.0, rho=1000.0)


@dataclass(frozen=True)
class Rect:
    """강체 직사각형 [m]."""

    x0: float
    x1: float
    y0: float
    y1: float


@dataclass(frozen=True)
class Emitter:
    """강체 표면의 방사 구간.

    axis: 면 법선 축 ('x' 면은 x = position 인 수직선, 'y' 면은 y = position 인 수평선).
    position: 면 좌표 [m]. span: 면을 따라가는 구간 (lo, hi) [m].
    normal: +1 / −1 = 유체 쪽 법선이 +축 / −축 방향.
    amplitude: 면 변위 진폭 ξ0 [m]. phase: φ [rad].
    """

    axis: Literal["x", "y"]
    position: float
    span: tuple[float, float]
    normal: int
    amplitude: float = 1e-5
    phase: float = 0.0
    enabled: bool = True
    name: str = ""


@dataclass(frozen=True)
class WaveConfig:
    medium: Medium = WATER
    frequency: float = 3000.0  # [Hz]
    domain: tuple[float, float, float, float] = (-1.0, 4.0, -2.5, 2.5)  # 관측 영역 (x0, x1, y0, y1) [m], PML 제외
    cells_per_wavelength: float = 20.0
    pml_wavelengths: float = 1.5
    pml_sides: tuple[bool, bool, bool, bool] = (True, True, True, True)  # x0, x1, y0, y1
    pml_reflection: float = 1e-6  # 이론적 수직 입사 반사율 R0 (σ_max 결정)
    pml_order: int = 3
    solids: tuple[Rect, ...] = ()
    emitters: tuple[Emitter, ...] = ()
    max_cells: int = 2_000_000
    # FDTD 전용
    cfl: float = 0.5  # c dt / dx, 2D 안정 조건 ≤ 1/√2
    ramp_periods: float = 10.0
    min_transits: float = 1.0
    avg_periods: int = 4
    steady_tol: float = 0.01
    max_periods: int = 800

    @property
    def wavelength(self) -> float:
        return self.medium.c / self.frequency

    @property
    def dx(self) -> float:
        return self.wavelength / self.cells_per_wavelength

    def validate(self) -> None:
        self.medium.validate()
        if not (math.isfinite(self.frequency) and self.frequency > 0):
            raise ConfigError("주파수는 양수여야 합니다.")
        x0, x1, y0, y1 = self.domain
        if not (x1 > x0 and y1 > y0):
            raise ConfigError("계산 영역: x1 > x0, y1 > y0 이어야 합니다.")
        if self.cells_per_wavelength < 10:
            raise ConfigError("파장당 셀 수는 10 이상이어야 합니다 (수치 분산 오차).")
        if not (0 < self.cfl <= 1 / math.sqrt(2)):
            raise ConfigError(f"CFL 안정 조건 위반: c·dt/dx = {self.cfl} > 1/√2 (2D). 0.5 정도를 쓰세요.")
        if self.avg_periods < 1 or self.pml_wavelengths <= 0 or self.max_periods < 1:
            raise ConfigError("평균 주기 수, 최대 주기 수, PML 두께는 양수여야 합니다.")
        if not any(e.enabled and e.amplitude > 0 for e in self.emitters):
            raise ConfigError("켜진 방사면이 없습니다.")
        for e in self.emitters:
            if e.axis not in ("x", "y") or e.normal not in (1, -1) or e.span[1] <= e.span[0] or e.amplitude < 0:
                raise ConfigError(f"방사면 '{e.name}' 설정이 잘못되었습니다.")


@dataclass
class WaveResult:
    """관측 영역(PML 제외) 으로 잘라낸 정상상태 페이저. 배열 축 0 = x, 축 1 = y. 시간 의존 e^{−iωt}."""

    method: str  # "harmonic" | "fdtd"
    xc: np.ndarray  # (nx,) 셀 중심 x [m]
    yc: np.ndarray  # (ny,)
    solid: np.ndarray  # (nx,ny) bool
    P: np.ndarray  # (nx,ny) complex [Pa]
    V: np.ndarray  # (nx,ny,2) complex [m/s]
    omega: float
    dx: float
    curl_ratio: float  # max|∇×V| / max|∇V| (자유 유체 노드)
    pml_cells: int
    driven_faces: dict[str, int]
    dt: float | None = None
    n_steps: int | None = None
    steps_per_period: int | None = None
    convergence_history: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def U(self) -> np.ndarray:
        """변위 페이저 (nx,ny,2) [m]. v = ∂u/∂t → V = −iωU."""
        return 1j * self.V / self.omega

    @property
    def convergence(self) -> float:
        return self.convergence_history[-1] if self.convergence_history else float("nan")


# ---------------------------------------------------------------------------
# 격자·형상 (두 풀이 공용)
# ---------------------------------------------------------------------------
def _sigma_profile(n_cells: int, n_pml: int, lo: bool, hi: bool, sigma_max: float, order: int, half: bool):
    """PML 감쇠 σ [1/s]. half=True 면 면 위치(정수 인덱스 0..n), False 면 셀 중심(+½)."""
    idx = np.arange(n_cells + 1) if half else np.arange(n_cells) + 0.5
    s = np.zeros_like(idx, dtype=float)
    if n_pml == 0:
        return s
    if lo:
        s = np.maximum(s, sigma_max * np.clip((n_pml - idx) / n_pml, 0.0, 1.0) ** order)
    if hi:
        s = np.maximum(s, sigma_max * np.clip((idx - (n_cells - n_pml)) / n_pml, 0.0, 1.0) ** order)
    return s


@dataclass
class _Grid:
    nx: int
    ny: int
    dx: float
    xc: np.ndarray
    yc: np.ndarray
    solid: np.ndarray
    free_x: np.ndarray  # (nx+1, ny) 압력 기울기로 갱신되는 면
    free_y: np.ndarray  # (nx, ny+1)
    gx: np.ndarray  # (nx+1, ny) complex: 방사면 속도 페이저 (그 외 0)
    gy: np.ndarray
    sx_c: np.ndarray  # (nx,1)
    sy_c: np.ndarray  # (1,ny)
    sx_f: np.ndarray  # (nx+1,1)
    sy_f: np.ndarray  # (1,ny+1)
    sl_x: slice
    sl_y: slice
    n_pml: int
    counts: dict[str, int]
    warnings: list[str]


def _setup(cfg: WaveConfig) -> _Grid:
    cfg.validate()
    c = cfg.medium.c
    omega = 2 * math.pi * cfg.frequency
    dx = cfg.dx
    n_pml = int(math.ceil(cfg.pml_wavelengths * cfg.cells_per_wavelength))
    x0, x1, y0, y1 = cfg.domain
    p_lo_x, p_hi_x, p_lo_y, p_hi_y = (n_pml if s else 0 for s in cfg.pml_sides)
    # 면 좌표가 dx 의 정수배가 되도록 격자 원점을 잡는다 (형상 경계가 면에 정확히 놓이게).
    i0 = math.floor(x0 / dx + 1e-9) - p_lo_x
    j0 = math.floor(y0 / dx + 1e-9) - p_lo_y
    nx = math.ceil(x1 / dx - 1e-9) + p_hi_x - i0
    ny = math.ceil(y1 / dx - 1e-9) + p_hi_y - j0
    if nx * ny > cfg.max_cells:
        raise ConfigError(f"격자가 너무 큽니다 ({nx}×{ny}). 영역을 줄이거나 파장당 셀 수를 낮추세요.")
    X0, Y0 = i0 * dx, j0 * dx
    xc = X0 + (np.arange(nx) + 0.5) * dx
    yc = Y0 + (np.arange(ny) + 0.5) * dx

    warns: list[str] = []
    edges = [v for r in cfg.solids for v in (r.x0, r.x1, r.y0, r.y1)]
    edges += [e.position for e in cfg.emitters if e.enabled]
    snap = max((abs(v / dx - round(v / dx)) * dx for v in edges), default=0.0)
    if snap > 1e-6 * dx:
        warns.append(f"형상 경계가 격자(셀 {dx * 1000:.1f} mm)에 맞춰 최대 {snap * 1000:.1f} mm 이동했습니다. "
                     "판 두께 등 치수를 셀 크기의 배수로 맞추면 사라집니다.")
    solid = np.zeros((nx, ny), bool)
    for r in cfg.solids:
        solid |= (xc[:, None] > r.x0) & (xc[:, None] < r.x1) & (yc[None, :] > r.y0) & (yc[None, :] < r.y1)
    pad = np.pad(solid, 1, constant_values=True)  # 격자 밖 = 강체 (PML 바깥 끝)
    fixed_x = pad[:-1, 1:-1] | pad[1:, 1:-1]  # x-면 i: 셀 i−1, i 중 하나라도 강체
    fixed_y = pad[1:-1, :-1] | pad[1:-1, 1:]

    gx = np.zeros((nx + 1, ny), complex)
    gy = np.zeros((nx, ny + 1), complex)
    counts: dict[str, int] = {}
    for e in cfg.emitters:
        if not e.enabled or e.amplitude == 0:
            continue
        name = e.name or f"{e.axis}={e.position:g}"
        # 변위 ξ0 cos(ωt − φ) = Re(ξ0 e^{iφ} e^{−iωt}) → 속도 페이저 −iω ξ0 e^{iφ} (유체 쪽 법선 성분)
        vph = -1j * omega * e.amplitude * np.exp(1j * e.phase) * e.normal
        if e.axis == "x":
            i = round((e.position - X0) / dx)
            fi, si = (i, i - 1) if e.normal > 0 else (i - 1, i)
            js = [j for j in np.nonzero((yc > e.span[0]) & (yc < e.span[1]))[0]
                  if 0 <= fi < nx and not solid[fi, j] and (si < 0 or si >= nx or solid[si, j])]
            gx[i, js] = vph
            counts[name] = len(js)
        else:
            j = round((e.position - Y0) / dx)
            fj, sj = (j, j - 1) if e.normal > 0 else (j - 1, j)
            is_ = [i for i in np.nonzero((xc > e.span[0]) & (xc < e.span[1]))[0]
                   if 0 <= fj < ny and not solid[i, fj] and (sj < 0 or sj >= ny or solid[i, sj])]
            gy[is_, j] = vph
            counts[name] = len(is_)
        if counts[name] == 0:
            raise ConfigError(f"방사면 '{name}' 이 강체 표면과 맞지 않습니다 (한쪽은 강체, 반대쪽은 유체여야 함).")

    sigma_max = (cfg.pml_order + 1) * c * math.log(1 / cfg.pml_reflection) / (2 * n_pml * dx)
    lo_x, hi_x, lo_y, hi_y = cfg.pml_sides
    prof = lambda n, lo, hi, half: _sigma_profile(n, n_pml, lo, hi, sigma_max, cfg.pml_order, half)  # noqa: E731
    return _Grid(
        nx=nx, ny=ny, dx=dx, xc=xc, yc=yc, solid=solid, free_x=~fixed_x, free_y=~fixed_y, gx=gx, gy=gy,
        sx_c=prof(nx, lo_x, hi_x, False)[:, None], sy_c=prof(ny, lo_y, hi_y, False)[None, :],
        sx_f=prof(nx, lo_x, hi_x, True)[:, None], sy_f=prof(ny, lo_y, hi_y, True)[None, :],
        sl_x=slice(p_lo_x, nx - p_hi_x), sl_y=slice(p_lo_y, ny - p_hi_y), n_pml=n_pml, counts=counts,
        warnings=warns,
    )


def _curl_ratio(g: _Grid, Fx: np.ndarray, Fy: np.ndarray) -> float:
    """면 속도 페이저로 노드 회전 / 기울기 비. 자유 유체 노드만 (주변 4면 자유, PML 밖)."""
    dx = g.dx
    curl = (Fy[1:, 1:-1] - Fy[:-1, 1:-1]) / dx - (Fx[1:-1, 1:] - Fx[1:-1, :-1]) / dx  # 노드 (1..nx−1, 1..ny−1)
    ok = g.free_y[1:, 1:-1] & g.free_y[:-1, 1:-1] & g.free_x[1:-1, 1:] & g.free_x[1:-1, :-1]
    ok &= (g.sx_f[1:-1] == 0) & (g.sy_f[:, 1:-1] == 0)
    scale = max(np.abs(np.diff(Fx, axis=0)).max(), np.abs(np.diff(Fy, axis=1)).max()) / dx
    return float(np.abs(curl[ok]).max() / scale) if ok.any() and scale > 0 else float("nan")


def _result(method, g: _Grid, cfg: WaveConfig, omega, P, Fx, Fy, warnings=(), **kw) -> WaveResult:
    V = np.stack([0.5 * (Fx[1:] + Fx[:-1]), 0.5 * (Fy[:, 1:] + Fy[:, :-1])], -1)  # 셀 중심 = 양쪽 면 평균
    return WaveResult(
        method=method, xc=g.xc[g.sl_x], yc=g.yc[g.sl_y], solid=g.solid[g.sl_x, g.sl_y],
        P=P[g.sl_x, g.sl_y], V=V[g.sl_x, g.sl_y], omega=omega, dx=g.dx, curl_ratio=_curl_ratio(g, Fx, Fy),
        pml_cells=g.n_pml, driven_faces=g.counts, warnings=list(g.warnings) + list(warnings), **kw,
    )


# ---------------------------------------------------------------------------
# 주파수 영역 (기본)
# ---------------------------------------------------------------------------
def solve_harmonic(cfg: WaveConfig, omega_eff: float | None = None) -> WaveResult:
    """e^{−iωt} 정상상태를 희소 직접법으로 푼다.

    면: (σ_f − iω) v = −(1/ρ) ∂p           → 자유 면 v = −a_f Δp/dx,  a_f = 1/(ρ(σ_f − iω))
    셀: (σ_c − iω) p_x = −ρc² ∂x vx (y 도 같음) → p = −b_x Δvx/dx − b_y Δvy/dx,  b = ρc²/(σ_c − iω)
    omega_eff: 이산 연산자에 넣는 각주파수 (기본 = 물리 ω). FDTD 와 비교할 때 도약법의 2 sin(ωdt/2)/dt 를 넣는다.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    g = _setup(cfg)
    rho, c = cfg.medium.rho, cfg.medium.c
    omega = 2 * math.pi * cfg.frequency
    w = omega if omega_eff is None else omega_eff
    nx, ny, dx = g.nx, g.ny, g.dx
    a_x = (1.0 / (rho * (g.sx_f - 1j * w))) * g.free_x  # (nx+1, ny), 고정 면은 0
    a_y = (1.0 / (rho * (g.sy_f - 1j * w))) * g.free_y
    b_x = rho * c**2 / (g.sx_c - 1j * w)  # (nx,1)
    b_y = rho * c**2 / (g.sy_c - 1j * w)  # (1,ny)
    b_x = np.broadcast_to(b_x, (nx, ny))
    b_y = np.broadcast_to(b_y, (nx, ny))
    h2 = dx * dx

    idx = np.arange(nx * ny).reshape(nx, ny)
    aL, aR = a_x[:-1], a_x[1:]  # 셀 i 의 왼쪽/오른쪽 면
    aB, aT = a_y[:, :-1], a_y[:, 1:]
    diag = 1.0 + (b_x * (aL + aR) + b_y * (aB + aT)) / h2
    rows, cols, vals = [idx.ravel()], [idx.ravel()], [diag.ravel()]

    def off(mask_coef, shift_rows, shift_cols):
        rows.append(shift_rows.ravel())
        cols.append(shift_cols.ravel())
        vals.append(mask_coef.ravel())

    off(-(b_x * aR)[:-1] / h2, idx[:-1], idx[1:])  # 오른쪽 이웃
    off(-(b_x * aL)[1:] / h2, idx[1:], idx[:-1])  # 왼쪽 이웃
    off(-(b_y * aT)[:, :-1] / h2, idx[:, :-1], idx[:, 1:])
    off(-(b_y * aB)[:, 1:] / h2, idx[:, 1:], idx[:, :-1])
    rhs = -(b_x * (g.gx[1:] - g.gx[:-1]) + b_y * (g.gy[:, 1:] - g.gy[:, :-1])) / dx
    # 강체 셀: p = 0 (항등 행). 강체 셀 행의 비대각 성분은 aL..aT 가 모두 0 이라 이미 0.
    rhs = np.where(g.solid, 0.0, rhs)
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny,) * 2)
    P = spsolve(A.tocsc(), rhs.ravel()).reshape(nx, ny)

    Fx = g.gx.copy()
    Fy = g.gy.copy()
    Fx[1:-1] += -a_x[1:-1] * (P[1:] - P[:-1]) / dx
    Fy[:, 1:-1] += -a_y[:, 1:-1] * (P[:, 1:] - P[:, :-1]) / dx
    return _result("harmonic", g, cfg, omega, P, Fx, Fy)


# ---------------------------------------------------------------------------
# 시간 영역 FDTD (교차 검증·과도 응답)
# ---------------------------------------------------------------------------
def fdtd_time_step(cfg: WaveConfig) -> tuple[float, int]:
    """(dt [s], 주기당 스텝 수). 한 주기가 정수 스텝이 되게 해 이산 푸리에 평균이 정확하다."""
    period = 1 / cfg.frequency
    spp = math.ceil(period / (cfg.cfl * cfg.dx / cfg.medium.c))
    return period / spp, spp


def run_fdtd(cfg: WaveConfig, progress=None) -> WaveResult:
    """도약법 FDTD. 연속 창 페이저 차이가 steady_tol 미만이면 멈춤 (max_periods 초과 시 경고).

    주의: 창 사이 차이가 작아도 느리게 감쇠하는 공진이 있으면 최종값과의 차이는 더 클 수 있다
    (차이가 비율 q 로 줄면 남은 오차 ≈ 차이 × q/(1−q)). 정상상태 값은 solve_harmonic 을 기준으로 삼는다.
    """
    g = _setup(cfg)
    rho, c = cfg.medium.rho, cfg.medium.c
    omega = 2 * math.pi * cfg.frequency
    nx, ny, dx = g.nx, g.ny, g.dx
    dt, spp = fdtd_time_step(cfg)
    assert c * dt / dx <= 1 / math.sqrt(2) + 1e-12, "CFL 안정 조건 위반"
    period = 1 / cfg.frequency
    win = cfg.avg_periods * spp
    diag = math.hypot(nx * dx, ny * dx)
    n_first = math.ceil((cfg.min_transits * diag / c + cfg.ramp_periods * period) / period) * spp
    n_max = max(cfg.max_periods * spp, n_first + 2 * win)

    ax, ay = g.sx_f * dt / 2, g.sy_f * dt / 2
    bx, by = g.sx_c * dt / 2, g.sy_c * dt / 2
    k_v = dt / (rho * dx)
    k_p = rho * c**2 * dt / dx
    drv_x = np.nonzero(g.gx.ravel())[0]
    drv_y = np.nonzero(g.gy.ravel())[0]
    gxv, gyv = g.gx.ravel()[drv_x], g.gy.ravel()[drv_y]
    vx = np.zeros((nx + 1, ny))
    vy = np.zeros((nx, ny + 1))
    px = np.zeros((nx, ny))
    py = np.zeros((nx, ny))
    t_ramp = cfg.ramp_periods * period
    sl_x, sl_y = g.sl_x, g.sl_y
    roi_fluid = ~g.solid[sl_x, sl_y]
    warns: list[str] = []

    def ramp(t):
        return 1.0 if t >= t_ramp else 0.5 * (1 - math.cos(math.pi * t / t_ramp))

    def new_acc():
        return {"P": np.zeros((nx, ny), complex), "FX": np.zeros((nx + 1, ny), complex),
                "FY": np.zeros((nx, ny + 1), complex)}

    scale = 2.0 / win  # Q = (2/N) Σ q e^{iωt}, N = 정수 주기의 스텝 수
    acc, prev, cur, history = new_acc(), None, None, []
    n = 0
    while True:
        tv = (n + 0.5) * dt
        p = px + py
        vx[1:-1] = ((1 - ax[1:-1]) * vx[1:-1] - k_v * (p[1:] - p[:-1])) / (1 + ax[1:-1])
        vy[:, 1:-1] = ((1 - ay[:, 1:-1]) * vy[:, 1:-1] - k_v * (p[:, 1:] - p[:, :-1])) / (1 + ay[:, 1:-1])
        vx *= g.free_x
        vy *= g.free_y
        rot = ramp(tv) * np.exp(-1j * omega * tv)
        vx.flat[drv_x] = np.real(gxv * rot)  # 방사면: Re(G e^{−iωt})
        vy.flat[drv_y] = np.real(gyv * rot)
        px = ((1 - bx) * px - k_p * (vx[1:] - vx[:-1])) / (1 + bx)
        py = ((1 - by) * py - k_p * (vy[:, 1:] - vy[:, :-1])) / (1 + by)
        px[g.solid] = 0.0
        py[g.solid] = 0.0
        n += 1  # p 는 t_n = n dt, v 는 t_{n−½} = tv

        if n > n_first:
            ev = np.exp(1j * omega * tv)
            acc["P"] += (px + py) * np.exp(1j * omega * n * dt)
            acc["FX"] += vx * ev
            acc["FY"] += vy * ev
            if (n - n_first) % win == 0:
                cur = {k: v * scale for k, v in acc.items()}
                if prev is not None:
                    pb = np.abs(cur["P"][sl_x, sl_y][roi_fluid]).max()
                    diff = np.abs(cur["P"] - prev["P"])[sl_x, sl_y][roi_fluid].max()
                    history.append(float(diff / pb) if pb > 0 else float("nan"))
                    if history[-1] < cfg.steady_tol:
                        break
                prev, acc = cur, new_acc()
                if n + win > n_max:
                    last = f"{history[-1]:.1%}" if history else "-"
                    warns.append(f"FDTD 정상상태 미도달: {cfg.max_periods} 주기 안에 연속 창 차이가 "
                                 f"{cfg.steady_tol:.0%} 아래로 내려가지 않았습니다 (마지막 {last}).")
                    break
        if progress is not None and n % 200 == 0:
            progress(min(1.0, n / n_max))

    return _result("fdtd", g, cfg, omega, cur["P"], cur["FX"], cur["FY"], dt=dt, n_steps=n,
                   steps_per_period=spp, convergence_history=history, warnings=warns)


# ---------------------------------------------------------------------------
# AUV 품(박스) 형상
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PocketSpec:
    """삼면이 막힌 박스 (뒷벽 x=0, 윗판·아랫판, 입구 x=L). 각 내부 면이 피스톤처럼 방사.

    length L [m] (x 방향 깊이), gap D [m] (판 사이 간격, y ∈ [−D/2, D/2]), thickness [m] (판 두께),
    body_depth [m]: 뒷벽 뒤 AUV 몸체 길이 (강체). 각 면: (켜기, 변위 진폭 [m], 위상 [rad]).
    """

    length: float = 1.0
    gap: float = 1.0
    thickness: float = 0.1
    body_depth: float = 0.6
    back: tuple[bool, float, float] = (True, 1e-5, 0.0)
    top: tuple[bool, float, float] = (True, 1e-5, math.pi / 2)
    bottom: tuple[bool, float, float] = (True, 1e-5, math.pi / 2)

    def solids(self) -> tuple[Rect, ...]:
        h, t, L = self.gap / 2, self.thickness, self.length
        return (
            Rect(-self.body_depth, 0.0, -h - t, h + t),  # AUV 몸체 + 뒷벽
            Rect(0.0, L, h, h + t),  # 윗판
            Rect(0.0, L, -h - t, -h),  # 아랫판
        )

    def emitters(self) -> tuple[Emitter, ...]:
        h, L = self.gap / 2, self.length
        return (
            Emitter("x", 0.0, (-h, h), +1, self.back[1], self.back[2], self.back[0], "뒷벽 (+x 방사)"),
            Emitter("y", h, (0.0, L), -1, self.top[1], self.top[2], self.top[0], "윗판 (−y 방사)"),
            Emitter("y", -h, (0.0, L), +1, self.bottom[1], self.bottom[2], self.bottom[0], "아랫판 (+y 방사)"),
        )

    def cutoff_frequencies(self, c: float, f_max: float) -> list[float]:
        """판 사이 횡모드 차단 주파수 f_m = m c / (2D), m = 1.. (f_max 이하). 이 근처에서 공진."""
        return [m * c / (2 * self.gap) for m in range(1, int(2 * self.gap * f_max / c) + 1)]


def pocket_config(spec: PocketSpec, **kw) -> WaveConfig:
    return WaveConfig(solids=spec.solids(), emitters=spec.emitters(), **kw)


# ---------------------------------------------------------------------------
# 지표 (엔진 A 와 같은 정의, 단일 주파수 페이저에서 정확한 주기 평균)
# ---------------------------------------------------------------------------
def wave_metrics(res: WaveResult, medium: Medium, settings=None) -> dict:
    """WaveResult → 셀별 지표 (nx,ny): L, e, planarity, T, rms [m], I (nx,ny,2) [W/m²]. 강체 셀은 NaN.

    2D 이므로 변위의 z 성분은 0 (평면성은 1 또는 NaN). 마디 판정 기준 A_ref = 유체 영역 최대 |U|.
    """
    from .config import MetricSettings
    from .metrics import phasor_metrics

    settings = settings or MetricSettings()
    nx, ny = res.P.shape
    zeros = np.zeros((nx * ny, 1))
    U = np.concatenate([res.U.reshape(-1, 2), zeros], axis=1)
    V = np.concatenate([res.V.reshape(-1, 2), zeros], axis=1)
    fluid = ~res.solid.ravel()
    a_ref = float(np.abs(U[fluid]).max())
    i_ref = 0.5 * medium.rho * medium.c * res.omega**2 * a_ref**2
    m = phasor_metrics(U, res.P.ravel(), V, a_ref, i_ref, settings)
    out = {k: np.where(fluid, m[k], np.nan).reshape(nx, ny) for k in ("L", "e", "planarity", "T", "rms")}
    out["I"] = np.where(fluid[:, None], m["intensity"][:, :2], np.nan).reshape(nx, ny, 2)
    out["a_ref"], out["i_ref"] = a_ref, i_ref
    return out


def radial_ellipticity(res: WaveResult, met: dict, center: tuple[float, float], wavelength: float,
                       edges_lambda=(0, 1, 2, 4, 8), half_plane_x: float | None = None) -> list[dict]:
    """center 로부터 거리 구간별 타원도 통계. half_plane_x 를 주면 x > half_plane_x 인 셀만 (품 앞쪽).

    반환 각 구간: lo, hi [λ], n (셀 수), e_median, e_energy (|I| 가중 평균), frac_e03 (e > 0.3 비율),
    coverage (구간의 (반)고리 중 계산 영역 안에 든 비율).
    에너지 가중 평균은 진폭이 거의 0 인 곳(지향성 영점 등) 의 큰 e 에 끌려가지 않는다.
    주의: coverage < 1 인 구간은 계산 영역 모서리만 남아 특정 방향(예: 지향성 영점)만 표본이 되므로
    다른 구간과 비교하면 안 된다.
    """
    X, Y = np.meshgrid(res.xc, res.yc, indexing="ij")
    r = np.hypot(X - center[0], Y - center[1]) / wavelength
    # 고리 면적 대비 영역 안 셀 비율 (강체 포함해서 셈 → 순수하게 영역 잘림만 반영)
    geo = np.ones_like(r, bool) if half_plane_x is None else X > half_plane_x
    cell_area = (res.dx / wavelength) ** 2
    frac = 0.5 if half_plane_x is not None else 1.0
    base = np.isfinite(met["e"])
    if half_plane_x is not None:
        base &= X > half_plane_x
    Imag = np.linalg.norm(np.nan_to_num(met["I"]), axis=-1)
    rows = []
    for lo, hi in zip(edges_lambda[:-1], edges_lambda[1:]):
        s = base & (r >= lo) & (r < hi)
        e = met["e"][s]
        w = Imag[s]
        n_geo = int((geo & (r >= lo) & (r < hi)).sum())
        full = frac * np.pi * (hi**2 - lo**2) / cell_area
        rows.append(dict(lo=lo, hi=hi, n=int(s.sum()), coverage=min(1.0, n_geo / full),
                         e_median=float(np.median(e)) if e.size else float("nan"),
                         e_energy=float(np.sum(w * e) / w.sum()) if e.size and w.sum() > 0 else float("nan"),
                         frac_e03=float(np.mean(e > 0.3)) if e.size else float("nan")))
    return rows


def emitted_power(res: WaveResult, cfg: WaveConfig) -> float:
    """방사면이 유체에 넣는 시간평균 일률 [W/m] (z 방향 단위 길이당) ≈ Σ ½Re(p v_n*) dx.

    p 는 방사면에 붙은 유체 셀의 값 (면에서 반 셀 떨어짐 → O(k dx) 근사)."""
    g = _setup(cfg)
    P_full = np.zeros((g.nx, g.ny), complex)
    P_full[g.sl_x, g.sl_y] = res.P
    total = 0.0
    for i, j in zip(*np.nonzero(g.gx)):
        fi = i if (i < g.nx and not g.solid[i, j]) else i - 1  # 유체 쪽 셀
        sign = 1 if fi == i else -1  # 유체 쪽 법선 (+x 면이면 +1)
        total += 0.5 * np.real(P_full[fi, j] * np.conj(g.gx[i, j] * sign)) * g.dx
    for i, j in zip(*np.nonzero(g.gy)):
        fj = j if (j < g.ny and not g.solid[i, j]) else j - 1
        sign = 1 if fj == j else -1
        total += 0.5 * np.real(P_full[i, fj] * np.conj(g.gy[i, j] * sign)) * g.dx
    return float(total)
