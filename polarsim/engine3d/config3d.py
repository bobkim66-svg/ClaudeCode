"""엔진 3D 설정 (dataclass) 과 입력 검증.

단위: 길이 m, 시간 s, 주파수 Hz, 각도 입력은 도(°) (이 모듈 안에서 rad 로 바꿔 쓴다), 진폭 m.
좌표계: 오른손 직교좌표 (x, y, z), z 위쪽. 방향은 (방위각 az, 고각 el) 로 준다:
    방향 벡터 d = (cos el cos az, cos el sin az, sin el).  az = x 축에서 반시계 (xy 평면), el = xy 평면에서 위로.

물체 (Body) 는 강체 (법선 속도 0). 송신기 (Transducer) 는 원판 모양 강체이고 앞면이 피스톤처럼 진동한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..config import ConfigError, Medium

WATER = Medium(c=1500.0, rho=1000.0)
BODY_KINDS = ("box", "cylinder", "sphere", "stl")
BODY_KIND_LABELS = {"box": "직육면체", "cylinder": "원기둥", "sphere": "구", "stl": "STL 파일"}
MAX_BODIES = 16
MAX_TRANSDUCERS = 16
CFL_3D_MAX = 1 / math.sqrt(3)  # 3D 도약법 안정 조건 c dt / dx ≤ 1/√3


def direction(az_deg: float, el_deg: float) -> np.ndarray:
    """(방위각, 고각) [°] → 단위벡터 (3,)."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


def rotation(az_deg: float, el_deg: float) -> np.ndarray:
    """국소 → 전역 회전행렬 R (3,3). 열 0 = 국소 x 축 = direction(az, el), 열 1 = 수평 방향 국소 y,
    열 2 = 국소 z (az = el = 0 이면 단위행렬). 전역 = R @ 국소."""
    ex = direction(az_deg, el_deg)
    az = math.radians(az_deg)
    ey = np.array([-math.sin(az), math.cos(az), 0.0])  # 수평, ex 에 수직
    ez = np.cross(ex, ey)
    return np.stack([ex, ey, ez], axis=1)


def _finite(vals, tag: str) -> None:
    if not all(math.isfinite(float(v)) for v in vals):
        raise ConfigError(f"{tag}: NaN/inf 값이 있습니다.")


@dataclass(frozen=True)
class Body:
    """강체 물체.

    kind:
      box      : size = (길이 lx, 폭 ly, 높이 lz) [m], 국소 x 축을 (az, el) 방향으로 돌린 직육면체
      cylinder : size = (길이, 지름, -) [m], 축 = (az, el) 방향
      sphere   : size = (지름, -, -) [m]
      stl      : path 의 STL 삼각형 메시. size[0] = 배율 (STL 단위 → m, mm 파일이면 0.001),
                 center = 이동량 [m], (az, el) 회전을 원점 기준으로 적용한 뒤 이동
    """

    kind: str = "box"
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: tuple[float, float, float] = (1.0, 0.3, 0.3)
    az_deg: float = 0.0
    el_deg: float = 0.0
    name: str = ""
    path: str = ""

    def validate(self, index: int = 0) -> None:
        tag = self.name or f"물체 {index + 1}"
        if self.kind not in BODY_KINDS:
            raise ConfigError(f"{tag}: 종류는 {BODY_KINDS} 중 하나여야 합니다 (받은 값 {self.kind!r}).")
        _finite((*self.center, *self.size, self.az_deg, self.el_deg), tag)
        need = {"box": 3, "cylinder": 2, "sphere": 1, "stl": 1}[self.kind]
        if any(s <= 0 for s in self.size[:need]):
            raise ConfigError(f"{tag}: 크기(또는 배율)는 양수여야 합니다.")
        if self.kind == "stl" and not self.path:
            raise ConfigError(f"{tag}: STL 파일 경로가 비어 있습니다.")

    @property
    def R(self) -> np.ndarray:
        return rotation(self.az_deg, self.el_deg)


@dataclass(frozen=True)
class Transducer:
    """원판 송신기 (강체 원판, 앞면만 방사).

    center [m]: 원판 중심. (az, el): 앞면 법선 방향. diameter [m], thickness [m] (원판 두께, 앞뒤로 절반씩).
    앞면 변위 = amplitude · s(t − delay) (법선 방향), s 는 신호 파형 (Signal). phase_deg 는 연속파 위상 φ [°]:
    연속파 변위 ξ0 cos(ω(t − delay) − φ).
    """

    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    az_deg: float = 0.0
    el_deg: float = 0.0
    diameter: float = 0.2
    thickness: float = 0.06
    amplitude: float = 1e-6
    phase_deg: float = 0.0
    delay: float = 0.0
    enabled: bool = True
    name: str = ""

    @property
    def normal(self) -> np.ndarray:
        return direction(self.az_deg, self.el_deg)

    def min_thickness(self, dx: float) -> float:
        """계단 근사에서 원판에 구멍이 생기지 않는 최소 두께 (막대와 같은 규칙: 이웃 셀로 갈 때 법선 방향 거리 변화의 최대)."""
        return dx * float(np.max(np.abs(self.normal)))

    def validate(self, index: int = 0) -> None:
        tag = self.name or f"송신기 {index + 1}"
        _finite((*self.center, self.az_deg, self.el_deg, self.diameter, self.thickness, self.amplitude,
                 self.phase_deg, self.delay), tag)
        if self.diameter <= 0 or self.thickness <= 0:
            raise ConfigError(f"{tag}: 지름과 두께는 양수여야 합니다.")
        if self.amplitude < 0 or self.delay < 0:
            raise ConfigError(f"{tag}: 진폭과 지연은 0 이상이어야 합니다.")


@dataclass(frozen=True)
class Signal:
    """송신 파형 s(t) (최대 1, 무차원).

    cw  : 연속파 cos(ωt − φ). 처음 ramp_cycles 주기 동안 부드럽게 켠다 (갑자기 켜면 넓은 대역 잡음).
    ping: cycles 주기짜리 펄스 sin²(π t / T_p) · cos(ωt − φ), T_p = cycles / f (Hann 창).
    """

    kind: Literal["cw", "ping"] = "cw"
    frequency: float = 5000.0
    cycles: float = 5.0
    ramp_cycles: float = 3.0

    @property
    def omega(self) -> float:
        return 2 * math.pi * self.frequency

    @property
    def duration(self) -> float:
        """핑 길이 T_p [s] (연속파는 inf)."""
        return self.cycles / self.frequency if self.kind == "ping" else math.inf

    def envelope_and_rate(self, t: np.ndarray | float):
        """(창 w(t), dw/dt). t [s] (지연 적용 후 시각)."""
        t = np.asarray(t, dtype=float)
        if self.kind == "cw":
            tr = self.ramp_cycles / self.frequency
            if tr <= 0:
                return np.where(t >= 0, 1.0, 0.0), np.zeros_like(t)
            x = np.clip(t / tr, 0.0, 1.0)
            w = 0.5 * (1 - np.cos(np.pi * x))
            dw = np.where((t > 0) & (t < tr), 0.5 * np.pi / tr * np.sin(np.pi * x), 0.0)
            return w, dw
        tp = self.duration
        inside = (t >= 0) & (t <= tp)
        w = np.where(inside, np.sin(np.pi * t / tp) ** 2, 0.0)
        dw = np.where(inside, np.pi / tp * np.sin(2 * np.pi * t / tp), 0.0)
        return w, dw

    def displacement(self, t, phase: float = 0.0):
        """정규화 변위 s(t) = w(t) cos(ωt − φ)."""
        w, _ = self.envelope_and_rate(t)
        return w * np.cos(self.omega * np.asarray(t, float) - phase)

    def velocity(self, t, phase: float = 0.0):
        """ds/dt [1/s] (해석적 미분: 수치 미분 잡음 없음)."""
        t = np.asarray(t, float)
        w, dw = self.envelope_and_rate(t)
        arg = self.omega * t - phase
        return dw * np.cos(arg) - w * self.omega * np.sin(arg)

    def validate(self) -> None:
        if self.kind not in ("cw", "ping"):
            raise ConfigError(f"신호 종류는 'cw' 또는 'ping' 이어야 합니다 (받은 값 {self.kind!r}).")
        if not (math.isfinite(self.frequency) and self.frequency > 0):
            raise ConfigError("주파수는 양수여야 합니다.")
        if self.kind == "ping" and not (0.5 <= self.cycles <= 200):
            raise ConfigError("핑 주기 수는 0.5 ~ 200 이어야 합니다.")
        if self.ramp_cycles < 0:
            raise ConfigError("켜짐 구간(ramp) 주기 수는 0 이상이어야 합니다.")


@dataclass(frozen=True)
class Sim3DConfig:
    medium: Medium = WATER
    signal: Signal = field(default_factory=Signal)
    domain: tuple[float, float, float, float, float, float] = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)  # x0,x1,y0,y1,z0,z1
    cells_per_wavelength: float = 10.0
    pml_wavelengths: float = 1.0
    pml_sides: tuple[bool, bool, bool, bool, bool, bool] = (True,) * 6  # x0,x1,y0,y1,z0,z1 (False = 강체 벽)
    pml_reflection: float = 1e-4
    pml_order: int = 3
    bodies: tuple[Body, ...] = ()
    transducers: tuple[Transducer, ...] = ()
    cfl: float = 0.5
    max_cells: int = 12_000_000
    # 연속파: 켜짐 + 영역 대각선 transits 번 횡단 뒤 avg_periods 주기 평균 (정상상태 페이저)
    transits: float = 2.0
    avg_periods: int = 2
    # 핑: 펄스 길이 + 영역 대각선 ping_transits 번 횡단까지 계산, 주기당 frames_per_period 장 기록
    ping_transits: float = 1.2
    frames_per_period: int = 8
    max_frames: int = 400
    particles_per_axis: int = 28  # 입자 격자 (축당 최대)

    @property
    def wavelength(self) -> float:
        return self.medium.c / self.signal.frequency

    @property
    def dx(self) -> float:
        return self.wavelength / self.cells_per_wavelength

    def grid_shape(self) -> tuple[int, int, int]:
        """PML 포함 대략 격자 크기 (견적용)."""
        npml = math.ceil(self.pml_wavelengths * self.cells_per_wavelength)
        x0, x1, y0, y1, z0, z1 = self.domain
        s = self.pml_sides
        return (math.ceil((x1 - x0) / self.dx) + npml * (s[0] + s[1]),
                math.ceil((y1 - y0) / self.dx) + npml * (s[2] + s[3]),
                math.ceil((z1 - z0) / self.dx) + npml * (s[4] + s[5]))

    def validate(self) -> None:
        self.medium.validate()
        self.signal.validate()
        x0, x1, y0, y1, z0, z1 = self.domain
        _finite(self.domain, "계산 영역")
        if not (x1 > x0 and y1 > y0 and z1 > z0):
            raise ConfigError("계산 영역: x1 > x0, y1 > y0, z1 > z0 이어야 합니다.")
        if not (6 <= self.cells_per_wavelength <= 40):
            raise ConfigError("파장당 셀 수는 6 ~ 40 이어야 합니다 (6 미만은 수치 분산이 너무 큼).")
        if not (0 < self.cfl <= CFL_3D_MAX):
            raise ConfigError(f"CFL 안정 조건 위반: c·dt/dx = {self.cfl} > 1/√3 (3D).")
        if self.pml_wavelengths <= 0 or self.avg_periods < 1 or self.frames_per_period < 2:
            raise ConfigError("PML 두께, 평균 주기 수, 주기당 프레임 수를 확인하세요.")
        nx, ny, nz = self.grid_shape()
        if nx * ny * nz > self.max_cells:
            raise ConfigError(f"격자가 너무 큽니다 ({nx}×{ny}×{nz} = {nx * ny * nz / 1e6:.1f}M 셀, 상한 "
                              f"{self.max_cells / 1e6:.0f}M). 영역을 줄이거나 파장당 셀 수를 낮추세요.")
        if len(self.bodies) > MAX_BODIES or len(self.transducers) > MAX_TRANSDUCERS:
            raise ConfigError(f"물체는 {MAX_BODIES}개, 송신기는 {MAX_TRANSDUCERS}개까지입니다.")
        for i, b in enumerate(self.bodies):
            b.validate(i)
        for i, t in enumerate(self.transducers):
            t.validate(i)
            if t.enabled and t.thickness <= 1.05 * t.min_thickness(self.dx):
                raise ConfigError(
                    f"{t.name or f'송신기 {i + 1}'}: 두께 {t.thickness * 1000:.0f} mm 가 격자(셀 {self.dx * 1000:.0f} mm)에 비해 "
                    f"얇아 구멍이 생길 수 있습니다. {1.1 * t.min_thickness(self.dx) * 1000:.0f} mm 이상으로 하세요.")
            c = np.asarray(t.center)
            if not (x0 <= c[0] <= x1 and y0 <= c[1] <= y1 and z0 <= c[2] <= z1):
                raise ConfigError(f"{t.name or f'송신기 {i + 1}'}: 중심이 계산 영역 밖입니다.")
        if not any(t.enabled and t.amplitude > 0 for t in self.transducers):
            raise ConfigError("켜진 송신기가 없습니다.")
