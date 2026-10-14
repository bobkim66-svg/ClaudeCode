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


# ---------------------------------------------------------------------------
# JSON (저장·불러오기). STL 경로는 JSON 파일 기준 상대경로로 저장해 폴더째 옮겨도 열리게 한다.
# ---------------------------------------------------------------------------
JSON_VERSION = 1


def to_dict(cfg: Sim3DConfig, base_dir=None) -> dict:
    from dataclasses import asdict
    from pathlib import Path

    def rel(path: str) -> str:
        if not path or base_dir is None:
            return path
        try:
            return Path(path).resolve().relative_to(Path(base_dir).resolve()).as_posix()
        except ValueError:
            return Path(path).resolve().as_posix()

    bodies = []
    for b in cfg.bodies:
        d = asdict(b)
        d["path"] = rel(b.path)
        bodies.append(d)
    return {
        "version": JSON_VERSION,
        "medium": {"c": cfg.medium.c, "rho": cfg.medium.rho},
        "signal": asdict(cfg.signal),
        "domain": list(cfg.domain),
        "cells_per_wavelength": cfg.cells_per_wavelength,
        "pml_sides": list(cfg.pml_sides),
        "bodies": bodies,
        "transducers": [asdict(t) for t in cfg.transducers],
    }


def from_dict(d: dict, base_dir=None) -> Sim3DConfig:
    """to_dict 의 역. 모르는 키는 무시하고, 빠진 키는 기본값."""
    from pathlib import Path

    def pick(cls, src: dict):
        names = cls.__dataclass_fields__
        kw = {k: v for k, v in src.items() if k in names}
        for k in ("center", "size"):
            if k in kw:
                kw[k] = tuple(float(x) for x in kw[k])
        return cls(**kw)

    try:
        bodies = []
        for bd in d.get("bodies", []):
            b = pick(Body, bd)
            if b.path and base_dir is not None and not Path(b.path).is_absolute():
                b = Body(**{**b.__dict__, "path": str((Path(base_dir) / b.path).resolve())})
            bodies.append(b)
        med = d.get("medium", {})
        return Sim3DConfig(
            medium=Medium(c=float(med.get("c", WATER.c)), rho=float(med.get("rho", WATER.rho))),
            signal=pick(Signal, d.get("signal", {})),
            domain=tuple(float(x) for x in d.get("domain", Sim3DConfig.domain)),
            cells_per_wavelength=float(d.get("cells_per_wavelength", 10.0)),
            pml_sides=tuple(bool(x) for x in d.get("pml_sides", (True,) * 6)),
            bodies=tuple(bodies),
            transducers=tuple(pick(Transducer, t) for t in d.get("transducers", [])),
        )
    except (TypeError, ValueError, KeyError) as e:
        raise ConfigError(f"설정 파일 형식이 잘못되었습니다: {e}") from None


# ---------------------------------------------------------------------------
# 예시 장면
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Preset3D:
    title: str
    note: str
    config: Sim3DConfig


def presets() -> dict[str, Preset3D]:
    """모두 물 (c = 1500 m/s), 5 kHz (λ = 0.3 m) 기준."""
    auv = (Body("cylinder", (-0.3, 0.0, 0.0), (1.6, 0.3, 0.0), name="AUV 몸체 (길이 1.6 m, 지름 0.3 m)"),
           Body("sphere", (0.5, 0.0, 0.0), (0.3, 0.0, 0.0), name="AUV 머리 (반구)"))
    sonar_front = Transducer(center=(0.68, 0.0, 0.0), diameter=0.2, thickness=0.06, name="앞쪽 소나")
    seabed_ping = Sim3DConfig(
        signal=Signal("ping", 5000.0, cycles=4), domain=(-1.0, 1.6, -1.0, 1.0, -1.0, 0.8),
        pml_sides=(True, True, True, True, False, True),
        bodies=(Body("sphere", (0.9, 0.0, -0.8), (0.4, 0.0, 0.0), name="해저 바위"),
                Body("cylinder", (-0.3, 0.0, 0.35), (1.2, 0.25, 0.0), name="AUV 몸체")),
        transducers=(Transducer(center=(0.1, 0.0, 0.18), az_deg=0, el_deg=-50, diameter=0.2, thickness=0.08,
                                name="하향 소나 (아래 50°)"),))
    d = 0.15  # λ/2 간격
    array = tuple(Transducer(center=(-0.6, (k - 1.5) * d, 0.0), diameter=0.13, thickness=0.06,
                             phase_deg=90.0 * k, name=f"소자 {k + 1} (위상 {90 * k}°)") for k in range(4))
    a = 0.225
    return {
        "auv_cw": Preset3D(
            "AUV + 앞쪽 소나 (연속파)",
            "AUV 몸체(원기둥+구) 앞에 지름 20 cm 송신기. 몸체 표면을 따라 도는 회절과 뒤쪽 그림자를 볼 수 있습니다.",
            Sim3DConfig(domain=(-1.4, 1.8, -1.0, 1.0, -1.0, 1.0), bodies=auv, transducers=(sonar_front,))),
        "seabed_ping": Preset3D(
            "해저 위 하향 소나 핑 (바닥·바위 반사)",
            "z = −1 m 바닥은 강체 벽(해저, 흡수층 끔). 비스듬히 아래로 쏜 4주기 펄스가 바닥과 바위에서 반사되는 모습.",
            seabed_ping),
        "array_steer": Preset3D(
            "4소자 배열 빔 조향 (위상차 +90° → 약 30°)",
            "y 방향으로 λ/2 간격 소자 4개, 이웃 위상차 +90° (y 가 클수록 늦게 진동). 빔이 +x 에서 +y 쪽으로 sin θ = Δφ/(k d) = 0.5, 약 30° 꺾입니다.",
            Sim3DConfig(domain=(-0.8, 1.4, -1.0, 1.4, -0.8, 0.8), transducers=array)),
        "piston_check": Preset3D(
            "배플 피스톤 (해석해 검증용)",
            "무한 강체 벽(배플)에 박힌 원형 피스톤. 축 위 |p| 를 닫힌 해와 비교합니다 (요약에 오차 표시).",
            Sim3DConfig(domain=(-0.45, 0.45, -0.45, 0.45, -0.06, 0.9), cells_per_wavelength=16,
                        pml_sides=(True, True, True, True, False, True),
                        bodies=(Body("box", (0, 0, -0.03), (4, 4, 0.06), name="배플 (큰 강체 판)"),),
                        transducers=(Transducer(center=(0, 0, -0.03), el_deg=90, diameter=2 * a, thickness=0.06,
                                                name="피스톤 (지름 45 cm)"),))),
    }
