"""시뮬레이션 설정 (dataclass) 과 입력 검증.

단위 규약 (모든 모듈 공통):
    길이 m, 시간 s, 주파수 Hz, 위상 rad, 밀도 kg/m^3, 음속 m/s.
    무차원 모드는 Medium.dimensionless() (c=1, rho=1) 와 f=1 을 쓰면 파장 = 1.
    GUI 에서 위상을 도(deg) 로 입력받더라도 이 모듈에는 rad 로 넘긴다.

좌표계: 오른손 직교좌표 (x, y, z). ENU/NED 같은 지리 좌표계는 쓰지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

MAX_SOURCES = 8
SourceKind = Literal["plane", "gaussian"]


class ConfigError(ValueError):
    """사용자 입력이 물리적으로/수치적으로 유효하지 않을 때. 메시지는 UI 에 그대로 표시한다."""


def _vec3(value, name: str) -> np.ndarray:
    """길이 3 실수 벡터로 변환하고 NaN/inf 를 거부한다."""
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ConfigError(f"{name}: 3성분 벡터여야 합니다 (받은 shape {arr.shape}).")
    if not np.all(np.isfinite(arr)):
        raise ConfigError(f"{name}: NaN/inf 가 포함되어 있습니다.")
    return arr


def unit_vector(value, name: str) -> np.ndarray:
    """영벡터를 거부하고 단위벡터를 반환. shape (3,), 무차원."""
    arr = _vec3(value, name)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ConfigError(f"{name}: 영벡터는 방향이 정의되지 않습니다.")
    return arr / norm


def _finite_positive(value, name: str) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ConfigError(f"{name}: 양의 유한한 값이어야 합니다 (받은 값 {value}).")
    return v


@dataclass(frozen=True)
class Medium:
    """이상 유체 (전단 강성 없음). c [m/s], rho [kg/m^3]."""

    c: float = 343.0
    rho: float = 1.2

    @staticmethod
    def dimensionless() -> "Medium":
        return Medium(c=1.0, rho=1.0)

    def validate(self) -> None:
        _finite_positive(self.c, "음속 c")
        _finite_positive(self.rho, "밀도 rho")


@dataclass(frozen=True)
class Source:
    """음원 하나.

    direction: 전파 방향 (3,), 내부에서 단위화. 크기는 무시된다.
    amplitude: 변위 진폭 [m]. plane 이면 그 파의 변위 진폭,
               gaussian 이면 초점(ref_point)에서 모든 성분이 동위상일 때의 스칼라 진폭 합.
    frequency: [Hz], > 0.
    phase: 초기 위상 [rad]. 위상 기준점은 ref_point.
    kind: "plane" (무한 평면파) 또는 "gaussian" (평면파 각스펙트럼 합으로 만든 유한 폭 빔).
    waist: gaussian 빔의 허리 반경 w0 [m] (스칼라 프로파일 exp(-rho^2/w0^2)).
    ref_point: 위상 기준점 / 빔 초점 [m], shape (3,).
    n_angular: 각스펙트럼 격자 한 변의 표본 수 (gaussian 전용, 홀수 권장).
    """

    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    amplitude: float = 1e-6
    frequency: float = 1000.0
    phase: float = 0.0
    enabled: bool = True
    kind: SourceKind = "plane"
    waist: float | None = None
    ref_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    n_angular: int = 21

    def validate(self, index: int = 0) -> None:
        tag = f"소스 {index + 1}"
        unit_vector(self.direction, f"{tag} 방향 벡터")
        _vec3(self.ref_point, f"{tag} 기준점")
        a = float(self.amplitude)
        if not np.isfinite(a) or a < 0.0:
            raise ConfigError(f"{tag} 진폭: 0 이상의 유한한 값이어야 합니다 (받은 값 {self.amplitude}).")
        _finite_positive(self.frequency, f"{tag} 주파수")
        if not np.isfinite(float(self.phase)):
            raise ConfigError(f"{tag} 위상: 유한한 값이어야 합니다.")
        if self.kind not in ("plane", "gaussian"):
            raise ConfigError(f"{tag} 종류: 'plane' 또는 'gaussian' 이어야 합니다 (받은 값 {self.kind!r}).")
        if self.kind == "gaussian":
            if self.waist is None:
                raise ConfigError(f"{tag}: 가우시안 빔은 허리 반경 waist 가 필요합니다.")
            _finite_positive(self.waist, f"{tag} 허리 반경")
            if int(self.n_angular) < 3:
                raise ConfigError(f"{tag} 각스펙트럼 표본 수: 3 이상이어야 합니다.")


@dataclass(frozen=True)
class ObservationPlane:
    """관측 평면 위 nx × ny 규칙 격자.

    origin: 평면 중심 [m] (3,).
    normal: 법선 (3,), 내부에서 단위화.
    spacing: 격자 간격 [m]. None 이면 최소 파장 / 10.
    u_axis: 평면 내 첫 축의 힌트 (3,). 법선 성분을 제거해 사용. None 이면 자동 선택.
    """

    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    nx: int = 40
    ny: int = 40
    spacing: float | None = None
    u_axis: tuple[float, float, float] | None = None

    def validate(self) -> None:
        _vec3(self.origin, "관측 평면 원점")
        unit_vector(self.normal, "관측 평면 법선")
        for name, n in (("nx", self.nx), ("ny", self.ny)):
            if int(n) != n or int(n) < 1:
                raise ConfigError(f"관측 격자 {name}: 1 이상의 정수여야 합니다 (받은 값 {n}).")
        if self.spacing is not None:
            _finite_positive(self.spacing, "격자 간격")
        if self.u_axis is not None:
            hint = _vec3(self.u_axis, "평면 축 힌트")
            n = unit_vector(self.normal, "관측 평면 법선")
            if np.linalg.norm(hint - np.dot(hint, n) * n) < 1e-9:
                raise ConfigError("평면 축 힌트가 법선과 평행(또는 영벡터)합니다.")


@dataclass(frozen=True)
class TimeWindow:
    """지표 계산용 시간창.

    mode="auto": 주파수들이 (허용오차 내) 유리수 비이면 공통 주기, 아니면 manual_periods 사용.
    mode="manual": duration [s] 사용 (None 이면 manual_periods × 최장 주기).
    samples_per_period: 최고 주파수 한 주기당 표본 수.
    """

    mode: Literal["auto", "manual"] = "auto"
    duration: float | None = None
    samples_per_period: int = 64
    max_auto_periods: int = 50
    manual_periods: int = 20
    max_samples: int = 8192

    def validate(self) -> None:
        if self.mode not in ("auto", "manual"):
            raise ConfigError(f"시간창 모드: 'auto' 또는 'manual' 이어야 합니다 (받은 값 {self.mode!r}).")
        if self.duration is not None:
            _finite_positive(self.duration, "시간창 길이")
        if int(self.samples_per_period) < 8:
            raise ConfigError("주기당 표본 수는 8 이상이어야 합니다 (2차량 평균의 앨리어싱 방지).")
        if int(self.max_auto_periods) < 1 or int(self.manual_periods) < 1:
            raise ConfigError("주기 수 설정은 1 이상이어야 합니다.")


@dataclass(frozen=True)
class MetricSettings:
    """지표의 '정의 불가(NaN)' 판정 임계값. 모두 무차원 상대값.

    node_tol: λ1 < node_tol · A_ref² 이면 변위가 사실상 0 (마디) → L, e, 평면성, T = NaN.
              A_ref = 활성 소스 진폭 합 (변위 진폭의 상한).
    linear_tol: λ2 < linear_tol · λ1 이면 궤적이 직선 → 평면성 = NaN (0/0).
    intensity_tol: |I| < intensity_tol · I_ref 이면 세기 방향 정의 불가 → T = NaN.
                   I_ref = ½ ρ c ω_max² A_ref² (A_ref 진폭 단일 평면파의 세기).
    curl_h_frac: 회전 검증용 중앙차분 간격 h = curl_h_frac × 최소 파장.
    curl_n_times: 회전 검증에 쓰는 시간 표본 수.
    """

    node_tol: float = 1e-8
    linear_tol: float = 1e-10
    intensity_tol: float = 1e-3
    curl_h_frac: float = 1e-3
    curl_n_times: int = 8


@dataclass(frozen=True)
class SimConfig:
    medium: Medium = field(default_factory=Medium)
    sources: tuple[Source, ...] = field(default_factory=lambda: (Source(),))
    plane: ObservationPlane = field(default_factory=ObservationPlane)
    window: TimeWindow = field(default_factory=TimeWindow)
    metrics: MetricSettings = field(default_factory=MetricSettings)

    def active_sources(self) -> tuple[Source, ...]:
        return tuple(s for s in self.sources if s.enabled)

    def validate(self) -> None:
        """모든 입력을 검사. 문제가 있으면 ConfigError (사용자용 한글 메시지)."""
        self.medium.validate()
        if len(self.sources) > MAX_SOURCES:
            raise ConfigError(f"소스는 최대 {MAX_SOURCES}개까지 지원합니다 (받은 개수 {len(self.sources)}).")
        for i, s in enumerate(self.sources):
            s.validate(i)
        active = self.active_sources()
        if not active:
            raise ConfigError("활성화된 소스가 없습니다. 최소 1개를 켜 주세요.")
        if sum(float(s.amplitude) for s in active) <= 0.0:
            raise ConfigError("활성 소스의 진폭이 모두 0 입니다.")
        self.plane.validate()
        self.window.validate()

    def min_wavelength(self) -> float:
        """활성 소스 중 최소 파장 [m]."""
        return self.medium.c / max(s.frequency for s in self.active_sources())
