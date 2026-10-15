"""시뮬레이션 설정 (모든 파라미터를 한 곳에 모은다). JSON 장면 파일과 상호 변환된다.

좌표계 (문서 전체에서 고정):
- 월드: ENU.  x = 동(East) [m], y = 북(North) [m], z = 위(Up) [m].  해저 기준면 z = 0.
- 진행 방위 heading_deg: +x 축에서 반시계(수학 각도) [deg].  0 = 동쪽, 90 = 북쪽.
- 차체: 전방 f = (cos psi, sin psi),  우현 s = (sin psi, -cos psi)  (위에서 볼 때 전방의 오른쪽).
- 자세: roll > 0  -> 우현이 아래로 (우현 빔이 더 아래를 봄),
        pitch > 0 -> 기수 위로 (빔 평면이 앞으로 기울어 발자국이 전방으로 이동),
        yaw   = heading 의 섭동.
- 영상: 행 = 핑(시간 순), 열 = [좌현 먼 거리 ... 좌현 가까운 | 우현 가까운 ... 우현 먼 거리].
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import List, Optional, Tuple

SOUND_SPEED_MPS = 1500.0


@dataclass
class SonarConfig:
    """소나 장비 파라미터. 기본값은 400~500 kHz 급 일반 SSS 의 대략값 (특정 장비 아님, 확인 필요)."""

    max_range_m: float = 40.0  # 한쪽 최대 경사거리
    n_bins: int = 512  # 한쪽 거리 bin 수
    range_resolution_m: float = 0.08  # c*tau/2 (펄스 길이에 의한 거리 해상도, -3 dB 폭)
    horizontal_beamwidth_deg: float = 0.6  # 진행방향 -3 dB 전폭 -> 진행방향 해상도 ~ R * theta_h
    vertical_halfwidth_deg: float = 25.0  # 수직 빔 -3 dB 반폭
    tilt_deg: float = 15.0  # 빔 중심 내림각
    sidelobe_db: float = -25.0  # 빔 밖 최소 이득 (수주/수면 쪽 약한 수신)
    absorption_db_per_m: float = 0.1  # 400 kHz 대 해수 흡수 근사 (확인 필요)
    tvg_exponent: float = 40.0  # 수신기 TVG 보상 (dB = k log10 r). 실제 확산은 40 log r
    beam_correction: bool = True  # 평탄 해저 기대응답으로 나누는 경험적 빔패턴 보정
    ping_rate_hz: float = 15.0
    # 수치 파라미터 (물리 아님)
    ground_oversample: int = 4  # 거리 bin 당 지상 표본 수 (클수록 정확, 느림)
    render_chunk_pings: int = 64  # 메모리 절약용 핑 묶음 크기
    min_along_sigma_pings: float = 0.3  # 근거리 진행방향 PSF 하한 (핑 간격보다 빔 발자국이 작을 때)

    @property
    def bin_size_m(self) -> float:
        return self.max_range_m / self.n_bins


@dataclass
class TrajectoryConfig:
    """항로와 차체 운동. 직선 (start -> end) 위를 일정 속도로 이동."""

    start_xy: Tuple[float, float] = (0.0, 0.0)
    end_xy: Tuple[float, float] = (40.0, 0.0)
    speed_mps: float = 1.5
    altitude_m: float = 8.0  # 해저 기준면 위 공칭 고도
    water_depth_m: float = 20.0  # 해저 기준면에서 수면까지
    # 운동 (진폭, 주기). 0 이면 끔. 각 성분 = 진폭 * sin(2 pi t / 주기 + 무작위 위상) + 저주파 랜덤
    heave_amp_m: float = 0.15
    heave_period_s: float = 6.0
    roll_amp_deg: float = 1.5
    roll_period_s: float = 5.0
    pitch_amp_deg: float = 0.8
    pitch_period_s: float = 7.0
    yaw_amp_deg: float = 1.0
    yaw_period_s: float = 12.0


@dataclass
class SeabedConfig:
    """해저 지형과 퇴적물."""

    grid_resolution_m: float = 0.05
    roughness_rms_m: float = 0.08  # 멱법칙 지형의 RMS 높이
    roughness_beta: float = 3.5  # 2D 파워 스펙트럼 P(k) ~ k^-beta (확인 필요: 해저마다 다름)
    roughness_cutoff_m: float = 30.0  # 이보다 긴 파장은 제거 (전체 기울기 방지)
    ripple_amp_m: float = 0.03
    ripple_wavelength_m: float = 0.8
    ripple_direction_deg: float = 30.0  # 마루선 법선 방향
    ripple_coverage: float = 0.35  # 물결이 보이는 면적 비율 (0~1)
    regional_slope_deg: Tuple[float, float] = (0.0, 0.0)  # (x 방향, y 방향) 경사
    # 퇴적물: 매끄러운 랜덤장을 문턱값으로 나눠 sand/mud/gravel. 반사율은 선형 강도 상대값
    sand_reflectivity: float = 1.0
    mud_reflectivity: float = 0.3
    gravel_reflectivity: float = 2.2
    mud_fraction: float = 0.2
    gravel_fraction: float = 0.1
    sediment_corr_m: float = 6.0
    texture_shape: Optional[float] = 8.0  # 감마 텍스처 nu (K-분포형 꼬리). None 이면 없음
    texture_corr_m: float = 0.3
    n_rocks: int = 15
    rock_radius_m: Tuple[float, float] = (0.2, 0.8)
    rock_reflectivity: float = 3.0
    flat_patch: Optional[Tuple[float, float, float, float]] = None  # (x0, x1, y0, y1) 완전 평탄 모래 (검증용)
    object_specular_width_deg: float = 10.0  # 물체 정반사 로브 폭 (입사각 기준)


@dataclass
class ObjectSpec:
    """해저 물체. kind = cylinder | box | sphere.  위치/크기 m, 방위 deg (x 축에서 반시계)."""

    kind: str
    x: float
    y: float
    length: float = 2.0  # cylinder/box 축 길이, sphere 는 무시
    width: float = 0.6  # cylinder 지름, box 폭, sphere 지름
    height: float = 0.6  # box 높이 (cylinder/sphere 는 지름에서 자동)
    heading_deg: float = 0.0
    burial_m: float = 0.0  # 매몰 깊이
    reflectivity: float = 6.0  # 확산 반사율
    specular: float = 30.0  # 정반사 강도 (금속 표면의 반짝임)
    name: str = ""


@dataclass
class NoiseConfig:
    """잡음과 아티팩트. 각각 켜고 끌 수 있다."""

    speckle: bool = True
    n_looks: int = 1
    snr_far_db: Optional[float] = 18.0  # 최대거리 모래 평균 대비 열잡음. None 이면 열잡음 없음
    surface_return_db: Optional[float] = -12.0  # 수면 산란 (모래 평균 대비). None 이면 없음
    multipath_db: Optional[float] = -18.0  # 수면 1회 반사 경로 이득. None 이면 없음
    volume_scatter_db: Optional[float] = -12.0  # 수주 체적 산란: TVG/빔보정 후 거의 일정한 바탕 (상수 근사)
    n_fish: int = 6  # 수주 점 산란체 무리 (물고기/부유물)
    fish_level_db: float = 15.0  # 점 산란체 최대 에너지 (PSF 적용 전, 모래 평균 대비)
    interference_prob: float = 0.0  # 타 장비 간섭 줄 (0 이면 끔)
    interference_db: float = 8.0
    dropout_prob: float = 0.0  # 핑 누락 확률


@dataclass
class SceneConfig:
    """시뮬레이션 전체 설정."""

    name: str = "default"
    seed: int = 0
    sonar: SonarConfig = field(default_factory=SonarConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    seabed: SeabedConfig = field(default_factory=SeabedConfig)
    objects: List[ObjectSpec] = field(default_factory=list)
    noise: NoiseConfig = field(default_factory=NoiseConfig)

    def validate(self) -> None:
        s, t = self.sonar, self.trajectory
        max_prf = SOUND_SPEED_MPS / (2 * s.max_range_m)
        if s.ping_rate_hz > max_prf:
            raise ValueError(
                f"ping_rate {s.ping_rate_hz} Hz > c/(2 R_max) = {max_prf:.2f} Hz: 거리 모호성 (다음 핑 전에 메아리가 안 돌아옴)"
            )
        if not 0 < t.altitude_m < s.max_range_m:
            raise ValueError("0 < altitude < max_range")
        if t.water_depth_m <= t.altitude_m:
            raise ValueError("water_depth > altitude 이어야 한다 (소나가 수중에 있어야 함)")
        if t.speed_mps <= 0 or s.n_bins < 8:
            raise ValueError("speed > 0, n_bins >= 8")
        if self.noise.n_looks < 1:
            raise ValueError("n_looks >= 1")
        for o in self.objects:
            if o.kind not in ("cylinder", "box", "sphere"):
                raise ValueError(f"알 수 없는 물체 종류: {o.kind}")

    # ---- JSON ----------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SceneConfig":
        return _build(cls, d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SceneConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _build(cls, d: dict):
    """중첩 dataclass 를 dict 에서 만든다. 모르는 키는 오류 (오타 방지)."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(known)
    if unknown:
        raise KeyError(f"{cls.__name__}: 알 수 없는 설정 키 {sorted(unknown)}")
    kwargs = {}
    sub = {"sonar": SonarConfig, "trajectory": TrajectoryConfig, "seabed": SeabedConfig, "noise": NoiseConfig}
    for k, v in d.items():
        if cls is SceneConfig and k in sub:
            kwargs[k] = _build(sub[k], v)
        elif cls is SceneConfig and k == "objects":
            kwargs[k] = [_build(ObjectSpec, o) for o in v]
        elif isinstance(v, list):
            kwargs[k] = tuple(v)
        else:
            kwargs[k] = v
    obj = cls(**kwargs)
    assert is_dataclass(obj)
    return obj
