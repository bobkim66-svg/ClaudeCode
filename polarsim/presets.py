"""프리셋 3종과 해석적으로 예상되는 결과 (가이드/테스트에서 실제 결과와 비교)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Medium, ObservationPlane, SimConfig, Source

DEFAULT_AMPLITUDE = 1e-6  # [m] 실제 변위 진폭. 1 kHz 에서 파장 0.343 m 의 약 3e-6 배.
DEFAULT_FREQ = 1000.0  # [Hz]


@dataclass(frozen=True)
class Preset:
    key: str
    title: str
    expected: str
    config: SimConfig


def _cfg(sources, medium: Medium | None = None) -> SimConfig:
    return SimConfig(
        medium=medium or Medium(),
        sources=tuple(sources),
        plane=ObservationPlane(origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), nx=40, ny=40),
    )


def standing_wave(medium: Medium | None = None, freq: float = DEFAULT_FREQ) -> Preset:
    a = DEFAULT_AMPLITUDE
    src = [
        Source(direction=(1, 0, 0), amplitude=a, frequency=freq),
        Source(direction=(-1, 0, 0), amplitude=a, frequency=freq),
    ]
    return Preset(
        "standing",
        "정면 대향 2파 → 정재파",
        "u = 2A x̂ sin(kx) sin(ωt). 변위 마디 간격 λ/2, 마디(원점 포함)에서 진폭 0. "
        "모든 입자가 x 방향 직선 진동 → L = 1 (마디는 NaN). 진행 에너지가 없어 I = 0 → T 는 전 영역 NaN(회색).",
        _cfg(src, medium),
    )


def orthogonal_quadrature(medium: Medium | None = None, freq: float = DEFAULT_FREQ) -> Preset:
    a = DEFAULT_AMPLITUDE
    src = [
        Source(direction=(1, 0, 0), amplitude=a, frequency=freq, phase=0.0),
        Source(direction=(0, 1, 0), amplitude=a, frequency=freq, phase=np.pi / 2),
    ]
    return Preset(
        "circular",
        "직교 2파 + 90° 위상차 → 원 궤적",
        "원점에서 u = A(cos ωt, sin ωt, 0) → e = 1 (원), L = 0, T = 0.5 (Î ∥ (1,1,0)/√2). "
        "x−y 성분의 상대위상 δ = k(y−x)+π/2 를 따라 직선(δ=0,π)↔원(δ=±π/2) 궤적이 대각선 줄무늬로 반복, "
        "같은 편광으로 돌아오는 수직 거리 λ/√2. 세기 I = I0(1+cos δ)(x̂+ŷ): δ=0 에서는 (1,1) 방향 직선 진동이 "
        "I 와 평행 (T = 0), δ=π 에서는 (1,−1) 방향 직선 진동이지만 I = 0 → T = NaN, 그 근처는 T → 1.",
        _cfg(src, medium),
    )


def three_wave_hex(medium: Medium | None = None, freq: float = DEFAULT_FREQ) -> Preset:
    a = DEFAULT_AMPLITUDE
    src = [
        Source(direction=(np.cos(th), np.sin(th), 0.0), amplitude=a, frequency=freq)
        for th in np.deg2rad([0.0, 120.0, 240.0])
    ]
    return Preset(
        "hex",
        "같은 평면 120° 간격 3파 → 육각 간섭 무늬",
        "변위 진폭이 육각 대칭 무늬. 각 파의 세기 합은 0 이고 간섭 교차항만 남아 I 가 소용돌이 격자를 이룸. "
        "원점(및 그 격자점)은 Σk̂ = 0 이라 변위 마디 (L, e, T 모두 NaN) 이고 I 흐름은 안장점. "
        "소용돌이 중심은 (0, ±2λ/(3√3)) 등: 세 파의 위상이 0, ±2π/3 이 되어 '3상 모터'처럼 |u| = 1.5A 인 "
        "정확한 원 궤적 (e = 1) 이고, 그곳에서 I = 0 → T = NaN. 인접 소용돌이는 회전 방향이 반대. "
        "그 밖의 점은 대부분 타원 (0 < e < 1).",
        _cfg(src, medium),
    )


PRESETS = {p.key: p for p in (standing_wave(), orthogonal_quadrature(), three_wave_hex())}
