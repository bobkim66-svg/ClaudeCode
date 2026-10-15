"""항로와 차체 자세 시계열."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .config import SonarConfig, TrajectoryConfig


@dataclass
class Navigation:
    """핑마다의 소나 위치/자세. 모든 배열 shape (n_pings,)."""

    t: np.ndarray  # [s]
    x: np.ndarray  # [m] ENU
    y: np.ndarray
    z: np.ndarray  # 해저 기준면 위 높이 [m]
    heading: np.ndarray  # [rad], +x 에서 반시계
    roll: np.ndarray  # [rad], + = 우현 아래
    pitch: np.ndarray  # [rad], + = 기수 위

    @property
    def n(self) -> int:
        return len(self.t)

    def forward(self):
        return np.cos(self.heading), np.sin(self.heading)

    def starboard(self):
        return np.sin(self.heading), -np.cos(self.heading)


def _motion(rng: np.random.Generator, t: np.ndarray, amp: float, period: float) -> np.ndarray:
    """주기 성분 (무작위 위상) + 같은 크기의 저주파 랜덤 성분. 진폭 0 이면 0."""
    if amp == 0 or len(t) == 0:
        return np.zeros_like(t)
    periodic = np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    slow = gaussian_filter1d(rng.standard_normal(len(t)), sigma=max(period / dt / 2, 1.0), mode="nearest")
    slow /= max(np.abs(slow).max(), 1e-12)
    return amp * (0.7 * periodic + 0.3 * slow)


def make_navigation(traj: TrajectoryConfig, sonar: SonarConfig, rng: np.random.Generator) -> Navigation:
    """직선 항로를 핑 주기로 표본화하고 운동을 더한다."""
    p0, p1 = np.asarray(traj.start_xy, float), np.asarray(traj.end_xy, float)
    length = float(np.linalg.norm(p1 - p0))
    if length <= 0:
        raise ValueError("start_xy 와 end_xy 가 같다")
    dt = 1.0 / sonar.ping_rate_hz
    n = int(length / (traj.speed_mps * dt)) + 1
    t = np.arange(n) * dt
    u = (p1 - p0) / length
    s = traj.speed_mps * t
    psi0 = np.arctan2(u[1], u[0])
    yaw = _motion(rng, t, np.radians(traj.yaw_amp_deg), traj.yaw_period_s)
    return Navigation(
        t=t,
        x=p0[0] + u[0] * s,
        y=p0[1] + u[1] * s,
        z=traj.altitude_m + _motion(rng, t, traj.heave_amp_m, traj.heave_period_s),
        heading=psi0 + yaw,
        roll=_motion(rng, t, np.radians(traj.roll_amp_deg), traj.roll_period_s),
        pitch=_motion(rng, t, np.radians(traj.pitch_amp_deg), traj.pitch_period_s),
    )
