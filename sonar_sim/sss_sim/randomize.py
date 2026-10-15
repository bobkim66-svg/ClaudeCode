"""도메인 랜덤화: 학습 데이터셋/sim-to-real 용 무작위 장면 설정."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .config import ObjectSpec, SceneConfig

OBJECT_KINDS = ("cylinder", "box", "sphere")


def random_object(rng: np.random.Generator, x_range, y_abs_range, name: str = "") -> ObjectSpec:
    kind = str(rng.choice(OBJECT_KINDS))
    side = rng.choice([-1, 1])
    return ObjectSpec(
        kind=kind,
        x=float(rng.uniform(*x_range)),
        y=float(side * rng.uniform(*y_abs_range)),
        length=float(rng.uniform(1.0, 4.0)),
        width=float(rng.uniform(0.4, 1.5)),
        height=float(rng.uniform(0.3, 1.2)),
        heading_deg=float(rng.uniform(0, 180)),
        burial_m=float(rng.choice([0.0, rng.uniform(0.0, 0.3)])),
        reflectivity=float(rng.uniform(2.0, 10.0)),
        specular=float(rng.uniform(5.0, 60.0)),
        name=name or kind,
    )


def random_scene(seed: int, base: SceneConfig | None = None) -> SceneConfig:
    """seed 로 결정되는 무작위 장면 (같은 seed -> 같은 장면)."""
    rng = np.random.default_rng(seed)
    base = base or SceneConfig()
    R = base.sonar.max_range_m
    length = float(rng.uniform(30, 60))
    altitude = float(rng.uniform(0.1, 0.25) * R)
    traj = replace(
        base.trajectory,
        start_xy=(0.0, 0.0),
        end_xy=(length, 0.0),
        altitude_m=altitude,
        water_depth_m=float(altitude + rng.uniform(3.0, 40.0)),
        heave_amp_m=float(rng.uniform(0.0, 0.3)),
        roll_amp_deg=float(rng.uniform(0.0, 3.0)),
        pitch_amp_deg=float(rng.uniform(0.0, 1.5)),
        yaw_amp_deg=float(rng.uniform(0.0, 2.0)),
    )
    seabed = replace(
        base.seabed,
        roughness_rms_m=float(rng.uniform(0.02, 0.15)),
        roughness_beta=float(rng.uniform(3.0, 4.0)),
        ripple_amp_m=float(rng.uniform(0.0, 0.05)),
        ripple_wavelength_m=float(rng.uniform(0.5, 1.5)),
        ripple_direction_deg=float(rng.uniform(0, 180)),
        ripple_coverage=float(rng.uniform(0.0, 0.7)),
        mud_fraction=float(rng.uniform(0.0, 0.5)),
        gravel_fraction=float(rng.uniform(0.0, 0.3)),
        texture_shape=float(rng.uniform(2.0, 20.0)),
        n_rocks=int(rng.integers(0, 40)),
        regional_slope_deg=(float(rng.normal(0, 1.0)), float(rng.normal(0, 1.0))),
    )
    noise = replace(
        base.noise,
        snr_far_db=float(rng.uniform(5.0, 25.0)),
        multipath_db=float(rng.uniform(-30.0, -12.0)),
        volume_scatter_db=float(rng.uniform(-20.0, -8.0)),
        n_fish=int(rng.integers(0, 15)),
        interference_prob=float(rng.choice([0.0, 0.0, rng.uniform(0.005, 0.03)])),
        dropout_prob=float(rng.choice([0.0, 0.0, rng.uniform(0.002, 0.01)])),
    )
    sonar = replace(base.sonar, tilt_deg=float(rng.uniform(8.0, 22.0)),
                    tvg_exponent=float(rng.uniform(36.0, 42.0)))
    n_obj = int(rng.integers(0, 6))
    objects = [random_object(rng, (3.0, length - 3.0), (0.25 * R, 0.85 * R), name=f"obj{k + 1}") for k in range(n_obj)]
    return SceneConfig(name=f"random_{seed}", seed=seed, sonar=sonar, trajectory=traj, seabed=seabed,
                       objects=objects, noise=noise)
