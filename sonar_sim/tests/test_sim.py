from dataclasses import replace

import numpy as np
import pytest

from sss_sim.config import NoiseConfig, ObjectSpec, SceneConfig, SeabedConfig, SonarConfig, TrajectoryConfig
from sss_sim.noise import along_sigma_pings, apply_psf
from sss_sim.randomize import random_scene
from sss_sim.sonar import simulate

ALT = 4.0


def flat_cfg(**kw) -> SceneConfig:
    """작고 빠른 평탄 해저 장면: 운동/잡음/아티팩트 모두 끔."""
    cfg = SceneConfig(
        name="flat",
        sonar=SonarConfig(max_range_m=20.0, n_bins=200, ping_rate_hz=20.0),
        trajectory=TrajectoryConfig(start_xy=(0, 0), end_xy=(10, 0), speed_mps=2.0, altitude_m=ALT, water_depth_m=10.0,
                                    heave_amp_m=0, roll_amp_deg=0, pitch_amp_deg=0, yaw_amp_deg=0),
        seabed=SeabedConfig(grid_resolution_m=0.05, roughness_rms_m=0, ripple_amp_m=0, mud_fraction=0, gravel_fraction=0,
                            texture_shape=None, n_rocks=0),
        noise=NoiseConfig(speckle=False, snr_far_db=None, surface_return_db=None, multipath_db=None,
                          volume_scatter_db=None, n_fish=0),
    )
    return replace(cfg, **kw)


def test_config_json_roundtrip(tmp_path):
    cfg = flat_cfg(objects=[ObjectSpec("box", 1, 2)])
    p = tmp_path / "s.json"
    cfg.save(p)
    assert SceneConfig.load(p) == cfg
    with pytest.raises(KeyError):
        SceneConfig.from_dict({"sonarr": {}})  # 오타는 조용히 무시하지 않는다


def test_validation_range_ambiguity():
    cfg = flat_cfg(sonar=SonarConfig(max_range_m=40.0, ping_rate_hz=30.0))  # c/(2R) = 18.75 Hz
    with pytest.raises(ValueError, match="거리 모호성"):
        cfg.validate()


def test_first_bottom_return_and_flat_level():
    rec = simulate(flat_cfg())
    nb, dr = rec.config.sonar.n_bins, rec.config.sonar.bin_size_m
    stbd = rec.clean[:, nb:]
    prof = stbd.mean(axis=0)
    first = int(np.argmax(prof > 0.5))
    assert abs(first * dr - ALT) < 2 * rec.config.sonar.range_resolution_m + dr  # 첫 해저 반사 = 고도
    assert np.all(prof[: int(ALT / dr) - 3] < 1e-6)  # 그 전은 수주 (끈 상태 = 0)
    far = prof[int(1.5 * ALT / dr): -3]
    assert np.allclose(far, 1.0, atol=0.02)  # 빔패턴 보정 후 평탄 모래 = 1


def test_port_starboard_symmetric_on_flat_floor():
    rec = simulate(flat_cfg())
    nb = rec.config.sonar.n_bins
    assert np.allclose(rec.clean[:, :nb][:, ::-1], rec.clean[:, nb:], rtol=1e-6, atol=1e-12)


def test_shadow_length_matches_geometry():
    """평탄 해저 위 상자 (높이 ht, 먼쪽 모서리 y_far): 그림자 끝 지상거리 = y_far * H / (H - ht)."""
    ht, yc, w = 1.0, 9.0, 1.0
    cfg = flat_cfg(objects=[ObjectSpec("box", 5.0, -yc, length=3.0, width=w, height=ht, heading_deg=0.0)])  # 우현 = -y
    rec = simulate(cfg)
    nb, dr = cfg.sonar.n_bins, cfg.sonar.bin_size_m
    ping = int(np.argmin(np.abs(rec.along_m - 5.0)))
    sh = rec.labels["shadow"][ping, nb:]
    y_far = yc + w / 2
    r_start = np.hypot(y_far, ALT - ht)
    r_end = np.hypot(y_far * ALT / (ALT - ht), ALT)
    idx = np.flatnonzero(sh)
    assert len(idx) > 0
    assert abs(idx[0] * dr - r_start) < 3 * dr
    assert abs((idx[-1] + 1) * dr - r_end) < 3 * dr
    assert rec.labels["object_id"][ping, nb:].max() == 1
    assert rec.labels["object_shadow_id"][ping, nb:].max() == 1


@pytest.mark.parametrize("L", [1, 4])
def test_speckle_enl_equals_looks(L):
    cfg = flat_cfg(noise=replace(flat_cfg().noise, speckle=True, n_looks=L))
    rec = simulate(cfg)
    nb, dr = cfg.sonar.n_bins, cfg.sonar.bin_size_m
    sl = (slice(5, -5), slice(nb + int(1.5 * ALT / dr), 2 * nb - 5))
    ratio = rec.intensity[sl] / rec.clean[sl]
    assert ratio.mean() == pytest.approx(1.0, abs=0.03)
    assert ratio.mean() ** 2 / ratio.var() == pytest.approx(L, rel=0.12)


def test_along_track_resolution_grows_with_range():
    s = SonarConfig(max_range_m=40.0, n_bins=400, horizontal_beamwidth_deg=1.0)
    spacing = 0.1
    sig = along_sigma_pings(s, spacing)
    r = (np.arange(400) + 0.5) * 0.1
    expect = r * np.radians(1.0) / 2.355 / spacing
    big = expect > s.min_along_sigma_pings
    assert np.allclose(sig[big], expect[big])
    # 점 응답의 실제 폭 (강도 PSF) 이 거리에 비례
    img = np.zeros((201, 400))
    img[100, :] = 1.0
    out = apply_psf(img, sig, 0.5, power=2)
    t = np.arange(201) - 100
    widths = {}
    for b in (250, 399):  # sigma >= 1.5 핑: 정수 격자 이산화 영향이 작은 구간
        p = out[:, b] / out[:, b].sum()
        widths[b] = np.sqrt(np.sum(p * t ** 2))
        sig_q = np.round(sig[b] / 0.1) * 0.1  # apply_psf 는 sigma 를 0.1 핑 단위로 묶는다
        assert widths[b] == pytest.approx(sig_q / np.sqrt(2), rel=0.05)  # g^2 의 표준편차 = sigma/sqrt2
    assert widths[399] / widths[250] == pytest.approx(r[399] / r[250], rel=0.05)  # 폭 ~ 거리


def test_multipath_ghost_at_water_depth():
    """수직 아래 해저점의 수면 1회 반사 경로 등가거리 = (H + (2D - H)) / 2 = D."""
    D = 10.0
    base = simulate(flat_cfg()).clean
    cfg = flat_cfg(noise=replace(flat_cfg().noise, multipath_db=0.0))
    mp = simulate(cfg).clean
    nb, dr = cfg.sonar.n_bins, cfg.sonar.bin_size_m
    ratio = mp[:, nb:].mean(0) / np.maximum(base[:, nb:].mean(0), 1e-12)
    i_D = int(D / dr)
    assert np.all(ratio[int(1.5 * ALT / dr): i_D - 4] < 1.001)  # D 이전엔 유령 없음
    assert ratio[i_D + 2: i_D + 20].max() > 1.01  # D 부근부터 유령 에너지


def test_surface_return_before_bottom_in_shallow_tow():
    """소나가 수면 가까이 (수면까지 2 m) 이면 수면 반사가 r = 2 m 에서 첫 해저 반사(4 m) 보다 먼저 나온다."""
    cfg = flat_cfg(trajectory=replace(flat_cfg().trajectory, water_depth_m=ALT + 2.0),
                   noise=replace(flat_cfg().noise, surface_return_db=10.0))
    rec = simulate(cfg)
    nb, dr = cfg.sonar.n_bins, cfg.sonar.bin_size_m
    prof = rec.clean[:, nb:].mean(0)
    assert prof[int(2.0 / dr) + 1: int(ALT / dr) - 3].max() > 1e-3
    assert np.all(prof[: int(2.0 / dr) - 3] < 1e-9)


def test_reproducible_and_seed_dependent():
    cfg = replace(SceneConfig.load(__import__("pathlib").Path(__file__).parents[1] / "scenes" / "default.json"),
                  sonar=SonarConfig(max_range_m=20.0, n_bins=128),
                  trajectory=replace(TrajectoryConfig(), end_xy=(8.0, 0.0), altitude_m=4.0),
                  seabed=replace(SeabedConfig(), grid_resolution_m=0.1))
    a, b = simulate(cfg), simulate(cfg)
    assert np.array_equal(a.intensity, b.intensity)
    c = simulate(replace(cfg, seed=cfg.seed + 1))
    assert not np.array_equal(a.intensity, c.intensity)


def test_random_scenes_valid_and_run():
    for s in range(5):
        random_scene(s).validate()
    cfg = random_scene(3)
    cfg = replace(cfg, sonar=replace(cfg.sonar, n_bins=128),
                  trajectory=replace(cfg.trajectory, end_xy=(8.0, 0.0)),
                  seabed=replace(cfg.seabed, grid_resolution_m=0.1))
    rec = simulate(cfg)
    assert rec.intensity.shape[1] == 256 and np.all(np.isfinite(rec.intensity))


def test_motion_and_artifacts_do_not_break_invariants():
    cfg = flat_cfg(trajectory=replace(flat_cfg().trajectory, roll_amp_deg=3, pitch_amp_deg=2, yaw_amp_deg=2, heave_amp_m=0.3),
                   noise=NoiseConfig(interference_prob=0.05, dropout_prob=0.05))
    rec = simulate(cfg)
    assert np.all(rec.intensity >= 0) and np.all(np.isfinite(rec.intensity))
    assert (rec.intensity.max(axis=1) == 0).any()  # 핑 누락 행 존재
