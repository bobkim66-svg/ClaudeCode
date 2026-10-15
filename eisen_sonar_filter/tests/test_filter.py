import numpy as np
import pytest

from eisen_sonar.filters.baseline_filters import frost_filter, kuan_filter, lee_filter
from eisen_sonar.filters.gauss_filter import EisensteinGaussFilter
from eisen_sonar.utils.data_loader import SonarSimConfig, make_synthetic_sonar
from eisen_sonar.utils.metrics import enl

MODES = ["adaptive", "linear", "resample_only"]


@pytest.mark.parametrize("mode", MODES)
def test_constant_image_preserved(mode):
    img = np.full((30, 41), 3.7)
    out = EisensteinGaussFilter(mode=mode).denoise(img)
    assert out.shape == img.shape
    assert np.allclose(out, 3.7, atol=1e-10)


@pytest.mark.parametrize("f", [lambda I: lee_filter(I), lambda I: kuan_filter(I), lambda I: frost_filter(I)])
def test_baselines_preserve_constant(f):
    assert np.allclose(f(np.full((20, 20), 2.0)), 2.0)


def test_input_validation():
    F = EisensteinGaussFilter()
    with pytest.raises(ValueError):
        F.denoise(np.ones(10))
    with pytest.raises(ValueError):
        F.denoise(-np.ones((10, 10)))
    with pytest.raises(ValueError):
        F.denoise(np.full((10, 10), np.nan))
    with pytest.raises(ValueError):
        EisensteinGaussFilter(window_size=4)
    with pytest.raises(ValueError):
        EisensteinGaussFilter(prime_a=1, prime_b=2)  # q=3: 적응형 불가


def test_adaptive_invariant_to_prime_choice():
    rng = np.random.default_rng(0)
    img = rng.exponential(size=(48, 64))
    a = EisensteinGaussFilter(prime_a=3, prime_b=1).denoise(img)  # q=7
    b = EisensteinGaussFilter(prime_a=4, prime_b=1).denoise(img)  # q=13
    assert np.allclose(a, b, atol=1e-10)


def test_speckle_reduction_on_pure_speckle_and_mean_preserved():
    rng = np.random.default_rng(1)
    img = 5.0 * rng.exponential(size=(96, 96))  # L=1, 백색 스페클
    m = np.zeros(img.shape, bool)
    m[8:-8, 8:-8] = True
    out = EisensteinGaussFilter().denoise(img)
    assert enl(out, m) > 5 * enl(img, m)
    assert out[m].mean() == pytest.approx(img[m].mean(), rel=0.03)


def test_step_edge_selects_parallel_axis_and_keeps_contrast():
    """수직 계단 경계 (x 방향 기울기) -> 접선은 y 방향. 60/120 deg 축이 선택되어야 한다."""
    rng = np.random.default_rng(2)
    clean = np.ones((64, 64))
    clean[:, 32:] = 8.0
    img = clean * rng.gamma(4, 1 / 4, size=clean.shape)
    F = EisensteinGaussFilter()
    out = F.denoise(img)
    d = F.last_diagnostics
    x, y = d["grid"].lattice_xy()
    near_edge = (np.abs(x - 31.5) < 1.0) & d["inside"] & (y > 4) & (y < 59)
    assert d["k"][near_edge].mean() > 0.8  # 경계를 구조로 검출
    assert np.isin(d["axis"][near_edge & (d["k"] > 0.5)], [1, 2]).mean() > 0.9
    # 경계 양쪽 1화소 거리의 대비가 원판 평균(비적응)보다 잘 보존되어야 함
    lin = EisensteinGaussFilter(mode="resample_only").denoise(clean)
    contrast_out = out[8:-8, 33].mean() / out[8:-8, 30].mean()
    assert contrast_out > 0.7 * (lin[8:-8, 33].mean() / lin[8:-8, 30].mean())


def test_reproducible_with_seed():
    cfg = SonarSimConfig(n_pings=64, n_bins=128, max_range_m=20.0, altitude_m=5.0,
                         homogeneous_patch=(1.0, 4.0, 10.0, 18.0), targets=[], mud_patch=(0, 0, 0, 0))
    a = make_synthetic_sonar(cfg, seed=3)
    b = make_synthetic_sonar(cfg, seed=3)
    assert np.array_equal(a.intensity, b.intensity)
    F = EisensteinGaussFilter()
    assert np.array_equal(F.denoise(a.intensity), F.denoise(b.intensity))


@pytest.mark.parametrize("L", [1, 4])
def test_simulator_enl_matches_looks(L):
    s = make_synthetic_sonar(SonarSimConfig(n_looks=L), seed=0)
    m = s.masks["homogeneous"]
    assert s.clean[m].std() / s.clean[m].mean() < 1e-6  # 정답은 동질 영역에서 상수
    assert enl(s.intensity, m) == pytest.approx(L, rel=0.15)


def test_shadow_length_flat_floor():
    """평탄 해저 위 상자 표적의 그림자 길이 = h_t * y_t / (H - h_t) (지상거리)."""
    from eisen_sonar.utils.data_loader import TargetSpec

    H, ht, yt, w = 8.0, 1.0, 20.0, 1.0
    cfg = SonarSimConfig(altitude_m=H, ripple_amp_m=0.0, texture_shape=None, n_rocks=0, mud_patch=(0, 0, 0, 0),
                         targets=[TargetSpec("box", 10.0, yt, 3.0, w, ht)])
    s = make_synthetic_sonar(cfg, seed=0)
    ping = int(10.0 / cfg.ping_spacing_m)
    row = s.clean[ping]
    y_far = yt + w / 2  # 상자 먼쪽 모서리
    r_start = np.hypot(y_far, H - ht)  # 그림자 시작 경사거리 (상자 윗면 끝)
    y_end = y_far * H / (H - ht)  # 평탄 해저에서 그림자 끝 지상거리
    r_end = np.hypot(y_end, H)
    dr = cfg.max_range_m / cfg.n_bins
    i0, i1 = int(r_start / dr) + 3, int(r_end / dr) - 3
    assert row[i0:i1].max() < 1e-6 * row.max()  # 예측 구간 안은 완전히 그림자
    assert row[int(r_end / dr) + 4] > 0.3  # 그림자 끝 바로 뒤는 다시 밝음
