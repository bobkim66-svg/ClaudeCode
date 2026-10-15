import numpy as np
import pytest

from eisen_sonar.core.eisenstein import (
    UNIT_OFFSETS,
    EisensteinInt,
    cartesian_to_eisenstein,
    eisenstein_to_cartesian,
    hex_disk,
    hex_to_cartesian,
    omega_residue,
    resample_to_hex_grid,
)

W = complex(-0.5, np.sqrt(3) / 2)


def _rand_ints(rng, n=200):
    return [EisensteinInt(*map(int, rng.integers(-20, 21, 2))) for _ in range(n)]


def test_arithmetic_matches_complex():
    rng = np.random.default_rng(0)
    zs, ws = _rand_ints(rng), _rand_ints(rng)
    for z, w in zip(zs, ws):
        assert np.isclose((z * w).to_complex(), z.to_complex() * w.to_complex())
        assert np.isclose((z + w).to_complex(), z.to_complex() + w.to_complex())
        assert np.isclose(z.conj().to_complex(), np.conj(z.to_complex()))
        assert z.norm() == pytest.approx(abs(z.to_complex()) ** 2)
        assert (z * w).norm() == z.norm() * w.norm()  # 노름의 곱셈성


def test_euclidean_division():
    rng = np.random.default_rng(1)
    for z, w in zip(_rand_ints(rng), _rand_ints(rng)):
        if w.norm() == 0:
            continue
        q, r = divmod(z, w)
        assert q * w + r == z
        assert r.norm() < w.norm()


def test_example_primes_and_omega_residue():
    for (a, b), q in [((1, 2), 3), ((2, 3), 7), ((3, 1), 7), ((4, 1), 13)]:
        r, qq = omega_residue(EisensteinInt(a, b))
        assert qq == q and (r * r + r + 1) % q == 0
    with pytest.raises(ValueError):
        omega_residue(EisensteinInt(2, 0))  # N=4, 관성소수 2 -> 소수체 아님


def test_seven_cluster_is_complete_residue_system_mod_3_plus_w():
    r, q = omega_residue(EisensteinInt(3, 1))
    res = sorted(int((a + b * r) % q) for a, b in hex_disk(1))
    assert res == list(range(7))


def test_units_are_sixth_roots_in_angle_order():
    for j, (a, b) in enumerate(UNIT_OFFSETS):
        z = complex(a + b * W.real, b * W.imag)
        assert np.isclose(z, np.exp(1j * np.pi * j / 3))


def test_hex_disk_counts():
    assert [len(hex_disk(R)) for R in range(4)] == [1, 7, 19, 37]


def test_coordinate_roundtrip():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(2, 100)) * 50
    a, b = cartesian_to_eisenstein(x, y)
    x2, y2 = eisenstein_to_cartesian(a, b)
    assert np.allclose(x, x2, atol=1e-12) and np.allclose(y, y2, atol=1e-12)


@pytest.mark.parametrize("spacing", [0.7, 1.0, 1.6])
def test_resample_roundtrip_exact_for_linear_field(spacing):
    yy, xx = np.mgrid[0:40, 0:57].astype(float)
    lin = 0.3 * xx - 0.7 * yy + 2.0
    hexv, grid = resample_to_hex_grid(lin, spacing)
    back = hex_to_cartesian(hexv, grid)
    # 경계는 격자점이 영상 밖이라 최근접 복제 -> 비선형. 안쪽만 정확성 검사
    m = int(np.ceil(2 * spacing)) + 1
    assert np.abs(back - lin)[m:-m, m:-m].max() < 1e-10
