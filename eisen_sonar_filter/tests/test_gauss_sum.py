import numpy as np
import pytest

from eisen_sonar.core.eisenstein import EisensteinInt, hex_disk
from eisen_sonar.core.gauss_sum import GaussSumKernel

PRIMES = [EisensteinInt(1, 2), EisensteinInt(3, 1), EisensteinInt(4, 1), EisensteinInt(5, 2)]  # q = 3, 7, 13, 19


@pytest.mark.parametrize("prime", PRIMES)
def test_gauss_sum_magnitude(prime):
    K = GaussSumKernel(prime)
    q = K.q
    assert K.gauss_sum(0) == pytest.approx(-1.0)  # 자명 지표
    for k in range(1, q - 1):
        assert abs(K.gauss_sum(k)) == pytest.approx(np.sqrt(q), rel=1e-12)


@pytest.mark.parametrize("prime", PRIMES)
def test_quadratic_gauss_sum_classical_value(prime):
    K = GaussSumKernel(prime)
    q = K.q
    G = K.gauss_sum((q - 1) // 2)
    expected = np.sqrt(q) if q % 4 == 1 else 1j * np.sqrt(q)
    assert G == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("prime", PRIMES)
def test_character_orthogonality_and_multiplicativity(prime):
    K = GaussSumKernel(prime)
    q = K.q
    x = np.arange(1, q)
    for k in range(1, q - 1):
        assert abs(K.chi(x, k).sum()) < 1e-10
    assert abs(K.psi(np.arange(q)).sum()) < 1e-10
    xx, yy = np.meshgrid(x, x)
    assert np.allclose(K.chi(xx * yy), K.chi(xx) * K.chi(yy))


@pytest.mark.parametrize("prime", PRIMES)
def test_gauss_sum_links_additive_and_multiplicative_transforms(prime):
    """sum_t chi(t) psi(t x) = conj(chi(x)) G  (x != 0)."""
    K = GaussSumKernel(prime)
    q = K.q
    G = K.gauss_sum()
    t = np.arange(1, q)
    for x in range(1, q):
        lhs = np.sum(K.chi(t) * np.exp(2j * np.pi * t * x / q))
        assert lhs == pytest.approx(np.conj(K.chi(x)) * G, abs=1e-10)


def test_linear_kernel_normalizer_is_gauss_sum_for_7_cluster():
    K = GaussSumKernel(EisensteinInt(3, 1))
    d = hex_disk(1)
    res = K.residue(d[:, 0], d[:, 1])
    raw = K.chi(res) * K.psi(res)
    assert raw.sum() == pytest.approx(K.gauss_sum())
    w = K.kernel_weights(d)
    assert w.sum() == pytest.approx(1.0)
    # 실수부 잡음 분산 이득 = 2/7 (7-점 평균은 1/7) -- 해석적 예측
    assert np.sum(w.real ** 2) == pytest.approx(2 / 7)


@pytest.mark.parametrize("prime", [EisensteinInt(3, 1), EisensteinInt(4, 1), EisensteinInt(5, 2)])
def test_ring_characters_are_angular_harmonics(prime):
    K = GaussSumKernel(prime)
    rc = K.ring_characters((1, 2, 3))
    j = np.arange(6)
    for m, c in rc.items():
        assert np.allclose(c, np.exp(1j * np.pi * m * j / 3))
        assert abs(c.sum()) < 1e-12  # DC 상쇄


def test_ramified_prime_rejected_for_unit_harmonics():
    with pytest.raises(ValueError):
        GaussSumKernel(EisensteinInt(1, 2)).unit_harmonic(1)
