import numpy as np
import pytest

from polarsim.config import Medium

C = 343.0
F = 1000.0
LAM = C / F
A = 1e-6


@pytest.fixture
def medium():
    return Medium()


@pytest.fixture
def rng():
    """테스트용 난수는 항상 시드 고정."""
    return np.random.default_rng(12345)


def one_period(f=F, n=64):
    """끝점 제외 한 주기 표본 (Nt,) [s]."""
    return np.arange(n) / (n * f)


def random_unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def random_rotation(rng):
    """균일 무작위 회전 행렬 (det = +1)."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q
