"""Ground-truth tests for the Clenshaw evaluation kernels."""

import numpy as np
import pytest
from numpy.polynomial import chebyshev as npcheb

from chebyshevnd import clenshaw_1d, clenshaw_2d, clenshaw_3d, clenshaw_4d, clenshaw_nd
from chebyshevnd.clenshaw import partial_clenshaw_4d
from helpers import decaying_coeffs, direct_eval, scale_to_unit


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def test_clenshaw_1d_matches_chebval(rng):
    coeffs = decaying_coeffs(rng, (20,))
    a, b = -2.0, 1.5
    for x in np.linspace(a, b, 9):
        xt = scale_to_unit(x, a, b)
        assert clenshaw_1d(coeffs, x, len(coeffs), a, b) == pytest.approx(
            npcheb.chebval(xt, coeffs), rel=1e-12, abs=1e-14
        )


def test_clenshaw_1d_k_max_is_a_count(rng):
    coeffs = decaying_coeffs(rng, (12,))
    a, b = 0.0, 3.0
    x = 1.234
    xt = scale_to_unit(x, a, b)
    for m in [1, 2, 5, 11, 12]:
        assert clenshaw_1d(coeffs, x, m, a, b) == pytest.approx(
            npcheb.chebval(xt, coeffs[:m]), rel=1e-12, abs=1e-14
        )


def test_clenshaw_1d_k_max_zero_returns_zero(rng):
    coeffs = decaying_coeffs(rng, (8,))
    assert clenshaw_1d(coeffs, 0.3, 0) == 0.0


def test_clenshaw_2d_matches_chebval2d(rng):
    coeffs = decaying_coeffs(rng, (6, 7))
    (ax, bx), (ay, by) = (-1.0, 2.0), (0.5, 4.0)
    for x in np.linspace(ax, bx, 4):
        for y in np.linspace(ay, by, 4):
            tx, ty = scale_to_unit(x, ax, bx), scale_to_unit(y, ay, by)
            assert clenshaw_2d(coeffs, x, y, ax=ax, bx=bx, ay=ay, by=by) == pytest.approx(
                npcheb.chebval2d(tx, ty, coeffs), rel=1e-12, abs=1e-14
            )


def test_clenshaw_3d_matches_direct_sum(rng):
    coeffs = decaying_coeffs(rng, (5, 6, 4))
    doms = [(-1.0, 1.0), (0.0, 2.0), (-3.0, -1.0)]
    for point in [(-0.5, 0.3, -2.7), (0.9, 1.9, -1.1), (0.0, 1.0, -2.0)]:
        ts = [scale_to_unit(p, a, b) for p, (a, b) in zip(point, doms)]
        expected = direct_eval(coeffs, ts)
        got = clenshaw_3d(
            coeffs, *point,
            ax=doms[0][0], bx=doms[0][1],
            ay=doms[1][0], by=doms[1][1],
            az=doms[2][0], bz=doms[2][1],
        )
        assert got == pytest.approx(expected, rel=1e-12, abs=1e-14)


def test_clenshaw_3d_kmax_trims_block(rng):
    coeffs = decaying_coeffs(rng, (6, 6, 6))
    kmax = (4, 3, 5)
    point = (0.2, -0.4, 0.7)
    expected = direct_eval(coeffs[: kmax[0], : kmax[1], : kmax[2]], point)
    got = clenshaw_3d(coeffs, *point, kmax=kmax)
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-14)


def test_clenshaw_4d_matches_direct_sum(rng):
    coeffs = decaying_coeffs(rng, (4, 5, 3, 4))
    doms = [(0.0, 1.0), (-2.0, 0.0), (1.0, 3.0), (-1.0, 1.0)]
    for point in [(0.3, -1.2, 2.4, 0.5), (0.9, -0.1, 1.2, -0.8)]:
        ts = [scale_to_unit(p, a, b) for p, (a, b) in zip(point, doms)]
        expected = direct_eval(coeffs, ts)
        got = clenshaw_4d(
            coeffs, *point,
            ax=doms[0][0], bx=doms[0][1],
            ay=doms[1][0], by=doms[1][1],
            az=doms[2][0], bz=doms[2][1],
            aw=doms[3][0], bw=doms[3][1],
        )
        assert got == pytest.approx(expected, rel=1e-12, abs=1e-14)


def test_clenshaw_nd_matches_direct_sum(rng):
    for shape in [(9,), (5, 6), (4, 5, 3), (3, 4, 3, 5)]:
        coeffs = decaying_coeffs(rng, shape)
        ts = rng.uniform(-1.0, 1.0, size=len(shape))
        expected = direct_eval(coeffs, ts)
        assert clenshaw_nd(coeffs, ts) == pytest.approx(expected, rel=1e-12, abs=1e-14)


def test_clenshaw_nd_with_domains(rng):
    coeffs = decaying_coeffs(rng, (4, 5, 3))
    doms = [(0.0, 2.0), (-3.0, 1.0), (1.0, 1.5)]
    a = [lo for lo, _ in doms]
    b = [hi for _, hi in doms]
    point = np.array([1.3, -0.7, 1.1])
    ts = [scale_to_unit(p, lo, hi) for p, (lo, hi) in zip(point, doms)]
    expected = direct_eval(coeffs, ts)
    assert clenshaw_nd(coeffs, point, a=a, b=b) == pytest.approx(expected, rel=1e-12, abs=1e-14)


def test_clenshaw_nd_kmax_trims(rng):
    coeffs = decaying_coeffs(rng, (6, 6))
    ts = np.array([0.3, -0.2])
    expected = direct_eval(coeffs[:4, :3], ts)
    assert clenshaw_nd(coeffs, ts, kmax=(4, 3)) == pytest.approx(expected, rel=1e-12, abs=1e-14)


def test_partial_clenshaw_4d_consistent_with_full(rng):
    coeffs = decaying_coeffs(rng, (5, 4, 5, 4))
    doms = [(0.0, 2.0), (-1.0, 1.0), (1.0, 4.0), (-2.0, -1.0)]
    x = 0.7
    collapsed = partial_clenshaw_4d(coeffs, x, a=doms[0][0], b=doms[0][1])
    for point in [(-0.3, 2.2, -1.5), (0.8, 3.7, -1.9)]:
        full = clenshaw_4d(
            coeffs, x, *point,
            ax=doms[0][0], bx=doms[0][1],
            ay=doms[1][0], by=doms[1][1],
            az=doms[2][0], bz=doms[2][1],
            aw=doms[3][0], bw=doms[3][1],
        )
        partial = clenshaw_3d(
            collapsed, *point,
            ax=doms[1][0], bx=doms[1][1],
            ay=doms[2][0], by=doms[2][1],
            az=doms[3][0], bz=doms[3][1],
        )
        assert partial == pytest.approx(full, rel=1e-12, abs=1e-14)
