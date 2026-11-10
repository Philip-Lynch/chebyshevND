"""Tests for fused evaluation: raw tables, eval_into / eval_grad_into, value_and_grad (Step 4)."""

import numpy as np
import pytest
from numba import njit
from numpy.polynomial import chebyshev as npcheb

from chebyshevnd import (
    ChebyshevInterpolantND,
    OutOfDomainError,
    RawTable,
    Workspace,
    eval_grad_into,
    eval_into,
)
from chebyshevnd.kernels import chebyshev_basis_and_derivative
from helpers import sample_on_grid

DOMAINS = [(0.0, 2.0), (-1.0, 3.0), (1.0, 4.0)]
NS = (22, 16, 12)
FUNCS = [
    lambda x, y, z: np.sin(x) * np.cos(y) * np.exp(-z),
    lambda x, y, z: np.exp(1j * x) * np.cos(0.5 * y) / z,
    lambda x, y, z: 1e-3 * (x * y + z**2),
]


def build(dtype=float, multi=True, domains=DOMAINS, ns=NS, funcs=FUNCS):
    axes = sample_on_grid(lambda *x: x[0], domains, ns)[0]
    grids = np.meshgrid(*axes, indexing="ij")
    vals = np.stack([f(*grids) for f in funcs], axis=-1)
    vals = vals.real if dtype is float else vals.astype(dtype)
    if multi:
        names = ["x", "y", "z"] if len(domains) == 3 else None
        return ChebyshevInterpolantND(vals, domains, axis_names=names,
                                      output_names=[f"f{i}" for i in range(len(funcs))])
    return ChebyshevInterpolantND(vals[..., 0], domains)


def random_points(n, domains=DOMAINS, seed=3):
    rng = np.random.default_rng(seed)
    lo, hi = np.array(domains).T
    return rng.uniform(lo, hi, size=(n, len(domains)))


def corner_points(domains=DOMAINS):
    """Every corner of the box, where t = +-1 on each axis."""
    grids = np.meshgrid(*[list(dom) for dom in domains], indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


def reference_gradient(nd, pts):
    """Gradient from Step 3's coefficient-space derivatives: shape (n, d[, F])."""
    columns = []
    for ax in range(nd.ndim):
        orders = [0] * nd.ndim
        orders[ax] = 1
        columns.append(nd.derivative(orders).evaluate_batch(pts))
    return np.stack(columns, axis=1)


def assert_close_relative(got, expected, rtol):
    """Max error relative to the largest magnitude, per output."""
    got, expected = np.asarray(got), np.asarray(expected)
    scale = np.abs(expected).max()
    np.testing.assert_allclose(got, expected, rtol=0, atol=rtol * max(scale, 1e-300))


# ------------------------------------------------------------- basis values


def test_basis_and_derivative_match_numpy():
    n = 15
    basis, deriv = np.empty(n), np.empty(n)
    for t in [-1.0, -0.73, 0.0, 0.41, 1.0]:
        chebyshev_basis_and_derivative(n, t, basis, deriv)
        for k in range(n):
            e = np.zeros(k + 1)
            e[k] = 1.0
            assert basis[k] == pytest.approx(npcheb.chebval(t, e), abs=1e-14)
            assert deriv[k] == pytest.approx(npcheb.chebval(t, npcheb.chebder(e)), abs=1e-12)


# ------------------------------------------------------- single-point fused


@pytest.mark.parametrize("dtype", [float, complex])
@pytest.mark.parametrize("multi", [False, True])
def test_value_and_grad_matches_derivative_tensors(dtype, multi):
    nd = build(dtype, multi)
    pts = np.concatenate([random_points(40), corner_points()])
    expected_grad = reference_gradient(nd, pts)
    expected_val = nd.evaluate_batch(pts)
    for p, ev, eg in zip(pts, expected_val, expected_grad):
        value, grad = nd.value_and_grad(*p)
        assert grad.shape == ((3,) if not multi else (3, 3))
        assert grad.dtype == nd.dtype
        if multi:
            assert value.shape == (3,)
        else:
            assert isinstance(value, complex if dtype is complex else float)
        assert_close_relative(value, ev, 1e-14)
        assert_close_relative(grad, eg, 1e-13)


def test_value_and_grad_against_analytic_function():
    nd = build(multi=False, ns=(30, 26, 22))
    for p in random_points(10):
        x, y, z = p
        _, grad = nd.value_and_grad(*p)
        exact = [np.cos(x) * np.cos(y) * np.exp(-z),
                 -np.sin(x) * np.sin(y) * np.exp(-z),
                 -np.sin(x) * np.cos(y) * np.exp(-z)]
        np.testing.assert_allclose(grad, exact, atol=1e-11)


@pytest.mark.parametrize("ns, domains", [
    ((17,), [(-2.0, 1.0)]),
    ((14, 9), [(0.0, 1.0), (2.0, 5.0)]),
    ((8, 7, 6, 5, 4), [(0.0, 1.0), (-1.0, 2.0), (0.5, 1.5), (-2.0, -1.0), (0.0, 0.5)]),
    ((1, 6), [(0.0, 1.0), (-1.0, 1.0)]),
])
def test_value_and_grad_other_dimensions(ns, domains):
    def f(*xs):
        out = np.ones_like(xs[0])
        for i, x in enumerate(xs):
            out = out * np.cos(0.3 * (i + 1) * x + 0.1)
        return out

    nd = build(multi=True, domains=domains, ns=ns, funcs=[f, lambda *xs: f(*xs) ** 2])
    pts = np.concatenate([random_points(15, domains), corner_points(domains)])
    expected = reference_gradient(nd, pts)
    for p, eg in zip(pts, expected):
        _, grad = nd.value_and_grad(*p)
        assert_close_relative(grad, eg, 1e-13)


def test_value_and_grad_after_optimize_and_precontract():
    nd = build()
    nd.optimize(1e-8)
    assert nd.kmax != nd.grid_shape
    child = nd.precontract({"y": 0.7})
    for interp in (nd, child):
        pts = random_points(20)[:, :interp.ndim]
        if interp is child:
            pts = random_points(20)[:, [0, 2]]
        expected = reference_gradient(interp, pts)
        for p, eg in zip(pts, expected):
            assert_close_relative(interp.value_and_grad(*p)[1], eg, 1e-13)


def test_value_and_grad_raises_outside_domain():
    nd = build()
    with pytest.raises(OutOfDomainError, match="y = 3.5"):
        nd.value_and_grad(1.0, 3.5, 2.0)
    with pytest.raises(TypeError, match="value_and_grad"):
        nd.value_and_grad(1.0, 2.0)


# ----------------------------------------------------------------- raw tables


def test_raw_table_fields_and_read_only():
    nd = build()
    table = nd.raw()
    assert isinstance(table, RawTable)
    assert table.ndim == 3 and table.n_outputs == 3 and table.dtype == np.float64
    np.testing.assert_array_equal(table.shape, nd.kmax)
    np.testing.assert_array_equal(table.bounds, np.array(DOMAINS).T)
    assert table.tol == nd.domain_tol
    np.testing.assert_array_equal(table.coeffs, nd.get_coefficients().ravel())
    for array in (table.coeffs, table.shape, table.bounds):
        with pytest.raises(ValueError):
            array[0] = 0
    ws = table.workspace()
    assert isinstance(ws, Workspace)


def test_raw_table_is_a_snapshot():
    nd = build()
    table = nd.raw()
    p = random_points(1)[0]
    before = nd.evaluate(*p)
    nd.optimize(1e-4)
    out = np.empty(3)
    eval_into(*table, p, *table.workspace(), out)
    np.testing.assert_allclose(out, before, rtol=1e-14)
    assert tuple(nd.raw().shape) == nd.kmax


@pytest.mark.parametrize("dtype", [float, complex])
def test_eval_into_and_eval_grad_into_match_methods(dtype):
    nd = build(dtype)
    table = nd.raw()
    ws = table.workspace()
    out = np.empty(table.n_outputs, dtype=table.dtype)
    out_grad = np.empty((table.ndim + 1, table.n_outputs), dtype=table.dtype)
    for p in random_points(20):
        assert eval_into(*table, p, *ws, out) == 0
        np.testing.assert_allclose(out, nd.evaluate(*p), rtol=1e-14)
        assert eval_grad_into(*table, p, *ws, out_grad) == 0
        value, grad = nd.value_and_grad(*p)
        np.testing.assert_allclose(out_grad[0], value, rtol=1e-14)
        np.testing.assert_allclose(out_grad[1:], grad, rtol=1e-14)


@njit
def _user_rhs(x, coeffs, shape, bounds, tol, scratch, work, out, out_grad):
    """A user's jitted function calling both entry points."""
    s1 = eval_into(coeffs, shape, bounds, tol, x, scratch, work, out)
    s2 = eval_grad_into(coeffs, shape, bounds, tol, x, scratch, work, out_grad)
    return s1 | s2, out[0] + out_grad[1, 0]


@njit
def _user_loop(table, ws, x, out):
    """Tuples are fine inside compiled code, field by field."""
    return eval_into(table.coeffs, table.shape, table.bounds, table.tol, x, ws.scratch, ws.work, out)


def test_called_from_user_njit_function():
    nd = build(multi=False)
    table = nd.raw()
    ws = table.workspace()
    out, out_grad = np.empty(1), np.empty((4, 1))
    p = random_points(1)[0]
    assert _user_loop(table, ws, p, out) == 0
    assert out[0] == pytest.approx(nd.evaluate(*p), rel=1e-14)
    status, total = _user_rhs(p, *table, *ws, out, out_grad)
    value, grad = nd.value_and_grad(*p)
    assert status == 0
    assert total == pytest.approx(value + grad[0], rel=1e-14)


def test_status_bits_and_values_outside_domain():
    nd = build()
    table = nd.raw()
    ws = table.workspace()
    out = np.empty((4, 3))
    status = eval_grad_into(*table, np.array([1.0, 3.5, 0.0]), *ws, out)
    assert status == 0b110
    assert np.all(np.isfinite(out))  # extrapolated, not NaN
    assert eval_grad_into(*table, np.array([np.nan, 1.0, 2.0]), *ws, out) == 0b001
    # On the boundary, within the tolerance
    assert eval_into(*table, np.array([2.0, -1.0, 4.0]), *ws, np.empty(3)) == 0


def test_size_checks():
    nd = build()
    table = nd.raw()
    ws = table.workspace()
    p = random_points(1)[0]
    with pytest.raises(ValueError, match="one coordinate per axis"):
        eval_into(*table, p[:2], *ws, np.empty(3))
    with pytest.raises(ValueError, match="one entry per output"):
        eval_into(*table, p, *ws, np.empty(2))
    with pytest.raises(ValueError, match=r"shape \(d \+ 1, F\)"):
        eval_grad_into(*table, p, *ws, np.empty((3, 3)))
    small = Workspace(ws.scratch, ws.work[:2])
    assert eval_into(*table, p, *small, np.empty(3)) == 0  # enough for values
    with pytest.raises(ValueError, match="workspace too small"):
        eval_grad_into(*table, p, *small, np.empty((4, 3)))
    other = build(ns=(22, 30, 12)).raw()  # larger after contracting the first axis
    with pytest.raises(ValueError, match="workspace too small"):
        eval_into(*other, p, *ws, np.empty(3))


# ------------------------------------------------- toy Hamiltonian end to end


def _kepler_table():
    """H(k, r, p_r, p_phi) = p_r^2/2 + p_phi^2/(2 r^2) - k/r, k a fixed parameter."""
    domains = [(0.8, 1.2), (6.0, 16.0), (-0.15, 0.15), (2.9, 3.4)]
    names = ["k", "r", "p_r", "p_phi"]
    _, vals = sample_on_grid(
        lambda k, r, pr, pphi: 0.5 * pr**2 + 0.5 * pphi**2 / r**2 - k / r, domains, (3, 40, 4, 5))
    return ChebyshevInterpolantND(vals, domains, axis_names=names)


@njit
def _hamilton_rhs(y, coeffs, shape, bounds, tol, scratch, work, out):
    """y = (r, phi, p_r, p_phi); Hamilton's equations from the interpolated H."""
    x = np.array([y[0], y[2], y[3]])
    status = eval_grad_into(coeffs, shape, bounds, tol, x, scratch, work, out)
    if status:
        raise ValueError("left the table")
    dH_dr, dH_dpr, dH_dpphi = out[1, 0], out[2, 0], out[3, 0]
    return np.array([dH_dpr, dH_dpphi, -dH_dr, 0.0])


def test_kepler_orbit_from_interpolated_hamiltonian():
    from scipy.integrate import solve_ivp

    H = _kepler_table().precontract({"k": 1.0})
    table = H.raw()
    ws = table.workspace()
    out = np.empty((table.ndim + 1, 1))

    # Bound orbit with semi-latus rectum p = 10 and eccentricity e = 0.3
    p, e = 10.0, 0.3
    y0 = np.array([p / (1 + e), 0.0, 0.0, np.sqrt(p)])
    sol = solve_ivp(lambda t, y: _hamilton_rhs(y, *table, *ws, out), (0.0, 600.0), y0,
                    method="DOP853", rtol=1e-12, atol=1e-12, dense_output=True)
    assert sol.success
    ts = np.linspace(0.0, 600.0, 400)
    r, phi, pr, pphi = sol.sol(ts)
    assert phi[-1] > 4 * np.pi  # a few radial periods

    # The flow of the interpolated H conserves the interpolated H exactly
    energy = H.evaluate_batch(np.stack([r, pr, pphi], axis=1))
    assert np.max(np.abs(energy - energy[0])) < 1e-11
    # ... and follows the analytic Kepler orbit r = p / (1 + e cos phi)
    np.testing.assert_allclose(r, p / (1 + e * np.cos(phi)), rtol=1e-10)
    exact_energy = -(1 - e**2) / (2 * p)
    assert energy[0] == pytest.approx(exact_energy, rel=1e-11)
