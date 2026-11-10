"""Tests for domain maps and MappedInterpolant (Step 5a)."""

import numpy as np
import pytest
from numba import njit

from chebyshevnd import (
    AxisMap,
    ChebyshevInterpolantND,
    MappedBatch,
    MappedInterpolant,
    OutOfDomainError,
    chebyshev_nodes,
    eval_grad_physical_into,
    eval_physical_into,
)
from chebyshevnd.backend import array_namespace, to_numpy
from chebyshevnd.maps import forward_map, inverse_map, pack_maps

# --------------------------------------------------------------- the fixture
#
# Axes (r, p, e, q):
#   r in [6, 16], stored as u = 1/r;
#   p in the window [lo(r, e), lo(r, e) + w(r, e)];
#   e in [0, 0.5], not mapped;
#   q in [sep(r, e), 5], stored as log((q - sep) / (5 - sep) + 0.1).
# The boundary functions lo, w, sep depend on (r, e).

R_RANGE, E_RANGE = (6.0, 16.0), (0.0, 0.5)
MAPS = {
    "r": AxisMap("power", power=-1),
    "p": AxisMap(lower="lo", width="w"),
    "q": AxisMap("log", lower="sep", upper=5.0, offset=0.1),
}


def lo(r, e):
    return np.sqrt(r) + 0.2 * e


def w(r, e):
    return 0.1 + 0.01 * r + 0.05 * e**2


def sep(r, e):
    return 1.0 + 0.1 * r * e


def F(r, p, e, q):
    return np.sin(2 / r) + e * np.cos(p) + 0.1 * q**2 + 0.05 * r * q


def grad_F(r, p, e, q):
    return np.stack([np.cos(2 / r) * (-2 / r**2) + 0.05 * q, -e * np.sin(p), np.cos(p),
                     0.2 * q + 0.05 * r], axis=-1)


def hess_F(r, p, e, q):
    H = np.zeros(r.shape + (4, 4))
    H[..., 0, 0] = -np.sin(2 / r) * 4 / r**4 + 4 * np.cos(2 / r) / r**3
    H[..., 0, 3] = H[..., 3, 0] = 0.05
    H[..., 1, 1] = -e * np.cos(p)
    H[..., 1, 2] = H[..., 2, 1] = -np.sin(p)
    H[..., 3, 3] = 0.2
    return H


def G(r, p, e, q):
    """A complex second output."""
    return np.exp(1j * p) * (1 + e) / r + 1j * q


def make_boundary():
    rn, en = chebyshev_nodes(24, *R_RANGE), chebyshev_nodes(10, *E_RANGE)
    rr, ee = np.meshgrid(rn, en, indexing="ij")
    vals = np.stack([lo(rr, ee), w(rr, ee), sep(rr, ee)], axis=-1)
    return ChebyshevInterpolantND(vals, [R_RANGE, E_RANGE], axis_names=["r", "e"],
                                  output_names=["lo", "w", "sep"])


def fit_mapped(names, shape, domains, maps, boundary, funcs, output_names=None):
    """
    Sample functions of the physical coordinates at the mapped nodes and fit
    them: the nodes are placed in table coordinates and mapped to physical
    ones with forward_map.
    """
    nodes = [chebyshev_nodes(n, *dom) for n, dom in zip(shape, domains)]
    y = np.stack([m.ravel() for m in np.meshgrid(*nodes, indexing="ij")], axis=1)
    codes, params = pack_maps([maps.get(n, AxisMap()) for n in names],
                              boundary.output_names if boundary else ())
    baxes = [names.index(n) for n in boundary.axis_names] if boundary else []
    x = forward_map(y, codes, params, np, baxes=baxes,
                    boundary=boundary.evaluate_batch if boundary else None)
    vals = np.stack([fn(*x.T).reshape(shape) for fn in funcs], axis=-1)
    if output_names is None:
        vals = vals[..., 0]
    table = ChebyshevInterpolantND(vals, domains, axis_names=names, output_names=output_names)
    return MappedInterpolant(table, maps, boundary)


SHAPE = (26, 14, 10, 16)
DOMAINS = [MAPS["r"].table_domain(*R_RANGE), (0.0, 1.0), E_RANGE, (np.log(0.1), np.log(1.1))]


@pytest.fixture(scope="module")
def boundary():
    return make_boundary()


@pytest.fixture(scope="module")
def mapped(boundary):
    """Real single-function MappedInterpolant."""
    return fit_mapped(["r", "p", "e", "q"], SHAPE, DOMAINS, MAPS, boundary, [F])


@pytest.fixture(scope="module")
def mapped_multi(boundary):
    """Complex two-output MappedInterpolant."""
    return fit_mapped(["r", "p", "e", "q"], SHAPE, DOMAINS, MAPS, boundary, [F, G],
                      output_names=["F", "G"])


def random_physical(n, seed=0, margin=0.0):
    """Random physical points inside the mapped region."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(*R_RANGE, n)
    e = rng.uniform(*E_RANGE, n)
    s = rng.uniform(margin, 1 - margin, n)
    p = lo(r, e) + w(r, e) * s
    sq = rng.uniform(margin, 1 - margin, n)
    q = sep(r, e) + (5 - sep(r, e)) * sq
    return np.stack([r, p, e, q], axis=1)


# ------------------------------------------------------------------ AxisMap


def test_axis_map_validation():
    with pytest.raises(ValueError, match="transform"):
        AxisMap("exp")
    with pytest.raises(ValueError, match="power="):
        AxisMap("power")
    with pytest.raises(ValueError, match="nonzero"):
        AxisMap("power", power=0)
    with pytest.raises(ValueError, match="only applies"):
        AxisMap("log", power=2.0)
    with pytest.raises(ValueError, match="offset only"):
        AxisMap(offset=1.0)
    with pytest.raises(ValueError, match="either upper or width"):
        AxisMap(upper=1.0, width=2.0)
    with pytest.raises(ValueError, match="nonzero"):
        AxisMap(width=0.0)
    with pytest.raises(ValueError, match="finite"):
        AxisMap(lower=np.inf)


def test_axis_map_round_trip_and_properties():
    m = AxisMap("log", lower="sep", upper=5.0, offset=0.1)
    assert AxisMap.from_dict(m.to_dict()) == m
    assert m.references == ("sep",)
    assert not m.is_identity and AxisMap().is_identity
    bound = m.bind({"sep": 1.5})
    assert bound.references == () and bound.lower == 1.5 and bound.upper == 5.0
    assert "log" in repr(m) and "sep" in repr(m)


def test_table_domain():
    assert AxisMap("power", power=-1).table_domain(6.0, 16.0) == pytest.approx((1 / 16, 1 / 6))
    assert AxisMap("log", lower=1.0, offset=0.5).table_domain(1.0, 3.0) == pytest.approx(
        (np.log(0.5), np.log(2.5)))
    assert AxisMap(lower=2.0, width=-4.0).table_domain(2.0, 6.0) == pytest.approx((-1.0, 0.0))
    with pytest.raises(ValueError, match="constant window"):
        AxisMap(lower="lo").table_domain(0.0, 1.0)
    with pytest.raises(ValueError, match="not inside"):
        AxisMap("log").table_domain(-1.0, 1.0)


# ------------------------------------------------------------ map functions


def _packed(boundary):
    codes, params = pack_maps([MAPS.get(n, AxisMap()) for n in ("r", "p", "e", "q")],
                              boundary.output_names)
    return codes, params, boundary, [0, 2]


def _boundary_derivs(boundary, xb):
    vals, grads = boundary.value_and_grad_batch(xb)
    hess = np.empty((len(xb), 2, 2, 3))
    for i in range(2):
        for j in range(2):
            orders = [0, 0]
            orders[i] += 1
            orders[j] += 1
            hess[:, i, j] = boundary.derivative(orders).evaluate_batch(xb)
    return vals, grads, hess


def test_inverse_and_forward_maps_round_trip(boundary):
    codes, params, boundary, baxes = _packed(boundary)
    x = random_physical(50)
    bvals = boundary.evaluate_batch(x[:, baxes])
    y = inverse_map(x, codes, params, np, bvals=bvals, baxes=baxes)
    np.testing.assert_allclose(y[:, 0], 1 / x[:, 0], rtol=1e-14)
    np.testing.assert_allclose(y[:, 2], x[:, 2])
    back = forward_map(y, codes, params, np, boundary=boundary.evaluate_batch, baxes=baxes)
    np.testing.assert_allclose(back, x, rtol=1e-13)


def test_map_jacobian_and_hessian_against_finite_differences(boundary):
    codes, params, boundary, baxes = _packed(boundary)
    x = random_physical(20, seed=1, margin=0.05)

    def jac_hess(points):
        bv, bg, bh = _boundary_derivs(boundary, points[:, baxes])
        return inverse_map(points, codes, params, np, bv, bg, bh, baxes, order=2)

    y, jac, hess = jac_hess(x)
    h = 1e-6
    for k in range(4):
        step = np.zeros(4)
        step[k] = h * (1.0 if k != 0 else 10.0)
        yp, jp, _ = jac_hess(x + step)
        ym, jm, _ = jac_hess(x - step)
        fd_jac = (yp - ym) / (2 * step[k])
        np.testing.assert_allclose(jac[:, :, k], fd_jac, rtol=1e-7, atol=1e-8)
        fd_hess = (jp - jm) / (2 * step[k])
        np.testing.assert_allclose(hess[:, :, :, k], fd_hess, rtol=1e-6, atol=1e-7)
    # Only the boundary inputs (r, e) appear off the diagonal
    assert np.all(jac[:, 0, 1:] == 0) and np.all(jac[:, 2, [0, 1, 3]] == 0)


# --------------------------------------------------------------- evaluation


def test_evaluate_matches_function(mapped):
    x = random_physical(200)
    np.testing.assert_allclose(mapped.evaluate_batch(x), F(*x.T), rtol=0, atol=2e-12)
    for p in x[:10]:
        assert mapped.evaluate(*p) == pytest.approx(F(*p), abs=2e-12)


def test_physical_gradient_matches_function(mapped):
    x = random_physical(200, seed=2)
    values, grads = mapped.value_and_grad_batch(x)
    np.testing.assert_allclose(values, F(*x.T), rtol=0, atol=2e-12)
    np.testing.assert_allclose(grads, grad_F(*x.T), rtol=0, atol=2e-10)
    for p, v, g in zip(x[:20], values, grads):
        value, grad = mapped.value_and_grad(*p)
        assert value == pytest.approx(v, rel=1e-13)
        np.testing.assert_allclose(grad, g, rtol=1e-12, atol=1e-14)


def test_physical_hessian_matches_function(mapped):
    x = random_physical(100, seed=3)
    values, grads, hess = mapped.value_grad_hess_batch(x)
    assert hess.shape == (100, 4, 4)
    np.testing.assert_allclose(values, F(*x.T), rtol=0, atol=2e-12)
    np.testing.assert_allclose(grads, grad_F(*x.T), rtol=0, atol=2e-10)
    np.testing.assert_allclose(hess, hess_F(*x.T), rtol=0, atol=4e-8)


def test_complex_multi_output(mapped_multi):
    x = random_physical(50, seed=4)
    values, grads = mapped_multi.value_and_grad_batch(x)
    assert values.shape == (50, 2) and grads.shape == (50, 4, 2)
    assert values.dtype == np.complex128
    np.testing.assert_allclose(values[:, 0], F(*x.T), rtol=0, atol=2e-12)
    np.testing.assert_allclose(values[:, 1], G(*x.T), rtol=0, atol=2e-12)
    value, grad = mapped_multi.value_and_grad(*x[0])
    np.testing.assert_allclose(value, values[0], rtol=1e-13)
    np.testing.assert_allclose(grad, grads[0], rtol=1e-12, atol=1e-14)
    _, _, hess = mapped_multi.value_grad_hess_batch(x)
    assert hess.shape == (50, 4, 4, 2)


def test_to_table_to_physical(mapped):
    x = random_physical(30)
    y = mapped.to_table(x)
    np.testing.assert_allclose(mapped.to_physical(y), x, rtol=1e-13)
    np.testing.assert_allclose(y[:, 0], 1 / x[:, 0])


def test_out_of_domain_in_physical_coordinates(mapped):
    r, e = 9.0, 0.3
    below_window = [r, lo(r, e) - 0.01, e, 3.0]
    with pytest.raises(OutOfDomainError, match=r"p = .* maps to .*outside the table domain"):
        mapped.evaluate(*below_window)
    with pytest.raises(OutOfDomainError, match=r"r = 20\.0 maps to"):
        mapped.value_and_grad(20.0, lo(16.0, e), e, 3.0)
    # log of a negative number: q far below sep
    _, status = mapped.evaluate_batch(np.array([[r, lo(r, e), e, -5.0]]), return_status=True)
    assert int(status[0]) == 0b1000
    # e outside both the table and the boundary table
    _, status = mapped.evaluate_batch(np.array([[r, lo(r, 0.5), 0.7, 3.0]]), return_status=True)
    assert int(status[0]) & 0b0100
    with pytest.raises(OutOfDomainError, match="1 of 2 points"):
        mapped.evaluate_batch(np.array([below_window, [r, lo(r, e), e, 3.0]]))


def test_construction_checks():
    boundary = make_boundary()
    grid_shape = (6, 4, 4, 4)
    vals = np.zeros(grid_shape)
    domains = [(1 / 16, 1 / 6), (0, 1), E_RANGE, (np.log(0.1), np.log(1.1))]
    table = ChebyshevInterpolantND(vals, domains, axis_names=["r", "p", "e", "q"])
    with pytest.raises(ValueError, match="no boundary table"):
        MappedInterpolant(table, MAPS)
    with pytest.raises(ValueError, match="does not have"):
        MappedInterpolant(table, {"p": AxisMap(lower="nope")}, boundary)
    with pytest.raises(ValueError, match="inputs of the boundary"):
        MappedInterpolant(table, {"e": AxisMap(lower="lo")}, boundary)
    with pytest.raises(TypeError, match="AxisMap"):
        MappedInterpolant(table, {"p": "lo"}, boundary)
    narrow = ChebyshevInterpolantND(np.zeros((5, 3, 1)), [(7.0, 16.0), E_RANGE], axis_names=["r", "e"],
                                    output_names=["lo"])
    with pytest.raises(ValueError, match="covers r in"):
        MappedInterpolant(table, {"r": MAPS["r"], "p": AxisMap(lower="lo")}, narrow)


# --------------------------------------------------------------- raw / Numba


@njit
def _rhs_like(x, coeffs, shape, bounds, tol, codes, params, bcoeffs, bshape, bbounds, baxes,
              scratch, work, out, out_grad):
    s1 = eval_physical_into(coeffs, shape, bounds, tol, codes, params, bcoeffs, bshape, bbounds,
                            baxes, x, scratch, work, out)
    s2 = eval_grad_physical_into(coeffs, shape, bounds, tol, codes, params, bcoeffs, bshape,
                                 bbounds, baxes, x, scratch, work, out_grad)
    return s1 | s2


def test_raw_from_user_njit(mapped_multi):
    table = mapped_multi.raw()
    ws = table.workspace()
    out = np.empty(2, dtype=np.complex128)
    out_grad = np.empty((5, 2), dtype=np.complex128)
    x = random_physical(5, seed=6)
    values, grads = mapped_multi.value_and_grad_batch(x)
    for p, v, g in zip(x, values, grads):
        assert _rhs_like(p, *table, *ws, out, out_grad) == 0
        np.testing.assert_allclose(out, v, rtol=1e-13)
        np.testing.assert_allclose(out_grad[0], v, rtol=1e-13)
        np.testing.assert_allclose(out_grad[1:], g, rtol=1e-12, atol=1e-14)
    with pytest.raises(ValueError, match="workspace too small"):
        eval_physical_into(*table, x[0], ws.scratch[:5], ws.work, out)


def test_raw_without_boundary():
    # 1/r + 1/r^2 is a quadratic in u = 1/r: exact on 3 nodes
    m = AxisMap("power", power=-1)
    domain = m.table_domain(2.0, 10.0)
    un = chebyshev_nodes(3, *domain)
    table = ChebyshevInterpolantND(un + un**2, [domain], axis_names=["r"])
    mapped = MappedInterpolant(table, {"r": m})
    value, grad = mapped.value_and_grad(4.0)
    assert value == pytest.approx(1 / 4 + 1 / 16, rel=1e-14)
    assert grad[0] == pytest.approx(-1 / 16 - 2 / 64, rel=1e-13)
    raw = mapped.raw()
    assert raw.bshape.shape == (0,)
    out = np.empty((2, 1))
    assert eval_grad_physical_into(*raw, np.array([4.0]), *raw.workspace(), out) == 0
    np.testing.assert_allclose(out[:, 0], [value, grad[0]], rtol=1e-14)


# ------------------------------------------------------------ precontraction


def test_precontract_partial_boundary(mapped):
    x = random_physical(20, seed=7)
    child = mapped.precontract({"e": 0.3})
    assert child.axis_names == ("r", "p", "q")
    assert child.boundary.axis_names == ("r",)
    x[:, 2] = 0.3
    x[:, 1] = lo(x[:, 0], 0.3) + w(x[:, 0], 0.3) * 0.4
    x[:, 3] = sep(x[:, 0], 0.3) + (5 - sep(x[:, 0], 0.3)) * 0.7
    full_v, full_g = mapped.value_and_grad_batch(x)
    v, g = child.value_and_grad_batch(x[:, [0, 1, 3]])
    np.testing.assert_allclose(v, full_v, rtol=1e-13)
    np.testing.assert_allclose(g, full_g[:, [0, 1, 3]], rtol=1e-11, atol=1e-13)


def test_precontract_all_boundary_inputs_binds_constants(mapped):
    r, e = 11.0, 0.2
    child = mapped.precontract({"r": r, "e": e})
    assert child.boundary is None
    assert child.maps["p"].references == () and child.maps["p"].lower == pytest.approx(lo(r, e))
    p, q = lo(r, e) + 0.3 * w(r, e), 3.0
    v, g = child.value_and_grad(p, q)
    full_v, full_g = mapped.value_and_grad(r, p, e, q)
    assert v == pytest.approx(full_v, rel=1e-13)
    np.testing.assert_allclose(g, full_g[[1, 3]], rtol=1e-11)


def test_precontract_dependent_axis_needs_boundary_inputs(mapped):
    with pytest.raises(ValueError, match="can only be fixed together"):
        mapped.precontract({"p": 3.0})
    r, e = 9.0, 0.1
    p = lo(r, e) + 0.5 * w(r, e)
    child = mapped.precontract({"r": r, "e": e, "p": p})
    assert child.axis_names == ("q",)
    assert child.evaluate(2.5) == pytest.approx(mapped.evaluate(r, p, e, 2.5), rel=1e-13)
    with pytest.raises(OutOfDomainError, match="p = "):
        mapped.precontract({"r": r, "e": e, "p": lo(r, e) - 1.0})


def test_precontract_batch_matches_single(mapped_multi):
    x = random_physical(8, seed=8)
    for fixed_axes in (["e"], ["r", "e"], ["r", "e", "p"]):
        fixed = {n: x[:, mapped_multi.axis_names.index(n)] for n in fixed_axes}
        batch = mapped_multi.precontract_batch(fixed)
        assert isinstance(batch, MappedBatch) and batch.n_members == 8
        live = [i for i, n in enumerate(mapped_multi.axis_names) if n not in fixed_axes]
        values, grads = batch.value_and_grad(x[:, live])
        evals = batch.evaluate(x[:, live])
        for j in range(8):
            child = mapped_multi.precontract({n: x[j, mapped_multi.axis_names.index(n)]
                                              for n in fixed_axes})
            v, g = child.value_and_grad(*x[j, live])
            np.testing.assert_allclose(values[j], v, rtol=1e-12)
            np.testing.assert_allclose(evals[j], v, rtol=1e-12)
            np.testing.assert_allclose(grads[j], g, rtol=1e-10, atol=1e-13)


def test_precontract_batch_status(mapped):
    r = np.array([8.0, 30.0])
    with pytest.raises(OutOfDomainError, match="member 1"):
        mapped.precontract_batch({"r": r})
    batch, status = mapped.precontract_batch({"r": r}, return_status=True)
    assert list(status) == [0, 1]
    pts = random_physical(2, seed=9)[:, 1:]
    pts[0, 0] = -100.0
    with pytest.raises(OutOfDomainError, match="member 0"):
        batch.evaluate(pts)


# ------------------------------------------- PM-like Hamiltonian end to end


def _kepler_mapped():
    """
    H(r, p_r, p_phi) = p_r^2/2 + p_phi^2/(2 r^2) - 1/r, with r stored as
    u = 1/r and p_phi relative to the circular value sqrt(r) (a boundary
    table over r).
    """
    r_range = (6.0, 16.0)
    rn = chebyshev_nodes(20, *r_range)
    boundary = ChebyshevInterpolantND(np.sqrt(rn)[:, None], [r_range], axis_names=["r"],
                                      output_names=["pphi_circ"])
    inv = AxisMap("power", power=-1)
    return fit_mapped(["r", "pr", "pphi"], (30, 4, 5),
                      [inv.table_domain(*r_range), (-0.15, 0.15), (-0.8, 0.6)],
                      {"r": inv, "pphi": AxisMap(lower="pphi_circ")}, boundary,
                      [lambda r, pr, pphi: 0.5 * pr**2 + 0.5 * pphi**2 / r**2 - 1 / r])


@njit
def _hamilton(y, coeffs, shape, bounds, tol, codes, params, bcoeffs, bshape, bbounds, baxes,
              scratch, work, out):
    x = np.array([y[0], y[2], y[3]])
    status = eval_grad_physical_into(coeffs, shape, bounds, tol, codes, params, bcoeffs, bshape,
                                     bbounds, baxes, x, scratch, work, out)
    if status:
        raise ValueError("left the table")
    return np.array([out[2, 0], out[3, 0], -out[1, 0], 0.0])


def test_kepler_orbit_with_maps():
    from scipy.integrate import solve_ivp

    H = _kepler_mapped()
    table = H.raw()
    ws = table.workspace()
    out = np.empty((4, 1))
    p, e = 10.0, 0.3
    y0 = np.array([p / (1 + e), 0.0, 0.0, np.sqrt(p)])
    sol = solve_ivp(lambda t, y: _hamilton(y, *table, *ws, out), (0.0, 600.0), y0,
                    method="DOP853", rtol=1e-12, atol=1e-12, dense_output=True)
    assert sol.success
    r, phi, pr, pphi = sol.sol(np.linspace(0.0, 600.0, 400))
    energy = H.evaluate_batch(np.stack([r, pr, pphi], axis=1))
    assert np.max(np.abs(energy - energy[0])) < 1e-11
    np.testing.assert_allclose(r, p / (1 + e * np.cos(phi)), rtol=1e-10)


# ------------------------------------------------------------------- GPU


@pytest.fixture
def cupy():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("no CUDA device")
    return cp


@pytest.mark.gpu
def test_gpu_mapped_batch_matches_cpu(cupy, mapped_multi):
    x = random_physical(40, seed=10)
    cpu = mapped_multi.value_and_grad_batch(x)
    gpu = mapped_multi.value_and_grad_batch(cupy.asarray(x))
    assert array_namespace(gpu[1]) is cupy
    for c, g in zip(cpu, gpu):
        np.testing.assert_allclose(to_numpy(g), c, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(to_numpy(mapped_multi.to_table(cupy.asarray(x))),
                               mapped_multi.to_table(x), rtol=1e-14)


@pytest.mark.gpu
def test_gpu_mapped_precontract_batch(cupy, mapped_multi):
    x = random_physical(12, seed=11)
    for fixed_axes in (["e"], ["r", "e"]):
        live = [i for i, n in enumerate(mapped_multi.axis_names) if n not in fixed_axes]
        cpu = mapped_multi.precontract_batch(
            {n: x[:, mapped_multi.axis_names.index(n)] for n in fixed_axes}).value_and_grad(x[:, live])
        gpu = mapped_multi.precontract_batch(
            {n: cupy.asarray(x[:, mapped_multi.axis_names.index(n)]) for n in fixed_axes}
        ).value_and_grad(cupy.asarray(x[:, live]))
        for c, g in zip(cpu, gpu):
            np.testing.assert_allclose(to_numpy(g), c, rtol=1e-12, atol=1e-14)
