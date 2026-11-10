"""Tests for ChebyshevInterpolantND."""

import numpy as np
import pytest

from chebyshevnd import (
    ChebyshevInterpolant1D,
    ChebyshevInterpolant2D,
    ChebyshevInterpolant3D,
    ChebyshevInterpolant4D,
    ChebyshevInterpolantND,
)
from helpers import sample_on_grid

# One smooth test function per dimension, with distinct domains and node
# counts on every axis so that axis or bound mix-ups cannot cancel out
CASES = {
    1: (lambda x: np.exp(x) * np.sin(2 * x), [(0.0, 2.0)], (30,)),
    2: (lambda x, y: np.sin(x) * np.cos(y), [(-1.0, 2.0), (0.0, 3.0)], (24, 18)),
    3: (lambda x, y, z: np.sin(x) * np.cos(y) * np.exp(-z),
        [(0.0, 2.0), (-1.0, 3.0), (1.0, 4.0)], (22, 16, 12)),
    4: (lambda x, y, z, w: np.sin(x) * np.cos(y) * np.exp(-z) * (w**3 + w),
        [(0.0, 2.0), (-1.0, 3.0), (1.0, 4.0), (-2.0, 0.0)], (13, 14, 12, 8)),
}
V01_CLASSES = {1: ChebyshevInterpolant1D, 2: ChebyshevInterpolant2D,
               3: ChebyshevInterpolant3D, 4: ChebyshevInterpolant4D}


def random_points(domains, n, seed=11):
    rng = np.random.default_rng(seed)
    return [tuple(rng.uniform(a, b) for a, b in domains) for _ in range(n)]


def build(dim, nodes="Chebyshev", **kwargs):
    f, domains, Ns = CASES[dim]
    axes, vals = sample_on_grid(f, domains, Ns, nodes=nodes)
    return f, domains, axes, vals, ChebyshevInterpolantND(vals, domains, nodes=nodes, **kwargs)


# ------------------------------------------------------------------ accuracy

@pytest.mark.parametrize("nodes", ["Chebyshev", "GaussLobatto"])
@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_matches_v01_classes(dim, nodes):
    f, domains, _, vals, nd = build(dim, nodes=nodes)
    bounds = [v for dom in domains for v in dom]
    old = V01_CLASSES[dim](vals, *bounds, nodes=nodes)
    for p in random_points(domains, 20):
        assert nd.evaluate(*p) == pytest.approx(old.evaluate(*p), rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("nodes", ["Chebyshev", "GaussLobatto"])
def test_reproduces_grid_values_exactly(nodes):
    _, _, axes, vals, nd = build(3, nodes=nodes)
    for i, j, k in [(0, 0, 0), (3, 7, 2), (21, 15, 11), (10, 4, 6)]:
        assert nd.evaluate(axes[0][i], axes[1][j], axes[2][k]) == pytest.approx(vals[i, j, k], abs=1e-13)


def test_5d_separable_function():
    f = lambda a, b, c, d, e: np.exp(0.3 * a) * np.cos(0.5 * b) * (1 + 0.2 * c) ** 2 * np.sin(0.4 * d + 0.1) * np.cos(e)
    domains = [(0.0, 1.0), (-1.0, 2.0), (0.5, 1.5), (-2.0, -1.0), (0.0, 0.5)]
    _, vals = sample_on_grid(f, domains, (12, 14, 6, 10, 11))
    nd = ChebyshevInterpolantND(vals, domains)
    for p in random_points(domains, 30):
        assert nd.evaluate(*p) == pytest.approx(f(*p), abs=1e-12)


def test_6d_separable_function():
    f = lambda a, b, c, d, e, g: np.cos(a) * np.exp(-b) * (c**2 + 1) * np.sin(d + 1) * np.cos(2 * e) * (1 + g)
    domains = [(0.0, 1.0), (0.0, 0.5), (-1.0, 1.0), (0.0, 1.0), (-0.5, 0.5), (1.0, 2.0)]
    _, vals = sample_on_grid(f, domains, (11, 10, 4, 12, 13, 3))
    nd = ChebyshevInterpolantND(vals, domains)
    for p in random_points(domains, 30):
        assert nd.evaluate(*p) == pytest.approx(f(*p), abs=1e-11)


def test_evaluate_returns_python_float():
    _, domains, _, _, nd = build(2)
    assert type(nd.evaluate(*random_points(domains, 1)[0])) is float


# ---------------------------------------------------------------- truncation

def test_optimize_bound_holds_pointwise():
    f, domains, _, _, nd = build(4)
    points = random_points(domains, 40)
    before = np.array([nd.evaluate(*p) for p in points])
    scale = np.abs(nd.get_coefficients(full=True)).max()

    rel_tol = 1e-6
    nd.optimize(rel_tol)
    assert nd.kmax < nd.grid_shape
    assert nd.get_coefficients().shape == nd.kmax
    after = np.array([nd.evaluate(*p) for p in points])
    # The discarded coefficient magnitudes bound the pointwise change
    assert np.max(np.abs(after - before)) <= rel_tol * scale


def test_optimize_restarts_from_full_tensor():
    _, domains, _, _, nd = build(3)
    p = random_points(domains, 1)[0]
    full = nd.evaluate(*p)
    nd.optimize(1e-3)
    loose = nd.kmax
    nd.optimize(1e-13)
    assert all(t >= l for t, l in zip(nd.kmax, loose))
    assert nd.evaluate(*p) == pytest.approx(full, abs=1e-12)


def test_optimize_without_full_tensor():
    _, _, _, _, nd = build(2)
    nd.optimize(1e-8, keep_full=False)
    with pytest.raises(RuntimeError):
        nd.optimize(1e-6)
    with pytest.raises(RuntimeError):
        nd.get_coefficients(full=True)


@pytest.mark.parametrize("rel_tol", [0.0, 1.0, -1e-3])
def test_optimize_rejects_bad_tolerance(rel_tol):
    _, _, _, _, nd = build(2)
    with pytest.raises(ValueError):
        nd.optimize(rel_tol)


def test_effective_grid_is_kmax():
    _, _, _, _, nd = build(3)
    assert nd.effective_grid() == nd.grid_shape
    nd.optimize(1e-6)
    assert nd.effective_grid() == nd.kmax


# ------------------------------------------------------------ error estimates

def test_error_estimate_matches_v01_and_bounds_actual_error():
    f, domains, _, vals, nd = build(3)
    old = ChebyshevInterpolant3D(vals, *[v for dom in domains for v in dom])
    assert nd.absolute_error() == pytest.approx(old.absolute_error(), rel=1e-12)
    assert nd.relative_error() == pytest.approx(old.relative_error(), rel=1e-12)
    actual = max(abs(nd.evaluate(*p) - f(*p)) for p in random_points(domains, 200))
    assert actual <= nd.absolute_error()


def test_absolute_error_scales_with_matching_axis():
    # White-box: a trailing coefficient along axis 1 is scaled by N_1, as in
    # the 2D-4D classes
    vals = np.zeros((4, 9, 5))
    nd = ChebyshevInterpolantND(vals, [(-1, 1)] * 3)
    c = np.zeros((4, 9, 5, 1))
    c[0, -1, 0, 0] = 5.0
    from chebyshevnd.interpnd import _trailing_error_estimate
    assert _trailing_error_estimate(np.abs(c))[0] == pytest.approx(5.0 * 9)


def test_absolute_error_includes_truncation():
    _, _, _, _, nd = build(3)
    before = nd.absolute_error()
    nd.optimize(1e-6)
    assert nd.absolute_error() > before


# ---------------------------------------------------------------- validation

def test_axis_names_default_and_custom():
    _, domains, _, vals, nd = build(3)
    assert nd.axis_names == ("x0", "x1", "x2")
    named = ChebyshevInterpolantND(vals, domains, axis_names=("nu", "u", "e"))
    assert named.axis_names == ("nu", "u", "e")
    assert "nu" in repr(named)


def test_properties():
    _, domains, _, _, nd = build(3)
    assert nd.ndim == 3
    assert nd.grid_shape == (22, 16, 12)
    assert nd.domains == tuple(domains)
    assert nd.nodes == "Chebyshev"


@pytest.mark.parametrize("names", [("a", "b"), ("a", "a", "b")])
def test_bad_axis_names(names):
    _, domains, _, vals, _ = build(3)
    with pytest.raises(ValueError):
        ChebyshevInterpolantND(vals, domains, axis_names=names)


def test_domain_count_must_match_vals():
    _, domains, _, vals, _ = build(3)
    with pytest.raises(ValueError, match="one \\(lower, upper\\) pair per axis"):
        ChebyshevInterpolantND(vals, domains[:2])


@pytest.mark.parametrize("bad", [(2.0, 0.0), (1.0, 1.0), (0.0, np.inf), (0.0,), "ab"])
def test_bad_domain_pairs(bad):
    with pytest.raises(ValueError):
        ChebyshevInterpolantND(np.ones((4, 4)), [(0.0, 1.0), bad])


def test_bad_node_family():
    with pytest.raises(ValueError):
        ChebyshevInterpolantND(np.ones((4, 4)), [(0, 1), (0, 1)], nodes="bogus")


def test_gauss_lobatto_needs_two_nodes_per_axis():
    with pytest.raises(ValueError):
        ChebyshevInterpolantND(np.ones((4, 1)), [(0, 1), (0, 1)], nodes="GaussLobatto")
    # A single root node per axis is a valid (constant) interpolant
    nd = ChebyshevInterpolantND(np.full((4, 1), 2.0), [(0, 1), (0, 1)])
    assert nd.evaluate(0.3, 0.7) == pytest.approx(2.0)


def test_evaluate_argument_checks():
    _, _, _, _, nd = build(3)
    with pytest.raises(TypeError):
        nd.evaluate(0.5, 0.5)
    with pytest.raises(TypeError):
        nd.evaluate(np.array([0.5, 0.6]), 0.5, 2.0)


# -------------------------------------------------------------------- complex

COMPLEX_CASES = {
    3: (lambda x, y, z: np.exp(1j * x) * np.cos(y) * z,
        [(0.0, 2.0), (-1.0, 1.0), (0.5, 1.5)], (22, 18, 3)),
    4: (lambda x, y, z, w: np.exp(1j * (x + 0.5 * w)) * np.cos(y) * (1 + z**2),
        [(0.0, 2.0), (-1.0, 1.0), (0.0, 1.0), (-1.0, 0.0)], (22, 18, 4, 16)),
    5: (lambda x, y, z, w, v: np.exp(1j * x) * np.sin(y + 1) * z * np.exp(-1j * w) * (1 + v),
        [(0.0, 1.0), (-1.0, 0.0), (1.0, 2.0), (0.0, 1.0), (-0.5, 0.5)], (17, 16, 2, 17, 2)),
}


@pytest.mark.parametrize("nodes", ["Chebyshev", "GaussLobatto"])
@pytest.mark.parametrize("dim", [3, 4, 5])
def test_complex_functions(dim, nodes):
    f, domains, Ns = COMPLEX_CASES[dim]
    _, vals = sample_on_grid(f, domains, Ns, nodes=nodes)
    nd = ChebyshevInterpolantND(vals, domains, nodes=nodes)
    assert nd.dtype == np.complex128
    for p in random_points(domains, 20):
        got = nd.evaluate(*p)
        assert type(got) is complex
        assert abs(got - f(*p)) < 1e-14


def test_complex_equals_real_and_imaginary_parts():
    f, domains, Ns = COMPLEX_CASES[4]
    _, vals = sample_on_grid(f, domains, Ns)
    nd = ChebyshevInterpolantND(vals, domains)
    re = ChebyshevInterpolantND(vals.real, domains)
    im = ChebyshevInterpolantND(vals.imag, domains)
    for p in random_points(domains, 10):
        assert nd.evaluate(*p) == pytest.approx(complex(re.evaluate(*p), im.evaluate(*p)), abs=1e-15)


def test_complex_optimize_bound_holds():
    f, domains, Ns = COMPLEX_CASES[4]
    _, vals = sample_on_grid(f, domains, Ns)
    nd = ChebyshevInterpolantND(vals, domains)
    points = random_points(domains, 30)
    before = np.array([nd.evaluate(*p) for p in points])
    scale = np.abs(nd.get_coefficients(full=True)).max()
    nd.optimize(1e-7)
    assert nd.kmax < nd.grid_shape
    after = np.array([nd.evaluate(*p) for p in points])
    assert np.max(np.abs(after - before)) <= 1e-7 * scale


def test_real_input_stays_real():
    _, _, _, _, nd = build(2)
    assert nd.dtype == np.float64
    integer_vals = ChebyshevInterpolantND(np.arange(12).reshape(3, 4), [(0, 1), (0, 1)])
    assert integer_vals.dtype == np.float64


# --------------------------------------------------------------- multi-output

MULTI_DOMAINS = [(0.0, 2.0), (-1.0, 3.0), (1.0, 4.0)]
MULTI_NS = (22, 16, 12)
MULTI_FUNCS = {
    "wave": lambda x, y, z: np.sin(x) * np.cos(y) * np.exp(-z),
    "tiny": lambda x, y, z: 1e-6 * (x * y + z**2),
    "bump": lambda x, y, z: np.exp(-((x - 1) ** 2) - 0.1 * y**2) / z,
}


def build_multi(dtype=float):
    axes = [np.asarray(a) for a in sample_on_grid(lambda *x: x[0], MULTI_DOMAINS, MULTI_NS)[0]]
    grids = np.meshgrid(*axes, indexing="ij")
    vals = np.stack([f(*grids) for f in MULTI_FUNCS.values()], axis=-1).astype(dtype)
    nd = ChebyshevInterpolantND(vals, MULTI_DOMAINS, output_names=list(MULTI_FUNCS))
    singles = [ChebyshevInterpolantND(vals[..., i], MULTI_DOMAINS) for i in range(vals.shape[-1])]
    return nd, singles


@pytest.mark.parametrize("dtype", [float, complex])
def test_multi_output_equals_stacked_single_outputs(dtype):
    nd, singles = build_multi(dtype)
    for p in random_points(MULTI_DOMAINS, 20):
        got = nd.evaluate(*p)
        assert got.shape == (3,)
        np.testing.assert_allclose(got, [s.evaluate(*p) for s in singles], rtol=1e-15, atol=1e-18)


def test_output_names_and_index():
    nd, _ = build_multi()
    assert nd.n_outputs == 3
    assert nd.output_names == ("wave", "tiny", "bump")
    assert nd.output_index == {"wave": 0, "tiny": 1, "bump": 2}
    p = random_points(MULTI_DOMAINS, 1)[0]
    assert nd.evaluate(*p)[nd.output_index["tiny"]] == pytest.approx(MULTI_FUNCS["tiny"](*p), rel=1e-10)
    single = ChebyshevInterpolantND(np.ones((3, 3)), [(0, 1), (0, 1)])
    assert single.n_outputs == 1 and single.output_names == () and single.output_index == {}


def test_evaluate_returns_independent_arrays():
    nd, _ = build_multi()
    p, q = random_points(MULTI_DOMAINS, 2)
    first = nd.evaluate(*p)
    second = nd.evaluate(*q)
    assert not np.shares_memory(first, second)
    np.testing.assert_array_equal(first, nd.evaluate(*p))


def test_multi_output_coefficients_and_errors_per_function():
    nd, singles = build_multi()
    assert nd.get_coefficients().shape == MULTI_NS + (3,)
    assert nd.get_coefficients(full=True).shape == MULTI_NS + (3,)
    np.testing.assert_allclose(nd.absolute_error(), [s.absolute_error() for s in singles], rtol=1e-14)
    np.testing.assert_allclose(nd.relative_error(), [s.relative_error() for s in singles], rtol=1e-14)


def test_multi_output_optimize_bound_holds_per_function():
    # "tiny" is 10^6 times smaller than the others; the shared truncation
    # must still bound its error relative to its own scale
    nd, _ = build_multi()
    points = random_points(MULTI_DOMAINS, 40)
    before = np.array([nd.evaluate(*p) for p in points])
    scales = np.abs(nd.get_coefficients(full=True)).reshape(-1, 3).max(axis=0)
    rel_tol = 1e-6
    nd.optimize(rel_tol)
    assert nd.kmax < nd.grid_shape
    assert nd.get_coefficients().shape == nd.kmax + (3,)
    after = np.array([nd.evaluate(*p) for p in points])
    assert np.all(np.abs(after - before).max(axis=0) <= rel_tol * scales)


def test_function_axis_requires_output_names():
    nd, _ = build_multi()
    vals = np.ones(MULTI_NS + (3,))
    with pytest.raises(ValueError, match="pass output_names"):
        ChebyshevInterpolantND(vals, MULTI_DOMAINS)
    with pytest.raises(ValueError, match="trailing function axis"):
        ChebyshevInterpolantND(np.ones(MULTI_NS), MULTI_DOMAINS, output_names=["a"])
    with pytest.raises(ValueError):
        ChebyshevInterpolantND(vals, MULTI_DOMAINS, output_names=["a", "b"])
    with pytest.raises(ValueError):
        ChebyshevInterpolantND(vals, MULTI_DOMAINS, output_names=["a", "a", "b"])
    with pytest.raises(ValueError):
        ChebyshevInterpolantND(np.ones(MULTI_NS + (0,)), MULTI_DOMAINS, output_names=[])


def test_single_named_output_keeps_function_axis():
    vals = np.ones((4, 5, 1))
    nd = ChebyshevInterpolantND(vals, [(0, 1), (0, 1)], output_names=["H"])
    assert nd.evaluate(0.2, 0.3).shape == (1,)


def test_gauss_lobatto_multi_output_checks_only_grid_axes():
    axes = [np.asarray(a) for a in sample_on_grid(lambda *x: x[0], MULTI_DOMAINS, MULTI_NS, nodes="GaussLobatto")[0]]
    grids = np.meshgrid(*axes, indexing="ij")
    vals = MULTI_FUNCS["wave"](*grids)[..., np.newaxis]
    nd = ChebyshevInterpolantND(vals, MULTI_DOMAINS, nodes="GaussLobatto", output_names=["wave"])
    single = ChebyshevInterpolantND(vals[..., 0], MULTI_DOMAINS, nodes="GaussLobatto")
    for p in random_points(MULTI_DOMAINS, 5):
        assert nd.evaluate(*p)[0] == pytest.approx(single.evaluate(*p), rel=1e-15)
        assert nd.evaluate(*p)[0] == pytest.approx(MULTI_FUNCS["wave"](*p), abs=1e-10)


# -------------------------------------------------------------- domain checks

def test_out_of_domain_raises_and_names_the_axis():
    from chebyshevnd import OutOfDomainError
    _, domains, _, vals, _ = build(3)
    nd = ChebyshevInterpolantND(vals, domains, axis_names=("nu", "u", "e"))
    with pytest.raises(OutOfDomainError, match="u = 3.5") as info:
        nd.evaluate(1.0, 3.5, 2.0)
    assert isinstance(info.value, ValueError)
    assert "nu" not in str(info.value) and "e =" not in str(info.value)


def test_out_of_domain_reports_every_bad_axis():
    from chebyshevnd import OutOfDomainError
    _, domains, _, _, nd = build(3)
    with pytest.raises(OutOfDomainError, match="x0 .*x2"):
        nd.evaluate(-1.0, 0.5, 9.0)


def test_boundaries_and_tolerance():
    from chebyshevnd import OutOfDomainError
    _, domains, _, _, nd = build(2)
    (ax, bx), (ay, by) = domains
    for x, y in [(ax, ay), (bx, by), (ax, by), (bx, ay)]:
        nd.evaluate(x, y)  # exact boundaries are inside
    half_width = (bx - ax) / 2
    nd.evaluate(bx + 0.5e-12 * half_width, ay)  # within the default tolerance
    with pytest.raises(OutOfDomainError):
        nd.evaluate(bx + 1e-9 * half_width, ay)


def test_custom_domain_tolerance():
    from chebyshevnd import OutOfDomainError
    _, domains, _, vals, _ = build(2)
    nd = ChebyshevInterpolantND(vals, domains, domain_tol=0.1)
    assert nd.domain_tol == 0.1
    (ax, bx), (ay, by) = domains
    nd.evaluate(bx + 0.05 * (bx - ax) / 2, ay)
    with pytest.raises(OutOfDomainError):
        nd.evaluate(bx + 0.2 * (bx - ax) / 2, ay)
    with pytest.raises(ValueError):
        ChebyshevInterpolantND(vals, domains, domain_tol=-1.0)


def test_nan_coordinate_is_out_of_domain():
    from chebyshevnd import OutOfDomainError
    _, _, _, _, nd = build(2)
    with pytest.raises(OutOfDomainError, match="nan"):
        nd.evaluate(np.nan, 1.0)


# ------------------------------------------------------------- error_by_axis


def test_error_by_axis_shapes_and_maximum():
    domains = [(0.0, 2.0), (-1.0, 3.0), (1.0, 4.0)]
    _, vals = sample_on_grid(lambda x, y, z: np.sin(x) * np.cos(y) * np.exp(-z), domains, (14, 22, 9))
    interp = ChebyshevInterpolantND(vals, domains)
    per_axis = interp.error_by_axis()
    assert per_axis.shape == (3,)
    assert per_axis.max() == pytest.approx(interp.absolute_error(), rel=1e-15)
    # z has the fewest nodes for its variation, so it limits the accuracy
    assert np.argmax(per_axis) == 2
    np.testing.assert_allclose(interp.error_by_axis(relative=True),
                               per_axis / np.abs(interp.get_coefficients()).max())
    multi = ChebyshevInterpolantND(np.stack([vals, 2 * vals], axis=-1), domains, output_names=["a", "b"])
    np.testing.assert_allclose(multi.error_by_axis()[:, 1], 2 * multi.error_by_axis()[:, 0])
    np.testing.assert_allclose(multi.error_by_axis(relative=True)[:, 0],
                               multi.error_by_axis(relative=True)[:, 1])


def test_error_by_axis_on_exact_polynomial_axis():
    # Linear in x: with 3 nodes the top genuine coefficient sits in the
    # second-to-last slice and is reported; one more node shows the series ended
    f = lambda x, y: (1.0 + 0.5 * x) * np.exp(np.sin(y))
    domains = [(0.0, 1.0), (0.0, 2.0)]
    tight = ChebyshevInterpolantND(sample_on_grid(f, domains, (3, 30))[1], domains)
    spare = ChebyshevInterpolantND(sample_on_grid(f, domains, (4, 30))[1], domains)
    assert tight.error_by_axis(relative=True)[0] > 1e-2
    assert spare.error_by_axis(relative=True)[0] < 1e-14
    assert spare.error_by_axis(relative=True)[1] < 1e-12


def test_error_by_axis_of_derivative_skips_zero_top_coefficients():
    domains = [(0.0, 2.0), (-1.0, 3.0)]
    _, vals = sample_on_grid(lambda x, y: np.sin(x) * np.cos(y), domains, (16, 20))
    interp = ChebyshevInterpolantND(vals, domains)
    deriv = interp.derivative((2, 0))
    per_axis = deriv.error_by_axis()
    # The derivative's last two x coefficients are exactly zero; the estimate
    # comes from the slices before them, not from those zeros
    assert np.all(deriv.get_coefficients(full=True)[-2:] == 0)
    assert per_axis[0] > 0
    assert per_axis[0] <= deriv.absolute_error()
