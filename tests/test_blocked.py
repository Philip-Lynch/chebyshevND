"""Tests for per-output truncation and blocked storage."""

import itertools

import numpy as np
import pytest
from numba import njit

from chebyshevnd import (
    AxisMap,
    ChebyshevBatch,
    ChebyshevInterpolantND,
    MappedBatch,
    MappedInterpolant,
    RawBlockedTable,
    RawMappedBlockedTable,
    RawTable,
    chebyshev_nodes,
    chebyshev_values_at_nodes_nd,
    eval_blocked_into,
    eval_grad_blocked_into,
    eval_grad_physical_blocked_into,
    eval_physical_blocked_into,
    prune_trailing_slices_greedy_nd,
)
from chebyshevnd.blocks import partition_outputs
from helpers import decaying_coeffs
import test_mapped
from test_mapped import DOMAINS, MAPS, SHAPE, F, G, fit_mapped, make_boundary, random_physical

SHAPES = {1: (14,), 2: (12, 9), 3: (10, 9, 7), 4: (8, 7, 6, 5), 5: (6, 6, 5, 4, 4)}
NAMES = ["x", "y", "z", "u", "v"]
#: Two families of harmonics; member n has relative size 10^-(2n) (n = 5 is
#: below every budget used here, so it is stored in no block)
FAMILIES = (("A", 1.0), ("B", 50.0))
HARMONICS = 6


def family_table(d, dtype=float, seed=0, noise=1e-13):
    """
    A table whose outputs are the harmonics of two families: decaying
    series, each at its own rate, plus noise at ``noise`` times its family's
    scale. Returns the table and one group label per output.
    """
    rng = np.random.default_rng(seed)
    shape = SHAPES[d]
    coeffs, names, labels = [], [], []
    for family, amp in FAMILIES:
        for n in range(HARMONICS):
            # Converged on the grid (0.08^9 ~ 1e-10), each harmonic at its own rate
            base = 0.08 + 0.03 * n
            c = amp * 10.0 ** (-2 * n) * decaying_coeffs(rng, shape, base=base)
            if dtype is complex:
                c = c + 1j * amp * 10.0 ** (-2 * n) * decaying_coeffs(rng, shape, base=base)
            c = c + amp * noise * rng.standard_normal(shape)
            coeffs.append(c)
            names.append(f"{family}_{n}")
            labels.append(family)
    coeffs = np.stack(coeffs, axis=-1)
    domains = [(-1.0 + 0.5 * i, 1.0 + i) for i in range(d)]
    vals = chebyshev_values_at_nodes_nd(coeffs, axes=range(d))
    table = ChebyshevInterpolantND(vals, domains, axis_names=NAMES[:d], output_names=names)
    return table, labels


def dense_like(table):
    """A dense table with exactly the coefficients a (blocked) table stores, zero elsewhere."""
    return ChebyshevInterpolantND._from_tables(
        table.get_coefficients().reshape(table.kmax + (table.n_outputs,)), None,
        *np.array(table.domains).T, nodes=table.nodes, axis_names=table.axis_names,
        output_names=table.output_names, domain_tol=table.domain_tol,
        abs_error=np.zeros(table.n_outputs), grid_shape=table.grid_shape)


def random_points(table, n=40, seed=1):
    lo, hi = np.array(table.domains).T
    return np.random.default_rng(seed).uniform(lo, hi, size=(n, table.ndim))


def group_scale(table, labels):
    """The largest coefficient magnitude of each output's group, per output."""
    c = np.abs(table.get_coefficients(full=True)).reshape(-1, table.n_outputs).max(axis=0)
    return np.array([max(c[j] for j in range(len(labels)) if labels[j] == g) for g in labels])


def assert_close(blocked, dense, scale):
    """Agree to 1e-14 of each output's group scale (exactly zero where both are)."""
    blocked, dense = np.asarray(blocked), np.asarray(dense)
    assert blocked.shape == dense.shape
    np.testing.assert_array_less(np.abs(blocked - dense), 1e-14 * scale + 1e-300)


def blocked_table(d, dtype=float, max_blocks=2, rel_tol=1e-8, norm="l2", keep_full=False):
    table, labels = family_table(d, dtype)
    table.optimize(rel_tol, keep_full=keep_full, groups=labels, norm=norm, per_output=True,
                   max_blocks=max_blocks)
    return table, labels


# ------------------------------------------------------ the default is today's


def old_optimize(table, rel_tol):
    """optimize() as it was before groups, norms and blocks (dev at 094ff24)."""
    abs_c = np.abs(table.get_coefficients(full=True))
    abs_c = abs_c.reshape(table.grid_shape + (table.n_outputs,))
    n_out = abs_c.shape[-1]
    scale = abs_c.reshape(-1, n_out).max(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    magnitude = (abs_c / scale).max(axis=-1)
    kmax, _ = prune_trailing_slices_greedy_nd(magnitude, delta=rel_tol, scale=1.0)
    kmax = tuple(max(1, int(k)) for k in kmax)
    kept = abs_c[tuple(slice(k) for k in kmax)].reshape(-1, n_out).sum(axis=0)
    total = abs_c.reshape(-1, n_out).sum(axis=0)
    return kmax, np.maximum(total - kept, 0.0)


@pytest.mark.parametrize("d", [1, 3, 5])
@pytest.mark.parametrize("rel_tol", [1e-4, 1e-8, 1e-12])
def test_default_optimize_is_unchanged(d, rel_tol):
    table, _ = family_table(d)
    interp_error = table.absolute_error()
    kmax, discarded = old_optimize(table, rel_tol)
    full = table.get_coefficients(full=True)
    table.optimize(rel_tol)
    assert table.kmax == kmax
    assert len(table.blocks) == 1 and isinstance(table.raw(), RawTable)
    np.testing.assert_array_equal(table.get_coefficients(), full[tuple(slice(k) for k in kmax)])
    np.testing.assert_array_equal(table.absolute_error(), interp_error + discarded)
    assert "blocks" not in repr(table)


# ------------------------------------------------ blocked against dense tables


@pytest.mark.parametrize("d", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("dtype", [float, complex])
def test_blocked_evaluation_matches_dense(d, dtype):
    table, labels = blocked_table(d, dtype)
    ref = dense_like(table)
    scale = np.abs(table.get_coefficients()).max()
    pts = random_points(table)

    assert_close([table.evaluate(*p) for p in pts], [ref.evaluate(*p) for p in pts], scale)
    assert_close(table.evaluate_batch(pts), ref.evaluate_batch(pts), scale)
    for p in pts[:8]:
        (v, g), (rv, rg) = table.value_and_grad(*p), ref.value_and_grad(*p)
        assert_close(v, rv, scale)
        assert_close(g, rg, 1e2 * scale)
    for got, want in zip(table.value_and_grad_batch(pts), ref.value_and_grad_batch(pts)):
        assert_close(got, want, 1e2 * scale)


@pytest.mark.parametrize("d", [2, 3, 5])
@pytest.mark.parametrize("dtype", [float, complex])
def test_blocked_derivative_matches_dense(d, dtype):
    table, _ = blocked_table(d, dtype)
    ref = dense_like(table)
    pts = random_points(table)
    scale = np.abs(table.get_coefficients()).max()
    for orders in ([1] + [0] * (d - 1), [0] * (d - 1) + [2], [1] * d):
        got, want = table.derivative(orders), ref.derivative(orders)
        assert_close(got.evaluate_batch(pts), want.evaluate_batch(pts), 1e4 * scale)
        assert_close([got.evaluate(*p) for p in pts[:5]], [want.evaluate(*p) for p in pts[:5]],
                     1e4 * scale)


@pytest.mark.parametrize("d", [2, 3, 5])
@pytest.mark.parametrize("dtype", [float, complex])
def test_blocked_precontraction_matches_dense(d, dtype):
    table, _ = blocked_table(d, dtype)
    ref = dense_like(table)
    pts = random_points(table)
    scale = np.abs(table.get_coefficients()).max()
    fixings = [{"x": 0.1}, {NAMES[d - 1]: pts[0, d - 1]}]
    if d >= 3:
        fixings.append({"x": 0.1, "z": pts[1, 2]})
    for fixed in fixings:
        live = [i for i, n in enumerate(table.axis_names) if n not in fixed]
        got, want = table.precontract(fixed), ref.precontract(fixed)
        assert not got._dense and got.n_outputs == table.n_outputs
        np.testing.assert_array_equal(got.output_kmax[:, 0] == 0, table.output_kmax[:, 0] == 0)
        assert_close(got.evaluate_batch(pts[:, live]), want.evaluate_batch(pts[:, live]), scale)
        assert_close([got.evaluate(*p) for p in pts[:5, live]],
                     [want.evaluate(*p) for p in pts[:5, live]], scale)

        batch = {name: pts[:, table.axis_names.index(name)] for name in fixed}
        got_b, want_b = table.precontract_batch(batch), ref.precontract_batch(batch)
        assert isinstance(got_b, ChebyshevBatch) and got_b.n_blocks == len(table.blocks)
        assert_close(got_b.evaluate(pts[:, live]), want_b.evaluate(pts[:, live]), scale)
        for g, w in zip(got_b.value_and_grad(pts[:, live]), want_b.value_and_grad(pts[:, live])):
            assert_close(g, w, 1e2 * scale)
        np.testing.assert_array_equal(got_b.tables, np.asarray(want_b.tables))


def test_precontraction_with_the_full_tensor_can_be_optimized_again():
    table, labels = blocked_table(3, keep_full=True)
    child = table.precontract({"x": 0.3})
    assert child.get_coefficients(full=True).shape == (9, 7, table.n_outputs)
    child.optimize(1e-6, groups=labels, per_output=True)
    pts = random_points(child)
    exact = table.precontract({"x": 0.3})
    err = np.abs(child.evaluate_batch(pts) - exact.evaluate_batch(pts)).max(axis=0)
    np.testing.assert_array_less(err, child.absolute_error() + 1e-15)


def test_single_block_and_one_block_per_output():
    for max_blocks in (1, None):
        table, _ = blocked_table(3, max_blocks=max_blocks)
        stored = np.flatnonzero(table.output_kmax[:, 0] > 0)
        assert len(table.blocks) == (1 if max_blocks == 1 else len(stored))
        ref = dense_like(table)
        pts = random_points(table)
        assert_close(table.evaluate_batch(pts), ref.evaluate_batch(pts), 1e2)
        assert_close([table.evaluate(*p) for p in pts], [ref.evaluate(*p) for p in pts], 1e2)


# ----------------------------------------------------------- what is stored


def test_layout_properties():
    table, labels = blocked_table(3, max_blocks=3)
    blocks = table.blocks
    assert 1 <= len(blocks) <= 3
    seen = np.concatenate([o for o, _ in blocks])
    assert len(set(seen.tolist())) == len(seen)  # no output in two blocks
    kmax = table.output_kmax
    for outputs, coeffs in blocks:
        assert not coeffs.flags.writeable and not outputs.flags.writeable
        assert np.all(np.diff(outputs) > 0)
        np.testing.assert_array_equal(kmax[outputs], np.broadcast_to(coeffs.shape[:-1], kmax[outputs].shape))
    empty = np.setdiff1d(np.arange(table.n_outputs), seen)
    np.testing.assert_array_equal(kmax[empty], 0)
    assert table.kmax == tuple(kmax.max(axis=0))
    assert table.effective_grid() == table.kmax
    dense = table.get_coefficients()
    assert dense.shape == table.kmax + (table.n_outputs,)
    np.testing.assert_array_equal(dense[..., empty], 0)
    text = repr(table)
    assert f"blocks={len(blocks)}" in text and "stored=" in text


def test_negligible_outputs_are_empty_and_exactly_zero():
    table, labels = blocked_table(3)
    names = table.output_names
    empty = [names[j] for j in np.flatnonzero(table.output_kmax[:, 0] == 0)]
    # Harmonic 5 of each family is 1e-10 of its family, below the budget
    assert {"A_5", "B_5"} <= set(empty)
    idx = [table.output_index[n] for n in empty]
    pts = random_points(table)
    assert np.all(table.evaluate_batch(pts)[:, idx] == 0)
    assert np.all(np.array([table.evaluate(*p) for p in pts])[:, idx] == 0)
    v, g = table.value_and_grad_batch(pts)
    assert np.all(v[:, idx] == 0) and np.all(g[..., idx] == 0)
    assert np.all(table.derivative([1, 1, 0]).evaluate_batch(pts)[:, idx] == 0)
    assert np.all(table.precontract({"y": 0.5}).evaluate_batch(pts[:, [0, 2]])[:, idx] == 0)
    # Their whole series counts as discarded
    full = np.abs(family_table(3)[0].get_coefficients(full=True))
    np.testing.assert_allclose(table._discarded[idx], full[..., idx].reshape(-1, len(idx)).sum(axis=0))


def test_a_table_of_zeros_stores_nothing():
    table = ChebyshevInterpolantND(np.zeros((6, 5, 2)), [(0, 1), (0, 1)], output_names=["a", "b"])
    table.optimize(1e-8, per_output=True)
    assert table.blocks == () and table.kmax == (0, 0)
    assert np.all(table.evaluate(0.3, 0.4) == 0)
    assert table.evaluate_batch(np.full((3, 2), 0.5)).shape == (3, 2)
    v, g = table.value_and_grad(0.3, 0.4)
    assert np.all(v == 0) and g.shape == (2, 2)
    assert np.all(table.precontract({0: 0.2}).evaluate_batch(np.full((2, 1), 0.5)) == 0)
    raw = table.raw()
    out = np.ones(2)
    assert eval_blocked_into(*raw, np.array([0.3, 0.4]), *raw.workspace(), out) == 0
    assert np.all(out == 0)


@pytest.mark.parametrize("norm", ["l1", "l2"])
def test_budgets_per_output(norm):
    rel_tol = 1e-8
    table, labels = family_table(3)
    full = table.get_coefficients(full=True)
    S = group_scale(table, labels)
    table.optimize(rel_tol, groups=labels, norm=norm, per_output=True)
    stored = np.zeros(full.shape, dtype=bool)
    for outputs, coeffs in table.blocks:
        stored[tuple(slice(k) for k in coeffs.shape[:-1]) + (outputs,)] = True
    removed = np.where(stored, 0.0, np.abs(full)).reshape(-1, table.n_outputs)
    # L1 is the total minus the kept part, so it carries rounding of the total
    total = np.abs(full).reshape(-1, table.n_outputs).sum(axis=0)
    np.testing.assert_allclose(table._discarded, removed.sum(axis=0), rtol=1e-6, atol=1e-14 * total.max())
    np.testing.assert_allclose(table.discarded_l2, np.sqrt((removed ** 2).sum(axis=0)), rtol=1e-12)
    # Within budget, against the group's scale (the pruner is conservative)
    budget = rel_tol * S
    used = table.discarded_l2 if norm == "l2" else table._discarded
    assert np.all(used <= budget * (1 + 1e-12))
    # The L2 budget prunes far more than the L1 one
    if norm == "l2":
        l1 = family_table(3)[0]
        l1.optimize(rel_tol, groups=labels, norm="l1", per_output=True)
        assert sum(c.size for _, c in table.blocks) < sum(c.size for _, c in l1.blocks)


def test_groups_against_own_scale():
    # Measured against its own scale, a tiny harmonic keeps its noise; against
    # its family's, it is dropped
    own, labels = family_table(3)
    grouped, _ = family_table(3)
    own.optimize(1e-8, per_output=True)
    grouped.optimize(1e-8, groups=labels, per_output=True)
    j = own.output_index["A_5"]
    assert own.output_kmax[j, 0] > 0 and grouped.output_kmax[j, 0] == 0


def test_shared_truncation_with_groups_and_l2():
    # per_output=False: one box for all, normalised by the group scale
    table, labels = family_table(3)
    full = table.get_coefficients(full=True)
    table.optimize(1e-8, groups=labels, norm="l2")
    assert len(table.blocks) == 1 and isinstance(table.raw(), RawTable)
    S = group_scale(table, labels)
    assert np.all(table.discarded_l2 <= 1e-8 * S)
    plain, _ = family_table(3)
    plain.optimize(1e-8)
    assert np.prod(table.kmax) < np.prod(plain.kmax)
    np.testing.assert_array_equal(table.get_coefficients(), full[tuple(slice(k) for k in table.kmax)])


def test_groups_as_a_mapping():
    a, labels = family_table(2)
    b, _ = family_table(2)
    mapping = {"A": [n for n in a.output_names if n.startswith("A")],
               "B": [j for j, n in enumerate(a.output_names) if n.startswith("B")]}
    a.optimize(1e-8, groups=labels, per_output=True, max_blocks=2)
    b.optimize(1e-8, groups=mapping, per_output=True, max_blocks=2)
    np.testing.assert_array_equal(a.output_kmax, b.output_kmax)
    assert b._truncation["groups"] == tuple(labels)


@pytest.mark.parametrize("kwargs, error", [
    ({"norm": "max"}, ValueError),
    ({"max_blocks": 2}, ValueError),  # without per_output
    ({"per_output": True, "max_blocks": 0}, ValueError),
    ({"per_output": True, "max_blocks": 2.5}, ValueError),
    ({"groups": "AB"}, TypeError),
    ({"groups": ["A"]}, ValueError),
    ({"groups": [["unhashable"]] * 12}, TypeError),
    ({"groups": {"A": ["A_0"], "B": ["A_0"]}}, ValueError),
    ({"groups": {"A": ["A_0"]}}, ValueError),
    ({"groups": {"A": ["nope"]}}, ValueError),
    ({"groups": {"A": [99]}}, ValueError),
    ({"groups": {"A": "A_0"}}, TypeError),
])
def test_optimize_argument_errors(kwargs, error):
    table, _ = family_table(2)
    with pytest.raises(error):
        table.optimize(1e-8, **kwargs)


def test_optimize_starts_again_from_the_full_tensor():
    table, labels = blocked_table(3, keep_full=True)
    table.optimize(1e-8)
    plain, _ = family_table(3)
    plain.optimize(1e-8)
    assert table.kmax == plain.kmax and len(table.blocks) == 1
    np.testing.assert_array_equal(table.get_coefficients(), plain.get_coefficients())
    np.testing.assert_array_equal(table.discarded_l2, plain.discarded_l2)


# -------------------------------------------------------------- error bounds


@pytest.mark.parametrize("norm", ["l1", "l2"])
@pytest.mark.parametrize("dtype", [float, complex])
def test_error_bounds_at_random_points(norm, dtype):
    exact, labels = family_table(3, dtype)
    table, _ = family_table(3, dtype)
    table.optimize(1e-7, groups=labels, norm=norm, per_output=True, max_blocks=3)
    pts = random_points(table, n=400, seed=5)
    err = np.abs(table.evaluate_batch(pts) - exact.evaluate_batch(pts)).max(axis=0)
    # L1: rigorous (plus rounding)
    np.testing.assert_array_less(err, table._discarded + 1e-14 * group_scale(exact, labels))
    # L2: an estimate of the typical error, not a bound; on a noise tail the
    # largest error over many points stays within a few times it
    assert np.all(err <= 10 * table.discarded_l2 + 1e-14 * group_scale(exact, labels))


def test_derivative_bounds_without_the_full_tensor():
    exact, labels = family_table(2)
    table, _ = family_table(2)
    table.optimize(1e-8, groups=labels, per_output=True, max_blocks=2, keep_full=False)
    pts = random_points(table, n=200, seed=2)
    for orders in ([1, 0], [0, 2]):
        err = np.abs(table.derivative(orders).evaluate_batch(pts)
                     - exact.derivative(orders).evaluate_batch(pts)).max(axis=0)
        np.testing.assert_array_less(err, table.derivative(orders).absolute_error())


def test_error_by_axis_without_the_full_tensor():
    table, _ = blocked_table(3, keep_full=False)
    ref = dense_like(table)
    err = table.error_by_axis()
    assert err.shape == (3, table.n_outputs)
    empty = table.output_kmax[:, 0] == 0
    assert np.all(err[:, empty] == 0)
    np.testing.assert_allclose(err[:, ~empty], ref.error_by_axis()[:, ~empty], rtol=1e-12)


# ------------------------------------------------------------- the partition


def brute_force_cost(kmax, max_blocks):
    """The cheapest split of the volume-sorted outputs into at most max_blocks runs."""
    order = np.argsort(np.prod(kmax, axis=1), kind="stable")
    k = kmax[order]
    n = len(k)
    best = np.inf
    for b in range(1, min(max_blocks, n) + 1):
        for cuts in itertools.combinations(range(1, n), b - 1):
            edges = (0,) + cuts + (n,)
            cost = sum((j - i) * np.prod(k[i:j].max(axis=0)) for i, j in zip(edges, edges[1:]))
            best = min(best, cost)
    return best


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("max_blocks", [1, 2, 3, 4])
def test_partition_is_optimal_over_sorted_runs(seed, max_blocks):
    rng = np.random.default_rng(seed)
    kmax = rng.integers(1, 9, size=(7, 3))
    runs = partition_outputs(kmax, max_blocks)
    assert len(runs) <= max_blocks
    np.testing.assert_array_equal(np.sort(np.concatenate(runs)), np.arange(7))
    cost = sum(len(r) * np.prod(kmax[r].max(axis=0)) for r in runs)
    assert cost == brute_force_cost(kmax, max_blocks)


def test_partition_one_block_per_output():
    kmax = np.array([[3, 3], [1, 2], [5, 1]])
    runs = partition_outputs(kmax, None)
    assert [r.tolist() for r in runs] == [[1], [2], [0]]
    assert len(partition_outputs(kmax, 10)) == 3


# ------------------------------------------------------------------- Numba


def test_raw_of_a_blocked_table():
    table, _ = blocked_table(3, dtype=complex)
    raw = table.raw()
    assert isinstance(raw, RawBlockedTable) and raw.n_blocks == len(table.blocks)
    assert raw.ndim == 3 and raw.n_outputs == table.n_outputs and raw.dtype == table.dtype
    assert not raw.coeffs.flags.writeable
    with pytest.raises(ValueError, match="blocks"):
        table.raw(blocked=False)
    ws = raw.workspace()
    out = np.empty(raw.n_outputs, dtype=raw.dtype)
    grad = np.empty((4, raw.n_outputs), dtype=raw.dtype)
    for p in random_points(table, 5):
        status = eval_blocked_into(*raw, p, *ws, out)
        assert status == 0
        np.testing.assert_array_equal(out, table.evaluate(*p))
        eval_grad_blocked_into(*raw, p, *ws, grad)
        v, g = table.value_and_grad(*p)
        np.testing.assert_array_equal(grad[0], v)
        np.testing.assert_array_equal(grad[1:], g)
    assert eval_blocked_into(*raw, np.array([5.0, 0.0, 0.0]), *ws, out) == 1


def test_raw_blocked_for_a_dense_table():
    table, _ = family_table(2)
    table.optimize(1e-8)
    raw = table.raw(blocked=True)
    assert isinstance(raw, RawBlockedTable) and raw.n_blocks == 1
    out = np.empty(raw.n_outputs)
    p = random_points(table, 1)[0]
    eval_blocked_into(*raw, p, *raw.workspace(), out)
    np.testing.assert_array_equal(out, table.evaluate(*p))


@njit
def user_rhs(coeffs, offsets, shapes, out_index, out_offsets, n_outputs, bounds, tol, x, scratch,
             work, out, grad):
    s = eval_blocked_into(coeffs, offsets, shapes, out_index, out_offsets, n_outputs, bounds, tol, x,
                          scratch, work, out)
    s |= eval_grad_blocked_into(coeffs, offsets, shapes, out_index, out_offsets, n_outputs, bounds, tol,
                                x, scratch, work, grad)
    return s


def test_blocked_entry_points_from_user_njit():
    table, _ = blocked_table(2)
    child = table.precontract({"x": 0.2})
    raw = child.raw()
    ws = raw.workspace()
    out = np.empty(raw.n_outputs)
    grad = np.empty((2, raw.n_outputs))
    x = np.array([1.1])
    assert user_rhs(*raw, x, *ws, out, grad) == 0
    np.testing.assert_array_equal(out, child.evaluate(1.1))
    np.testing.assert_array_equal(grad[1], child.value_and_grad(1.1)[1][0])


def test_blocked_entry_points_check_their_arguments():
    table, _ = blocked_table(3)
    raw = table.raw()
    scratch, work = raw.workspace()
    x = random_points(table, 1)[0]
    out = np.empty(raw.n_outputs)
    with pytest.raises(ValueError, match="one coordinate per axis"):
        eval_blocked_into(*raw, x[:2], scratch, work, out)
    with pytest.raises(ValueError, match="one entry per output"):
        eval_blocked_into(*raw, x, scratch, work, out[:-1])
    with pytest.raises(ValueError, match="workspace too small"):
        eval_blocked_into(*raw, x, scratch, work[:, :1], out)
    with pytest.raises(ValueError, match="workspace too small"):
        eval_grad_blocked_into(*raw, x, scratch, work[:6], np.empty((4, raw.n_outputs)))
    with pytest.raises(ValueError, match=r"shape \(d \+ 1, F\)"):
        eval_grad_blocked_into(*raw, x, scratch, work, np.empty((3, raw.n_outputs)))
    bad = raw._replace(out_index=np.full_like(raw.out_index, raw.n_outputs))
    with pytest.raises(ValueError, match="inconsistent"):
        eval_blocked_into(*bad, x, scratch, work, out)


# ------------------------------------------------------------ mapped tables

MAPPED_FUNCS = {
    float: [F, lambda *x: 1e-3 * F(*x) * np.cos(x[1]), lambda *x: 1e-6 * F(*x) ** 2,
            lambda *x: 1e-14 * F(*x)],
    complex: [G, F, lambda *x: 1e-4 * G(*x) ** 2, lambda *x: 1e-14 * G(*x)],
}


def mapped_pair(dtype, boundary, max_blocks=2):
    """A blocked MappedInterpolant, and the dense one with the same coefficients."""
    names = ["r", "p", "e", "q"]
    m = fit_mapped(names, SHAPE, DOMAINS, MAPS, boundary, MAPPED_FUNCS[dtype],
                   output_names=["f0", "f1", "f2", "f3"])
    m.optimize(1e-9, groups=[0, 0, 0, 0], norm="l2", per_output=True, max_blocks=max_blocks,
               keep_full=False)
    return m, MappedInterpolant(dense_like(m.table), MAPS, boundary)


@pytest.fixture(scope="module")
def boundary():
    return make_boundary()


@pytest.mark.parametrize("dtype", [float, complex])
def test_mapped_blocked_matches_dense(dtype, boundary):
    m, ref = mapped_pair(dtype, boundary)
    assert not m.table._dense and m.table.output_kmax[3, 0] == 0  # the 1e-14 output is empty
    scale = np.abs(m.table.get_coefficients()).max()
    x = random_physical(30, seed=3, margin=0.01)
    assert_close([m.evaluate(*p) for p in x], [ref.evaluate(*p) for p in x], scale)
    for p in x[:6]:
        (v, g), (rv, rg) = m.value_and_grad(*p), ref.value_and_grad(*p)
        assert_close(v, rv, scale)
        assert_close(g, rg, 1e3 * scale)
    assert_close(m.evaluate_batch(x), ref.evaluate_batch(x), scale)
    for got, want in zip(m.value_and_grad_batch(x), ref.value_and_grad_batch(x)):
        assert_close(got, want, 1e3 * scale)
    for got, want in zip(m.value_grad_hess_batch(x[:8]), ref.value_grad_hess_batch(x[:8])):
        assert_close(got, want, 1e6 * scale)
    assert np.all(m.evaluate_batch(x)[:, 3] == 0)


def physical_points(n, seed, r=None, e=None):
    """Random physical points in the mapped region, with r and/or e fixed if given."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(6.0, 16.0, n) if r is None else np.full(n, r)
    e = rng.uniform(0.0, 0.5, n) if e is None else np.full(n, e)
    lo, width, sep = (test_mapped.lo(r, e), test_mapped.w(r, e), test_mapped.sep(r, e))
    p = lo + width * rng.uniform(0.01, 0.99, n)
    q = sep + (5 - sep) * rng.uniform(0.01, 0.99, n)
    return np.stack([r, p, e, q], axis=1)


@pytest.mark.parametrize("dtype", [float, complex])
@pytest.mark.parametrize("fixed_axes", [("r", "e"), ("e",), ("r", "e", "q")])
def test_mapped_blocked_precontraction(dtype, fixed_axes, boundary):
    m, ref = mapped_pair(dtype, boundary)
    scale = np.abs(m.table.get_coefficients()).max()
    names = ["r", "p", "e", "q"]
    cols = [names.index(a) for a in fixed_axes]
    live = [i for i in range(4) if i not in cols]
    # Both boundary inputs fixed (the boundary becomes numbers), or only e
    # (the boundary table is precontracted too)
    x0 = physical_points(1, seed=4)[0]
    x = physical_points(30, seed=5, r=x0[0] if "r" in fixed_axes else None,
                        e=x0[2] if "e" in fixed_axes else None)
    if "q" in fixed_axes:
        x[:, 3] = x0[3]  # inside the window: r and e are fixed at x0's values too
    fixed = {a: float(x0[c]) for a, c in zip(fixed_axes, cols)}
    got, want = m.precontract(fixed), ref.precontract(fixed)
    assert not got.table._dense
    pts = x[:, live]
    assert_close(got.evaluate_batch(pts), want.evaluate_batch(pts), scale)
    assert_close([got.evaluate(*p) for p in pts[:4]], [want.evaluate(*p) for p in pts[:4]], scale)
    for g, w in zip(got.value_and_grad_batch(pts), want.value_and_grad_batch(pts)):
        assert_close(g, w, 1e3 * scale)

    members = physical_points(30, seed=6)
    batch = {a: members[:, c] for a, c in zip(fixed_axes, cols)}
    got_b, want_b = m.precontract_batch(batch), ref.precontract_batch(batch)
    assert isinstance(got_b, MappedBatch)
    assert_close(got_b.evaluate(members[:, live]), want_b.evaluate(members[:, live]), scale)
    for g, w in zip(got_b.value_and_grad(members[:, live]), want_b.value_and_grad(members[:, live])):
        assert_close(g, w, 1e3 * scale)


@njit
def user_mapped(coeffs, offsets, shapes, out_index, out_offsets, n_outputs, bounds, tol, codes,
                params, bcoeffs, bshape, bbounds, baxes, x, scratch, work, out, grad):
    s = eval_physical_blocked_into(coeffs, offsets, shapes, out_index, out_offsets, n_outputs, bounds,
                                   tol, codes, params, bcoeffs, bshape, bbounds, baxes, x, scratch,
                                   work, out)
    s |= eval_grad_physical_blocked_into(coeffs, offsets, shapes, out_index, out_offsets, n_outputs,
                                         bounds, tol, codes, params, bcoeffs, bshape, bbounds, baxes,
                                         x, scratch, work, grad)
    return s


def test_mapped_blocked_raw_from_user_njit(boundary):
    m, _ = mapped_pair(complex, boundary)
    raw = m.raw()
    assert isinstance(raw, RawMappedBlockedTable) and raw.n_blocks == len(m.table.blocks)
    ws = raw.workspace()
    out = np.empty(4, dtype=complex)
    grad = np.empty((5, 4), dtype=complex)
    for p in random_physical(4, seed=6, margin=0.01):
        assert user_mapped(*raw, p, *ws, out, grad) == 0
        v, g = m.value_and_grad(*p)
        np.testing.assert_array_equal(out, m.evaluate(*p))
        np.testing.assert_array_equal(grad[0], v)
        np.testing.assert_array_equal(grad[1:], g)
    with pytest.raises(ValueError, match="workspace too small"):
        eval_physical_blocked_into(*raw, p, ws.scratch[:5], ws.work, out)
    with pytest.raises(ValueError):
        m.raw(blocked=False)


def test_mapped_raw_blocked_for_a_dense_table(boundary):
    names = ["r", "p", "e", "q"]
    m = fit_mapped(names, SHAPE, DOMAINS, MAPS, boundary, [F, G], output_names=["F", "G"])
    raw = m.raw(blocked=True)
    assert isinstance(raw, RawMappedBlockedTable)
    out = np.empty(2, dtype=complex)
    p = random_physical(1, seed=2, margin=0.01)[0]
    eval_physical_blocked_into(*raw, p, *raw.workspace(), out)
    np.testing.assert_allclose(out, m.evaluate(*p), rtol=0, atol=1e-15)


def test_mapped_without_boundary():
    u = chebyshev_nodes(20, 1 / 16, 1 / 6)
    e = chebyshev_nodes(9, 0.0, 0.5)
    uu, ee = np.meshgrid(u, e, indexing="ij")
    vals = np.stack([np.sin(2 * uu) * (1 + ee), 1e-5 * uu * ee, 1e-15 * uu], axis=-1)
    table = ChebyshevInterpolantND(vals, [(1 / 16, 1 / 6), (0.0, 0.5)], axis_names=["r", "e"],
                                   output_names=["a", "b", "c"])
    m = MappedInterpolant(table, {"r": AxisMap("power", power=-1)})
    dense = MappedInterpolant(ChebyshevInterpolantND(vals, table.domains, axis_names=["r", "e"],
                                                     output_names=["a", "b", "c"]),
                              {"r": AxisMap("power", power=-1)})
    m.optimize(1e-10, groups=["g"] * 3, per_output=True)
    assert m.table.output_kmax[2, 0] == 0
    x = np.stack([np.linspace(6.5, 15.5, 7), np.linspace(0.05, 0.45, 7)], axis=1)
    np.testing.assert_allclose(m.evaluate_batch(x), dense.evaluate_batch(x), rtol=0, atol=1e-9)
    np.testing.assert_allclose([m.evaluate(*p) for p in x], dense.evaluate_batch(x), rtol=0, atol=1e-9)


def test_mapped_caches_follow_the_table(boundary):
    names = ["r", "p", "e", "q"]
    m = fit_mapped(names, SHAPE, DOMAINS, MAPS, boundary, MAPPED_FUNCS[float],
                   output_names=["f0", "f1", "f2", "f3"])
    p = random_physical(1, seed=8, margin=0.01)[0]
    before = m.evaluate(*p)
    assert isinstance(m.raw(), type(m.raw(blocked=False)))
    # Optimizing the wrapped table directly, after evaluating, must not
    # leave the wrapper evaluating the old tables
    m.table.optimize(1e-9, groups=[0] * 4, per_output=True)
    assert isinstance(m.raw(), RawMappedBlockedTable)
    after = m.evaluate(*p)
    assert after[3] == 0 and before[3] != 0
    np.testing.assert_allclose(after[:3], before[:3], rtol=0, atol=1e-8)
    np.testing.assert_array_equal(m.evaluate_batch(p[None])[0], after)


def test_blocked_boundary_tables_are_refused(boundary):
    blocked = make_boundary()
    blocked.optimize(1e-6, per_output=True)
    names = ["r", "p", "e", "q"]
    table = fit_mapped(names, SHAPE, DOMAINS, MAPS, boundary, [F]).table
    with pytest.raises(NotImplementedError, match="boundary"):
        MappedInterpolant(table, MAPS, blocked)


# --------------------------------------------------------------------- GPU


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
@pytest.mark.parametrize("dtype", [float, complex])
def test_gpu_blocked_tables(cupy, dtype):
    from chebyshevnd.backend import to_numpy

    table, _ = blocked_table(3, dtype)
    table.to_device("cuda")
    pts = random_points(table)
    gpu = table.evaluate_batch(cupy.asarray(pts))
    np.testing.assert_allclose(to_numpy(gpu), table.evaluate_batch(pts), rtol=1e-12, atol=1e-14)
    for g, c in zip(table.value_and_grad_batch(cupy.asarray(pts)), table.value_and_grad_batch(pts)):
        np.testing.assert_allclose(to_numpy(g), c, rtol=1e-12, atol=1e-12)
    batch = table.precontract_batch({"x": cupy.asarray(pts[:, 0])})
    host = table.precontract_batch({"x": pts[:, 0]})
    np.testing.assert_allclose(to_numpy(batch.evaluate(cupy.asarray(pts[:, 1:]))),
                               host.evaluate(pts[:, 1:]), rtol=1e-12, atol=1e-14)
