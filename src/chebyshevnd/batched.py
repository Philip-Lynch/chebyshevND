"""
Batch-first evaluation of compact Chebyshev tables, written once for NumPy
and CuPy (design doc A8).

A batch is B points, each evaluated against either one **shared** table of
shape (k_1, ..., k_d, F) or its own **per-member** table, given as one array
of shape (B, k_1, ..., k_d, F). Per-member tables arise when each waveform
in a batch fixes some axes (e.g. its own mass ratio and spins) at different
values.

The contraction follows the leading-axis-first scheme of the Numba kernel,
expressed as matrix multiplications: for a shared table, the first axis is
one GEMM over the whole batch, and every further axis is a batched
matrix-vector product. Complex tables are contracted as real arrays with
real and imaginary parts interleaved along the output axis, so every
product stays real-valued.

The functions here are pure: they return new arrays and never raise on
points outside the domain. Instead, they return a per-point status bitmask.
"""

import math
from dataclasses import dataclass
from typing import Any

from .backend import array_namespace

#: Memory allowed for the intermediate array of one chunk when evaluating a
#: shared table. Its size is (chunk size) x (table size / k_1) entries.
DEFAULT_MAX_CHUNK_BYTES = 1 << 28


def chebyshev_basis_batch(ts, n, xp, derivative=False):
    """
    Chebyshev polynomials T_0, ..., T_{n-1} at each point of ``ts``.

    Parameters
    ----------
    ts : array
        Points in [-1, 1], any shape, e.g. (B,) for one axis or (B, d) for
        every axis of a batch at once (fewer, larger array operations, which
        matters on the GPU).
    n : int
        Number of polynomials.
    xp : module
        ``numpy`` or ``cupy``.
    derivative : bool, optional
        Also return the derivatives T'_k(t).

    Returns
    -------
    T : array, shape ``ts.shape + (n,)``
        ``T[..., k] = T_k(ts)``, from the three-term recurrence, which keeps
        full accuracy near +-1.
    D : array, shape ``ts.shape + (n,)``
        Only if ``derivative``: ``D[..., k] = T'_k(ts) = k U_{k-1}(ts)``,
        with the Chebyshev polynomials of the second kind U_k from the same
        recurrence (no division, so t = +-1 needs no special case).

    Notes
    -----
    The results are views of arrays stored with k as the leading axis, so
    that each step of the recurrence is two in-place operations on
    contiguous memory: about 2n array operations for T and 2n more for D,
    each a kernel launch on the GPU. The closed forms ``T_k = cos(k theta)``
    and ``T'_k = k sin(k theta) / sin(theta)``, with ``theta = arccos(t)``,
    would need fewer. The first is as accurate as the recurrence, but the
    second loses digits near t = -1, where theta is close to pi (and
    ``sin(pi)`` is not 0 in floating point), and neither extrapolates
    outside [-1, 1].
    """
    T = _recurrence(ts, n, ts, xp)
    if not derivative:
        return xp.moveaxis(T, 0, -1)
    D = xp.empty_like(T)
    D[0] = 0.0
    if n > 1:
        U = _recurrence(ts, n - 1, 2.0 * ts, xp)
        k = xp.arange(1, n, dtype=ts.dtype).reshape((n - 1,) + (1,) * ts.ndim)
        xp.multiply(k, U, out=D[1:])
    return xp.moveaxis(T, 0, -1), xp.moveaxis(D, 0, -1)


def _recurrence(ts, n, first, xp):
    """
    P_0 = 1, P_1 = ``first``, P_k = 2 t P_{k-1} - P_{k-2}, stacked along a new
    leading axis: T_k for ``first = t``, U_k for ``first = 2 t``.
    """
    P = xp.empty((n,) + ts.shape, dtype=ts.dtype)
    P[0] = 1.0
    if n > 1:
        P[1] = first
        two_t = 2.0 * ts
        for k in range(2, n):
            xp.multiply(two_t, P[k - 1], out=P[k])
            xp.subtract(P[k], P[k - 2], out=P[k])
    return P


def evaluate_batch(coeffs, a, b, points, tol=1e-12, max_chunk_bytes=DEFAULT_MAX_CHUNK_BYTES):
    """
    Evaluate a compact Chebyshev table at a batch of points.

    Parameters
    ----------
    coeffs : array
        The coefficients, float64 or complex128, on the same device as
        ``points``. Either a shared table of shape (k_1, ..., k_d, F), or
        per-member tables of shape (B, k_1, ..., k_d, F).
    a, b : array-like, shape (d,)
        Lower and upper bound of each axis.
    points : array, shape (B, d)
        One point per row, in the coordinates of the domain.
    tol : float, optional
        Allowed overshoot outside the domain, as a fraction of each axis's
        half-width, before a point is flagged in ``status``.
    max_chunk_bytes : int, optional
        Memory budget for the intermediate array when evaluating a shared
        table; larger batches are processed in chunks.

    Returns
    -------
    values : array, shape (B, F)
        The table evaluated at each point (same dtype as ``coeffs``).
        Values for flagged points are extrapolations and should not be used.
    status : int64 array, shape (B,)
        Bit i of ``status[j]`` is set when coordinate i of point j lies
        outside its domain (beyond ``tol``) or is NaN; 0 means inside.

    See Also
    --------
    value_and_grad_batch : the values together with their gradients.
    """
    batch = _prepare(coeffs, a, b, points, tol)
    xp, ts, k = batch.xp, batch.ts, batch.k
    n_points = ts.shape[0]
    if n_points == 0:
        return xp.empty((0, batch.n_out), dtype=batch.dtype), batch.status

    if batch.per_member:
        bases = _axis_bases(ts, k, xp)
        values = _contract_axes(batch.table.reshape(n_points, k[0], -1), bases, 0, xp)
    else:
        flat = batch.table.reshape(k[0], -1)
        rows = max(1, max_chunk_bytes // (flat.shape[1] * flat.itemsize))
        chunks = []
        for start in range(0, n_points, rows):
            bases = _axis_bases(ts[start:start + rows], k, xp)
            # The first axis of a shared table is one GEMM over the chunk
            chunks.append(_contract_axes(bases[0] @ flat, bases, 1, xp))
        values = chunks[0] if len(chunks) == 1 else xp.concatenate(chunks, axis=0)
    return batch.finish(values), batch.status


def value_and_grad_batch(coeffs, a, b, points, tol=1e-12, max_chunk_bytes=DEFAULT_MAX_CHUNK_BYTES):
    """
    Evaluate a compact Chebyshev table and its gradient at a batch of points.

    The batched form of :func:`chebyshevnd.eval_grad_into`. Each
    axis is contracted with both T and T'; the first axis of the table is
    read once (a single GEMM, or batched GEMM for per-member tables), and
    the derivative branches only touch the much smaller partial sums.

    Parameters
    ----------
    coeffs, a, b, points, tol, max_chunk_bytes
        As for :func:`evaluate_batch`.

    Returns
    -------
    values : array, shape (B, F)
        The table evaluated at each point.
    grads : array, shape (B, d, F)
        ``grads[j, i]`` is the derivative of every output with respect to
        coordinate i (in the domain's units), at point j.
    status : int64 array, shape (B,)
        As for :func:`evaluate_batch`.
    """
    batch = _prepare(coeffs, a, b, points, tol)
    xp, ts, k = batch.xp, batch.ts, batch.k
    n_points, d = ts.shape
    if n_points == 0:
        return (xp.empty((0, batch.n_out), dtype=batch.dtype),
                xp.empty((0, d, batch.n_out), dtype=batch.dtype), batch.status)

    if batch.per_member:
        bases = _axis_bases(ts, k, xp, derivative=True)
        branches = _contract_axes_with_gradient(batch.table.reshape(n_points, 1, -1), bases, 0, xp)
    else:
        flat = batch.table.reshape(k[0], -1)
        # The intermediate holds two rows per point: value and first derivative
        rows = max(1, max_chunk_bytes // (2 * flat.shape[1] * flat.itemsize))
        chunks = []
        for start in range(0, n_points, rows):
            bases = _axis_bases(ts[start:start + rows], k, xp, derivative=True)
            T, D = bases[0]
            m = T.shape[0]
            # First axis: one GEMM for both, (2m, k_1) @ (k_1, rest)
            both = xp.stack([T, D], axis=1).reshape(2 * m, k[0])
            partial = (both @ flat).reshape(m, 2, -1)
            chunks.append(_contract_axes_with_gradient(partial, bases, 1, xp))
        branches = chunks[0] if len(chunks) == 1 else xp.concatenate(chunks, axis=0)

    branches = batch.finish(branches)
    scale = 2.0 / (batch.b - batch.a)
    return xp.ascontiguousarray(branches[:, 0]), branches[:, 1:] * scale[None, :, None], batch.status


@dataclass
class _Batch:
    """Validated inputs of a batched evaluation, made by :func:`_prepare`."""

    xp: Any
    table: Any  # coefficients, complex ones viewed as interleaved float64
    ts: Any  # points mapped to [-1, 1], shape (B, d)
    a: Any
    b: Any
    status: Any
    k: tuple  # coefficients per axis
    n_out: int
    per_member: bool
    is_complex: bool
    dtype: Any

    def finish(self, values):
        """Turn a contracted real array back into the table's dtype."""
        values = self.xp.ascontiguousarray(values)
        if self.is_complex:
            values = values.view(self.xp.complex128)
        return values


def _prepare(coeffs, a, b, points, tol):
    """Validate the inputs of a batched evaluation and map the points to [-1, 1]."""
    xp = array_namespace(coeffs, points)
    points = xp.asarray(points, dtype=xp.float64)
    if points.ndim != 2:
        raise ValueError(f"points must have shape (B, d), got {points.shape}.")
    n_points, d = points.shape
    coeffs = xp.asarray(coeffs)
    if coeffs.ndim == d + 1:
        per_member = False
    elif coeffs.ndim == d + 2:
        per_member = True
        if coeffs.shape[0] != n_points:
            raise ValueError(
                f"Per-member tables hold {coeffs.shape[0]} members but {n_points} points were given."
            )
    else:
        raise ValueError(
            f"coeffs must have shape (k_1, ..., k_{d}, F) or (B, k_1, ..., k_{d}, F) for "
            f"{d}-dimensional points; got {coeffs.shape}."
        )
    is_complex = coeffs.dtype.kind == "c"
    dtype = xp.complex128 if is_complex else xp.float64
    coeffs = xp.ascontiguousarray(coeffs, dtype=dtype)
    a = xp.asarray(a, dtype=xp.float64)
    b = xp.asarray(b, dtype=xp.float64)

    ts = (2.0 * points - a - b) / (b - a)
    outside = ~(xp.abs(ts) <= 1.0 + tol)
    weights = xp.asarray([1 << i for i in range(d)], dtype=xp.int64)
    status = (outside * weights).sum(axis=1)

    return _Batch(
        xp=xp, ts=ts, a=a, b=b, status=status, per_member=per_member, is_complex=is_complex,
        dtype=dtype, n_out=coeffs.shape[-1],
        k=coeffs.shape[1:-1] if per_member else coeffs.shape[:-1],
        # Interleave real and imaginary parts so every product stays real
        table=coeffs.view(xp.float64) if is_complex else coeffs,
    )


def _axis_bases(ts, k, xp, derivative=False):
    """
    Per-axis Chebyshev values (and derivatives) for points ``ts`` of shape (B, d).

    All axes are computed together, up to the largest k, so the number of
    array operations does not grow with d. Returns a list with, for each
    axis, T of shape (B, k[ax]), or the pair (T, D) if ``derivative``.
    """
    n = max(k)

    def axis(P, ax):
        # Contiguous (B, k) blocks, so the matrix products below can use BLAS
        return xp.ascontiguousarray(P[:, ax, :k[ax]])

    if derivative:
        T, D = chebyshev_basis_batch(ts, n, xp, derivative=True)
        return [(axis(T, ax), axis(D, ax)) for ax in range(len(k))]
    T = chebyshev_basis_batch(ts, n, xp)
    return [axis(T, ax) for ax in range(len(k))]


def _contract_axes(partial, bases, first, xp):
    """
    Contract axes ``first, ..., d-1`` of a per-point array.

    ``partial`` has shape (B, ...), whose remaining axes flatten to
    (k[first], ..., k[d-1], F); ``bases[ax]`` has shape (B, k[ax]). The
    result has shape (B, F).
    """
    n_points = partial.shape[0]
    for ax in range(first, len(bases)):
        basis = bases[ax]
        partial = partial.reshape(n_points, basis.shape[1], -1)
        partial = xp.matmul(basis[:, None, :], partial)[:, 0, :]
    return partial


def _contract_axes_with_gradient(partial, bases, first, xp):
    """
    Contract axes ``first, ..., d-1``, keeping the derivative along each.

    ``partial`` has shape (B, n_branches, ...): branch 0 is the value and
    branch i + 1 the derivative along axis i, for every axis before
    ``first``; the remaining axes flatten to (k[first], ..., k[d-1], F).
    ``bases[ax]`` is the pair (T, D), each of shape (B, k[ax]). The value
    branch is contracted with T and D in one batched GEMM; the other
    branches with T only. The result has shape (B, d + 1, F).
    """
    n_points = partial.shape[0]
    for ax in range(first, len(bases)):
        T, D = bases[ax]
        n_branches = partial.shape[1]
        partial = partial.reshape(n_points, n_branches, T.shape[1], -1)
        both = xp.matmul(xp.stack([T, D], axis=1), partial[:, 0])  # (B, 2, rest)
        if n_branches == 1:
            partial = both
            continue
        others = xp.matmul(T[:, None, None, :], partial[:, 1:])[:, :, 0, :]
        partial = xp.concatenate([both[:, :1], others, both[:, 1:]], axis=1)
    return partial


def precontract_tables(coeffs, a, b, fixed_axes, fixed_values, tol=1e-12, status_axes=None):
    """
    Build per-member tables by fixing some axes of a shared table at
    per-member values.

    Member j's table is the shared table with axis ``fixed_axes[i]`` fixed at
    ``fixed_values[j, i]``. The fixed axes are moved to the front and each
    member's Chebyshev values on them are multiplied into one row of a
    matrix W, so every table comes out of a single GEMM,
    ``W @ table.reshape(k_fixed, rest)``. That avoids the large
    intermediate of contracting one fixed axis at a time.

    Parameters
    ----------
    coeffs : array, shape (k_1, ..., k_d, F)
        Shared table, float64 or complex128.
    a, b : array-like, shape (d,)
        Lower and upper bound of each axis.
    fixed_axes : sequence of int
        Distinct axes to fix.
    fixed_values : array, shape (B, len(fixed_axes))
        The value of each fixed axis for each member.
    tol : float, optional
        Allowed overshoot outside the domain, as a fraction of each axis's
        half-width.
    status_axes : sequence of int, optional
        The status bit to use for each fixed axis; defaults to
        ``fixed_axes``. Useful when ``coeffs`` is a reordered copy of a
        table whose own axis numbering should appear in the status.

    Returns
    -------
    tables : array, shape (B, k_live..., F)
        One table per member over the remaining axes, in their original order.
    status : int64 array, shape (B,)
        Bit ``status_axes[i]`` of ``status[j]`` is set when member j's value
        for fixed axis i lies outside its domain (beyond ``tol``) or is NaN;
        such a member's table is an extrapolation and should not be used.

    Notes
    -----
    If the fixed axes are not already the leading axes, the table is
    transposed (copied) on each call. Callers that fix the same axes
    repeatedly should store the table with those axes first.
    """
    fixed_axes = [int(ax) for ax in fixed_axes]
    xp = array_namespace(coeffs, fixed_values)
    values = xp.asarray(fixed_values, dtype=xp.float64)
    coeffs = xp.asarray(coeffs)
    d = coeffs.ndim - 1
    if values.ndim != 2 or values.shape[1] != len(fixed_axes):
        raise ValueError(
            f"fixed_values must have shape (B, {len(fixed_axes)}); got {tuple(values.shape)}."
        )
    if len(set(fixed_axes)) != len(fixed_axes) or not all(0 <= ax < d for ax in fixed_axes):
        raise ValueError(f"fixed_axes must be distinct axes in [0, {d}); got {fixed_axes}.")
    is_complex = coeffs.dtype.kind == "c"
    dtype = xp.complex128 if is_complex else xp.float64
    n_members = values.shape[0]

    a = xp.asarray(a, dtype=xp.float64)[fixed_axes]
    b = xp.asarray(b, dtype=xp.float64)[fixed_axes]
    ts = (2.0 * values - a - b) / (b - a)
    outside = ~(xp.abs(ts) <= 1.0 + tol)
    bits = fixed_axes if status_axes is None else [int(ax) for ax in status_axes]
    weights = xp.asarray([1 << ax for ax in bits], dtype=xp.int64)
    status = (outside * weights).sum(axis=1)

    live = [ax for ax in range(d) if ax not in fixed_axes]
    k_fixed = [coeffs.shape[ax] for ax in fixed_axes]
    k_live = [coeffs.shape[ax] for ax in live]
    n_out = coeffs.shape[-1]

    # Fixed axes first, then the live ones, then the output axis
    table = xp.ascontiguousarray(xp.transpose(coeffs, fixed_axes + live + [d]), dtype=dtype)
    if is_complex:
        table = table.view(xp.float64)
    flat = table.reshape(math.prod(k_fixed), -1)

    # Row j of W is the outer product of member j's bases on the fixed axes,
    # flattened in the same (C) order as the fixed axes of `flat`
    weights_matrix = chebyshev_basis_batch(ts[:, 0], k_fixed[0], xp)
    for i in range(1, len(fixed_axes)):
        basis = chebyshev_basis_batch(ts[:, i], k_fixed[i], xp)
        weights_matrix = (weights_matrix[:, :, None] * basis[:, None, :]).reshape(n_members, -1)

    tables = xp.ascontiguousarray(weights_matrix @ flat)
    if is_complex:
        tables = tables.view(xp.complex128)
    return tables.reshape((n_members,) + tuple(k_live) + (n_out,)), status
