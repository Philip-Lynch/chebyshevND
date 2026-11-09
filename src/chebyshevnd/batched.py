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

from .backend import array_namespace

#: Memory allowed for the intermediate array of one chunk when evaluating a
#: shared table. Its size is (chunk size) x (table size / k_1) entries.
DEFAULT_MAX_CHUNK_BYTES = 1 << 28


def chebyshev_basis_batch(ts, n, xp):
    """
    Chebyshev polynomials T_0, ..., T_{n-1} at each point of ``ts``.

    Parameters
    ----------
    ts : array, shape (B,)
        Points in [-1, 1].
    n : int
        Number of polynomials.
    xp : module
        ``numpy`` or ``cupy``.

    Returns
    -------
    array, shape (B, n)
        ``T[b, k] = T_k(ts[b])``, from the three-term recurrence, which keeps
        full accuracy near +-1 (unlike ``cos(k * arccos(t))``).
    """
    T = xp.empty((ts.shape[0], n), dtype=ts.dtype)
    T[:, 0] = 1.0
    if n > 1:
        T[:, 1] = ts
        for k in range(2, n):
            T[:, k] = 2.0 * ts * T[:, k - 1] - T[:, k - 2]
    return T


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
    """
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

    k = coeffs.shape[1:-1] if per_member else coeffs.shape[:-1]
    n_out = coeffs.shape[-1]
    if n_points == 0:
        return xp.empty((0, n_out), dtype=dtype), status

    # Interleave real and imaginary parts so every product below is real
    table = coeffs.view(xp.float64) if is_complex else coeffs

    if per_member:
        values = _contract_axes(table.reshape(n_points, k[0], -1), ts, k, 0, xp)
    else:
        flat = table.reshape(k[0], -1)
        rows = max(1, max_chunk_bytes // (flat.shape[1] * flat.itemsize))
        chunks = []
        for start in range(0, n_points, rows):
            t = ts[start:start + rows]
            # The first axis of a shared table is one GEMM over the chunk
            partial = chebyshev_basis_batch(t[:, 0], k[0], xp) @ flat
            chunks.append(_contract_axes(partial, t, k, 1, xp))
        values = chunks[0] if len(chunks) == 1 else xp.concatenate(chunks, axis=0)

    values = xp.ascontiguousarray(values)
    if is_complex:
        values = values.view(xp.complex128)
    return values, status


def _contract_axes(partial, ts, k, first, xp):
    """
    Contract axes ``first, ..., d-1`` of a per-point array.

    ``partial`` has shape (B, ...), whose remaining axes flatten to
    (k[first], ..., k[d-1], F); the result has shape (B, F).
    """
    n_points = partial.shape[0]
    for ax in range(first, len(k)):
        basis = chebyshev_basis_batch(ts[:, ax], k[ax], xp)
        partial = partial.reshape(n_points, k[ax], -1)
        partial = xp.matmul(basis[:, None, :], partial)[:, 0, :]
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
