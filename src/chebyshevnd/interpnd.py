"""n-dimensional Chebyshev interpolation."""

import numpy as np

from .kernels import evaluate_point
from .utils import compute_chebyshev_coefficients_nd, prune_trailing_slices_greedy_nd

_NODE_FAMILIES = ("Chebyshev", "GaussLobatto")


class ChebyshevInterpolantND:
    """
    Chebyshev interpolant of a function of d variables on a box
    [a_1, b_1] x ... x [a_d, b_d], for any d >= 1.

    Coefficients are computed from values sampled on a tensor grid with a
    single d-dimensional DCT. Evaluation contracts the coefficient tensor
    one axis at a time with Chebyshev polynomial values at the point, in
    compiled code.

    Evaluation reuses scratch buffers held by the instance, so one instance
    must not be evaluated from several threads at once.
    """

    def __init__(self, vals, domains, nodes="Chebyshev", axis_names=None):
        """
        Construct an nD Chebyshev interpolant from values sampled on a tensor grid.

        Parameters
        ----------
        vals : array-like
            Function values of shape (N_1, ..., N_d), sampled on the tensor
            grid of the chosen node family (``chebyshev_nodes`` per axis for
            'Chebyshev', ``chebyshev_gauss_lobatto_nodes`` for 'GaussLobatto')
            with ``np.meshgrid(..., indexing='ij')`` ordering, so that axis k
            of ``vals`` is variable k. Sample each axis at its nodes in the
            order returned (descending).
        domains : sequence of (float, float)
            One (lower, upper) pair per axis of ``vals``, in axis order: the
            same pairs used to generate the nodes. Each lower bound must be
            smaller than its upper bound.
        nodes : str, optional
            Node family the values were sampled at: 'Chebyshev' (default) or
            'GaussLobatto'.
        axis_names : sequence of str, optional
            One name per axis, e.g. ``("nu", "u", "e")``. Defaults to
            ``("x0", "x1", ...)``.
        """
        vals = np.asarray(vals)
        if np.iscomplexobj(vals):
            raise TypeError("Complex values are not supported yet.")
        vals = vals.astype(np.float64, copy=False)

        a, b = _parse_domains(domains)
        d = a.shape[0]
        if vals.ndim != d:
            raise ValueError(
                f"vals has {vals.ndim} axes but {d} domains were given; "
                "give one (lower, upper) pair per axis of vals."
            )
        if nodes not in _NODE_FAMILIES:
            raise ValueError(f"nodes must be one of {_NODE_FAMILIES}, got {nodes!r}.")
        min_nodes = 2 if nodes == "GaussLobatto" else 1
        if any(n < min_nodes for n in vals.shape):
            raise ValueError(
                f"Every axis needs at least {min_nodes} node(s) for nodes={nodes!r}; "
                f"got shape {vals.shape}."
            )

        self._a, self._b = a, b
        self._nodes = nodes
        self._axis_names = _parse_names(axis_names, d, "x", "axis_names")
        self._grid_shape = tuple(int(n) for n in vals.shape)

        # Coefficients always carry a trailing output axis internally
        coeffs = compute_chebyshev_coefficients_nd(vals, nodes=nodes)
        self._coeffs_full = np.ascontiguousarray(coeffs[..., np.newaxis])

        abs_c = np.abs(self._coeffs_full)
        n_out = abs_c.shape[-1]
        self._scale = abs_c.reshape(-1, n_out).max(axis=0)
        self._interp_error = _trailing_error_estimate(abs_c)
        self._discarded = np.zeros(n_out)
        self._ts = np.empty(d)
        self._set_truncation(self._grid_shape)

    # ------------------------------------------------------------------ setup

    def _set_truncation(self, kmax):
        """Keep the leading ``kmax`` block as the compact evaluation tensor (A2)."""
        block = tuple(slice(k) for k in kmax)
        self._kmax = tuple(int(k) for k in kmax)
        self._coeffs = np.ascontiguousarray(self._coeffs_full[block])
        self._flat = self._coeffs.reshape(-1)
        self._shape = np.array(self._kmax, dtype=np.int64)
        rest = self._flat.shape[0] // self._kmax[0]
        dtype = self._coeffs.dtype
        self._basis = np.empty(max(self._kmax))
        self._work_a = np.empty(rest, dtype=dtype)
        self._work_b = np.empty(rest, dtype=dtype)
        self._out = np.empty(self._coeffs.shape[-1], dtype=dtype)

    # ------------------------------------------------------------- properties

    @property
    def ndim(self):
        """Number of variables d."""
        return len(self._grid_shape)

    @property
    def grid_shape(self):
        """Number of nodes per axis, (N_1, ..., N_d)."""
        return self._grid_shape

    @property
    def kmax(self):
        """Number of coefficients used per axis after truncation, (k_1, ..., k_d)."""
        return self._kmax

    @property
    def domains(self):
        """The (lower, upper) bounds of each axis."""
        return tuple(zip(self._a.tolist(), self._b.tolist()))

    @property
    def axis_names(self):
        """The name of each axis."""
        return self._axis_names

    @property
    def nodes(self):
        """Node family the values were sampled at."""
        return self._nodes

    # ------------------------------------------------------------- evaluation

    def evaluate(self, *xs):
        """
        Evaluate the interpolant at a single point.

        Parameters
        ----------
        *xs : float
            One coordinate per axis, in axis order.

        Returns
        -------
        float
            Value of the interpolant at the point.
        """
        x = self._point(xs)
        evaluate_point(self._flat, self._shape, self._a, self._b, x, self._ts,
                       self._basis, self._work_a, self._work_b, self._out)
        return self._out[0].item()

    def _point(self, xs):
        """Validate scalar coordinates and return them as a float64 array."""
        if len(xs) != self.ndim:
            raise TypeError(
                f"evaluate() takes {self.ndim} coordinates {self._axis_names}, got {len(xs)}."
            )
        try:
            x = np.array(xs, dtype=np.float64)
        except (TypeError, ValueError):
            x = None
        if x is None or x.ndim != 1:
            raise TypeError("evaluate() takes one scalar per axis.")
        return x

    # ----------------------------------------------------- coefficients, errors

    def get_coefficients(self, full=False):
        """
        Get the Chebyshev coefficients of the interpolant.

        Parameters
        ----------
        full : bool, optional
            If False (default), return the truncated block used for
            evaluation, shape ``kmax``. If True, return the full tensor,
            shape ``grid_shape``.

        Returns
        -------
        np.ndarray
            The coefficient tensor.
        """
        if full:
            if self._coeffs_full is None:
                raise RuntimeError("The full coefficient tensor was dropped by optimize(keep_full=False).")
            return self._coeffs_full[..., 0]
        return self._coeffs[..., 0]

    def absolute_error(self):
        """
        Estimate the absolute error of the interpolant.

        The estimate is the interpolation error, judged from the trailing
        coefficients of the full tensor along each axis, plus the sum of
        the magnitudes of any coefficients removed by :meth:`optimize`
        (an upper bound on the error the truncation introduces).

        Returns
        -------
        float
            Absolute error estimate.
        """
        return float((self._interp_error + self._discarded)[0])

    def relative_error(self):
        """
        Estimate the relative error: :meth:`absolute_error` divided by the
        largest coefficient magnitude.

        Returns
        -------
        float
            Relative error estimate.
        """
        scale = self._scale[0]
        return float((self._interp_error + self._discarded)[0] / scale) if scale > 0 else 0.0

    # ------------------------------------------------------------- truncation

    def optimize(self, rel_tol=1e-8, keep_full=True):
        """
        Truncate the series by greedily removing trailing coefficient slices
        while the sum of the removed magnitudes stays below
        ``rel_tol`` times the largest coefficient magnitude.

        Truncation always starts from the full tensor, so ``optimize`` can
        be called again with a different tolerance.

        Parameters
        ----------
        rel_tol : float, optional
            Allowed relative error of the truncation, in (0, 1).
        keep_full : bool, optional
            If False, drop the full coefficient tensor to save memory. Later
            calls to ``optimize`` are then impossible.
        """
        if self._coeffs_full is None:
            raise RuntimeError(
                "optimize() needs the full coefficient tensor, which was dropped by "
                "an earlier optimize(keep_full=False)."
            )
        if not 0.0 < rel_tol < 1.0:
            raise ValueError(f"rel_tol must lie in (0, 1), got {rel_tol}.")

        abs_c = np.abs(self._coeffs_full)
        n_out = abs_c.shape[-1]
        # Normalise each output by its own scale; the greedy pruner then
        # bounds every output's discarded sum by rel_tol times its scale
        scale = np.where(self._scale > 0, self._scale, 1.0)
        magnitude = (abs_c / scale).max(axis=-1)
        kmax, _ = prune_trailing_slices_greedy_nd(magnitude, delta=rel_tol, scale=1.0)
        kmax = tuple(max(1, int(k)) for k in kmax)

        kept = abs_c[tuple(slice(k) for k in kmax)].reshape(-1, n_out).sum(axis=0)
        total = abs_c.reshape(-1, n_out).sum(axis=0)
        self._discarded = np.maximum(total - kept, 0.0)
        self._set_truncation(kmax)
        if not keep_full:
            self._coeffs_full = None

    def effective_grid(self):
        """
        Get the effective grid size after truncation.

        Returns
        -------
        tuple of int
            Per-axis truncation counts (``kmax``).
        """
        return self._kmax

    def __repr__(self):
        return (
            f"ChebyshevInterpolantND(axes={self._axis_names}, grid_shape={self._grid_shape}, "
            f"kmax={self._kmax}, nodes={self._nodes!r})"
        )


def _parse_domains(domains):
    """Return the lower and upper bounds as float64 arrays, checking each pair."""
    pairs = list(domains)
    if not pairs:
        raise ValueError("domains must contain at least one (lower, upper) pair.")
    a = np.empty(len(pairs))
    b = np.empty(len(pairs))
    for i, pair in enumerate(pairs):
        try:
            lo, hi = pair
        except (TypeError, ValueError):
            raise ValueError(f"domains[{i}] must be a (lower, upper) pair, got {pair!r}.") from None
        lo, hi = float(lo), float(hi)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(f"domains[{i}] = {pair!r} must be finite.")
        if not lo < hi:
            raise ValueError(f"domains[{i}] = {pair!r}: the lower bound must be smaller than the upper bound.")
        a[i], b[i] = lo, hi
    return a, b


def _parse_names(names, count, prefix, argname):
    """Return `count` unique names, defaulting to prefix0, prefix1, ..."""
    if names is None:
        return tuple(f"{prefix}{i}" for i in range(count))
    names = tuple(str(n) for n in names)
    if len(names) != count:
        raise ValueError(f"{argname} has {len(names)} entries but {count} are needed.")
    if len(set(names)) != len(names):
        raise ValueError(f"{argname} must be unique, got {names}.")
    return names


def _trailing_error_estimate(abs_c):
    """
    Per-output error estimate from the trailing coefficients along each axis.

    ``abs_c`` has shape (N_1, ..., N_d, F). For every axis with at least two
    coefficients, the last and second-to-last slices are scaled by N and
    N - 1 respectively (the convention of the 2D-4D classes), and the
    largest result over all axes is taken.
    """
    d = abs_c.ndim - 1
    n_out = abs_c.shape[-1]
    err = np.zeros(n_out)
    for ax in range(d):
        n = abs_c.shape[ax]
        if n < 2:
            continue
        c = np.moveaxis(abs_c, ax, 0)
        last = c[-1].reshape(-1, n_out).max(axis=0) * n
        second = c[-2].reshape(-1, n_out).max(axis=0) * (n - 1)
        err = np.maximum(err, np.maximum(last, second))
    return err
