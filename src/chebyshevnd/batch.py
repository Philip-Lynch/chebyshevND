"""Per-member Chebyshev tables for a batch of members, e.g. waveforms."""

from .backend import array_namespace, to_numpy
from .batched import evaluate_batch
from .errors import OutOfDomainError, describe_out_of_domain


class ChebyshevBatch:
    """
    One small Chebyshev table per batch member, all on the same device.

    Made by :meth:`ChebyshevInterpolantND.precontract_batch`: member j's
    table is the interpolant with the fixed axes at member j's values. Each
    member is then evaluated at its own point, e.g. one waveform's current
    state in a batched ODE solve. The tables live on the device of the
    arrays they were made from (NumPy on the CPU, CuPy on the GPU), and
    evaluation stays there.
    """

    def __init__(self, tables, a, b, *, axis_names, output_names, domain_tol):
        """
        Wrap per-member tables; normally created by ``precontract_batch``.

        Parameters
        ----------
        tables : array, shape (B, k_1, ..., k_d, F)
            One table per member (NumPy or CuPy).
        a, b : array, shape (d,)
            Lower and upper bound of each remaining axis.
        axis_names : sequence of str
            Names of the remaining axes.
        output_names : sequence of str
            Output names; empty for a single function without a function axis.
        domain_tol : float
            Allowed overshoot outside the domain, as a fraction of each
            axis's half-width.
        """
        xp = array_namespace(tables)
        self._tables = tables
        self._a = xp.asarray(a, dtype=xp.float64)
        self._b = xp.asarray(b, dtype=xp.float64)
        self._a_host, self._b_host = to_numpy(self._a), to_numpy(self._b)
        self._axis_names = tuple(axis_names)
        self._output_names = tuple(output_names)
        self._single = len(self._output_names) == 0
        self._domain_tol = domain_tol

    @property
    def n_members(self):
        """Number of batch members B."""
        return self._tables.shape[0]

    @property
    def ndim(self):
        """Number of remaining (live) axes d."""
        return len(self._axis_names)

    @property
    def axis_names(self):
        """Names of the remaining axes."""
        return self._axis_names

    @property
    def output_names(self):
        """The name of each function; empty without a function axis."""
        return self._output_names

    @property
    def output_index(self):
        """Mapping from function name to its position in evaluation results."""
        return {name: i for i, name in enumerate(self._output_names)}

    @property
    def n_outputs(self):
        """Number of functions F (1 without a function axis)."""
        return self._tables.shape[-1]

    @property
    def kmax(self):
        """Number of coefficients per remaining axis."""
        return tuple(int(k) for k in self._tables.shape[1:-1])

    @property
    def domains(self):
        """The (lower, upper) bounds of each remaining axis."""
        return tuple(zip(self._a_host.tolist(), self._b_host.tolist()))

    @property
    def dtype(self):
        """Coefficient dtype: float64 or complex128."""
        return self._tables.dtype

    @property
    def tables(self):
        """The per-member coefficient tables, shape (B, k_1, ..., k_d, F)."""
        return self._tables

    def evaluate(self, points, return_status=False):
        """
        Evaluate each member at its own point.

        Parameters
        ----------
        points : array, shape (B, d)
            Row j is the point for member j, coordinates in axis order. For
            one remaining axis a shape (B,) array is also accepted. Arrays
            from the other library are copied to the tables' device.
        return_status : bool, optional
            If True, return the per-member status instead of raising for
            points outside the domain.

        Returns
        -------
        values : array
            Shape (B,) for a single function, (B, F) with a function axis,
            on the tables' device.
        status : int64 array, shape (B,)
            Only if ``return_status``. Bit i of ``status[j]`` is set when
            coordinate i of member j's point lies outside its domain or is
            NaN; that member's values should not be used.

        Raises
        ------
        OutOfDomainError
            If any point is outside the domain and ``return_status`` is False.
        """
        xp = array_namespace(self._tables)
        pts = xp.asarray(points, dtype=xp.float64)
        if pts.ndim == 1 and self.ndim == 1:
            pts = pts[:, None]
        if pts.ndim != 2 or pts.shape != (self.n_members, self.ndim):
            raise ValueError(
                f"points must have shape ({self.n_members}, {self.ndim}): one point per member, "
                f"axes {self._axis_names}; got {tuple(pts.shape)}."
            )
        values, status = evaluate_batch(self._tables, self._a, self._b, pts, tol=self._domain_tol)
        if self._single:
            values = values[:, 0]
        if return_status:
            return values, status
        bad = status != 0
        if bool(xp.any(bad)):
            first = int(xp.flatnonzero(bad)[0])
            raise OutOfDomainError(
                f"{int(bad.sum())} of {self.n_members} members are outside the interpolation domain; "
                f"the first is member {first}: "
                + describe_out_of_domain(self._axis_names, self._a_host, self._b_host,
                                         to_numpy(pts[first]), int(status[first])) + "."
            )
        return values

    def __repr__(self):
        return (
            f"ChebyshevBatch(members={self.n_members}, axes={self._axis_names}, kmax={self.kmax}, "
            f"outputs={self.n_outputs}, dtype={self.dtype})"
        )
