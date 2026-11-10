"""
Tensor grids of Chebyshev nodes: where to sample, in what order, and how to
put samples computed in parallel chunks back together.

Sampling the functions to tabulate is usually the expensive part of
building a table (hours to days for orbit-averaged EOB forcing), so a grid
gives every node a stable linear index, splits the index range into chunks
that can be computed independently, and checks chunk files against the
grid before they are merged and fitted. The parallel driver itself (job
arrays, MPI, ...) is up to the caller.
"""

import hashlib
import json
import numbers
import os

import numpy as np

from .batched import evaluate_batch as _evaluate_batch
from .interpnd import ChebyshevInterpolantND, _parse_domains
from .mapped import MappedInterpolant, check_boundary_covers, validate_maps
from .maps import AxisMap, forward_map, pack_maps
from .utils import chebyshev_gauss_lobatto_nodes, chebyshev_nodes

_NODE_FAMILIES = {"Chebyshev": chebyshev_nodes, "GaussLobatto": chebyshev_gauss_lobatto_nodes}
_FORMAT = "chebyshevnd-grid-1"


class Axis:
    """
    One axis of a :class:`ChebyshevGrid`.

    Parameters
    ----------
    name : str
        The physical variable, e.g. ``"e"``.
    n : int
        Number of nodes.
    domain : (float, float)
        The interval in *table* coordinates (the coordinates the nodes are
        placed in): the physical interval for an unmapped axis, otherwise
        the interval of the map's output (see :meth:`AxisMap.table_domain`).
    map : AxisMap, optional
        Map from the physical variable to the table coordinate.
    """

    def __init__(self, name, n, domain, map=None):
        if not isinstance(name, str) or not name:
            raise ValueError(f"Axis name must be a non-empty string, got {name!r}.")
        if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 1:
            raise ValueError(f"Axis {name!r}: n must be a positive integer, got {n!r}.")
        a, b = _parse_domains([domain])
        self.name = name
        self.n = int(n)
        self.domain = (float(a[0]), float(b[0]))
        self.map = AxisMap() if map is None else map
        if not isinstance(self.map, AxisMap):
            raise TypeError(f"Axis {name!r}: map must be an AxisMap, got {map!r}.")

    def to_dict(self):
        """A JSON-compatible description."""
        return {"name": self.name, "n": self.n, "domain": list(self.domain), "map": self.map.to_dict()}

    def __repr__(self):
        extra = "" if self.map.is_identity else f", map={self.map!r}"
        return f"Axis({self.name!r}, {self.n}, {self.domain}{extra})"


class ChebyshevGrid:
    """
    A tensor grid of Chebyshev nodes, possibly mapped, for sampling functions.

    Nodes are numbered in C order, the order of
    ``np.meshgrid(*grid.table_nodes(), indexing="ij")`` flattened: the last
    axis varies fastest. Sampling a function at ``grid.physical_points()``
    (or at a chunk of them) and passing the values to :meth:`fit` builds the
    interpolant.

    Parameters
    ----------
    axes : sequence of Axis
    nodes : {"Chebyshev", "GaussLobatto"}, optional
        Node family for every axis. Use "Chebyshev" (roots), which avoids
        the domain's end points: for NIT tables these may be e = 0 or the
        separatrix, where the functions are singular or cannot be sampled.
    boundary : ChebyshevInterpolantND, optional
        The boundary table for maps that use boundary functions (see
        :class:`MappedInterpolant`).
    """

    def __init__(self, axes, nodes="Chebyshev", boundary=None):
        axes = list(axes)
        if not axes:
            raise ValueError("A grid needs at least one axis.")
        for axis in axes:
            if not isinstance(axis, Axis):
                raise TypeError(f"Grid axes must be Axis objects, got {axis!r}.")
        names = [axis.name for axis in axes]
        if len(set(names)) != len(names):
            raise ValueError(f"Axis names must be unique, got {names}.")
        if nodes not in _NODE_FAMILIES:
            raise ValueError(f"nodes must be one of {tuple(_NODE_FAMILIES)}, got {nodes!r}.")
        if nodes == "GaussLobatto" and any(axis.n < 2 for axis in axes):
            raise ValueError("Gauss-Lobatto nodes need at least 2 nodes per axis.")
        self._axes = tuple(axes)
        self._nodes = nodes
        maps, self._baxes = validate_maps(names, [axis.map for axis in axes], boundary)
        self._boundary = boundary
        self._codes, self._params = pack_maps(maps, boundary.output_names if boundary else ())
        if boundary is not None:
            check_boundary_covers(names, self.domains, self._codes, self._params, self._baxes,
                                  boundary)
        self._hash = None

    # ------------------------------------------------------------- properties

    @property
    def axes(self):
        """The axes."""
        return self._axes

    @property
    def axis_names(self):
        """The name of each axis."""
        return tuple(axis.name for axis in self._axes)

    @property
    def ndim(self):
        """Number of axes."""
        return len(self._axes)

    @property
    def shape(self):
        """Number of nodes per axis."""
        return tuple(axis.n for axis in self._axes)

    @property
    def size(self):
        """Total number of nodes."""
        return int(np.prod(self.shape))

    @property
    def domains(self):
        """The (lower, upper) bounds of each axis, in table coordinates."""
        return tuple(axis.domain for axis in self._axes)

    @property
    def nodes(self):
        """Node family."""
        return self._nodes

    @property
    def maps(self):
        """Axis name -> AxisMap."""
        return {axis.name: axis.map for axis in self._axes}

    @property
    def boundary(self):
        """The boundary table, or None."""
        return self._boundary

    # ------------------------------------------------------------------ nodes

    def table_nodes(self):
        """
        The node coordinates along each axis, in table coordinates.

        Returns
        -------
        tuple of np.ndarray
            In the order the node functions return them (descending), as
            :meth:`ChebyshevInterpolantND.axis_nodes` does.
        """
        node_fn = _NODE_FAMILIES[self._nodes]
        return tuple(node_fn(axis.n, *axis.domain) for axis in self._axes)

    def table_points(self, start=0, stop=None):
        """
        Nodes ``start, ..., stop - 1`` in table coordinates.

        Returns
        -------
        np.ndarray, shape (stop - start, d)
        """
        start, stop = self._range(start, stop)
        index = np.unravel_index(np.arange(start, stop), self.shape)
        return np.stack([nodes[i] for nodes, i in zip(self.table_nodes(), index)], axis=1)

    def physical_points(self, start=0, stop=None):
        """
        Nodes ``start, ..., stop - 1`` in physical coordinates: the points to sample at.

        Returns
        -------
        np.ndarray, shape (stop - start, d)
        """
        y = self.table_points(start, stop)
        boundary = None
        if self._boundary is not None:
            coeffs, a, b = self._boundary._table_on(np)

            def boundary(xb):
                return _evaluate_batch(coeffs, a, b, xb, tol=np.inf)[0]

        return forward_map(y, self._codes, self._params, np, boundary=boundary, baxes=self._baxes)

    def multi_index(self, index):
        """Per-axis node indices of linear index (or indices) ``index``."""
        return np.unravel_index(index, self.shape)

    def linear_index(self, multi_index):
        """Linear index of per-axis node indices ``multi_index``."""
        return np.ravel_multi_index(multi_index, self.shape)

    def chunks(self, size):
        """
        Split the linear index range into chunks.

        Parameters
        ----------
        size : int
            Nodes per chunk (the last one may be smaller).

        Returns
        -------
        list of (int, int)
            ``(start, stop)`` ranges covering ``[0, grid.size)`` in order.
        """
        if isinstance(size, bool) or not isinstance(size, numbers.Integral) or size < 1:
            raise ValueError(f"size must be a positive integer, got {size!r}.")
        return [(s, min(s + int(size), self.size)) for s in range(0, self.size, int(size))]

    def grid_hash(self):
        """
        A fingerprint of the grid: node family, axes (names, node counts,
        domains, maps) and the boundary table's coefficients.

        Chunk files record it, and :meth:`load_chunks` refuses chunks made
        for a different grid. The boundary table enters through its exact
        coefficients, so build it once and give every worker the same one,
        rather than rebuilding it on machines whose arithmetic may differ in
        the last bit.

        Returns
        -------
        str
            A SHA-256 hex digest.
        """
        if self._hash is None:
            spec = {"format": _FORMAT, "nodes": self._nodes,
                    "axes": [axis.to_dict() for axis in self._axes], "boundary": None}
            if self._boundary is not None:
                b = self._boundary
                coeffs = np.ascontiguousarray(b.get_coefficients(), dtype=np.float64)
                spec["boundary"] = {
                    "axes": list(b.axis_names), "outputs": list(b.output_names),
                    "domains": [list(dom) for dom in b.domains], "shape": list(coeffs.shape),
                    "coeffs": hashlib.sha256(coeffs.tobytes()).hexdigest(),
                }
            text = json.dumps(spec, sort_keys=True)
            self._hash = hashlib.sha256(text.encode()).hexdigest()
        return self._hash

    # ------------------------------------------------------- chunks and fits

    def save_chunk(self, path, start, stop, values):
        """
        Write the samples of one chunk to an ``.npz`` file.

        Parameters
        ----------
        path : str or path-like
            Output file.
        start, stop : int
            The chunk's linear index range.
        values : mapping
            Function name -> array of ``stop - start`` samples (real or
            complex), at ``physical_points(start, stop)`` in order.
        """
        start, stop = self._range(start, stop)
        names, arrays = self._check_values(values, (stop - start,), "chunk values")
        np.savez(path, grid_hash=np.array(self.grid_hash()), start=np.int64(start),
                 stop=np.int64(stop), names=np.array(names),
                 **{f"values_{i}": arr for i, arr in enumerate(arrays)})

    def load_chunks(self, paths):
        """
        Read chunk files and assemble the samples on the whole grid.

        Checks that every file was written for this grid, that the chunks
        hold the same functions, and that together they cover every node
        exactly once.

        Parameters
        ----------
        paths : iterable of str or path-like

        Returns
        -------
        dict
            Function name -> array of shape ``grid.shape``, ready for
            :meth:`fit`.
        """
        chunks = []
        names = None
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                if str(data["grid_hash"]) != self.grid_hash():
                    raise ValueError(f"{os.fspath(path)} was written for a different grid.")
                chunk_names = tuple(str(n) for n in data["names"])
                if names is None:
                    names = chunk_names
                elif chunk_names != names:
                    raise ValueError(f"{os.fspath(path)} holds functions {list(chunk_names)}, but "
                                     f"other chunks hold {list(names)}.")
                arrays = [data[f"values_{i}"] for i in range(len(chunk_names))]
                chunks.append((int(data["start"]), int(data["stop"]), arrays, os.fspath(path)))
        if not chunks:
            raise ValueError("No chunk files were given.")
        chunks.sort(key=lambda c: (c[0], c[1]))
        expected = 0
        for start, stop, _, path in chunks:
            if start < expected:
                raise ValueError(f"{path} covers nodes [{start}, {stop}), which overlap an earlier "
                                 "chunk.")
            if start > expected:
                raise ValueError(f"Nodes [{expected}, {start}) are in no chunk.")
            expected = stop
        if expected != self.size:
            raise ValueError(f"Nodes [{expected}, {self.size}) are in no chunk.")

        result = {}
        for i, name in enumerate(names):
            dtype = np.result_type(*[arrays[i] for _, _, arrays, _ in chunks])
            full = np.empty(self.size, dtype=dtype)
            for start, stop, arrays, _ in chunks:
                full[start:stop] = arrays[i]
            result[name] = full.reshape(self.shape)
        return result

    def fit(self, values, domain_tol=1e-12):
        """
        Fit samples on the whole grid.

        Parameters
        ----------
        values : mapping or array
            Function name -> samples, each of shape ``grid.shape`` (or
            ``(grid.size,)`` in linear-index order), all fitted into one
            multi-output table; or a single array for one unnamed function.
        domain_tol : float, optional
            Passed to :class:`ChebyshevInterpolantND`.

        Returns
        -------
        ChebyshevInterpolantND or MappedInterpolant
            A :class:`MappedInterpolant` if any axis is mapped, otherwise
            the plain table.
        """
        if hasattr(values, "items"):
            names, arrays = self._check_values(values, None, "values")
            vals = np.stack([arr.reshape(self.shape) for arr in arrays], axis=-1)
            output_names = names
        else:
            arr = np.asarray(values)
            if arr.shape not in (self.shape, (self.size,)):
                raise ValueError(f"values must have shape {self.shape} or ({self.size},); "
                                 f"got {arr.shape}.")
            vals = arr.reshape(self.shape)
            output_names = None
        table = ChebyshevInterpolantND(vals, self.domains, nodes=self._nodes,
                                       axis_names=self.axis_names, output_names=output_names,
                                       domain_tol=domain_tol)
        if self._boundary is None and all(axis.map.is_identity for axis in self._axes):
            return table
        return MappedInterpolant(table, self.maps, self._boundary)

    # -------------------------------------------------------------- helpers

    def _range(self, start, stop):
        stop = self.size if stop is None else stop
        start, stop = int(start), int(stop)
        if not 0 <= start <= stop <= self.size:
            raise ValueError(f"Need 0 <= start <= stop <= {self.size}; got start={start}, "
                             f"stop={stop}.")
        return start, stop

    def _check_values(self, values, length, what):
        """Names and arrays of a mapping of samples; arrays flattened to linear-index order."""
        if not hasattr(values, "items") or not values:
            raise ValueError(f"{what} must be a non-empty mapping from function name to samples.")
        names, arrays = [], []
        for name, arr in values.items():
            arr = np.asarray(arr)
            if np.iscomplexobj(arr):
                arr = arr.astype(np.complex128)
            else:
                arr = arr.astype(np.float64)
            if length is not None:
                if arr.shape != length:
                    raise ValueError(f"{what}[{name!r}] must have shape {length}; got {arr.shape}.")
            elif arr.shape not in (self.shape, (self.size,)):
                raise ValueError(f"{what}[{name!r}] must have shape {self.shape} or "
                                 f"({self.size},); got {arr.shape}.")
            names.append(str(name))
            arrays.append(arr.reshape(-1))
        return names, arrays

    def __repr__(self):
        return f"ChebyshevGrid({list(self._axes)!r}, nodes={self._nodes!r})"
