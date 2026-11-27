"""
Blocked coefficient storage, for tables truncated output by output.

After ``ChebyshevInterpolantND.optimize(per_output=True)`` each output keeps
its own box of coefficients. The outputs are then stored in a few dense
*blocks*: block b is a C-contiguous array of shape (K^b_1, ..., K^b_d, F_b)
holding the outputs listed in its index array, with K^b large enough for
every member. Outputs whose whole series fits in the truncation budget
belong to no block and evaluate to exactly zero. A dense table is the
special case of one block holding every output in order.
"""

import numbers

import numpy as np

from .backend import array_namespace


class BlockedCoeffs:
    """
    The coefficients of a blocked table, on one device.

    What batched evaluation and precontraction receive in place of a single
    coefficient array when a table is stored in several blocks. Each block
    is either shared, of shape (K^b_1, ..., K^b_d, F_b), or holds one table
    per batch member, of shape (B, K^b_1, ..., K^b_d, F_b).

    Parameters
    ----------
    blocks : sequence of arrays
        The blocks, all NumPy or all CuPy, float64 or complex128.
    outputs : sequence of int64 arrays
        For each block, the output index of each of its columns, on the
        same device as the blocks.
    n_outputs : int
        Total number of outputs F, including those in no block.
    dtype : dtype
        The coefficient dtype (needed when there are no blocks).
    box : tuple of int, optional
        The elementwise largest block shape (K_1, ..., K_d). Computed from
        the blocks if not given; needed when there are none.
    members : int, optional
        The number of batch members for per-member blocks; None for shared
        blocks.
    xp : module, optional
        The array module. Found from the blocks if not given; needed when
        there are none.
    """

    def __init__(self, blocks, outputs, n_outputs, dtype, box=None, members=None, xp=None):
        self.blocks = tuple(blocks)
        self.outputs = tuple(outputs)
        self.n_outputs = int(n_outputs)
        self.members = None if members is None else int(members)
        if xp is None:
            xp = array_namespace(*self.blocks) if self.blocks else np
        self.xp = xp
        self.dtype = np.dtype(dtype)
        lead = 0 if self.members is None else 1
        if box is None:
            d = self.blocks[0].ndim - 1 - lead
            box = tuple(max(int(c.shape[lead + i]) for c in self.blocks) for i in range(d))
        self.box = tuple(int(k) for k in box)

    @property
    def per_member(self):
        """True for per-member blocks, (B, K..., F_b) each."""
        return self.members is not None

    @property
    def shape(self):
        """The shape of the equivalent zero-filled dense table."""
        lead = () if self.members is None else (self.members,)
        return lead + self.box + (self.n_outputs,)

    @property
    def ndim(self):
        """Number of dimensions of :attr:`shape`."""
        return len(self.shape)

    @property
    def size(self):
        """Number of stored coefficients."""
        return sum(int(c.size) for c in self.blocks)

    def to_dense(self):
        """The zero-filled dense table, shape :attr:`shape`, on the same device."""
        xp = self.xp
        out = xp.zeros(self.shape, dtype=self.dtype)
        lead = () if self.members is None else (slice(None),)
        for c, idx in zip(self.blocks, self.outputs):
            k = c.shape[len(lead):-1]
            out[lead + tuple(slice(n) for n in k) + (idx,)] = c
        return out

    def to(self, xp):
        """A copy on the device of array module ``xp``."""
        return BlockedCoeffs([xp.asarray(c) for c in self.blocks], [xp.asarray(o) for o in self.outputs],
                             self.n_outputs, self.dtype, box=self.box, members=self.members, xp=xp)


def table_namespace(table):
    """The array module of a coefficient array or :class:`BlockedCoeffs`."""
    return table.xp if isinstance(table, BlockedCoeffs) else array_namespace(table)


def pack_blocks(blocks, d, dtype, flat=None):
    """
    The blocks as flat arrays for the Numba kernels.

    Parameters
    ----------
    blocks : sequence of (coeffs, outputs)
        Host blocks; ``coeffs`` of shape (K^b_1, ..., K^b_d, F_b).
    d : int
        Number of axes.
    dtype : dtype
        Coefficient dtype.
    flat : 1D array, optional
        A buffer that already holds the blocks one after the other (e.g.
        written there by a precontraction), used instead of a copy.

    Returns
    -------
    coeffs : 1D array
        Every block's coefficients, flattened and concatenated. For a single
        contiguous block this is a view, not a copy.
    offsets : int64 array, shape (n_blocks + 1,)
        Block b occupies ``coeffs[offsets[b]:offsets[b + 1]]``.
    shapes : int64 array, shape (n_blocks, d)
        (K^b_1, ..., K^b_d) per block.
    out_index : int64 array
        The output index of each column of each block, concatenated.
    out_offsets : int64 array, shape (n_blocks + 1,)
        Block b's outputs are ``out_index[out_offsets[b]:out_offsets[b + 1]]``.
    """
    n = len(blocks)
    if flat is not None:
        if flat.size != sum(c.size for c, _ in blocks):
            raise ValueError("flat does not hold exactly the blocks.")
    elif n == 1 and blocks[0][0].flags.c_contiguous:
        flat = blocks[0][0].reshape(-1)
    elif n:
        flat = np.concatenate([np.ascontiguousarray(c).reshape(-1) for c, _ in blocks])
    else:
        flat = np.zeros(0, dtype=dtype)
    offsets = np.zeros(n + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([c.size for c, _ in blocks])
    shapes = np.array([c.shape[:-1] for c, _ in blocks], dtype=np.int64).reshape(n, d)
    out_index = (np.concatenate([np.asarray(o, dtype=np.int64) for _, o in blocks]) if n
                 else np.zeros(0, dtype=np.int64))
    out_offsets = np.zeros(n + 1, dtype=np.int64)
    out_offsets[1:] = np.cumsum([len(o) for _, o in blocks])
    return flat, offsets, shapes, out_index, out_offsets


def partition_outputs(kmax, max_blocks=None):
    """
    Group outputs into at most ``max_blocks`` dense blocks.

    The outputs are sorted by the volume of their boxes and the sorted list
    is split into contiguous runs, minimising the total stored size
    ``sum_b F_b * prod_i K^b_i``, where K^b is the elementwise largest box of
    run b. The split is exact (dynamic programming over the runs, O(F^2 B)
    array operations); only the restriction to runs of the sorted order is
    a heuristic.

    Parameters
    ----------
    kmax : int array, shape (F, d)
        The box of each output; every entry at least 1.
    max_blocks : int or None
        The largest number of blocks; None means one block per output.

    Returns
    -------
    list of int64 arrays
        For each block, its outputs (row indices of ``kmax``), ascending.
    """
    kmax = np.asarray(kmax, dtype=np.int64)
    n = kmax.shape[0]
    order = np.argsort(np.prod(kmax, axis=1), kind="stable")
    if max_blocks is None or max_blocks >= n:
        return [np.array([i], dtype=np.int64) for i in order]

    boxes = kmax[order].astype(np.float64)
    # cost[i, j]: stored size of one block holding sorted outputs i, ..., j - 1
    cost = np.full((n, n + 1), np.inf)
    for i in range(n):
        box = np.maximum.accumulate(boxes[i:], axis=0)
        cost[i, i + 1:] = np.prod(box, axis=1) * np.arange(1, n - i + 1)
    best = cost[0].copy()  # best[j]: the first j outputs in at most one block
    splits = []
    for _ in range(1, max_blocks):
        # The first j outputs in at most one more block: the last block
        # holds outputs i, ..., j - 1 (or the previous answer stands)
        total = best[:n, None] + cost
        start = total.argmin(axis=0)
        new = total[start, np.arange(n + 1)]
        better = new < best
        splits.append(np.where(better, start, -1))
        best = np.where(better, new, best)

    runs, j = [], n
    for split in reversed(splits):
        i = int(split[j])
        if i >= 0:
            runs.append((i, j))
            j = i
    if j > 0:
        runs.append((0, j))
    return [np.sort(order[i:j]) for i, j in reversed(runs)]


def parse_groups(groups, output_names, n_outputs):
    """
    One group label per output, from the ``groups`` argument of ``optimize``.

    Parameters
    ----------
    groups : None, sequence or mapping
        None: every output is its own group (returns None). A sequence of
        ``n_outputs`` hashable labels. Or a mapping from label to a sequence
        of output names or indices, which must put every output in exactly
        one group.
    output_names : tuple of str
        The output names (empty for a single unnamed function).
    n_outputs : int

    Returns
    -------
    tuple or None
        One label per output.
    """
    if groups is None:
        return None
    if isinstance(groups, (str, bytes)):
        raise TypeError("groups must be a sequence of labels (one per output) or a mapping from "
                        "label to outputs, not a string.")
    if hasattr(groups, "items"):
        labels = [None] * n_outputs
        for label, members in groups.items():
            if isinstance(members, (str, bytes)) or not hasattr(members, "__iter__"):
                raise TypeError(f"groups[{label!r}] must be a sequence of output names or indices.")
            for member in members:
                j = _output_position(member, output_names, n_outputs)
                if labels[j] is not None:
                    raise ValueError(f"Output {_describe(j, output_names)} is in groups {labels[j]!r} "
                                     f"and {label!r}; each output belongs to exactly one group.")
                labels[j] = label
        missing = [_describe(j, output_names) for j, label in enumerate(labels) if label is None]
        if missing:
            raise ValueError(f"Every output must be in a group; {', '.join(missing)} "
                             f"{'is' if len(missing) == 1 else 'are'} in none.")
        return tuple(labels)
    labels = tuple(groups)
    if len(labels) != n_outputs:
        raise ValueError(f"groups has {len(labels)} labels but the table has {n_outputs} outputs.")
    for label in labels:
        try:
            hash(label)
        except TypeError:
            raise TypeError(f"Group labels must be hashable, got {label!r}.") from None
    return labels


def group_scale(labels, scale):
    """Per output, the largest of ``scale`` over the outputs of its group."""
    scale = np.asarray(scale, dtype=np.float64)
    if labels is None:
        return scale.copy()
    largest = {}
    for label, s in zip(labels, scale):
        largest[label] = max(largest.get(label, 0.0), float(s))
    return np.array([largest[label] for label in labels])


def outside_box_sum(values, box):
    """
    Per-output sum of ``values`` outside the leading ``box``.

    ``values`` has shape (N_1, ..., N_d, F). The complement of the box is
    split into disjoint slabs, so nothing is summed twice and no copy of
    ``values`` is made.
    """
    total = np.zeros(values.shape[-1])
    head = ()
    for k in box:
        total += values[head + (slice(int(k), None),)].reshape(-1, values.shape[-1]).sum(axis=0)
        head += (slice(0, int(k)),)
    return total


def _output_position(member, output_names, n_outputs):
    if isinstance(member, str):
        try:
            return output_names.index(member)
        except ValueError:
            raise ValueError(f"No output named {member!r}.") from None
    if isinstance(member, numbers.Integral) and not isinstance(member, bool):
        if not -n_outputs <= member < n_outputs:
            raise ValueError(f"Output index {member} out of range for {n_outputs} outputs.")
        return int(member) % n_outputs
    raise TypeError(f"Outputs are given by name or integer index, not {member!r}.")


def _describe(j, output_names):
    return repr(output_names[j]) if output_names else str(j)
