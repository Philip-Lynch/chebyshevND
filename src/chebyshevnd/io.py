"""
Saving and loading tables in HDF5.

Needs ``h5py`` (``pip install chebyshevnd[io]``), imported only when a
function here is called.

:func:`save` writes a :class:`ChebyshevInterpolantND`, a
:class:`MappedInterpolant` or a :class:`ChebyshevGrid` to a file (or to a
group of an open file, so several tables can share one file), and
:func:`load` reads it back, bit for bit, optionally straight onto the GPU.
The schema is documented in ``docs/file_format.md`` so that other code
(e.g. a C reader) can use the tables without this package.
"""

import datetime
import importlib.metadata
import numbers
import os

import numpy as np

FORMAT = "chebyshevnd"
FORMAT_VERSION = 1

_KINDS = ("ChebyshevInterpolantND", "MappedInterpolant", "ChebyshevGrid")
#: Written by save() itself and returned by read_metadata()
_RESERVED = ("kind", "format_version", "chebyshevnd_version", "created")


def _h5py():
    try:
        import h5py
    except ImportError:
        raise ImportError("Saving and loading tables needs h5py: pip install chebyshevnd[io]") from None
    return h5py


def _version():
    try:
        return importlib.metadata.version("ChebyshevND")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


# --------------------------------------------------------------------- API


def save(target, obj, *, metadata=None, float32=False, full=False, overwrite=False):
    """
    Save a table or grid to an HDF5 file.

    Parameters
    ----------
    target : str, path-like or h5py.Group
        A file name, or a group of an open ``h5py.File`` (to keep several
        tables in one file, e.g. ``save(f.create_group("rhs"), rhs)``).
    obj : ChebyshevInterpolantND, MappedInterpolant or ChebyshevGrid
        What to save. Batches (``ChebyshevBatch``, ``MappedBatch``) are
        built per run and are not saved.
    metadata : mapping, optional
        Anything to keep with the table, e.g. the model name, a hash of the
        code that sampled it, the grid hash. Values may be strings, numbers,
        booleans, sequences or arrays of those, or nested mappings. Read it
        back with :func:`read_metadata`.
    float32 : bool, optional
        Store the coefficients of the table in single precision, for tables
        that need only ~1e-7 relative accuracy (e.g. waveform amplitudes).
        They are converted back to double precision on load, and the
        rounding they lost is added to the stored error estimate. Boundary
        tables are always stored in double precision.
    full : bool, optional
        Also store the full (untruncated) coefficient tensor of an optimized
        table, so the loaded table can be optimized again and supports
        :meth:`~ChebyshevInterpolantND.values_at_nodes`. By default only the
        coefficients used for evaluation are stored; for a table that was
        never optimized, those are the full tensor anyway.
    overwrite : bool, optional
        Replace an existing file, or the contents of a non-empty group.
        Without it, saving over an existing table raises an error.
    """
    h5py = _h5py()
    kind = _kind(obj)
    if hasattr(target, "create_dataset"):
        _prepare_group(target, overwrite)
        _write(target, obj, kind, metadata, float32, full)
        return
    path = os.fspath(target)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it.")
    with h5py.File(path, "w") as f:
        _write(f, obj, kind, metadata, float32, full)


def load(source, *, device="cpu", return_metadata=False):
    """
    Load a table or grid saved by :func:`save`.

    Parameters
    ----------
    source : str, path-like or h5py.Group
        A file name, or the group a table was saved to.
    device : {"cpu", "cuda"}, optional
        With "cuda", the coefficients are also copied to the GPU now (see
        :meth:`ChebyshevInterpolantND.to_device`), so batched evaluation
        with CuPy arrays never copies them later.
    return_metadata : bool, optional
        Also return the metadata (see :func:`read_metadata`).

    Returns
    -------
    ChebyshevInterpolantND, MappedInterpolant or ChebyshevGrid
        Evaluates bit for bit like the saved object (up to the precision
        of ``float32`` storage).
    metadata : dict
        Only if ``return_metadata``.
    """
    h5py = _h5py()
    if hasattr(source, "attrs") and hasattr(source, "keys"):
        obj, meta = _read(source)
    else:
        with h5py.File(os.fspath(source), "r") as f:
            obj, meta = _read(f)
    if device != "cpu" and hasattr(obj, "to_device"):
        obj.to_device(device)
    return (obj, meta) if return_metadata else obj


def read_metadata(source):
    """
    Read the metadata of a saved table without loading the table.

    Returns
    -------
    dict
        The user metadata given to :func:`save`, plus ``"kind"``,
        ``"format_version"``, ``"chebyshevnd_version"`` (of the writer) and
        ``"created"`` (UTC, ISO 8601).
    """
    h5py = _h5py()
    if hasattr(source, "attrs") and hasattr(source, "keys"):
        _check_format(source)
        return _read_metadata(source)
    with h5py.File(os.fspath(source), "r") as f:
        _check_format(f)
        return _read_metadata(f)


# ------------------------------------------------------------------ writing


def _kind(obj):
    from .grid import ChebyshevGrid
    from .interpnd import ChebyshevInterpolantND
    from .mapped import MappedInterpolant

    for cls in (ChebyshevInterpolantND, MappedInterpolant, ChebyshevGrid):
        if isinstance(obj, cls):
            return cls.__name__
    raise TypeError(
        f"save() takes a ChebyshevInterpolantND, MappedInterpolant or ChebyshevGrid, not "
        f"{type(obj).__name__}."
    )


def _prepare_group(group, overwrite):
    if len(group.keys()) or len(group.attrs):
        if not overwrite:
            raise ValueError(f"Group {group.name!r} is not empty; pass overwrite=True to replace "
                             "its contents.")
        for key in list(group.keys()):
            del group[key]
        for key in list(group.attrs.keys()):
            del group.attrs[key]


def _write(group, obj, kind, metadata, float32, full):
    group.attrs["format"] = FORMAT
    group.attrs["format_version"] = FORMAT_VERSION
    group.attrs["kind"] = kind
    group.attrs["chebyshevnd_version"] = _version()
    group.attrs["created"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    clash = [key for key in (metadata or {}) if key in _RESERVED]
    if clash:
        raise ValueError(f"metadata keys {clash} are reserved: save() records them itself.")
    _write_metadata(group.create_group("metadata"), metadata or {})
    if kind == "ChebyshevInterpolantND":
        _write_table(group, obj, float32, full)
    elif kind == "MappedInterpolant":
        _write_table(group.create_group("table"), obj.table, float32, full)
        if obj.boundary is not None:
            _write_table(group.create_group("boundary"), obj.boundary, False, False)
        maps = group.create_group("maps")
        for name, axis_map in obj.maps.items():
            if not axis_map.is_identity:
                _write_map(maps.create_group(name), axis_map)
    else:
        if float32 or full:
            raise ValueError("float32 and full apply to tables, not to grids.")
        group.attrs["nodes"] = obj.nodes
        group.attrs["grid_hash"] = obj.grid_hash()
        axes = group.create_group("axes")
        for i, axis in enumerate(obj.axes):
            g = axes.create_group(str(i))
            g.attrs["name"] = axis.name
            g.attrs["n"] = axis.n
            g.attrs["domain"] = np.array(axis.domain, dtype=np.float64)
            if not axis.map.is_identity:
                _write_map(g.create_group("map"), axis.map)
        if obj.boundary is not None:
            _write_table(group.create_group("boundary"), obj.boundary, False, False)


def _strings(values):
    return np.array(list(values), dtype=_h5py().string_dtype())


def _write_table(group, interp, float32, full):
    """Write one ChebyshevInterpolantND (schema: docs/file_format.md)."""
    coeffs = interp._coeffs
    complex_table = coeffs.dtype.kind == "c"
    storage = (np.complex64 if complex_table else np.float32) if float32 else coeffs.dtype
    if full and interp._coeffs_full is None:
        raise ValueError("full=True, but the full coefficient tensor was dropped by "
                         "optimize(keep_full=False).")

    truncation = np.array(interp._discarded, dtype=np.float64)

    def stored(array):
        out = np.ascontiguousarray(array, dtype=storage)
        if float32:
            # The rounding to single precision adds at most this much to |error|
            lost = np.abs(array - out.astype(array.dtype)).reshape(-1, array.shape[-1]).sum(axis=0)
            return out, lost
        return out, 0.0

    data, lost = stored(coeffs)
    group.create_dataset("coeffs", data=data)
    # Untruncated, the evaluation tensor is the full tensor: no need to store it twice
    untruncated = interp._coeffs_full is not None and interp._kmax == interp._grid_shape
    group.attrs["full_is_coeffs"] = untruncated
    if full and not untruncated:
        full_data, _ = stored(interp._coeffs_full)
        group.create_dataset("coeffs_full", data=full_data)
    group.create_dataset("domains", data=np.stack([interp._a, interp._b], axis=1))
    group.create_dataset("error_interpolation", data=np.asarray(interp._interp_error, dtype=np.float64))
    group.create_dataset("error_truncation", data=truncation + lost)
    group.create_dataset("coeff_scale", data=np.asarray(interp._scale, dtype=np.float64))
    group.attrs["axis_names"] = _strings(interp.axis_names)
    group.attrs["output_names"] = _strings(interp.output_names)
    group.attrs["grid_shape"] = np.array(interp.grid_shape, dtype=np.int64)
    group.attrs["nodes"] = interp.nodes
    group.attrs["domain_tol"] = float(interp.domain_tol)
    group.attrs["dtype"] = np.dtype(coeffs.dtype).name
    group.attrs["storage_dtype"] = np.dtype(storage).name


def _write_map(group, axis_map):
    group.attrs["transform"] = axis_map.transform
    group.attrs["lower"] = axis_map.lower
    if axis_map.upper is not None:
        group.attrs["upper"] = axis_map.upper
    else:
        group.attrs["width"] = axis_map.width
    group.attrs["offset"] = axis_map.offset
    if axis_map.power is not None:
        group.attrs["power"] = axis_map.power


def _write_metadata(group, metadata):
    if not hasattr(metadata, "items"):
        raise TypeError("metadata must be a mapping.")
    for key, value in metadata.items():
        if not isinstance(key, str) or not key or "/" in key:
            raise ValueError(f"metadata keys must be non-empty strings without '/', got {key!r}.")
        if hasattr(value, "items"):
            _write_metadata(group.create_group(key), value)
        else:
            group.attrs[key] = _metadata_value(key, value)


def _metadata_value(key, value):
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_, numbers.Number, np.number)):
        return value
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value)
        if array.dtype.kind in "U" or (array.dtype == object and all(isinstance(v, str) for v in array.ravel())):
            return _strings(array.ravel()).reshape(array.shape)
        if array.dtype.kind in "biufc":
            return array
    raise TypeError(f"metadata[{key!r}] = {value!r}: values must be strings, numbers, booleans, "
                    "sequences or arrays of those, or mappings.")


# ------------------------------------------------------------------ reading


def _check_format(group):
    fmt = group.attrs.get("format")
    if fmt != FORMAT:
        raise ValueError(f"{group.file.filename}:{group.name} is not a ChebyshevND table "
                         f"(no format attribute {FORMAT!r}).")
    version = int(group.attrs.get("format_version", -1))
    if version != FORMAT_VERSION:
        writer = group.attrs.get("chebyshevnd_version", "an unknown version")
        newer = "newer " if version > FORMAT_VERSION else ""
        raise ValueError(
            f"{group.file.filename}:{group.name} uses {newer}file format version {version} "
            f"(written by chebyshevnd {writer}); this chebyshevnd ({_version()}) reads format "
            f"version {FORMAT_VERSION}."
            + (" Upgrade chebyshevnd to read it." if version > FORMAT_VERSION else "")
        )
    kind = group.attrs.get("kind")
    if kind not in _KINDS:
        raise ValueError(f"{group.file.filename}:{group.name} holds an unknown kind {kind!r}.")
    return kind


def _read(group):
    from .grid import Axis, ChebyshevGrid
    from .mapped import MappedInterpolant

    kind = _check_format(group)
    meta = _read_metadata(group)
    if kind == "ChebyshevInterpolantND":
        return _read_table(group), meta
    if kind == "MappedInterpolant":
        boundary = _read_table(group["boundary"]) if "boundary" in group else None
        maps = {name: _read_map(g) for name, g in group["maps"].items()}
        return MappedInterpolant(_read_table(group["table"]), maps, boundary), meta
    boundary = _read_table(group["boundary"]) if "boundary" in group else None
    axes_group = group["axes"]
    axes = []
    for i in range(len(axes_group)):
        g = axes_group[str(i)]
        axes.append(Axis(str(g.attrs["name"]), int(g.attrs["n"]), tuple(float(v) for v in g.attrs["domain"]),
                         _read_map(g["map"]) if "map" in g else None))
    grid = ChebyshevGrid(axes, nodes=str(group.attrs["nodes"]), boundary=boundary)
    if grid.grid_hash() != group.attrs["grid_hash"]:
        raise ValueError(f"{group.file.filename}:{group.name}: the grid read back does not match its "
                         "stored grid_hash; the file is corrupted or was written with a different "
                         "hash definition.")
    return grid, meta


def _read_table(group):
    from .interpnd import ChebyshevInterpolantND

    dtype = np.dtype(str(group.attrs["dtype"]))
    coeffs = np.ascontiguousarray(group["coeffs"][()], dtype=dtype)
    if "coeffs_full" in group:
        full = np.ascontiguousarray(group["coeffs_full"][()], dtype=dtype)
    elif bool(group.attrs["full_is_coeffs"]):
        full = coeffs
    else:
        full = None
    domains = group["domains"][()]
    output_names = [str(n) for n in group.attrs["output_names"]]
    return ChebyshevInterpolantND._from_tables(
        coeffs, full, domains[:, 0], domains[:, 1],
        nodes=str(group.attrs["nodes"]),
        axis_names=[str(n) for n in group.attrs["axis_names"]],
        output_names=output_names,
        domain_tol=float(group.attrs["domain_tol"]),
        abs_error=group["error_interpolation"][()],
        discarded=group["error_truncation"][()],
        scale=group["coeff_scale"][()],
        grid_shape=[int(n) for n in group.attrs["grid_shape"]],
    )


def _read_map(group):
    from .maps import AxisMap

    def entry(name):
        if name not in group.attrs:
            return None
        value = group.attrs[name]
        return str(value) if isinstance(value, (str, bytes)) else float(value)

    return AxisMap(str(group.attrs["transform"]), lower=entry("lower"), upper=entry("upper"),
                   width=entry("width"), offset=float(group.attrs["offset"]),
                   power=entry("power"))


def _read_metadata(group):
    meta = {key: _python_value(group.attrs[key]) for key in _RESERVED}
    meta.update(_read_metadata_group(group["metadata"]) if "metadata" in group else {})
    return meta


def _read_metadata_group(group):
    out = {key: _python_value(value) for key, value in group.attrs.items()}
    for key, sub in group.items():
        out[key] = _read_metadata_group(sub)
    return out


def _python_value(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [v.decode() if isinstance(v, bytes) else str(v) for v in value.ravel()]
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value
