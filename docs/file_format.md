# File format

{func}`chebyshevnd.save` writes tables and grids to HDF5 files that
{func}`chebyshevnd.load` reads back bit for bit. This page documents the
layout (format version 1), so that other programs can read the tables
without ChebyshevND, for example with plain `h5py` or the HDF5 C library.

Saving and loading need `h5py`: `pip install chebyshevnd[io]`.

```python
from chebyshevnd import save, load, read_metadata

save("rhs.h5", interp, metadata={"model": "SEOBNRv6EHM", "grid_hash": grid.grid_hash()})
interp = load("rhs.h5")                  # or load("rhs.h5", device="cuda")
read_metadata("rhs.h5")["model"]
```

Several tables can share one file by saving each into its own group:

```python
import h5py

with h5py.File("tables.h5", "w") as f:
    save(f.create_group("rhs"), rhs)
    save(f.create_group("amplitudes"), amplitudes, float32=True)

with h5py.File("tables.h5", "r") as f:
    rhs = load(f["rhs"])
```

## Every saved object

The group an object is saved into (the file's root group, or the group
passed to `save`) has these attributes:

| Attribute | Type | Meaning |
|---|---|---|
| `format` | string | always `"chebyshevnd"` |
| `format_version` | integer | `1`. Readers should refuse other versions |
| `kind` | string | `"ChebyshevInterpolantND"`, `"MappedInterpolant"` or `"ChebyshevGrid"` |
| `chebyshevnd_version` | string | version of the package that wrote the file |
| `created` | string | UTC time of writing, ISO 8601 |

It also has a subgroup `metadata` holding the user's metadata: one
attribute per entry, and one nested subgroup per nested mapping.

## Tables (`kind = "ChebyshevInterpolantND"`)

A table stores F functions of d variables as a tensor Chebyshev series on
the box [a₁, b₁] × … × [a_d, b_d]:

f_j(x) = Σ c[k₁, …, k_d, j] T_{k₁}(t₁) ⋯ T_{k_d}(t_d), with t_i = (2 x_i − a_i − b_i) / (b_i − a_i),

where T_k is the Chebyshev polynomial of the first kind, the sum runs over
0 ≤ k_i < K_i, and x is in table coordinates. The group holds:

| Name | Kind | Shape, type | Meaning |
|---|---|---|---|
| `coeffs` | dataset | (K₁, …, K_d, F), float64 or complex128 (float32 / complex64 if saved with `float32=True`) | the coefficients c, C order. The output axis is always present, even for a single function (F = 1) |
| `coeffs_full` | dataset, optional | (N₁, …, N_d, F) | the untruncated tensor, if saved with `full=True` and the table was truncated |
| `domains` | dataset | (d, 2), float64 | row i is (a_i, b_i) |
| `error_interpolation` | dataset | (F,), float64 | interpolation error estimate per function |
| `error_truncation` | dataset | (F,), float64 | bound on the error from truncation (and from single-precision storage) |
| `coeff_scale` | dataset | (F,), float64 | largest coefficient magnitude per function |
| `axis_names` | attribute | d strings | variable names, in axis order |
| `output_names` | attribute | F strings, or none | function names; empty for a single unnamed function |
| `grid_shape` | attribute | d integers | nodes per axis the table was fitted on (N_i ≥ K_i) |
| `nodes` | attribute | string | node family of the fit: `"Chebyshev"` (roots) or `"GaussLobatto"` |
| `domain_tol` | attribute | float | allowed overshoot outside the box, as a fraction of each half-width |
| `dtype` | attribute | string | `"float64"` or `"complex128"`: the type to use in memory |
| `storage_dtype` | attribute | string | the type of `coeffs` in the file |
| `full_is_coeffs` | attribute | boolean | true if `coeffs` is the untruncated tensor (the table was never optimized) |

The absolute error estimate is `error_interpolation + error_truncation`,
and the relative one divides it by `coeff_scale`.

## Mapped tables (`kind = "MappedInterpolant"`)

A mapped table is evaluated in physical coordinates. Each axis's physical
coordinate x is mapped to its table coordinate

y = f((x − L) / W),

and the table is then evaluated at y. The group holds:

| Name | Kind | Meaning |
|---|---|---|
| `table` | group | the table, as above |
| `boundary` | group, optional | the boundary table, as above (always float64): its outputs are the boundary functions, and its axes are physical coordinates of other axes |
| `maps` | group | one subgroup per mapped axis, named after the axis. Axes without one are not mapped (y = x) |

Each map subgroup has these attributes:

| Attribute | Type | Meaning |
|---|---|---|
| `transform` | string | `"identity"` (f(s) = s), `"log"` (f(s) = log(s + offset)) or `"power"` (f(s) = (s + offset)^power) |
| `lower` | float or string | L: a number, or the name of a boundary output evaluated at the physical point |
| `upper` | float or string, optional | if present, W = upper − L |
| `width` | float or string, optional | if present, W = width (exactly one of `upper` and `width` is present) |
| `offset` | float | added to s before a log or power transform |
| `power` | float, optional | the exponent of a power transform |

A point is outside the domain if any table coordinate lies outside its
interval (or is NaN), or if a boundary input lies outside the boundary
table's domain.

## Grids (`kind = "ChebyshevGrid"`)

| Name | Kind | Meaning |
|---|---|---|
| `nodes` | attribute | node family of every axis |
| `grid_hash` | attribute | the grid's fingerprint; checked when the grid is loaded |
| `axes` | group | subgroups `0`, `1`, … in axis order, each with attributes `name`, `n` (node count) and `domain` (2 floats, table coordinates), and an optional `map` subgroup as above |
| `boundary` | group, optional | the boundary table, as above |

## Versions

The format version changes whenever a reader of the old version would
misread a new file. A reader should refuse versions it does not know;
{func}`chebyshevnd.load` does, and its error names the version that wrote
the file.
