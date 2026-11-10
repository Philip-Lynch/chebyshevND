# ChebyshevND — design document for the NIT / SEOBNRv6EPHM extension

**Revision:** 3.7 (2026-11-10).
- Steps 1–6 and 8 are implemented; §3 records what was built and where it
  differs from the plan.
- Step 7 is deferred (D7): the NIT now uses an (α, β) equinoctial
  parametrisation, which removes the `1/e` behaviour of `f_w`.
- GPU support with CUDA via CuPy is now a requirement (§0.3, A8, D6).
- The second consumer is an interpolated post-Minkowskian (PM) Hamiltonian
  (§0.2).
- Step 0 is complete (v0.1.1).
- §9 lists all revisions and maps the old step numbers to the new ones.

**Branch:** `dev` (v0.1.1 plus Steps 1–6 and 8; version 0.2.0).

**Purpose:** turn ChebyshevND from a 1D–4D single-function interpolator into the
tabulation-and-evaluation backend for:
- near-identity-averaged (NIT) EOB dynamics, first for nonspinning /
  aligned-spin SEOBNRv6EHM and then for the precessing SEOBNRv6EPHM model;
- fast multivoice waveform construction for the same models;
- an interpolated PM Hamiltonian whose gradient gives the equations of motion
  (§0.2).

Every run-time path must work both on the CPU and in batches on the GPU with
CuPy, the setup pySEOBNR already uses (§0.3).

This document records *what* must change, *why* (traced back to the NIT
requirements), and *in what order*. It is meant to be the brief for separate
development sessions, so it is self-contained.

---

## 0. What the NIT project needs from the interpolator

The averaged dynamics evolve a small set of slow variables in the
radial phase χ, with a right-hand side built entirely from tabulated,
orbit-averaged functions. Everything below follows from the shape of those
tables and how often they are evaluated.

| Requirement | Where it comes from |
|---|---|
| **5 grid axes** `(ν, u=log p⁻³, e, χ₁·L̂N, χ₂·L̂N)` | aligned-spin forcing terms; the precessing model adds no axes (spins enter only via the two projections; in-plane spin scalars enter via linearised derivative tables) |
| **Many functions on one grid** (dozens of RHS terms; seven scalar moments `g₅…g₁₁` per κ for the spin block; hundreds of `(ℓ,m,n)` mode-amplitude harmonics) | the 0/1/2PA averaged equations, the precession moments, the multivoice waveform |
| **First and mixed second partial derivatives** in `(p, e)` and the spin-projection axes | 1PA/2PA terms contain `∂F̃/∂P̃ₖ` and `∂²F̃⁽⁰⁾/∂P̃ₖ∂P̃ᵢ`; the cheap adiabaticity estimate `Q` uses first derivatives. For the NIT, needed at **grid-build time** to assemble the higher-PA tables (D2); in production the NIT RHS is folded into one interpolant and evaluates no derivatives. The PM-Hamiltonian project (§0.2) needs **gradients at every RHS call** |
| **Complex coefficients** | Fourier coefficients of forcing functions, mode amplitudes |
| **Micro-second evaluation of the whole RHS** | the χ-ODE takes ~10³–10⁴ steps per waveform (×12 RHS calls per DOP853 step); see the budget below |
| **Axes held fixed per waveform** (`ν` always; both spin axes in the aligned-spin case; all of them for the nonspinning model) | precontracting those axes once per waveform is the dominant speed-up |
| **Non-rectangular physical domain** (`p_min` depends on `e` and spins; separatrix) | tensor Chebyshev needs a hyper-rectangle |
| **Singular prefactors** (`f_w ∼ 1/e`; √-type behaviour near the separatrix) | Chebyshev cannot represent these directly |
| **Portable tables** | tables built in the Mathematica/Python prototype must be the tables pySEOBNR ships |
| **Batched GPU evaluation (CUDA via CuPy)** | pySEOBNR runs batches of waveforms in parallel on GPUs with CuPy; the same tables must evaluate inside that pipeline (§0.3) |

### 0.1 Measured baseline and performance budget

Measured 2026-10-18 on `v0.1.0` (12⁴ grid, Apple laptop, the `cheb-env`
conda env: Python 3.13, NumPy 2.1, Numba 0.61). v0.1.1 did not change these
paths' speed.

| Operation | µs per call |
|---|---|
| `ChebyshevInterpolant4D.evaluate` | 15 |
| … after `optimize(1e-8)` (`kmax = (10, 12, 12, 4)`) | 5.3 |
| … `evaluate_yzw` after `precompute_x` | 1.0 |
| `clenshaw_nd` (pure Python), 12⁴ | 34 000 |
| Numba dispatch overhead (`@njit`, array + float args) | 0.14 |
| `scipy.solve_ivp` DOP853 overhead per RHS call (trivial jitted RHS) | 3.6 |

Revision 1 quoted 80 / 26 / 3 µs for the first three rows. The ratios agree,
so precontraction is still the main lever, but the absolute numbers came from
a different setup. `benchmarks/bench_evaluate.py` (added in v0.1.1) is the
source of truth.

**CPU budget.** RHS cost ≈ Σ over tables of (pruned coefficient count ×
number of outputs F × derivative factor). A prototype of the Step 1 kernel
evaluates a precontracted 30×20 table in 1.6 µs for F = 1, 16 µs for F = 50
and 86 µs for F = 300, linear in F. A waveform needs 10⁴–10⁵ RHS calls, so
every 10 µs of RHS costs 0.1–1 s per waveform. The levers, in order of
impact:
1. fold the whole RHS (all PA orders) into one multi-output table at build
   time, so F is the number of evolved variables rather than the number of
   terms (D2);
2. precontract the fixed axes (Step 2);
3. keep node counts low on the spin axes (Step 5) and prune (Step 1);
4. prune per function group, not just on the union (Step 8, if profiling says so).

For the nonspinning and aligned-spin models, a folded table precontracted to
`(u, e)` should cost a few µs per RHS call. The precessing model is the
remaining performance risk. With the spin projections live, its table stays
4D, and in the prototype a 4D table of ~2×10⁴ coefficients cost ~11 µs per
output.

Below ~5 µs per RHS call, `solve_ivp`'s own overhead dominates. The choice of
integrator is outside this package (§7), but it sets the floor.

**GPU budget.** On the GPU, cost is per batched call: a fixed launch
overhead (a few CuPy kernel launches at roughly 5–20 µs each) plus B × the
per-member work. The GPU only wins once B is large enough for the second
term to dominate. Single-waveform runs therefore stay on the CPU path, and
Step 4b measures where the crossover lies.

### 0.2 Second consumer: an interpolated PM Hamiltonian

A separate project tabulates a post-Minkowskian (PM) EOB Hamiltonian
`H(r, p_r, p_φ, …; ν, spins)` and gets the equations of motion from its
partial derivatives with respect to the coordinates and momenta
(`ṙ = ∂H/∂p_r`, `ṗ_r = −∂H/∂r + F_r`, …). The parameters (`ν`, spins) are
fixed per waveform and are precontracted (Step 2). The phase-space axes stay
live. This adds three requirements:

- **Run-time gradients.** Value and gradient along the live axes at every
  RHS call. The Hessian is assumed to be needed only occasionally (e.g. for
  initial conditions); revisit this if an implicit integrator or the
  radiation-reaction force needs it every step.
- **Maps on the hot path.** The region orbits visit in `(r, p_r, p_φ)` is far
  from a box (`p_φ` follows its circular-orbit value along the inspiral). The
  table will likely use mapped coordinates, such as `u = 1/r` and momenta
  relative to an `r`-dependent reference. The gradient with respect to the
  physical variables then needs the map Jacobian at every call (Step 5a).
- **Differentiate one interpolant.** The gradient of a single interpolant
  `H̃` gives a vector field that is exactly Hamiltonian for `H̃`. Separately
  fitted derivative tables would not.

Cost: in the prototype, a fused value + gradient evaluation costs ~2× a value
evaluation at any dimension (Step 4). The interpolant only pays off if it is
cheaper than the analytic Hamiltonian + gradient it replaces, so measure that
on the first real table. On the GPU, the baseline is the analytic PM
Hamiltonian evaluated in batch on the same GPU (§5.1 Q5), which is
comparatively cheaper there than on a CPU. The levers are pruning, maps, and
subtracting an analytic baseline (done in the user's RHS: tabulate `H − H₀`
and add `H₀` and its gradient back; see D7).

Sampling `H` should be far cheaper than sampling the NIT forcing, which
needs orbit integrations and Fourier decompositions. That makes this project
a good first end-to-end test of Steps 1–5a.

### 0.3 GPU requirement: batched evaluation with CuPy

pySEOBNR now runs on GPUs, computing batches of waveforms in parallel with
CUDA through CuPy (the software engineer confirmed the stack on
2026-11-09). Tables built with ChebyshevND must also evaluate inside that
pipeline. The CPU path stays: single-waveform runs, grid builds and tests use
it.

- **Batches, not single points.** Each CuPy kernel launch from Python costs
  roughly 5–20 µs, more than a whole CPU evaluation, so the GPU path always
  evaluates B batch members per call:
  `(table, points[B, d]) → (values[B, F], status[B])` (A8).
- **One table per member.** Each waveform in a batch has its own `ν` and
  spins, so batched precontraction produces one small table per member
  (Step 2b).
- **Matrix multiplications.** Batched contractions are GEMMs and einsums,
  which CuPy runs through cuBLAS (A3, A8).
- **No exceptions on the device.** Leaving the domain is reported per member
  in `status`, so the batched integrator can mask finished members (A8).
- **float64.** The dynamics tables need it. Datacenter GPUs (A100/H100) run
  FP64 at a useful rate; consumer GPUs run it at a small fraction of FP32
  speed.
- **No automatic differentiation in CuPy.** The PM Hamiltonian's gradient on
  the GPU uses the same analytic `T′` scheme as on the CPU (Step 4b).
- **Tables live on the device.** They are loaded once (Step 6). Nothing
  copies between host and device inside the ODE loop.

What we still need to learn about the engineer's pipeline is listed in §5.1.

---

## 1. Current state of the code (v0.1.1, what the changes build on)

- `clenshaw.py`: `clenshaw_1d/2d/3d/4d` Numba kernels; `partial_clenshaw_4d`
  (collapse leading axis); `clenshaw_nd` as a **pure-Python `nditer`
  recursion**, correct but ~2000× slower than `clenshaw_4d`.
- `interp1d.py … interp4d.py`: four near-duplicate classes. Each holds
  `coeffs`, domain bounds, a truncation, error estimates and `optimize`; 3D/4D
  add `precompute_x` / `evaluate_yz(w)`.
- `utils.py`: nodes (roots and Gauss–Lobatto), DCT-based coefficient
  computation for arbitrary dimension (always transforms *all* axes), greedy
  structured pruning (`prune_trailing_slices_greedy_nd`), per-row staircase
  truncation for 2D.
- Complex support: 1D and 2D kernels accept complex coefficients (Numba
  type-unifies the accumulator). 3D/4D kernels **fail** to compile because
  intermediate buffers are hard-coded `np.empty(...)` float64.
- Two truncation conventions coexist: per-row `NyMaxes` (2D) vs rectangular
  `kmax` (3D/4D). Truncation is passed into every kernel call, which is what
  caused the bugs fixed in v0.1.1.
- **Fixed in v0.1.1** (2026-10-19), each with a regression test:
  - `partial_clenshaw_4d` ignored the `ix` truncation count;
  - 3D `evaluate_yz` returned NaN for `Nx > Ny` (`NyMaxes` had length `Nx`);
  - 3D `precompute_x` ignored `kmax[0]`;
  - `optimize()` after `precompute_x()` desynchronised the precomputed array
    (NaN in 4D).
- Also in v0.1.1:
  - `cache=True` on every kernel;
  - CI on pushes to `dev`;
  - `benchmarks/bench_evaluate.py`;
  - documentation and examples on axis/bound ordering with non-square grids
    (issue #1).
- No out-of-domain checks: evaluation outside `[a, b]` silently extrapolates.
- Error estimates come from the full tensor and are not updated by `optimize`.
- No serialization, no derivatives, no multi-output, no domain maps, no
  batched or GPU evaluation.

---

## 2. Architectural decisions (apply to every step)

These are fixed up front so the steps below do not each re-decide them.

**A1 — The kernel works in box coordinates only.** Evaluation happens at
`t ∈ [−1, 1]^d`. Affine domain scaling, domain maps, prefactors and the
chain rule are layers on top; the kernels never see them. This keeps the hot
path small, which matters twice over now that the same algorithm exists for
Numba and for CuPy (D1, D6).

**A2 — Compact coefficient storage.** The tensor the kernel evaluates is
always C-contiguous with shape `(k₁, …, k_d[, F])`. `optimize` *slices and
copies*, so the kernel never receives `kmax`. This removes by construction
the class of bug fixed in v0.1.1, and gives the fixed shapes GPU code
prefers. The full tensor is kept for error estimates and re-optimizing
(`optimize(..., keep_full=False)` drops it for large 5D multi-output tables).

**A3 — Leading-axis-first basis contraction.** For each axis, compute
`T_k(t_j)` by the three-term recurrence. Contract axis 0 first, as a sequence
of axpy operations over contiguous memory, then axis 1, and so on.
- The function axis is the trailing contiguous block and is never contracted,
  so multi-output costs nothing extra in the kernel.
- Complex dtype only changes the work-buffer dtype.
- Run-time derivatives use the same kernel with `T′` (gradient) or `T″`
  (Hessian) on the differentiated axis: `T′ₖ₊₁ = 2Tₖ + 2tT′ₖ − T′ₖ₋₁` and
  `T″ₖ₊₁ = 4T′ₖ + 2tT″ₖ − T″ₖ₋₁`, with no division at `t = ±1`. Step 4
  builds the gradient version for the PM Hamiltonian (§0.2).
- Precontraction is the same operation with the point fixed (`np.tensordot`
  with one basis vector).
- Over a batch, the first contraction of a shared table is a single GEMM,
  `(B, k₁) @ (k₁, rest)`, which is why the same algorithm suits the GPU (A8).

A throwaway prototype (not committed), against a flat nD Clenshaw kernel
written the same way:

| Coefficient shape | flat nD Clenshaw | basis contraction | max error vs direct sum |
|---|---|---|---|
| 12⁴ | 17.6 µs | 11.3 µs | 5e-14 |
| 12⁵ | 201 µs | 118 µs | 2e-13 |
| 20×30×8×6×6 | 117 µs | 98 µs | 2e-13 |
| 30×20 | 1.4 µs | 1.65 µs | 3e-14 |
| 30×20, F = 50 | — | 15.6 µs | 8e-14 |

The prototype still allocated once per axis; ping-pong work buffers should
improve it further. On `[−1, 1]`, `|T_k| ≤ 1` and the recurrence is stable, so
rounding error is comparable to Clenshaw's. Keep the flat-Clenshaw kernel as
a test oracle. Remove the README sentence "evaluation never reconstructs
polynomial values explicitly" or reword it.

**A4 — Two compiled dtypes:** float64 and complex128, on both CPU and GPU.
float32 is a storage format only and is upcast on load (Step 6).

**A5 — Named axes and named outputs from the first version** of the nD class.
Every later API uses them: `precontract(fixed={"nu": 0.2})`,
`out[idx["Ftilde_p_0PA"]]`, map specs, file schema.

**A6 — Every `@njit` uses `cache=True`** (done for the v0.1 kernels in
v0.1.1).

**A7 — The 1D–4D classes stay as they are** (bugfixes only) until the nD
class is proven. Turning them into wrappers is Step 8, not Step 1. It brings
no NIT capability and carries compatibility risk (1D array evaluation, 2D
staircase semantics, attribute names the tests poke at).

**A8 — Batch-first evaluation, written once for NumPy and CuPy.** Next to the
Numba single-point kernels, the run-time evaluator is written against an
array namespace `xp`, chosen from the arrays passed in (NumPy or CuPy).
- **Signature:** `evaluate_batch(table, points) -> (values, status)`, with
  `points` of shape `(B, d)`, `values` `(B, F)` and `status` `(B,)` integers.
  It is a pure function: it returns new arrays and raises nothing. The
  Python convenience API raises on a nonzero status; the batched integrator
  reads `status` instead.
- **Tables:** `table` holds compact coefficients on the same device as
  `points`. They are either **shared** (`(k₁, …, k_d[, F])`) or **per member**
  (a leading batch axis `B`, produced by Step 2b).
- **Basis values:** the three-term recurrence, vectorised over the batch.
  - It is a Python loop over `k` of array operations, so on CuPy each
    step is a kernel launch. Since Step 4b, all axes are computed together
    and each step is two in-place operations, about `2·max(k)` launches for
    T and as many again for T′ (previously about `3·Σk`).
  - Computing all axes' bases in a single `cupy.RawKernel` (CUDA C compiled
    at run time by NVRTC, so no CUDA build step) is deferred to the GPU
    follow-up (Step 4b, as built). It could not be compiled or tested
    without a CUDA machine, and GPU benchmarks should justify it first.
  - Closed forms are not used. Checked against exact rational arithmetic
    (Step 4b): `cos(k·arccos t)` is as accurate as the recurrence, but the
    derivative `k·sin(kθ)/sin θ` is O(1) wrong at `t = −1` and loses digits
    near it, and neither extrapolates outside `[−1, 1]` (flagged points and
    `check_derivatives`' stencils need extrapolation).
- **Contractions:**
  - For a shared table, the first axis is one GEMM and the rest are
    batched einsums on small arrays.
  - Complex tables are contracted as real arrays, with real and imaginary
    parts interleaved along the function axis. Every GEMM therefore stays
    real (a complex × real product), about half the work of a complex GEMM.
- **Dependencies:** CuPy is optional (`chebyshevnd[gpu]`). Importing
  chebyshevnd never imports CuPy.
- **Testing:** NumPy is the reference implementation and runs in CI. The
  CuPy run must match it to rounding.

---

## 3. The changes, in implementation order

Ordering principle: each step is independently testable, and later steps
build on earlier ones rather than running in parallel with them.
- Step 0 stabilised what existed (done, v0.1.1).
- Steps 1–4 are the evaluation stack the RHS needs, for both the CPU
  (Numba) and batched GPU (CuPy) paths. GPU work sits in its own sub-steps
  (1e, 2b, 4b), so the CPU milestones never wait for it.
- Step 5 is maps (needed on the hot path) and the build side: grid and
  sampling.
- Steps 6–7 are production engineering.
- Step 8 is consolidation and polish.

Step 5 stays after Step 4 (D3).

### Step 0 — Stabilise v0.1 (bugfix release `0.1.1`) — DONE

Released as v0.1.1 on 2026-10-19 (PR #2; DOI follow-up PR #3). What it
contains:
- the four bug fixes listed in §1, each with a regression test;
- `cache=True`;
- `benchmarks/bench_evaluate.py`, run manually because timing tests are
  flaky on shared runners;
- CI on `dev`;
- the ordering documentation (issue #1).

### Step 1 — Core nD engine (`ChebyshevInterpolantND`) — IMPLEMENTED

Implemented on branch `step1-nd-core` with one commit per sub-step, and
merged into `dev` on 2026-11-09. "As built" below records where it differs
from the design.

**Why first:** 5D is needed, and the right fix is not an `interp5d.py`. One
nD class gives every later feature a single home instead of four copies. It
also fixes the kernel signature once, for complex dtype, the function axis
and batches, instead of rewriting it several times.

**Design:**
- `ChebyshevInterpolantND(vals, domains, nodes='Chebyshev', axis_names=None,
  output_names=None, domain_tol=1e-12)`. `domains` is a list of `(a, b)`
  pairs. Values only; no callables (consistent with v0.1).
- Coefficients from the existing DCT path, restricted to the spatial axes
  (`compute_chebyshev_coefficients_nd` and both renormalisers gain an `axes`
  argument).
- Jitted kernel in a new `kernels.py`, implementing A3:
  `contract_into(coeffs_flat, shape, ts, work, out) -> status`, with
  preallocated work buffers held by the interpolant. Reimplement the public
  `clenshaw_nd` on top of it, keeping its signature (~1000× faster).
- `evaluate(*xs)` returns a float when there is no function axis (as in
  v0.1) and an `(F,)` array otherwise.
- `evaluate_batch(points)` is the A8 evaluator, also used for large CPU
  batches.
- Error estimates per function: `(F,)` arrays, `np.abs`-based so complex
  works.
- `optimize(rel_tol)`: one rectangular `kmax` shared by all functions.
  Normalise each function by its own scale `S_f`, take the max over F, then
  run the existing greedy pruner on that d-dim magnitude tensor. The
  guarantee `Σ_discarded |c_f| ≤ rel_tol·S_f` then holds for every function
  separately. Apply A2 (slice and copy).
- Box-domain check: the kernels set `status ≠ 0` when any `|t| > 1 + tol`.
  `evaluate` raises `OutOfDomainError(ValueError)`. The tolerance is
  configurable; boundary points must pass.

**As built** (differences from the design above):
- **The function axis is opt-in.** Passing `output_names` declares the
  trailing function axis. Without it, `vals` must have exactly one axis per
  domain. Inferring the axis from `vals.ndim == len(domains) + 1` would have
  silently turned a missing domain into a function axis.
- **Bounds are checked:** each `(a, b)` must have `a < b`, finite.
- **Kernels** (`kernels.py`):
  - `contract(coeffs, shape, ts, basis, work_a, work_b, out)`;
  - `evaluate_point(coeffs, shape, bounds, xs, tol, scratch, work, out) -> status`,
    with `bounds` of shape `(2, d)` and `work` of shape `(2, n)`. It takes
    8 arguments, because Numba's per-argument dispatch cost dominated small
    tables.
  - The two work buffers must stay separate arrays: halves of one array
    stopped the inner loop from vectorising (12⁴: 12.3 → 20.7 µs).
- **`absolute_error()`** is the trailing-coefficient estimate plus the
  coefficient magnitudes discarded by `optimize`, so it covers truncation.
- **Batches:**
  - `evaluate_batch` processes shared-table batches in chunks under a
    256 MB memory budget, since the intermediate after the first GEMM has
    B × (table size / k₁) entries.
  - The first CuPy call copies the table to the device and caches it;
    `optimize` clears the cache.
- **Measured** (per call, Apple laptop; v0.1 class in brackets):

  | Grid | `evaluate` | Batched, per point (B = 1000) |
  |---|---|---|
  | 1D (64) | 1.34 µs (2.03) | 0.29 µs |
  | 2D (32×24) | 1.63 µs (1.36) | 0.32 µs |
  | 3D (20×16×12) | 3.29 µs (3.28) | 0.55 µs |
  | 4D (12⁴) | 12.3 µs (15.0) | 1.6 µs |
  | 5D (12×10×8×8×6) | 26 µs | 5.6 µs |

  `clenshaw_nd` takes 20 µs on 12⁴, down from ~34 ms. The 2D gap is fixed
  Python-level overhead, which Step 4a's compiled entry point avoids.
  Step 8 must benchmark the 2D/3D wrappers before replacing the v0.1
  classes (D5). It did, and the v0.1 classes were kept (D5, revised).
- **GPU tests** (`pytest -m gpu`) are written but have not yet been run on a
  CUDA machine.

Sub-steps, each with its own tests:
- **1a** real, single output (kernel + class);
- **1b** complex128;
- **1c** function axis and names;
- **1d** domain checks;
- **1e** the batch-first `xp` evaluator (A8), with the NumPy reference and
  CuPy support behind the optional import.

**Tests:**
- nD class agrees with the v0.1 1D–4D classes on the same data (≤1e-13).
- Agreement with `tests/helpers.direct_eval`, extended beyond the current
  `"ijkl"` 4D limit.
- 5D/6D separable functions.
- `exp(ix)cos(y)z` and friends in 3D–5D to 1e-14.
- Multi-output equals the stacked single-output results.
- Out-of-domain raises; `t = ±1` is accepted.
- `evaluate_batch` agrees with single-point `evaluate` to 1e-14 relative,
  for shared and per-member tables, real and complex. `status` flags exactly
  the out-of-domain members.
- `@pytest.mark.gpu` (skipped without CuPy and a CUDA device): the CuPy run
  matches NumPy to 1e-13 relative.
- Benchmark (not CI): the nD kernel is no slower than `clenshaw_4d` on 12⁴.

### Step 2 — Precontraction over an arbitrary subset of axes — IMPLEMENTED

Implemented on branch `step2-precontraction` (2026-11-09). "As built" below
records where it differs from the design.

**Why:** this is the biggest runtime lever. In the nonspinning model every
axis but `(u, e)` is fixed per waveform; in the aligned-spin model `ν, χ₁, χ₂`
are fixed; in the precessing model only `ν` is fixed (the projections drift
slowly). In a GPU batch, each member has its own fixed values.

**Design:**
- **2a — single waveform.** `precontract(fixed={axis: value, …})` returns a
  new, lower-dimensional `ChebyshevInterpolantND` with the same outputs and
  names. For each fixed axis, `np.tensordot` the compact tensor with that
  axis's basis vector `T(t)`. No special kernel is needed: this runs once per
  waveform, and A2 makes truncation automatic.
- **2b — batch.** `precontract(fixed={"nu": nu[B], …})` returns a per-member
  table with a leading batch axis, on the same device as the inputs.
  - Store the axes that are fixed per waveform first (one transpose at
    load).
  - Form each member's outer product of its fixed-axis bases,
    `W (B, k_ν·k_χ₁·k_χ₂)`.
  - Then do one GEMM:
    `W @ C.reshape(k_ν·k_χ₁·k_χ₂, rest) → (B, rest)`.
  - This avoids the large intermediate of contracting one fixed axis at a
    time. For example, with B = 1000, a live `(u, e)` block of 30×20 and
    F = 10, the result is 48 MB.
- Optional `optimize` on the result: the precontracted tensor often decays
  faster than the parent. For batches, prune once on the union over members,
  so the shape stays fixed.
- `precompute_x` / `evaluate_yz(w)` become wrappers over 2a in Step 8.

**As built** (differences from the design above):
- **One routine for both.** `batched.precontract_tables` (NumPy/CuPy) does
  the outer-product-plus-GEMM for 2b, and 2a calls it with a batch of one.
  The per-axis `np.tensordot` planned for 2a was much slower on 5D tables
  (3 ms per waveform), because every contraction on a non-leading axis
  copies the whole tensor.
- **Fixed axes first, cached.** Rather than reordering axes at load, the
  table with a given set of fixed axes moved to the front is cached on
  first use, per device. That costs one extra copy of the table per set
  of fixed axes in use; `optimize` clears the cache.
- **2a result:**
  - an ordinary `ChebyshevInterpolantND` that evaluates exactly like the
    parent, including its truncation;
  - the full tensor is also contracted when the parent still holds it and
    it differs from the truncated one, so the result can be optimized
    again;
  - the error estimate is inherited from the parent.
- **2b result:** `precontract_batch(fixed, return_status=False)` returns
  a `ChebyshevBatch` (new module `batch.py`).
  - It holds the `(B, k…, F)` tables on their device, with
    `evaluate(points, return_status=False)` taking one point per member.
  - Scalars are allowed alongside arrays and apply to every member.
  - Status bits for fixed values use the parent's axis numbering.
  - No per-batch `optimize` yet: a batch inherits the parent's
    truncation, which already keeps its shape fixed.
- **Measured** (4 outputs, Apple laptop; `benchmarks/bench_evaluate.py`):

  | Case | Full evaluate | `precontract` | Then evaluate | Batch build / member | Batch evaluate / member |
  |---|---|---|---|---|---|
  | nonspinning (ν, u, e) = 10×30×20, fix ν | 17–20 µs | 50 µs | 4.1 µs | 1–2 µs | 0.9 µs |
  | aligned-spin (ν, u, e, χ₁, χ₂) = 10×24×16×6×6, fix ν, χ₁, χ₂ | 292 µs | 0.2–0.6 ms | 3.6 µs | 11 µs | 0.7 µs |

  - Batch columns use B = 1000.
  - The single-waveform `precontract` on the 5D table is limited by memory
    bandwidth: one pass over the 4.4 MB table takes about 0.1 ms. It varies
    with the laptop's cache state.
- **GPU tests** (`pytest -m gpu`) are written but have not yet been run on a
  CUDA machine.

**Tests:**
- Precontracted evaluation equals full evaluation to 1e-14 at random points,
  for every subset of axes including non-leading ones.
- Works on complex, multi-output and optimized tensors.
- 2b equals running 2a member by member (NumPy in CI; CuPy under
  `@pytest.mark.gpu`).

### Step 3 — Analytic derivatives — IMPLEMENTED

Implemented on branch `step3-derivatives` (2026-11-10). "As built" below
records where it differs from the design.

**Why:** the 1PA/2PA tables are assembled at build time from first and
mixed second derivatives of lower-order tables, evaluated at the grid nodes
(`∂F̃/∂P̃ₖ`, `∂²F̃⁽⁰⁾/∂P̃ₖ∂P̃ᵢ`, spin→orbit cross terms). For the NIT this is
the only use: production folds the whole RHS into one table, so derivatives
never enter its ODE loop (D2). Occasional diagnostics such as `Q` can
evaluate a derivative interpolant through the ordinary `evaluate`. For the
PM Hamiltonian (§0.2), the derivative tensors are the test oracle for Step
4's fused gradient, and they supply the occasional Hessian (initial
conditions). This step is build-side and CPU-only.

**Design:**
- `derivative(orders)` takes `orders` as a tuple of non-negative integers per
  axis and returns a new `ChebyshevInterpolantND`. It differentiates in
  coefficient space along each axis with `c′ₖ₋₁ = c′ₖ₊₁ + 2k cₖ` (halving at
  `k = 0`), vectorised over fibres, times `2/(b−a)` per application.
  Derivatives therefore compose with precontraction and multi-output
  automatically. Cache results on the parent, keyed by `orders`.
- `values_at_nodes()`: inverse DCT back to the node grid (exact up to
  rounding), for both node families. This gives the build workflow: fit
  `F⁽⁰⁾`, call `F0.derivative((0, 1, 0)).values_at_nodes()`, combine with
  other node values, fit `F⁽¹⁾`.
- Error estimates are recomputed on the derivative tensor. Differentiation
  loses ~N² of relative accuracy per order, so document that node counts
  must be set by second-derivative accuracy.
- `check_derivatives(points)`: compare against central finite differences of
  the interpolant itself. Run it routinely during grid builds, not once.

**As built** (differences from the design above):
- **`derivative(orders)`:**
  - `orders` may also be a mapping, e.g. `{"u": 1, "e": 1}`.
  - It differentiates the *truncated* series, so it is the exact
    derivative of what `evaluate` computes, and that tensor shrinks by the
    order along each differentiated axis. The full series is
    differentiated too when it is held.
  - The recurrence is public as `differentiate_coefficients(coeffs, axis,
    order)`.
- **No cache on the parent.** A derivative costs one pass over the tensor
  per order, and a cached object could be changed by the caller (e.g. by
  `optimize`), which would surprise later calls. Callers keep the object.
- **The error estimate has three parts:**
  - the derivative's trailing coefficients;
  - an exact bound on the truncation's contribution,
    `Σ|D(full) − D(truncated)|` (scaled by Markov's inequality when the full
    tensor was dropped);
  - a **rounding term**, `eps · max|c| · ∏ (N²·2/(b−a))^order`.

  The rounding term was added because tests showed that for a function
  resolved to rounding level, the second-derivative error is amplified
  coefficient noise, which the trailing coefficients miss. On the test
  function the estimate bounds the actual error in every case, by 1–100×.
  Since it grows with N, more nodes eventually make high derivatives
  worse.
- **`values_at_nodes()`** uses the full series, via the new public
  `chebyshev_values_at_nodes_nd`, the inverse of the coefficient DCT.
- **`axis_nodes()`** (new) returns the node coordinates per axis, so other
  quantities can be computed at the same nodes. A test runs the whole
  build workflow (F⁽¹⁾ = ∂F⁽⁰⁾/∂y · g).
- **`check_derivatives(orders, points, step=None)`:**
  - supports first, pure second and mixed second derivatives;
  - returns `max|analytic − FD| / max|analytic|` per output (typically
    1e-10 for first and 1e-8 for second derivatives);
  - evaluates its stencils without the domain check, so boundary points
    work.

**Tests:**
- Derivatives of separable test functions to ~1e-10 (first) and ~1e-8
  (second) on 20-node axes.
- Mixed-partial symmetry `∂ₓ∂_y = ∂_y∂ₓ`.
- Chain rule under domain rescaling.
- `derivative ∘ precontract == precontract ∘ derivative`.
- `values_at_nodes` round trip reproduces the input samples to 1e-14 for
  both node families.

### Step 4 — Fused evaluation for the ODE RHS (values and gradients) — IMPLEMENTED

Implemented on branch `step4-fused-evaluation` (2026-11-10). "As built"
below records where it differs from the design.

**Why:** each Python-level interpolant call adds dispatch and glue around a
µs-scale kernel, so the whole RHS should be one compiled function on the CPU,
or a few kernel launches per batch on the GPU. The PM Hamiltonian (§0.2) also
needs the gradient at every call.

**4a — CPU (Numba).**
- `interp.raw()` returns the precontracted tensor and metadata as plain
  arrays (`coeffs_flat, shape, a, b, n_out`). A module-level `@njit`
  function `eval_into(coeffs_flat, shape, a, b, xs, work, out) -> status`
  can be called directly from the user's own jitted RHS. With the NIT RHS
  folded into one table (D2), that is one call per RHS evaluation.
- `eval_grad_into(…) -> status` returns the value and the gradient along
  every live axis in one call, scaled by `2/(b − a)` per axis. It also works
  with the function axis, giving a gradient for every output. The class
  exposes the same thing as `value_and_grad(*xs)`.
  - **Algorithm (prototyped):** contract axis 0 with both `T` and `T′` (two
    passes over the full tensor). Then finish each of the `1 + d` outputs on
    the much smaller remaining buffer, using `T′` on the differentiated axis.
  - **Measured (unpruned tables):**

    | Table | Value only | Value + gradient |
    |---|---|---|
    | 30³ | 17 µs | 35 µs |
    | 20³×10 | 51 µs | 109 µs |
    | 40×30×30 | 22 µs | 45 µs |

    That is 2.0–2.1× a value evaluation. Stacking the Step 3 derivative
    tensors as extra outputs costs `(1 + d)×` instead. The fused results
    agree with coefficient-space derivatives to ~1e-15 relative.
- **Hessian:** the same scheme with `T″` (the basis recurrence already
  produces it), exposed as `max_order=2`. Build it only once a per-step use
  appears; one-off Hessians can use Step 3's derivative interpolants.

**4b — GPU (CuPy).**
- `evaluate_batch` and `value_and_grad_batch` on per-member tables (Step 2b),
  following A8.
- The gradient uses the 4a algorithm in batched form. CuPy has no automatic
  differentiation, so the analytic `T′` route is the only one.
  - Compute `T` and `T′` for every axis in the same RawKernel.
  - Contract the first axis with both, giving a `(B, 2, …)` array.
  - Then finish the `1 + d` outputs with einsums.
- **Fused kernel, if needed.** Measure first. If launch overhead dominates
  at the engineer's batch sizes, fuse the whole per-member evaluation into a
  single `RawKernel`, giving one launch per RHS call. The same CUDA C source
  can then also be offered as a `__device__` function, for RHS code written
  as raw CUDA (§5.1 Q1).

**Portability:** A1–A3 and the self-describing schema (Step 6) keep a
Cython/C port cheap as well, if plans change.

**As built, 4a** (differences from the design above):
- **API.**
  - `raw()` returns a `RawTable(coeffs, shape, bounds, tol)` named tuple
    of read-only views: a snapshot that a later `optimize()` does not
    change. `bounds` is `(2, d)`, as in the internal kernels.
  - `RawTable.workspace()` returns a `Workspace(scratch, work)` that
    serves both entry points.
  - The entry points take **flat arrays** in field order:
    `eval_into(coeffs, shape, bounds, tol, x, scratch, work, out)`, and the
    same for `eval_grad_into`, whose `out` is `(d + 1, F)` (row 0 the
    values, row i + 1 the derivative along axis i). From Python,
    `eval_into(*table, x, *ws, out)`.
  - Why flat: Numba types array arguments in C but tuples in Python. A
    jitted RHS that takes the named tuples costs about 1.5 µs more per call
    from Python (2.5 µs vs 1.0 µs of kernel for a 30×20 table), against
    about 0.5 µs for 8 arrays. `solve_ivp` (and pySEOBNR's integrator) call
    the RHS from Python at every step. Numba 0.61 also cannot star-unpack a
    named tuple inside `@njit` code.
  - Both check the sizes of `x`, the workspace and `out`, and raise
    `ValueError`: Numba does no bounds checking, so an undersized
    workspace would corrupt memory.
  - `value_and_grad(*xs)` returns `(value, grad)`, `grad` of shape `(d,)`,
    or `(d, F)` with a function axis.
- **Algorithm: branches instead of 1 + d finishing passes.** Contracting
  axis i turns the value branch into two (with T and T′) in one pass over
  its data, and contracts every existing derivative branch with T. Axis 0
  is read once with two accumulators, and only the small partial sums are
  touched more than once. The hot loops are written out by hand: calling a
  helper for them cost 10–18%.
- **Measured** (raw kernels from an `@njit` loop, single function):

  | Table | Value | Value + gradient | Ratio |
  |---|---|---|---|
  | 30³ | 15.1 µs | 17.0 µs | 1.12 |
  | 20³×10 | 43.1 µs | 49.3 µs | 1.14 |
  | 40×30×30 | 19.9 µs | 22.1 µs | 1.11 |
  | 30×20, F = 4 | 2.18 µs | 2.49 µs | 1.14 |
  | 12⁴ | 11.8 µs | 14.2 µs | 1.20 |
  | 10×24×16×6×6, F = 4 | 300 µs | 397 µs | 1.32 |
  | 20³, complex | 8.5 µs | 11.9 µs | 1.40 |

  That is 1.0–1.4×, against 2.0–2.1× for the prototype and the ≤ 2.5×
  target. Through the Python methods (`evaluate` vs `value_and_grad`) the
  ratio is 1.2–1.3×. The gradient agrees with Step 3's derivative tensors
  to ~1e-15 relative.
- **No Hessian** (`max_order=2`): there is no per-step use yet, and Step 3's
  derivative interpolants cover one-off Hessians.
- **Toy Kepler Hamiltonian.** `H(k, r, p_r, p_φ)` on a 3×40×4×5 grid,
  `k` precontracted, Hamilton's equations from `eval_grad_into` in a
  jitted RHS, DOP853 at `rtol = 1e-12` over ~2.6 orbits (1643 calls).
  The interpolated energy drifts by 7e-13 and the orbit satisfies
  `r = p / (1 + e cos φ)` to 5e-12.
- **ODE RHS** (`bench_evaluate.py`): a toy 2-variable averaged RHS from a
  (ν, u, e) = 10×30×20 table precontracted at ν, with 2 outputs, DOP853.

  | Right-hand side | µs per call |
  |---|---|
  | trivial jitted RHS (integrator floor) | 3.6 |
  | Python RHS calling `evaluate()` | 5.7 |
  | jitted RHS calling `eval_into()` | 5.4 |

  The table costs about 1 µs and the dispatch about 0.5 µs, so
  `solve_ivp`'s own overhead dominates (§0.1). The next lever for a folded
  NIT table is a compiled integrator, which is outside this package (§7).

**As built, 4b** (differences from the design above):
- **API.** `value_and_grad_batch(coeffs, a, b, points)` (module level)
  returns `(values (B, F), grads (B, d, F), status)`. The class has
  `value_and_grad_batch(points, return_status=False)`, and `ChebyshevBatch`
  has `value_and_grad(points, return_status=False)`. Shapes drop the
  function axis for a single function, as for `evaluate_batch`.
- **Algorithm.** The same branches as 4a:
  - For a shared table, the first axis is one GEMM,
    `(2B, k₁) @ (k₁, rest)`, with the T and T′ rows of each point
    interleaved.
  - For per-member tables, the first axis is one batched GEMM,
    `(B, 2, k₁) @ (B, k₁, rest)`, which reads each table once.
  - Later axes are batched matrix products on the partial sums.
- **Basis values.** `chebyshev_basis_batch` takes points of any shape and
  `derivative=True`.
  - All axes are computed together, up to the largest k.
  - The arrays are stored k-first, so each step is two in-place
    operations on contiguous memory.
  - T′ comes from the second-kind polynomials, `T′_k = k·U_{k−1}`.

  This cuts the array operations (GPU launches) from about `3·Σk` for
  values and about `10·max(k)` for values + derivatives to about
  `2·max(k)` and `4·max(k)`. On the CPU the bases are about 2.5× faster.
- **No RawKernel, no fused kernel.** CUDA C cannot be compiled or tested
  on the development Mac. Both wait for the GPU follow-up, together with
  running `pytest -m gpu` and `bench_evaluate.py --gpu` (the batch-size
  sweep is written and runs on NumPy).
- **CPU cost.** A batched gradient costs 2–4× a batched value on NumPy,
  unlike the Numba kernel. A shared table's first-axis GEMM is
  compute-bound, so twice the rows cost twice the time, and the derivative
  branches add contractions (`j + 2` at axis j). Per point, batched NumPy
  is about as fast as the Numba loop (1.4–2.0 µs vs 2.1–2.2 µs per member
  for B ≥ 100 in the sweep).
- **GPU tests** (3) are written and have not been run.

**Tests:**
- `eval_into` and `eval_grad_into` called from inside a user `@njit`
  function.
- The fused gradient matches the Step 3 derivative tensors to 1e-13 at
  random points, including complex, multi-output and `t = ±1` cases.
- 4b matches 4a member by member to 1e-13 relative (`@pytest.mark.gpu`; the
  NumPy run of the same code is in CI).
- A toy Hamiltonian (harmonic oscillator or Kepler, tabulated on a box):
  integrate Hamilton's equations from the fused gradient. Check energy
  conservation and agreement with the analytic trajectory.
- A toy 2-variable averaged RHS integrated with `scipy.solve_ivp` (DOP853).
  Report the evaluation share of total time against the 3.6 µs per-call
  integrator overhead (§0.1).
- Benchmarks (not CI):
  - value + gradient ≤ 2.5× a value evaluation;
  - a batch-size sweep on the GPU, recording where CuPy overtakes the CPU
    path;
  - for the PM Hamiltonian, a comparison with the analytic Hamiltonian
    batched on the same GPU.

### Step 5 — Domain maps, grid definition and sampling workflow — IMPLEMENTED

Implemented on branch `step5-maps-grid` (2026-11-10). "As built" below
records where it differs from the design.

**Why grouped:** the map decides where the nodes sit in physical variables,
so the grid and the maps are one design.
**Why here:** sampling at the nodes is the expensive part of the project
(hours to days per grid), but it does not need to start before the
evaluation stack is done (D3), so this step follows Step 4.

**Sub-steps:**
- **5a — maps and the chain rule.** The PM Hamiltonian (§0.2) needs these on
  the hot path, so do 5a straight after Step 4. Each map needs a Numba
  version (CPU hot path) and an `xp` version (batched / GPU).
- **5b — grid, sampling and checkpoints.** CPU only.

**Design:**
- `Axis(name, N, domain, nodes='Chebyshev', map=None)` and
  `ChebyshevGrid(axes)`. The grid provides:
  - node coordinates in box and physical variables;
  - a stable C-order linear index (matching `meshgrid(indexing='ij')`);
  - `chunks(size)`;
  - `grid_hash()` over domains, N, node family and map specs.
  Per-axis N makes the spin axes cheap (4–6 nodes).
- **Node family:** use root nodes for NIT tables. Gauss–Lobatto includes the
  endpoints `e = 0` (where `f_w ∼ 1/e`) and the separatrix, where sampling
  may be impossible.
- **Maps:** a small named registry. Each entry provides the forward map
  (box→physical), the inverse, and the Jacobian and Hessian of the inverse.
  They are implemented over a parameter array, both as `@njit` functions and
  as `xp` functions. Maps are therefore serialisable (Step 6) and callable
  from both hot paths (Step 4). Every map is written twice, so keep the
  registry small. Initial set:
  - affine (the default);
  - log-type, `y = log(x − x₀ + δ)`;
  - **table-bounded**: affine between a lower boundary given by a
    lower-dimensional `ChebyshevInterpolantND` (e.g. the separatrix in
    `(e, χ₁, χ₂, ν)`, which is numerical for EOB) and a fixed upper bound.
    The boundary table is evaluated with the same evaluators (single-point or
    batched).
  A map for one axis may depend on other axes, in which case the box
  Jacobian is not diagonal.
- **Chain rule helper:** physical gradient and Hessian from the box ones plus
  the map Jacobian `J` and Hessian `H`:
  `∂ᵢF = Σₐ ∂ₐF̂ Jₐᵢ` and `∂ᵢ∂ⱼF = Σₐᵦ ∂ₐ∂ᵦF̂ JₐᵢJᵦⱼ + Σₐ ∂ₐF̂ Hᵃᵢⱼ`.
  The kernels stay map-free (A1).
  - The gradient part is on the PM Hamiltonian hot path.
    `eval_grad_physical_into` (Numba) and `value_and_grad_physical_batch`
    (`xp`) each combine three things: the inverse map, the box gradient and
    the gradient chain rule.
  - The Hessian part, which needs the map Hessian, runs off the hot path.
- **Precontraction constraint:** an axis whose map depends on live axes
  cannot be precontracted at a fixed physical value. When a map depends on a
  precontracted axis, that value is bound into the map (per member, in a
  batch). A table-bounded map precontracts its boundary table at the same
  values.
- `evaluate_physical(*x)` applies the inverse maps, then the box kernel, then
  the domain check. Leaving the mapped domain doubles as the NIT handover
  trigger.
- `from_samples(grid, values_dict)` fits all functions at once into one
  multi-output interpolant (Step 1c).
- **Checkpoints:** one `.npz` per chunk holding `grid_hash`, the linear-index
  range and values per function name. `merge(paths, grid)` validates the
  hash, full coverage and absence of duplicates. The parallel driver stays
  outside the package.

**Tests:**
- Map round trip.
- Jacobian and Hessian vs finite differences.
- Physical-variable derivatives vs finite differences.
- The `xp` map versions match the Numba versions (NumPy in CI, CuPy under
  `@pytest.mark.gpu`).
- Grid order matches `meshgrid(..., indexing='ij')`.
- `chunks → merge → from_samples` equals a direct fit.
- Out-of-domain detection in physical variables.

**As built, 5a** (differences from the design above):
- **One parametrised map instead of a registry of map types.** Every axis
  has `y = f((x − L)/W)`.
  - The transform f is `identity`, `log(s + offset)` or
    `(s + offset)^power`.
  - L is a constant or a boundary function.
  - W is a width, or `upper − L`; each is a constant or a boundary
    function.

  This covers the planned set (affine; log-type; affine between a
  boundary table and a fixed upper bound). It also covers `u = 1/r`
  (`power = −1`), which §0.2 needs but the initial set lacked; momenta
  relative to an `r`-dependent reference; and the log of the distance to
  a separatrix. Every map is still written twice (Numba and `xp`), but as
  one function with three transforms.
- **One boundary table per interpolant.** It is a real
  `ChebyshevInterpolantND` whose named outputs are the boundary functions,
  and maps refer to those outputs by name. The kernels therefore take a
  fixed number of arrays, and one multi-output table costs little more
  than one function. The restriction is that all boundary functions share
  one grid over the same inputs.
- **One level of dependence.** The boundary table's inputs must be axes
  with constant windows. The physical↔table conversion then takes one pass,
  and the Jacobian is diagonal apart from the boundary inputs' columns,
  which the Numba kernel exploits to apply the chain rule in place.
  Constructing an interpolant checks that the boundary table covers the
  physical range of its inputs.
- **Domains are in table coordinates.** `AxisMap.table_domain(lo, hi)`
  converts a physical interval for maps with constant windows.
- **Maps live in a wrapper class, `MappedInterpolant`** (new `mapped.py`),
  not in `ChebyshevInterpolantND`. Every method of the wrapper takes
  physical coordinates, and every method of `ChebyshevInterpolantND` stays
  in table coordinates, so `evaluate_physical` is
  `MappedInterpolant.evaluate`.
  - Single points: `evaluate`, `value_and_grad`.
  - Batches: `evaluate_batch`, `value_and_grad_batch`, and
    `value_grad_hess_batch` (build time: Step 3's derivative tables plus
    the map Hessian).
  - Coordinate changes: `to_table`, `to_physical`.
  - Precontraction: `precontract`, and `precontract_batch`, which returns a
    `MappedBatch`.
  - `raw()` returns a `RawMappedTable` for the Numba entry points
    `eval_physical_into` / `eval_grad_physical_into` (14 flat arguments;
    see Step 4 on why arguments are flat). The batched `xp` path is
    `MappedInterpolant.value_and_grad_batch`; there is no separate
    module-level `value_and_grad_physical_batch`.
- **Status and errors.** Bit i is set when axis i's table coordinate is
  outside the domain or NaN (an undefined map, e.g. the log of a negative
  number), or when coordinate i is outside the boundary table's domain.
  Messages are in physical terms, e.g. "p = 3.1 maps to 0.53, outside the
  table domain [−0.5, 0.5]".
- **Precontraction** maps the fixed values to table coordinates first.
  - If all of the boundary's inputs are fixed, the boundary functions
    become numbers bound into the remaining maps (one per member in a
    batch).
  - If only some are fixed, the boundary table is precontracted too.
  - An axis whose map uses the boundary can only be fixed together with
    all of the boundary's inputs.
- **Measured** on a PM-like (r, p_r, p_φ) = 30×8×10 table, value +
  gradient:

  | Variant | Raw kernel | Python `value_and_grad` | Batch, per point |
  |---|---|---|---|
  | unmapped | 2.06 µs | 3.7 µs | 1.1–1.3 µs |
  | u = 1/r | 2.23 µs | 4.7 µs | 1.2–1.4 µs |
  | u = 1/r, p_φ − ref(r) (boundary table) | 2.40 µs | 4.8 µs | 1.5 µs |

  On the compiled path the maps cost 0.2–0.35 µs.
- **Accuracy.** A 4D fixture with every kind of map (26×14×10×16, with
  boundary functions of two axes) matches the analytic function to 4e-13,
  its physical gradient to 4e-11 and its physical Hessian to 1.4e-9. The
  Kepler test of Step 4, rebuilt with `u = 1/r` and `p_φ` relative to a
  boundary `√r`, conserves energy to 7e-13 and follows the orbit to 5e-12.
- **GPU tests** (2) are written and have not been run.

**As built, 5b:**
- **Grid.** `Axis(name, n, domain, map=None)` and
  `ChebyshevGrid(axes, nodes="Chebyshev", boundary=None)`.
  - The node family belongs to the grid, not to each axis, because an
    interpolant has a single family.
  - The grid provides `table_nodes()`, `table_points(start, stop)`,
    `physical_points(start, stop)` (the points to sample at),
    `multi_index` / `linear_index`, `chunks(size)` and `grid_hash()`.
- **`from_samples` became `grid.fit(values)`.** It returns a
  `MappedInterpolant` if any axis is mapped, otherwise the plain table.
- **Checkpoints.**
  - `grid.save_chunk(path, start, stop, values)` writes one `.npz` per
    chunk.
  - `grid.load_chunks(paths)` (the planned `merge`) checks the hash, full
    coverage, overlaps and that every chunk holds the same functions.
- **`grid_hash()`** covers the node family, the axes (names, node counts,
  domains, maps) and the boundary table's exact coefficient bytes. Build
  the boundary table once and give every worker the same one: rebuilding
  it on a machine whose arithmetic differs in the last bit changes the
  hash.

### Step 6 — Serialization and metadata — IMPLEMENTED

Implemented on branch `step6-serialization` (2026-11-10). "As built" below
records where it differs from the design.

**Why:** the tables are built once (hours to days of Cython-force sampling),
reused everywhere, and must be identical between the prototype and pySEOBNR,
on CPU and GPU.

**Design:**
- **One canonical format: HDF5** (`h5py` as an optional extra,
  `chebyshevnd[io]`), with a documented schema and a `format_version`, so
  pySEOBNR or a C reader can load tables without importing chebyshevnd.
  Revision 1's npz fallback is dropped (D4).
- **Contents:**
  - the compact coefficients, plus the original N per axis and full-tensor
    error estimates;
  - node family, domains and axis names;
  - map specs (registry name + params; nested groups for table-bounded maps);
  - output names;
  - prefactor specs (Step 7);
  - dtype;
  - a metadata dict: model name, source hash of the Hamiltonian/force code,
    κ_max, build date, chebyshevnd version, `grid_hash`.
- Derivative tensors are not stored. They are only needed at build time and
  regenerate in milliseconds (D2).
- **Device placement:** `load(path, device="cpu" | "cuda")` reads on the
  host and copies to the GPU once (`cupy.asarray`); `interp.to_device(...)`
  moves an existing table.
- Optional float32 storage for amplitude tables (~1e-3 relative accuracy
  needed); upcast on load (A4). Dynamics tables stay float64.
- Cold start: loading compiles nothing beyond the evaluation kernels. Numba
  kernels are cached (A6), and CuPy caches its NVRTC-compiled RawKernels.

**Tests:**
- Round trip is bit-exact for float64 and complex128; float32 within 1e-7
  relative.
- Maps, names and metadata survive.
- An unknown `format_version` gives a clear error.
- `load(..., device="cuda")` gives the same results as the CPU load
  (`@pytest.mark.gpu`).

**As built** (differences from the design above):
- **API** (new `io.py`; h5py is imported on first use, from the `io` extra,
  which CI now installs):
  - `save(target, obj, *, metadata=None, float32=False, full=False,
    overwrite=False)`;
  - `load(source, *, device="cpu", return_metadata=False)`;
  - `read_metadata(source)`.

  `target` and `source` are file names or groups of an open `h5py.File`,
  so several tables can share a file. Saving refuses to replace an
  existing file or a non-empty group without `overwrite=True`: a table may
  take days to sample.
- **Grids are saved too** (`kind = "ChebyshevGrid"`). Workers can then load
  the very same grid and boundary table, which settles Step 5's caveat that
  a rebuilt boundary table can change `grid_hash`. On load the hash is
  recomputed and must match.
- **The schema is in `docs/file_format.md`**, part of the documentation.
  - Root attributes: `format`, `format_version` (1), `kind`,
    `chebyshevnd_version`, `created`.
  - A `metadata` group.
  - Per table: `coeffs` (always with the output axis), optional
    `coeffs_full`, `domains`, and the error estimate in its parts
    (`error_interpolation`, `error_truncation`, `coeff_scale`), so that
    `absolute_error` and `relative_error` survive exactly. Also the names,
    `grid_shape`, `nodes`, `domain_tol`, `dtype` and `storage_dtype`.
  - Mapped tables: groups `table`, `boundary` and `maps/<axis>`, with the
    map parameters as attributes. Step 5's single parametrised map needs
    no registry name.
- **Full tensor.** Optional (`full=True`) for an optimized table. A table
  that was never optimized stores its single tensor and gets it back as
  the full tensor, so `values_at_nodes` and `optimize` work after loading.
- **float32** applies to the table only (boundary tables stay double).
  The exact sum of what the rounding lost, `Σ|c − c₃₂|`, is added to
  `error_truncation`, so the stored estimate still bounds the error.
- **Metadata** is free-form: strings, numbers, booleans, arrays and
  sequences of those, and nested mappings (as subgroups). The automatic
  fields (`kind`, `format_version`, `chebyshevnd_version`, `created`) are
  reserved keys. `grid_hash` is not recorded automatically, because an
  interpolant does not know its grid; pass `metadata={"grid_hash":
  grid.grid_hash()}`.
- **Devices.** `ChebyshevInterpolantND.to_device("cuda")` and
  `MappedInterpolant.to_device("cuda")` fill the device caches now (the
  host copy is always kept for the Numba path), and
  `load(device="cuda")` calls them. The caches are keyed by module, so they
  serve CuPy's current device only; several GPUs would need one load per
  device.
- **Errors** name the file and group:
  - not a ChebyshevND table;
  - another format version (with the writer's version and an
    "upgrade" hint);
  - an unknown kind;
  - a grid whose hash does not match.
- **Prefactor specs** (Step 7) are not in format 1. Step 7 is deferred
  (D7), so format 1 stands. Adding prefactors later would need a new format
  version if a version-1 reader would otherwise silently ignore them.

### Step 7 — Regularised tabulation — DEFERRED (D7)

**Deferred on 2026-11-10.** Nothing that needs it remains:
- The NIT now uses an (α, β) equinoctial parametrisation, which removes the
  `1/e` behaviour of `f_w`. If the averaged equations take the form
  `F_α = α·G(p, e²)`, tabulate the regular `G` with
  `AxisMap("power", power=2)` on the e axis. That gives the even-in-e
  reduction considered below, with no new code, and avoids tabulating on a
  box in (α, β), which would not match the disk `e ≤ e_max`.
- Square-root behaviour at the separatrix is handled by Step 5's maps,
  which the design below already preferred.
- The PM Hamiltonian's additive baseline lives in the user's RHS. The
  package only has to record `H₀`'s name, which Step 6's metadata does.

Dropping the step also avoids a third run-time layer written twice (Numba
and `xp`) and a file format version 2. When the first (α, β) tables exist,
check the coefficient decay along e and run `check_derivatives` near e = 0.
If singular behaviour turns up after all (e.g. in the precession moments or
the mode amplitudes), the design below still applies.

**Why:** `f_w ∼ 1/e` as `e → 0`; √-type behaviour near the separatrix.

**Design:**
- Handle the separatrix primarily with maps (Step 5).
- For `e → 0`: an optional prefactor `P(x)` per output, chosen from a small
  named library: powers of one axis variable (`e⁻¹`, `e^|κ|`) and
  `√(x − x₀)`. Tabulate `g = f/P` and return `f = g·P`. Derivatives use the
  product rule, with `P′` and `P″` analytic per library entry. Prefactor
  specs are stored in the file (Step 6). Prefactors used at run time need an
  `xp` version too, like maps.
- **Alternative to evaluate first: parity on the e axis.** If `g` is even (or
  odd) in `e`, represent it on `[−e_max, e_max]` keeping only the even (odd)
  coefficients. That halves the coefficients and needs samples only at
  `e > 0` (root nodes are symmetric). It is also regular at `e = 0` by
  construction. Try it on test functions before choosing.
- **Additive baseline (PM Hamiltonian).** Tabulate `H − H₀`, where `H₀`
  is a cheap analytic approximation, and add `H₀` and its gradient back in
  the user's RHS. This can remove the dominant variation and shrink the
  coefficient count. The package only records `H₀`'s name in the file
  metadata.
- Hand-tabulating `e·f_w` in the NIT builder works with no package support.
  For the NIT, this step adds value only for derivatives and self-describing
  files, which makes it the lowest priority of the NIT-critical steps. For
  the PM Hamiltonian, decide once the first table has been pruned and
  benchmarked.

**Tests:**
- `sin(e)/e`-type functions through the prefactor route recover spectral
  convergence.
- Product-rule derivatives vs finite differences.

### Step 8 — Consolidation and polish (release `0.2.0`) — IMPLEMENTED

- Turn `ChebyshevInterpolant1D…4D` into thin wrappers over the nD class
  with the same public API, and route `precompute_x` / `evaluate_yz(w)`
  through `precontract`. Run the existing test suite unchanged. Drop the 2D
  staircase in favour of rectangular `kmax` unless the benchmark shows the
  staircase is faster (D5).
- Per-function-group pruning if the union `kmax` is measurably wasteful.
  This means several multi-output tensors grouped by resolution, which keeps
  A3's contiguous layout and A8's fixed shapes.
- Chebfun-style resolution diagnostics ("is this axis resolved?") per
  function. The current trailing-coefficient estimate (inherited from the
  2D–4D classes) also takes the second-to-last slice, to guard against
  parity zeros. It therefore overstates the error badly on axes where the
  function is an exact low-degree polynomial sampled with only one or two
  nodes to spare. For example, the Kepler table of Step 4 (linear in `k`
  on 3 nodes, quadratic in `p_r` on 4) reports a relative error of 1.07
  against an actual 9e-16.
- A benchmarks notebook covering:
  - cost vs dimension, vs F and vs precontraction;
  - derivative accuracy vs N;
  - the CPU-vs-GPU batch-size crossover.
- README (A3 wording, a GPU section, roadmap), `docs/api.rst`, examples;
  version `0.2.0`.

**As built** (2026-11-10, branch `step8-consolidation`):
- **The 1D–4D classes are kept as they are (D5, revised): no wrappers.**
  The benchmark Step 1 asked for:

  | v0.1 class → nD class, µs per call | evaluate | after `optimize` |
  |---|---|---|
  | 1D | 2.1 → 1.4 | 1.9 → 1.0 |
  | 2D | 1.4 → 1.7 | 0.58 → 1.27 |
  | 3D | 3.3 → 3.5 | 1.56 → 2.16 |
  | 4D | 15.2 → 12.5 | 5.3 → 4.3 |

  The nD path carries ~0.8 µs more Python and dispatch overhead per call,
  so wrappers would slow down the 2D/3D cases and the `precompute_x` paths.
  Keeping v0.1 behaviour would also need shims: no domain check, the 1D
  error estimate, the 2D staircase, settable `coeffs`. The classes stay as
  the simple fixed-dimension API, already fixed in v0.1.1, and
  `test_matches_v01_classes` keeps them in agreement with the nD class.
  The staircase question is moot.
- **No per-function-group pruning.** There is no real multi-output table
  yet to show that the union `kmax` is wasteful; revisit with the first NIT
  tables.
- **Resolution diagnostics:** `error_by_axis(relative=False)` gives the
  trailing-coefficient estimate per axis and function, to set node counts
  per axis.
  - The estimate now skips trailing slices that are exactly zero (a
    derivative's top coefficients), which leaves every other estimate
    unchanged.
  - The polynomial-axis case is inherent: from N samples one cannot tell
    an exact polynomial of degree N − 2 from an unresolved series. It is
    therefore documented, not changed, and one or two more nodes on that
    axis settle it.
- **An example notebook instead of a benchmarks notebook**
  (`examples/example_nd_workflow.ipynb`, rendered in the docs). It covers a
  multi-function table, precontraction (11.9 → 1.2 µs per call in its
  example), derivatives, a Numba RHS, a Hamiltonian through maps, chunked
  sampling and files. Benchmarks stay in `benchmarks/bench_evaluate.py`.
- **README and docs landing page** updated:
  - features;
  - extras;
  - an nD quickstart;
  - A3's wording;
  - the domain-check difference;
  - the roadmap;
  - GPU tests.
- **Version 0.2.0** in `pyproject.toml` and `CITATION.cff`. Run
  `pytest -m gpu` on a CUDA machine before publishing the release.

---

## 4. How the steps map to the project milestones

| Milestone | Steps required |
|---|---|
| Nonspinning v6EHM NIT (3D grid, 0/1/2PA, handover) | 0–6 (done). Step 7 is not needed: the (α, β) parametrisation removes the `1/e` (D7) |
| Aligned-spin v6EHM NIT (5D grid) | Step 5 checkpointing becomes essential (grid builds get expensive); cheap spin axes via per-axis N |
| Precessing v6EPHM NIT (5D grid + moment tables `g_n`, derivative tables for in-plane scalars) | All steps. Step 2 must support precontracting only `ν` while the spin-projection axes stay live; that live 4D table is the main performance risk (§0.1) |
| Multivoice waveform interpolation (mode amplitudes) | Steps 1, 2, 6 (float32), 8 (the amplitude grid is coarser and separate from the dynamics grid) |
| Interpolated PM Hamiltonian (§0.2) | Steps 0–4 and 5a (done). Step 6 to share tables; 5b only to organise sampling; an additive baseline, if pruning leaves the table too large, lives in the user's RHS (D7) |
| Batched GPU runs (CuPy) of any milestone above | Additionally 1e, 2b, 4b, the `xp` maps of 5a, and Step 6's device loading, all validated on a CUDA machine (§6) |

---

## 5. Decisions (settled 2026-10-18; D1 revised and D6 added 2026-11-09; D5 revised and D7 added 2026-11-10)

| # | Question | Answer | Consequence |
|---|---|---|---|
| D1 | Language of the production RHS | **CPU:** Numba, for single waveforms. **GPU:** CuPy, for batches (D6). The same tables and the same algorithm serve both. Keep the kernels portable | Step 4 has a Numba entry point (4a) and a CuPy batched path (4b). A1–A3 and the self-describing schema keep a Cython/C port cheap too |
| D2 | Are derivatives evaluated in every RHS call? | **NIT: no.** Derivatives build the higher-PA grids, and production folds the entire RHS into a single interpolant. **PM Hamiltonian: yes** (§0.2): value + gradient every call | Step 3 builds the NIT tables and is the oracle for Step 4. Step 4 includes the fused value + gradient kernel on both paths (Hessian on demand). The maps' Jacobian and chain rule join the hot path (5a). No derivative-aware precontraction and no stored derivative tensors |
| D3 | Must sampling start before the evaluation stack is done? | No | Step order unchanged: Step 5 after Step 4 |
| D4 | Serialization format | HDF5 only, for now | Step 6 |
| D5 | 1D–4D wrappers; 2D staircase pruning | **Revised 2026-11-10: keep the 1D–4D classes as they are.** Measured, wrappers would slow the 2D/3D and `precompute_x` paths (~0.8 µs more overhead per call) and change documented v0.1 behaviour | Step 8 adds no wrappers; the staircase stays in the 2D class; a cross-check test keeps the classes in agreement with the nD class |
| D7 | Regularised tabulation (Step 7) | **Deferred (2026-11-10):** the NIT uses an (α, β) equinoctial parametrisation without the `1/e`; Step 5's maps handle the separatrix; the PM baseline lives in the user's RHS | No `prefactors.py`; file format stays at version 1; Step 7's design kept for later |
| D6 | GPU framework | **CUDA via CuPy**, matching pySEOBNR's GPU port | A8; sub-steps 1e, 2b and 4b; `xp` maps (5a); device loading (Step 6); CuPy as an optional `gpu` extra; GPU tests run on a CUDA machine (§6) |

### 5.1 Open questions for the software engineer

These change details of 1e, 2b and 4b, not the plan's structure.

| # | Question | What it decides |
|---|---|---|
| Q1 | Is the batched RHS written as **CuPy array operations** over the batch, or as **custom CUDA kernels** (`RawKernel` / `RawModule`)? | Array operations: the A8 evaluator plugs in directly. Custom kernels: ship the evaluator as CUDA C `__device__` source to include in theirs (4b's fused kernel) |
| Q2 | How does the batched integrator step: fixed or adaptive, lockstep with masks? How are per-member events (handover, leaving the domain, merger) handled? | The meaning of `status`, and whether evaluation must skip masked members |
| Q3 | Typical batch sizes, and which GPUs (FP64 rate, memory)? | Whether per-member tables (Step 2b) fit in memory, and the CPU/GPU crossover |
| Q4 | Which CuPy and CUDA versions (e.g. `cupy-cuda12x`)? | The pin in the `gpu` extra, and the CUDA test machine |
| Q5 | Has the PM Hamiltonian already been ported to the GPU, and how? | The baseline the interpolated Hamiltonian must beat on the GPU (§0.2) |

---

## 6. Development conventions for these steps

- One step (or sub-step) per branch off `dev`, merged with tests green.
- Run `benchmarks/bench_evaluate.py` before and after every step, and record
  the numbers in the PR description.
- **GPU-touching changes:**
  - CI runs the shared `xp` code path with NumPy on every push.
  - Run `pytest -m gpu` on a CUDA machine before merging (the development
    Mac has no CUDA) and paste the result into the PR.
  - GPU tests skip cleanly when CuPy or a CUDA device is missing.
- Keep the v0.1 conventions: values-only constructors; truncation as counts,
  not indices; scalar `evaluate` returns a float; `clenshaw_nd(coeffs, xs,
  a=None, b=None, kmax=None)` signature preserved.
- New public API goes in `__all__` and `docs/api.rst` in the same change.

---

## 7. Non-goals (for this package)

- The NIT grid builder itself (sampling the EOB force, Fourier
  decomposition, 0/1/2PA algebra). That lives in the NIT package and only
  *uses* `ChebyshevGrid` / `from_samples` / `derivative` / `values_at_nodes`.
- Parallel job orchestration beyond the chunk/merge convention.
- The ODE integrators: the CPU one, and the batched GPU one in the
  engineer's pySEOBNR port. They set the per-call overhead floor (§0.1);
  ChebyshevND only provides the evaluation they call.
- Sparse or adaptive (non-tensor) grids. If 5D tensor grids become too
  expensive on the spin axes, the first remedy is fewer nodes on those axes,
  not a new grid type.
- FastEMRIWaveforms mode-amplitude tables. These were discussed on
  2026-10-20 and deferred. The CuPy path (A8, 4b) would make them possible
  later, but they would need amplitude data sampled at Chebyshev nodes.

---

## 8. Quick reference: files touched

| Step | New / modified |
|---|---|
| 0 (done) | `interp3d.py`, `interp4d.py`, `clenshaw.py`, `utils.py`, `.github/workflows/tests.yml`, `benchmarks/`, README, docs, examples |
| 1 (done) | new `kernels.py`, `interpnd.py`, `batched.py`, `backend.py`; `clenshaw.py` (`clenshaw_nd` on the kernel); `utils.py` (`axes=`); `__init__.py`; `docs/api.rst`; `benchmarks/bench_evaluate.py`; `pyproject.toml` (`gpu` extra and marker); tests `test_interpnd.py`, `test_batched.py`, `helpers.py` |
| 2 (done) | `interpnd.py` (`precontract`, `precontract_batch`), `batched.py` (`precontract_tables`), new `batch.py` (`ChebyshevBatch`), new `errors.py`; `docs/api.rst`; benchmarks; tests `test_precontract.py` |
| 3 (done) | new `derivatives.py` (`differentiate_coefficients`); `utils.py` (`chebyshev_values_at_nodes_nd`); `interpnd.py` (`derivative`, `values_at_nodes`, `axis_nodes`, `check_derivatives`); `docs/api.rst`; tests `test_derivatives.py` |
| 4 (done) | `kernels.py` (`value_and_grad_point`, `eval_into`, `eval_grad_into`, `RawTable`, `Workspace`), `batched.py` (`value_and_grad_batch`, basis rewrite), `batch.py` (`value_and_grad`), `interpnd.py` (`raw`, `value_and_grad`, `value_and_grad_batch`); `docs/api.rst`; benchmarks; tests `test_fused.py`. `cuda_kernels.py` (RawKernel) deferred to the GPU follow-up |
| 5 (done) | 5a: new `maps.py` (`AxisMap`, packing, `xp` and Numba maps, `eval_physical_into`, `eval_grad_physical_into`, `RawMappedTable`), new `mapped.py` (`MappedInterpolant`, `MappedBatch`), `interpnd.py` (shared per-member value stacking); 5b: new `grid.py` (`Axis`, `ChebyshevGrid`); `docs/api.rst`, `docs/conf.py`; benchmarks; tests `test_mapped.py`, `test_grid.py` |
| 6 (done) | new `io.py` (`save`, `load`, `read_metadata`), `interpnd.py` and `mapped.py` (`to_device`), `backend.py` (`resolve_device`), `pyproject.toml` (`io` extra), CI (installs it), new `docs/file_format.md`, `docs/api.rst`, `docs/index.md`; tests `test_io.py` |
| 7 (deferred) | — |
| 8 (done) | `interpnd.py` (`error_by_axis`), new `examples/example_nd_workflow.ipynb`, README, `docs/index.md`, `pyproject.toml` and `CITATION.cff` (0.2.0); tests in `test_interpnd.py` |

---

## 9. Revision history

**Revision 3.7 (2026-11-10).** Step 7 deferred (D7) and Step 8 implemented
(branch `step8-consolidation`):
- D5 revised: the 1D–4D classes are kept, on the evidence of the
  benchmarks;
- `error_by_axis`;
- an example notebook instead of a benchmarks notebook;
- README and docs updated;
- version 0.2.0.

**Revision 3.6 (2026-11-10).** Step 6 implemented (branch
`step6-serialization`). Step 6 now records how it was built and where it
differs from the plan:
- grids are saved too;
- the schema page `docs/file_format.md`;
- tables saved into groups;
- exact error estimates;
- a never-optimized tensor doubles as the full tensor;
- the float32 rounding is added to the error estimate;
- `to_device`.

**Revision 3.5 (2026-11-10).** Step 5 implemented (branch `step5-maps-grid`).
Step 5 now records how it was built and where it differs from the plan:
- one parametrised map per axis, with boundary functions from a single
  boundary table;
- one level of dependence;
- the `MappedInterpolant` wrapper instead of `evaluate_physical` on the
  nD class;
- `grid.fit` / `save_chunk` / `load_chunks`;
- a node family per grid rather than per axis;
- measured costs and accuracy.

**Revision 3.4 (2026-11-10).** Step 4 implemented (branch
`step4-fused-evaluation`). Step 4 now records how it was built and where it
differs from the plan:
- flat-array entry points (Numba dispatches tuples slowly);
- the branch scheme, at 1.0–1.4× a value evaluation;
- no Hessian;
- the batched basis rewrite;
- no RawKernel until GPU testing;
- measured timings, including the ODE RHS benchmark.

A8 now records the closed-form basis check. Step 8 notes the error
estimate's weakness on low-degree polynomial axes.

**Revision 3.3 (2026-11-10).** Step 3 implemented (branch
`step3-derivatives`). Step 3 now records how it was built and where it
differs from the plan:
- no derivative cache on the parent;
- a rounding term in the derivative error estimate;
- `axis_nodes()`;
- `check_derivatives` covers first and second orders.

**Revision 3.2 (2026-11-09).** Step 2 implemented (branch
`step2-precontraction`). Step 2 now records how it was built and where it
differs from the plan:
- 2a uses 2b's single-GEMM routine with a batch of one;
- the fixed-axes-first table is cached per set of axes;
- `precontract_batch` returns a `ChebyshevBatch`;
- measured timings.

**Revision 3.1 (2026-11-09).** Step 1 implemented and merged into `dev`.
Step 1 now records how it was built and where it differs from the plan:
- the function axis is opt-in via `output_names`;
- kernel signatures and the reason the work buffers stay separate;
- `absolute_error` includes truncation;
- batch chunking and device caching;
- measured timings.

The RawKernel for the Chebyshev values moved from 1e to 4b (A8, §8).

**Revision 3 (2026-11-09).** GPU support and the PM Hamiltonian.
- pySEOBNR's GPU port uses CUDA via CuPy, so batched GPU evaluation is now a
  requirement (§0.3):
  - D1 revised (Numba on the CPU, CuPy for GPU batches) and D6 added;
  - A8 added: a batch-first evaluator written once for NumPy and CuPy;
  - new sub-steps 1e (batched evaluator), 2b (batched precontraction) and 4b
    (CuPy value and value + gradient);
  - `xp` versions of maps (5a) and run-time prefactors (7), and device
    placement on load (6);
  - a GPU budget (§0.1), open questions for the engineer (§5.1), and GPU
    test conventions (§6).
- The second consumer is now named as a PM Hamiltonian (§0.2). Its GPU
  baseline is the analytic Hamiltonian batched on the same GPU.
- Step 0 is marked done (v0.1.1), and §1 now describes v0.1.1.
- FastEMRIWaveforms amplitudes are recorded as deferred (§7).

**Revision 2.2 (2026-10-18).** D2 revised: a second project interpolates an
EOB Hamiltonian and needs value + gradient at every RHS call (§0.2).
- Reinstated the fused derivative kernel in Step 4 (gradient now, Hessian on
  demand). A prototype measured it at ~2× a value evaluation.
- Split Step 5 into 5a (maps and chain rule, now on the hot path) and 5b
  (grid and sampling).
- Added the additive-baseline option to Step 7 and a Hamiltonian row to the
  milestones.

**Revision 2.1 (2026-10-18).** Settled D1–D5 (§5). The production RHS is
Numba and folds all PA orders into one table, so derivatives are build-time
only. Dropped:
- the fused derivative kernel (rev 2's Step 4b);
- `linear_combine`;
- derivative-aware precontraction;
- stored derivative tensors.

The budget (§0.1) now names the precessing model's live 4D table as the main
performance risk.

**Revision 2 (2026-10-18).**
- Checked revision 1's claims against the code:
  - the complex-dtype limits and the `partial_clenshaw_4d` bug are confirmed;
  - two further `ChebyshevInterpolant3D` bugs were found (§1);
  - the timings were re-measured (§0.1). Dispatch overhead is ~0.14 µs, not
    ~1 µs.
- Added Step 0 (bugfix release, benchmarks, CI on `dev`), the cross-cutting
  architecture section (§2), the open decisions (§5) and the conventions (§6).
- Step 1 now uses a leading-axis-first basis contraction with compact
  storage, chosen from prototype numbers. Complex dtype and the function
  axis are designed in from the start.
- The 1D–4D wrapper conversion moved to the end (A7).
- Precontraction is now a tensordot, with `linear_combine` added.
- Derivatives gained the build-time workflow (`values_at_nodes`). The fused
  derivative kernel is now conditional (D2).
- Maps are a serialisable named registry with a chain-rule helper. The
  box-domain check moved into Step 1; maps merged with the grid.
- Serialization uses HDF5 only and does not store derivative tensors by
  default.
- Prefactors gained the parity alternative and lower priority.

Old → new step numbers:

| Rev 1 | Rev 2 and later |
|---|---|
| 1 nD core | 1 (wrappers → 8) |
| 2 complex | 1b |
| 3 multi-output | 1c |
| 4 precontraction | 2 (2a; batched 2b added in rev 3) |
| 5 derivatives | 3 (fused part → Step 4: dropped in 2.1, reinstated in 2.2) |
| 6 fused evaluation | 4 (4a CPU; 4b GPU added in rev 3) |
| 7 maps / out-of-domain | box check → 1d; maps → 5 |
| 8 prefactors | 7 |
| 9 serialization | 6 |
| 10 grid / checkpointing | 5 |
| 11 polish | 8 |
| — | 0 (new; done in v0.1.1) |
