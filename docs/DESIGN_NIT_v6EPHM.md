# ChebyshevND — design document for the NIT / SEOBNRv6EPHM extension

**Revision:** 3.1 (2026-11-09).
- Step 1 is implemented and merged into `dev`; §3 records what was built
  and where it differs from the plan.
- GPU support with CUDA via CuPy is now a requirement (§0.3, A8, D6).
- The second consumer is an interpolated post-Minkowskian (PM) Hamiltonian
  (§0.2).
- Step 0 is complete (v0.1.1).
- §9 lists all revisions and maps the old step numbers to the new ones.

**Branch:** `dev` (v0.1.1 plus Step 1).

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
subtracting an analytic baseline (Step 7).

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
  - As implemented in Step 1e, this is a Python loop over `k` of array
    operations, so on CuPy it launches one kernel per `k`.
  - Computing all axes' bases in a single `cupy.RawKernel` (CUDA C compiled
    at run time by NVRTC, so no CUDA build step) is deferred to Step 4b. It
    could not be compiled or tested without a CUDA machine, and GPU
    benchmarks should justify it first.
  - `cos(k·arccos t)` is not used: it loses about half the digits near
    `t = ±1`.
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
  classes (D5).
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

### Step 2 — Precontraction over an arbitrary subset of axes

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

**Tests:**
- Precontracted evaluation equals full evaluation to 1e-14 at random points,
  for every subset of axes including non-leading ones.
- Works on complex, multi-output and optimized tensors.
- 2b equals running 2a member by member (NumPy in CI; CuPy under
  `@pytest.mark.gpu`).

### Step 3 — Analytic derivatives

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

**Tests:**
- Derivatives of separable test functions to ~1e-10 (first) and ~1e-8
  (second) on 20-node axes.
- Mixed-partial symmetry `∂ₓ∂_y = ∂_y∂ₓ`.
- Chain rule under domain rescaling.
- `derivative ∘ precontract == precontract ∘ derivative`.
- `values_at_nodes` round trip reproduces the input samples to 1e-14 for
  both node families.

### Step 4 — Fused evaluation for the ODE RHS (values and gradients)

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

### Step 5 — Domain maps, grid definition and sampling workflow

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

### Step 6 — Serialization and metadata

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

### Step 7 — Regularised tabulation

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

### Step 8 — Consolidation and polish (release `0.2.0`)

- Turn `ChebyshevInterpolant1D…4D` into thin wrappers over the nD class
  with the same public API, and route `precompute_x` / `evaluate_yz(w)`
  through `precontract`. Run the existing test suite unchanged. Drop the 2D
  staircase in favour of rectangular `kmax` unless the benchmark shows the
  staircase is faster (D5).
- Per-function-group pruning if the union `kmax` is measurably wasteful.
  This means several multi-output tensors grouped by resolution, which keeps
  A3's contiguous layout and A8's fixed shapes.
- Chebfun-style resolution diagnostics ("is this axis resolved?") per
  function.
- A benchmarks notebook covering:
  - cost vs dimension, vs F and vs precontraction;
  - derivative accuracy vs N;
  - the CPU-vs-GPU batch-size crossover.
- README (A3 wording, a GPU section, roadmap), `docs/api.rst`, examples;
  version `0.2.0`.

---

## 4. How the steps map to the project milestones

| Milestone | Steps required |
|---|---|
| Nonspinning v6EHM NIT (3D grid, 0/1/2PA, handover) | 0–6. Step 7 is optional: the `1/e` prefactor can be applied by hand in the builder |
| Aligned-spin v6EHM NIT (5D grid) | Step 5 checkpointing becomes essential (grid builds get expensive); cheap spin axes via per-axis N |
| Precessing v6EPHM NIT (5D grid + moment tables `g_n`, derivative tables for in-plane scalars) | All steps. Step 2 must support precontracting only `ν` while the spin-projection axes stay live; that live 4D table is the main performance risk (§0.1) |
| Multivoice waveform interpolation (mode amplitudes) | Steps 1, 2, 6 (float32), 8 (the amplitude grid is coarser and separate from the dynamics grid) |
| Interpolated PM Hamiltonian (§0.2) | Steps 0–4 and 5a. Step 6 to share tables; 5b only to organise sampling; Step 7's baseline if pruning leaves the table too large |
| Batched GPU runs (CuPy) of any milestone above | Additionally 1e, 2b, 4b, the `xp` maps of 5a, and Step 6's device loading, all validated on a CUDA machine (§6) |

---

## 5. Decisions (settled 2026-10-18; D1 revised and D6 added 2026-11-09)

| # | Question | Answer | Consequence |
|---|---|---|---|
| D1 | Language of the production RHS | **CPU:** Numba, for single waveforms. **GPU:** CuPy, for batches (D6). The same tables and the same algorithm serve both. Keep the kernels portable | Step 4 has a Numba entry point (4a) and a CuPy batched path (4b). A1–A3 and the self-describing schema keep a Cython/C port cheap too |
| D2 | Are derivatives evaluated in every RHS call? | **NIT: no.** Derivatives build the higher-PA grids, and production folds the entire RHS into a single interpolant. **PM Hamiltonian: yes** (§0.2): value + gradient every call | Step 3 builds the NIT tables and is the oracle for Step 4. Step 4 includes the fused value + gradient kernel on both paths (Hessian on demand). The maps' Jacobian and chain rule join the hot path (5a). No derivative-aware precontraction and no stored derivative tensors |
| D3 | Must sampling start before the evaluation stack is done? | No | Step order unchanged: Step 5 after Step 4 |
| D4 | Serialization format | HDF5 only, for now | Step 6 |
| D5 | 1D–4D wrappers; 2D staircase pruning | Wrappers in Step 8; drop the staircase unless a benchmark shows it beats rectangular `kmax` | Step 8 |
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
| 2 | `interpnd.py` (`precontract`), `batched.py` (batched precontraction) |
| 3 | new `derivatives.py`, `utils.py` (inverse DCT), `interpnd.py` (`derivative`, `values_at_nodes`, `check_derivatives`) |
| 4 | `kernels.py` (`eval_into`, `eval_grad_into`), `batched.py` (`value_and_grad_batch`), new `cuda_kernels.py` (RawKernel sources, moved from Step 1), `interpnd.py` (`raw`, `value_and_grad`) |
| 5 | 5a: new `maps.py` (Numba and `xp` versions, `eval_grad_physical_into`, `value_and_grad_physical_batch`), `interpnd.py` (`evaluate_physical`); 5b: new `grid.py` |
| 6 | new `io.py` (with device placement), `pyproject.toml` (`io` extra) |
| 7 | new `prefactors.py` |
| 8 | `interp1d–4d.py`, README, `docs/api.rst`, examples, benchmarks |

---

## 9. Revision history

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
