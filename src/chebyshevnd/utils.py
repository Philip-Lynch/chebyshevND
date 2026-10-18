"""Node generation, coefficient computation, and pruning utilities."""

import numpy as np
from scipy.fft import dctn
from numba import njit


@njit(cache=True)
def rescale_variable(x, a, b):
    """
    Map x from [a, b] to [-1, 1].

    Parameters
    ----------
    x : float
        Point in [a, b].
    a, b : float
        Domain bounds.

    Returns
    -------
    float
        The image of x in [-1, 1].
    """
    return 2 * (x - a) / (b - a) - 1


def chebyshev_gauss_lobatto_nodes(N,a,b):
    """
    Generate the N Chebyshev-Gauss-Lobatto (extrema) nodes on [a, b].

    These are the extrema of the Chebyshev polynomial T_{N-1} and include
    the endpoints a and b. The nodes are returned in descending order;
    sample your function at them in this order, since the coefficient
    computation assumes it.

    Parameters
    ----------
    N : int
        Number of nodes.
    a, b : float
        Domain bounds.

    Returns
    -------
    np.ndarray
        Chebyshev-Gauss-Lobatto nodes in [a, b].
    """
    # Chebyshev-Gauss-Lobatto nodes on [-1, 1]
    x_cheb =  np.cos(np.pi * np.arange(N) / (N - 1))
    # Rescale to [a, b]
    x = 0.5 * (b - a) * x_cheb + 0.5 * (a + b)
    return x


def chebyshev_nodes(N,a,b):
    """
    Generate the N Chebyshev (root) nodes on [a, b].

    These are the roots of the Chebyshev polynomial T_N and exclude the
    endpoints a and b. The nodes are returned in descending order;
    sample your function at them in this order, since the coefficient
    computation assumes it.

    Parameters
    ----------
    N : int
        Number of nodes.
    a, b : float
        Domain bounds.

    Returns
    -------
    np.ndarray
        Chebyshev nodes in (a, b).
    """
    # Chebyshev nodes on [-1, 1]
    x_cheb = np.cos(np.pi * (2*np.arange(N)+1) / (2*N))
    # Rescale to [a, b]
    x = 0.5 * (b - a) * x_cheb + 0.5 * (a + b)
    return x


def renormalize_coefficients_GL(coeffs,copy=True):
    """
    Renormalize raw DCT-I output on a Gauss-Lobatto (extrema) tensor grid to
    standard Chebyshev-series convention, for arbitrary dimension.

    This generalises:
    2D: divide by (Nx-1)(Ny-1), then halve rows/cols at indices 0 and -1;
    3D: divide by (Nx-1)(Ny-1)(Nz-1), then halve each of the 6 faces, etc.

    Parameters
    ----------
    coeffs : np.ndarray
        Output of ``scipy.fft.dctn(..., type=1, norm=None)``, shape (N1, ..., Nd).
    copy : bool, optional
        If True (default), work on a copy and return it. If False, scale in place.

    Returns
    -------
    np.ndarray
        Renormalized Chebyshev coefficients, same shape as the input.
    """
    arr = coeffs.copy() if copy else coeffs
    shape = arr.shape

    if any(N < 2 for N in shape):
        raise ValueError("All dimensions must have size >= 2 for DCT-I normalization.")

    # 1) divide by prod(Nr - 1)
    scale = 1.0
    for N in shape:
        scale *= (N - 1)
    arr /= scale

    # 2) halve endpoint hyperplanes along each axis
    ndim = arr.ndim
    for axis, N in enumerate(shape):
        # first index plane
        sl = [slice(None)] * ndim
        sl[axis] = 0
        arr[tuple(sl)] *= 0.5
        # last index plane
        sl[axis] = N - 1
        arr[tuple(sl)] *= 0.5

    return arr


def renormalize_coefficients_Chebyshev(out: np.ndarray) -> np.ndarray:
    """
    Renormalize raw DCT-II output on a Chebyshev (root) tensor grid to
    standard Chebyshev-series convention, for arbitrary dimension.

    Parameters
    ----------
    out : np.ndarray
        Output of ``scipy.fft.dctn(..., type=2, norm=None)``, n-dimensional.

    Returns
    -------
    np.ndarray
        Renormalized Chebyshev coefficients, same shape as the input.
    """
    shape = out.shape
    ndim = out.ndim

    # Create factor arrays for each axis
    factors = []
    for n in shape:
        f = np.ones(n)
        f[0] = 2
        factors.append(f)

    # Build the scale tensor via outer products (einsum over all dims)
    einsum_subscripts = ",".join(chr(ord("a") + i) for i in range(ndim)) + "->" + "".join(chr(ord("a") + i) for i in range(ndim))
    scale = np.einsum(einsum_subscripts, *factors)

    # Normalize by scale * product of sizes
    normalized = out / (scale * np.prod(shape))

    return normalized


def compute_chebyshev_coefficients_nd(f_values: np.ndarray,nodes = 'Chebyshev') -> np.ndarray:
    """
    Compute Chebyshev coefficients for an nD array of function values using a DCT.

    Parameters
    ----------
    f_values : np.ndarray
        n-dimensional array of function values sampled on a Chebyshev tensor
        grid of the chosen node family (``chebyshev_nodes`` along each axis
        for 'Chebyshev', ``chebyshev_gauss_lobatto_nodes`` for 'GaussLobatto').
    nodes : str, optional
        Node family the values were sampled at: 'Chebyshev' (default) or
        'GaussLobatto'.

    Returns
    -------
    np.ndarray
        n-dimensional array of Chebyshev coefficients, same shape as the input.
    """
    if(nodes == 'Chebyshev'):
        coeffs = dctn(f_values, type=2, norm=None)
        coeffs = renormalize_coefficients_Chebyshev(coeffs)
    elif(nodes == 'GaussLobatto'):
        coeffs = dctn(f_values, type=1, norm=None)
        coeffs = renormalize_coefficients_GL(coeffs)
    else:
        raise ValueError('Incorrect node type. Please choose either Chebyshev or GaussLobatto')
    return coeffs


def prune_trailing_slices_greedy_nd(coeffs, delta=1e-10, scale='max'):
    """
    Structured pruning of nD Chebyshev coefficients: greedily remove trailing
    slices along axes to reduce Clenshaw lengths while controlling a global
    relative error.

    Error control: ``sum(|discarded coeffs|) <= delta * S``, with S given by
    `scale`.

    Parameters
    ----------
    coeffs : np.ndarray
        Chebyshev coefficients, n-dimensional.
    delta : float, optional
        Allowed global relative error.
    scale : {'max', 'l1', 'l2'} or float, optional
        Scale S for the relative tolerance:

        - 'max' (default): ``S = max(abs(coeffs))``
        - 'l1' : ``S = sum(abs(coeffs))``
        - 'l2' : ``S = norm(coeffs.ravel(), 2)``
        - float: use the provided value

    Returns
    -------
    kmax : tuple of int
        Per-axis truncation counts; keep
        ``coeffs[:kmax[0], :kmax[1], ..., :kmax[n-1]]``.
    discarded_abs_sum : float
        Sum of ``|coeffs|`` discarded — an upper bound on the absolute uniform
        error introduced by the truncation.
    """
    c = np.asarray(coeffs)
    shape = c.shape
    ndim = c.ndim
    if c.size == 0:
        return (0,) * ndim, 0.0

    abs_c = np.abs(c)
    if scale == 'max':
        S = abs_c.max()
    elif scale == 'l1':
        S = abs_c.sum()
    elif scale == 'l2':
        S = np.sqrt(np.sum(abs_c**2))
    elif isinstance(scale, (int, float)):
        S = float(scale)
    else:
        raise ValueError("scale must be 'max', 'l1', 'l2', or a float")

    if S == 0.0:
        # all-zero; keep nothing
        return (0,) * ndim, 0.0

    tau = delta * S

    # Precompute L1 costs of each trailing slice along each axis
    # cost[axis][i] = L1 norm of slice at index i along that axis
    costs = []
    for ax in range(ndim):
        summed = abs_c.sum(axis=tuple(j for j in range(ndim) if j != ax))
        costs.append(summed)

    # Start from full sizes
    kmax = list(shape)
    discarded = 0.0

    while True:
        # compute cost of trimming one more slice on each axis
        next_costs = []
        for ax in range(ndim):
            idx = kmax[ax] - 1
            if idx >= 0:
                next_costs.append(costs[ax][idx])
            else:
                next_costs.append(np.inf)

        axis = int(np.argmin(next_costs))
        inc = next_costs[axis]

        if not np.isfinite(inc):
            break
        if discarded + inc > tau:
            break

        discarded += inc
        kmax[axis] -= 1

        if any(k == 0 for k in kmax):
            break

    return tuple(kmax), discarded


def find_significant_coeffs_1d(coefficients, rel_tol=1e-8, max_coeff=None, x_max_minus_i=1.):
    """
    Count how many leading 1D Chebyshev coefficients are worth keeping.

    Scans from the tail for the last coefficient whose magnitude exceeds
    ``rel_tol * max_coeff / (x_max_minus_i * len(coefficients))``.

    Parameters
    ----------
    coefficients : np.ndarray
        1D array of Chebyshev coefficients.
    rel_tol : float, optional
        Relative tolerance for coefficient significance.
    max_coeff : float, optional
        Reference magnitude for the relative threshold. Defaults to
        ``max(abs(coefficients))``.
    x_max_minus_i : float, optional
        Extra scale factor used by the 2D routine to tighten the per-row
        threshold. Leave at 1 for standalone use.

    Returns
    -------
    int
        Number of leading coefficients to keep (a count, not an index);
        0 if no coefficient is significant.
    """
    if max_coeff is None:
        max_coeff = np.max(np.abs(coefficients))

    nymax = 0
    # Find the last significant coefficient
    n = len(coefficients)
    for i in range(n - 1, -1, -1):
        if abs(coefficients[i]) > (rel_tol * max_coeff) /x_max_minus_i/(n):
            nymax = i + 1
            break
    else:
        return 0  # If all coefficients are below tol, keep none

    return nymax


def find_significant_coeffs_2d(coefficients, rel_tol=1e-8,max_coeff=None, x_max_minus_i=1.):
    """
    Per-row truncation counts for a 2D Chebyshev coefficient matrix.

    For each row i, counts the significant coefficients along the second
    axis; trailing all-insignificant rows are trimmed from the result. The
    returned array is suitable as the ``NyMaxes`` argument of
    :func:`chebyshevnd.clenshaw.clenshaw_2d`.

    Parameters
    ----------
    coefficients : np.ndarray
        2D array of Chebyshev coefficients, shape (Nx, Ny).
    rel_tol : float, optional
        Relative tolerance for coefficient significance.
    max_coeff : float, optional
        Reference magnitude for the relative threshold. Defaults to
        ``max(abs(coefficients))``.
    x_max_minus_i : float, optional
        Extra scale factor for the per-row threshold.

    Returns
    -------
    np.ndarray of int64
        Truncation count for each kept row (a count per row, not indices).
    """
    nx, ny = coefficients.shape
    tmp = np.zeros(nx)
    nyMax = np.zeros(nx)

    if coefficients.ndim != 2:
        raise ValueError("Input coefficients must be a 2D array.")

    if rel_tol <= 0:
        raise ValueError("Relative tolerance must be positive.")
    if max_coeff is None:
        max_coeff = np.max(np.abs(coefficients))


    # Evaluate in the y-direction first
    for i in range(nx):
        nyMax[i] = find_significant_coeffs_1d(coefficients[i, :], rel_tol / x_max_minus_i ,max_coeff=max_coeff, x_max_minus_i = nx)

    # Trim off the trailing zeros
    last_nz = np.max(np.nonzero(nyMax))
    nyMaxTrimmed = nyMax[:last_nz + 1]

    return nyMaxTrimmed.astype(np.int64)
