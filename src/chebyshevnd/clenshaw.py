
import numpy as np
from numba import njit
from .utils import rescale_variable

@njit
def clenshaw_1d(coeffs, x,  k_max=None, a=-1.0, b=1.0):
    """
    Evaluate Chebyshev series at a single point x using Clenshaw's algorithm.
    
    Parameters:
        coeffs: array of Chebyshev coefficients
        x: point of evaluation in [a, b]
        a, b: domain bounds (default [-1, 1])
        k_max : int, optional
            Number of coefficients to use (truncate at this order).
            Defaults to len(coeffs).
    
    Returns:
        interpolated value at x
    """
    if k_max is None:
        k_max = len(coeffs)
    if k_max <= 0:
        return 0.0

    # Map x from [a, b] to [-1, 1]
    xt = (2.0 * x - a - b) / (b - a)
    
    b_k1 = 0.0
    b_k2 = 0.0
    for k in range(k_max-1, 0, -1):
        b_k = 2 * xt * b_k1 - b_k2 + coeffs[k]
        b_k2 = b_k1
        b_k1 = b_k
    return coeffs[0] + xt * b_k1 - b_k2

@njit
def clenshaw_2d(coeffs, x, y, NyMaxes = None, ax = -1., bx=1, ay=-1., by=1):
    """
    Evaluate a 2D Chebyshev interpolant using the Clenshaw algorithm.

    Parameters:

        coeffs : 2D np.ndarray
            Matrix of Chebyshev coefficients [Nx, Ny].
        x, y : float
            Point to evaluate.
        Nx : int, optional
            Number of Chebyshev points in x-direction. If None, defaults to shape(coeffs)[0].
        Ny : int, optional
            Number of Chebyshev points in y-direction. If None, defaults to shape(coeffs)[1].
        ax, bx : float
            Domain bounds in x-direction.
        ay, by : float
            Domain bounds in y-direction.

    Returns:
        float
            Value of interpolant at (x, y)
    """
    if NyMaxes is None:
        NyMaxes = np.ones(coeffs.shape[0], dtype=np.int64) * coeffs.shape[1]

    # Map x, y from [ax, bx] and [ay, by] to [-1, 1]
    tx = (2 * x - (bx + ax)) / (bx - ax)
    ty = (2 * y - (by + ay)) / (by - ay)

    # Evaluate in y-direction first for each fixed x
    temp = np.array([clenshaw_1d(coeffs[i, :], ty,NyMaxes[i]) for i in range(NyMaxes.shape[0])])
    
    # Then evaluate resulting 1D Chebyshev series in x-direction
    return clenshaw_1d(temp, tx,NyMaxes.shape[0])

''' 
@njit
def clenshaw_3d(coeffs, x, y, z, kmax=None,
                ax=-1., bx=1., ay=-1., by=1., az=-1., bz=1.):
    """
    3D Clenshaw, reusing clenshaw_2d inside.
    """
    # Nx, Ny, Nz = coeffs.shape
    if kmax is None:
        kmax = coeffs.shape

    # Map z
    tz = (2 * z - (bz + az)) / (bz - az)
    NzMaxes = np.ones((kmax[0], kmax[1]), dtype=np.int64) * kmax[2]
    # Evaluate along z for each (i, j)
    temp_xy = np.empty((kmax[0], kmax[1]))
    for i in range(kmax[0]):
        for j in range(kmax[1]):
            temp_xy[i, j] = clenshaw_1d(coeffs[i, j, :], tz, NzMaxes[i, j])

    # Now evaluate the 2D interpolant in (x, y)
    return clenshaw_2d(temp_xy, x, y, ax=ax, bx=bx, ay=ay, by=by)

''' 
@njit
def clenshaw_3d(coeffs, x, y, z, 
                ax = -1., bx = 1., 
                ay = -1., by = 1., 
                az = -1., bz = 1.,
                kmax=None):
    """
    coeffs: (nx,ny,nz) full coefficients (you may pass a trimmed block)
    kmax: (ix,iy,iz) optional; if None uses coeffs.shape
    """
    nx, ny, nz = coeffs.shape
    if kmax is None:
        ix, iy, iz = nx, ny, nz
    else:
        ix, iy, iz = kmax

    # Scale x,y,z to [-1,1]
    tx = rescale_variable(x, ax, bx)
    ty = rescale_variable(y, ay, by)
    tz = rescale_variable(z, az, bz)

    # Evaluate along z first using only [:,:iz]
    tmp_yx = np.empty((ix, iy))
    for i in range(ix):
        for j in range(iy):
            tmp_yx[i, j] = clenshaw_1d(coeffs[i, j, :iz], tz)

    # Then along y using only [:iy]
    tmp_x = np.empty(ix)
    for i in range(ix):
        tmp_x[i] = clenshaw_1d(tmp_yx[i, :iy], ty)

    # Finally along x using only [:ix]
    return clenshaw_1d(tmp_x[:ix], tx)

@njit
def clenshaw_4d(coeffs, x, y, z, w,
                ax=-1., bx=1., ay=-1., by=1.,
                az=-1., bz=1., aw=-1., bw=1.,
                kmax=None):
    """
    4D Clenshaw, reusing clenshaw_3d inside.
    """
    Nx, Ny, Nz, Nw = coeffs.shape

    if kmax is None:
        ix, iy, iz,iw = Nx, Ny, Nz, Nw
    else:
        ix, iy, iz, iw = kmax
    # Map w
    tw = (2 * w - (aw + bw)) / (bw - aw)

    # Evaluate along w for each (i, j, k)
    temp_xyz = np.empty((ix, iy, iz))
    for i in range(ix):
        for j in range(iy):
            for k in range(iz):
                temp_xyz[i, j, k] = clenshaw_1d(coeffs[i, j, k, :iw], tw)

    # Now evaluate the 3D interpolant in (x, y, z)
    return clenshaw_3d(temp_xyz, x, y, z,
                       ax=ax, bx=bx,
                       ay=ay, by=by,
                       az=az, bz=bz,
                       kmax = (ix, iy, iz))


@njit
def partial_clenshaw_4d(coeffs, x_value, a=-1., b=1., kmax = None):
        """
        Collapse x dimension at fixed x_value.
        Returns a 3D array of effective coefficients in (y, z, w).
        """
        tx = rescale_variable(x_value, a, b)

        Nx, Ny, Nz, Nw = coeffs.shape

        if kmax is None:
            ix, iy, iz,iw = Nx, Ny, Nz, Nw
        else:
            ix, iy, iz, iw = kmax

        temp_yzw = np.empty((iy, iz, iw))

        for j in range(iy):
            for k in range(iz):
                for l in range(iw):
                    coeffs_x = coeffs[:Nx, j, k, l]
                    temp_yzw[j, k, l] = clenshaw_1d(coeffs_x, tx)

        return temp_yzw

def _clenshaw_nd_unit(ts, coeffs):
    d = ts.shape[0]

    # Base case
    if d == 1:
        return clenshaw_1d(coeffs, ts[0])

    t_last = ts[-1]
    shape = coeffs.shape[:-1]
    tmp = np.empty(shape, dtype=np.float64)

    # Loop explicitly, no np.ndindex (Numba safer with nested loops)
    it = np.nditer(tmp, flags=['multi_index'], op_flags=['writeonly'])
    while not it.finished:
        idx = it.multi_index
        # Force 1D view along the last axis
        coeff_vec = coeffs[idx + (slice(None),)]
        it[0] = clenshaw_1d(coeff_vec, t_last)
        it.iternext()

    return _clenshaw_nd_unit(ts[:-1], tmp)


def clenshaw_nd(coeffs, xs, a=None, b=None, kmax=None):
    """
    Evaluate an nD Chebyshev series at a single point.

    Parameters:
        coeffs : np.ndarray
            nD array of Chebyshev coefficients.
        xs : array-like
            The n coordinates of the evaluation point, one per axis of coeffs.
        a, b : array-like, optional
            Per-axis lower and upper domain bounds (default [-1, 1] on every axis).
        kmax : tuple, optional
            Per-axis truncation counts; only coeffs[:kmax[0], ..., :kmax[n-1]] are used.

    Returns:
        float: value of the interpolant at xs.
    """
    xs = np.asarray(xs, dtype=np.float64)
    d = coeffs.ndim
    a = -np.ones(d) if a is None else np.asarray(a, dtype=np.float64)
    b = np.ones(d) if b is None else np.asarray(b, dtype=np.float64)

    ts = (2.0 * xs - a - b) / (b - a)
    if kmax is not None:
        coeffs = coeffs[tuple(slice(int(k)) for k in kmax)]
    return _clenshaw_nd_unit(ts, coeffs)