"""Exceptions and messages shared by the interpolant classes."""


class OutOfDomainError(ValueError):
    """Raised when an interpolant is evaluated outside its domain."""


def describe_out_of_domain(names, a, b, x, status):
    """
    Describe the coordinates of a point that lie outside the domain.

    Parameters
    ----------
    names : sequence of str
        Axis names.
    a, b : sequence of float
        Lower and upper bound of each axis.
    x : sequence of float
        The point, one coordinate per axis.
    status : int
        Bitmask: bit i is set when coordinate i is outside its domain.

    Returns
    -------
    str
        For example ``"u = 3.5 not in [-1.0, 3.0]; e = 0.95 not in [0.0, 0.9]"``.
    """
    return "; ".join(
        f"{names[i]} = {float(x[i])!r} not in [{float(a[i])!r}, {float(b[i])!r}]"
        for i in range(len(names)) if status >> i & 1
    )
