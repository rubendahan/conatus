"""Chebyshev-series helpers for the sensitivity densities (exact polynomial arithmetic).

Mode shapes come out of the tau discretization as Chebyshev coefficients on [a, b]; products,
derivatives and integrals of those series are done exactly with ``numpy.polynomial.Chebyshev``
(no quadrature, no aliasing). A non-polynomial coefficient c(y) (a base flow, a feedback weight)
is converted to a Chebyshev series by *adaptive* interpolation: the degree is doubled until the
8 trailing coefficients fall below max(``CHEB_TAIL_TOL``, 10·deg·eps) relative to the largest (the
second term is the rounding floor of the interpolation itself), the series is then chopped at that
level, and the degree used is returned so that callers report it (never chosen silently).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import sympy
from numpy.polynomial import Chebyshev

from ..errors import ConvergenceError
from ..nonmodal.system import Y

CHEB_TAIL_TOL = 1e-14        # relative size of the 8 trailing coefficients of an adaptive fit
CHEB_MAX_DEGREE = 4096       # refusal threshold for the adaptive fit


def series(coeffs, bounds) -> Chebyshev:
    """Chebyshev series Σ c_n T_n(t), t the affine image of y ∈ bounds on [−1, 1]."""
    return Chebyshev(np.asarray(coeffs, dtype=complex), domain=list(bounds))


def as_function(f) -> Callable[[np.ndarray], np.ndarray]:
    """sympy expression in Y, number or callable -> vectorized numpy function of y."""
    if callable(f) and not isinstance(f, sympy.Basic):
        return lambda y: np.asarray(f(y), dtype=complex) + 0 * y
    e = sympy.sympify(f)
    fn = sympy.lambdify(Y, e, "numpy")
    return lambda y: np.asarray(fn(y), dtype=complex) + 0 * y


def adaptive(f, bounds, *, what: str) -> tuple[Chebyshev, int]:
    """Chebyshev interpolant of ``f`` on ``bounds`` with trailing coefficients below
    max(``CHEB_TAIL_TOL``, 10·deg·eps) (relative); returns (chopped series, degree tried)."""
    fn = as_function(f)
    deg = 16
    while deg <= CHEB_MAX_DEGREE:
        s = Chebyshev.interpolate(fn, deg, domain=list(bounds))
        c = np.abs(s.coef)
        scale = c.max()
        if scale == 0.0:
            return series([0.0], bounds), 0
        tol = max(CHEB_TAIL_TOL, 10 * deg * np.finfo(float).eps) * scale
        if c[-8:].max() <= tol:
            keep = int(np.nonzero(c > tol)[0].max()) + 1       # chop the rounding-level tail
            return series(s.coef[:keep], bounds), deg
        deg *= 2
    raise ConvergenceError(f"adaptive Chebyshev fit of {what} did not converge by degree "
                           f"{CHEB_MAX_DEGREE} (tail > {CHEB_TAIL_TOL:g} relative)",
                           fix="the function is not smooth on the interval, or not resolved")


def conj(s: Chebyshev) -> Chebyshev:
    """Complex conjugate of a Chebyshev series (real basis: conjugate the coefficients)."""
    return Chebyshev(np.conj(s.coef), domain=s.domain)


def integral(s: Chebyshev) -> complex:
    """∫_a^b s(y) dy, exact."""
    a, b = s.domain
    F = s.integ()
    return complex(F(b) - F(a))
