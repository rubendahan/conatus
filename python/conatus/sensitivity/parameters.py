"""Eigenvalue sensitivity to parameters: one entry point over the three existing mechanisms.

All three evaluate the same first-order formula for the pencil (L + σ M) x = 0 with its left vector
y (yᴴ (L + σ M) = 0):

    dσ/dp = −yᴴ (∂L/∂p + σ ∂M/∂p) x / (yᴴ M x),

and differ only in where the matrices and their derivatives come from:

* ``EVPResult`` (``conatus.solvers.solve_dense(..., left=True)``): the user passes ∂L/∂p, ∂M/∂p
  (``conatus.solvers.sensitivity``);
* ``ParametricEVP`` + ``TrackedMode``: *exact* derivatives of the Laurent-polynomial pencil
  (``ParametricEVP.sensitivity``);
* ``ModePair`` (a ``ModalSystem`` built by a factory ``factory(p)``): derivatives of the tau
  descriptor by Richardson-extrapolated central differences with a *declared* relative step
  (``conatus.solvers.matrix_derivative``; error O(h⁴) in the matrices, reported by comparing two
  steps) — or an explicit (δL, δM).

:func:`eigenvalue_sensitivity` dispatches on the type and returns an :class:`EigenvalueSensitivity`
that says which mechanism ran. See
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..errors import InputTypeError, OptionError
from ..solvers.evp import EVPResult, matrix_derivative
from ..solvers.evp import sensitivity as _evp_sensitivity
from ..solvers.parametric import ParametricEVP, TrackedMode
from .modes import ModePair, mode_pair

__all__ = ["EigenvalueSensitivity", "eigenvalue_sensitivity", "parameter_sensitivity"]


@dataclass
class EigenvalueSensitivity:
    """dσ/dp of one eigenvalue, with the mechanism that produced it."""

    sigma: complex
    value: complex
    parameter: str
    method: str
    check: float = float("nan")      # |difference| between two FD steps (FD mechanism only)

    def explain(self) -> str:
        c = "" if np.isnan(self.check) else f"; step-halving check {self.check:.1e}"
        return (f"dσ/d{self.parameter} = {self.value:.12g} at σ = {self.sigma:.12g} "
                f"[{self.method}]{c}. Formula −yᴴ(∂L + σ∂M)x / (yᴴ M x).")


def parameter_sensitivity(factory: Callable[[float], object], value: float, *, parameter: str,
                          N: int, target: complex, infinite_tol: float, rel_h: float
                          ) -> EigenvalueSensitivity:
    """dσ/dp for the eigenvalue nearest ``target`` of ``factory(value)`` (a ModalSystem).

    ∂L/∂p, ∂M/∂p of the tau descriptor by Richardson central differences with relative step
    ``rel_h`` (mandatory); the same derivative with 2·rel_h is reported as ``check``.
    """
    if not rel_h > 0:
        raise OptionError("rel_h must be > 0 (declared finite-difference step)")
    pair = mode_pair(factory(value), N=N, target=target, adjoint="discrete",
                     infinite_tol=infinite_tol)

    def build(p):
        d = factory(p).discretize(N)
        return d.L, d.M

    dL, dM = matrix_derivative(build, value, rel_h)
    dL2, dM2 = matrix_derivative(build, value, 2 * rel_h)
    v1, v2 = pair.shift(dL, dM), pair.shift(dL2, dM2)
    return EigenvalueSensitivity(pair.sigma, v1, parameter,
                                 f"ModalSystem tau descriptor, Richardson FD (rel_h = {rel_h:g}), "
                                 f"N = {N}, pencil residual {pair.residual_right:.1e}",
                                 float(abs(v1 - v2)))


def eigenvalue_sensitivity(source, *args, parameter: str = "p", **kw) -> EigenvalueSensitivity:
    """dσ/dp from any of the three mechanisms (see module doc):

    * ``eigenvalue_sensitivity(res: EVPResult, i, M, dL, dM=None)``
    * ``eigenvalue_sensitivity(pevp: ParametricEVP, mode: TrackedMode, name)``
    * ``eigenvalue_sensitivity(pair: ModePair, dL, dM=None)``
    """
    if isinstance(source, EVPResult):
        i, M, dL = args[:3]
        dM = args[3] if len(args) > 3 else kw.get("dM")
        v = _evp_sensitivity(source, i, M, dL, dM)
        return EigenvalueSensitivity(complex(source.values[i]), v, parameter,
                                     f"EVPResult ({source.method}), user-supplied dL/dp, dM/dp")
    if isinstance(source, ParametricEVP):
        mode, name = args[:2]
        if not isinstance(mode, TrackedMode):
            raise InputTypeError("eigenvalue_sensitivity(ParametricEVP, mode, name): mode must be "
                                 "a TrackedMode (ParametricEVP.track / least_stable)")
        v = source.sensitivity(mode, name)
        return EigenvalueSensitivity(mode.sigma, v, name,
                                     "ParametricEVP: exact derivative of the Laurent pencil")
    if isinstance(source, ModePair):
        dL = args[0]
        dM = args[1] if len(args) > 1 else kw.get("dM")
        return EigenvalueSensitivity(source.sigma, source.shift(dL, dM), parameter,
                                     "ModePair: pencil left/right vectors, user-supplied dL, dM")
    raise InputTypeError(f"eigenvalue_sensitivity: unsupported source {type(source).__name__}",
                         fix="EVPResult (solve_dense(left=True)), ParametricEVP or ModePair")
