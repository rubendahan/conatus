"""The existing transient-growth machinery (``conatus.growth``) on the same energy norm.

``conatus.growth`` computes G = σ_max²(F Φ Z R⁻¹) from a time-stepped propagator Φ (IMEX
Runge–Kutta, all columns at once) over the admissible inputs Z. For a *steady* base flow the same
code applies with L₀ = L and B± = 0; with the exact Gram factor F of ``Descriptor`` the result
must equal G(t) = ‖e^{tA}‖²_E of ``StateSpace`` up to the time-stepping error (tested with the
too (the historical ``growth.energy_factor`` uses the N-point Fejér rule, exact to degree N − 1
only, while |u|² has degree 2N − 2: both converge spectrally, they differ at finite N).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from .._deprecation import renames
from ..errors import OptionError
from ..growth.propagator import gain, propagator
from ..linearize.inner_product import chebyshev_gram
from .system import Descriptor
from .._repr import compact_repr

__all__ = ["SteadyOperator", "steady_operator", "propagator_growth", "gram_energy_factor"]


@dataclass
class SteadyOperator:
    """Duck-typed operator for ``growth.propagator``: M X' + L0 X = −B(t) X with B = 0."""
    __repr__ = compact_repr   # one line, arrays by shape

    M: np.ndarray
    L0: np.ndarray
    Bp: np.ndarray
    Bm: np.ndarray
    state_cols: np.ndarray

    @property
    def n(self) -> int:
        return self.M.shape[0]

    def admissible_basis(self) -> np.ndarray:
        """Z whose columns span the admissible states: every algebraic row (no mass: continuity
        and boundary conditions) holds. Built from the null space of those rows of L0 restricted
        to the state columns; they must not involve pressure or taus (checked)."""
        alg = np.nonzero(~np.any(self.M != 0, axis=1))[0]
        other = np.setdiff1d(np.arange(self.n), self.state_cols)
        C = self.L0[np.ix_(alg, self.state_cols)]
        leak = self.L0[np.ix_(alg, other)]
        involved = np.any(leak != 0, axis=1)          # rows touching p/taus are not constraints
        if involved.all():
            raise RuntimeError("no pure state constraint found among the algebraic rows")
        return sla.null_space(C[~involved])

    def explain(self) -> str:
        """The steady operator handed to the growth propagator."""
        return (f"SteadyOperator: M X' + L0 X = 0 (B(t) = 0), n = {self.n}, "
                f"{len(self.state_cols)} state column(s); the propagator of conatus.growth "
                f"applied to a time-independent operator")


def steady_operator(desc: Descriptor) -> SteadyOperator:
    """The steady descriptor as a ``growth`` operator (B = 0)."""
    Z = np.zeros_like(desc.L)
    return SteadyOperator(desc.M, desc.L, Z, Z, desc.cols)


@renames("conatus.nonmodal.transient:propagator_growth")
def propagator_growth(desc: Descriptor, T: float, steps: int, scheme: str) -> float:
    """G(T) from the IMEX propagator of ``conatus.growth`` in the exact energy norm."""
    op = steady_operator(desc)
    Phi = propagator(op, 0.0, T, steps, scheme, "cpu")
    return gain(Phi, desc.energy_factor(), op.admissible_basis()).G


def gram_energy_factor(op: object, theta_weight: float | None = None) -> np.ndarray:
    """Exact-Gram counterpart of ``growth.energy_factor``: ‖F x‖² = ½ Σ w_c ∫|c|² dy exactly."""
    N = op.N
    with np.errstate(divide="ignore"):
        Gz = chebyshev_gram(N, 0.0, op.Ly)
    Fc = np.linalg.cholesky(0.5 * Gz).T
    blocks = []
    for comp in op.state:
        if comp == "b":
            if theta_weight is None or theta_weight <= 0:
                raise OptionError("theta is in the state: give theta_weight > 0 (1 or N^2)")
            blocks.append(np.sqrt(theta_weight) * Fc)
        else:
            blocks.append(Fc)
    return sla.block_diag(*blocks)
