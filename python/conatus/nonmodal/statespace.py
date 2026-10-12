"""From the descriptor pencil M x' + L x = 0 to the generator on the admissible states.

The tau discretization of an incompressible problem is a DAE: continuity, boundary rows and taus
have no mass, and the pencil (−L, M) has infinite eigenvalues. The dynamics lives on the right
deflating subspace of the *finite* eigenvalues. We take it from an ordered generalized Schur form
(QZ, unitary, no eigenvectors: robust even when the eigenvectors are nearly parallel, which is the
non-normal case of interest):

    (−L) Z = Q S,   M Z = Q T,   finite eigenvalues first  =>  x = Z₁ q,   T₁₁ q' = S₁₁ q.

The energy of x is ‖F C Z₁ q‖² (C selects the state columns, F the Cholesky factor of the exact
Gram); with the thin QR F C Z₁ = Q_e R_e, the coordinates q̃ = R_e q are energy-orthonormal and

    Ã = R_e T₁₁⁻¹ S₁₁ R_e⁻¹,     ‖(z − A)⁻¹‖_E = ‖(z − Ã)⁻¹‖₂,   ‖e^{tA}‖_E = ‖e^{tÃ}‖₂.

The finite/infinite split uses |β| > infinite_tol·|α| on the QZ pairs (mandatory tolerance);
the gap between the largest finite and the smallest infinite |λ| is reported, and a split with a
gap under 1e4 is refused (it would depend on the tolerance). See
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from .pencil import reduce_pencil
from .system import Descriptor
from .._repr import compact_repr

__all__ = ["StateSpace", "state_space"]


@dataclass
class StateSpace:
    """Generator Ã in energy-orthonormal coordinates, with the maps back to state coefficients."""
    __repr__ = compact_repr   # one line, arrays by shape

    A: np.ndarray                  # Ã (k × k)
    to_state: np.ndarray           # state coefficients = to_state @ q̃   (= C Z₁ R_e⁻¹)
    descriptor: Descriptor
    n_infinite: int
    gap: float                     # min |λ_inf| / max |λ_fin|
    cond_Re: float                 # condition number of R_e (energy vs unitary coordinates)
    infinite_tol: float

    @property
    def k(self) -> int:
        return self.A.shape[0]

    def eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvals(self.A)

    def schur(self):
        """Complex Schur form Ã = U T Uᴴ (cached): the matrix functions of Ã work on T."""
        if not hasattr(self, "_schur"):
            T, U = sla.schur(self.A, output="complex")
            self._schur = (T, U)
        return self._schur

    def explain(self) -> str:
        d = self.descriptor
        return (f"StateSpace of '{d.system.name}' (N = {d.N}): {self.k} finite eigenvalues kept, "
                f"{self.n_infinite} infinite removed (|beta| <= {self.infinite_tol:g} |alpha|, gap "
                f"{self.gap:.2e}); energy-orthonormal coordinates (cond R_e = {self.cond_Re:.2e}); "
                "norms of resolvent and exponential are 2-norms of the generator here.")


def state_space(desc: Descriptor, *, infinite_tol: float) -> StateSpace:
    """Reduce M x' + L x = 0 to q̃' = Ã q̃ in energy-orthonormal coordinates (see module doc)."""
    At, to_state, n_inf, gap, cond = reduce_pencil(desc.M, desc.L, desc.cols, desc.energy_factor(),
                                                   infinite_tol=infinite_tol)
    return StateSpace(At, to_state, desc, n_inf, gap, cond, infinite_tol)
