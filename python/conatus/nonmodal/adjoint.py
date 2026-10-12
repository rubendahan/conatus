"""Continuous ("optimize, then discretize") versus discrete ("discretize, then optimize") adjoint.

* **Discrete adjoint** of the generator in the declared energy product: A† = W⁻¹ Aᴴ W, i.e.
  Ãᴴ in energy-orthonormal coordinates (``StateSpace``). It is the exact adjoint of the
  *discrete* operator: ⟨A†a, x⟩_E = ⟨a, A x⟩_E to rounding, at every N.
* **Continuous adjoint**: the adjoint PDE derived by formal integration by parts
  (``ModalSystem.formal_adjoint``, declared and checked boundary conditions), then discretized
  by the same tau method. It is the discretization of the exact adjoint operator, *not* the
  adjoint of the discretization: at finite N the two differ (the tau rows, the lifts and the
  truncation of the NCC products do not commute with the transposition).

``compare_adjoints`` measures the difference on invariant quantities (adjoint eigenvalue,
eigenfunction in the energy norm, eigenvalue condition number κ = ‖x‖‖a‖/|⟨a, x⟩|) so that
tests can show both converge to the same limit as N grows. See
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .statespace import StateSpace, state_space
from .system import ModalSystem

__all__ = ["AdjointComparison", "compare_adjoints", "discrete_adjoint_mode",
           "continuous_adjoint_mode"]


def _eig_near(A: np.ndarray, target: complex):
    w, V = np.linalg.eig(A)
    i = int(np.argmin(np.abs(w - target)))
    return complex(w[i]), V[:, i]


def discrete_adjoint_mode(ss: StateSpace, sigma: complex):
    """(σ̄, a): eigenpair of Ãᴴ near σ̄; ``a`` in state coefficients, unit energy."""
    lam, q = _eig_near(ss.A.conj().T, np.conj(sigma))
    return lam, ss.to_state @ (q / np.linalg.norm(q))


def continuous_adjoint_mode(adj: ModalSystem, N: int, sigma: complex, *, infinite_tol: float):
    """(σ̄_c, a_c) from the discretized continuous adjoint, unit energy (its own exact Gram)."""
    ssa = state_space(adj.discretize(N), infinite_tol=infinite_tol)
    lam, q = _eig_near(ssa.A, np.conj(sigma))
    return lam, ssa.to_state @ (q / np.linalg.norm(q)), ssa


@dataclass
class AdjointComparison:
    N: int
    sigma: complex                 # direct eigenvalue (nearest the target)
    residual: float                # ‖Ã x − σ x‖ / (‖Ã‖ ‖x‖)
    sigma_adj_discrete: complex    # = conj(σ) to rounding
    sigma_adj_continuous: complex  # → conj(σ) as N grows
    kappa_discrete: float
    kappa_continuous: float
    mode_distance: float           # ‖a_c − a_d‖_E after unit energy and ⟨a, x⟩ > 0

    def explain(self) -> str:
        return (f"N = {self.N}: σ = {self.sigma:.12g} (residual {self.residual:.1e}); adjoint "
                f"eigenvalue discrete {self.sigma_adj_discrete:.12g}, continuous "
                f"{self.sigma_adj_continuous:.12g}; κ discrete {self.kappa_discrete:.10g}, "
                f"continuous {self.kappa_continuous:.10g}; "
                f"‖a_c − a_d‖_E = {self.mode_distance:.2e}")


def compare_adjoints(system: ModalSystem, adjoint_bcs, N: int, target: complex, *,
                     infinite_tol: float) -> AdjointComparison:
    """Discrete and continuous adjoint of the eigenmode of ``system`` nearest ``target``."""
    desc = system.discretize(N)
    ss = state_space(desc, infinite_tol=infinite_tol)
    sigma, xt = _eig_near(ss.A, target)
    res = float(np.linalg.norm(ss.A @ xt - sigma * xt)
                / (np.linalg.norm(ss.A, 2) * np.linalg.norm(xt)))
    x = ss.to_state @ (xt / np.linalg.norm(xt))
    W = desc.W
    adj = system.formal_adjoint(adjoint_bcs)
    lam_d, a_d = discrete_adjoint_mode(ss, sigma)
    lam_c, a_c, ssa = continuous_adjoint_mode(adj, N, sigma, infinite_tol=infinite_tol)

    def normed(a):                 # unit energy, phase such that ⟨a, x⟩_E > 0
        a = a / np.sqrt((a.conj() @ W @ a).real)
        ip = a.conj() @ W @ x
        return a * (ip / abs(ip)), abs(ip)

    a_d, ip_d = normed(a_d)
    a_c, ip_c = normed(a_c)
    xn = np.sqrt((x.conj() @ W @ x).real)
    d = a_c - a_d
    return AdjointComparison(N, sigma, res, lam_d, lam_c, float(xn / ip_d), float(xn / ip_c),
                             float(np.sqrt((d.conj() @ W @ d).real)))


# ------------------------------------------------------------------ resolvent: R† continuous vs Rᴴ




