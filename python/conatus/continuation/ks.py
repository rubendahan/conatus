"""Continuation of Kuramoto–Sivashinsky equilibria in the domain length L.

Parametrization by L (rescaling x̃ = 2π x / L). The state is the RealFourier coefficient vector
of u on (0, L): u(x) = Σ c_n cos(2π n x / L) + s_n (−sin(2π n x / L)). The *same* vector
describes ũ(x̃) = u(L x̃ / 2π) on the fixed interval (0, 2π) for every L, so continuing the
coefficient vector in L is exactly continuing ũ. L enters only through the derivative
coefficients: ∂x = (2π/L) ∂x̃, hence KS becomes ũ_t = −(2π/L) ũ ũ_x̃ − (2π/L)² ũ_x̃x̃ −
(2π/L)⁴ ũ_x̃x̃x̃x̃. In practice each value of L gets its own :func:`conatus.ecs.ks_solver`
(wavenumbers 2πn/L), whose vector field v_L(X) = M⁻¹(F(X) − L X) is used unchanged.

Unknowns: the antisymmetric subspace R u = u (sin components; translations broken, mean = 0
automatically), where E1, E2, E3 live. G(x, L) = v_L restricted to that subspace; its Jacobian
is formed densely from exact JVPs (the subspace has N/2 − 1 unknowns); G_L is a central finite
difference with the declared step. Stability is counted in the **full** space (mean frozen) with
:func:`conatus.ecs.equilibrium_eigenvalues`, so symmetry-breaking bifurcations are seen too.
The translation mode is exactly marginal there: real parts with |Re λ| ≤ ``marginal_tol``
(declared) are counted as zero, not as unstable.

Nothing in :mod:`conatus.ecs` is modified; this module only composes its public objects.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from ..ecs import Flow, FourierOps, equilibrium_eigenvalues, ks_solver
from ..errors import OptionError
from .core import Branch, ContinuationSettings, PseudoArclength, _count
from .._deprecation import renames
from .._repr import compact_repr

__all__ = ["KSDomainContinuation"]


@dataclass
class _AtL:
    __repr__ = compact_repr   # one line, arrays by shape
    solver: object
    v: Flow
    ops: FourierOps
    mask: np.ndarray            # dofs of the antisymmetric subspace (sin components)


class KSDomainContinuation:
    """Equilibria of KS continued in L within R u = u; every numerical choice is an argument."""

    def __init__(self, *, N: int, dealias: float, dp_fd: float, marginal_tol: float,
                 eig_count: int, cache: int = 6) -> None:
        if not marginal_tol > 0:
            raise OptionError(f"KSDomainContinuation: marginal_tol={marginal_tol!r} must be > 0",
                              fix="e.g. marginal_tol=1e-8 (the translation mode is marginal)")
        # counts are integers, never truncated (N=64.5 used to become 64 without a word)
        for name, value in (("N", N), ("eig_count", eig_count), ("cache", cache)):
            _count(value, name, "KSDomainContinuation")
        self.N, self.dealias, self.dp_fd = int(N), float(dealias), float(dp_fd)
        self.marginal_tol, self.eig_count = float(marginal_tol), int(eig_count)
        self._cache: OrderedDict[float, _AtL] = OrderedDict()
        self._cache_size = int(cache)

    # -- per-L objects (rebuilt when L changes; small LRU cache) -------------------------------
    def at(self, L: float) -> _AtL:
        """Solver, vector field and Fourier operators at domain length L (small LRU cache)."""
        L = float(L)
        if not L > 0:
            # a predictor step can overshoot below L = 0: a corrector failure (step halved),
            # reported by the continuation, not a crash
            raise RuntimeError(f"KS domain length must be > 0 (corrector reached L = {L:.4g})")
        if L in self._cache:
            self._cache.move_to_end(L)
            return self._cache[L]
        # the time step is irrelevant here (only the vector field is used); RK scheme required by ecs
        s = ks_solver(L=L, N=self.N, dealias=self.dealias, scheme="RK443", timestep=0.01)
        ops = FourierOps(s)
        at = _AtL(s, Flow(s), ops, ops.odd_mask(np.zeros(s.system.n)))
        self._cache[L] = at
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return at

    def full(self, x: np.ndarray, L: float) -> np.ndarray:
        """Full state from the antisymmetric-subspace unknowns (other entries zero)."""
        at = self.at(L)
        X = np.zeros(at.solver.system.n)
        X[at.mask] = x
        return X

    def active(self, X: np.ndarray, L: float) -> np.ndarray:
        """Unknowns of the antisymmetric subspace (sin components) of a full state."""
        return np.asarray(X)[self.at(L).mask]

    # -- continuation callbacks ------------------------------------------------------------------
    def residual(self, x: np.ndarray, L: float) -> np.ndarray:
        """G(x, L): the KS vector field restricted to the subspace."""
        return self.active(self.at(L).v(self.full(x, L)), L)

    def jacobian(self, x: np.ndarray, L: float) -> np.ndarray:
        """Dense Jacobian of ``residual`` from exact JVPs (one per unknown)."""
        at, X = self.at(L), self.full(x, L)
        cols = [self.active(at.v.jvp(X, self.full(e, L)), L) for e in np.eye(x.size)]
        return np.column_stack(cols)

    def eigenvalues(self, x: np.ndarray, L: float) -> np.ndarray:
        """Leading eigenvalues of the full-space Jacobian (mean frozen); |Re| ≤ marginal_tol → 0."""
        at = self.at(L)
        sol = SimpleNamespace(X=self.full(x, L), params={}, kind="equilibrium")   # c = 0 (R5)
        # dense path (active size < dense_max = 10 N): arnoldi_tol is not used by it
        ev = np.asarray(equilibrium_eigenvalues(at.solver, sol, n_modes=self.eig_count,
                                                dense_max=10 * self.N, arnoldi_tol=1e-12),
                        dtype=complex)
        re = np.where(np.abs(ev.real) <= self.marginal_tol, 0.0, ev.real)
        return re + 1j * ev.imag

    def energy(self, x: np.ndarray, L: float) -> float:
        """E = (1/L) ∫ u²/2 dx (Cvitanović et al. 2010, eq. 3.6); only sin components here."""
        c = self.at(L).solver.system.scatter(self.full(x, L))[0]
        return float(0.5 * (c[0] ** 2 + 0.5 * np.sum(c[2:] ** 2)))

    def continuation(self, settings: ContinuationSettings, *, eigen: bool) -> PseudoArclength:
        """The :class:`PseudoArclength` driver on these callbacks (stability if ``eigen``)."""
        return PseudoArclength(self.residual, self.jacobian, settings,
                               eigen=self.eigenvalues if eigen else None,
                               marginal_tol=self.marginal_tol if eigen else None,
                               scale=lambda x, L: max(1.0, float(np.max(np.abs(x)))))

    @renames("conatus.continuation.ks:KSDomainContinuation.run")
    def run(self, x0: np.ndarray, L0: float, settings: ContinuationSettings, *, direction: float,
            L_bounds: tuple[float, float], eigen: bool = True) -> Branch:
        """Continue the equilibrium x0 (full state at L0, must satisfy R u = u) in L."""
        at = self.at(L0)
        if np.abs(at.ops.reflect(x0) - x0).max() > 1e-12 * max(1.0, np.abs(x0).max()):
            raise OptionError("starting state is not in the antisymmetric subspace R u = u",
                              fix="project it first (x0 = (x0 + R x0) / 2 with FourierOps.reflect) "
                                  "if it is meant to be antisymmetric")
        return self.continuation(settings, eigen=eigen).run(self.active(x0, L0), L0,
                                                            direction=direction, p_bounds=L_bounds)

    def explain(self) -> str:
        """Discretization, unknowns, Jacobian, G_L and stability settings."""
        return (f"KS equilibria continued in L: RealFourier N={self.N}, dealias {self.dealias:g}; "
                "state = Fourier coefficients (= ũ(x̃), x̃ = 2πx/L, L-independent representation); "
                "unknowns: antisymmetric subspace R u = u (sin components); Jacobian dense from exact "
                f"JVPs; G_L by central FD, dL = {self.dp_fd:g}; stability in the full space (mean "
                f"frozen), {self.eig_count} leading eigenvalues, |Re| ≤ {self.marginal_tol:g} counted "
                "as marginal (translation mode)")
