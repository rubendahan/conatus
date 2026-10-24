"""Continuation adapter for an NLBVP in one continuation parameter (Newton system of discretization plans).

The discrete residual is the one Newton uses, F_h(x; p) = A(x; p) x - b(x; p), with A = F'(x) the
exact Fréchet Jacobian of the IR (see solvers/nlbvp.py); the unknown vector x is the packed
coefficient vector of the problem's single group. Problems with several Fourier groups are refused
for now (the bordered system would couple the groups through p: to be done with block elimination).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import scipy.sparse as sp

from .._deprecation import renames
from ..errors import InputTypeError, OptionError, UnsupportedError
from ..solvers.nlbvp import NewtonSolver, unpack_group
from .core import Branch, ContinuationSettings, PseudoArclength, fd_step

__all__ = ["NLBVPContinuation"]


class NLBVPContinuation:
    """Pseudo-arclength continuation of ``solver.problem`` in the parameter ``parameter``
    (``param``: its deprecated former name, v0.1).

    ``settings`` are the explicit continuation settings (no defaults); G_p is computed by the
    declared central finite difference ``settings.dp_fd`` (the Newton system is rebuilt at p ± dp).
    ``eigen(coeffs, p) -> growth rates`` (optional) gives the stability of each branch point, e.g.
    the eigenvalues of the linearized IVP (``linearize`` + ``eigenvalues``): an NLBVP has no time
    derivative, so its own Jacobian cannot provide growth rates (see ``__init__``).
    """

    @renames("conatus.continuation.nlbvp:NLBVPContinuation.__init__")
    def __init__(self, solver: NewtonSolver, parameter: str | None = None,
                 settings: ContinuationSettings | None = None,
                 eigen: Callable[[Mapping[str, np.ndarray], float], np.ndarray] | None = None,
                 marginal_tol: float | None = None) -> None:
        from .._deprecation import given        # param= (v0.1 name) is mapped by @renames
        param = given("NLBVPContinuation", "parameter", parameter)
        if settings is None:
            raise InputTypeError("NLBVPContinuation: settings (a ContinuationSettings) must be given",
                                 fix="settings=ContinuationSettings(ds=..., ds_min=..., "
                                     "ds_max=..., tolerance=..., max_points=...)")
        if settings.dp_fd is None:
            raise OptionError("NLBVP continuation needs a finite-difference G_p: dp_fd=None "
                              "declares an analytic one, which this driver does not have",
                              fix="leave dp_fd at its default 'auto' (declared step rule "
                                  "eps^(1/3) max(1, |p|)), or give a step, e.g. dp_fd=1e-6")
        if isinstance(eigen, bool):
            # eigen=True used to count the eigenvalues of the steady Jacobian A = F'(x) with Re > 0:
            # opposite sign to growth rates (F = LHS - RHS), no mass matrix, tau/BC rows mixed in;
            # a stable branch was reported with every mode unstable (tests/exact).
            raise InputTypeError("eigen must be a callable (coeffs, p) -> growth rates of the branch "
                            "point (e.g. eigenvalues of the linearized IVP); the eigenvalues of the "
                            "steady Newton Jacobian are not growth rates (no dt, sign F = LHS - RHS, "
                            "tau rows)")
        self.solver, self.param, self.settings = solver, param, settings
        self._eigen_user = eigen
        self._template: dict[str, np.ndarray] | None = None
        self._plan0 = None
        self._rows = None                     # row scales (audit invariance): set by run()
        self._group = None
        self._cache = None
        self.core = PseudoArclength(self._G, self._J, settings, scale=self._scale,
                                    eigen=self._eig if eigen is not None else None,
                                    marginal_tol=marginal_tol if eigen is not None else None)

    # --- vector <-> coefficients ---------------------------------------------------------------
    def coeffs_of(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Coefficient arrays (per variable) of the packed vector x."""
        if self._template is None:
            # used to fail with "'NoneType' object has no attribute 'items'"
            raise OptionError("NLBVPContinuation.coeffs_of: no layout yet",
                              fix="call vector_of(start, p0) (or run) first: it fixes the layout")
        out = {n: a.copy() for n, a in self._template.items()}
        unpack_group(self._plan0, self._group, x, out)
        return out

    def vector_of(self, coeffs: Mapping[str, np.ndarray], p: float) -> np.ndarray:
        """Packed vector of the coefficients at parameter p (fixes the layout)."""
        plan, systems = self.solver.discrete_system(coeffs, {self.param: p})
        if len(systems) != 1:
            raise UnsupportedError(f"continuation of a problem with {len(systems)} Fourier "
                                   "groups is not implemented (single group only)",
                                   fix="continue a problem with no Fourier direction (one group)")
        self._plan0, (self._group, _, _, x) = plan, systems[0]
        self._template = {n: np.zeros_like(a) for n, a in coeffs.items()}
        return np.real_if_close(x)

    def _system(self, x: np.ndarray, p: float):
        key = (x.tobytes(), float(p))          # G, J and scale at the same point share one build
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        out = self._build(x, p)
        self._cache = (key, out)
        return out

    def _build(self, x: np.ndarray, p: float):
        # A = F'(x) is singular at a fold: its rank report is skipped on purpose; the bordered
        # matrix [[A, G_p], [t^T]] is what is factorized, and a singular one fails loudly.
        _, systems = self.solver.discrete_system(self.coeffs_of(x), {self.param: p}, check=False)
        (_, A, b, _), = systems
        return A, b

    def _row_scale(self, A) -> np.ndarray:
        """1 / (largest coefficient of each row of A at the start point), fixed for the run (audit
        invariance): G, J and the residual are those of D (A x - b) with this
        constant diagonal D, the same branch and the same Newton iterates in exact arithmetic,
        but a relative residual that no longer depends on the scale of an equation (Bratu's
        equation written times 1e-6 was measured against its O(1) boundary rows: no fold)."""
        if self._rows is None:
            r = np.asarray(abs(A).max(axis=1).todense()).ravel() if sp.issparse(A) \
                else np.abs(A).max(axis=1)
            self._rows = 1.0 / np.where(r > 0, r, 1.0)
        return self._rows

    def _G(self, x: np.ndarray, p: float) -> np.ndarray:
        A, b = self._system(x, p)
        return self._row_scale(A) * (A @ x - b)

    def _J(self, x: np.ndarray, p: float):
        A = self._system(x, p)[0]
        d = self._row_scale(A)
        return sp.diags(d) @ A if sp.issparse(A) else d[:, None] * A

    def _scale(self, x: np.ndarray, p: float) -> float:
        """Residual scale max(|A x|, |b|, |p G_p|) (∞-norms; G_p by the declared central
        difference). At the fold of a uniform state A x = 0 and b = 0 as assembled vectors (the
        terms cancel), and the former scale max(|A x|, |b|) vanished: the corrector could not
        converge there. |p G_p| is the size of the
        parameter's own terms, which do not cancel; a norm ‖A‖ |x| would instead be dominated by
        the high derivative rows of a spectral matrix and loosen every other point."""
        A, b = self._system(x, p)
        d = self._row_scale(A)
        base = max(float(np.max(np.abs(d * (A @ x)))), float(np.max(np.abs(d * b))))
        h = fd_step(self.settings.dp_fd, p)
        gp = (self._G(x, p + h) - self._G(x, p - h)) / (2.0 * h)
        self._system(x, p)                    # leave the cache on (x, p) for the corrector
        return max(base, abs(p) * float(np.max(np.abs(gp))), 1e-300)

    def _eig(self, x: np.ndarray, p: float) -> np.ndarray:
        """Growth rates of the branch point from the user callback (coefficients, parameter)."""
        return np.asarray(self._eigen_user(self.coeffs_of(x), p))

    # --- driver ---------------------------------------------------------------------------------
    def run(self, start: Mapping[str, np.ndarray], p0: float, direction: float = +1.0,
            p_bounds: tuple[float, float] | None = None) -> Branch:
        """Continue from the converged coefficients ``start`` at p0 (see PseudoArclength.run)."""
        x0 = self.vector_of(start, p0)
        self._rows = None                     # row scales of this run: taken at (x0, p0)
        self._row_scale(self._system(x0, p0)[0])
        return self.core.run(x0, p0, direction, p_bounds)

    def explain(self) -> str:
        """Residual, G_p, norm, stability and rank-report settings."""
        return "\n".join([
            f"NLBVP continuation of '{self.solver.problem.name}' in parameter '{self.param}'",
            "  residual: Newton system F_h(x; p) = A(x; p) x - b(x; p) (IR Fréchet Jacobian)",
            "  G_p: central finite difference (declared step), norm: unweighted coefficients",
            "  relative residual: max|G| / max(|A x|, |b|, |p G_p|) (∞-norms), each row divided by"
            " its largest coefficient at the start point (no equation's scale enters)",
            "  stability: " + ("user growth-rate callback (coeffs, p)" if self._eigen_user
                               is not None else "none"),
            "  rank report of A skipped along the branch (A is singular at folds); the start point"
            " is checked, the bordered matrix is factorized and a singular one is refused",
            self.settings.explain()])
