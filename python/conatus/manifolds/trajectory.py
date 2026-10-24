"""Trajectories with a *declared* dense output, for event detection (Poincaré sections, slice
phases, connection end points). Notes:

Two integrators, one interface (``integrate(x0, until) -> Trajectory``):

* :class:`ODEIntegrator` — ``dx/dt = G(x)`` of a small :class:`~conatus.bifurcation.DynamicalSystem`
  (M = I) with scipy's embedded Runge–Kutta pairs; ``rtol``, ``atol`` and the method are required.
  Dense output: the method's own continuous extension (DOP853: order 7; RK45: order 4), declared.
* :class:`SolverIntegrator` — the **exact discrete map** of an IMEX Runge–Kutta
  :class:`~conatus.timestep.IVPSolver` at a fixed, declared step (the map that the ECS solvers
  use, so a Poincaré point of a periodic orbit found by ``conatus.ecs`` lies on the same discrete
  trajectory). Dense output: cubic Hermite from the states and the vector field v = M⁻¹(F − LX) at
  both ends of each step (local error O(dt⁴)), declared; :meth:`Trajectory.restep` replaces it by
  the exact partial RK step X_n → Φ_θdt(X_n), and the difference is reported as the
  interpolation defect.

No step size is chosen for the user and no tolerance is tightened silently.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..errors import OptionError, UnsupportedError
from .._deprecation import renames
from .._repr import compact_repr

__all__ = ["Trajectory", "ODEIntegrator", "SolverIntegrator"]


@dataclass
class Trajectory:
    """Samples ``t`` (m,), states ``X`` (m, n) and the dense output between consecutive samples.

    ``dense_kind`` names the interpolant; ``_dense(i, t)`` evaluates it on segment i
    (t in [t_i, t_{i+1}]); ``_restep(i, t)`` (discrete maps only) is the exact partial step."""
    __repr__ = compact_repr   # one line, arrays by shape

    t: np.ndarray
    X: np.ndarray
    dense_kind: str
    _dense: Callable[[int, float], np.ndarray] = field(repr=False)
    _restep: Callable[[int, float], np.ndarray] | None = field(default=None, repr=False)
    notes: list[str] = field(default_factory=list)

    def segment(self, t: float) -> int:
        """Index i of the segment [t_i, t_{i+1}] holding t (refuses t outside the trajectory)."""
        if not self.t[0] <= t <= self.t[-1]:
            raise OptionError(f"t = {t:.6g} outside the trajectory [{self.t[0]:.6g}, {self.t[-1]:.6g}]")
        return int(min(max(np.searchsorted(self.t, t, side="right") - 1, 0), len(self.t) - 2))

    def __call__(self, t: float) -> np.ndarray:
        """Dense output at time t (declared interpolant)."""
        return self._dense(self.segment(t), float(t))

    def restep(self, t: float) -> np.ndarray:
        """Exact state of the discrete map from the previous sample by a partial step (discrete
        maps only; an ODE trajectory has no discrete map and refuses)."""
        if self._restep is None:
            raise UnsupportedError("restep: only for trajectories of a discrete map (SolverIntegrator)")
        return self._restep(self.segment(t), float(t))

    def explain(self) -> str:
        lines = [f"Trajectory: {len(self.t)} samples on [{self.t[0]:.6g}, {self.t[-1]:.6g}], "
                 f"n = {self.X.shape[1]}, dense output: {self.dense_kind}"]
        return "\n".join(lines + [f"  note: {n}" for n in self.notes])


class ODEIntegrator:
    """dx/dt = G(x; par) of a DynamicalSystem with M = I, by ``scipy.integrate.solve_ivp``.

    ``method`` ('DOP853', 'RK45', 'Radau', ...), ``rtol`` and ``atol`` are required (declared);
    the samples are the integrator's own accepted steps, the dense output is its continuous
    extension (order stated in ``explain()``)."""

    _ORDER = {"DOP853": 7, "RK45": 4, "RK23": 3, "Radau": 3, "BDF": 1, "LSODA": 1}

    def __init__(self, system, *, method: str, rtol: float, atol: float, par=None) -> None:
        if not getattr(system, "identity_mass", True):
            raise UnsupportedError("ODEIntegrator: M != I (DAE rows); use SolverIntegrator")
        if method not in self._ORDER:
            raise OptionError(f"ODEIntegrator: unknown method {method!r}",
                              fix=f"one of {sorted(self._ORDER)}")
        self.system, self.method, self.rtol, self.atol = system, method, float(rtol), float(atol)
        self.par = None if par is None else dict(par)
        self.n = system.n

    def with_par(self, par) -> ODEIntegrator:
        """Same integrator at other parameter values."""
        return ODEIntegrator(self.system, method=self.method, rtol=self.rtol, atol=self.atol, par=par)

    def vector_field(self, x: np.ndarray) -> np.ndarray:
        return self.system.G(x, self.par)

    def _jac(self) -> dict:
        """Analytic Jacobian for the implicit methods only (explicit ones ignore it)."""
        if self.method in ("Radau", "BDF", "LSODA"):
            return {"jac": lambda _t, x: self.system.J(x, self.par)}
        return {}

    def flow(self, x0: np.ndarray, t: float) -> np.ndarray:
        """End point only (no dense output stored)."""
        import scipy.integrate as si
        if t == 0:
            return np.array(x0, float)
        sol = si.solve_ivp(lambda _t, x: self.system.G(x, self.par), (0.0, float(t)),
                           np.asarray(x0, float), method=self.method, rtol=self.rtol, atol=self.atol,
                           **self._jac())
        if sol.status != 0:
            raise UnsupportedError(f"ODEIntegrator: solve_ivp failed ({sol.message})")
        return sol.y[:, -1]

    @renames("conatus.manifolds.trajectory:ODEIntegrator.integrate")
    def integrate(self, x0: np.ndarray, until: float, *, events=None) -> Trajectory:
        import scipy.integrate as si
        sol = si.solve_ivp(lambda _t, x: self.system.G(x, self.par), (0.0, float(until)),
                           np.asarray(x0, float), method=self.method, rtol=self.rtol, atol=self.atol,
                           dense_output=True, events=events, **self._jac())
        if sol.status < 0:
            raise UnsupportedError(f"ODEIntegrator: solve_ivp failed ({sol.message})")
        dense = sol.sol
        tr = Trajectory(np.asarray(sol.t), np.asarray(sol.y).T,
                        f"{self.method} continuous extension (order {self._ORDER[self.method]})",
                        lambda i, t: dense(t))
        tr.notes.append(f"{self.method}, rtol = {self.rtol:g}, atol = {self.atol:g}, "
                        f"{sol.nfev} evaluations")
        return tr

    def explain(self) -> str:
        return (f"ODEIntegrator: scipy {self.method} (rtol {self.rtol:g}, atol {self.atol:g}), "
                f"dense output order {self._ORDER[self.method]}; system: {self.system.origin}")


class SolverIntegrator:
    """Exact discrete map of an IMEX Runge–Kutta IVPSolver at the declared step ``dt``.

    ``embed(x) -> X`` and ``restrict(X) -> x`` map reduced coordinates (an invariant subspace, the
    frozen mean...) to the solver vector and back; identity by default. The vector field for the
    Hermite interpolant is v = M⁻¹(F − L X) (``conatus.ecs.Flow``)."""

    def __init__(self, solver, *, dt: float, embed=None, restrict=None) -> None:
        from ..ecs.solutions import Flow
        if solver.order or getattr(solver, "family", "rk") != "rk":       # SBDF and CN multistep
            raise UnsupportedError("SolverIntegrator: multistep scheme (no one-step map); use RK443, "
                                   "RK222, RKSMR or RKGFY")
        if not dt > 0:
            raise OptionError("SolverIntegrator: dt must be > 0")
        self.solver, self.dt = solver, float(dt)
        self.v = Flow(solver)
        self.embed = embed if embed is not None else (lambda x: np.asarray(x, float))
        self.restrict = restrict if restrict is not None else (lambda X: np.asarray(X, float))
        self.n = int(np.asarray(self.restrict(np.zeros(solver.system.n))).size)

    def step(self, X: np.ndarray, h: float) -> np.ndarray:
        """One RK step of size h of the solver's scheme (solver vector)."""
        return self.solver.rk_stages(X, 0.0, float(h))[-1]

    def vector_field(self, x: np.ndarray) -> np.ndarray:
        return self.restrict(self.v(self.embed(x)))

    def flow(self, x0: np.ndarray, t: float) -> np.ndarray:
        """Φ_t(x0) with nsteps = round(t/dt) steps of dt; t must be a multiple of dt (1e-9)."""
        n = int(round(t / self.dt))
        if abs(n * self.dt - t) > 1e-9 * max(1.0, abs(t)):
            raise OptionError(f"SolverIntegrator.flow: t = {t:.12g} is not a multiple of dt = {self.dt:g}")
        X = self.embed(x0)
        for _ in range(n):
            X = self.step(X, self.dt)
        return self.restrict(X)

    @renames("conatus.manifolds.trajectory:SolverIntegrator.integrate")
    def integrate(self, x0: np.ndarray, until: float, *, every: int = 1) -> Trajectory:
        """States at every step (``every`` > 1 keeps one sample in ``every`` for storage; the
        dense output then spans ``every`` steps with the same Hermite rule, declared)."""
        from ..errors import check_count
        check_count(every, "every", "SolverIntegrator.integrate")
        n = int(round(until / self.dt))
        if n < 1:
            raise OptionError("SolverIntegrator.integrate: until < dt")
        if abs(n * self.dt - until) > 1e-9 * max(1.0, abs(until)):     # as flow(): never rounded
            raise OptionError(f"SolverIntegrator.integrate: until = {until:.12g} is not a multiple "
                              f"of dt = {self.dt:g}", fix=f"until = {n * self.dt:.12g} ({n} steps)")
        X = self.embed(x0)
        Xs, Vs = [self.restrict(X)], [self.restrict(self.v(X))]
        full = [X]
        for i in range(1, n + 1):
            X = self.step(X, self.dt)
            if i % every == 0 or i == n:
                Xs.append(self.restrict(X))
                Vs.append(self.restrict(self.v(X)))
                full.append(X)
        t = np.array([0.0] + [i * self.dt for i in range(1, n + 1) if i % every == 0 or i == n])
        Xa, Va = np.array(Xs), np.array(Vs)

        def dense(i, tt):              # cubic Hermite on [t_i, t_{i+1}]
            h = t[i + 1] - t[i]
            s = (tt - t[i]) / h
            h00, h10 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s
            h01, h11 = -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
            return h00 * Xa[i] + h10 * h * Va[i] + h01 * Xa[i + 1] + h11 * h * Va[i + 1]

        def restep(i, tt):             # exact partial step(s) of the discrete map from sample i
            Xi, rem = full[i], tt - t[i]
            while rem > self.dt * (1 + 1e-12):
                Xi, rem = self.step(Xi, self.dt), rem - self.dt
            return self.restrict(self.step(Xi, rem) if rem > 0 else Xi)

        tr = Trajectory(t, Xa, "cubic Hermite (states and vector field at the samples; O(h^4) local)",
                        dense, restep)
        tr.notes.append(f"exact discrete map: {self.solver.scheme}, dt = {self.dt:g}, {n} steps, "
                        f"sample every {every} step(s)")
        return tr

    def explain(self) -> str:
        return (f"SolverIntegrator: exact discrete map of {self.solver.scheme} at dt = {self.dt:g} "
                "(fixed); dense output: cubic Hermite, exact partial step on request (restep)")
