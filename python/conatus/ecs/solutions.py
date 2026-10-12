"""Invariant solutions of an autonomous IVP: equilibria, travelling waves, (relative / pre-)
periodic orbits, with Floquet analysis. Theory and conventions:

Unknown vector z = (active state dofs, extra unknowns). The conserved mean (cos_0) is frozen, and
the antisymmetric subspace R u = u is available as an explicit mask. The flow map is the exact
discrete map of :class:`conatus.adjoint.FlowMap` (RK scheme, dt = T / nsteps), so the Newton
residual is the residual of the discrete problem; ∂Φ/∂T is taken as the vector field at Φ_T(u)
(continuous-time approximation, exact up to the time-discretization error — enough for Newton,
the reported residual is always the exact discrete one).
"""
from __future__ import annotations

import warnings
import weakref
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse.linalg as spla

from ..adjoint import FlowMap
from ..errors import InputTypeError, NewtonNotConverged, OptionError, SymmetryError, UnsupportedError
from .fourier_ops import FourierOps
from .newton import NewtonKrylovHookstep, NewtonKrylovResult
from .._deprecation import renames
from ..tolerances import REQUIRED, require

__all__ = ["Flow", "VectorField", "Equilibrium", "TravellingWave", "RelativePeriodicOrbit",
           "PreperiodicOrbit", "ECSSolution", "find_equilibrium", "find_travelling_wave",
           "find_relative_periodic_orbit", "floquet_multipliers", "equilibrium_eigenvalues",
           "symmetry_breaking_check", "SCOPE"]

_DOC = "docs/site/guide/dynamics/invariant_solutions.md"

#: Scope of find_equilibrium / find_travelling_wave / find_relative_periodic_orbit,
#: stated in every refusal and in the guide.
SCOPE = ("one periodic direction with a real Fourier basis and no other axis (1D periodic "
         "problems: Kuramoto-Sivashinsky, 1D Swift-Hohenberg, ...); find_equilibrium searches "
         "the odd subspace u(-x) = -u(x) (R u = u with R u(x) = -u(-x))")
_ALTERNATIVES = ("for flows with walls or several directions: conatus.ecs.steady."
                 "AffineSteadyProblem (steady states of large IVPs, symmetry subspaces, Couette), "
                 "conatus.ecs.pipe.PipeTravellingWave (pipe travelling waves), "
                 "conatus.ecs.kolmogorov (2D Kolmogorov flow); these finders are not yet "
                 "general")


class _MissingSpeedError(OptionError, KeyError):
    """A travelling solution without its frame speed ``params['c']``: an OptionError that keeps the
    built-in base of the historical raise (KeyError), so ``except KeyError`` still catches it
    (errors.py, compatibility rule)."""


def _solver_of(obj, where: str, *, N=None, need_scheme: bool = False):
    """The IVPSolver behind ``obj``: an expert solver, a native ``Simulation`` (its ``.solver``)
    or, when the time scheme plays no role (``need_scheme=False``), a ``cn.Model`` with ``N=``
    (compiled with a declared placeholder scheme: v(u) = M⁻¹(F − L u) does not depend on it).
    Refuses, with the scope and the alternatives, a problem outside the 1D periodic scope."""
    from ..model.model import Model
    from ..model.run import Simulation
    note = None
    if isinstance(obj, Model):
        if need_scheme:
            raise InputTypeError(f"{where}: a cn.Model has no time scheme",
                                 why="a periodic orbit is a fixed point of the discrete flow map "
                                     "Phi_T: the scheme and nsteps define it",
                                 fix="sim = m.simulate(N, scheme='RK443', dt=...); then pass sim",
                                 doc=_DOC)
        if N is None:
            raise InputTypeError(f"{where}: a cn.Model needs its resolution",
                                 fix=f"{where}(m, X0, newton, N=64), or pass "
                                     "m.simulate(N, scheme=..., dt=...)", doc=_DOC)
        obj = obj.simulate(N, scheme="RK443", dt=1.0)
        note = ("model compiled at N = " + str(N) + " (time scheme irrelevant for v(u) = "
                "M^-1 (F - L u); RK443 placeholder)")
    elif N is not None:
        raise OptionError(f"{where}(..., N=...): N is only for a cn.Model", doc=_DOC)
    if isinstance(obj, Simulation):
        obj = obj.solver
    if not (hasattr(obj, "plan") and hasattr(obj, "system")):
        raise InputTypeError(f"{where}: expected a Simulation (m.simulate(...)), an expert "
                             f"IVPSolver or a cn.Model, got {type(obj).__name__}", doc=_DOC)
    from ..ir.bases import RealFourier
    bases = [b for b in obj.plan.domain.bases if b is not None]
    if len(bases) != 1 or not isinstance(bases[0], RealFourier):
        raise UnsupportedError(
            f"{where}: this problem has bases {[type(b).__name__ for b in bases]}",
            why=f"the finder works on {SCOPE}",
            fix=_ALTERNATIVES, doc=_DOC)
    return obj, note


def _seed_state(solver, x0, where: str) -> tuple[np.ndarray, str | None]:
    """The state vector of ``solver`` for the initial guess ``x0`` (grid values from ``sim.grid("u")`` hit a bare IndexError inside the finder). Accepted:
    the state vector itself (``sim.solver.X``, ``solver.system.n`` entries), a ``Simulation`` or
    an ``IVPSolver`` (its current state), or the grid values of the single unknown (the shape of
    ``sim.grid(name)``), converted to coefficients with the loss of the round trip declared in the
    returned note. Anything else is refused, naming the accepted objects."""
    from ..model.run import Simulation, _from_grid
    n = solver.system.n
    if isinstance(x0, Simulation):
        x0 = x0.solver
    if hasattr(x0, "system") and hasattr(x0, "X"):
        if x0.system.n != n:
            raise InputTypeError(f"{where}: the simulation given as initial guess has a state of "
                                 f"{x0.system.n} entries, this solver {n}", fix="a simulation "
                                 "of the same model at the same N", doc=_DOC)
        X = x0.X
        X = np.array(X.get() if hasattr(X, "get") else X, dtype=float)
        return X, "initial guess: the current state of the given simulation"
    arr = np.asarray(x0.get() if hasattr(x0, "get") and not isinstance(x0, np.ndarray) else x0)
    names = [v.name for v in solver.system.variables]
    gshape = solver.grid(names[0]).shape if len(names) == 1 else None
    if arr.shape == (n,) and not (gshape == (n,) and _from_grid(x0)):
        return np.array(arr, dtype=float), None
    if gshape is not None and arr.shape == gshape:
        if not np.all(np.isfinite(arr)):
            raise InputTypeError(f"{where}: the initial guess has non-finite values", doc=_DOC)
        v = solver.system.variables[0]
        X = solver.system.gather_state([solver._grid_to_coeffs(v, arr)])
        X = np.array(X.get() if hasattr(X, "get") else X, dtype=float)
        back = np.asarray(solver.plan.to_grid(names[0], solver.system.scatter(X)[0]))
        loss = float(np.linalg.norm(back - arr) / max(np.linalg.norm(arr), 1e-300))
        return X, (f"initial guess: grid values of {names[0]} (shape {arr.shape}) converted to "
                   f"the state vector ({n} entries); relative change of the round trip "
                   f"grid -> coefficients -> grid {loss:.1e} (the modes the state does not hold)")
    want = f"grid values of shape {gshape} (sim.grid({names[0]!r})), " if gshape else ""
    raise InputTypeError(
        f"{where}: initial guess of shape {arr.shape} is neither the state vector ({n} "
        f"entries) nor the grid values of the unknown{'s' if len(names) > 1 else ''} "
        + ", ".join(names),
        fix=f"pass {want}the state vector sim.solver.X ({n} entries), or the simulation itself "
            "(its current state)", doc=_DOC)


class _CachedFlowMap(FlowMap):
    """FlowMap whose trajectory is cached for the last X0 (GMRES calls jvp many times at one X0)."""

    def trajectory(self, X0):
        key = np.asarray(X0).tobytes()
        if getattr(self, "_traj_key", None) != key:
            self._traj = super().trajectory(X0)
            self._traj_key = key
        return self._traj


class Flow:
    """v(X) = M⁻¹(F(X) − L X) and its Jacobian, for an RK :class:`IVPSolver` on CPU."""

    def __init__(self, solver) -> None:
        if solver.order or getattr(solver, "family", "rk") != "rk":       # SBDF and CN multistep
            raise OptionError("ECS: the solver uses a multistep scheme; the ECS solvers need a "
                              "Runge–Kutta one (dt = T/nsteps changes with T)",
                              fix="simulate with scheme='RK443' (or another RK scheme)", doc=_DOC)
        self.solver = solver
        self.system = solver.system
        self._lu_M = spla.splu(self.system.M.tocsc())       # refuses singular M (tau rows)
        self._lin = FlowMap(solver, 1)                        # IR linearization of F (exact)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self._lu_M.solve(self.solver.F(X, 0.0) - self.system.L @ X)

    def jvp(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        return self._lu_M.solve(self._lin._JF(X, V) - self.system.L @ V)

    def explain(self) -> str:
        """What v(X) and its Jacobian are made of."""
        return (f"Flow: v(X) = M⁻¹ (F(X, 0) − L X) of the {self.solver.scheme} solver "
                f"(n = {self.system.n}); M factorized once (SuperLU; a singular M with tau rows "
                f"is refused); Jacobian-vector products from the IR linearization of F (exact)")


class VectorField(Flow):
    """Deprecated alias of :class:`Flow` (name clash with ``conatus.ir.VectorField``)."""

    def __init__(self, solver) -> None:
        import warnings
        warnings.warn("conatus.ecs.VectorField is deprecated: use conatus.ecs.Flow",
                      DeprecationWarning, stacklevel=2)
        super().__init__(solver)


TRIVIAL_TOL = 1e-8      # ‖u‖ < TRIVIAL_TOL · scale ⇒ trivial state (declared in every explain())
#: A travelling wave whose speed term is below the Newton accuracy, |c| ‖∂x U‖ < TW_SPEED_TOL ·
#: tol · scale (tol, scale: those of the Newton solve), is an equilibrium: c = 0 within the
#: accuracy of the solve (from c0 = 0 an equilibrium came back as a
#: "travelling_wave" with c = -3.7e-17). Relabelled kind "equilibrium" (class Equilibrium),
#: ``params['c']`` kept, the measured value in the notes.
TW_SPEED_TOL = 1.0


def _guard_trivial(sol: ECSSolution, scale: float) -> ECSSolution:
    """A Newton run that lands on u ≈ 0 is not a new solution: retract it explicitly."""
    sol.notes.append(f"trivial-state threshold: ‖u‖ < {TRIVIAL_TOL:g} × {scale:.3g} ⇒ retracted")
    if np.linalg.norm(sol.X) < TRIVIAL_TOL * scale:
        sol.status = "retracted"
        sol.notes.append("converged to the trivial state u = 0 (not a new solution)")
    return sol


#: Steady-state guard of the orbit finders. An orbit search can
#: converge onto an equilibrium or a travelling wave (they solve S τ_ℓ Φ_T(u) = u for a continuum
#: of (T, ℓ)); on the discrete map such a state is a discrete travelling wave, every step a pure
#: shift, so along its period it keeps its shape to the accuracy of the solve. The guard measures
#: the shape change of the converged orbit over its period, A = max_j min_s ‖τ_s u_j − u_0‖ / ‖u_0‖
#: (j over STEADY_SAMPLES time steps of the period, s over the shifts), and retracts the orbit when A < STEADY_TOL x the
#: accuracy of the solve (max of its residual and the Newton relative residual). It is relative to
#: the orbit amplitude, not to ‖u‖: a small orbit near a Hopf point, u = U + ε φ(t), has A ~ ε
#: whatever ‖U‖. The former test, comoving velocity min_c ‖v(u) + c ∂x u‖/‖u‖ < 1e-2, retracted
#: such orbits (velocity ε ω / ‖U‖) and depended on dt (a travelling wave on the discrete map: 9.8e-4
#: at dt = 0.05, RK443); it is still reported in the notes. Measured
#: (tests/ecs/test_orbit_steady_guard.py, tests/ecs/test_ux4c_steady_guard.py): E2 and TW1 reached by
#: orbit searches A <= 1e-10, the PPO 10.25 A = O(1), an orbit of amplitude 1e-5 about a state of
#: norm 1 A ~ 1e-5.
STEADY_TOL = 1e3
#: Number of states of the period at which the shape change is measured (all of them up to it).
STEADY_SAMPLES = 64
#: Size up to which the stability problems of invariant solutions are solved densely (a budget
#: has a declared default): QZ on the full Jacobian (every eigenvalue,
#: marginal modes included); above it ARPACK (equilibrium_eigenvalues, floquet_multipliers, with
#: a required arnoldi_tol) or a refusal (symmetry_breaking_check, a dense-only check).
DENSE_MAX = 2000


def _arpack_tol(arnoldi_tol, where: str) -> float:
    """The ARPACK accuracy (required whichever path runs, so that a script does not
    change meaning when its size crosses dense_max). the v0.1 surface is frozen, so
    omitting it keeps working through v0.1 with the former default 1e-12 and a DeprecationWarning
    that names the value used; it becomes a hard requirement in v0.2."""
    if arnoldi_tol is REQUIRED:
        warnings.warn(
            f"{where}(): arnoldi_tol not given; using the v0.1 default arnoldi_tol={V01_ARNOLDI_TOL:g} "
            "(the relative accuracy of the Ritz values above dense_max). It will be required in "
            "v0.2: pass it explicitly, e.g. arnoldi_tol=1e-12.", DeprecationWarning, stacklevel=3)
        return V01_ARNOLDI_TOL
    return require(arnoldi_tol, "arnoldi_tol", where=f"{where}()", suggest="arnoldi_tol=1e-12",
                   why="above dense_max ARPACK stops at this relative accuracy of the Ritz values: "
                       "it decides which digits of the eigenvalues are right")


#: Former default of arnoldi_tol, used with a DeprecationWarning when it is omitted.
V01_ARNOLDI_TOL = 1e-12


def _min_shift_distance(ops, U: np.ndarray, W: np.ndarray) -> float:
    """min over shifts s of ‖τ_s U − W‖ (1D periodic): a dense scan of s, then golden-section
    refinement around the best sample (the distance is a smooth trigonometric sum in s)."""
    L = ops.L
    n = 8 * max(4, ops.basis.size)

    def d(sh: float) -> float:
        return float(np.linalg.norm(ops.translate(U, sh) - W))
    grid = np.arange(n) * (L / n)
    vals = [d(sh) for sh in grid]
    i = int(np.argmin(vals))
    a, b = grid[i] - L / n, grid[i] + L / n
    g = (np.sqrt(5.0) - 1) / 2
    c, e = b - g * (b - a), a + g * (b - a)
    fc, fe = d(c), d(e)
    for _ in range(80):
        if fc < fe:
            b, e, fe = e, c, fc
            c = b - g * (b - a)
            fc = d(c)
        else:
            a, c, fc = c, e, fe
            e = a + g * (b - a)
            fe = d(e)
        if b - a < 1e-13 * L:
            break
    return min(fc, fe, vals[i])


def _guard_steady(sol: ECSSolution, orb: _Orbit) -> ECSSolution:
    """An equilibrium or travelling wave solves S τ_ℓ Φ_T(u) = u for a continuum of (T, ℓ): an
    orbit search that lands there has not found an orbit. Judged by the shape change of the orbit
    over its period relative to the accuracy of the solve (STEADY_TOL); retracted explicitly."""
    X = sol.X
    n0 = float(np.linalg.norm(X))
    if not n0 > 0:
        return sol
    v, dx = orb.v(X), orb.ops.ddx(X)
    c = -float(v @ dx) / float(dx @ dx) if float(dx @ dx) > 0 else 0.0
    rel = float(np.linalg.norm(v + c * dx) / n0)
    T = sol.params.get("T")
    if T is None:
        sol.notes.append("steady-state check: no period T in the solution, not judged")
        return sol
    states = orb.flow(float(T)).trajectory(X)[1:]
    if len(states) > STEADY_SAMPLES:          # evenly spaced along the period (cost, declared)
        states = [states[i] for i in np.linspace(0, len(states) - 1, STEADY_SAMPLES).astype(int)]
    amp = max((_min_shift_distance(orb.ops, np.asarray(u), X) for u in states), default=0.0) / n0
    newton_rel = getattr(sol.newton, "relative_residual", 0.0) or 0.0
    floor = max(float(sol.residual), float(newton_rel), 1e-15)
    sol.notes.append(f"steady-state check: shape change over the period max_j min_s "
                     f"‖τ_s u_j − u_0‖/‖u_0‖ = {amp:.3e} vs STEADY_TOL x accuracy of the solve = "
                     f"{STEADY_TOL:g} x {floor:.1e} = {STEADY_TOL * floor:.1e} (below ⇒ retracted); "
                     f"comoving velocity min_c ‖v(u) + c ∂x u‖/‖u‖ = {rel:.3e} (c = {c:.6g}), "
                     "reported only")
    if amp < STEADY_TOL * floor:
        sol.status = "retracted"
        sol.notes.append("converged to a steady state (equilibrium or travelling wave), not an orbit")
    return sol


@dataclass
class ECSSolution:
    """A converged invariant solution with its evidence."""

    kind: str                 # equilibrium / travelling_wave / relative_periodic_orbit / preperiodic_orbit
    X: np.ndarray
    params: dict              # c (TW), T and ell (RPO/PPO), nsteps, dt
    residual: float           # relative residual of the discrete problem at the solution
    newton: NewtonKrylovResult
    energy: float             # E = (1/L) ∫ u²/2 dx (Cvitanović et al. 2010, eq. 3.6)
    status: str = "provisional"
    notes: list[str] = field(default_factory=list)
    stability: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        # one line (the dataclass repr printed both state
        # arrays in full, about 40 lines in a log)
        p = "".join(f", {k}={v:.10g}" for k, v in self.params.items() if isinstance(v, float))
        return (f"{type(self).__name__}({self.kind}{p}, residual {self.residual:.1e}, "
                f"{self.newton.iterations} Newton iterations, energy {self.energy:.6g}, status "
                f"{self.status}; state of {np.asarray(self.X).size} entries in .X; .explain())")

    def explain(self) -> str:
        p = ", ".join(f"{k}={v:.10g}" if isinstance(v, float) else f"{k}={v}" for k, v in self.params.items())
        lines = [f"{self.kind}: {p}", f"  relative residual {self.residual:.3e} "
                 f"({self.newton.iterations} Newton iterations), energy E = {self.energy:.6f}, "
                 f"status {self.status}"]
        lines += [f"  note: {n}" for n in self.notes]
        for k, v in self.stability.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


class Equilibrium(ECSSolution):
    """v(u) = 0."""


class TravellingWave(ECSSolution):
    """u(x, t) = U(x − c t) (relative equilibrium); ``params['c']``."""


class RelativePeriodicOrbit(ECSSolution):
    """τ_ℓ Φ_T(u) = u; ``params['T']``, ``params['ell']``."""


class PreperiodicOrbit(ECSSolution):
    """R τ_ℓ Φ_T(u) = u (periodic with period 2T); ``params['T']``, ``params['ell']``."""


def _energy(ops: FourierOps, X: np.ndarray) -> float:
    c = ops.system.scatter(X)[0]
    # ∫ cos² = ∫ sin² = L/2 for n ≥ 1, mean contributes c0²·L
    return float(0.5 * (c[0] ** 2 + 0.5 * np.sum(c[2:] ** 2)))


class _Space:
    """Active dofs: all but the frozen mean, optionally restricted to R u = u (odd subspace)."""

    def __init__(self, ops: FourierOps, n: int, odd: bool) -> None:
        mask = np.ones(n, bool)
        mask[ops.mean_index] = False
        if odd:
            mask &= ops.odd_mask(np.zeros(n))
        self.mask, self.odd, self.n = mask, odd, n

    def full(self, x: np.ndarray) -> np.ndarray:
        X = np.zeros(self.n)
        X[self.mask] = x
        return X

    def active(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X)[self.mask]

    def describe(self) -> str:
        return ("antisymmetric subspace R u = u (cos components frozen at 0), mean frozen"
                if self.odd else "full space, mean frozen (conserved)")


# --------------------------------------------------------------------------- equilibria / TW
@renames("conatus.ecs.solutions:find_equilibrium")
def find_equilibrium(solver, x0: np.ndarray, newton: NewtonKrylovHookstep, *, odd: bool = True,
                     log=None, N=None) -> ECSSolution:
    """v(u) = 0. In the antisymmetric subspace (default) translations are broken: no phase
    condition needed. In the full space use :func:`find_travelling_wave` (c ≈ 0).

    ``solver``: a native ``Simulation``, an expert ``IVPSolver`` or a ``cn.Model`` with ``N=``.
    Scope (``SCOPE``): one periodic real-Fourier direction; other problems are refused with the
    alternatives."""
    solver, note = _solver_of(solver, "find_equilibrium", N=N)
    x0, seed = _seed_state(solver, x0, "find_equilibrium")
    if not odd:
        raise OptionError("find_equilibrium(odd=False): full-space equilibria have a translation "
                          "family", fix="find_travelling_wave (phase condition, c unknown, c ≈ 0 "
                          "for an equilibrium) instead", doc=_DOC)
    ops, v = FourierOps(solver), Flow(solver)
    sp = _Space(ops, solver.system.n, odd=True)
    if np.abs(ops.reflect(x0) - x0).max() > 1e-12 * max(1.0, np.abs(x0).max()):
        raise OptionError("find_equilibrium: the initial guess is not in the antisymmetric "
                          "subspace R u = u", fix="project it: x0 = (x0 + R x0) / 2 with "
                          "R = FourierOps(solver).reflect", doc=_DOC)
    scale = max(1.0, float(np.linalg.norm(x0)))
    res = newton.solve(lambda z: sp.active(v(sp.full(z))),
                       lambda z, d: sp.active(v.jvp(sp.full(z), sp.full(d))),
                       sp.active(x0), scale=scale, log=log)
    X = sp.full(res.z)
    rel = float(np.linalg.norm(v(X)) / max(np.linalg.norm(X), 1e-300))  # meaningless if X ≈ 0
    sol = Equilibrium("equilibrium", X, {"space": sp.describe()}, rel, res, _energy(ops, X))
    sol.notes.append("residual = ‖v(u)‖/‖u‖ with v = M⁻¹(F − L u), evaluated in the full space")
    sol.notes += [t for t in (note, seed) if t]
    return _guard_trivial(sol, scale)


@renames("conatus.ecs.solutions:find_travelling_wave")
def find_travelling_wave(solver, x0: np.ndarray, c0: float, newton: NewtonKrylovHookstep, *,
                         log=None, N=None) -> ECSSolution:
    """u(x, t) = U(x − c t): v(U) + c ∂x U = 0, unknowns (U, c), phase ⟨∂x U*, δU⟩ = 0.
    ``solver``: a ``Simulation``, an ``IVPSolver`` or a ``cn.Model`` with ``N=``; scope as
    :func:`find_equilibrium` (``SCOPE``). A solution whose speed is zero within the Newton
    accuracy (``TW_SPEED_TOL``) is returned as an :class:`Equilibrium` (kind "equilibrium",
    ``params['c']`` kept, the measured value in the notes)."""
    solver, note = _solver_of(solver, "find_travelling_wave", N=N)
    x0, seed = _seed_state(solver, x0, "find_travelling_wave")
    ops, v = FourierOps(solver), Flow(solver)
    sp = _Space(ops, solver.system.n, odd=False)
    ref = {"t": ops.ddx(x0)}

    def R(z):
        X, c = sp.full(z[:-1]), z[-1]
        return np.concatenate([sp.active(v(X) + c * ops.ddx(X)), [0.0]])

    def J(z, d):
        X, c = sp.full(z[:-1]), z[-1]
        dX, dc = sp.full(d[:-1]), d[-1]
        body = v.jvp(X, dX) + c * ops.ddx(dX) + dc * ops.ddx(X)
        return np.concatenate([sp.active(body), [ref["t"] @ dX]])

    scale = max(1.0, float(np.linalg.norm(x0)))
    try:
        res = newton.solve(R, J, np.concatenate([sp.active(x0), [c0]]), scale=scale, log=log)
    except NewtonNotConverged as err:
        _note_repeated_stagnation(err, solver, c0)
        raise
    X, c = sp.full(res.z[:-1]), float(res.z[-1])
    rel = float(np.linalg.norm(v(X) + c * ops.ddx(X)) / np.linalg.norm(X))
    rel = rel if np.linalg.norm(X) > 0 else float("inf")
    speed = abs(c) * float(np.linalg.norm(ops.ddx(X)))
    still = speed < TW_SPEED_TOL * float(newton.tol) * scale
    cls, kind = (Equilibrium, "equilibrium") if still else (TravellingWave, "travelling_wave")
    sol = cls(kind, X, {"c": c, "space": sp.describe()}, rel, res, _energy(ops, X))
    sol.notes.append(f"speed check: |c| ‖∂x U‖ = {speed:.3e} vs TW_SPEED_TOL·tol·scale = "
                     f"{TW_SPEED_TOL * float(newton.tol) * scale:.3e}")
    if still:
        sol.notes.append(f"c = {c:.3g} is zero within the Newton accuracy: this is an equilibrium "
                         "(relabelled from travelling_wave), not a travelling wave; start from a "
                         "c0 away from 0 or a seed of a wave to find one")
    sol.notes += [t for t in (note, seed) if t]
    return _guard_trivial(sol, scale)


#: Stalled travelling-wave searches of each solver in this session: (‖R‖/scale, final c, c0),
#: oldest first (five far seeds stopped on the same ‖R‖ = 4.013e-03, one
#: stagnation point found five times, each refusal saying only "not converged"). Weak keys: a
#: solver that is gone takes its history with it.
_TW_STAGNATIONS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _note_repeated_stagnation(err: NewtonNotConverged, solver: object, c0: float) -> None:
    """Add to the refusal of a failed find_travelling_wave that an earlier search of the same
    solver stalled at the same point: the same final residual and the same final wave speed, each
    to a relative 1e-3 (INTERNAL SAME_STAGNATION). Only a stalled search counts (``err.stalled``:
    no decrease of ‖R‖ at the end); one cut by its budget while ‖R‖ still fell is no stagnation
    point (two 2-iteration searches from the same seed were told they
    had found "a local minimum of ‖R‖"). A note on the error only; nothing is retried or changed."""
    rel = getattr(err, "relative_residual", None)
    last = getattr(err, "last_iterate", None)
    if (not getattr(err, "stalled", False) or rel is None or not np.isfinite(rel)
            or last is None or not np.size(last)):
        return
    c_end = float(np.real(np.ravel(last)[-1]))
    try:
        seen = _TW_STAGNATIONS.setdefault(solver, [])
    except TypeError:                                  # a solver without weak references
        return
    same = [(r, c, s0) for r, c, s0 in seen if abs(rel - r) <= 1e-3 * max(r, rel)
            and abs(c_end - c) <= 1e-3 * max(1.0, abs(c), abs(c_end))]
    seen.append((float(rel), c_end, float(c0)))
    if not same:
        return
    s0 = same[0][2]
    note = (f"stalled at the same point as {len(same)} earlier search(es) of this solver "
            f"(‖R‖/scale = {rel:.4e}, c = {c_end:.6g}; the first from c0 = {s0:g}): these "
            "seeds lead to one point where Newton stalls; unless more iterations converge from "
            "it, it is a stagnation point of the residual (a local minimum of ‖R‖), not a "
            "travelling wave, and another far seed may reach it again")
    err.why = note if not err.why else f"{err.why}; {note}"
    seed_fix = ("seed from a state near the wave (a run that approaches it, a continuation from a "
                "known wave, the wave at a nearby parameter) rather than another far seed")
    err.fix = seed_fix if not err.fix else f"{err.fix}; or {seed_fix}"


# --------------------------------------------------------------------------- periodic orbits
class _Orbit:
    """G(u, T, ℓ) = S τ_ℓ Φ_T(u) − u with S = I (RPO) or R (pre-periodic orbit)."""

    def __init__(self, solver, nsteps: int, reflect: bool) -> None:
        self.solver, self.nsteps, self.reflect = solver, int(nsteps), bool(reflect)
        self.ops, self.v = FourierOps(solver), Flow(solver)
        self._maps: dict[float, _CachedFlowMap] = {}

    def flow(self, T: float) -> _CachedFlowMap:
        if T not in self._maps:
            if len(self._maps) > 4:
                self._maps.clear()
            self.solver.set_dt(T / self.nsteps)
            self._maps[T] = _CachedFlowMap(self.solver, self.nsteps)
        return self._maps[T]

    def S(self, X):
        return self.ops.reflect(X) if self.reflect else X

    def G(self, X, T, ell):
        return self.S(self.ops.translate(self.flow(T)(X), ell)) - X

    def dG(self, X, T, ell, dX, dT, dell):
        fm = self.flow(T)
        PhX = fm.trajectory(X)[-1]
        body = fm.jvp(X, dX) + dT * self.v(PhX)
        body = self.ops.translate(body, ell) + dell * self.ops.ddx(self.ops.translate(PhX, ell))
        return self.S(body) - dX


@renames("conatus.ecs.solutions:find_relative_periodic_orbit")
def find_relative_periodic_orbit(solver, x0: np.ndarray, T0: float, ell0: float,
                                 newton: NewtonKrylovHookstep, *, steps: int, reflect: bool = False,
                                 log=None) -> ECSSolution:
    """S τ_ℓ Φ_T(u) = u with unknowns (u, T, ℓ) and phase conditions ⟨v(u*), δu⟩ = 0 (time),
    ⟨∂x u*, δu⟩ = 0 (translation). ``reflect=True`` gives pre-periodic orbits R τ_ℓ Φ_T u = u
    (the ℓ freedom there is the choice of reflection point, pinned by the translation phase).
    ``solver``: a ``Simulation`` or an ``IVPSolver`` (its scheme defines the discrete map); scope
    as :func:`find_equilibrium` (``SCOPE``)."""
    solver, _ = _solver_of(solver, "find_relative_periodic_orbit", need_scheme=True)
    x0, seed = _seed_state(solver, x0, "find_relative_periodic_orbit")
    orb = _Orbit(solver, steps, reflect)
    sp = _Space(orb.ops, solver.system.n, odd=False)
    ref = {"v": orb.v(x0), "t": orb.ops.ddx(x0)}

    def R(z):
        X, T, ell = sp.full(z[:-2]), z[-2], z[-1]
        return np.concatenate([sp.active(orb.G(X, T, ell)), [0.0, 0.0]])

    def J(z, d):
        X, T, ell = sp.full(z[:-2]), z[-2], z[-1]
        dX = sp.full(d[:-2])
        body = orb.dG(X, T, ell, dX, d[-2], d[-1])
        return np.concatenate([sp.active(body), [ref["v"] @ dX, ref["t"] @ dX]])

    scale = max(1.0, float(np.linalg.norm(x0)))
    res = newton.solve(R, J, np.concatenate([sp.active(x0), [T0, ell0]]), scale=scale, log=log)
    X, T, ell = sp.full(res.z[:-2]), float(res.z[-2]), float(res.z[-1])
    ell = float((ell + orb.ops.L / 2) % orb.ops.L - orb.ops.L / 2)       # shift in [−L/2, L/2)
    rel = float(np.linalg.norm(orb.G(X, T, ell)) / np.linalg.norm(X))
    kind = "preperiodic_orbit" if reflect else "relative_periodic_orbit"
    cls = PreperiodicOrbit if reflect else RelativePeriodicOrbit
    sol = cls(kind, X, {"T": T, "ell": ell, "nsteps": steps, "dt": T / steps,
                                "scheme": solver.scheme}, rel, res, _energy(orb.ops, X))
    sol.notes.append("residual = ‖S τ_ℓ Φ_T(u) − u‖/‖u‖ of the discrete map (exact FlowMap)")
    if seed:
        sol.notes.append(seed)
    return _guard_steady(_guard_trivial(sol, scale), orb)


# --------------------------------------------------------------------------- stability
@renames("conatus.ecs.solutions:equilibrium_eigenvalues")
def equilibrium_eigenvalues(solver, sol: ECSSolution, n_modes: int, *, dense_max: int = DENSE_MAX,
                            arnoldi_tol: float = REQUIRED) -> np.ndarray:
    """``n_modes`` leading eigenvalues (largest real part) of the Jacobian of v (+ c ∂x for a TW),
    full space without the mean: dense QZ if the active size ≤ ``dense_max`` (declared default
    DENSE_MAX), ARPACK otherwise at the accuracy ``arnoldi_tol`` (required: the
    budget is declared, the tolerance required; ``k=`` is the former name of
    ``n_modes``, k being a wavenumber everywhere else).
    Marginal eigenvalues (translation) are part of the output. A travelling wave must carry
    ``params['c']`` (it used to fall back to the fixed frame, c = 0); an equilibrium
    without it has c = 0 by definition."""
    if "c" in sol.params:
        c = sol.params["c"]
    elif sol.kind == "equilibrium":
        c = 0.0
    else:
        raise _MissingSpeedError(f"equilibrium_eigenvalues: {sol.kind} without params['c'] (the "
                                 "co-moving frame speed); refusing to analyse it in the fixed frame",
                                 fix="pass the solution returned by find_travelling_wave (it "
                                 "carries params['c'])", doc=_DOC)
    tol = _arpack_tol(arnoldi_tol, "equilibrium_eigenvalues")
    # a Simulation (or a cn.Model refused with its fix) like the finders
    solver, _ = _solver_of(solver, "equilibrium_eigenvalues")
    ops, v = FourierOps(solver), Flow(solver)
    sp = _Space(ops, solver.system.n, odd=False)
    m = int(sp.mask.sum())

    def matvec(d):
        return sp.active(v.jvp(sol.X, sp.full(d)) + c * ops.ddx(sp.full(d)))
    if m <= dense_max:                     # small systems: dense (ARPACK can miss marginal modes)
        A = np.column_stack([matvec(e) for e in np.eye(m)])
        ev = np.linalg.eigvals(A)
    else:
        ev = spla.eigs(spla.LinearOperator((m, m), dtype=float, matvec=matvec), k=n_modes,
                       which="LR", return_eigenvectors=False, tol=tol)
    return ev[np.argsort(-ev.real)][:n_modes]


@renames("conatus.ecs.solutions:floquet_multipliers")
def floquet_multipliers(solver, sol: ECSSolution, n_modes: int, *, dense_max: int = DENSE_MAX,
                        marginal_overlap_min: float = 0.99, arnoldi_tol: float = REQUIRED) -> dict:
    """``n_modes`` leading Floquet multipliers of S τ_ℓ J_T (monodromy of the discrete map, full
    space without the mean): dense if the active size ≤ ``dense_max`` (declared default), ARPACK
    otherwise with its required accuracy ``arnoldi_tol`` (``k=``: former name of ``n_modes``). The
    marginal time and translation directions are identified by the overlap of the eigenvectors
    with v(u) and ∂x u (overlap > ``marginal_overlap_min``, declared and returned), and counted."""
    tol = _arpack_tol(arnoldi_tol, "floquet_multipliers")
    solver, _ = _solver_of(solver, "floquet_multipliers", need_scheme=True)   # as the finders
    T, ell = sol.params["T"], sol.params["ell"]
    orb = _Orbit(solver, sol.params["nsteps"], sol.kind == "preperiodic_orbit")
    sp = _Space(orb.ops, solver.system.n, odd=False)
    fm = orb.flow(T)
    m = int(sp.mask.sum())
    def matvec(d):
        return sp.active(orb.S(orb.ops.translate(fm.jvp(sol.X, sp.full(d)), ell)))
    if m <= dense_max:
        lam, vec = np.linalg.eig(np.column_stack([matvec(e) for e in np.eye(m)]))
    else:
        lam, vec = spla.eigs(spla.LinearOperator((m, m), dtype=float, matvec=matvec), k=n_modes,
                             which="LM", tol=tol)
    order = np.argsort(-np.abs(lam))
    lam, vec = lam[order][:n_modes], vec[:, order][:, :n_modes]
    vt = sp.active(orb.v(sol.X))
    vx = sp.active(orb.ops.ddx(sol.X))
    marg = []
    for i in range(len(lam)):
        w = vec[:, i] / np.linalg.norm(vec[:, i])
        ot = abs(np.vdot(w, vt)) / np.linalg.norm(vt)
        ox = abs(np.vdot(w, vx)) / np.linalg.norm(vx)
        if max(ot, ox) > marginal_overlap_min:
            marg.append((i, "time" if ot >= ox else "translation", complex(lam[i])))
    return {"multipliers": lam, "exponents": np.log(lam.astype(complex)) / T, "marginal": marg,
            "marginal_overlap_min": float(marginal_overlap_min)}


def symmetry_breaking_check(solver, sol: ECSSolution, *, dense_max: int = DENSE_MAX,
                            unstable_tol: float,
                            coupling_tol: float, symmetry_tol: float = 1e-10) -> dict:
    """Stability of a reflection-symmetric equilibrium (R u = u) in **each isotypic component**
    of the reflection R: the Jacobian commutes with R, so its spectrum
    splits into the symmetric block (R v = v, what a subspace computation sees) and the
    antisymmetric block (R v = −v, invisible there). Both blocks are computed (dense, size
    ≤ ``dense_max``, explicit). The cross blocks (sym → anti and anti → sym) are **measured**; the
    check is refused if their norm relative to the diagonal blocks exceeds ``coupling_tol``.
    An eigenvalue counts as unstable if Re λ > ``unstable_tol`` (explicit, reported). The
    antisymmetric block holds the marginal translation mode ∂x u (λ ≈ 0): it is **not** identified
    and removed; it stays out of the count only because its Re λ is below ``unstable_tol`` (so
    ``unstable_tol`` must exceed its numerical |Re λ|). The solution must satisfy
    max|R u − u| <= ``symmetry_tol`` · max|u| (declared, returned)."""
    solver, _ = _solver_of(solver, "symmetry_breaking_check")   # a Simulation too
    ops, v = FourierOps(solver), Flow(solver)
    asym = float(np.abs(ops.reflect(sol.X) - sol.X).max() / np.abs(sol.X).max())
    if asym > symmetry_tol:
        raise SymmetryError(f"symmetry_breaking_check: the solution is not reflection-symmetric "
                            f"(max|R u − u| / max|u| = {asym:.1e} > symmetry_tol = {symmetry_tol:g})",
                            fix="pass an equilibrium found in the symmetric subspace "
                            "(find_equilibrium(odd=True), the default), or a looser symmetry_tol if the "
                            "asymmetry is rounding", doc=_DOC)
    n = solver.system.n
    base = np.ones(n, bool)
    base[ops.mean_index] = False
    sym = base & ops.odd_mask(np.zeros(n))          # R v = v : sin components
    anti = base & ~sym                               # R v = −v: cos components
    m = int(base.sum())
    if m > dense_max:
        raise OptionError(f"symmetry_breaking_check: size {m} > dense_max = {dense_max} (the "
                          "isotypic blocks are computed dense)",
                          fix=f"pass dense_max >= {m} if the memory allows (m² doubles), or a "
                          "coarser solver", doc=_DOC)
    idx = np.flatnonzero(base)
    Jfull = np.empty((n, m))
    for j, i in enumerate(idx):
        e = np.zeros(n)
        e[i] = 1.0
        Jfull[:, j] = v.jvp(sol.X, e)
    col = {k: np.isin(idx, np.flatnonzero(msk)) for k, msk in (("symmetric", sym), ("antisymmetric", anti))}
    blk = lambda r, c: Jfull[np.flatnonzero(r)][:, col[c]]          # noqa: E731
    diag = max(np.linalg.norm(blk(sym, "symmetric")), np.linalg.norm(blk(anti, "antisymmetric")))
    coupling = max(np.linalg.norm(blk(anti, "symmetric")), np.linalg.norm(blk(sym, "antisymmetric"))) / diag
    if coupling > coupling_tol:
        raise SymmetryError(f"symmetry_breaking_check: the Jacobian couples the isotypic blocks "
                            f"(relative cross-block norm {coupling:.2e} > coupling_tol = "
                            f"{coupling_tol:g})",
                            why="the reflection is not a symmetry of the equations at this solution",
                            fix="check that the forcing and the solution share the reflection; "
                            "coupling_tol only absorbs rounding", doc=_DOC)
    blocks = {"symmetric": np.linalg.eigvals(blk(sym, "symmetric")),
              "antisymmetric": np.linalg.eigvals(blk(anti, "antisymmetric"))}
    blocks = {k: b[np.argsort(-b.real)] for k, b in blocks.items()}
    lead = {k: b[0] for k, b in blocks.items()}
    anti_unstable = [complex(x) for x in blocks["antisymmetric"] if x.real > unstable_tol]
    verdict = (f"symmetry-breaking instability (Re λ > {unstable_tol:g})" if anti_unstable else
               f"no symmetry-breaking instability (antisymmetric block: Re λ ≤ {unstable_tol:g} "
               "up to the marginal translation mode)")
    out = {"blocks": blocks, "leading": lead, "unstable_antisymmetric": anti_unstable,
           "verdict": verdict, "coupling": float(coupling), "unstable_tol": float(unstable_tol),
           "coupling_tol": float(coupling_tol), "symmetry_tol": float(symmetry_tol)}
    sol.stability["symmetry_breaking_check"] = f"{verdict}; cross-block coupling {coupling:.1e}"
    return out
