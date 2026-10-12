"""Step-size information, kept separate by design (spec §18: CFL ≠ error control).

- ``CFL``: advective *stability* indicator from a velocity field. It never estimates an error.
- ``EmbeddedError``: *temporal error* estimate from a companion scheme of lower order
  (RK443/RK222, RK222/RK111, RKSMR/RK111, RKGFY/RK111, SBDFq/SBDF(q-1) on the same history;
  the CN multistep family has no companion here and is refused explicitly).
- ``TauWatch``: records the size of the tau variables along a run and warns when they grow by
  more than a declared factor (in a 4th-order equation the taus amplify round-off by ~1e7
  in a few steps while the physical fields are unaffected; long runs must report it).

Neither changes dt unless the user asks (``EmbeddedError(adapt=True, tol=..., safety=...)``, Runge–Kutta
only; SBDF coefficients are constant-step).
"""
from __future__ import annotations

import numpy as np

from ..reference import transforms as T
from ..errors import OptionError
from .integrator import IVPSolver, require_family, rk_tableau, sbdf_coefficients

_DOC = "docs/site/guide/timesteppers.md"
# adaptive step limiter (declared defaults: conatus.tolerances.DECLARED, shown by explain()):
# the step changes by a factor in [DT_FACTOR_MIN, DT_FACTOR_MAX] per step, whatever the estimate
DT_FACTOR_MIN = 0.2
DT_FACTOR_MAX = 2.0

__all__ = ["CFL", "EmbeddedError", "TauWatch"]

# main scheme -> (companion of lower order, its order): the companion recomputes the step from
# the same X^n (not an embedded pair sharing stages); RKSMR / RKGFY are second order (IMEX).
_COMPANION = {"RK443": ("RK222", 2), "RK222": ("RK111", 1), "RKSMR": ("RK111", 1),
              "RKGFY": ("RK111", 1)}


class CFL:
    """dt_cfl = safety / max_x sum_i |u_i(x)| / dx_i(x) on the collocation grid."""

    def __init__(self, solver: IVPSolver, velocity: str, safety: float) -> None:
        self.solver, self.velocity, self.safety = solver, velocity, float(safety)
        plan = solver.plan
        self.coords = plan.domain.coordsys.coords
        self.spacing = []
        for i, c in enumerate(self.coords):
            b = plan.domain.basis(c)
            g = T.grid(b, dealias=False)
            if b.periodic:
                d = np.full(g.size, b.length / b.size)
            else:
                # node spacing of every non-periodic basis (only
                # Chebyshev used it; Legendre/Jacobi nodes got the uniform L/N, a step ~3x too
                # large at N = 16, and Hermite/Laguerre L = inf, their direction ignored)
                d = np.gradient(g)
            shape = [1] * len(self.coords)
            shape[i] = g.size
            self.spacing.append(d.reshape(shape))

    def compute(self) -> float:
        """The CFL step of the current state (inf for a fluid at rest)."""
        u = self.solver.grid(self.velocity)
        rate = 0
        for i in range(len(self.coords)):
            rate = rate + np.abs(u[i]) / self.spacing[i]
        m = float(np.max(rate))
        return np.inf if m == 0 else self.safety / m

    def _repr_html_(self) -> str:       # Jupyter display (conatus.notebook)
        from ..notebook import explain_html
        return explain_html(self)

    def explain(self) -> str:
        """What the indicator is (and is not)."""
        return (f"CFL(velocity={self.velocity}, safety={self.safety}): advective stability "
                "indicator on the collocation grid; not an error estimate (EmbeddedError is one)")


class EmbeddedError:
    """Temporal error of one step, from a lower-order companion on the same data."""

    def __init__(self, solver: IVPSolver, adapt: bool = False, tol: float | None = None,
                 safety: float | None = None, dt_min: float = 0.0, dt_max: float = np.inf) -> None:
        self.solver = solver
        s = solver.scheme
        require_family(solver, ("sbdf", "rk"), "EmbeddedError",
                       fix="no lower-order companion is defined for the CN multistep family; use "
                           "an SBDF2-4 or IMEX Runge–Kutta scheme (" + ", ".join(_COMPANION)
                           + "), or compare two runs at dt and dt/2")
        if solver.order:
            if solver.order < 2:
                raise OptionError("EmbeddedError: SBDF1 has no lower-order companion",
                                  fix="SBDF2-4 (companion SBDF(q-1)) or an IMEX Runge–Kutta "
                                      "scheme (" + ", ".join(_COMPANION) + ")", doc=_DOC)
            self.companion, self.p_low = f"SBDF{solver.order - 1}", solver.order - 1
            if adapt:
                raise OptionError("EmbeddedError(adapt=True) with a constant-step SBDF scheme",
                                  why="the SBDF coefficients are constant-step: dt cannot change",
                                  fix="adapt=False (estimate only), or an IMEX Runge–Kutta scheme",
                                  doc=_DOC)
        else:
            if s not in _COMPANION:
                raise OptionError(f"EmbeddedError: no embedded companion for {s}",
                                  fix="one of " + ", ".join(_COMPANION) + " or SBDF2-4", doc=_DOC)
            self.companion, self.p_low = _COMPANION[s]
        if adapt and tol is None:
            raise OptionError("EmbeddedError(adapt=True): tol must be given explicitly (no default)",
                              why="tol is the per-step error the time step is adapted to: it sets "
                                  "the accuracy of the whole run",
                              fix="write tol=1e-6 (or the per-step relative error you accept)",
                              doc=_DOC)
        if adapt and safety is None:
            # the safety factor pilots the time step; it was 0.9 by default
            raise OptionError("EmbeddedError(adapt=True): safety must be given explicitly (no "
                              "default since v0.1)",
                              why="the new step is safety x dt x (tol/err)^(1/(p+1)): the safety "
                                  "factor decides how many steps are rejected or taken, a "
                                  "numerical choice that changes the run",
                              fix="write safety=0.9 (the value used before v0.1; 0.8 rejects fewer "
                                  "steps)", doc=_DOC)
        if safety is not None and not 0 < safety <= 1:
            raise OptionError(f"EmbeddedError: safety must be in (0, 1], got {safety!r}", doc=_DOC)
        self.adapt, self.tol, self.safety = adapt, tol, safety
        self.dt_min, self.dt_max = dt_min, dt_max
        self.last = np.nan

    def step(self) -> float:
        """Advance one step with the main scheme; return ||X - X_low|| / ||X|| (max norm)."""
        sv = self.solver
        if sv.order:
            sv.step()
            if len(sv._MXh) < sv.order:          # still in startup: no estimate yet
                self.last = np.nan
                return self.last
            q = sv.order - 1
            a, c = sbdf_coefficients(q)
            rhs = sv.xp.zeros(sv.system.n, dtype=sv.system.dtype)
            for j in range(1, q + 1):
                rhs += c[j] * sv._Fh[j - 1] - (a[j] / sv.dt) * sv._MXh[j - 1]
            # the boundary data of the step at t_{n+1}, as in the step itself (
            # the companion kept the extrapolated rows, so with time-dependent
            # boundary data the estimate measured that extrapolation, not the scheme)
            rhs = sv.impose_bc(rhs, sv.t, 1.0)
            X_low = sv._solve_state(a[0] / sv.dt, 1.0, rhs)   # same projection as the step
        else:
            X0, t0, dt = sv.X.copy(), sv.t, sv.dt
            sv.step()
            X_low = sv._rk_step(X0, t0, dt, rk_tableau(self.companion))
        scale = max(np.max(np.abs(sv.X)), 1e-300)
        self.last = float(np.max(np.abs(sv.X - X_low)) / scale)
        fe = getattr(sv, "_finite_every", 0)
        # a NaN estimate is local to a rank: it may trigger the check only of a solver whose
        # check is not collective (one rank alone would wait in the allreduce)
        local = not np.isfinite(self.last) and not getattr(sv, "_finite_collective", False)
        if fe and (sv.iteration % fe == 0 or local):
            # the finite-state check of the solver: this loop steps the solver itself, so
            # IVPSolver.run's check never ran here (a NaN estimate only skipped the
            # adaptation and the steps went on); a NaN estimate triggers it at once, for free
            from .integrator import solver_finite_check
            solver_finite_check(sv, "EmbeddedError.step")
        if self.adapt and np.isfinite(self.last):
            fac = self.safety * (self.tol / max(self.last, 1e-300)) ** (1.0 / (self.p_low + 1))
            new = min(max(sv.dt * min(DT_FACTOR_MAX, max(DT_FACTOR_MIN, fac)), self.dt_min), self.dt_max)
            if new != sv.dt:
                sv.set_dt(new)
        return self.last

    def _repr_html_(self) -> str:       # Jupyter display (conatus.notebook)
        from ..notebook import explain_html
        return explain_html(self)

    def explain(self) -> str:
        """Main scheme, companion and whether dt is adapted."""
        mode = (f"adaptive (tol={self.tol:g}, safety={self.safety:g}, dt in [{self.dt_min:g}, "
                f"{self.dt_max:g}], change per step in [DT_FACTOR_MIN, DT_FACTOR_MAX] = "
                f"[{DT_FACTOR_MIN:g}, {DT_FACTOR_MAX:g}])" if self.adapt else "report only, dt unchanged")
        return (f"EmbeddedError({self.solver.scheme} vs {self.companion}): relative max-norm "
                f"difference of one step, {mode}; a temporal error estimate, unrelated to CFL")


class TauWatch:
    """Max |coefficient| of each tau variable, recorded at every call. Use it as the ``callback``
    of :meth:`IVPSolver.run` **with an explicit** ``every`` (``run(n, callback=tw, every=10)``;
    a callback without ``every`` is refused), or call it yourself between
    :meth:`IVPSolver.run_graph` blocks.

    ``growth_limit`` is explicit (no default): when a tau grows by more than that factor over its
    first recorded value, a ``RuntimeWarning`` names it. The history stays in ``self.history``
    ({tau name: [values]}) so a run can report it; nothing is changed in the solver."""

    def __init__(self, solver: IVPSolver, growth_limit: float) -> None:
        import warnings
        self._warn = warnings.warn
        self.solver, self.growth_limit = solver, float(growth_limit)
        taus = solver.plan.taus
        self.names = [v.name for v in solver.system.variables if v in taus]
        self.history: dict[str, list[float]] = {n: [] for n in self.names}
        self.flagged: set[str] = set()

    def __call__(self, solver: IVPSolver | None = None) -> None:
        s = solver or self.solver
        for n in self.names:
            v = float(np.abs(s.coeffs(n)).max())
            h = self.history[n]
            h.append(v)
            ref = next((x for x in h if x > 0), 0.0)
            if ref and v > self.growth_limit * ref and n not in self.flagged:
                self.flagged.add(n)
                self._warn(f"TauWatch: tau '{n}' grew by {v / ref:.2e} (> {self.growth_limit:g}) "
                           f"at iteration {s.iteration}; physical fields are not necessarily "
                           "affected, check them", RuntimeWarning, stacklevel=2)

    def _repr_html_(self) -> str:       # Jupyter display (conatus.notebook)
        from ..notebook import explain_html
        return explain_html(self)

    def explain(self) -> str:
        """Taus watched, the limit, last values and the flagged ones."""
        last = ", ".join(f"{n}={h[-1]:.2e}" for n, h in self.history.items() if h)
        return (f"TauWatch over {len(self.names)} tau variables, growth limit {self.growth_limit:g}"
                + (f"; last: {last}" if last else "") + (f"; flagged: {sorted(self.flagged)}"
                                                         if self.flagged else ""))
