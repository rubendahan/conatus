"""Step-size information, kept separate by design (CFL ≠ error control).

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

import numbers

import numpy as np

from ..reference import transforms as T
from ..errors import ConvergenceError, OptionError
from .integrator import IVPSolver, require_family, rk_tableau, sbdf_coefficients

_DOC = "docs/site/guide/timesteppers.md"
# adaptive step limiter (declared defaults: conatus.tolerances.DECLARED, shown by explain()):
# the step changes by a factor in [DT_FACTOR_MIN, DT_FACTOR_MAX] per step, whatever the estimate
DT_FACTOR_MIN = 0.2
DT_FACTOR_MAX = 2.0
# adaptive mode: a step whose estimate exceeds tol is rejected (state restored) and retried with
# a smaller dt, at most MAX_REJECTIONS times in a row (then ConvergenceError: never accepted)
MAX_REJECTIONS = 20

__all__ = ["CFL", "EmbeddedError", "TauWatch"]

# main scheme -> (companion of lower order, its order): the companion recomputes the step from
# the same X^n (not an embedded pair sharing stages); RKSMR / RKGFY are second order (IMEX).
_COMPANION = {"RK443": ("RK222", 2), "RK222": ("RK111", 1), "RKSMR": ("RK111", 1),
              "RKGFY": ("RK111", 1)}


class CFL:
    """dt_cfl = safety / max_x sum_i |u_i(x)| / dx_i(x) on the collocation grid.

    Geometric contract: Cartesian coordinates only. ``u_i`` are the
    components of the velocity along the coordinate axes (lengths per time) and ``dx_i`` the
    distance between neighbouring collocation nodes of axis i (L/N on a periodic axis, the node
    spacing on the others), both in the same length unit. On a curvilinear geometry (disk,
    annulus, pipe, sphere, ball, shell) a coordinate is an angle and the physical spacing is
    r dphi or r sin(theta) dphi: not implemented, refused rather than returned in mixed units."""

    def __init__(self, solver: IVPSolver, velocity: str, safety: float) -> None:
        # safety a finite real number (audit inputs: float("0.5") parsed text)
        from ..errors import _scalar_of
        safety = _scalar_of(safety)              # np.array(0.5): one number
        if isinstance(safety, (bool, np.bool_)) or not isinstance(safety, numbers.Real):
            raise OptionError(f"CFL: safety={safety!r} must be a finite positive number",
                              fix="safety=0.5", doc=_DOC)
        self.solver, self.velocity, self.safety = solver, velocity, float(safety)
        plan = solver.plan
        cs = plan.domain.coordsys
        # the velocity is a vector unknown of the problem, or the scalar unknown of a 1D problem
        # (Burgers) (audit inputs: an unknown name was a bare ValueError, and a scalar
        # was indexed as a vector: u[0], its FIRST GRID VALUE, was taken for the velocity of the
        # whole 1D domain, and a scalar on a 2D grid gave its first rows as components)
        names = getattr(plan, "variables", None)
        var = None if names is None else next((v for v in names if v.name == velocity), None)
        rank = None if var is None else len(getattr(var, "tensorsig", ()))
        dim = len(cs.coords)
        if names is not None and not (rank == 1 or (rank == 0 and dim == 1)):
            from ..errors import InputTypeError, did_you_mean
            vectors = [v.name for v in names if len(getattr(v, "tensorsig", ())) == 1]
            raise InputTypeError(
                f"CFL: {velocity!r} is not a vector unknown of the problem"
                + did_you_mean(str(velocity), vectors),
                why="the CFL step reads the velocity components u_i along each axis",
                fix="the name of the velocity: " + (", ".join(vectors) or "no vector unknown "
                                                    "here (the CFL step needs a velocity)"),
                doc=_DOC)
        if getattr(cs, "curvilinear", False):
            from ..errors import UnsupportedError
            raise UnsupportedError(
                f"CFL on the curvilinear coordinates {tuple(c.name for c in cs.coords)}",
                why="the CFL step needs the physical node spacing (r dphi, r sin(theta) dphi); the "
                    "spacing of an angle in radians would mix units with the velocity",
                fix="estimate the advective step from the grid yourself (e.g. min over the grid "
                    "of r dphi / |u_phi| and dr / |u_r|), or check the step with EmbeddedError",
                doc=_DOC)
        self._scalar = rank == 0
        if not np.isfinite(self.safety) or self.safety <= 0:
            raise OptionError(f"CFL: safety must be a finite positive number, got {safety!r}",
                              fix="safety=0.5", doc=_DOC)
        self.coords = cs.coords
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
        if self._scalar:                         # 1D: the scalar is the velocity component
            u = np.asarray(u)[None]
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
        # validated before any step (review finding 10)
        if tol is not None and not (np.isfinite(tol) and tol > 0):
            raise OptionError(f"EmbeddedError: tol must be a finite positive number, got {tol!r}",
                              fix="tol=1e-6 (the per-step relative error you accept)", doc=_DOC)
        if not (np.isfinite(dt_min) and dt_min >= 0) or not dt_max > dt_min:
            raise OptionError(f"EmbeddedError: need 0 <= dt_min < dt_max, got dt_min={dt_min!r}, "
                              f"dt_max={dt_max!r}", fix="dt_min=0.0, dt_max=np.inf (no bound)",
                              doc=_DOC)
        self.adapt, self.tol, self.safety = adapt, tol, safety
        self.dt_min, self.dt_max = float(dt_min), float(dt_max)
        self.last = np.nan
        self.rejected = 0             # adaptive mode: steps rejected (and retried) so far

    def step(self) -> float:
        """Advance one step with the main scheme; return ||X - X_low|| / ||X|| (max norm).

        ``adapt=True``: a step whose estimate exceeds ``tol`` is rejected, the state (X, t,
        iteration) restored and the step retried with a smaller dt; the accepted estimate is
        returned and the next dt proposed from it. ``ConvergenceError`` if ``tol`` cannot be met
        at ``dt_min`` or after ``MAX_REJECTIONS`` retries."""
        if not self.adapt:
            return self._attempt()
        sv = self.solver
        X0, t0, it0 = sv.X.copy(), sv.t, sv.iteration
        for _ in range(MAX_REJECTIONS + 1):
            err = self._attempt()
            # a non-finite estimate (the step blew up) is a rejection with the smallest factor
            fac = (self.safety * (self.tol / max(err, 1e-300)) ** (1.0 / (self.p_low + 1))
                   if np.isfinite(err) else 0.0)
            if err <= self.tol:
                new = min(max(sv.dt * min(DT_FACTOR_MAX, max(DT_FACTOR_MIN, fac)), self.dt_min),
                          self.dt_max)
                if new != sv.dt:
                    sv.set_dt(new)
                return err
            # rejected: back to the start of the step, retry with a smaller dt
            self.rejected += 1
            sv.X, sv.t, sv.iteration = X0.copy(), t0, it0
            if sv.dt <= self.dt_min:
                sv.set_dt(sv.dt)                   # clock origin back at t0
                raise ConvergenceError(
                    f"EmbeddedError: estimate {err:.3e} > tol = {self.tol:g} at dt = dt_min = "
                    f"{self.dt_min:g}", where=f"t = {t0:g}",
                    why="the step cannot be made accurate enough within the declared bounds; it "
                        "is never accepted above tol",
                    fix="lower dt_min, raise tol, or check the state (a blow-up?)", doc=_DOC)
            new = max(sv.dt * max(DT_FACTOR_MIN, min(fac, 1.0)), self.dt_min)
            sv.set_dt(new)
        raise ConvergenceError(
            f"EmbeddedError: {MAX_REJECTIONS} rejections in a row at t = {t0:g} (last estimate "
            f"{err:.3e} > tol = {self.tol:g}, dt = {sv.dt:g})",
            why="the estimate does not decrease with dt (tol below round-off, or a non-smooth "
                "right-hand side)", fix="raise tol or check the state", doc=_DOC)

    def _attempt(self) -> float:
        """One step with the main scheme and its estimate (no adaptation)."""
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
        return self.last

    def _repr_html_(self) -> str:       # Jupyter display (conatus.notebook)
        from ..notebook import explain_html
        return explain_html(self)

    def explain(self) -> str:
        """Main scheme, companion and whether dt is adapted."""
        mode = (f"adaptive (tol={self.tol:g}, safety={self.safety:g}, dt in [{self.dt_min:g}, "
                f"{self.dt_max:g}], change per step in [DT_FACTOR_MIN, DT_FACTOR_MAX] = "
                f"[{DT_FACTOR_MIN:g}, {DT_FACTOR_MAX:g}]; a step above tol is rejected and "
                f"retried with a smaller dt, {self.rejected} rejected so far)"
                if self.adapt else "report only, dt unchanged")
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
