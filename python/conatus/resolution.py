"""Resolution diagnostic: spectral tails, fitted decay, "how many modes?" (a-posteriori, v0.1).

``resolution_report(plan, state)`` (also ``plan.resolution_report(state)``) looks at the spectral
coefficients of every variable of a Cartesian plan, direction by direction, and says whether the
declared resolution resolves them. ``estimate_modes(coeffs, basis)`` answers "how many modes?" for
one coefficient array. Nothing here changes the plan, the solver or the state: the recommended
sizes are printed, never applied (no silent substitution).

Definitions (all declared; the thresholds are module constants printed in every report):

* **mode number** ``m`` along a direction: the polynomial degree for Chebyshev/Legendre/Jacobi
  (``m = 0..N-1``); ``|n|`` for Fourier (RealFourier: pair index ``n = 0..N/2-1``; ComplexFourier:
  ``|fftfreq|``, the always-zero Nyquist entry excluded, ``+n`` and ``-n`` merged).
* **envelope** ``e[m]``: max of ``|c|`` over every coefficient with mode number ``m`` along the
  direction (all other axes and all tensor components included, so one line per variable and
  direction), then made monotone from the right, ``E[m] = max_{m' >= m} e[m']`` (spectra with
  zero odd/even modes, e.g. by symmetry, do not fool the fit).
* **tail**: ``E[M - n_tail] / max(E)``, i.e. the largest coefficient among the last
  ``n_tail = max(TAIL_MIN_MODES, ceil(TAIL_FRACTION * M))`` mode numbers relative to the largest
  coefficient (``M`` = number of mode numbers).
* **decay rate** ``sigma`` (per mode, natural log): least-squares slope of ``-log E[m]`` over the
  last ``FIT_FRACTION`` of the mode numbers whose ``E`` is above ``FIT_FLOOR * max(E)`` (the
  round-off plateau is excluded); needs ``MIN_FIT_POINTS`` points.
* **verdict**: ``resolved`` if tail <= target (``TAIL_TARGET``), ``marginal`` if tail <=
  ``TAIL_MARGINAL``, ``under-resolved`` above; ``zero`` for an identically zero field;
  ``too few modes`` (tail NaN, not judged, not resolved) when the tail window would cover every
  mode number, the mean included.
* **modes for a target** (the "how many modes?" assistant): smallest ``M'`` whose predicted tail,
  ``exp(a - sigma (M' - n_tail(M'))) / max(E)`` from the fitted line ``log E = a - sigma m``, is
  <= target; converted to a basis size (``N = M'`` for polynomials, ``N = 2 M'`` for Fourier).
  **Assumption**: the exponential (geometric) decay seen in the resolved modes continues beyond
  them. It holds for analytic fields in the asymptotic regime; it is wrong for pre-asymptotic
  spectra, fronts or shocks, and for a nonlinear state the dynamics at the new N may differ.
* **advective CFL** (optional): ``dt * max_x sum_i |u_i| / dx_i`` through ``conatus.timestep
  .control.CFL`` (safety 1). A *stability indicator*, not an error control.

Curvilinear plans (polar, sphere) are refused with an ``UnsupportedError``: their radial/spherical
bases have mode numbers per azimuthal group that this module does not interpret yet.
"""
from __future__ import annotations

import html
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

from .errors import InputTypeError, OptionError, UnsupportedError
from .ir.tails import TAIL_FRACTION, TAIL_MIN_MODES, envelope_tail, mode_envelope  # noqa: F401
from .ir.tails import basis_family as _family
from .ir.tails import mode_numbers as _mode_numbers
from .ir.tails import tail_window as _n_tail
from ._repr import compact_repr

# Documented names only (M5, 0.2); the others stay importable from this module.
__all__ = ["resolution_report", "estimate_modes", "ResolutionReport", "TAIL_TARGET",
           "TAIL_MARGINAL"]

#: Statement on a direct solve (``LBVPSolution``, ``NLBVPSolution``, ``TerrainSolution``): the
#: relative envelope tail of each unknown along each direction, above which the solution is
#: stated under-resolved. It is ``TAIL_MARGINAL``, the "under-resolved" verdict of
#: ``resolution_report``. The small residual of a direct solve says nothing here: it measures the
#: discrete system, which is solved to round-off whether or not the grid resolves the solution.
#: Level, measured on every direct solve of the fast suite, guide pages and acceptance scripts: no
#: resolved case above 2.6e-5; the deliberately coarse N = 10 rows of a convergence study at 1.6e-3
#: and above; an annulus Poisson solve with a 3 % error at 1.8e-4. The time steppers' 1e-3 is looser
#: because a turbulent state sits at higher tails. A solution with real content in the last 10 % of its modes is warned even when it
#: is exact there (a polynomial of degree N - 1, a Fourier mode just below Nyquist): the coefficients
#: cannot tell the two apart.
SOLVE_TAIL_TOL = 1e-4
#: The same tail of the data of a Cartesian linear solve (its right-hand sides on the grid). The level
#: is higher because integration damps the top modes of the data. Measured: a step of width 1e-3 at
#: N = 64 has data tail 2.5e-2, solution tail 8.8e-5 and values 3 % off; a Gaussian of width 0.02 at
#: N = 128 has data tail 4.2e-3 and solution tail 9.5e-7.
SOLVE_DATA_TAIL_TOL = 1e-2
#: An unknown whose largest coefficient is below this fraction of the largest of the solution is
#: round-off-sized and not judged (the time steppers' ``RUN_TAIL_FLOOR``).
SOLVE_TAIL_FLOOR = 1e-10
#: Fewer mode numbers than this along a direction: not judged and stated. On so few coefficients
#: an exact low-degree polynomial fills the tail window.
SOLVE_TAIL_MIN_MODES = 8

#: A callable given field is sampled at points off the grid and compared with the interpolant of its
#: grid values. A misfit above this fraction of its largest value means the grid does not see it (T64
#: at N = 32 is constant on the grid). Same level as ``SOLVE_DATA_TAIL_TOL``.
SOLVE_DATA_PROBE_TOL = 1e-2
#: Directions larger than this are not probed (the interpolation costs N^2 per line).
SOLVE_DATA_PROBE_MAX = 4096
#: At most this many points off the grid per line: the misfit needs a sample, and each point costs N.
SOLVE_DATA_PROBE_POINTS = 1031
#: A callable that takes longer than this for one evaluation on a line is not probed (seconds).
SOLVE_DATA_PROBE_BUDGET = 1.0

# ---------------------------------------------------------------- declared thresholds
# TAIL_FRACTION (0.10), TAIL_MIN_MODES (2): the tail window, defined in conatus.ir.tails
TAIL_TARGET = 1e-8         # tail <= this: "resolved" (default target of the recommendation)
TAIL_MARGINAL = 1e-4       # tail <= this: "marginal"; above: "under-resolved"
FIT_FLOOR = 1e-13          # envelope below FIT_FLOOR * max: round-off plateau, not fitted
FIT_FRACTION = 0.5         # the decay is fitted on the last half of the modes above the floor
MIN_FIT_POINTS = 3         # fewer points: no decay rate, no estimate
CFL_REFERENCE = 1.0        # CFL number printed against 1 (indicator only)
MAX_MODES_ESTIMATE = 1 << 16   # the estimate gives up above this many mode numbers

_NOT_IN_STATE = "not in the state"
_ASSUMPTION = ("assumes the exponential decay fitted on the resolved modes continues beyond them "
               "(analytic field, asymptotic regime); unreliable for fronts/shocks or "
               "pre-asymptotic spectra, and a nonlinear state may change at the new N")


# ---------------------------------------------------------------- basis helpers
def _size_for(modes: int, basis) -> int:
    """Basis size N that holds ``modes`` mode numbers."""
    return modes if _family(basis) == "poly" else 2 * modes


def _envelope(c: np.ndarray, axis: int, basis) -> np.ndarray:
    """Monotone envelope E[m] of |c| along ``axis`` (max over every other axis)."""
    a = np.moveaxis(np.abs(np.asarray(c)), axis, 0).reshape(c.shape[axis], -1).max(axis=1)
    mnum, M = _mode_numbers(basis)
    if not np.all(np.isfinite(a)):
        # a NaN/inf coefficient makes the whole envelope non-finite (audit "assurances",
        # Python's max(0.0, nan) is 0.0, so an all-NaN field was "zero" and the report
        # said "all resolved"; a NaN in the tail was dropped from the tail)
        return np.full(M, np.nan)
    return mode_envelope(a, mnum, M)


# ---------------------------------------------------------------- one direction
@dataclass
class DirectionResolution:
    """Resolution of one variable along one direction (see the module docstring)."""
    __repr__ = compact_repr   # one line, arrays by shape

    variable: str
    coord: str
    basis: str
    N: int
    n_modes: int
    dealias: float
    grid_size: int
    envelope: np.ndarray = field(repr=False)
    tail: float
    n_tail: int
    decay_rate: float | None          # sigma per mode number (natural log); None: no fit
    fit_points: int
    verdict: str
    target: float
    marginal: float
    _fit: tuple[float, float] | None = field(default=None, repr=False)   # (a, sigma)
    _family: str = field(default="poly", repr=False)

    @property
    def scale(self) -> float:
        """max |c| of the variable (absolute size: a tail of round-off-sized data means nothing)."""
        return float(self.envelope.max()) if self.envelope.size else 0.0

    def modes_for(self, target: float | None = None) -> int | None:
        """Basis size N predicted to reach ``target`` (default: the report's target), from the
        fitted exponential decay; None when no decay could be fitted. Never applied."""
        target = self.target if target is None else float(target)
        if self._fit is None or self.decay_rate is None or self.decay_rate <= 0:
            return None
        a, sigma = self._fit
        emax = float(self.envelope.max())
        for M in range(max(TAIL_MIN_MODES, 1), MAX_MODES_ESTIMATE):
            if M - _n_tail(M) < 1:            # a window holding the mean judges nothing
                continue
            if math.exp(a - sigma * (M - _n_tail(M))) / emax <= target:
                return M if self._family == "poly" else 2 * M
        return None

    def recommendation(self, target: float | None = None) -> str:
        """One declared line: keep, or increase N to ~value (never applied)."""
        target = self.target if target is None else float(target)
        name = f"N{self.coord}"
        if self.verdict == "zero":
            return f"{self.variable}/{self.coord}: identically zero, nothing to judge"
        if self.verdict == "non-finite":
            return (f"{self.variable}/{self.coord}: non-finite coefficients (NaN or inf): the state "
                    "is not a number, its resolution cannot be judged")
        if self.verdict == "too few modes":
            return (f"{self.variable}/{self.coord}: {self.n_modes} mode number(s), no tail window "
                    f"apart from the mean: not judged; increase {name} to judge the resolution")
        if self.tail <= target:
            return (f"{self.variable}/{self.coord}: keep {name}={self.N} "
                    f"(tail {self.tail:.1e} <= {target:.0e} target)")
        n = self.modes_for(target)
        if n is None:
            return (f"{self.variable}/{self.coord}: tail {self.tail:.1e} > {target:.0e} target; no "
                    "exponential decay could be fitted, so no size is estimated: increase "
                    f"{name} (e.g. x1.5) and run the report again")
        return (f"{self.variable}/{self.coord}: increase {name} from {self.N} to ~{max(n, self.N + 1)}"
                f": tail {self.tail:.1e} > {target:.0e} target; estimated from the fitted decay "
                f"rate {self.decay_rate:.3g}/mode, assuming it continues beyond the resolved modes; "
                "not applied")

    def to_dict(self) -> dict:
        return {"variable": self.variable, "coord": self.coord, "basis": self.basis, "N": self.N,
                "n_modes": self.n_modes, "dealias": self.dealias, "grid_size": self.grid_size,
                "scale": self.scale, "tail": self.tail, "n_tail": self.n_tail, "decay_rate": self.decay_rate,
                "fit_points": self.fit_points, "verdict": self.verdict,
                "N_for_target": self.modes_for(), "target": self.target,
                "envelope": [float(x) for x in self.envelope]}


def _analyse(variable: str, c: np.ndarray, axis: int, basis, target: float,
             marginal: float) -> DirectionResolution:
    """Tail, fitted decay and verdict of ``c`` along ``axis`` (basis ``basis``)."""
    E = _envelope(c, axis, basis)
    M, emax = E.size, float(E.max()) if E.size else 0.0
    nt = _n_tail(M)
    fit, sigma, npts = None, None, 0
    if emax > 0 and np.isfinite(emax):
        # a tail window that covers every mode number holds the mean (mode 0) too: its "tail"
        # would be 1 whatever the data (earlier versions: a constant on N = 4 Fourier
        # modes was "under-resolved, tail 1.0"; same rule as the run-time check of lot B)
        tail = envelope_tail(E, emax) if M - nt >= 1 else float("nan")
        idx = np.nonzero(E > FIT_FLOOR * emax)[0]
        idx = idx[int(len(idx) * (1 - FIT_FRACTION)):] if len(idx) >= 2 * MIN_FIT_POINTS else idx
        npts = int(idx.size)
        if npts >= MIN_FIT_POINTS:
            slope, a = np.polyfit(idx.astype(float), np.log(E[idx]), 1)
            sigma, fit = float(-slope), (float(a), float(-slope))
        verdict = ("too few modes" if M - nt < 1 else "resolved" if tail <= target
                   else "marginal" if tail <= marginal else "under-resolved")
    elif emax == 0:
        tail, verdict = 0.0, "zero"
    else:
        tail, verdict = float("nan"), "non-finite"
    return DirectionResolution(variable, basis.coord.name, type(basis).__name__, basis.size, M,
                               float(basis.dealias), int(basis.grid_size), E, tail, nt, sigma,
                               npts, verdict, target, marginal, fit, _family(basis))


# ---------------------------------------------------------------- "how many modes?"
@dataclass
class ModeEstimate:
    """``estimate_modes`` result: current tail, fitted decay, N predicted for the target."""

    direction: DirectionResolution
    target: float

    @property
    def N(self) -> int | None:
        return self.direction.modes_for(self.target)

    def explain(self) -> str:
        d = self.direction
        rate = "none (no decay fitted)" if d.decay_rate is None else f"{d.decay_rate:.3g}/mode"
        n = self.N
        return (f"How many modes? {d.basis} N={d.N}: tail {d.tail:.2e} ({d.verdict}), decay rate "
                f"{rate}; N for tail <= {self.target:.0e}: "
                + (f"~{n}" if n is not None else "not estimable")
                + f". Assumption: {_ASSUMPTION}.")

    def __str__(self) -> str:
        return self.explain()


def estimate_modes(coeffs: Any, basis, *, axis: int = -1,
                   target: float = TAIL_TARGET) -> ModeEstimate:
    """"How many modes?" for one coefficient array along ``axis`` in ``basis`` (layout of
    ``conatus.reference.transforms``): the N predicted to bring the tail under ``target`` from the
    fitted exponential decay. Declared estimate, never applied."""
    c = np.asarray(coeffs)
    ax = axis % c.ndim
    if c.shape[ax] != basis.size:
        raise OptionError(f"estimate_modes: {c.shape[ax]} coefficients along axis {axis}, "
                          f"basis has N={basis.size}",
                          fix="pass the coefficient array of this basis (not grid values)")
    return ModeEstimate(_analyse("f", c, ax, basis, float(target), TAIL_MARGINAL), float(target))


# ---------------------------------------------------------------- the report
@dataclass
class ResolutionReport:
    """Per (variable, direction) resolution, declared thresholds, optional CFL, recommendations."""

    problem: str
    entries: list[DirectionResolution]
    skipped: list[tuple[str, str]]              # (variable, reason)
    target: float
    marginal: float
    cfl: dict | None = None

    @property
    def unmeasured(self) -> list[str]:
        """Variables of the plan the report did not measure because the state lacks them (tau
        variables and constants are not counted). An empty state never gives zero
        entries, ``ok=True`` and "all resolved"."""
        return [v for v, why in self.skipped if why == _NOT_IN_STATE]

    @property
    def complete(self) -> bool:
        """True when every variable of the plan with a spectral direction was measured."""
        return not self.unmeasured and bool(self.entries)

    @property
    def ok(self) -> bool:
        """True when the report is complete and every (variable, direction) is resolved (or
        identically zero)."""
        return self.complete and all(e.verdict in ("resolved", "zero") for e in self.entries)

    def _headline(self) -> str:
        if not self.entries:
            return "NOT MEASURED (no variable of the plan measured)"
        if self.unmeasured:
            return ("INCOMPLETE: not measured " + ", ".join(self.unmeasured)
                    + ("; the measured ones all resolved" if all(
                        e.verdict in ("resolved", "zero") for e in self.entries)
                       else "; NOT all measured ones resolved"))
        return "all resolved" if self.ok else "NOT all resolved"

    def entry(self, variable: str, coord: str) -> DirectionResolution:
        for e in self.entries:
            if e.variable == variable and e.coord == coord:
                return e
        raise KeyError(f"no entry for ({variable!r}, {coord!r})")

    def recommend(self, target: float | None = None) -> list[str]:
        """Declared recommendation lines (one per entry needing more modes); nothing is changed."""
        t = self.target if target is None else float(target)
        return [e.recommendation(t) for e in self.entries
                if e.verdict not in ("zero",) and not e.tail <= t]   # a NaN tail is no pass

    def thresholds(self) -> str:
        return (f"tail = max |c| over the last max({TAIL_MIN_MODES}, ceil({TAIL_FRACTION:g} M)) "
                f"mode numbers / max |c| (M mode numbers per direction); resolved <= "
                f"{self.target:.0e}, marginal <= {self.marginal:.0e}, under-resolved above; decay "
                f"fitted on the last {FIT_FRACTION:g} of the modes above {FIT_FLOOR:.0e} x max")

    def explain(self) -> str:
        lines = [f"Resolution report for '{self.problem}': " + self._headline(),
                 "  " + self.thresholds(),
                 f"  {'variable':10s} {'dir':4s} {'basis':14s} {'N':>5s} {'dealias':>7s} "
                 f"{'max|c|':>9s} {'tail':>9s} {'decay/mode':>10s}  verdict"]
        for e in self.entries:
            rate = "-" if e.decay_rate is None else f"{e.decay_rate:.3g}"
            lines.append(f"  {e.variable:10s} {e.coord:4s} {e.basis:14s} {e.N:5d} "
                         f"{e.dealias:7.3g} {e.scale:9.1e} {e.tail:9.1e} {rate:>10s}  {e.verdict}")
        for v, why in self.skipped:
            lines.append(f"  skipped {v}: {why}")
        lines.append("  dealias: the tail is measured on the N kept coefficients; aliasing of the "
                     "nonlinear terms on the dealiased grid is not measured here")
        if self.cfl is not None:
            c = self.cfl
            lines.append(f"  advective CFL ({c['velocity']}, dt={c['dt']:g}{c['dt_source']}): "
                         f"{c['cfl']:.3g} (reference {CFL_REFERENCE:g}); it limits the step for "
                         "stability and does not estimate the time error")
        recs = self.recommend()
        lines.append("  suggested resolutions (the plan is unchanged):")
        lines.append(f"    estimated N: {_ASSUMPTION}")
        if recs:
            lines += ["    " + r for r in recs]
        elif not self.entries:
            lines.append("    none: nothing was measured (pass the state of every variable)")
        else:
            lines.append("    none: every measured tail is under the target"
                         + (f" (not measured: {', '.join(self.unmeasured)})"
                            if self.unmeasured else ""))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.explain()

    def to_dict(self) -> dict:
        return {"problem": self.problem, "ok": self.ok, "complete": self.complete,
                "unmeasured": self.unmeasured, "target": self.target,
                "marginal": self.marginal, "definition": self.thresholds(),
                "entries": [e.to_dict() for e in self.entries],
                "skipped": [list(s) for s in self.skipped], "cfl": self.cfl,
                "recommendations": self.recommend()}

    def _repr_html_(self) -> str:
        esc = html.escape
        rows = "".join(
            f"<tr><td>{esc(e.variable)}</td><td>{esc(e.coord)}</td><td>{esc(e.basis)}</td>"
            f"<td>{e.N}</td><td>{e.dealias:g}</td><td>{e.scale:.1e}</td><td>{e.tail:.1e}</td>"
            f"<td>{'-' if e.decay_rate is None else f'{e.decay_rate:.3g}'}</td>"
            f"<td>{esc(e.verdict)}</td></tr>" for e in self.entries)
        recs = "".join(f"<li>{esc(r)}</li>" for r in self.recommend()) or "<li>none</li>"
        cfl = ("" if self.cfl is None else
               f"<p>Advective CFL: {self.cfl['cfl']:.3g} (a stability limit; it does not estimate the time "
               "error)</p>")
        return (f"<div><b>Resolution report: {esc(self.problem)}</b> "
                f"({esc(self._headline())})"
                f"<p><small>{esc(self.thresholds())}</small></p>"
                "<table><tr><th>variable</th><th>dir</th><th>basis</th><th>N</th><th>dealias</th>"
                f"<th>max|c|</th><th>tail</th><th>decay/mode</th><th>verdict</th></tr>{rows}</table>{cfl}"
                f"<p>Suggested resolutions (the plan is unchanged):</p><ul>{recs}</ul></div>")


# ---------------------------------------------------------------- state -> coefficients
def _coefficients(plan, state: Any, layout: str | None) -> dict[str, np.ndarray]:
    """Variable name -> coefficient array, from an IVPSolver, an LBVPSolution or a mapping."""
    from .reference.evaluate import FieldData
    names = {v.name: v for v in plan.variables}
    if hasattr(state, "coeffs") and hasattr(state, "system") and hasattr(state, "plan"):
        if state.plan is not plan:
            raise OptionError("resolution report: the solver was built on another plan",
                              fix="call solver.plan.resolution_report(solver)")
        return {v.name: np.asarray(state.coeffs(v.name)) for v in state.system.variables}
    if hasattr(state, "coeffs") and isinstance(state.coeffs, Mapping):  # LBVPSolution
        return {k: np.asarray(v) for k, v in state.coeffs.items()}
    if not isinstance(state, Mapping):
        raise InputTypeError(
            f"resolution report: unsupported state {type(state).__name__}",
            fix="pass an IVPSolver, an LBVPSolution, or a mapping {variable or name: FieldData "
                "(or array with layout='coeff'/'grid' declared)}")
    out = {}
    for k, d in state.items():
        name = k if isinstance(k, str) else getattr(k, "name", None)
        if name not in names:
            raise OptionError(f"resolution report: {name!r} is not a variable of the plan",
                              fix="valid names: " + ", ".join(names))
        v = names[name]
        if callable(d) and not isinstance(d, (np.ndarray, FieldData)):
            out[name] = plan._to_coeffs(plan._as_grid(v, d), v.domain, v.rank)   # f(X, ...)
        elif isinstance(d, FieldData):
            if d.layout == "coeff":
                out[name] = _check_shape(plan, v, d.data, "coeff")
            else:
                out[name] = plan._to_coeffs(plan._as_grid(v, d), v.domain, v.rank)
        elif layout == "coeff":
            out[name] = _check_shape(plan, v, d, "coeff")
        elif layout == "grid":
            out[name] = plan._to_coeffs(_check_shape(plan, v, d, "grid"), v.domain, v.rank)
        else:
            raise InputTypeError(
                f"resolution report: bare array for {name!r} with no declared layout",
                why="grid values and coefficients have the same shape; guessing would give a "
                    "meaningless tail",
                fix="wrap it in FieldData(field, data, layout=...) or pass layout='coeff'/'grid'")
    return out


def _check_shape(plan, v, data: Any, layout: str) -> np.ndarray:
    """``data`` as an array of the shape of variable ``v`` on ``plan`` (components first, then
    one axis per coordinate; coefficient and grid layouts have the same shape on a Cartesian
    plan). A wrongly shaped array is refused with its shape (a 2-entry array for an N = 16
    variable would otherwise raise a bare IndexError inside the tail measurement)."""
    a = np.asarray(data)
    want = plan._grid_shape(v.domain, v.tensorsig)
    if a.shape != want:
        raise InputTypeError(
            f"resolution report: {v.name!r} has shape {a.shape}, the plan's {layout} layout of "
            f"{v.name} has shape {want}",
            why=f"the tail is read along each axis of the {layout} array; another shape has no "
                "mode numbers to read",
            fix=f"pass the {layout} array of {v.name} on this plan (shape {want}), e.g. "
                f"solver.coeffs({v.name!r}), or the solver itself")
    return a


class _GridView:
    """Minimal solver face (``plan``, ``grid(name)``) for ``conatus.timestep.control.CFL``."""

    def __init__(self, plan, coeffs: Mapping[str, np.ndarray]) -> None:
        self.plan, self._c = plan, coeffs

    def grid(self, name: str) -> np.ndarray:
        return self.plan.to_grid(name, self._c[name])


def resolution_report(plan, state: Any, *, target: float = TAIL_TARGET,
                      marginal: float = TAIL_MARGINAL, velocity: str | None = None,
                      dt: float | None = None, layout: str | None = None,
                      include_taus: bool = False) -> ResolutionReport:
    """Spectral-tail resolution report of ``state`` on ``plan`` (module docstring: definitions).

    ``state``: an ``IVPSolver`` (its current coefficients), an ``LBVPSolution``, or a mapping
    {variable or name: FieldData | callable of the grids | array} (bare arrays need
    ``layout='coeff'`` or ``'grid'``).
    ``velocity`` (+ ``dt``, default ``solver.dt``): adds the advective CFL number. Tau variables
    are skipped unless ``include_taus``. Read-only: the plan and the state are never modified."""
    from .discretize.plan import Plan
    if not isinstance(plan, Plan):
        raise UnsupportedError(
            f"resolution report: {type(plan).__name__} is not supported (Cartesian Plan only)",
            where="conatus.resolution.resolution_report",
            why="polar/spherical bases have per-group radial degrees and (l, m) spectra that the "
                "tail definition here does not interpret; a guessed tail would be wrong",
            fix="inspect the radial/spherical coefficients yourself; curvilinear support is not "
                "implemented yet")
    if not (0 < target <= marginal):
        raise OptionError(f"resolution report: need 0 < target <= marginal "
                          f"(got {target!r}, {marginal!r})")
    coeffs = _coefficients(plan, state, layout)
    coords = plan.domain.coordsys.coords
    entries, skipped = [], []
    for v in plan.variables:
        if v in plan.taus and not include_taus:
            skipped.append((v.name, "tau variable (include_taus=True to report it)"))
            continue
        dirs = [(i, v.domain.basis(crd)) for i, crd in enumerate(coords)
                if v.domain is not None and v.domain.basis(crd) is not None
                and v.domain.basis(crd).size > 1]
        if not dirs:
            skipped.append((v.name, "no spectral direction (constant)"))
            continue
        if v.name not in coeffs:
            skipped.append((v.name, _NOT_IN_STATE))
            continue
        for i, b in dirs:
            _family(b)                     # refuses an unknown basis by name
            entries.append(_analyse(v.name, coeffs[v.name], v.rank + i, b, float(target),
                                    float(marginal)))
    cfl = None
    if velocity is not None:
        from .timestep.control import CFL
        src = ""
        if dt is None:
            if not hasattr(state, "dt"):
                raise OptionError("resolution report: velocity given without dt",
                                  fix="pass dt=... (or an IVPSolver, whose dt is used)")
            dt, src = state.dt, " from solver.dt"
        dt_max = CFL(_GridView(plan, coeffs), velocity, 1.0).compute()
        # a fluid at rest (dt_max = inf) has CFL 0; a NaN velocity gives a NaN CFL, never 0
        # (audit "assurances": every non-finite dt_max used to print CFL 0)
        # an infinite velocity has dt_max = 0: CFL inf (review: ZeroDivisionError)
        cfl = {"velocity": velocity, "dt": float(dt), "dt_source": src,
               "cfl": (0.0 if dt_max == np.inf else np.inf if dt_max == 0
                       else float(dt) / float(dt_max))}
    return ResolutionReport(plan.problem.name, entries, skipped, float(target), float(marginal), cfl)


@dataclass
class SolutionTails:
    """Spectral tails of a solved boundary-value problem (``solution_tails``)."""
    bad: list[str] = field(default_factory=list)        # "u along z: tail 2e-01 (N = 32)"
    short: list[str] = field(default_factory=list)      # too few modes to judge
    worst: tuple[str, float] | None = None              # largest judged tail and where
    skipped: str = ""                                   # why nothing was judged
    nonfinite: list[str] = field(default_factory=list)  # unknowns holding NaN or inf
    unread: list[str] = field(default_factory=list)     # directions without readable mode numbers
    small: list[str] = field(default_factory=list)      # unknowns too small beside the others
    data_worst: float | None = None                     # largest tail of the data
    aliased: list[str] = field(default_factory=list)    # given fields the grid cannot see
    unprobed: list[str] = field(default_factory=list)   # callables the probe left alone, and why

    def statement(self) -> str:
        """The explain() line(s) of the check."""
        lines = []
        if self.nonfinite:
            lines.append("resolution check: non-finite coefficients in "
                         + ", ".join(self.nonfinite))
        if self.bad:
            lines.append(f"resolution check: UNDER-RESOLVED, {'; '.join(self.bad)} (relative "
                         f"spectral tail above SOLVE_TAIL_TOL = {SOLVE_TAIL_TOL:g}): the solution "
                         "is not converged although its residual may be small; raise N "
                         "(resolution_report(sol.plan, sol) estimates how many modes)")
        elif self.worst is not None:
            lines.append(f"resolution check: largest spectral tail {self.worst[1]:.1e} "
                         f"({self.worst[0]}), below SOLVE_TAIL_TOL = {SOLVE_TAIL_TOL:g}; a tail "
                         "is a necessary sign of convergence, not a proof")
        elif self.skipped:
            lines.append(f"resolution check: not run ({self.skipped})")
        if self.unprobed:
            lines.append("resolution check: not sampled off the grid: " + "; ".join(self.unprobed))
        if self.aliased:
            lines.append("resolution check: INPUT DATA NOT RESOLVED, " + "; ".join(self.aliased)
                         + f" (largest misfit off the grid, above SOLVE_DATA_PROBE_TOL = "
                         f"{SOLVE_DATA_PROBE_TOL:g}): the grid values of this data do not represent "
                         "it, whatever the tails say; raise N")
        if self.short:
            lines.append(f"resolution check: not judged, fewer than {SOLVE_TAIL_MIN_MODES} mode "
                         "numbers: " + "; ".join(self.short))
        if self.unread:
            lines.append("resolution check: not judged, no mode numbers read here (a mapped domain, "
                         "or a compound, Hermite or Laguerre basis): " + "; ".join(self.unread)
                         + "; compare two values of N")
        if self.small:
            lines.append(f"resolution check: not judged, below {SOLVE_TAIL_FLOOR:g} of the largest "
                         "unknown: " + ", ".join(self.small))
        return "\n  ".join(lines)

    def warn(self) -> None:
        """UnderResolvedWarning when a direction is under-resolved; nothing else changes."""
        if self.bad:
            from .errors import UnderResolvedWarning, _warn_user
            _warn_user("the boundary-value solution is under-resolved: " + "; ".join(self.bad)
                       + f" (relative spectral tail above SOLVE_TAIL_TOL = {SOLVE_TAIL_TOL:g}; the "
                       "solve's small residual measures the discrete system only). Its values are "
                       "not converged: raise N along these directions "
                       "(resolution_report(sol.plan, sol) estimates how many modes). Nothing was "
                       "changed.", UnderResolvedWarning)
        warn_aliased(self.aliased)


def _solution_axes(plan, v, c: np.ndarray, unread: list[str]
                   ) -> list[tuple[str, int, np.ndarray, int, int]]:
    """(coordinate, axis, mode number per index, number of mode numbers, size) of each direction
    of the coefficient array ``c`` of variable ``v``, in the layout of its plan. A direction whose
    mode numbers cannot be read goes to ``unread``."""
    rank = int(getattr(v, "rank", 0) or 0)
    axes = []
    if getattr(plan, "is_polar", False) or getattr(plan, "is_sphere", False):
        s = plan.space
        mm = np.abs(np.asarray(s.ms, dtype=int))
        if c.shape[rank] == mm.size:
            axes.append(("phi", rank, mm, int(mm.max()) + 1 if mm.size else 0, c.shape[rank]))
        rest = c.shape[rank + 1:]
        if getattr(plan, "is_sphere", False):
            names = ("theta", "r")
        else:
            names = ("r", "z") if getattr(s, "cyl", False) else ("r",)  # layout: m, radial[, k_z]
        for i, (name, size) in enumerate(zip(names, rest, strict=False)):
            if name == "z":              # Fourier in z: |n|, the Nyquist entry excluded
                m = np.abs(np.asarray(s.zfreq)).copy()
                if m.size != size:
                    continue
                m[size // 2] = -1
                axes.append((name, rank + 1 + i, m, int(m.max()) + 1, size))
            else:
                axes.append((name, rank + 1 + i, np.arange(size), size, size))
        return axes
    coords = plan.domain.coordsys.coords
    dom = v.domain
    for i, co in enumerate(coords):
        b = dom.basis(co) if dom is not None else None
        if b is None:
            continue
        if rank + i >= c.ndim or c.shape[rank + i] != b.size:
            unread.append(f"{v.name} along {co.name}")
            continue
        try:
            mnum, M = _mode_numbers(b)
        except UnsupportedError:      # no interpretable mode numbers: not judged
            unread.append(f"{v.name} along {co.name}")
            continue
        axes.append((co.name, rank + i, mnum, M, b.size))
    return axes


def _axis_tails(plan, v, a: np.ndarray, unread: list[str], short: list[str], label: str
                ) -> list[tuple[str, int, float]]:
    """(coordinate, size, tail) of the coefficient array ``a`` of ``v`` along each direction that
    has enough modes to judge; a direction with too few goes to ``short``."""
    from .ir.tails import envelope_tail, mode_envelope, tail_window
    abs_a = np.abs(a)
    out = []
    for cname, axis, mnum, M, size in _solution_axes(plan, v, a, unread):
        if size <= 2 or M < 1:
            continue
        host = np.moveaxis(abs_a, axis, 0).reshape(size, -1).max(axis=1)
        E = mode_envelope(host, mnum, M)
        emax = float(E.max()) if E.size else 0.0
        if not emax > 0:
            continue
        if M < SOLVE_TAIL_MIN_MODES or M - tail_window(M) < 1:
            short.append(f"{label} along {cname} (N = {size})")
            continue
        out.append((cname, size, envelope_tail(E, emax)))
    return out


def _rhs_tails(out: SolutionTails, plan, rhs: Any) -> None:
    """The right-hand sides as the grid sees them (``SOLVE_DATA_TAIL_TOL``). Integrating data smooths
    it, so a step in the data can leave a tail below ``SOLVE_TAIL_TOL`` on the solution and values 3 %
    off. Data of round-off size beside the others is left out."""
    arrays = [(i, eq, np.asarray(r)) for i, (eq, r) in enumerate(zip(plan.problem.equations, rhs,
                                                                     strict=True)) if np.size(r)]
    if not arrays or not all(np.all(np.isfinite(a)) for _, _, a in arrays):
        return
    top = max(float(np.abs(a).max()) for _, _, a in arrays)
    for i, eq, a in arrays:
        if not float(np.abs(a).max()) > SOLVE_TAIL_FLOOR * top:
            continue
        label = f"data of equation {i}"
        v = SimpleNamespace(name=label, rank=int(eq.rhs.rank), domain=eq.domain)
        for cname, size, tail in _axis_tails(plan, v, a, [], [], label):
            if out.data_worst is None or tail > out.data_worst:
                out.data_worst = tail
            if tail > SOLVE_DATA_TAIL_TOL:
                out.bad.append(f"{label} along {cname}: tail {tail:.1e} (N = {size}), level "
                               f"SOLVE_DATA_TAIL_TOL = {SOLVE_DATA_TAIL_TOL:g}")


def _probe_points(basis: Any, count: int) -> np.ndarray:
    """``count`` points off the grid: equidistributed in the interval, denser at the ends of a
    polynomial basis, as its nodes are; the ends themselves are left out."""
    lo, hi = basis.bounds
    frac = (np.arange(count) * 0.6180339887498949 + 0.25) % 1.0
    if getattr(basis, "periodic", False):
        return lo + (hi - lo) * frac
    return 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos(np.pi * (0.02 + 0.96 * frac))


def _interpolate_fourier(values: np.ndarray, g0: float, length: float, x: np.ndarray) -> np.ndarray:
    """Trigonometric interpolant of the columns of ``values`` (equispaced nodes along axis 0) at x."""
    n = values.shape[0]
    c = np.fft.fft(values, axis=0) / n
    k = np.fft.fftfreq(n, 1.0 / n)
    E = np.exp(2j * np.pi * np.outer(x - g0, k) / length)
    if n % 2 == 0:                    # the Nyquist mode, as a cosine
        E[:, n // 2] = np.cos(2 * np.pi * (n // 2) * (x - g0) / length)
    out = E @ c
    return out.real if np.isrealobj(values) else out


def _interpolate_polynomial(values: np.ndarray, nodes: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Barycentric interpolant of the columns of ``values`` at x (distinct nodes, x off the nodes)."""
    d = nodes[:, None] - nodes[None, :]
    np.fill_diagonal(d, 1.0)
    logw = -np.sum(np.log(np.abs(d)), axis=1)
    w = np.exp(logw - logw.max()) * np.where(np.sum(d < 0, axis=1) % 2 == 0, 1.0, -1.0)
    r = w[None, :] / (x[:, None] - nodes[None, :])
    return (r @ values) / np.sum(r, axis=1, keepdims=True)


def _on_line(fn: Any, native: list, i: int, x: np.ndarray, frac: float) -> np.ndarray:
    """Values of the callable ``fn`` at x along direction i, the other coordinates at the grid
    point at fraction ``frac``: one row per point, one column per component."""
    grids = []
    for j, (_, g) in enumerate(native):
        g = np.asarray(g).ravel()
        v = x if j == i else g[int(frac * g.size)][None]
        sh = [1] * len(native)
        sh[j] = v.size
        grids.append(v.reshape(sh))
    raw = fn(*grids)
    if isinstance(raw, (tuple, list)):
        comps = [np.broadcast_to(np.asarray(r), np.broadcast_shapes(np.shape(r), grids[i].shape))
                 for r in raw]
        return np.stack([q.ravel() for q in comps], axis=1)
    arr = np.broadcast_to(np.asarray(raw), np.broadcast_shapes(np.shape(raw), grids[i].shape))
    return np.moveaxis(arr, arr.ndim - len(native) + i, 0).reshape(x.size, -1)


def _reads_generator(fn: Any) -> bool:
    """True when ``fn`` holds or names a random generator (its closure, its globals, its ``self``):
    evaluating it again would advance a stream the user's next draws come from."""
    from random import Random
    kinds = (np.random.Generator, np.random.RandomState, np.random.BitGenerator, Random)
    fn = getattr(fn, "__wrapped__", fn)
    seen = [getattr(fn, "__self__", None)]
    seen += [c.cell_contents for c in (getattr(fn, "__closure__", None) or ()) if _cell_full(c)]
    code, names = getattr(fn, "__code__", None), getattr(fn, "__globals__", {})
    seen += [names[n] for n in (code.co_names if code else ()) if n in names]
    return any(isinstance(v, kinds) for v in seen)


def _cell_full(cell: Any) -> bool:
    try:
        cell.cell_contents  # noqa: B018 -- an empty closure cell raises on reading
    except ValueError:
        return False
    return True


def data_alias_probe(plan: Any, unprobed: list[str] | None = None) -> list[tuple[str, str, int, float]]:
    """(given field, coordinate, N, misfit) of each callable given field and each direction whose
    grid does not represent it: the callable at points off the grid against the interpolant of its
    values on the grid, along lines through the grid, relative to its largest value. A sampled
    tail cannot see this: T64 is constant on the N = 32 Chebyshev grid. On Fourier and Jacobi
    directions; a callable that fails or gives non-finite values off the grid is not judged.

    The probe calls the user's function again, up to 5 times per direction. The values of the solve
    were computed before it and do not change. The global NumPy and Python random states are
    restored afterwards. A callable that holds a random generator, gives two different answers at
    the same points, or takes longer than ``SOLVE_DATA_PROBE_BUDGET`` for one evaluation is left
    alone and named in ``unprobed``. A callable that counts its calls in its own state does see
    the extra calls."""
    import random
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        return _probe_callables(plan, [] if unprobed is None else unprobed)
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def _probe_callables(plan: Any, unprobed: list[str]) -> list[tuple[str, str, int, float]]:
    """The body of :func:`data_alias_probe`, inside its restored random states."""
    import time
    from .ir.bases import Jacobi, _Fourier
    from .reference import transforms as T
    bad = []
    callables = getattr(plan, "_ncc_callables", None) or {}
    coords = plan.domain.coordsys.coords
    # data of round-off size beside the other data is not judged, as in the tails
    top_all = max((float(np.abs(plan._ncc_data[f]).max(initial=0.0)) for f in callables
                   if np.all(np.isfinite(plan._ncc_data[f]))), default=0.0)
    for f, fn in callables.items():
        if f.domain is None or not float(np.abs(plan._ncc_data[f]).max(initial=0.0)) > SOLVE_TAIL_FLOOR * top_all:
            continue
        if _reads_generator(fn):
            unprobed.append(f"given {f.name} holds a random generator")
            continue
        native = []
        for c in coords:
            b = f.domain.basis(c)
            native.append((b, T.grid(b, dealias=False) if b is not None else np.zeros(1)))
        repeatable = None
        for i, c in enumerate(coords):
            b = native[i][0]
            if not isinstance(b, (Jacobi, _Fourier)) or not 4 <= b.size <= SOLVE_DATA_PROBE_MAX:
                continue
            nodes = np.asarray(native[i][1]).ravel()
            pts = _probe_points(b, min(2 * nodes.size + 7, SOLVE_DATA_PROBE_POINTS))
            worst = 0.0
            reason = None
            for line in (0.37, 0.71):
                try:
                    started = time.perf_counter()
                    v = _on_line(fn, native, i, nodes, line)
                    took = time.perf_counter() - started
                    if took > SOLVE_DATA_PROBE_BUDGET:
                        reason = f"given {f.name} takes {took:.1f} s for one evaluation on a line"
                        worst = 0.0
                        break
                    if repeatable is None:
                        again = _on_line(fn, native, i, nodes, line)
                        repeatable = bool(np.array_equal(v, again, equal_nan=True))
                    if not repeatable:
                        reason = f"given {f.name} gives different values at the same points"
                        worst = 0.0
                        break
                    p = _on_line(fn, native, i, pts, line)
                except Exception:  # noqa: BLE001 -- user code off its grid: stated as not judged
                    worst = 0.0
                    break
                if v.shape[1] != p.shape[1] or not (np.all(np.isfinite(v)) and np.all(np.isfinite(p))):
                    worst = 0.0
                    break
                top = max(float(np.abs(v).max()), float(np.abs(p).max()))
                if not top > 0:
                    continue
                if isinstance(b, _Fourier):
                    q = _interpolate_fourier(v, float(nodes[0]), float(b.bounds[1] - b.bounds[0]), pts)
                else:
                    q = _interpolate_polynomial(v, nodes, pts)
                worst = max(worst, float(np.abs(q - p).max()) / top)
            if reason:
                unprobed.append(reason)
                break
            if worst > SOLVE_DATA_PROBE_TOL:
                bad.append((f.name, c.name, int(b.size), worst))
    return bad


def _aliased_text(found: list[tuple[str, str, int, float]]) -> list[str]:
    return [f"given {n} along {c}: misfit {m:.1e} off the grid (N = {size})" for n, c, size, m in found]


def warn_aliased(aliased: list[str]) -> None:
    """UnderResolvedWarning naming the given fields the grid does not represent."""
    if aliased:
        from .errors import UnderResolvedWarning, _warn_user
        _warn_user("input data is not resolved by this grid: " + "; ".join(aliased)
                   + f" (relative misfit between the callable off the grid and the interpolant of "
                   f"its grid values, above SOLVE_DATA_PROBE_TOL = {SOLVE_DATA_PROBE_TOL:g}). The "
                   "computed fields see the aliased data; raise N. Nothing was changed.",
                   UnderResolvedWarning)


def check_given_data(plan: Any) -> None:
    """The data probe alone, for a time stepper or an eigenvalue problem built from callables."""
    warn_aliased(_aliased_text(data_alias_probe(plan)))


def solution_tails(plan, coeffs: Mapping[str, np.ndarray], rhs: Any = None) -> SolutionTails:
    """Relative spectral tail of each unknown of a solved boundary-value problem along each
    direction, and of the data ``rhs`` when given (coefficients per equation, ``SOLVE_TAIL_TOL``).
    Read-only, and cheap beside the solve."""
    out = SolutionTails()
    out.aliased = _aliased_text(data_alias_probe(plan, out.unprobed))
    taus = set(getattr(plan, "taus", ()) or ())
    variables = [v for v in getattr(plan, "variables", ()) if v not in taus]
    arrays = {}
    for v in variables:
        a = coeffs.get(v.name)
        if isinstance(a, Mapping):        # mapped domains: {component: array}
            parts = [np.asarray(x) for x in a.values()]
            if not parts or any(x.shape != parts[0].shape for x in parts):
                continue
            a = np.stack(parts) if v.rank else parts[0]
        if a is not None:
            arrays[v.name] = np.asarray(a)
    if not arrays:
        out.skipped = "no coefficients of an unknown in this solution"
        return out
    for name, a in arrays.items():
        if a.size and not np.all(np.isfinite(a)):
            out.nonfinite.append(name)
    tops = [float(np.abs(a).max()) for n, a in arrays.items()
            if a.size and n not in out.nonfinite]
    top = max(tops, default=0.0)
    if not top > 0:
        out.skipped = "the solution is identically zero" if not out.nonfinite else ""
        return out
    for v in variables:
        a = arrays.get(v.name)
        if a is None or not a.size or v.name in out.nonfinite:
            continue
        if not float(np.abs(a).max()) > SOLVE_TAIL_FLOOR * top:
            if float(np.abs(a).max()) > 0:
                out.small.append(v.name)
            continue
        for cname, size, tail in _axis_tails(plan, v, a, out.unread, out.short, v.name):
            if out.worst is None or tail > out.worst[1]:
                out.worst = (f"{v.name} along {cname}, N = {size}", tail)
            if tail > SOLVE_TAIL_TOL:
                out.bad.append(f"{v.name} along {cname}: tail {tail:.1e} (N = {size})")
    if rhs is not None:
        _rhs_tails(out, plan, rhs)
    return out
