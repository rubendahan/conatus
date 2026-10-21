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
from typing import Any

import numpy as np

from .errors import InputTypeError, OptionError, UnsupportedError
from ._repr import compact_repr

# Documented names only (M5, 0.2); the others stay importable from this module.
__all__ = ["resolution_report", "estimate_modes", "ResolutionReport", "TAIL_TARGET",
           "TAIL_MARGINAL"]

# ---------------------------------------------------------------- declared thresholds
TAIL_FRACTION = 0.10       # the tail window is the last 10 % of the mode numbers ...
TAIL_MIN_MODES = 2         # ... and at least 2 mode numbers
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
def _family(basis) -> str:
    """'poly' (degree = index), 'rfourier' or 'cfourier'; UnsupportedError otherwise."""
    from .ir.bases import ComplexFourier, Jacobi, RealFourier
    if isinstance(basis, RealFourier):
        return "rfourier"
    if isinstance(basis, ComplexFourier):
        return "cfourier"
    if isinstance(basis, Jacobi):
        return "poly"
    raise UnsupportedError(
        f"resolution report: basis {type(basis).__name__} is not supported",
        where=f"direction {basis.coord.name}",
        why="only Chebyshev/Legendre/Jacobi (degree) and Fourier (|wavenumber|) coefficients have "
            "a mode number this module interprets; guessing one would give a wrong tail",
        fix="inspect the coefficients yourself (solver.coeffs(name)) for this basis")


def _mode_numbers(basis) -> tuple[np.ndarray, int]:
    """Mode number of every coefficient index (-1 = excluded) and the number of mode numbers."""
    N, fam = basis.size, _family(basis)
    if fam == "poly":
        return np.arange(N), N
    if fam == "rfourier":
        return np.arange(N) // 2, N // 2
    m = np.abs(np.rint(np.fft.fftfreq(N) * N)).astype(int)
    if N % 2 == 0:
        m[N // 2] = -1                 # Nyquist entry, always zero by convention
    return m, (N + 1) // 2


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
    e = np.zeros(M)
    for i, m in enumerate(mnum):
        if m >= 0:
            e[m] = max(e[m], a[i])
    return np.maximum.accumulate(e[::-1])[::-1]


def _n_tail(M: int) -> int:
    return min(M, max(TAIL_MIN_MODES, math.ceil(TAIL_FRACTION * M)))


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
        # would be 1 whatever the data (F101-10-16: a constant on N = 4 Fourier
        # modes was "under-resolved, tail 1.0"; same rule as the run-time check of lot B)
        tail = float(E[M - nt] / emax) if M - nt >= 1 else float("nan")
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
        variables and constants are not counted). Review 0.1.1, F10: an empty state gave zero
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
    plan). Review 0.1.1, F17: a 2-entry array for an N = 16 variable raised a bare IndexError
    from inside the tail measurement."""
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
