"""Edge tracking by iterated bisection (Skufca, Yorke & Eckhardt 2006, Schneider,
Eckhardt & Yorke 2007, Schneider et al. 2008 PRE 78 037301).

Given two states on either side of the laminar–turbulent boundary (``lo`` decays, ``hi`` becomes
turbulent), the family X(λ) = lo + λ (hi − lo) is bisected in λ until the bracket is narrower than
``bisection_tol`` (relative to ‖hi − lo‖); the two bracketing trajectories are then followed
together until they separate by more than ``separation`` (relative distance ‖A − B‖ / ‖A − ref‖,
``ref`` = the laminar state) or one of them is classified; the pair reached one check earlier
becomes the new (lo, hi) and the bisection restarts. The edge trajectory (the lower member of
every pair, sampled at each check) is recorded with the user's ``measure`` callback.

Everything the classification depends on is supplied and reported: ``classify(X) -> -1
(laminar), +1 (turbulent), 0 (undecided)``, the check interval ``t_check`` and the horizon
``t_max`` after which an undecided trial is refused (never guessed), and ``t_min``: no verdict
before it (the initial transient of a perturbation can cross the thresholds: measured on plane
Couette at Re = 400, D > 2.5 during the first time units of perturbations that relaminarize, run
```` ❌, whose bisection tracked the threshold crossing instead)."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..errors import OptionError, check_count

__all__ = ["EdgeTracker", "ThresholdBisection", "UndecidedTrajectory"]


class UndecidedTrajectory(RuntimeError):
    """A bisection trial neither relaminarized nor became turbulent within t_max."""


class EdgeTracker:
    """``advance(X, T) -> X`` integrates T time units (T a multiple of the time step, declared by
    the caller); ``classify``; ``measure(X) -> dict`` (recorded along the edge); ``norm(X)`` the
    distance used for the separation test; ``ref``: laminar state."""

    def __init__(self, advance: Callable, classify: Callable, measure: Callable, norm: Callable,
                 ref, *, t_check: float, t_max: float, bisection_tol: float, separation: float,
                 t_min: float, project: Callable | None = None, log=None) -> None:
        self.advance, self.classify, self.measure, self.norm = advance, classify, measure, norm
        self.ref, self.project = ref, project
        self.t_check, self.t_max, self.t_min = float(t_check), float(t_max), float(t_min)
        # fate() classifies at j t_check up to t_max: a t_max between two checks would be overshot
        # (the last check past t_max), so it must be a whole number of checks (never rounded)
        n_checks = round(self.t_max / self.t_check) if self.t_check > 0 else 0
        from ..timestep.integrator import is_whole
        if n_checks < 1 or not is_whole(self.t_max / self.t_check):
            raise OptionError(f"t_max = {self.t_max:g} is not a positive multiple of t_check = {self.t_check:g}",
                              why="the state is classified every t_check up to and including t_max",
                              fix="choose t_max = n * t_check (n >= 1)")
        self.tol, self.separation = float(bisection_tol), float(separation)
        self.log = log
        self.trajectory: list[dict] = []        # edge samples (t, measures)
        self.rounds: list[dict] = []
        self.t = 0.0
        self.n_trials = 0

    def fate(self, X) -> tuple[int, float]:
        """(+1 turbulent / −1 laminar, decision time); raises UndecidedTrajectory.

        The state is classified at every check t = j t_check (j >= 0, from t_min on) up to and
        including t_max; the times are j * t_check, not a running sum (no drift past t_max)."""
        j, t = 0, 0.0
        while True:
            c = self.classify(X) if t >= self.t_min else 0      # no verdict on the transient
            if c not in (-1, 0, 1):
                raise OptionError(f"classify returned {c!r} at t = {t:g}",
                                  why="the verdict is -1 (laminar), +1 (turbulent) or 0 (undecided)",
                                  fix="return one of -1, 0, +1 from classify(X)")
            if c:
                return int(c), t
            if t >= self.t_max * (1 - 1e-12):                  # j t_check = t_max up to rounding
                raise UndecidedTrajectory(f"undecided after t_max = {self.t_max:g}")
            X = self.advance(X, self.t_check)
            j += 1
            t = j * self.t_check

    def bisect(self, lo, hi) -> tuple:
        """Bracket (A, B) of the edge on the segment [lo, hi] (A laminar side, B turbulent side)."""
        a, b = 0.0, 1.0
        d = hi - lo
        info = []
        while b - a > self.tol:
            m = 0.5 * (a + b)
            X = lo + m * d
            if self.project is not None:
                X = self.project(X)
            f, td = self.fate(X)
            self.n_trials += 1
            info.append((m, f, td))
            if f > 0:
                b = m
            else:
                a = m
        A, B = lo + a * d, lo + b * d
        if self.project is not None:
            A, B = self.project(A), self.project(B)
        return A, B, info

    def run(self, lo, hi, *, rounds: int, on_round: Callable | None = None,
            verify_bracket: bool = True) -> list[dict]:
        """``rounds`` bisection + follow cycles from the bracket (lo, hi).

        ``verify_bracket``: the fates of ``lo`` (must relaminarize) and ``hi`` (must become
        turbulent) are computed first and a wrong bracket is refused (a bisection on a segment
        whose two ends share a fate converges to an end point, not to the edge).
        The fates are kept in ``bracket_check``."""
        rounds = check_count(rounds, "rounds", "EdgeTracker.run")
        if verify_bracket:
            X0 = lo if self.project is None else self.project(lo)
            X1 = hi if self.project is None else self.project(hi)
            (f0, t0), (f1, t1) = self.fate(X0), self.fate(X1)
            self.bracket_check = dict(lo=f0, t_lo=t0, hi=f1, t_hi=t1)
            if self.log:
                self.log(f"bracket check: lo {'laminar' if f0 < 0 else 'turbulent'} at t = {t0:g}, "
                         f"hi {'laminar' if f1 < 0 else 'turbulent'} at t = {t1:g}")
            if not (f0 < 0 < f1):
                hint = ("hi relaminarizes: take a larger perturbation for hi" if f1 < 0 else
                        "lo becomes turbulent: take a smaller perturbation for lo")
                raise OptionError(f"not a bracket of the edge: fate(lo) = {f0:+d} (t = {t0:g}), "
                                  f"fate(hi) = {f1:+d} (t = {t1:g}); need lo laminar, hi turbulent",
                                  why="a bisection on a segment whose two ends share a fate "
                                      "converges to an end point, not to the edge",
                                  fix=f"{hint}, or revisit the declared classify thresholds")
        for r in range(rounds):
            A, B, info = self.bisect(lo, hi)
            t0 = self.t
            prevA, prevB = A, B
            while True:
                dist = float(self.norm(A - B)) / max(float(self.norm(A - self.ref)), 1e-300)
                rec = dict(t=self.t, distance=dist, **self.measure(A))
                self.trajectory.append(rec)
                if dist > self.separation:      # separation only (a verdict needs t >= t_min)
                    break
                if self.t - t0 >= self.t_max:
                    raise UndecidedTrajectory(f"edge pair not separated after t_max = {self.t_max:g}")
                prevA, prevB = A, B
                A, B = self.advance(A, self.t_check), self.advance(B, self.t_check)
                self.t += self.t_check
            lo, hi = prevA, prevB
            self.rounds.append(dict(round=r, t_start=t0, t_end=self.t, trials=len(info),
                                    last=self.trajectory[-1]))
            if self.log:
                self.log(f"round {r}: t {t0:.1f} -> {self.t:.1f}, {len(info)} trials, "
                         f"edge: {self.trajectory[-1]}")
            if on_round is not None:
                on_round(self, lo, hi)
        return self.trajectory

    def explain(self) -> str:
        return (f"EdgeTracker: bisection to {self.tol:g} (relative λ), separation "
                f"{self.separation:g}, checks every {self.t_check:g}, no verdict before t_min = "
                f"{self.t_min:g}, trials refused after "
                f"t_max = {self.t_max:g}; {len(self.rounds)} rounds, {self.n_trials} trials, "
                f"edge time {self.t:g}")


class ThresholdBisection:
    """Bisection on one amplitude (e.g. the seed energy E0 of a minimal seed) between a
    value that relaminarizes and one that transitions: the transition threshold as a bracket
    ]lo, hi], never as a single guessed number.

    ``fate(a) -> -1`` (relaminarizes) / ``+1`` (transitions) / ``0`` (undecided) is the user's
    declared verdict for amplitude ``a`` (e.g. one forward DNS of the seed rescaled to E0 = a,
    classified with declared thresholds); an undecided run stops the bisection
    (:class:`UndecidedTrajectory`), it is never counted on either side. ``known`` gives fates
    already computed (e.g. by finished jobs): they are used first (no run is repeated, an end of
    [lo, hi] already settled by a known fate is not recomputed) and checked for consistency (a
    transitioning amplitude below a relaminarizing one is refused: the threshold would not be a
    single crossing on this family). Bisection is in the middle of the bracket (``scale=
    "linear"``) or of its logarithm (``"log"``) until (hi - lo) / hi <= ``rtol``.

        tb = ThresholdBisection(fate, rtol=1e-3, known={1.576778e-7: -1, 1.60e-7: +1})
        lo, hi = tb.run(1.5e-7, 1.7e-7); print(tb.explain())
    """

    def __init__(self, fate: Callable, *, rtol: float, scale: str = "linear",
                 known: dict | None = None, log=None) -> None:
        if not 1e-14 <= rtol < 1:
            raise OptionError(f"ThresholdBisection: rtol = {rtol!r} out of range",
                              why="rtol is the relative width of the final bracket; below 1e-14 "
                                  "the midpoint is no longer distinct from the ends in float64",
                              fix="pass rtol in [1e-14, 1), e.g. rtol=1e-3")
        if scale not in ("linear", "log"):
            raise OptionError(f"ThresholdBisection: scale = {scale!r}",
                              why="scale says where the bracket is cut",
                              fix="pass scale='linear' (arithmetic midpoint) or 'log' (geometric)")
        self.fate, self.rtol, self.scale, self.log = fate, float(rtol), scale, log
        self.fates: dict[float, int] = {}
        self.history: list[dict] = []          # runs made by this object (not the known fates)
        for a, f in (known or {}).items():
            self._record(float(a), f, source="known")

    @staticmethod
    def _as_fate(a: float, f, source: str) -> int:
        if f not in (-1, 1):                   # compared as given: 0.7 is not rounded to anything
            raise UndecidedTrajectory(f"amplitude {a:.8g}: fate {f!r} ({source}) is not -1 "
                                      "(relaminarizes) or +1 (transitions); an undecided run is not "
                                      "counted (run it longer, or declare other thresholds)")
        return int(f)

    def _record(self, a: float, f, source: str) -> int:
        f = self._as_fate(a, f, source)
        lam = [b for b, g in self.fates.items() if g < 0] + ([a] if f < 0 else [])
        tur = [b for b, g in self.fates.items() if g > 0] + ([a] if f > 0 else [])
        if lam and tur and min(tur) < max(lam):           # checked before anything is stored
            raise OptionError(
                f"ThresholdBisection: fates not monotone in the amplitude ({min(tur):.8g} "
                f"transitions, {max(lam):.8g} relaminarizes)",
                why="the threshold is not one crossing on this family of amplitudes",
                fix="check the verdict thresholds and the horizon of fate(), or bisect a narrower "
                    "interval")
        self.fates[a] = f
        return f

    def _fate(self, a: float) -> int:
        if a in self.fates:
            return self.fates[a]
        f = self._record(a, self.fate(a), source="fate()")
        self.history.append(dict(amplitude=a, fate=f))
        if self.log:
            self.log(f"threshold bisection: {a:.10g} -> {'transitions' if f > 0 else 'relaminarizes'}")
        return f

    def run(self, lo: float, hi: float) -> tuple[float, float]:
        """Bracket ]lo, hi] of the threshold in [lo, hi]: lo must relaminarize and hi transition.
        Known fates inside [lo, hi] narrow the bracket first; an end is run only when no known
        fate settles it (monotonicity: a relaminarizing amplitude >= lo settles lo)."""
        lo, hi = float(lo), float(hi)
        if not 0 < lo < hi:
            raise OptionError(f"ThresholdBisection.run: lo = {lo!r}, hi = {hi!r}",
                              why="the bracket needs 0 < lo < hi", fix="swap or correct the ends")
        lam = [a for a, f in self.fates.items() if f < 0 and lo <= a <= hi]
        tur = [a for a, f in self.fates.items() if f > 0 and lo <= a <= hi]
        if not lam and self._fate(lo) > 0:
            raise OptionError(f"ThresholdBisection.run: lo = {lo:.8g} transitions",
                              why="lo is not a lower bound of the threshold",
                              fix="pass a smaller lo (an amplitude whose run relaminarizes)")
        if not tur and self._fate(hi) < 0:
            raise OptionError(f"ThresholdBisection.run: hi = {hi:.8g} relaminarizes",
                              why="hi is not an upper bound of the threshold",
                              fix="pass a larger hi (an amplitude whose run transitions)")
        lo = max(a for a, f in self.fates.items() if f < 0 and lo <= a <= hi)
        hi = min(a for a, f in self.fates.items() if f > 0 and lo <= a <= hi)
        while (hi - lo) / hi > self.rtol:
            m = 0.5 * (lo + hi) if self.scale == "linear" else float(np.sqrt(lo * hi))
            if not lo < m < hi:                # float resolution reached: the bracket is final
                break
            if self._fate(m) > 0:
                hi = m
            else:
                lo = m
        self.bracket = (lo, hi)
        return lo, hi

    def explain(self) -> str:
        b = getattr(self, "bracket", None)
        head = ("ThresholdBisection: " + (f"threshold in ]{b[0]:.8g}, {b[1]:.8g}] "
                                          f"(relative width {(b[1] - b[0]) / b[1]:.1e})"
                                          if b else "not run yet")
                + f"; rtol {self.rtol:g}, {self.scale} cuts, {len(self.history)} new runs, "
                f"{len(self.fates) - len(self.history)} known fates")
        rows = [f"  {a:.10g}: {'transitions' if f > 0 else 'relaminarizes'}"
                for a, f in sorted(self.fates.items())]
        return "\n".join([head] + rows)
