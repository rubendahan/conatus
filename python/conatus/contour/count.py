"""Eigenvalue counting in a region by the argument principle on ``det T(z)``.

For ``T(z)`` holomorphic on and inside a closed contour ``Gamma`` except at ``P`` poles of
``det T``, with no zero of ``det T`` on ``Gamma``, the winding number of ``det T`` along
``Gamma`` is

    N_Gamma = (1 / 2 pi) [arg det T]_Gamma = Z - P,

``Z`` and ``P`` counted with multiplicity (the argument principle). For a pencil ``L + z M`` the
determinant is a polynomial in ``z``: ``P = 0`` and ``N_Gamma`` is the number of eigenvalues
inside, multiplicities included.

The phase of ``det T`` is read at points of the boundary from an LU factorisation (unit phase and
log modulus, never the determinant itself: :mod:`conatus.contour.operators`). Between two points
only the principal value of the phase increment is known, so the walk is adaptive: a step is
accepted when

* its principal increment and those of its two halves are at most ``phase_step`` (the
  principal-increment bound: a full turn of the phase inside a step shows up in a half unless it
  happens within one quarter of the step, which the next condition catches through the modulus;
  an acceptance test on sampled values, not a proof; ``tests/contour/test_contour_count.py``
  places clusters of zeros 1e-3 .. 1e-12 off an edge, between the initial points: each is
  counted right or refused, never miscounted),
  and
* ``log det T`` is resolved by the step: the second difference of ``log|det T| + i arg det T``
  over the two halves is at most ``phase_step`` as well (a zero close to the step makes the
  modulus dip between the points),

otherwise it is bisected. A zero of ``det T`` on (or within rounding of) the boundary, or a
determinant that is not continuous (a branch-labelled product, a sign that depends on the order
in which some factor was computed), makes the bisection stall: the walk is then refused with the
place where it stalled, never reported. The winding of the accepted walk is an integer up to
rounding by construction (the last point is the first one, evaluated again); a non-integer
winding -- an evaluation of ``det T`` that does not return the same value at the same point --
is refused.

The count is the winding number of the *discretised* operator. It counts every eigenvalue of
the matrix pencil in the region, the spurious ones of a discretisation included.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import ConvergenceError, OptionError
from .operators import Operator, SingularPoint, as_operator
from .region import ContourBox

__all__ = ["count", "CountResult", "PHASE_STEP_MAX", "INITIAL_POINTS", "MAX_EVALUATIONS"]

_DOC = "docs/site/guide/dynamics/contour.md"
#: Principal-increment bound (radians): the largest phase increment of det T accepted over one
#: step of the boundary walk (and over each of its halves). pi/4 leaves a margin of 3 against
#: aliasing (a true increment is only known modulo 2 pi): halving it doubles the evaluations of a
#: well-resolved walk and changes no count of the test suite.
PHASE_STEP_MAX = math.pi / 4
#: Points per edge of the box before any refinement (corners included).
INITIAL_POINTS = 16
#: Budget of determinant evaluations of one walk; reaching it refuses the count (a zero on the
#: boundary or a discontinuous determinant), with the place where the walk stalled.
MAX_EVALUATIONS = 20000
_END = 4.0          # the perimeter parameter runs over [0, 4] (conatus.contour.region)


@dataclass
class CountResult:
    """Number of eigenvalues of ``T(z)`` in a region, by the argument principle (:func:`count`)."""

    count: int | None              # Z, the zeros inside (None when P is not known)
    winding: int                   # N_Gamma = Z - P
    poles: int | None              # P, by construction or declared; None = not declared
    region: ContourBox
    operator: str                  # what T is and how det T was evaluated
    evaluations: int
    smallest_step: float           # shortest accepted step, as a fraction of the perimeter
    smallest_at: complex           # where (the closest approach of a zero to the boundary)
    winding_error: float           # |winding - nearest integer| before rounding
    phase_step: float
    control: str = ""              # the control-box sentence, when one ran
    log: list[str] = field(default_factory=list)

    @property
    def count_method(self) -> str:
        """How the count was certified, in words (walk, evaluations, increment bound)."""
        return (f"argument principle: winding of det T along the boundary of the box "
                f"({self.evaluations} evaluations, principal increment <= {self.phase_step:.3g} "
                f"rad per step)")

    def __repr__(self) -> str:
        what = (f"Z = {self.count}" if self.count is not None else
                f"winding Z - P = {self.winding} (P not declared)")
        return f"CountResult({what} in {self.region.describe()}; .explain())"

    def explain(self) -> str:
        """The count, Z - P, how the boundary was walked and what was checked."""
        P = "not declared" if self.poles is None else str(self.poles)
        head = (f"eigenvalue count in {self.region.describe()}: "
                + (f"Z = {self.count} (zeros of det T inside, multiplicities included)"
                   if self.count is not None else
                   f"Z - P = {self.winding}; P (poles of det T inside) not declared, so Z is not "
                   "known: poles=0 declares T holomorphic in the box"))
        lines = [head,
                 f"  N_Gamma = Z - P = {self.winding} (winding number of det T along the boundary); "
                 f"P = {P}",
                 f"  operator: {self.operator}",
                 f"  walk: {self.evaluations} evaluations of det T, every step with a principal "
                 f"phase increment <= {self.phase_step:.3g} rad (and its two halves), log det T "
                 "resolved by every step",
                 f"  winding before rounding: off an integer by {self.winding_error:.1e} (a "
                 "non-integer winding is refused, never reported)",
                 f"  shortest step {self.smallest_step:.1e} of the perimeter, at z = "
                 f"{self.smallest_at:.6g} (where a zero comes closest to the boundary)"]
        if self.control:
            lines.append("  control: " + self.control)
        lines += ["  " + s for s in self.log]
        return "\n".join(lines)


def _check_options(phase_step: Any, initial_points: Any, max_evaluations: Any) -> tuple[float, int, int]:
    ok = (not isinstance(phase_step, bool) and isinstance(phase_step, numbers.Real)
          and 0 < phase_step <= math.pi / 2)
    if not ok:
        raise OptionError(f"phase_step={phase_step!r}: a bound in (0, pi/2] radians",
                          why="above pi/2 a full turn inside a step is no longer seen in its halves",
                          fix="phase_step=math.pi/4 (the declared default PHASE_STEP_MAX)", doc=_DOC)
    for name, v, lo in (("initial_points", initial_points, 2), ("max_evaluations", max_evaluations, 8)):
        if isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < lo:
            raise OptionError(f"{name}={v!r}: an integer >= {lo}", doc=_DOC)
    return float(phase_step), int(initial_points), int(max_evaluations)


def _walk(op: Operator, region: ContourBox, phase_step: float, initial_points: int,
          max_evaluations: int) -> dict:
    """Adaptive walk of the boundary (module docstring). Returns the winding and the statistics;
    raises ConvergenceError where the walk cannot be completed."""
    cache: dict[float, tuple[complex, float]] = {}

    def ev(s: float) -> tuple[complex, float]:
        if s in cache:
            return cache[s]
        if len(cache) >= max_evaluations:
            raise _Stall(s, "budget")
        z = complex(region.point(s))
        try:
            v = op.slogdet(z)
        except SingularPoint:
            raise _Stall(s, "singular") from None
        cache[s] = v
        return v

    def dlog(a: tuple[complex, float], b: tuple[complex, float]) -> complex:
        """log det(b) - log det(a), imaginary part as the principal increment."""
        return complex(b[1] - a[1], float(np.angle(b[0] * np.conj(a[0]))))

    knots = [k + j / (initial_points - 1) for k in range(4) for j in range(initial_points - 1)]
    knots.append(_END)
    total, smallest, smallest_at = 0.0, _END, 0.0
    # the closing point s = 4 is the first point evaluated again (the integer check)
    try:
        for s0, s1 in zip(knots[:-1], knots[1:], strict=True):
            stack = [(s0, s1)]
            while stack:
                a, b = stack.pop()
                if b - a < 1e-12 * _END:
                    raise _Stall(0.5 * (a + b), "step")
                fa = ev(a)
                fb = _closing(op, region, cache) if b == _END else ev(b)
                m = 0.5 * (a + b)
                fm = ev(m)
                d1, d2 = dlog(fa, fm), dlog(fm, fb)
                whole = abs(float(np.angle(fb[0] * np.conj(fa[0]))))
                if (whole <= phase_step and abs(d1.imag) <= phase_step
                        and abs(d2.imag) <= phase_step and abs(d2 - d1) <= phase_step):
                    total += d1.imag + d2.imag
                    if b - a < smallest:
                        smallest, smallest_at = b - a, m
                else:
                    stack += [(m, b), (a, m)]          # left half first
    except _Stall as st:
        z = complex(region.point(st.s))
        why = {"singular": f"det T is exactly zero at z = {z:.12g} on the boundary (an eigenvalue "
                           "on the contour)",
               "step": f"the walk could not resolve the phase near z = {z:.12g}: steps of "
                       f"{1e-12:.0e} of the perimeter still jump",
               "budget": f"the walk used its {max_evaluations} evaluations near z = {z:.12g}"}[st.kind]
        raise ConvergenceError(
            f"eigenvalue count refused in {region.describe()}: {why}",
            why="an eigenvalue on (or within rounding of) the boundary has no count, and a "
                "determinant that jumps (branch-labelled factors, a sign that follows a pivot "
                "or eigenvalue ordering) has no winding number; reporting an integer anyway "
                "would be a guess",
            fix="move or enlarge the box so that its edges stay clear of the eigenvalues; for a "
                "user-supplied determinant, make it a continuous function of z (the determinant "
                "of the whole matrix, with its permutation signs, not a product of factors "
                "labelled by their order); max_evaluations= raises the budget",
            doc=_DOC) from None
    w = total / (2 * math.pi)
    return {"winding": w, "evaluations": len(cache), "smallest": smallest / _END,
            "smallest_at": complex(region.point(smallest_at))}


def _closing(op: Operator, region: ContourBox, cache: dict) -> tuple[complex, float]:
    """det T at s = 4: the starting corner evaluated again (not taken from the cache)."""
    key = -1.0
    if key not in cache:
        try:
            cache[key] = op.slogdet(complex(region.point(0.0)))
        except SingularPoint:
            raise _Stall(0.0, "singular") from None
    return cache[key]


class _Stall(Exception):
    def __init__(self, s: float, kind: str) -> None:
        super().__init__(kind)
        self.s, self.kind = s, kind


def count(T: Any, region: ContourBox, *, poles: int | None = None, derivative: Any = None,
          control: tuple[ContourBox, int] | None = None, phase_step: float = PHASE_STEP_MAX,
          initial_points: int = INITIAL_POINTS, max_evaluations: int = MAX_EVALUATIONS) -> CountResult:
    """Number of eigenvalues of ``T(z)`` in ``region`` (a :func:`conatus.contour.box`) by the
    argument principle, certified as an integer (module docstring).

    ``T``: a pair ``(L, M)`` for the pencil ``L + z M``, a :class:`conatus.floquet.HillBlocks`,
    or a callable ``z -> matrix`` (or scalar). ``poles``: poles of ``det T`` inside the region,
    for a callable (0 if ``T`` is holomorphic there; a pencil and a linear Hill operator have none
    by construction); without it the result gives the winding ``Z - P`` and ``count=None``.
    ``control=(box2, n)``: count ``box2`` first and refuse unless it holds ``n`` (a box whose count
    is known checks the operator and the walk). ``phase_step``, ``initial_points``,
    ``max_evaluations``: the walk (declared defaults ``PHASE_STEP_MAX``, ``INITIAL_POINTS``,
    ``MAX_EVALUATIONS``). Raises ConvergenceError when the walk cannot be completed (an
    eigenvalue on the boundary, a discontinuous determinant) or does not close on an integer."""
    if not isinstance(region, ContourBox):
        raise OptionError(f"count(T, region): region must be cn.box(re=..., im=...), got "
                          f"{type(region).__name__}", doc=_DOC)
    phase_step, initial_points, max_evaluations = _check_options(phase_step, initial_points,
                                                                 max_evaluations)
    op = as_operator(T, poles=poles, derivative=derivative)
    ctrl = ""
    if control is not None:
        try:
            cbox, expected = control
        except (TypeError, ValueError):
            raise OptionError(f"control={control!r}: a pair (cn.box(...), known count)", doc=_DOC) from None
        c = count(op, cbox, phase_step=phase_step, initial_points=initial_points,
                  max_evaluations=max_evaluations)
        got = c.count if c.count is not None else c.winding
        if got != expected:
            raise ConvergenceError(
                f"control box {cbox.describe()}: counted {got}, expected {expected}",
                why="the control box has a known count; a different one means the operator or "
                    "its determinant is not what the count assumes",
                fix="check T (and poles=) on the control box before trusting any count", doc=_DOC)
        ctrl = f"box {cbox.describe()} counted {got}, as expected ({c.evaluations} evaluations)"
    w = _walk(op, region, phase_step, initial_points, max_evaluations)
    n = round(w["winding"])
    err = abs(w["winding"] - n)
    if err > 1e-6:
        raise ConvergenceError(
            f"eigenvalue count refused in {region.describe()}: the winding of det T is "
            f"{w['winding']:.6f}, not an integer",
            why="the walk ends on the point where it started, evaluated again: a non-integer "
                "winding means det T does not return the same value at the same point "
                "(a non-deterministic sign or phase), so no count can be read from it",
            fix="make the evaluation of T(z) deterministic; a user determinant must include "
                "every permutation sign", doc=_DOC)
    P = op.poles
    return CountResult(count=None if P is None else n + P, winding=n, poles=P, region=region,
                       operator=op.explain(), evaluations=w["evaluations"],
                       smallest_step=w["smallest"], smallest_at=w["smallest_at"],
                       winding_error=err, phase_step=phase_step, control=ctrl)
