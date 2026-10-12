"""Numerical-uncertainty report of a scalar result (spec §75–81): resolution and time-step
studies, observed orders, Richardson extrapolation, an error estimate with its assumptions.

::

    def nusselt(N, dt):
        sim = model.simulate(N=(2 * N, N), scheme="RK443", dt=dt)
        ...
        return float(nu)

    rep = cn.uncertainty(nusselt, N=[32, 48, 64], dt=[4e-3, 2e-3, 1e-3], scheme_order=3)
    print(rep.explain())          # value, estimate of the error, observed orders, verdicts
    rep.value, rep.error, rep.interval

What is computed (every number in the report, nothing hidden):

* **temporal study** at the finest N: q(dt_1), q(dt_2), q(dt_3) with a constant ratio r =
  dt_i / dt_{i+1}; observed order p = log(|q1 - q2| / |q2 - q3|) / log r; Richardson value
  q_R = q3 + (q3 - q2) / (r^p - 1); temporal error estimate e_t = F_s |q_R - q3|, F_s =
  ``SAFETY`` = 1.25 (the safety factor of Roache's grid convergence index for three-level
  studies; |q_R - q3| alone is an estimate, not a bound, and falls short of the true error about
  half the time); with the formal
  order ``scheme_order`` when given, p is compared to it: an observed order off by more than
  ``ORDER_TOL`` means the asymptotic range is not reached and the estimate is flagged);
* **spatial study** at the finest dt: q(N_1), q(N_2), q(N_3) (spectral: geometric decrease of the
  successive differences expected); spatial error estimate e_s = F_s |q(N_3) - q(N_2)|
  rho / (1 - rho), rho = |d_2| / |d_1| (d_i successive differences): the geometric tail of the
  next differences, times the same F_s; flagged when the differences do not decrease (rho >= 1) or
  sit at round-off;
* **total**: value q3 (finest N and dt; the Richardson value is reported, never substituted),
  error = e_t + e_s (a sum: independent sources, no quadrature assumed), interval = value +- error.

The finest-grid / finest-step value is the one reported; extrapolations are shown beside it. A
flagged study keeps its numbers and says why they should not be trusted.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..errors import InputTypeError, OptionError

__all__ = ["uncertainty", "UncertaintyReport", "ORDER_TOL", "ROUNDOFF_REL", "SAFETY"]

ORDER_TOL = 0.3          # |observed - formal| above this: not in the asymptotic range (flagged)
ROUNDOFF_REL = 1e-13     # successive differences below this (relative to |q|): round-off plateau
SAFETY = 1.25            # F_s of Roache's GCI (three-level studies); Roache, J. Fluids Eng. 116 (1994)
_DOC = "docs/site/guide/uncertainty.md"


@dataclass
class _Study:
    kind: str
    params: list
    values: list[float]
    order: float | None = None
    ratio: float | None = None
    estimate: float = float("nan")
    extrapolated: float | None = None
    flags: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        name = "dt" if self.kind == "temporal" else "N"
        out = [f"  {self.kind} study ({name} = {', '.join(f'{p:g}' for p in self.params)}): "
               + ", ".join(f"{v:.15g}" for v in self.values)]
        if self.order is not None:
            out.append(f"    observed order p = {self.order:.3f} (ratio r = {self.ratio:g})"
                       + (f", Richardson value {self.extrapolated:.15g}" if self.extrapolated is not None else ""))
        if self.kind == "spatial" and self.ratio is not None:
            out.append(f"    contraction of successive differences rho = {self.ratio:.3g}")
        out.append(f"    error estimate {self.estimate:.3e}")
        out += [f"    ** {f} **" for f in self.flags]
        return out


@dataclass
class UncertaintyReport:
    """Value, error estimate and the two studies behind them (see the module docstring)."""
    value: float
    error: float
    temporal: _Study
    spatial: _Study
    scheme_order: int | None
    name: str = "q"

    @property
    def interval(self) -> tuple[float, float]:
        return (self.value - self.error, self.value + self.error)

    @property
    def trusted(self) -> bool:
        """No study flagged (asymptotic range reached, differences above round-off)."""
        return not (self.temporal.flags or self.spatial.flags)

    def explain(self) -> str:
        lines = [f"Numerical uncertainty of {self.name}: {self.value:.15g} +- {self.error:.2e} "
                 f"(finest N and dt; error = temporal + spatial estimates)"
                 + ("" if self.trusted else "  [flagged: see below]")]
        lines += self.temporal.lines() + self.spatial.lines()
        if self.scheme_order is not None:
            lines.append(f"  formal order of the time scheme: {self.scheme_order} (tolerance {ORDER_TOL})")
        lines.append("  assumptions: smooth dependence on dt in the asymptotic range (Richardson); "
                     f"geometric (spectral) convergence in N; the two errors add; safety factor "
                     f"F_s = {SAFETY} on both (Roache's GCI)")
        return "\n".join(lines)

    __str__ = explain


def _temporal(values: list[float], dts: list[float], formal: int | None) -> _Study:
    st = _Study("temporal", dts, values)
    r1, r2 = dts[0] / dts[1], dts[1] / dts[2]
    if not math.isclose(r1, r2, rel_tol=1e-9):
        raise OptionError(f"uncertainty(dt={dts}): the ratio dt_i/dt_(i+1) must be constant "
                          f"({r1:g} vs {r2:g})", why="Richardson extrapolation with three steps "
                          "needs one refinement ratio", fix="dt=[4e-3, 2e-3, 1e-3]", doc=_DOC)
    st.ratio = r1
    d1, d2 = values[0] - values[1], values[1] - values[2]
    scale = max(abs(v) for v in values) or 1.0
    if abs(d2) <= ROUNDOFF_REL * scale or abs(d1) <= ROUNDOFF_REL * scale:
        st.flags.append("differences at round-off: temporal error below what this study resolves")
        st.estimate = abs(d2)
        return st
    if d1 * d2 <= 0:
        st.flags.append("non-monotone in dt: no observed order (not in the asymptotic range)")
        st.estimate = max(abs(d1), abs(d2))
        return st
    p = math.log(abs(d1) / abs(d2)) / math.log(r1)
    st.order = p
    if formal is not None and abs(p - formal) > ORDER_TOL:
        st.flags.append(f"observed order {p:.2f} differs from the formal order {formal} by more than "
                        f"{ORDER_TOL}: not in the asymptotic range, the estimate is unreliable")
    if p <= 0:
        st.flags.append("observed order <= 0: the values diverge as dt decreases")
        st.estimate = abs(d2)
        return st
    st.extrapolated = values[2] + (values[2] - values[1]) / (r1 ** p - 1)
    st.estimate = SAFETY * abs(st.extrapolated - values[2])
    return st


def _spatial(values: list[float], Ns: list[int]) -> _Study:
    st = _Study("spatial", Ns, values)
    d1, d2 = values[1] - values[0], values[2] - values[1]
    scale = max(abs(v) for v in values) or 1.0
    if abs(d2) <= ROUNDOFF_REL * scale:
        # resolved to the precision of the arithmetic: the last difference is the estimate (not
        # a flag: a converged spectral result is the expected case)
        st.estimate = abs(d2)
        st.ratio = abs(d2) / abs(d1) if d1 else 0.0
        return st
    rho = abs(d2) / abs(d1) if d1 else math.inf
    st.ratio = rho
    if rho >= 1:
        st.flags.append(f"successive differences do not decrease (rho = {rho:.3g}): not resolved")
        st.estimate = abs(d2)
        return st
    st.estimate = SAFETY * abs(d2) * rho / (1 - rho)  # geometric tail sum of the next differences
    return st


def uncertainty(quantity: Callable[..., Any], *, N: Sequence[int], dt: Sequence[float],
                scheme_order: int | None = None, name: str = "q") -> UncertaintyReport:
    """Run ``quantity(N=..., dt=...)`` (a number) on the declared studies and report its value
    with an error estimate: three time steps at the finest N, three resolutions at the finest
    dt (5 runs; the finest/finest run is shared). ``N``: increasing resolutions (an int or a
    tuple per run, passed as given; the last one is the finest); ``dt``: decreasing steps with a
    constant ratio; ``scheme_order``: formal order of the time scheme (checks the asymptotic
    range). Nothing is chosen for you: no default list, no adaptive refinement."""
    Ns, dts = list(N), [float(h) for h in dt]
    if len(Ns) != 3 or len(dts) != 3:
        raise InputTypeError(f"uncertainty(N={Ns}, dt={dts}): three resolutions and three time steps",
                             why="an observed order and a geometric contraction need three values each",
                             fix="N=[32, 48, 64], dt=[4e-3, 2e-3, 1e-3]", doc=_DOC)
    if not all(a > b > 0 for a, b in zip(dts, dts[1:], strict=False)):
        raise InputTypeError(f"uncertainty(dt={dts}): decreasing positive steps", doc=_DOC)

    def _sizes(n: Any) -> tuple:
        return tuple(n) if isinstance(n, (tuple, list)) else (n,)

    # the last N is reported as the finest; a list given in
    # the wrong order used to report the coarsest run as "the value" without a word
    sz = [_sizes(n) for n in Ns]
    if not (len({len(s) for s in sz}) == 1 and all(
            all(b >= a for a, b in zip(s0, s1, strict=True)) and s1 != s0
            for s0, s1 in zip(sz, sz[1:], strict=False))):
        raise InputTypeError(f"uncertainty(N={Ns}): increasing resolutions, the finest last",
                             why="the value reported is the one at the last N (and the last dt)",
                             fix="N=[32, 48, 64] (or tuples growing in every entry)", doc=_DOC)
    if not callable(quantity):
        raise InputTypeError("uncertainty(quantity, ...): a function quantity(N=..., dt=...)", doc=_DOC)

    def q(n, h) -> float:
        v = quantity(N=n, dt=h)
        try:
            v = float(v)
        except (TypeError, ValueError) as err:
            raise InputTypeError(f"uncertainty: quantity(N={n}, dt={h}) returned {v!r}, not a number",
                                 doc=_DOC) from err
        if not math.isfinite(v):
            raise InputTypeError(f"uncertainty: quantity(N={n}, dt={h}) = {v}", doc=_DOC)
        return v
    fine = q(Ns[-1], dts[-1])
    tvals = [q(Ns[-1], dts[0]), q(Ns[-1], dts[1]), fine]
    svals = [q(Ns[0], dts[-1]), q(Ns[1], dts[-1]), fine]
    tst, sst = _temporal(tvals, dts, scheme_order), _spatial(svals, Ns)
    return UncertaintyReport(fine, tst.estimate + sst.estimate, tst, sst, scheme_order, name)
