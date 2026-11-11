"""Numerical-uncertainty report of a scalar result: resolution and time-step
studies, observed orders, Richardson extrapolation, an error estimate with its assumptions.

::

    def nusselt(N, dt):
        sim = model.simulate(N=(2 * N, N), scheme="RK443", dt=dt)
        ...
        return float(nu)

    rep = cn.uncertainty(nusselt, N=[32, 40, 48, 56], dt=[4e-3, 2e-3, 1e-3], scheme_order=3)
    print(rep.explain())          # value, estimate of the error, observed orders, verdicts
    rep.value, rep.error, rep.interval

The report uses the following formulas and assumptions:

* **temporal study** at the finest N: q(dt_1), q(dt_2), q(dt_3) with a constant ratio r =
  dt_i / dt_{i+1}; observed order p = log(|q1 - q2| / |q2 - q3|) / log r; Richardson value
  q_R = q3 + (q3 - q2) / (r^p - 1); temporal error estimate e_t = F_s |q_R - q3|, F_s =
  ``SAFETY`` = 1.25 (the safety factor of Roache's grid convergence index for three-level
  studies; |q_R - q3| alone is an estimate, not a bound, and falls short of the true error about
  half the time); with the formal
  order ``scheme_order`` when given, p is compared to it: an observed order off by more than
  ``ORDER_TOL`` means the asymptotic range is not reached and the estimate is flagged; without
  ``scheme_order`` the study is flagged as unchecked);
* **spatial study** at the finest dt: q(N_1), ..., q(N_m), m >= 3, with successive differences
  d_i. Two laws are fitted to each three consecutive values at the actual N: geometric (spectral)
  q(N) = q_inf + C e^{-a N}, a from rho = |d_{i+1}| / |d_i| = e^{-a h_i} (1 - e^{-a h_{i+1}}) /
  (1 - e^{-a h_i}) (h_i the increments of N), and algebraic q(N) = q_inf + C N^{-p} (a corner, a
  discontinuity), the same relation with h_i = ln(N_{i+1} / N_i). The remaining error after the
  finest N is e_geo = F_s |d_last| / (e^{a h_last} - 1) (equal increments: F_s |d| rho / (1 - rho),
  the geometric tail of the next differences) or e_alg = F_s |d_last| / ((N_m / N_{m-1})^p - 1).
  Three resolutions fit either law: the larger estimate is given and the study is flagged (the
  law is not tested). With four or more, a law whose rate (a or p) drops by more than
  ``LAW_TOL`` = 10 % from one triple to the next is rejected: geometric when a holds and p grows
  (estimate e_geo), algebraic or not told apart otherwise (estimate max(e_geo, e_alg)), flagged
  when neither holds. Also flagged: differences that change sign (a real exponential is monotone
  in N), that do not decrease, that decrease more slowly than any power of N, tuple resolutions
  refined in different proportions, and a refinement that leaves the value unchanged (difference
  at round-off) before or after one that changes it: an even solution gives q(2m + 1) = q(2m),
  so such a zero is parity, not convergence;
* **total**: value q3 (finest N and dt; the Richardson value is reported, never substituted),
  error = e_t + e_s (a sum: independent sources, no quadrature assumed), interval = value +- error.

The finest-grid / finest-step value is the one reported; extrapolations are shown beside it. A
flagged study keeps its numbers and says why they should not be trusted.

What ``trusted`` does not establish: that the asymptotic behaviour seen over the given N and dt
continues beyond them (a law can change after the finest run); that the spatial and temporal
errors add without a cross term; anything about errors the two studies do not vary (solver
tolerances, a statistical error, a modelling error). The interval is an estimate with a safety
factor, not a bound.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..errors import InputTypeError, OptionError

__all__ = ["uncertainty", "UncertaintyReport", "ORDER_TOL", "ROUNDOFF_REL", "SAFETY", "LAW_TOL"]

ORDER_TOL = 0.3          # |observed - formal| above this: not in the asymptotic range (flagged)
# a decay rate (geometric law) or an order (algebraic law) of three consecutive resolutions lower
# than the previous one by more than this fraction rejects that law (hunt 5D, Q5)
LAW_TOL = 0.1
ROUNDOFF_REL = 1e-13     # successive differences below this (relative to |q|): round-off plateau
SAFETY = 1.25            # F_s of Roache's GCI (three-level studies); Roache, J. Fluids Eng. 116 (1994)
_DOC = "docs/site/guide/uncertainty.md"
# increments of tuple resolutions are "in the same proportion" to this relative rounding
_SPACING_RTOL = 1e-12
# decay rate per mode beyond which the spatial fit gives up (e^-1e6: no remaining error at all)
_DECAY_RATE_MAX = 1e6
# a last difference at round-off is a stall (parity) when the geometric model of the previous
# differences predicts one this many times above round-off
_STALL_MARGIN = 100.0


def _fmt(p: Any) -> str:
    """A resolution or a time step: a number, or a tuple of sizes ``(8, 8)`` (earlier versions: ``f"{p:g}"`` raised TypeError on the documented tuple resolutions)."""
    if isinstance(p, (tuple, list)):
        return "(" + ", ".join(f"{x:g}" for x in p) + ")"
    return f"{p:g}"


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
    # spatial study: the convergence law found ("geometric", "algebraic", ...; None: not tested
    # or not found), the estimate under each law, and the decay rates a / orders p of each
    # three consecutive resolutions
    law: str | None = None
    estimate_geometric: float | None = None
    estimate_algebraic: float | None = None
    rates: list | None = None
    orders: list | None = None

    def lines(self) -> list[str]:
        name = "dt" if self.kind == "temporal" else "N"
        out = [f"  {self.kind} study ({name} = {', '.join(_fmt(p) for p in self.params)}): "
               + ", ".join(f"{v:.15g}" for v in self.values)]
        if self.order is not None:
            out.append(f"    observed order p = {self.order:.3f} (ratio r = {self.ratio:g})"
                       + (f", Richardson value {self.extrapolated:.15g}" if self.extrapolated is not None else ""))
        if self.kind == "spatial" and self.ratio is not None:
            out.append(f"    contraction of the last two differences rho = {self.ratio:.3g}")
        if self.kind == "spatial" and self.estimate_geometric is not None:
            rates = ", ".join("-" if x is None else f"{x:.3g}" for x in self.rates or [])
            orders = ", ".join("-" if x is None else f"{x:.3g}" for x in self.orders or [])
            out.append(f"    geometric model: decay rates a = {rates}, estimate "
                       f"{self.estimate_geometric:.3e}; algebraic model: orders p = {orders}, "
                       f"estimate {self.estimate_algebraic:.3e}")
        if self.kind == "spatial":
            out.append(f"    convergence law: {self.law or 'not established'}")
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
        """No study flagged. Temporal: the observed order matches ``scheme_order`` within
        ``ORDER_TOL`` (without ``scheme_order`` the asymptotic range is not checked, so never
        trusted). Spatial: four or more resolutions follow one convergence law (geometric or
        algebraic, rates within ``LAW_TOL``), with differences of one sign, no unexplained
        round-off step (parity) and a single N to model. It does not establish that the law
        continues past the finest run, nor anything about errors the studies do not vary."""
        return not (self.temporal.flags or self.spatial.flags)

    def explain(self) -> str:
        lines = [f"Numerical uncertainty of {self.name}: {self.value:.15g} +- {self.error:.2e} "
                 f"(finest N and dt; error = temporal + spatial estimates)"
                 + ("" if self.trusted else "  [flagged: see below]")]
        lines += self.temporal.lines() + self.spatial.lines()
        if self.scheme_order is not None:
            lines.append(f"  formal order of the time scheme: {self.scheme_order} (tolerance {ORDER_TOL})")
        lines.append("  assumptions: smooth dependence on dt in the asymptotic range (Richardson); "
                     "one convergence law in N, geometric or algebraic, tested with four or more "
                     f"resolutions; the two errors add; safety factor F_s = {SAFETY} on both "
                     "(Roache's GCI); not established: the behaviour past the finest N and dt, "
                     "and errors the studies do not vary")
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
    if formal is None:
        st.flags.append("no scheme_order given: the asymptotic range is not checked (pass the "
                        "formal order of the time scheme)")
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


def _increments(Ns: list) -> tuple[int, list[float]] | None:
    """(k, increments of the resolutions along entry k), k the first entry that grows, or None
    when the entries of tuple resolutions are refined in different proportions (no single N to
    model). Every growing entry is a candidate for the algebraic model, see :func:`_entries`."""
    sz = [tuple(n) if isinstance(n, (tuple, list)) else (n,) for n in Ns]
    hs = [[b - a for a, b in zip(s0, s1, strict=True)] for s0, s1 in zip(sz, sz[1:], strict=False)]
    k = next((i for i, h in enumerate(hs[0]) if h > 0), None)
    if k is None or any(h[k] <= 0 for h in hs):
        return None
    for h1, h2 in zip(hs, hs[1:], strict=False):
        ratio = h2[k] / h1[k]
        if any(not math.isclose(b, ratio * a, rel_tol=_SPACING_RTOL, abs_tol=0)
               for a, b in zip(h1, h2, strict=True)):
            return None
    return k, [float(h[k]) for h in hs]


def _entries(Ns: list) -> list[list[float]]:
    """The values of every entry of the resolutions that grows at each step and is positive (the
    algebraic model q_inf + C N^-p is applied to each, the largest estimate kept)."""
    sz = [tuple(n) if isinstance(n, (tuple, list)) else (n,) for n in Ns]
    out = []
    for e in range(len(sz[0])):
        col = [float(s[e]) for s in sz]
        if col[0] > 0 and all(b > a for a, b in zip(col, col[1:], strict=False)):
            out.append(col)
    return out


def _decay_rate(rho: float, h1: float, h2: float) -> float | None:
    """a > 0 with |d2|/|d1| = e^{-a h1} (1 - e^{-a h2}) / (1 - e^{-a h1}) (model q(N) = q_inf +
    C e^{-a N} at the actual N; equal steps give rho = e^{-a h}); None if rho is not reachable
    (rho >= h2 / h1, the limit a -> 0: no geometric decrease). With h_i = ln(N_{i+1} / N_i) the
    same relation gives the order p of the algebraic model q_inf + C N^-p."""
    if not 0 < rho < h2 / h1:
        return None

    def f(a: float) -> float:
        return (math.exp(-a * h1) * -math.expm1(-a * h2) / -math.expm1(-a * h1)) - rho
    lo, hi = 1e-12 / max(h1, h2), 1.0
    while f(hi) > 0:                     # f decreases from h2/h1 (a -> 0) to 0 (a -> inf)
        hi *= 2.0
        if hi > _DECAY_RATE_MAX:
            return None
    from scipy.optimize import brentq
    return float(brentq(f, lo, hi, xtol=1e-14, rtol=1e-12))


def _next_difference(d0: float, d1: float, h: list[float]) -> float | None:
    """|d2| predicted by the geometric model from the differences d0, d1 over the increments
    h = (h0, h1, h2); None when they do not decrease geometrically."""
    a = _decay_rate(abs(d1) / abs(d0), h[0], h[1])
    if a is None:
        return None
    return abs(d1) * math.exp(-a * h[1]) * -math.expm1(-a * h[2]) / -math.expm1(-a * h[1])


def _spatial(values: list[float], Ns: list) -> _Study:
    st = _Study("spatial", Ns, values)
    d = [b - a for a, b in zip(values, values[1:], strict=False)]
    scale = max(abs(v) for v in values) or 1.0
    floor = ROUNDOFF_REL * scale
    small = [abs(x) <= floor for x in d]
    st.ratio = abs(d[-1]) / abs(d[-2]) if d[-2] else (0.0 if small[-1] else math.inf)
    if any(small[i] and not small[j] for i in range(len(d)) for j in range(i + 1, len(d))):
        # hunt 5D, Q5: an even solution gives q(2m + 1) = q(2m) exactly, so N = [23, 24, 25]
        # had a last difference 0 and the error was reported 1400x too small
        st.flags.append("a refinement left the value unchanged (difference at round-off) and a "
                        "later one changed it: the added modes are zero there (a symmetric "
                        "solution: odd and even N give the same result) or the quantity does "
                        "not see them; take resolutions of one parity")
        st.estimate = max(abs(x) for x in d)
        return st
    inc = _increments(Ns)
    if small[-1]:
        big = [abs(x) for x, s in zip(d, small, strict=True) if not s]
        trailing = len(d) - len(big)
        st.law = "resolved to round-off"
        st.estimate = max(abs(x) for x, s in zip(d, small, strict=True) if s)
        if not big or trailing >= 2:
            # resolved to the precision of the arithmetic (a converged spectral result is the
            # expected case): two refinements in a row, or every one, left the value unchanged
            return st
        k = len(big) - 1                 # index of the last difference above round-off
        pred = None if (k == 0 or inc is None) else _next_difference(d[k - 1], d[k], inc[1][k - 1:k + 2])
        if pred is None or pred > _STALL_MARGIN * floor:
            st.flags.append("the last refinement left the value unchanged (difference at round-off) "
                            f"after a difference of {big[-1]:.3g}"
                            + ("" if pred is None else f" (the geometric model predicts {pred:.3g})")
                            + ": convergence to round-off cannot be told from a step that adds "
                            "only zero modes (parity); add a resolution of the same parity")
            st.law = None
            st.estimate = big[-1]
        return st
    if inc is None:
        # the tail formula assumes the actual increments of N;
        # entries refined in different proportions leave no single N to model it with
        st.flags.append("tuple resolutions refined in different proportions: the geometric "
                        "model of the spatial error cannot be applied (refine every entry in the "
                        "same proportion)")
        st.estimate = abs(d[-1])
        return st
    _k, h = inc
    rhos = [abs(d1) / abs(d0) for d0, d1 in zip(d, d[1:], strict=False)]
    rates = [_decay_rate(r, h0, h1) for r, h0, h1 in zip(rhos, h, h[1:], strict=False)]
    a = rates[-1]
    if a is None:
        h1, h2 = h[-2], h[-1]
        st.flags.append(f"successive differences do not decrease (rho = {rhos[-1]:.3g}"
                        + (f", steps in N {h1:g} then {h2:g}" if h1 != h2 else "")
                        + "): not resolved")
        st.estimate = abs(d[-1])
        return st
    if len({x > 0 for x in d}) > 1:
        # a > 0 has successive differences of one sign
        st.flags.append("successive differences change sign: the monotone exponential model of "
                        "the spatial error does not apply")
        mags = [abs(x) for x in d]
        # 0.3.1 review (hunt 5G, T4): with differences that still shrink, the coarsest one is
        # orders of magnitude above the error (2e4 times for the heat equation of the guide);
        # the last two bound it. Differences that do not shrink keep the largest.
        shrinking = all(m1 < m0 for m0, m1 in zip(mags, mags[1:], strict=False))
        st.estimate = max(mags[-2:]) if shrinking else max(mags)
        return st
    # remaining error after the finest N under q(N) = q_inf + C e^{-a N} at the actual N
    # (N = [1, 4, 5] was modelled as equally spaced, error 13.6x too small);
    # equal steps give |d2| rho / (1 - rho), the geometric tail sum of the next differences
    st.estimate_geometric = SAFETY * abs(d[-1]) / math.expm1(a * h[-1])
    # hunt 5D, Q5: an algebraic decrease (a corner, a discontinuity) fitted by the geometric
    # model gave errors 2-6x too small, "trusted"; the algebraic tail with the same last ratio
    est_alg, orders = 0.0, None
    for col in _entries(Ns):
        lh = [math.log(b / a_) for a_, b in zip(col, col[1:], strict=False)]
        ps = [_decay_rate(r, h0, h1) for r, h0, h1 in zip(rhos, lh, lh[1:], strict=False)]
        if ps[-1] is None:
            est_alg = math.inf
            break
        est_alg = max(est_alg, SAFETY * abs(d[-1]) / math.expm1(ps[-1] * lh[-1]))
        orders = orders or ps
    st.estimate_algebraic = est_alg
    st.rates, st.orders = rates, orders
    if not math.isfinite(est_alg):
        st.flags.append("the differences decrease more slowly than any power of N: not resolved")
        st.estimate = st.estimate_geometric
        return st
    both = max(st.estimate_geometric, est_alg)
    if len(d) < 3:
        st.flags.append("three resolutions fit any convergence law (geometric e^-aN or algebraic "
                        "N^-p): the law is not tested and the larger of the two estimates is "
                        "given; pass a fourth resolution to test it")
        st.estimate = both
        return st
    geo_ok = all(r1 is not None and r0 is not None and r1 >= (1 - LAW_TOL) * r0
                 for r0, r1 in zip(rates, rates[1:], strict=False))
    alg_ok = orders is not None and all(p1 is not None and p0 is not None
                                        and p1 >= (1 - LAW_TOL) * p0
                                        for p0, p1 in zip(orders, orders[1:], strict=False))
    alg_flat = alg_ok and all(p1 <= (1 + LAW_TOL) * p0 for p0, p1 in zip(orders, orders[1:], strict=False))
    if geo_ok and not alg_flat:
        st.law, st.estimate = "geometric", st.estimate_geometric
    elif alg_ok:
        st.law = "algebraic" if not geo_ok else "geometric or algebraic (not told apart)"
        st.estimate = both
    else:
        st.flags.append("the successive differences follow neither a geometric nor an algebraic "
                        f"law (decay rates {', '.join(f'{x:.3g}' for x in rates if x is not None)}): "
                        "not in the asymptotic range")
        st.estimate = both
    return st


def uncertainty(quantity: Callable[..., Any], *, N: Sequence[int], dt: Sequence[float],
                scheme_order: int | None = None, name: str = "q") -> UncertaintyReport:
    """Run ``quantity(N=..., dt=...)`` (a number) on the declared studies and report its value
    with an error estimate: three time steps at the finest N, three or more resolutions at the
    finest dt (len(N) + 2 runs; the finest/finest run is shared). ``N``: increasing resolutions
    (an int or a tuple per run, passed as given; the last one is the finest); four or more are
    needed to test the convergence law, so three are always flagged; ``dt``: decreasing steps
    with a constant ratio; ``scheme_order``: formal order of the time scheme (checks the
    asymptotic range). Nothing is chosen for you: no default list, no adaptive refinement."""
    from ..errors import check_real
    if isinstance(N, str) or isinstance(dt, str) or not (hasattr(N, "__iter__")
                                                         and hasattr(dt, "__iter__")):
        raise InputTypeError(f"uncertainty(N={N!r}, dt={dt!r}): two lists of three values",
                             fix="N=[32, 48, 64], dt=[4e-3, 2e-3, 1e-3]", doc=_DOC)
    # each step a finite real number > 0, never parsed from text (audit inputs)
    Ns, dts = list(N), [check_real(h, "dt", "uncertainty()", positive=True) for h in dt]
    if len(Ns) < 3 or len(dts) != 3:
        raise InputTypeError(f"uncertainty(N={Ns}, dt={dts}): three or more resolutions and three "
                             "time steps",
                             why="an observed order and a geometric contraction need three values "
                                 "each; a fourth resolution tests the convergence law in N",
                             fix="N=[32, 40, 48, 56], dt=[4e-3, 2e-3, 1e-3]", doc=_DOC)
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
    if scheme_order is not None:
        # checked before the five runs (audit inputs: scheme_order="2" failed with a
        # bare TypeError after them, 0 or 2.5 were accepted as a formal order)
        from ..errors import check_count
        check_count(scheme_order, "scheme_order", "uncertainty()")

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
    svals = [q(n, dts[-1]) for n in Ns[:-1]] + [fine]
    tst, sst = _temporal(tvals, dts, scheme_order), _spatial(svals, Ns)
    return UncertaintyReport(fine, tst.estimate + sst.estimate, tst, sst, scheme_order, name)
