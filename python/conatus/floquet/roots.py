"""Floquet exponents as certified roots.

A root is never reported as a bare number: it carries its residual, its convergence in the
discretisation parameters and a status ``certified`` / ``provisional`` / ``retracted``.
Im μ is defined modulo 1 for a period-2π base flow; the fold to [0, 1/2] (with the
conjugate symmetry of a real problem) is explicit and recorded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..errors import OptionError
from ..errors import check_count  # noqa: F401  (moved to conatus.errors, shared; re-exported here)

__all__ = ["fold_floquet", "CertifiedRoot", "SecantRoot", "SecantRun", "determinant_secant",
           "unconverged"]

# : Second point of the determinant secants (``secant_root``.find_root``,
#: secant stops on the caller's ``tol`` and reports a stall); declared in
#: ``conatus.tolerances.DECLARED`` (it was a literal signature default).
SECANT_STEP = 1e-4 * (1 + 1j)
#: Iteration cap of the secant searches (find_root, secant_root): a budget with a declared
#: default (rule); the accuracy tol stays required. A search that ends on the
#: cap or stalls raises ConvergenceError (carrying the last iterate as ``err.result``) unless
#: ``allow_unconverged=True`` is written (the reasons used to come back
#: in a second tuple element, so ``secant_root(...)[0]`` was a number with nothing attached).
SECANT_MAX_ITERATIONS = 50


#: Stall rule of the determinant secants of ``find_root`` (Hill, cascade): the search stops when
#: the step has not halved against the best step so far for this many consecutive iterations (a
#: budget, not an accuracy test: the stop is reported as a stall, never as convergence).
SECANT_STALL_STEPS = 6


@dataclass
class SecantRun:
    """Outcome of :func:`determinant_secant`: the last iterate and its determinant, the secant
    steps actually taken, the last step and why the search ended ("" when converged)."""

    mu: complex
    f: complex
    iterations: int
    last_step: float
    converged: bool
    reason: str


def determinant_secant(f, start: complex, step: complex, *, tol: float, max_iterations: int,
                       stall_after: int | None = None) -> SecantRun:
    """Secant iteration on ``f`` (a scaled determinant) from ``start`` and ``start + step``,
    (one copy: a zero secant denominator is not reported as "not converged"
    and the step not taken is not counted). Converged: the last step is below ``tol``. Otherwise the
    reason names the stop: equal determinants at the last two iterates (stalled, no step
    possible), no halving of the step for ``stall_after`` iterations (stalled), or the budget."""
    mus, fs = [start, start + step], [f(start), f(start + step)]
    best, stall, taken, reason = float("inf"), 0, 0, ""
    converged = False
    for _ in range(max_iterations):
        den = fs[-1] - fs[-2]
        if den == 0:
            reason = ("secant stalled, not converged: equal determinants at the last two iterates "
                      "(zero secant denominator, no step possible)")
            break
        mu_new = mus[-1] - fs[-1] * (mus[-1] - mus[-2]) / den
        d = abs(mu_new - mus[-1])
        mus.append(mu_new)
        fs.append(f(mu_new))
        taken += 1
        if d < tol:
            converged = True
            break
        if stall_after is not None:
            if d < 0.5 * best:
                best, stall = d, 0
            else:
                stall += 1
                if stall >= stall_after:
                    reason = (f"secant stalled, not converged: the step did not halve in "
                              f"{stall_after} consecutive iterations")
                    break
    else:
        reason = f"secant not converged: budget of {max_iterations} iterations exhausted"
    last = float(abs(mus[-1] - mus[-2]))
    if reason:
        reason += f" (last step {last:.1e}, tol {tol:g})"
    return SecantRun(complex(mus[-1]), fs[-1], taken, last, converged, reason)


@dataclass
class SecantRoot:
    """Result of :func:`conatus.floquet.hill_general.secant_root`.

    ``mu``: the last iterate; ``converged``: the last secant step is below ``tol``;
    ``residual``: the scaled determinant |det M(mu) / det M(start)| at ``mu`` (what the secant
    drives to zero); ``iterations``: secant steps taken (none for a stop on equal determinants); ``last_step``: |mu_k - mu_(k-1)|;
    ``reasons``: why it is not converged (empty when converged).

    v0.1 returned the tuple ``(mu, reasons)``; unpacking ``mu, why = secant_root(...)`` and
    indexing ``[0]``/``[1]`` keep working through v0.1 with a DeprecationWarning; use
    ``r.mu`` and ``r.reasons``."""

    mu: complex
    converged: bool
    residual: float
    iterations: int
    last_step: float
    tol: float
    reasons: list[str] = field(default_factory=list)

    def _legacy(self) -> tuple[complex, list[str]]:
        import warnings
        warnings.warn("secant_root(...) returns a SecantRoot (mu, converged, residual, iterations, "
                      "reasons); unpacking it as the v0.1 tuple (mu, reasons) or indexing it is "
                      "deprecated: use r.mu and r.reasons", DeprecationWarning, stacklevel=3)
        return (self.mu, self.reasons)

    def __iter__(self):
        return iter(self._legacy())

    def __getitem__(self, i):
        return self._legacy()[i]

    def __len__(self) -> int:
        return 2

    def explain(self) -> str:
        """The root, whether it converged (last step vs tol), iterations and scaled determinant."""
        state = "converged" if self.converged else "NOT converged"
        return (f"secant root mu = {self.mu.real:.12g}{self.mu.imag:+.12g}i: {state} (last step "
                f"{self.last_step:.1e}, tol {self.tol:g}) in {self.iterations} iteration(s); "
                f"scaled determinant |det M(mu)/det M(start)| = {self.residual:.2e}"
                + (f"; {'; '.join(self.reasons)}" if self.reasons else ""))

    __str__ = explain


def unconverged(result, where: str, *, max_iterations: int, tol: float):
    """The ConvergenceError of a root search that did not reach ``tol``, carrying ``result``
    (``err.result``): raised unless the caller wrote ``allow_unconverged=True``."""
    from ..errors import ConvergenceError
    err = ConvergenceError(
        f"{where}: not converged: {'; '.join(result.reasons) or 'stopped before tol'} "
        f"(max_iterations={max_iterations}, tol={tol:g}); last iterate mu = {complex(result.mu):.10g}",
        where=where,
        why="a root that did not reach the requested accuracy is not a root; returning it as a "
            "number would let it be used as one",
        fix="start closer (a better seed), raise max_iterations if the steps still decrease, or "
            "loosen tol above the round-off floor of the determinant; allow_unconverged=True "
            "returns the last iterate, flagged converged=False (err.result holds it)")
    err.result = result
    return err


def check_allow_unconverged(value, where: str) -> bool:
    """``allow_unconverged`` as a bool (anything else refused: it decides raise vs return)."""
    if not isinstance(value, bool):
        from ..errors import OptionError
        raise OptionError(f"{where}(allow_unconverged={value!r})",
                          why="allow_unconverged is True (return an unconverged root, flagged) or "
                              "False (raise ConvergenceError, the default)",
                          fix="allow_unconverged=True")
    return value




def fold_floquet(mu: complex) -> tuple[complex, str]:
    """Canonical representative of a Floquet exponent: Im μ in [0, 1/2].

    μ and μ + i k (k integer) are the same Floquet mode (harmonic relabelling); for a real
    operator μ and conj(μ) are both exponents. Returns the folded value and a description of
    what was done, so that the fold is never silent.
    """
    k = round(mu.imag)
    m = complex(mu.real, mu.imag - k)
    note = f"shifted Im by {-k:+d}" if k else "no shift"
    if m.imag < 0:
        m = m.conjugate()
        note += ", conjugated (real problem)"
    return m, note


#: Distance to the nearest integer under which a sampled winding number counts as that integer
#: in :meth:`CertifiedRoot.certify` (a contour sampled finely enough lands within 1e-3; 0.05
#: leaves room for a coarse contour while far from the half-integers of an under-sampled one).
WINDING_INTEGER_TOL = 0.05


@dataclass
class CertifiedRoot:
    """A Floquet exponent with the evidence that it is one."""

    mu: complex                              # as solved (unfolded)
    residual: float                          # ||M(μ) x|| / (||M||_F ||x||)
    iterations: int
    params: dict
    convergence: dict = field(default_factory=dict)   # resolution label -> μ
    status: str = "provisional"              # certified / provisional / retracted
    reasons: list[str] = field(default_factory=list)
    #: the secant reached its tol; False only with allow_unconverged=True
    converged: bool = True

    @property
    def folded(self) -> complex:
        return fold_floquet(self.mu)[0]

    def certify(self, *, residual_tol: float, convergence_tol: float,
                contrast: float | None = None, contrast_max: float | None = None,
                winding: float | None = None) -> CertifiedRoot:
        """Set the status from explicit tolerances (no default: the caller declares them).

        Certified requires: the secant converged (a root kept with ``allow_unconverged=True`` is
        never certified), the residual below ``residual_tol`` and a resolution study
        (>= 2 resolutions) whose spread is below ``convergence_tol``. Optional evidence, checked
        when given: ``contrast`` = σ_min/σ_2 of M(μ)
        must be below ``contrast_max``; ``winding`` = winding
        number of det M on a small circle around μ (``winding``) must be a positive integer to
        ``WINDING_INTEGER_TOL`` (``winding()`` returns the integer its adaptive walk certified, or
        raises). What was checked is listed in ``reasons`` / ``explain()``."""
        self.reasons = []
        ok = True
        mu = complex(self.mu)
        if not (np.isfinite(mu.real) and np.isfinite(mu.imag)):
            ok = False
            self.reasons.append(f"root μ = {mu} is not finite")
        if not self.converged:
            ok = False
            self.reasons.append("secant not converged (allow_unconverged=True)")
        if contrast is not None:
            if contrast_max is None:
                raise OptionError("certify(contrast=...) needs contrast_max",
                                  fix="contrast_max=1e-6 (σ_min/σ_2 at an isolated simple root)")
            if not contrast < contrast_max:
                ok = False
                self.reasons.append(f"singular-value contrast {contrast:.2e} >= {contrast_max:.1e}")
            else:
                self.reasons.append(f"checked: singular-value contrast {contrast:.2e} < "
                                    f"{contrast_max:.1e}")
        if winding is not None:
            if not (abs(winding - round(winding)) < WINDING_INTEGER_TOL and round(winding) >= 1):
                ok = False
                self.reasons.append(f"winding number {winding:.3f} is not a positive integer "
                                    f"(to {WINDING_INTEGER_TOL:g})")
            else:
                self.reasons.append(f"checked: winding number {winding:.3f}")
        if not self.residual < residual_tol:
            ok = False
            self.reasons.append(f"residual {self.residual:.2e} >= {residual_tol:.1e}")
        if len(self.convergence) < 2:
            ok = False
            self.reasons.append("no resolution study (need >= 2 resolutions)")
        else:
            vals = [complex(v) for v in self.convergence.values()]
            if not all(np.isfinite(v.real) and np.isfinite(v.imag) for v in vals):
                ok = False
                self.reasons.append("resolution study holds a non-finite value")
            else:
                spread = max(abs(v - vals[-1]) for v in vals[-2:])
                if not spread < convergence_tol * max(1.0, abs(vals[-1])):
                    ok = False
                    self.reasons.append(f"resolution spread {spread:.2e} >= {convergence_tol:.1e}")
                # the study must be about this root (audit of 0.1.2, F05: a root μ = 100 was
                # certified by a study converged at 0, and μ = NaN after it)
                gap = abs(mu - vals[-1])
                if not gap < convergence_tol * max(1.0, abs(vals[-1])):
                    ok = False
                    self.reasons.append(f"root μ differs from the finest resolution of the study "
                                        f"by {gap:.2e} >= {convergence_tol:.1e} (relative): the "
                                        "study is not evidence for this root")
        object.__setattr__(self, "status", "certified" if ok else "provisional")
        object.__setattr__(self, "_evidence", self._evidence_key() if ok else None)
        if ok and contrast is None and winding is None:
            self.reasons.append("certified on residual and resolution study only (no contrast "
                                "or winding evidence passed)")
        return self

    #: fields whose change after certify() withdraws the certification
    _EVIDENCE_FIELDS = ("mu", "residual", "iterations", "params", "convergence", "converged")

    def _evidence_key(self) -> tuple:
        return (complex(self.mu), float(self.residual), int(self.iterations), bool(self.converged),
                tuple((k, complex(v)) for k, v in self.convergence.items()))

    def __setattr__(self, name, value) -> None:
        # a certified root whose root or evidence is changed is no longer certified (audit of
        # 0.1.2, F05: the status survived replacing μ by NaN); certify() must run again
        object.__setattr__(self, name, value)
        if name in self._EVIDENCE_FIELDS:
            self._check_evidence()

    def _check_evidence(self) -> None:
        """Withdraw a certification whose root or evidence changed since certify()."""
        if self.__dict__.get("status") != "certified":
            return
        try:
            same = self._evidence_key() == self.__dict__.get("_evidence")
        except (TypeError, ValueError):
            same = False
        if not same:
            object.__setattr__(self, "status", "provisional")
            self.reasons.append("root or evidence changed after certify(): certify again")

    def retract(self, why: str) -> None:
        self.status = "retracted"
        self.reasons.append(why)

    def explain(self) -> str:
        self._check_evidence()          # a resolution study changed in place counts too
        f, note = fold_floquet(self.mu)
        head = "" if self.converged else "NOT CONVERGED (allow_unconverged=True): "
        conv = ", ".join(f"{k}: {v.real:.12f}{v.imag:+.12f}i" for k, v in self.convergence.items())
        return (head + f"Floquet exponent μ = {self.mu.real:.12f}{self.mu.imag:+.12f}i "
                f"(folded {f.real:.12f}{f.imag:+.12f}i, {note}); status {self.status}; "
                f"residual {self.residual:.2e}; {self.iterations} iterations; "
                f"resolution study [{conv}]" + (f"; reasons: {'; '.join(self.reasons)}" if self.reasons else ""))


def adaptive_winding(slogdet, center: complex, radius: float, n_points: int) -> float:
    """Winding number of det M on the circle |mu - center| = radius, by the adaptive walk of
    :mod:`conatus.contour.count` (principal phase increment <= pi/4 per step and per half step,
    log det resolved by every step; ``n_points`` initial points, refined where needed).

    ``slogdet(mu) -> (phase, log|det|)``. A sampled winding closed on its first point is an
    integer by construction, so "a non-integer value means under-resolved" never fired: a
    coarse sampling returned a clean but wrong integer (critic-science). The
    walk returns the integer it certified, or raises ConvergenceError (a root on the circle, a
    determinant that jumps)."""
    import math

    from ..contour.count import PHASE_STEP_MAX, MAX_EVALUATIONS, _walk
    from ..contour.operators import SingularPoint

    class _Circle:
        def point(self, s):
            import numpy as np
            return center + radius * np.exp(0.5j * math.pi * np.asarray(s))

        def describe(self) -> str:
            return f"the circle |mu - ({complex(center):.6g})| = {radius:g}"

    class _Det:
        def slogdet(self, z):
            ph, la = slogdet(z)
            if not (math.isfinite(la) and abs(ph) > 0):
                raise SingularPoint(z)
            return complex(ph), float(la)

    if not (isinstance(n_points, int) and n_points >= 4):
        raise OptionError(f"winding(n_points={n_points!r}): an integer >= 4 (initial points; "
                          "the walk refines where the phase needs it)")
    w = _walk(_Det(), _Circle(), PHASE_STEP_MAX, max(2, -(-n_points // 4) + 1), MAX_EVALUATIONS)
    return float(round(w["winding"]))
