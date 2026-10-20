"""Automatic verification report of an eigenmode / critical point.

Every check is measured and printed with its threshold; the report never says "correct".
A failed check is a step the user must investigate by hand, so ``manual_steps`` counts them.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np

__all__ = ["Check", "VerificationReport", "verify"]

# Pass thresholds of the report checks (each is printed beside its measured value; declared in
# ``conatus.tolerances.DECLARED``).
RECONSTRUCTION_TOL = 1e-10      # parametric L(p), M(p) vs a directly assembled plan
RESIDUAL_TOL = 1e-12            # right and left backward errors of the mode
TAIL_TOL = 1e-8                 # last 4 Chebyshev coefficients / max
TAYLOR_TOL = 1e-5               # exact sensitivity vs central difference
CRITICAL_GROWTH_TOL = 1e-10     # growth of the tracked mode at the neutral value


@dataclass
class Check:
    name: str
    value: float
    threshold: float
    what: str

    @property
    def status(self) -> str:
        return "PASS" if (np.isfinite(self.value) and self.value <= self.threshold) else "FAIL"

    def _repr_html_(self) -> str:       # Jupyter display (conatus.notebook)
        from ..notebook import check_html
        return check_html(self)

    def explain(self) -> str:
        return f"[{self.status}] {self.name}: {self.value:.2e} (threshold {self.threshold:.0e}) — {self.what}"

    def __str__(self) -> str:                  # print(check) shows the readable line
        return self.explain()


@dataclass
class VerificationReport:
    checks: list[Check] = dc_field(default_factory=list)

    @property
    def manual_steps(self) -> int:
        return sum(c.status == "FAIL" for c in self.checks)

    def _repr_html_(self) -> str:       # Jupyter display (conatus.notebook)
        from ..notebook import verification_html
        return verification_html(self)

    def explain(self) -> str:
        lines = [f"Verification report: {len(self.checks)} checks, {self.manual_steps} to investigate by hand"]
        lines += ["  " + c.explain() for c in self.checks]
        return "\n".join(lines)

    def __str__(self) -> str:                  # print(verify(...)) shows the report, repr stays raw
        return self.explain()


def _tail(pevp, x: np.ndarray, k: int = 4) -> float:
    """max over the components of every unknown of |last k coefficients| / max |coefficients|
    (a-posteriori resolution); NaN when no component has more than ``k`` coefficients.

    Per (variable, component) (audit "assurances"): the columns of a vector unknown
    hold its components one after the other, so the last k entries of the whole variable were the
    top coefficients of its LAST component only, and an unresolved first component passed; a
    pencil with no measurable component returned 0, a PASS on nothing measured."""
    cols = pevp.gm0.cols
    tails = []
    for key in dict.fromkeys((c[0], c[1]) for c in cols if c[2] != "tau"):
        idx = [i for i, c in enumerate(cols) if (c[0], c[1]) == key and c[2] != "tau"]
        if len(idx) <= k:
            continue
        a = np.abs(x[idx])
        tails.append(float(a[-k:].max() / max(a.max(), 1e-300)))
    return float(np.max(tails)) if tails else float("nan")     # np.max keeps a NaN


def verify(pevp, mode, critical=None, *, rel_h: float = 1e-4) -> VerificationReport:
    """Residuals, rank, reconstruction, spectral tail, sensitivity Taylor test, critical point."""
    r = VerificationReport()
    r.checks.append(Check("pencil reconstruction", pevp.recon_error, RECONSTRUCTION_TOL,
                          "parametric L(p), M(p) vs a directly assembled plan"))
    r.checks.append(Check("pencil regular", 0.0 if "regular" in pevp.rank_check else np.inf, 0.5,
                          pevp.rank_check))
    r.checks.append(Check("right residual", mode.residual, RESIDUAL_TOL, "||(L + sigma M) x|| backward error"))
    if mode.y is not None:
        L, M = pevp.matrices(**mode.values)
        yr = float(np.linalg.norm(mode.y.conj() @ (L + mode.sigma * M))
                   / ((np.linalg.norm(L, 1) + abs(mode.sigma) * np.linalg.norm(M, 1)) * np.linalg.norm(mode.y)))
        r.checks.append(Check("left residual", yr, RESIDUAL_TOL, "||y^H (L + sigma M)|| backward error"))
    r.checks.append(Check("spectral tail", _tail(pevp, mode.x), TAIL_TOL,
                          "last 4 Chebyshev coefficients / max, per component of each unknown "
                          "(mode resolved in N; NaN: no component long enough, not measured)"))
    for p in pevp.params:
        if mode.y is None:
            break
        v = mode.values[p.name]
        # relative step, absolute rel_h at v = 0 (h = 0 used to divide by zero: a parameter
        # declared at 0, e.g. a Richardson number, crashed the report)
        h = rel_h * abs(v) if v != 0 else rel_h
        sp_ = pevp.track(mode.sigma, mode.x, left=False, **{**mode.values, p.name: v + h}).sigma
        sm_ = pevp.track(mode.sigma, mode.x, left=False, **{**mode.values, p.name: v - h}).sigma
        fd = (sp_ - sm_) / (2 * h)
        an = pevp.sensitivity(mode, p.name)
        r.checks.append(Check(f"d sigma/d {p.name} Taylor test", abs(fd - an) / max(abs(an), 1e-300), TAYLOR_TOL,
                              "exact sensitivity vs central difference of tracked sigma"))
    if critical is not None:
        r.checks.append(Check("critical point mode", 0.0 if critical.verified else np.inf, 0.5,
                              "full QZ at the critical point confirms the tracked mode is the least stable"))
        r.checks.append(Check("critical growth", abs(critical.growth), CRITICAL_GROWTH_TOL,
                              "growth of the tracked mode at the neutral value"))
    return r
