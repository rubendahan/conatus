"""Automatic verification report of an eigenmode / critical point.

Every check is measured and printed with its threshold; the report never says "correct".
A failed check is a step the user must investigate by hand, so ``manual_steps`` counts them.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np

from ..ir.tails import coefficient_tail

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
    """One measured verification check: its value, threshold, interpretation and pass/fail status."""
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
    """A collection of measured checks; ``manual_steps`` counts the checks that failed."""
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


def _lines(pevp) -> tuple[list | None, str]:
    """The coefficient lines of the pencil's vectors along the non-periodic axis, one per
    (unknown, component, Fourier index) and per piece of a compound basis, as (index array into
    the vector, symmetry-removed entries dropped; True for a piece of a compound basis); (None,
    why) when the plan has no such layout."""
    from ..solvers.spurious import tail_layout
    try:
        plan, group, reduced = pevp._layout_of
        layout, why = tail_layout(plan, group, reduced, require_tau=False)
    except Exception as exc:  # noqa: BLE001 -- the fallback below is said in the check's text
        return None, f"{type(exc).__name__}: {exc}"
    if layout is None:
        return None, why or "no non-periodic axis"
    lines = []
    for _name, idx, pieces in layout:
        for row in idx:
            for a, b in pieces:
                seg = row[a:b]
                lines.append((seg[seg >= 0], len(pieces) > 1))
    return lines, ""


def _compound(pevp) -> bool:
    """True when a basis of the pencil's domain is stored piece after piece (``sizes``); False for
    a pencil without a plan (its columns are measured per (unknown, component))."""
    lay = getattr(pevp, "_layout_of", None)
    if lay is None:
        return False
    try:
        bases = lay[0].domain.bases
    except Exception:  # noqa: BLE001 -- a plan whose bases cannot be read: pieces may exist
        return True
    return any(len(getattr(b, "sizes", ()) or ()) > 1 for b in bases if b is not None)


def _tail(pevp, x: np.ndarray, k: int = 4) -> float:
    """Largest coefficient tail of the mode (a-posteriori resolution along the non-periodic axis):
    for every coefficient line of every unknown (one per component, per Fourier index of the
    group and per piece of a compound basis), max |last k coefficients of the line| over the
    largest coefficient of that (unknown, component). NaN when nothing is measured, when a
    component is zero or non-finite, or when a piece holds k coefficients or fewer.

    Per (variable, component) (audit "assurances"): the last k entries of a vector unknown were
    the top coefficients of its last component only. Per piece and per Fourier index (0.3.1):
    a CompoundChebyshev unknown stores its pieces one after the other and a group holds its
    Fourier lines one after the other, so the last k entries were the top of the last piece of
    the last line only, and an unresolved first piece passed (eigenvalue 2-5 % off)."""
    cols = pevp.gm0.cols
    keep = [i for i, c in enumerate(cols) if c[2] != "tau"]
    key = {i: (cols[i][0], cols[i][1]) for i in keep}
    scale: dict = {}
    for i in keep:
        a = abs(x[i])
        scale[key[i]] = max(scale.get(key[i], 0.0), a) if np.isfinite(a) else float("nan")
    lines, _why = _lines(pevp)
    if lines is None:
        if _compound(pevp):
            # one block per component holds several pieces whose boundaries this plan does not
            # expose: a tail of the last k entries would measure the last piece only
            return float("nan")
        lines = [(np.array([i for i in keep if key[i] == kk]), False)
                 for kk in dict.fromkeys(key.values())]
    tails = []
    for seg, piece in lines:
        if seg.size <= k:
            if piece and seg.size:
                tails.append(float("nan"))    # a piece too short to show a tail: not measured
            continue                          # a line without length (a constant): no tail
        s = scale.get(key[int(seg[0])], float("nan"))
        if not (np.isfinite(s) and s > 0):
            # a zero or non-finite component has no tail to measure (review 0.1.1, sweep of F4:
            # 0 / 1e-300 = 0 was a PASS on a zero vector)
            tails.append(float("nan"))
            continue
        a = np.abs(x[seg])
        # the one coefficient-tail definition, rescaled from the line's largest coefficient to
        # the component's
        ok = np.all(np.isfinite(a))
        tails.append(coefficient_tail(a, k) * float(a.max()) / s if ok else float("nan"))
    return float(np.max(tails)) if tails else float("nan")     # np.max keeps a NaN


def _vector(v, n: int, name: str) -> np.ndarray:
    """``v`` as a 1-D vector of the pencil's size; a wrong size is refused."""
    from ..errors import InputTypeError
    a = np.asarray(v)
    if a.ndim != 1 or a.size != n:
        raise InputTypeError(f"verify(): mode.{name} has shape {a.shape}, the pencil has size {n}",
                             fix=f"a mode of this pencil (shape ({n},)), e.g. from P.track(...)")
    return a


def _backward_error(L, M, sigma: complex, v: np.ndarray, *, left: bool) -> float:
    """Relative backward error of the eigenpair (sigma, v) of L + sigma M (right, or left for
    ``left``): NaN for a zero or non-finite vector (not measurable, so never a PASS)."""
    nv = float(np.linalg.norm(v))
    if not (np.isfinite(nv) and nv > 0):
        return float("nan")
    A = L + sigma * M
    rv = v.conj() @ A if left else A @ v
    return float(np.linalg.norm(rv) / ((np.linalg.norm(L, 1) + abs(sigma) * np.linalg.norm(M, 1)) * nv))


def verify(pevp, mode, critical=None, *, rel_h: float = 1e-4) -> VerificationReport:
    """Residuals, rank, reconstruction, spectral tail, sensitivity Taylor test, critical point."""
    r = VerificationReport()
    r.checks.append(Check("pencil reconstruction", pevp.recon_error, RECONSTRUCTION_TOL,
                          "parametric L(p), M(p) vs a directly assembled plan"))
    r.checks.append(Check("pencil regular", 0.0 if "regular" in pevp.rank_check else np.inf, 0.5,
                          pevp.rank_check))
    # both residuals are measured here, on the pencil at the mode's values and on the vectors the
    # mode holds now (the right residual was the one stored by track(), so a
    # mode whose x was changed afterwards still printed the old 3e-19 and PASS)
    from ..errors import ParameterError
    try:
        L, M = pevp.matrices(**mode.values)
    except ParameterError as err:
        # the pencil refuses this point (entries discarded at rounding level could matter here):
        # nothing below can be measured, and the report says so instead of raising (0.1.2)
        r.checks.append(Check("pencil at the mode's values", float("nan"), RECONSTRUCTION_TOL,
                              f"not measured: the pencil was refused ({err.message})"))
        return r
    x = _vector(mode.x, L.shape[0], "x")
    right = _backward_error(L, M, mode.sigma, x, left=False)
    r.checks.append(Check("right residual", right,
                          RESIDUAL_TOL, "||(L + sigma M) x|| backward error, measured on the "
                          "current vector"))
    if mode.y is not None:
        y = _vector(mode.y, L.shape[0], "y")
        r.checks.append(Check("left residual", _backward_error(L, M, mode.sigma, y, left=True),
                              RESIDUAL_TOL, "||y^H (L + sigma M)|| backward error, measured on "
                              "the current vector"))
    r.checks.append(Check("spectral tail", _tail(pevp, x), TAIL_TOL,
                          "last 4 coefficients / max along the non-periodic axis, per component, "
                          "Fourier line and piece of each unknown (NaN: not measured). A small "
                          "tail is necessary for a resolved mode, not sufficient: it does not "
                          "measure the eigenvalue error, the Fourier directions or aliasing"))
    for p in pevp.params:
        if mode.y is None:
            break
        if not np.isfinite(right):             # a zero or non-finite x: nothing to track from
            r.checks.append(Check(f"d sigma/d {p.name} Taylor test", float("nan"), TAYLOR_TOL,
                                  "not measured: the mode vector is zero or non-finite"))
            continue
        v = mode.values[p.name]
        # relative step, absolute rel_h at v = 0 (h = 0 used to divide by zero: a parameter
        # declared at 0, e.g. a Richardson number, crashed the report)
        h = rel_h * abs(v) if v != 0 else rel_h
        try:
            sp_ = pevp.track(mode.sigma, mode.x, left=False, **{**mode.values, p.name: v + h}).sigma
            sm_ = pevp.track(mode.sigma, mode.x, left=False, **{**mode.values, p.name: v - h}).sigma
        except ParameterError as err:
            r.checks.append(Check(f"d sigma/d {p.name} Taylor test", float("nan"), TAYLOR_TOL,
                                  f"not measured: the pencil was refused at {p.name} +- h "
                                  f"({err.message})"))
            continue
        fd = (sp_ - sm_) / (2 * h)
        an = pevp.sensitivity(mode, p.name)
        r.checks.append(Check(f"d sigma/d {p.name} Taylor test", abs(fd - an) / max(abs(an), 1e-300), TAYLOR_TOL,
                              "exact sensitivity vs central difference of tracked sigma"))
    if critical is not None:
        r.checks.append(Check("critical point mode", 0.0 if critical.verified else np.inf, 0.5,
                              "recorded by critical(): its full QZ at the critical point found the "
                              "tracked mode the least stable (not repeated here)"))
        # the growth is measured on the eigenvalue of the critical mode as it is now, with the
        # convention critical() used (review 0.1.1, sweep of F4)
        g = getattr(critical, "growth_rule", None)
        if callable(g):
            cg, what = abs(float(g(critical.mode.sigma))), ("growth of the critical mode at the "
                                                            "neutral value, measured on its "
                                                            "current eigenvalue")
        else:
            cg, what = abs(critical.growth), "growth recorded by critical()"
        r.checks.append(Check("critical growth", cg, CRITICAL_GROWTH_TOL, what))
    return r
