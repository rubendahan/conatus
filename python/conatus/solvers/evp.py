"""Generalized eigenvalue problems on a discretization plan.

Convention (``conatus.ir.problem.EVP``): the eigenvalue sigma appears in the LHS,

    (L + sigma M) x = 0        i.e.   -L x = sigma M x,

group by group. Every mode carries its relative backward residual
``||(L + sigma M) x|| / ((||L|| + |sigma| ||M||) ||x||)``. Nothing is dropped
silently: infinite eigenvalues (M singular on tau and boundary rows) are
counted, and spurious finite modes of the tau method (Gardner et al. 1989 for
Orr–Sommerfeld) are only removed by an explicit resolution-drift test
(Boyd 2001), whose outcome is part of the result.
"""
from __future__ import annotations

import numbers
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np
import scipy.linalg as sl
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .._deprecation import renames
from ..tolerances import REQUIRED, require

__all__ = ["EVPResult", "solve_dense", "solve_sparse", "drift_filter", "sensitivity",
           "matrix_derivative", "biorthogonality", "check_drift_tol"]

INF_TOL = 1e3 * np.finfo(float).eps     # |beta| <= INF_TOL |alpha| (scaled) -> infinite eigenvalue
# ARPACK stopping tolerance of shift-invert (declared default): 0 = machine precision, the
# strictest; every returned mode carries its residual (conatus.tolerances.DECLARED).
ARNOLDI_TOL = 0.0
DRIFT_WHY = ("the drift tolerance decides which modes are kept: a mode is returned only if it "
             "reappears at the finer resolution within it (relative to |sigma|, floored at "
             "DRIFT_FLOOR times the median |sigma| of the finer spectrum); 1e-6 "
             "keeps resolved modes and removes spurious tau modes in the usual cases, a tighter "
             "value also removes marginally resolved physical modes")


#: Floor of the drift ratio, as a fraction of the median |sigma| of the finer spectrum (audit
#: invariance). The ratio was |dsigma| / max(|sigma|, 1): the 1 is in the unit of the
#: eigenvalue, so the same modes written in another time unit (sigma / 1e4) were judged on an
#: absolute drift and 9 under-resolved Orr-Sommerfeld modes passed instead of 1. The median of
#: the finer spectrum is in the unit of sigma (a handful of spurious modes do not move it); a mode
#: below DRIFT_FLOOR of it is "near zero" for this spectrum and its drift is measured against
#: that floor (a neutral mode computed as rounding noise at both resolutions stays kept).
DRIFT_FLOOR = 1e-3


def drift_ratios(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """|sigma_lo - nearest sigma_hi| / max(|sigma_lo|, DRIFT_FLOOR median |sigma_hi|) for each
    eigenvalue of ``lo``: invariant under a change of unit of sigma and under any scaling of the
    equations (audit invariance)."""
    from .._scales import spectral_scale
    lo, hi = np.asarray(lo), np.asarray(hi)
    # the scale of the finer spectrum: the median of its nonzero moduli (critic review: the plain
    # median is 0 when most eigenvalues are 0, and the rounding-noise neutral modes, compared with
    # a floor of 0, were dropped as drifting); the largest when fewer than half are nonzero
    floor = DRIFT_FLOOR * spectral_scale(hi) if hi.size else 0.0
    return np.array([np.min(np.abs(hi - s)) / max(abs(s), floor, np.finfo(float).tiny)
                     for s in lo])


def check_drift_tol(tol: Any, where: str, name: str = "drift_tol") -> float:
    """``drift_tol`` given and meaningful: a finite real
    number in (0, DRIFT_TOL_MAX]. Zero or less keeps nothing; above DRIFT_TOL_MAX (1e-2) most
    modes, spurious ones included, find a neighbour at the finer resolution, so the test no
    longer filters (``drift_tol=1.0`` was accepted and looked filtered)."""
    import numbers

    from ..tolerances import DRIFT_TOL_MAX
    tol = require(tol, name, where=where, suggest=f"{name}=1e-6", why=DRIFT_WHY)
    if (isinstance(tol, bool) or not isinstance(tol, numbers.Real) or not np.isfinite(tol)
            or not 0 < tol <= DRIFT_TOL_MAX):
        from ..errors import OptionError
        raise OptionError(f"{where}: {name}={tol!r} must be a number in (0, {DRIFT_TOL_MAX:g}]",
                          why="drift_tol is the relative distance within which a mode must be found "
                              "again at the finer resolution: 0 keeps nothing, and above "
                              f"{DRIFT_TOL_MAX:g} (conatus.tolerances.DRIFT_TOL_MAX) most modes, "
                              "spurious tau modes included, find a neighbour, so nothing is "
                              "filtered while the result reads as filtered",
                          fix=f"{name}=1e-6 (resolved modes kept, tau modes removed in the "
                              "usual cases)")
    return float(tol)


def _norm1(A) -> float:
    if sp.issparse(A):
        return float(abs(A).sum(axis=0).max()) if A.nnz else 0.0
    return float(np.linalg.norm(A, 1))


@dataclass
class EVPResult:
    """Eigenvalues sigma of (L + sigma M) x = 0 for one group, with residuals and a filter log."""

    values: np.ndarray
    vectors: np.ndarray | None               # right eigenvectors (columns), or None
    residuals: np.ndarray                     # relative backward residual per mode
    method: str
    label: str = ""
    n_infinite: int = 0
    left: np.ndarray | None = None            # left eigenvectors y: y^H (L + sigma M) = 0
    drift: np.ndarray | None = None           # drift ratio per mode (if a drift test ran)
    log: list[str] = dc_field(default_factory=list)
    # (plan, group, reduced, {variable name: unknown name}) of the group that produced the
    # vectors, set by conatus.eigenvalues(plan, ...): what mode(i) needs to unpack them
    context: Any = dc_field(default=None, repr=False, compare=False)
    # growth convention of the eigenvalue: its name, the rule
    # ('real', 'imag' or a callable of sigma; None = not declared) and the sentence saying why;
    # set by conatus.eigenvalues(plan, ...) from the problem, used by order() and explain()
    eigenvalue: str = "sigma"
    growth: Any = dc_field(default=None, repr=False, compare=False)
    growth_note: str = ""
    # spurious-mode filter: None when the problem has no tau (nothing
    # to filter), else a sentence: the drift test that ran, or that none ran
    filter_note: str | None = None
    # spectral-tail spurious-mode flag (conatus.solvers.spurious):
    # relative tail energy per mode and the flag (None: the test did not run), and its sentence
    tails: np.ndarray | None = dc_field(default=None, repr=False, compare=False)
    spurious: np.ndarray | None = dc_field(default=None, repr=False, compare=False)
    spurious_note: str = ""
    # resolution of the selected mode (
    # conatus.solvers.spurious.check_selected): "" or the sentence saying that order()[0] of a
    # result without a drift test has a tail above UNRESOLVED_TAIL_TOL
    unresolved_note: str = ""

    @property
    def flagged(self) -> np.ndarray:
        """Indices of the modes flagged spurious by the spectral-tail test (empty if none, or if
        the test did not run: see ``spurious_note``)."""
        if self.spurious is None:
            return np.zeros(0, dtype=int)
        return np.nonzero(self.spurious)[0]

    @property
    def unresolved(self) -> np.ndarray:
        """Indices of the modes that pass the spectral-tail flag but whose tail exceeds the
        stricter ``conatus.tolerances.UNRESOLVED_TAIL_TOL`` (under-resolved physical modes). Empty if the tail test did not run. Kept and ranked as any mode: the result
        says it (``unresolved_note``) when the selected mode ``order()[0]`` is one of them."""
        if self.tails is None:
            return np.zeros(0, dtype=int)
        from ..tolerances import UNRESOLVED_TAIL_TOL
        ok = np.ones(self.values.size, bool) if self.spurious is None else ~self.spurious
        return np.nonzero(ok & (self.tails > UNRESOLVED_TAIL_TOL))[0]

    def mode(self, i: int):
        """Eigenfunction ``i`` on the grid (:class:`conatus.solvers.evp_modes.EVPMode`): named
        unknowns (tau variables left out) on the physical grid, ``.grid("v")``, ``.coordinate("y")``,
        normalised by the declared rule ``evp_modes.NORMALIZATION``."""
        from .evp_modes import mode_of
        return mode_of(self, i)

    def __len__(self) -> int:
        return self.values.size

    def _growth_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        g = self.growth
        if g == "imag":
            return lambda v: v.imag
        if callable(g):
            return lambda v: np.array([float(g(s)) for s in v])
        return lambda v: v.real

    def growth_rule(self) -> str:
        """How :meth:`order` (``key='growth'``) ranks the modes, in words."""
        e = self.eigenvalue
        if self.growth is None:
            return (f"sorted by Re {e}, least stable first (growth convention of {e!r} not "
                    f"declared: m.eigenvalue({e!r}, growth='imag') if it is a phase speed)")
        what = {"real": f"Re {e}", "imag": f"Im {e}"}.get(self.growth, f"growth({e})") \
            if isinstance(self.growth, str) else f"growth({e}) (a callable)"
        return f"sorted by growth = {what}, least stable first"

    def order(self, key: str = "growth") -> np.ndarray:
        """Indices from the least to the most stable mode by the declared growth convention of
        the eigenvalue (``key='growth'``: Re s for exp(s t), Im c for a phase speed c; an undeclared convention ranks by Re, said by :meth:`growth_rule`), or by
        decreasing real part (``'real'``) or imaginary part (``'imag'``).

        Modes flagged spurious by the spectral-tail test (:attr:`flagged`) come
        after every mode that passes it (each part ranked by ``key``): ``order()[0]`` is the
        least stable mode that passes; the result keeps one index per mode."""
        if key == "growth":
            k = -self._growth_fn()(self.values)
        elif key == "real":
            k = -self.values.real
        elif key == "imag":
            k = -self.values.imag
        else:
            from ..errors import OptionError, did_you_mean
            raise OptionError(f"order(key={key!r})", fix="key='growth' (declared growth "
                              "convention), 'real' (decreasing real part) or 'imag' (decreasing "
                              "imaginary part)" + did_you_mean(str(key), ("growth", "real", "imag")))
        if self.spurious is None or not self.spurious.any():
            return np.argsort(k, kind="stable")
        return np.lexsort((k, self.spurious.astype(int)))

    def select(self, idx: Sequence[int] | np.ndarray, why: str) -> EVPResult:
        """The modes ``idx``, with ``why`` appended to the log (nothing is dropped silently)."""
        idx = np.asarray(idx, dtype=int)
        take = lambda a: None if a is None else (a[:, idx] if a.ndim == 2 else a[idx])  # noqa: E731
        return EVPResult(self.values[idx], take(self.vectors), self.residuals[idx], self.method,
                         self.label, self.n_infinite, take(self.left), take(self.drift),
                         self.log + [f"kept {idx.size} of {self.values.size} modes: {why}"],
                         self.context, self.eigenvalue, self.growth, self.growth_note,
                         self.filter_note, take(self.tails), take(self.spurious),
                         self.spurious_note, self.unresolved_note)

    def __repr__(self) -> str:
        # one line: the dataclass repr printed every eigenvector
        head = f"EVPResult([{self.label}] {self.values.size} finite modes"
        if self.values.size:
            i = int(self.order()[0])
            head += f", least stable {self.eigenvalue} = {self.values[i]:.10g}"
        if self.filter_note is not None and self.drift is None:
            head += ", no spurious-mode filter"
        if self.spurious is not None and self.spurious.any():
            head += f", {int(self.spurious.sum())} flagged spurious (ranked last)"
        if self.unresolved_note:
            head += ", SELECTED MODE UNDER-RESOLVED"
        return head + "; .explain() for the log)"

    def _repr_html_(self) -> str:       # Jupyter display (conatus.notebook)
        from ..notebook import evp_html
        return evp_html(self)

    def explain(self, n: int = 6, key: str = "growth", *, full: bool = False) -> str:
        """Counts (finite, infinite, filtered), the filters that removed modes, the growth
        convention and the ``n`` first modes with residuals. ``full=True``: the whole log
        (pencil regularity, infinite-eigenvalue test, solver tolerances)."""
        from ..errors import check_count
        n = check_count(n, "n", "EVPResult.explain()", minimum=0)  # audit inputs: n=-3 sliced
        kept =" kept by the filters below" if any(s.startswith("kept ") for s in self.log) else ""
        lines = [f"EVP result [{self.label}] ({self.method}): {self.values.size} finite modes{kept}, "
                 f"{self.n_infinite} infinite (M singular on tau/boundary rows)"]
        if self.filter_note is not None and self.drift is None:
            lines.append("  " + self.filter_note)
        if self.spurious_note:
            lines.append("  " + self.spurious_note)
        if self.unresolved_note:
            lines.append("  WARNING: " + self.unresolved_note)
        # short text: the filters, and every check that was skipped (a warning)
        log = self.log if full else [s for s in self.log if s.startswith("kept ") or "skipped" in s]
        lines += ["  " + s for s in log]
        if self.growth_note:
            lines.append("  growth: " + self.growth_note)
        rule = self.growth_rule() if key == "growth" else f"sorted by decreasing {key} part"
        if key == "growth" and self.values.size > 1:
            gv = self._growth_fn()(self.values)
            other = self.values.imag if self.growth != "imag" else self.values.real
            if np.ptp(gv) <= 1e-12 * max(1.0, float(np.abs(self.values).max())) and np.ptp(other) > 0:
                lines.append(f"  growth {'Im' if self.growth == 'imag' else 'Re'} {self.eigenvalue}"
                             " is the same for every mode: this ranking says nothing (a temporal "
                             f"eigenvalue? m.eigenvalue({self.eigenvalue!r}, growth='real'))")
        lines.append(f"  modes {rule}:")
        e = self.eigenvalue
        flagged = set(self.flagged.tolist())
        for i in [j for j in self.order(key) if j not in flagged][:n]:
            d = "" if self.drift is None else f", drift {self.drift[i]:.1e}"
            lines.append(f"  {e} = {self.values[i]: .12e}  residual {self.residuals[i]:.1e}{d}")
        if flagged:
            lines.append(f"  flagged spurious or unresolved (energy in the last coefficients; kept, ranked last; "
                         f"spurious='none' turns the flag off), {len(flagged)} mode(s):")
            for i in [j for j in self.order(key) if j in flagged][:n]:
                lines.append(f"  {e} = {self.values[i]: .12e}  residual {self.residuals[i]:.1e}, "
                             f"tail {self.tails[i]:.1e}  [index {i}]")
            if len(flagged) > n:
                lines.append(f"  ... {len(flagged) - n} more (res.flagged)")
        if not full and len(log) < len(self.log):
            lines.append(f"  ({len(self.log) - len(log)} more log line(s): pencil and "
                         "infinite-eigenvalue checks, solver settings: explain(full=True))")
        return "\n".join(lines)


def _residuals(L, M, w, V) -> np.ndarray:
    """Relative backward residual ||L x + w M x|| / ((||L|| + |w| ||M||) ||x||). When ||L|| +
    |w| ||M|| vanishes (w = 0 in a group where L = 0: the k = 0 mode of a periodic advection-
    diffusion operator) the scale is ||M|| instead: the residual of such a mode was 0/0 = NaN
   ; M = 0 is refused before any solve."""
    nL, nM = _norm1(L), _norm1(M)
    R = L @ V + (M @ V) * w[None, :]
    scale = nL + np.abs(w) * nM
    scale = np.where(scale > 0, scale, nM)
    return np.linalg.norm(R, axis=0) / (scale * np.linalg.norm(V, axis=0))


def _require_mass(M, L=None) -> None:
    """M = 0 means the eigenvalue never enters the problem: refuse rather than return []. With
    ``L``: both square, of one shape, and finite (audit inputs: a shape mismatch or a
    NaN entry was a bare ValueError of SciPy)."""
    from ..errors import EquationError, InputTypeError
    if L is not None:
        if L.ndim != 2 or L.shape != M.shape or L.shape[0] != L.shape[1]:
            raise InputTypeError(f"pencil (L, M): shapes {L.shape} and {M.shape}, two square "
                                 "matrices of one shape are needed",
                                 fix="L and M of the same group (plan.matrices(group))")
        data = [A.data if sp.issparse(A) else np.asarray(A) for A in (L, M)]
        if not all(np.all(np.isfinite(d)) for d in data):
            raise InputTypeError("pencil (L, M): non-finite entries (NaN or inf)",
                                 why="eigenvalues of a NaN matrix mean nothing",
                                 fix="check the parameters and coefficients the matrices were "
                                     "built with")
    if _norm1(M) == 0.0:
        raise EquationError("M is identically zero: the eigenvalue appears in no equation "
                            "(see EVP.check())")


def solve_dense(L, M, *, left: bool = False, label: str = "") -> EVPResult:
    """All eigenvalues by the QZ algorithm (scipy.linalg.eig), infinite ones counted and removed."""
    Ld = L.toarray() if sp.issparse(L) else np.asarray(L)
    Md = M.toarray() if sp.issparse(M) else np.asarray(M)
    _require_mass(Md, Ld)
    # Row equilibration (audit invariance): D (L + s M) has the eigenvalues and right
    # vectors of L + s M for any diagonal D, but QZ is backward stable in the norm of the whole
    # pencil, so a row 1e-8 times smaller than the others (an equation multiplied by a constant,
    # the 1/Re in front of D^4 next to O(1) boundary rows) lost 8 digits: the Orr-Sommerfeld
    # equation written times 1e-8 had no mode left after the drift test. Each row is divided by
    # its largest entry in L or M; the infinite/finite split below is then invariant too.
    r = np.maximum(np.abs(Ld).max(axis=1, initial=0.0), np.abs(Md).max(axis=1, initial=0.0))
    r = np.where(r > 0, r, 1.0)[:, None]
    Le, Me = Ld / r, Md / r
    out = sl.eig(-Le, Me, left=left, right=True, homogeneous_eigvals=True)
    (a, b), *vecs = out
    if left:
        VL, VR = vecs
        VL = VL / r            # y^H (L + s M) = 0  <=>  (D^-1 y)^H D (L + s M) = 0, D = 1/r
    else:
        VL, VR = None, vecs[0]
    scale = max(_norm1(Le), 1e-300) / max(_norm1(Me), 1e-300)
    finite = np.abs(b) * scale > INF_TOL * np.abs(a)
    w = a[finite] / b[finite]
    V = VR[:, finite]
    res = _residuals(Ld, Md, w, V)
    r = EVPResult(w, V, res, "dense QZ (scipy.linalg.eig)", label, int((~finite).sum()),
                  VL[:, finite] if left else None)
    r.log.append(f"{int((~finite).sum())} infinite eigenvalues removed (|beta| <= {INF_TOL:.1e} "
                 "|alpha| after scaling by ||L||/||M||, rows equilibrated)")
    return r


def check_target(target: Any) -> None:
    """The shift of shift-invert: a finite (real or complex) number, never text or a flag (audit
    inputs: target=nan factorised a NaN matrix, "0.2" was a bare UFuncTypeError)."""
    from ..errors import OptionError
    if isinstance(target, np.ndarray) and target.ndim == 0:      # one number (np.array(0.2j))
        target = target[()]
    ok = isinstance(target, numbers.Number) and not isinstance(target, (bool, np.bool_))
    if ok:
        z = complex(target)
        ok = np.isfinite(z.real) and np.isfinite(z.imag)
    if not ok:
        raise OptionError(f"shift-invert: target={target!r} must be a finite number",
                          fix="target=0.1+0.2j (the eigenvalues nearest it are returned)")


@renames("conatus.solvers.evp:solve_sparse")
def solve_sparse(L, M, target: complex, n_modes: int = 6, *, label: str = "",
                 tol: float = ARNOLDI_TOL) -> EVPResult:
    """``n_modes`` eigenvalues nearest ``target`` by shift-invert Arnoldi on (L + target M)^{-1} M
    (``k=`` is the deprecated former name of ``n_modes``: k is a wavenumber everywhere else).

    Works for singular, non-symmetric M (no M-inner product assumed): the
    standard problem  -(L + s M)^{-1} M x = nu x  has nu = 1 / (sigma - s).
    """
    from ..errors import OptionError, check_count
    L = sp.csc_matrix(L, dtype=complex)
    M = sp.csc_matrix(M, dtype=complex)
    _require_mass(M, L)
    n = L.shape[0]
    # target: a finite number; n_modes: an integer the Arnoldi method can return (audit inputs,
    # target=nan factorised a NaN matrix, "0.2" was a bare UFuncTypeError, and
    # n_modes was clipped to n - 2 while the log said the requested count)
    check_target(target)
    check_count(n_modes, "n_modes", "shift-invert")
    if n_modes > n - 2:
        raise OptionError(f"shift-invert: n_modes={n_modes} > {n - 2} (= size - 2, the most the "
                          f"Arnoldi method returns for a pencil of size {n})",
                          fix=f"n_modes <= {n - 2}, or method='dense' for every mode")
    lu = spla.splu(L + target * M)
    op = spla.LinearOperator((n, n), matvec=lambda x: -lu.solve(M @ x), dtype=complex)
    nu, V = spla.eigs(op, k=n_modes, which="LM", tol=tol)
    # nu = 0 <=> sigma = infinity (M singular): counted, never returned as a finite mode.
    scale = max(np.abs(nu).max(initial=0.0), 1e-300)
    keep = np.abs(nu) > INF_TOL * scale
    w = target + 1.0 / nu[keep]
    V = V[:, keep]
    res = _residuals(L, M, w, V)
    n_inf = int((~keep).sum())
    r = EVPResult(w, V, res, f"shift-invert Arnoldi (scipy eigs), target {target}", label, n_inf)
    r.log.append(f"ARPACK tol = {tol:g}" + (" (machine precision, declared default ARNOLDI_TOL)"
                                             if tol == 0 else " (given)"))
    r.log.append(f"{n_modes} modes requested nearest {target}; {n_inf} infinite eigenvalues "
                 f"(|nu| <= {INF_TOL:.1e} max|nu|) removed")
    return r


def drift_filter(lo: EVPResult, hi: EVPResult, tol: float = REQUIRED) -> EVPResult:
    """Keep the modes of ``lo`` that reappear in ``hi`` (a finer resolution) within ``tol``.

    ``tol`` has no default: it decides which modes are returned.

    Drift ratio = |sigma_lo - nearest sigma_hi| / max(|sigma_lo|, DRIFT_FLOOR median |sigma_hi|)
    (:func:`drift_ratios`: no unit of sigma enters). Spurious tau modes move by
    orders of magnitude between resolutions; resolved physical modes do not.
    """
    tol = check_drift_tol(tol, "drift_filter()", "tol")      # given and in range
    if hi.values.size == 0:
        raise ValueError("drift test needs a non-empty high-resolution spectrum")
    d = drift_ratios(lo.values, hi.values)
    keep = np.nonzero(d <= tol)[0]
    out = lo.select(keep, f"resolution drift <= {tol:.0e} against {hi.values.size} modes at the "
                          f"finer resolution; {lo.values.size - keep.size} unresolved or spurious "
                          "(tau) modes removed")
    out.drift = d[keep]
    return out


def biorthogonality(res: EVPResult, M, idx: Sequence[int]) -> np.ndarray:
    """Matrix y_i^H M x_j over the chosen modes (diagonal-dominant for distinct eigenvalues)."""
    if res.left is None:
        raise ValueError("left eigenvectors not computed (solve_dense(..., left=True))")
    Y = res.left[:, idx]
    X = res.vectors[:, idx]
    return Y.conj().T @ (M @ X)


def sensitivity(res: EVPResult, i: int, M, dL, dM=None) -> complex:
    """d sigma / dp for mode i from its left and right eigenvectors.

    From (L + sigma M) x = 0 and y^H (L + sigma M) = 0:
        d sigma/dp = - y^H (dL/dp + sigma dM/dp) x / (y^H M x).
    """
    if res.left is None:
        raise ValueError("left eigenvectors not computed (solve_dense(..., left=True))")
    x, y, s = res.vectors[:, i], res.left[:, i], res.values[i]
    num = y.conj() @ (dL @ x)
    if dM is not None:
        num = num + s * (y.conj() @ (dM @ x))
    return complex(-num / (y.conj() @ (M @ x)))


def matrix_derivative(build: Callable[[float], tuple], p: float, rel_h: float = 1e-3):
    """(dL/dp, dM/dp) by Richardson-extrapolated central differences (error O(h^4)).

    ``build(p) -> (L, M)`` must keep the sparsity layout fixed (same plan structure).
    """
    h = rel_h * max(abs(p), 1.0)

    def cd(hh):
        Lp, Mp = build(p + hh)
        Lm, Mm = build(p - hh)
        return (Lp - Lm) / (2 * hh), (Mp - Mm) / (2 * hh)

    L1, M1 = cd(h)
    L2, M2 = cd(h / 2)
    return (4 * L2 - L1) / 3, (4 * M2 - M1) / 3
