"""Equilibrium branches G(x; p) = 0 with fold, branch-point and Hopf test functions.

Test functions (evaluated at every accepted point, located by secant):

* ``fold``: the parameter component t_p of the unit tangent (limit point);
* ``branch point``: det [G_x G_p ; t^T] (the bordered Jacobian stays regular at a fold and is
  singular at a branch point, where it changes sign); scaled by its value at the start point;
* ``hopf``: ψ_H = Π_{i<j} s(λ_i + λ_j), s(z) = z / (1 + |z|), over the finite generalized
  eigenvalues of (J, M) (the bialternate-product test written with eigenvalues; declared
  ``n_eigs``: the pairs are formed from the ``n_eigs`` eigenvalues of largest real part). A zero is
  a Hopf point if the pair summing to ~0 is complex, otherwise a *neutral saddle* (λ, −λ real),
  rejected with that reason (kept in ``Curve.rejected``).

Every point stores its eigenvalues (sorted by decreasing real part) and the number of unstable ones.
"""
from __future__ import annotations

from collections.abc import Mapping

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from ..continuation.core import ContinuationSettings
from ..errors import ConvergenceError, OptionError
from .driver import ArclengthDriver, Curve, CurvePoint, factor_test
from .system import DynamicalSystem
from .._deprecation import renames

#: Two roots closer than this, relative to their size, count as a conjugate pair in an event check.
_CONJUGATE_TOL = 1e-6

__all__ = ["continue_equilibria", "newton_equilibrium", "generalized_eigenvalues",
           "pencil_spectrum", "PencilSpectrum", "INF_EIG"]

#: Former threshold of the finite/infinite split, kept for code that imports the name. It decides
#: nothing since 0.2.4: :func:`pencil_spectrum` uses the rank rule of dense QZ.
INF_EIG = 1e10


@dataclass
class PencilSpectrum:
    """Finite spectrum of J v = λ M v with what was left out of it.

    ``values`` (sorted by decreasing real part), ``right`` and ``left`` (unit vectors, or None)
    hold the finite eigenvalues. ``infinite`` counts the pairs removed as infinite.
    ``ambiguous_removed`` holds the moduli of the removed pairs that only the numerical rank of M
    calls infinite: a finite root that large cannot be told from an infinite one in double
    precision. ``ambiguous_kept`` holds the moduli of the kept pairs the same test cannot settle.
    ``undetermined`` counts the pairs with α and β both at rounding level (a singular pencil),
    which define no eigenvalue and are not in ``values``. ``ambiguous`` is the number of roots whose
    finite or infinite status is unresolved, so an unstable count from ``values`` is exact only
    when it is 0. ``note`` states the rule that was applied."""

    values: np.ndarray
    right: np.ndarray | None
    left: np.ndarray | None
    infinite: int
    ambiguous_removed: np.ndarray
    ambiguous_kept: np.ndarray
    undetermined: int
    note: str

    @property
    def ambiguous(self) -> int:
        """Number of roots whose finite or infinite status is unresolved."""
        return int(self.ambiguous_removed.size + self.ambiguous_kept.size + self.undetermined)


def pencil_spectrum(J: np.ndarray, M, *, vectors: bool = False) -> PencilSpectrum:
    """Finite eigenvalues of J v = λ M v, classified as dense QZ classifies them.

    Each row of the pencil is divided by its largest entry in J or M. A nonsingular M (numerical
    rank n of those rows, singular values above n eps) has no infinite eigenvalue, however large
    its finite roots are: J = diag(1, -1), M = diag(1e-12, 1) keeps 1e12. A singular M has at
    least n - rank(M) infinite ones, removed by the rule of ``solvers.evp._finite_eigenvalues``.
    A row of M at rounding level beside its row of J counts as a zero row (J = -I,
    M = diag(1, 1e-17)): its root is removed and reported in ``ambiguous_removed``."""
    # M as given (audit inputs: a complex M was cast to float, its imaginary part
    # dropped with a ComplexWarning); square matrices of one shape, finite (bare SciPy errors)
    Jd = np.asarray(J.toarray() if hasattr(J, "toarray") else J)
    Md = np.asarray(M.toarray() if hasattr(M, "toarray") else M)
    Md = Md.astype(complex if np.iscomplexobj(Md) else float)
    if Jd.ndim != 2 or Jd.shape != Md.shape or Jd.shape[0] != Jd.shape[1]:
        from ..errors import InputTypeError
        raise InputTypeError(f"generalized_eigenvalues: J {Jd.shape} and M {Md.shape} must be "
                             "square matrices of one shape", fix="M of the size of J (identity: "
                                                                 "np.eye(n))")
    if not (np.all(np.isfinite(Jd)) and np.all(np.isfinite(Md))):
        from ..errors import InputTypeError
        raise InputTypeError("generalized_eigenvalues: non-finite entries in J or M",
                             fix="check the state and parameters the Jacobian was built at")
    from ..solvers.evp import _finite_eigenvalues
    # J and M each at unit size first (QZ balances them separately): the rows then mix them in a
    # proportion that scaling J or M does not change, and λ scales exactly as nJ / nM
    nJ, nM = float(np.abs(Jd).max(initial=0.0)), float(np.abs(Md).max(initial=0.0))
    nJ, nM = nJ if nJ > 0 else 1.0, nM if nM > 0 else 1.0
    J1, M1 = Jd / nJ, Md / nM
    r = np.maximum(np.abs(J1).max(axis=1, initial=0.0), np.abs(M1).max(axis=1, initial=0.0))
    r = np.where(r > 0, r, 1.0)[:, None]
    Je, Me = J1 / r, M1 / r
    if vectors:
        ab, vl, vr = sla.eig(Je, Me, left=True, right=True, homogeneous_eigvals=True)
        # y^H (J - λ M) = 0  <=>  (D^-1 y)^H D (J - λ M) = 0, with D = 1/r
        vl = vl / r
        vl = vl / np.maximum(np.linalg.norm(vl, axis=0), np.finfo(float).tiny)
    else:
        ab = sla.eigvals(Je, Me, homogeneous_eigvals=True)
        vl = vr = None
    alpha, beta = ab[0], ab[1]
    finite, note, removed, kept = _finite_eigenvalues(alpha, beta, Je, Me, Md)
    n = alpha.size
    level = n * np.finfo(float).eps
    both = (np.abs(alpha) <= level * max(np.abs(Je).max(initial=0.0), np.finfo(float).tiny)) & \
        (np.abs(beta) <= level * max(np.abs(Me).max(initial=0.0), np.finfo(float).tiny))
    with np.errstate(divide="ignore", invalid="ignore"):
        w = alpha / beta * (nJ / nM)
    keep = finite & ~both & np.isfinite(w)
    idx = np.flatnonzero(keep)
    idx = idx[np.argsort(-w[idx].real, kind="stable")]
    undetermined = int(both.sum())
    if undetermined:
        note += (f"; {undetermined} pair(s) with alpha and beta both at rounding level define no "
                 "eigenvalue (singular pencil) and are left out")
    return PencilSpectrum(w[idx], vr[:, idx] if vectors else None, vl[:, idx] if vectors else None,
                          int(n - idx.size - undetermined), removed, kept, undetermined, note)


def generalized_eigenvalues(J: np.ndarray, M, *, vectors: bool = False, left: bool = False):
    """Finite eigenvalues of J v = λ M v, sorted by decreasing real part. The split between finite
    and infinite roots is that of :func:`pencil_spectrum`, which also reports the roots it could
    not settle: use it when a count of unstable roots is read from the result. Scaling J, M or
    both does not change the result: the eigenvalues 1e12 of (1e12 I, I) and of (I, 1e-12 I) are
    treated alike."""
    s = pencil_spectrum(J, M, vectors=vectors)
    return (s.values, s.right, s.left) if vectors else s.values


@renames("conatus.bifurcation.equilibria:newton_equilibrium")
def newton_equilibrium(system: DynamicalSystem, x0: np.ndarray, *, tol: float,
                       max_iterations: int = 50, par: Mapping[str, float] | None = None
                       ) -> tuple[np.ndarray, float, int]:
    """Plain Newton on G(x; p) = 0 at fixed p; returns (x, max|G|, iterations) or raises.
    ``tol`` (max|G|) is required; ``max_iterations`` is a budget with a declared default:
    reaching it raises ConvergenceError, never returns an unconverged point."""
    x = np.asarray(x0, float).copy()
    for k in range(max_iterations + 1):
        g = system.G(x, par)
        r = float(np.max(np.abs(g)))
        if r <= tol:
            return x, r, k
        if k == max_iterations:
            break
        x = x - np.linalg.solve(system.J(x, par), g)
    raise ConvergenceError(f"newton_equilibrium: max|G| = {r:.2e} > tol = {tol:g} after {max_iterations} its")


def _hopf_test(ev: np.ndarray, n_eigs: int) -> float:
    """Test function with the sign of the bialternate product Π_{i<j} (λ_i + λ_j) and the size of
    its smallest factor (:func:`factor_test`). The product itself shrinks with every stable
    eigenvalue added: below crossing_tol at an ordinary point from dimension 8, zero by underflow
    at 24."""
    ev = ev[:n_eigs]
    if ev.size < 2:
        return 1.0
    return factor_test((ev[:, None] + ev[None, :])[np.triu_indices(ev.size, 1)])


def continue_equilibria(system: DynamicalSystem, x0: np.ndarray, parameter: str,
                        settings: ContinuationSettings, *, direction: float = 1.0,
                        bounds: tuple[float, float] | None = None, n_eigs: int | None = None,
                        marginal_tol: float, tests: tuple[str, ...] = ("fold", "branch point", "hopf"),
                        par: Mapping[str, float] | None = None,
                        weights: np.ndarray | None = None,
                        tangent: np.ndarray | None = None) -> Curve:
    """Pseudo-arclength branch of equilibria in ``parameter`` from a converged x0.

    ``marginal_tol`` (required): eigenvalues with |Re λ| <= marginal_tol are marginal (not counted
    as unstable). ``n_eigs``: eigenvalues used by the Hopf test (default: all finite ones; for
    large systems give it explicitly). ``tests``: which test functions are active. ``tangent``:
    initial direction in z = (x, p) (default: ``direction`` along p), used by ``branch_switch``.
    """
    if not marginal_tol > 0:
        raise OptionError("continue_equilibria: marginal_tol > 0 is required (declared)")
    unknown = set(tests) - {"fold", "branch point", "hopf"}
    if unknown:
        raise OptionError(f"unknown test(s) {sorted(unknown)}", fix="fold, branch point, hopf")
    system._check_parameter(parameter, "continue_equilibria")
    base = system._full(par)
    n = system.n
    M = system.M

    def pars(z):
        q = dict(base)
        q[parameter] = float(z[-1])
        return q

    def H(z):
        return system.G(z[:-1], pars(z))

    def Hz(z):
        q = pars(z)
        return np.hstack([system.J(z[:-1], q), system.Gp(z[:-1], parameter, q)[:, None]])

    ref = {}

    def data(z):
        q = pars(z)
        J = system.J(z[:-1], q)
        spec = pencil_spectrum(J, M)
        ev = spec.values
        out = {"eigenvalues": ev, "unstable": int(np.sum(ev.real > marginal_tol)),
               "ambiguous": spec.ambiguous, "parameters": q}
        if spec.ambiguous and not ref.get("warned"):
            ref["warned"] = True
            from ..errors import CorrectnessWarning, _warn_user
            _warn_user(f"continue_equilibria: {spec.ambiguous} root(s) of (J, M) cannot be told "
                       "finite or infinite in double precision (M is numerically singular at "
                       "rounding level), so the unstable count is a count of the settled roots "
                       f"only (data['ambiguous'] on each point): {spec.note}", CorrectnessWarning)
        return out

    ne = n if n_eigs is None else int(n_eigs)

    def t_fold(pt: CurvePoint) -> float:
        return float(pt.tangent[-1])

    def t_bp(pt: CurvePoint) -> float:
        # the sign of det A, the size of its smallest singular value: the determinant carries
        # every other direction of the system, so its size says nothing about this event
        A = np.vstack([Hz(pt.z), pt.tangent[None, :]])
        sg = float(np.real(np.linalg.slogdet(A)[0]))
        smin = float(np.linalg.svd(A, compute_uv=False).min())
        return sg * smin / (1.0 + smin)

    def t_hopf(pt: CurvePoint) -> float:
        return _hopf_test(pt.data["eigenvalues"], ne)

    all_tests = {"fold": t_fold, "branch point": t_bp, "hopf": t_hopf}

    def classify(name, ev):
        if name != "hopf":
            return name, None
        lam = ev.point.data["eigenvalues"]
        # the pair whose sum is closest to 0
        s = np.abs(lam[:, None] + lam[None, :]) + np.diag(np.full(lam.size, np.inf))
        i, j = np.unravel_index(np.argmin(s), s.shape)
        pair = lam[[i, j]]
        # the declared threshold of the split, in the result
        ev.data["classification"] = (f"Hopf if |Im λ| > hopf_imag_tol = {settings.hopf_imag_tol:g}"
                                     " x max(|λ|, 1e-3 x the median |λ| of the spectrum) for the "
                                     "pair of sum closest to 0, else neutral saddle")
        from .._scales import spectral_floor
        ev.data["critical_distance"] = float(np.abs(pair.real).max())
        size = float(np.abs(pair).max())
        if abs(pair[0] - np.conj(pair[1])) > _CONJUGATE_TOL * max(size, np.finfo(float).tiny):
            # λ and -λ of a real matrix with their conjugates: the sum vanishes off the axis
            word = "saddle-focus" if np.abs(pair.imag).max() > _CONJUGATE_TOL * size else "saddle"
            return None, (f"neutral {word} (pair {pair[0]:.4g}, {pair[1]:.4g} has sum 0 and "
                          "is not conjugate)")
        if np.all(np.abs(pair.imag) > settings.hopf_imag_tol * max(np.abs(pair).max(),
                                                                     spectral_floor(lam))):
            ev.data.update(omega=float(abs(pair[0].imag)), pair=pair)
            ev.detail = f"λ = {pair[0].real:+.2e} ± {abs(pair[0].imag):.12g} i"
            return "hopf", None
        return None, f"neutral saddle (real pair {pair[0].real:.4g}, {pair[1].real:.4g})"

    # a fold is reported by the fold test; the bordered determinant does not change sign there
    drv = ArclengthDriver(H, Hz, settings, tests={k: all_tests[k] for k in tests}, data=data,
                          classify=classify, weights=None if weights is None else
                          np.concatenate([weights, [1.0]]))
    z0 = np.concatenate([np.asarray(x0, float), [base[parameter]]])
    if tangent is None:
        t0 = np.zeros(n + 1)
        t0[-1] = np.sign(direction) or 1.0
    else:
        t0 = np.asarray(tangent, float)
    cv = drv.run(z0, t0, kind="equilibrium", names={parameter: n},
                 bounds=None if bounds is None else {parameter: bounds})
    cv.context = {"system": system, "parameter": parameter, "base": base, "driver": drv,
                  "marginal_tol": marginal_tol}
    cv.notes += [f"equilibria of {system.origin} in {parameter}",
                 f"stability: #Re λ > {marginal_tol:g} (generalized eigenvalues of (J, M); roots "
                 "are infinite only when M is numerically singular, and data['ambiguous'] counts "
                 "the roots that test cannot settle)",
                 f"tests: {', '.join(tests)}; Hopf test over {ne} eigenvalue(s)"]
    return cv
