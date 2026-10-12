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

import numpy as np
import scipy.linalg as sla

from ..continuation.core import ContinuationSettings
from ..errors import ConvergenceError, OptionError
from .driver import ArclengthDriver, Curve, CurvePoint
from .system import DynamicalSystem
from .._deprecation import renames

__all__ = ["continue_equilibria", "newton_equilibrium", "generalized_eigenvalues", "INF_EIG"]

INF_EIG = 1e10     # |λ| above INF_EIG · (1 + ‖J‖) is an infinite eigenvalue of (J, M) (DAE rows)


def generalized_eigenvalues(J: np.ndarray, M, *, vectors: bool = False, left: bool = False):
    """Finite eigenvalues of J v = λ M v, sorted by decreasing real part (infinite ones, from
    singular rows of M, removed with the declared threshold INF_EIG)."""
    Md = np.asarray(M.toarray() if hasattr(M, "toarray") else M, float)
    if vectors:
        w, vl, vr = sla.eig(J, Md, left=True, right=True)
    else:
        w = sla.eigvals(J, Md)
    bound = INF_EIG * (1.0 + np.abs(J).max())
    keep = np.isfinite(w) & (np.abs(w) < bound)
    idx = np.flatnonzero(keep)
    idx = idx[np.argsort(-w[idx].real, kind="stable")]
    if vectors:
        return w[idx], vr[:, idx], vl[:, idx]
    return w[idx]


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
    ev = ev[:n_eigs]
    if ev.size < 2:
        return 1.0
    s = ev[:, None] + ev[None, :]
    iu = np.triu_indices(ev.size, 1)
    z = s[iu]
    val = np.prod(z / (1.0 + np.abs(z)))
    return float(val.real)


def _logdet_sign(A: np.ndarray) -> tuple[float, float]:
    sign, logdet = np.linalg.slogdet(A)
    return float(np.real(sign)), float(logdet)


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
        ev = generalized_eigenvalues(J, M)
        return {"eigenvalues": ev, "unstable": int(np.sum(ev.real > marginal_tol)),
                "parameters": q}

    ne = n if n_eigs is None else int(n_eigs)

    def t_fold(pt: CurvePoint) -> float:
        return float(pt.tangent[-1])

    def t_bp(pt: CurvePoint) -> float:
        A = np.vstack([Hz(pt.z), pt.tangent[None, :]])
        sg, ld = _logdet_sign(A)
        if "bp" not in ref:
            ref["bp"] = ld
        return sg * float(np.exp(np.clip(ld - ref["bp"], -300, 300)))

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
                                     " x max(1, |λ|) for the pair of sum closest to 0, else "
                                     "neutral saddle")
        if np.all(np.abs(pair.imag) > settings.hopf_imag_tol * max(1.0, np.abs(pair).max())):
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
                 f"stability: #Re λ > {marginal_tol:g} (generalized eigenvalues of (J, M), "
                 f"|λ| > {INF_EIG:g}(1 + max|J|) removed as infinite)",
                 f"tests: {', '.join(tests)}; Hopf test over {ne} eigenvalue(s)"]
    return cv
