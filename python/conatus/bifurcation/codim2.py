"""Two-parameter continuation of fold and Hopf curves by minimally augmented systems (Govaerts,
*Numerical Methods for Bifurcations of Dynamical Equilibria*, SIAM 2000, ch. 4; as in MatCont).

**Fold curve.** z = (x, p1, p2); G(x; p) = 0 and g(x, p) = 0 with

    [ J   b ] [v]   [0]          [ Jᵀ  c ] [w]   [0]
    [ cᵀ  0 ] [g] = [1],         [ bᵀ  0 ] [h] = [1],

so g = 0 iff J is singular (b, c: bordering vectors, the last null vectors, refreshed after every
accepted point). g_z = −wᵀ J_z v with the exact IR forms: g_x = −(wᵀB(v, e_i))_i, g_pk = −wᵀ J_pk v.

**Hopf curve** (M = I). z = (x, κ, p1, p2), κ = ω²; the bordered matrix of J² + κI with two
columns B and rows Cᵀ gives a 2×2 G; the two entries of G used as conditions are chosen once at
the start (the pair giving the best-conditioned augmented Jacobian, declared). Derivatives:
(J²)_z v = J_z(J v) + J(J_z v), exact from B and J_p.

**Test functions** (located by secant):

* on a fold curve: ``cusp`` (wᵀB(v, v), normalized vectors), ``bt`` (wᵀ v: the zero eigenvalue
  becomes double), ``zh`` (Π s(λ_i + λ_j) over the eigenvalues other than the zero one; a zero with
  a complex pair = fold–Hopf, a real pair = neutral saddle, rejected);
* on a Hopf curve: ``bt`` (κ = ω² → 0), ``gh`` (Bautin / generalized Hopf: first Lyapunov
  coefficient l1 = 0, computed by ``normal_form`` at every point), ``zh`` (det J = 0),
  ``hh`` (Hopf–Hopf: Π s(λ_i + λ_j) over the eigenvalues other than ±iω; complex pair required).
"""
from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations

import numpy as np

from ..continuation.core import ContinuationSettings
from ..errors import ConvergenceError, OptionError, UnsupportedError
from .driver import ArclengthDriver, Curve
from .equilibria import generalized_eigenvalues
from .normal_form import normal_form
from .system import DynamicalSystem

# kernel tolerance of the first Lyapunov coefficient l1 computed along a Hopf curve (diagnostic
# column; declared default: conatus.tolerances.DECLARED); a failure is recorded in l1_error
L1_KERNEL_TOL = 1e-6
#: A zero-Hopf test zero on a Hopf curve with |κ| = ω² <= BT_KAPPA_TOL x max(1, |λ|max²) is the
#: Bogdanov–Takens point (ω → 0), not a fold–Hopf point (classification; declared).
BT_KAPPA_TOL = 1e-6

__all__ = ["continue_codim2"]


def _s(x):
    return x / (1.0 + np.abs(x))


def _Bmat(system, x, v, par) -> np.ndarray:
    """Matrix of u -> B(x)[v, u] (columns B(v, e_i)); exact IR second derivative."""
    n = system.n
    out = np.empty((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        out[:, i] = np.real(system.B(x, v, e, par))
    return out


def continue_codim2(system: DynamicalSystem, x0: np.ndarray, kind: str, parameters: tuple[str, str],
                    settings: ContinuationSettings, *, par: Mapping[str, float] | None = None,
                    omega: float | None = None, direction: float = 1.0,
                    bounds: Mapping[str, tuple[float, float]] | None = None,
                    tests: tuple[str, ...] | None = None) -> Curve:
    """Continue a fold (``kind="fold"``) or Hopf (``kind="hopf"``) point in two parameters.

    x0: the located point (event of ``continue_equilibria``); ``par`` its parameter values;
    ``omega``: Hopf frequency (default: pair closest to the axis). ``direction`` orients the start
    along +p2. ``tests`` default: all tests of the curve kind.
    """
    if kind not in ("fold", "hopf"):
        raise OptionError(f"continue_codim2: kind must be 'fold' or 'hopf', got {kind!r}")
    p1, p2 = parameters
    for pk in (p1, p2):
        system._check_parameter(pk, "continue_codim2")
    if p1 == p2:
        raise OptionError(f"continue_codim2: the two parameters are the same ({p1!r})",
                          fix="parameters=(p1, p2) with two different parameters of the system")
    base = system._full(par)
    x0 = np.asarray(x0, float)
    n = system.n
    M = system.M
    if kind == "hopf" and not system.identity_mass:
        raise UnsupportedError("continue_codim2(kind='hopf') needs M = I (J² + κI formulation)")
    Md = np.asarray(M.toarray() if hasattr(M, "toarray") else M, float)
    J0 = system.J(x0, base)
    lam, vr, vl = generalized_eigenvalues(J0, Md, vectors=True)
    state: dict = {}

    def pars(z):
        q = dict(base)
        q[p1], q[p2] = float(z[-2]), float(z[-1])
        return q

    if kind == "fold":
        i0 = int(np.argmin(np.abs(lam)))
        state["b"] = np.real(vl[:, i0]) / np.linalg.norm(np.real(vl[:, i0]))
        state["c"] = np.real(vr[:, i0]) / np.linalg.norm(np.real(vr[:, i0]))

        def bordered(z):
            x, q = z[:n], pars(z)
            J = system.J(x, q)
            K = np.block([[J, state["b"][:, None]], [state["c"][None, :], np.zeros((1, 1))]])
            rhs = np.zeros(n + 1)
            rhs[-1] = 1.0
            vg = np.linalg.solve(K, rhs)
            wh = np.linalg.solve(K.T, rhs)
            return J, vg[:n], vg[n], wh[:n]

        def H(z):
            x, q = z[:n], pars(z)
            _, _, g, _ = bordered(z)
            return np.concatenate([system.G(x, q), [g]])

        def Hz(z):
            x, q = z[:n], pars(z)
            J, v, g, w = bordered(z)
            top = np.hstack([J, system.Gp(x, p1, q)[:, None], system.Gp(x, p2, q)[:, None]])
            gx = -w @ _Bmat(system, x, v, q)
            gp = [-(w @ np.real(system.Jp(x, v, pk, q))) for pk in (p1, p2)]
            return np.vstack([top, np.concatenate([gx, gp])[None, :]])

        def on_accept(pt):
            _, v, _, w = bordered(pt.z)
            state["b"], state["c"] = w / np.linalg.norm(w), v / np.linalg.norm(v)

        def data(z):
            x, q = z[:n], pars(z)
            J, v, g, w = bordered(z)
            ev = generalized_eigenvalues(J, Md)
            vn, wn = v / np.linalg.norm(v), w / np.linalg.norm(w)
            return {"eigenvalues": ev, "v": vn, "w": wn,
                    "cusp": float(wn @ np.real(system.B(x, vn, vn, q))),
                    "bt": float(wn @ Md @ vn), "parameters": q}

        def t_zh(pt):
            ev = pt.data["eigenvalues"]
            ev = np.delete(ev, np.argmin(np.abs(ev)))
            if ev.size < 2:
                return 1.0
            s = ev[:, None] + ev[None, :]
            return float(np.real(np.prod(_s(s[np.triu_indices(ev.size, 1)]))))

        all_tests = {"cusp": lambda pt: pt.data["cusp"], "bt": lambda pt: pt.data["bt"], "zh": t_zh}
        z0 = np.concatenate([x0, [base[p1], base[p2]]])
        names = {p1: n, p2: n + 1}
    else:
        cand = np.flatnonzero(lam.imag > 0)
        if cand.size == 0:
            # was a bare "attempt to get argmin of an empty sequence" from numpy
            raise ConvergenceError("continue_codim2(kind='hopf'): no complex eigenvalue at x0 "
                                   "(not a Hopf point)",
                                   fix="start from a located Hopf event of continue_equilibria")
        if omega is not None:
            i0 = int(cand[np.argmin(np.abs(lam[cand].imag - omega))])
        else:
            i0 = int(cand[np.argmin(np.abs(lam[cand].real))])
        w0 = float(lam[i0].imag)
        qv, pv = vr[:, i0], vl[:, i0]
        state["B"] = np.linalg.qr(np.column_stack([pv.real, pv.imag]))[0]
        state["C"] = np.linalg.qr(np.column_stack([qv.real, qv.imag]))[0]
        I = np.eye(n)

        def bordered(z):
            x, kap, q = z[:n], z[n], pars(z)
            J = system.J(x, q)
            A = J @ J + kap * I
            K = np.block([[A, state["B"]], [state["C"].T, np.zeros((2, 2))]])
            rhs = np.zeros((n + 2, 2))
            rhs[n:] = np.eye(2)
            VG = np.linalg.solve(K, rhs)
            WH = np.linalg.solve(K.T, rhs)
            return J, A, VG[:n], VG[n:], WH[:n]

        def H_all(z):
            x, q = z[:n], pars(z)
            _, _, _, Gm, _ = bordered(z)
            return system.G(x, q), Gm

        def dG(z):
            """Derivatives of the four entries of G w.r.t. z (4 x (n+3))."""
            x, q = z[:n], pars(z)
            J, A, V, Gm, W = bordered(z)
            out = np.zeros((2, 2, n + 3))
            for j in range(2):
                v = V[:, j]
                Jv = J @ v
                Bv, BJv = _Bmat(system, x, v, q), _Bmat(system, x, Jv, q)
                dA_x = BJv + J @ Bv                       # columns: d/dx_i (J² v)
                dps = []
                for pk in (p1, p2):
                    dps.append(np.real(system.Jp(x, Jv, pk, q)) + J @ np.real(system.Jp(x, v, pk, q)))
                for i in range(2):
                    w = W[:, i]
                    out[i, j, :n] = -w @ dA_x
                    out[i, j, n] = -w @ v
                    out[i, j, n + 1] = -w @ dps[0]
                    out[i, j, n + 2] = -w @ dps[1]
            return out.reshape(4, n + 3), Gm.ravel()

        def top_rows(z):
            x, q = z[:n], pars(z)
            J = system.J(x, q)
            return np.hstack([J, np.zeros((n, 1)), system.Gp(x, p1, q)[:, None],
                              system.Gp(x, p2, q)[:, None]])

        z0 = np.concatenate([x0, [w0 ** 2, base[p1], base[p2]]])
        # choose the two entries of G once (best-conditioned augmented Jacobian), declared
        dg, _ = dG(z0)
        best, pick = -1.0, (0, 3)
        for a_, b_ in combinations(range(4), 2):
            A = np.vstack([top_rows(z0), dg[[a_, b_]]])
            smin = np.linalg.svd(A, compute_uv=False)[-1]
            if smin > best:
                best, pick = smin, (a_, b_)
        state["pick"] = pick

        def H(z):
            g, Gm = H_all(z)
            return np.concatenate([g, Gm.ravel()[list(pick)]])

        def Hz(z):
            dg, _ = dG(z)
            return np.vstack([top_rows(z), dg[list(pick)]])

        def on_accept(pt):
            x, kap, q = pt.z[:n], pt.z[n], pars(pt.z)
            J = system.J(x, q)
            A = J @ J + kap * I
            U, s, Vh = np.linalg.svd(A)
            state["B"], state["C"] = U[:, -2:], Vh[-2:].T

        def data(z):
            x, kap, q = z[:n], z[n], pars(z)
            J = system.J(x, q)
            ev = generalized_eigenvalues(J, Md)
            d = {"eigenvalues": ev, "parameters": q, "kappa": kap}
            if kap > 0:
                try:
                    nf = normal_form(system, x, kind="hopf", par=q, omega=np.sqrt(kap), kernel_tol=L1_KERNEL_TOL)
                    d["l1"] = nf.l1
                except Exception as e:  # noqa: BLE001 -- recorded in d['l1_error'], never hidden
                    d["l1"], d["l1_error"] = np.nan, str(e).splitlines()[0]
            else:
                d["l1"] = np.nan
            sg, ld = np.linalg.slogdet(J)
            d["logdet"] = (float(sg), float(ld))
            return d

        ref = {}

        def t_zh(pt):
            sg, ld = pt.data["logdet"]
            ref.setdefault("ld", ld)
            return sg * float(np.exp(np.clip(ld - ref["ld"], -300, 300)))

        def t_hh(pt):
            ev = pt.data["eigenvalues"]
            w = np.sqrt(max(pt.z[n], 0.0))
            drop = [int(np.argmin(np.abs(ev - 1j * w))), int(np.argmin(np.abs(ev + 1j * w)))]
            ev = np.delete(ev, drop)
            if ev.size < 2:
                return 1.0
            s = ev[:, None] + ev[None, :]
            return float(np.real(np.prod(_s(s[np.triu_indices(ev.size, 1)]))))

        all_tests = {"bt": lambda pt: float(pt.z[n]), "gh": lambda pt: pt.data["l1"],
                     "zh": t_zh, "hh": t_hh}
        names = {p1: n + 1, p2: n + 2, "kappa": n}

    if tests is None:
        tests = tuple(all_tests)
    unknown = set(tests) - set(all_tests)
    if unknown:
        raise OptionError(f"continue_codim2({kind}): unknown test(s) {sorted(unknown)}",
                          fix=f"available: {sorted(all_tests)}")

    def classify(name, ev):
        if name == "zh" and kind == "hopf":
            # det J = 0 also where ω → 0 (the Bogdanov–Takens point): not a fold–Hopf point
            kap = ev.point.z[n]
            if abs(kap) <= BT_KAPPA_TOL * max(1.0, np.abs(ev.point.data["eigenvalues"]).max() ** 2):
                return None, f"det J = 0 with κ = ω² = {kap:.1e}: the Bogdanov–Takens point"
        if name in ("zh", "hh") and kind == "fold" or name == "hh":
            lam_ = ev.point.data["eigenvalues"]
            if kind == "fold":
                lam_ = np.delete(lam_, np.argmin(np.abs(lam_)))
            else:
                w = np.sqrt(max(ev.point.z[n], 0.0))
                lam_ = np.delete(lam_, [int(np.argmin(np.abs(lam_ - 1j * w))),
                                        int(np.argmin(np.abs(lam_ + 1j * w)))])
            s = np.abs(lam_[:, None] + lam_[None, :]) + np.diag(np.full(lam_.size, np.inf))
            i, j = np.unravel_index(np.argmin(s), s.shape)
            pair = lam_[[i, j]]
            # same rule as the Hopf/real-crossing classification of equilibria and orbits (B8)
            if not np.all(np.abs(pair.imag) > settings.hopf_imag_tol * max(1.0, np.abs(pair).max())):
                return None, f"neutral saddle (real pair {pair[0].real:.4g}, {pair[1].real:.4g})"
            ev.detail = f"second pair ±{abs(pair[0].imag):.10g} i"
        if name == "gh" and ev.point is not None:
            ev.detail = f"l1 = {ev.point.data.get('l1', np.nan):.2e}"
        if name == "bt" and kind == "hopf":
            ev.detail = f"κ = ω² = {ev.point.z[n]:.2e}"
        return name, None

    drv = ArclengthDriver(H, Hz, settings, tests={k: all_tests[k] for k in tests}, data=data,
                          classify=classify, on_accept=on_accept)
    t0 = np.zeros(z0.size)
    t0[-1] = np.sign(direction) or 1.0
    cv = drv.run(z0, t0, kind=f"{kind} (codim 2)", names=names, bounds=bounds)
    cv.context = {"system": system, "parameters": parameters, "base": base, "state": state}
    cv.notes += [f"{kind} curve of {system.origin} in ({p1}, {p2}), minimally augmented system "
                 "(Govaerts 2000); bordering vectors refreshed after every accepted point",
                 f"tests: {', '.join(tests)}"]
    if kind == "hopf":
        cv.notes.append(f"Hopf conditions: entries {state['pick']} of the 2x2 bordered G (row-major), "
                        "chosen once at the start for the best-conditioned augmented Jacobian")
    return cv
