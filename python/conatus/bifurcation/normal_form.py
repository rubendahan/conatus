"""Normal forms on the centre manifold by projection (Kuznetsov 1999; *Elements of Applied
Bifurcation Theory*, 3rd ed. 2004, §3.5, §5.4, §8.7), in the generalized form M dx/dt = G(x; p).

The multilinear forms are the IR's symbolic Fréchet derivatives (``DynamicalSystem.B``, ``.C``);
the only linear algebra is on J = G_x and M (dense, bordered solves).

**Zero eigenvalue** (fold, transcritical, pitchfork). J q = 0, pᵀJ = 0, pᵀM q = 1, ‖q‖ = 1
(declared norm). Centre manifold x = x0 + a q + a²/2 h2 + ..., reduced equation

    da/dt = β Δp + μ Δp a + a2 a² + a3 a³,
    β = pᵀG_p,  a2 = ½ pᵀB(q, q),  J h2 = 2 a2 M q − B(q, q) with pᵀM h2 = 0,
    a3 = (1/6) pᵀ[C(q, q, q) + 3 B(q, h2)],  μ = pᵀ[J_p q + B(q, x_p)],  J x_p = −G_p (pᵀM x_p = 0).

β ≠ 0: fold; β = 0 (branch point on a branch, e.g. a trivial branch): transcritical (a2 ≠ 0) or
pitchfork (a2 = 0, e.g. by a declared symmetry). The zero pitchfork a2 is reported with its value.

**Hopf** (J q = iω M q, (J − iωM)ᴴ p = 0, pᴴM q = 1, ‖q‖ = 1):

    h11 = −J⁻¹B(q, q̄),  h20 = (2iω M − J)⁻¹ B(q, q),
    g21 = pᴴ[C(q, q, q̄) + B(q̄, h20) + 2 B(q, h11)],  c1 = g21 / 2,  l1 = Re c1 / ω

(Kuznetsov eq. (3.20)/(5.39); l1 < 0 supercritical, > 0 subcritical). With λ(p) = α(p) + iω(p),
λ' = pᴴ[J_p q + B(q, x_p)], the Landau/Stuart equation is dz/dt = λ' Δp z + c1 z|z|², so the
limit cycle has |z|² = −α' Δp / Re c1 and x(t) ≈ x0 + 2 Re(z q e^{iωt}). The magnitudes of c1, l1
depend on the normalization of q (declared: Euclidean norm of the coefficient vector unless a
weight is given); Re c1 / ‖q‖² and every prediction in physical variables do not.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np

from ..errors import ConvergenceError, OptionError, SymmetryError
from .equilibria import generalized_eigenvalues
from .system import DynamicalSystem
from .._repr import compact_repr

__all__ = ["normal_form", "ZeroEigenvalueNormalForm", "HopfNormalForm"]

#: Kernel parity under a declared symmetry: S q = ±q to PARITY_TOL x max|q|, else SymmetryError
#: (refusal; declared in ``conatus.tolerances.DECLARED``).
PARITY_TOL = 1e-8


def parity_under(Sv: np.ndarray, v: np.ndarray, tol: float) -> int | None:
    """+1 if S v = v, -1 if S v = -v, to ``tol`` x max|v| in every entry; None otherwise.

    The bound is absolute and only that: ``np.allclose`` with an atol alone
    kept its default rtol = 1e-5, so the declared PARITY_TOL (1e-8 here, 1e-7 in branching) was
    never the test; a kernel 1e-5 away from either parity, entry by entry, was accepted."""
    atol = tol * float(np.abs(v).max())
    if np.allclose(Sv, -v, rtol=0.0, atol=atol):
        return -1
    if np.allclose(Sv, v, rtol=0.0, atol=atol):
        return +1
    return None


def _dense(M) -> np.ndarray:
    return np.asarray(M.toarray() if hasattr(M, "toarray") else M)


def _bordered_solve(A: np.ndarray, qcol: np.ndarray, prow: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Solve A h = r with prow·h = 0 via [[A, qcol], [prow, 0]] (regular when A has a simple
    kernel spanned by q and prow·q ≠ 0). Returns h; the border unknown must vanish if r is in
    range(A), and its value is checked by the caller through the residual."""
    n = A.shape[0]
    K = np.zeros((n + 1, n + 1), dtype=np.result_type(A, qcol, prow, r))
    K[:n, :n], K[:n, n], K[n, :n] = A, qcol, prow
    sol = np.linalg.solve(K, np.concatenate([r, [0.0]]))
    return sol[:n]


@dataclass
class ZeroEigenvalueNormalForm:
    """Centre-manifold normal form at a simple zero eigenvalue (fold, transcritical, pitchfork)."""
    __repr__ = compact_repr   # one line, arrays by shape

    kind: str                    # "fold" | "transcritical" | "pitchfork"
    x0: np.ndarray
    params: dict
    parameter: str
    q: np.ndarray
    p: np.ndarray
    beta: float                  # constant term pᵀG_p
    mu: float                    # linear unfolding coefficient pᵀ(J_p q + B(q, x_p))
    a2: float
    a3: float
    h2: np.ndarray
    checks: dict = dc_field(default_factory=dict)
    notes: list = dc_field(default_factory=list)

    def amplitude(self, dp: float) -> float:
        """Steady amplitude a of the pitchfork branch at p0 + dp (a² = −μ dp / a3), NaN if none."""
        a2 = -self.mu * dp / self.a3
        return float(np.sqrt(a2)) if a2 >= 0 else float("nan")

    def explain(self) -> str:
        """Reduced equation, its coefficients, normalization and the residual checks."""
        eq = {"fold": "da/dt = β Δp + a2 a²",
              "transcritical": "da/dt = μ Δp a + a2 a²",
              "pitchfork": "da/dt = μ Δp a + a3 a³"}[self.kind]
        crit = ""
        if self.kind == "pitchfork":
            crit = " (supercritical)" if self.a3 < 0 else " (subcritical)"
        lines = [f"Normal form ({self.kind}{crit}) at {self.parameter} = {self.params[self.parameter]:.12g}: {eq}",
                 f"  β = {self.beta:.10g}, μ = {self.mu:.10g}, a2 = {self.a2:.10g}, a3 = {self.a3:.10g}",
                 "  x ≈ x0 + a q + a²/2 h2, ‖q‖ = 1 (declared norm), pᵀM q = 1"]
        lines += [f"  check {k}: {v:.1e}" for k, v in self.checks.items()]
        lines += ["  " + s for s in self.notes]
        return "\n".join(lines)


@dataclass
class HopfNormalForm:
    """Hopf normal form: frequency ω, c1, first Lyapunov coefficient l1 and the unfolding λ'."""
    __repr__ = compact_repr   # one line, arrays by shape

    x0: np.ndarray
    params: dict
    parameter: str | None
    omega: float
    q: np.ndarray
    p: np.ndarray
    c1: complex
    l1: float
    dlambda: complex | None      # λ'(p) = α' + i ω'
    h20: np.ndarray
    h11: np.ndarray
    checks: dict = dc_field(default_factory=dict)
    notes: list = dc_field(default_factory=list)

    @property
    def criticality(self) -> str:
        """"supercritical" (l1 < 0), "subcritical" (l1 > 0) or "degenerate (l1 = 0)"."""
        return "supercritical" if self.l1 < 0 else "subcritical" if self.l1 > 0 else "degenerate (l1 = 0)"

    def radius(self, dp: float) -> float:
        """|z| of the bifurcating cycle at p0 + dp (|z|² = −α' dp / Re c1), NaN if none."""
        if self.dlambda is None:
            raise OptionError("HopfNormalForm.radius needs the unfolding (normal_form(..., parameter=))")
        r2 = -self.dlambda.real * dp / self.c1.real
        return float(np.sqrt(r2)) if r2 >= 0 else float("nan")

    def period(self, dp: float) -> float:
        """Period of the cycle to first order: 2π / (ω + ω' dp + Im c1 |z|²)."""
        r2 = self.radius(dp) ** 2
        return float(2 * np.pi / (self.omega + self.dlambda.imag * dp + self.c1.imag * r2))

    def orbit(self, dp: float, tau: np.ndarray) -> np.ndarray:
        """First-order cycle x(τ) = x0 + 2 Re(z q e^{2πiτ}), τ ∈ [0, 1] (rows = times)."""
        z = self.radius(dp)
        return self.x0[None, :] + 2 * np.real(z * self.q[None, :] * np.exp(2j * np.pi * tau)[:, None])

    def explain(self) -> str:
        """ω, c1, l1 and criticality, normalization, unfolding λ' and the residual checks."""
        lines = [f"Hopf normal form at {self._ploc()}: ω = {self.omega:.12g}",
                 f"  c1 = {self.c1.real:.10g} {self.c1.imag:+.10g} i, first Lyapunov coefficient "
                 f"l1 = Re c1 / ω = {self.l1:.10g} ({self.criticality})",
                 "  normalization: ‖q‖ = 1 (declared norm), pᴴM q = 1 (Kuznetsov); "
                 "Re c1 / ‖q‖² is invariant"]
        if self.dlambda is not None:
            lines.append(f"  λ'({self.parameter}) = {self.dlambda.real:.10g} {self.dlambda.imag:+.10g} i "
                         f"-> Landau: dz/dt = λ' Δ{self.parameter} z + c1 z|z|²")
        lines += [f"  check {k}: {v:.1e}" for k, v in self.checks.items()]
        lines += ["  " + s for s in self.notes]
        return "\n".join(lines)

    def _ploc(self) -> str:
        return ", ".join(f"{k} = {v:.12g}" for k, v in self.params.items())


def _norm(v: np.ndarray, weight: np.ndarray | None) -> float:
    w = np.ones(v.size) if weight is None else weight
    return float(np.sqrt(np.sum(w * np.abs(v) ** 2)))


def normal_form(system: DynamicalSystem, x0: np.ndarray, *, kind: str,
                par: Mapping[str, float] | None = None, parameter: str | None = None,
                omega: float | None = None, weight: np.ndarray | None = None,
                symmetry: Any = None, kernel_tol: float = 1e-8) -> Any:
    """Normal form at the equilibrium x0 (a located event: fold, branch point or Hopf).

    ``kind``: "zero" (fold / transcritical / pitchfork, classified) or "hopf". ``parameter``: the
    unfolding parameter (needed for β, μ, λ'). ``omega``: the Hopf frequency guess (default: the
    critical pair closest to the imaginary axis). ``weight``: diagonal of the norm fixing ‖q‖ = 1
    (default Euclidean on the discrete vector, declared). ``symmetry``: optional linear map S
    (matrix or callable) with S G(x) = G(S x); the kernel parity S q = ±q is reported and a
    pitchfork is then *required* to have a2 = 0 up to ``kernel_tol`` (else refused).
    ``kernel_tol``: the critical eigenvalue must satisfy |λ| (zero) or |Re λ| (Hopf) <= kernel_tol·(1+‖J‖),
    and x0 must be an equilibrium, max|G(x0)| <= kernel_tol·max(1, ‖J‖)·max(1, ‖x0‖) (else refused).
    """
    if kind not in ("zero", "hopf"):
        raise OptionError(f"normal_form: kind must be 'zero' or 'hopf', got {kind!r}")
    q_par = system._full(par)
    x0 = np.asarray(x0, float)
    J = system.J(x0, q_par)
    M = _dense(system.M).astype(float)
    nJ = max(1.0, np.abs(J).max())
    res_eq = float(np.max(np.abs(system.G(x0, q_par))))
    # a normal form only exists at an equilibrium: x0 off the branch (
    # finding 91: u' = p + u^2 at p = 1, u = 0, G = 1, was classified a fold) is refused, with
    # the same relative tolerance as the critical eigenvalue
    eq_tol = kernel_tol * nJ * max(1.0, float(np.max(np.abs(x0), initial=0.0)))
    if not res_eq <= eq_tol:
        raise ConvergenceError(f"normal_form: x0 is not an equilibrium (max|G(x0)| = {res_eq:.2e} > "
                               f"kernel_tol·‖J‖·max(1, ‖x0‖) = {eq_tol:.2e})",
                               fix="converge x0 first (newton_equilibrium, or the located "
                                   "continue_equilibria event)")
    lam, vr, vl = generalized_eigenvalues(J, M, vectors=True)
    if kind == "zero":
        i = int(np.argmin(np.abs(lam)))
        if abs(lam[i]) > kernel_tol * nJ:
            raise ConvergenceError(f"normal_form: no zero eigenvalue at x0 (closest {lam[i]:.2e})",
                                   fix="locate the fold/branch point first (continue_equilibria event)")
        return _zero_nf(system, x0, q_par, J, M, lam, vr, vl, i, parameter, weight, symmetry,
                        kernel_tol, res_eq)
    # Hopf
    cand = np.flatnonzero(lam.imag > 0)
    if omega is not None:
        cand = cand[np.argsort(np.abs(lam[cand].imag - omega))][:1]
    else:
        cand = cand[np.argsort(np.abs(lam[cand].real))][:1]
    if cand.size == 0:
        raise ConvergenceError("normal_form: no complex eigenvalue at x0 (not a Hopf point)")
    i = int(cand[0])
    if abs(lam[i].real) > kernel_tol * nJ:
        raise ConvergenceError(f"normal_form: critical pair not on the axis (Re λ = {lam[i].real:.2e})",
                               fix="locate the Hopf point first (continue_equilibria event)")
    return _hopf_nf(system, x0, q_par, J, M, lam[i], vr[:, i], vl[:, i], parameter, weight, res_eq)


def _zero_nf(system, x0, par, J, M, lam, vr, vl, i, parameter, weight, symmetry, tol, res_eq):
    q = np.real(vr[:, i])
    q = q / _norm(q, weight)
    p = np.real(vl[:, i])
    p = p / (p @ M @ q)
    Bqq = system.B(x0, q, q, par)
    a2 = 0.5 * float(p @ Bqq)
    h2 = _bordered_solve(J, M @ q, p @ M, 2 * a2 * (M @ q) - Bqq)
    res_h2 = float(np.max(np.abs(J @ h2 - (2 * a2 * (M @ q) - Bqq))))
    a3 = float(p @ (system.C(x0, q, q, q, par) + 3 * system.B(x0, q, h2, par))) / 6.0
    checks = {"max|G(x0)|": res_eq, "|λ_0|": float(abs(lam[i])),
              "‖J q‖": float(np.max(np.abs(J @ q))), "‖pᵀJ‖": float(np.max(np.abs(p @ J))),
              "residual h2": res_h2}
    notes = []
    sym_parity = None
    if symmetry is not None:
        S = symmetry if callable(symmetry) else (lambda v, S=np.asarray(symmetry): S @ v)  # noqa: B008
        sym_parity = parity_under(S(q), q, PARITY_TOL)
        if sym_parity is None:
            raise SymmetryError("normal_form: the kernel is neither symmetric nor antisymmetric "
                                "under the declared S (S q ≠ ±q)")
        notes.append(f"declared symmetry: S q = {'+' if sym_parity > 0 else '−'}q")
    beta = mu = float("nan")
    if parameter is not None:
        Gp = system.Gp(x0, parameter, par)
        beta = float(p @ Gp)
        # x_p: J x_p = −G_p + β M q (consistent right-hand side), pᵀM x_p = 0
        xp = _bordered_solve(J, M @ q, p @ M, -Gp + beta * (M @ q))
        mu = float(p @ (system.Jp(x0, q, parameter, par) + system.B(x0, q, xp, par)))
        checks["residual x_p"] = float(np.max(np.abs(J @ xp + Gp - beta * (M @ q))))
    scale = max(abs(a2), abs(a3), abs(mu) if np.isfinite(mu) else 0.0, 1e-300)
    if parameter is not None and abs(beta) > tol * max(1.0, np.abs(system.Gp(x0, parameter, par)).max()):
        kind = "fold"
    elif sym_parity == -1:
        kind = "pitchfork"
        if abs(a2) > tol * scale:
            raise SymmetryError(f"normal_form: antisymmetric kernel but a2 = {a2:.2e} ≠ 0 "
                                "(the declared S is not a symmetry of G)")
    elif abs(a2) <= tol * scale:
        kind = "pitchfork"
        notes.append(f"a2 = {a2:.1e} <= {tol:g} x scale: pitchfork (no symmetry declared)")
    else:
        kind = "transcritical"
    return ZeroEigenvalueNormalForm(kind, x0, dict(par), parameter or "", q, p, beta, mu, a2, a3,
                                    h2, checks, notes)


def _hopf_nf(system, x0, par, J, M, lam, qv, pv, parameter, weight, res_eq):
    omega = float(lam.imag)
    q = qv / _norm(qv, weight)
    q = q * np.exp(-1j * np.angle(q[np.argmax(np.abs(q))]))     # declared phase: largest entry real
    p = pv / np.conj(pv.conj() @ M @ q)                          # pᴴ M q = 1
    qb = q.conj()
    Bqqb = system.B(x0, q, qb, par)
    Bqq = system.B(x0, q, q, par)
    h11 = -np.linalg.solve(J, Bqqb)
    h20 = np.linalg.solve(2j * omega * M - J, Bqq)
    g21 = p.conj() @ (system.C(x0, q, q, qb, par) + system.B(x0, qb, h20, par)
                      + 2 * system.B(x0, q, h11, par))
    c1 = complex(g21) / 2.0
    checks = {"max|G(x0)|": res_eq, "|Re λ|": float(abs(lam.real)),
              "‖(J − iωM) q‖": float(np.max(np.abs(J @ q - 1j * omega * (M @ q)))),
              "‖pᴴ(J − iωM)‖": float(np.max(np.abs(p.conj() @ J - 1j * omega * (p.conj() @ M)))),
              "|pᴴ M q − 1|": float(abs(p.conj() @ M @ q - 1)),
              "Re h11 imag part": float(np.max(np.abs(h11.imag)))}
    dl = None
    if parameter is not None:
        Gp = system.Gp(x0, parameter, par)
        xp = -np.linalg.solve(J, Gp)
        dl = complex(p.conj() @ (system.Jp(x0, q, parameter, par) + system.B(x0, q, xp, par)))
    return HopfNormalForm(x0, dict(par), parameter, omega, q, p, c1, c1.real / omega, dl,
                          h20, np.real(h11), checks, [])
