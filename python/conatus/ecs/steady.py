"""Steady states of large IVPs with a singular mass matrix, matrix-free: Newton–Krylov–
hookstep on the Stokes-preconditioned residual, in a symmetry subspace, with the Reynolds number
(or any parameter entering L affinely) as continuation parameter.

Problem: the steady IVP  L(p) X = F(X)  (M dX/dt = 0), with L(p) = A + B / p **affine in 1/p**
(the Reynolds number of Navier–Stokes: viscous terms / Re on the left-hand side). A and B are
extracted from the compiled systems at two values of p and the affine law is **checked** on a
third one (refused otherwise).

Residual (left preconditioning by a declared operator P: by default the Stokes operator
P = L(p_ref), i.e. Tuckerman's Stokes preconditioning with an infinite time step; better, the
operator linearized about the laminar flow, P = L(p_ref) − F'(X_lam) (Oseen), whose Krylov
spectrum is only perturbed by the deviation from laminar):

    R(x, p) = [ P⁻¹ ( F(X) − L(p) X ) ]_act ,   X = E x,

where the active entries ``act`` are the field coefficients of the symmetry subspace plus all tau
(algebraic) entries, and E puts x back into the state vector (other entries zero). The quantity
**reported** is the unpreconditioned relative residual of the full system,
‖F(X) − L(p) X‖ / ‖L(p) X‖ over **all** rows (not only the active ones): a solution found in the
subspace is checked in the full space. Jacobian: J v = [P⁻¹ (F'(X) V − L(p) V)]_act with the exact
linearized RHS (:class:`conatus.adjoint.rhs_transpose.LinearizedRHS`), dR/dp = [P⁻¹ B X / p²]_act.

**Full space (no symmetry group): neutral directions.** An equilibrium u of a problem invariant
under continuous translations along x and z has the neutral directions ∂x u and ∂z u (J singular).
In a symmetry subspace such as the shift-reflect S of plane Couette they are absent (∂x u, ∂z u
break S). In the full space the caller declares the generators (``translations``, e.g.
:class:`conatus.ecs.subspace.StateTranslation`) and the problem becomes the traveling-wave
system (Viswanath 2007; GHC 2009 §4), unknowns (x, c_1..c_m):

    R = [ P⁻¹ ( F(X) − L(p) X + Σ_j c_j M D_j X ) ]_act ,   φ_j = ⟨D_j X_ref, X⟩ / ‖D_j X_ref‖ = 0,

with X_ref the Newton seed (``set_phase_reference``); an equilibrium has c_j = 0, and the
reported residual ‖F(X) − L(p) X‖ / ‖L(p) X‖ does **not** include the c_j terms (a traveling wave
would show up as a non-zero reported residual, never be accepted silently); the c_j are reported.

Works on the CPU or the GPU backend of the solver (vectors cross to the host for the Krylov
algebra of :class:`NewtonKrylovHookstep`)."""
from __future__ import annotations

import numpy as np

from ..adjoint.rhs_transpose import LinearizedRHS
from ..errors import DiscretizationError, OptionError
from .newton import NewtonKrylovHookstep, NewtonKrylovResult

__all__ = ["AffineSteadyProblem", "SteadyResult", "isotypic_spectrum"]


class _PhaseReferenceError(OptionError, RuntimeError):
    """A solve before ``set_phase_reference``: an OptionError that keeps the built-in base of the
    historical raise (RuntimeError), so ``except RuntimeError`` still catches it (errors.py,
    compatibility rule)."""


class SteadyResult:
    """A converged steady state with its evidence (explain())."""

    def __init__(self, X, p, residual, newton: NewtonKrylovResult | None, notes: list[str]):
        self.X, self.p, self.residual, self.newton, self.notes = X, float(p), float(residual), newton, notes

    def explain(self) -> str:
        it = "" if self.newton is None else f", {self.newton.iterations} Newton iterations"
        return "\n".join([f"steady state at p = {self.p:.10g}: full-space relative residual "
                          f"{self.residual:.3e}{it}"] + [f"  note: {n}" for n in self.notes])


class AffineSteadyProblem:
    """Steady states of ``solver_ref``'s IVP for any p, with L(p) = A + B / p.

    ``solver_ref``: IVPSolver at p = p_ref (its factored L is the preconditioner);
    ``system_2``: the GlobalSystem of the same problem at another p (p_2), ``system_3`` at a third
    one (checks the affine law); ``group``: optional :class:`DiagonalSymmetryGroup` (fixed
    subspace = unknowns); ``affine_tol``: refusal threshold of the affine check."""

    def __init__(self, solver_ref, p_ref: float, system_2, p_2: float, system_3, p_3: float, *,
                 group=None, affine_tol: float, preconditioner=None,
                 preconditioner_name: str = "Stokes operator L(p_ref)",
                 translations: list | None = None) -> None:
        s = solver_ref
        self.s, self.xp = s, s.xp
        self.gpu = self.xp is not np
        host = getattr(s.system, "host", s.system)
        self.n = host.n
        L1, L2, L3 = host.L.tocsr(), system_2.L.tocsr(), system_3.L.tocsr()
        if L2.shape != L1.shape or L3.shape != L1.shape:
            raise DiscretizationError("the three systems must have the same layout",
                                      where=f"L shapes {L1.shape}, {L2.shape}, {L3.shape}",
                                      fix="discretize the same problem at the three parameter "
                                          "values (same bases, sizes and boundary conditions)")
        a1, a2 = 1.0 / p_ref, 1.0 / p_2
        self.B = ((L1 - L2) / (a1 - a2)).tocsr()
        self.A = (L1 - a1 * self.B).tocsr()
        pred = self.A + self.B / p_3
        err = abs(pred - L3).max() / max(abs(L3).max(), 1e-300)
        if err > affine_tol:
            raise DiscretizationError(f"L(p) is not affine in 1/p: max deviation {err:.2e} at "
                                      f"p = {p_3:g} (affine_tol = {affine_tol:g})",
                                      why="the steady solver interpolates L(p) = A + B / p from "
                                          "two systems; the third checks it",
                                      fix="the parameter must enter L as 1/p only (e.g. 1/Re on "
                                          "the viscous term); check the three systems")
        self.affine_error = float(err)
        self.p_ref = float(p_ref)
        if self.gpu:
            import cupyx.scipy.sparse as csp
            self._A, self._B = csp.csr_matrix(self.A), csp.csr_matrix(self.B)
        else:
            self._A, self._B = self.A, self.B
        self.lin = LinearizedRHS(s)
        self.group = group
        # tau (algebraic) entries: the entries of the variables the group does not label
        if group is not None:
            self.tau_entry = ~group.field_entry
            self.act = np.flatnonzero(group.mask | self.tau_entry)
        else:
            self.tau_entry = np.zeros(self.n, bool)
            self.act = np.arange(self.n)
        self._act_dev = self.xp.asarray(self.act)
        # phase conditions for the continuous translations (full space, declared)
        self.translations = list(translations or [])
        if self.translations and group is not None:
            raise OptionError("translations: phase conditions are for the full space (a symmetry "
                              "subspace such as S already removes the neutral directions)",
                              fix="pass either group= or translations=, not both")
        self.m = len(self.translations)
        self.n_x = self.act.size
        self._phase = None                      # rows ⟨D_j X_ref, ·⟩ / ‖D_j X_ref‖ (device)
        if self.m:
            Mh = host.M.tocsr()
            if self.gpu:
                import cupyx.scipy.sparse as csp
                self._M = csp.csr_matrix(Mh)
            else:
                self._M = Mh
        # preconditioner: the factored L of another solver with the same state layout (e.g. the
        # problem linearized about the laminar flow: Oseen preconditioner), declared
        self.P_solver, self.P_name = (preconditioner or s), preconditioner_name
        if preconditioner is not None:
            ph = getattr(preconditioner.system, "host", preconditioner.system)
            same = ph.n == host.n and len(ph._col_sel) == len(host._col_sel) and all(
                np.array_equal(a, b) for a, b in zip(ph._col_sel, host._col_sel, strict=True)) and \
                np.array_equal(ph.cols_flat, host.cols_flat)
            if not same:
                raise DiscretizationError("preconditioner: different state layout",
                                          fix="build the preconditioner solver from the same "
                                              "bases, sizes and boundary conditions")

    # ------------------------------------------------------------------ vectors
    def full(self, x):
        """State vector of the unknowns x (the trailing c_j, if any, are dropped)."""
        X = self.xp.zeros(self.n)
        X[self._act_dev] = self.xp.asarray(np.asarray(x)[:self.n_x])
        return X

    def active(self, X) -> np.ndarray:
        """Unknowns of a state (c_j = 0 appended when phase conditions are declared)."""
        a = X[self._act_dev]
        a = np.asarray(a.get() if self.gpu else a)
        return np.concatenate([a, np.zeros(self.m)]) if self.m else a

    def speeds(self, x) -> np.ndarray:
        """The c_j of the unknowns (empty without phase conditions)."""
        return np.asarray(x)[self.n_x:self.n_x + self.m]

    def set_phase_reference(self, X) -> None:
        """Phase conditions ⟨D_j X_ref, X⟩ / ‖D_j X_ref‖ = 0 about the state X_ref (declared)."""
        if not self.m:
            return
        rows = []
        for D in self.translations:
            d = D.apply(self.xp.asarray(X))
            nd = float(self.xp.linalg.norm(d))
            if nd == 0:
                raise OptionError("phase condition: D X_ref = 0 (the reference is invariant under "
                                  "this translation; no neutral direction to fix)",
                                  fix="drop this translation from translations=, or start from a "
                                      "state that depends on that coordinate")
            rows.append(d / nd)
        self._phase = rows

    def _phase_rows(self, X):
        if self._phase is None:
            raise _PhaseReferenceError("phase conditions declared but no reference",
                                        fix="call set_phase_reference(X0) before solving")
        return np.array([float(self.xp.dot(r, X)) for r in self._phase])

    def _translation_term(self, X, c):
        out = self.xp.zeros(self.n)
        for cj, D in zip(c, self.translations, strict=True):
            out = out + float(cj) * D.apply(X)
        return self._M @ out

    def L(self, X, p):
        return self._A @ X + (self._B @ X) / p

    def _P(self, r):
        return self.P_solver.system.solve(0.0, 1.0, r)   # P⁻¹ (the preconditioner's own factors)

    # ------------------------------------------------------------------ residual / Jacobian
    def _act(self, Y) -> np.ndarray:
        a = Y[self._act_dev]
        return np.asarray(a.get() if self.gpu else a)

    def residual(self, x, p):
        X = self.full(x)
        r = self.s.F(X, 0.0) - self.L(X, p)
        if not self.m:
            return self._act(self._P(r))
        r = r + self._translation_term(X, self.speeds(x))
        return np.concatenate([self._act(self._P(r)), self._phase_rows(X)])

    def jvp(self, x, v, p):
        X, V = self.full(x), self.full(v)
        r = self.lin.apply(X, V) - self.L(V, p)
        if not self.m:
            return self._act(self._P(r))
        r = r + self._translation_term(V, self.speeds(x)) + self._translation_term(X, self.speeds(v))
        return np.concatenate([self._act(self._P(r)), self._phase_rows(V)])

    def dRdp(self, x, p):
        X = self.full(x)
        r = self._act(self._P((self._B @ X) / p**2))
        return np.concatenate([r, np.zeros(self.m)]) if self.m else r

    def full_residual(self, x, p) -> float:
        """Unpreconditioned relative residual over all rows (the reported number)."""
        X = self.full(x)
        LX = self.L(X, p)
        r = self.s.F(X, 0.0) - LX
        return float(self.xp.linalg.norm(r) / self.xp.linalg.norm(LX))

    # ------------------------------------------------------------------ Newton
    def solve(self, X0, p: float, newton: NewtonKrylovHookstep, *, log=None) -> SteadyResult:
        """Newton–Krylov–hookstep from the state X0 (projected on the subspace) at fixed p."""
        X0 = self.xp.asarray(X0)
        if self.group is not None:
            X0 = self.group.project(X0) + X0 * self.xp.asarray(self.tau_entry.astype(float))
        if self.m:
            self.set_phase_reference(X0)
        x0 = self.active(X0)
        scale = max(1.0, float(np.linalg.norm(x0)))
        res = newton.solve(lambda z: self.residual(z, p), lambda z, d: self.jvp(z, d, p), x0,
                           scale=scale, log=log)
        X = self.full(res.z)
        notes = [f"L(p) = A + B/p checked at a third p: deviation {self.affine_error:.1e}",
                 f"left preconditioner: {self.P_name} (p_ref = {self.p_ref:g})",
                 "residual reported: ‖F(X) − L(p) X‖ / ‖L(p) X‖ over all rows"]
        if self.group is not None:
            notes.append(f"unknowns in the fixed subspace ({int(self.group.mask.sum())} field "
                         f"coefficients) + {int(self.tau_entry.sum())} tau entries")
        else:
            notes.append(f"full space: {self.n_x} unknowns (all state entries)")
        if self.m:
            c = self.speeds(res.z)
            notes.append("phase conditions about the seed for " + ", ".join(
                D.explain() for D in self.translations) + "; speeds c = "
                + ", ".join(f"{v:.2e}" for v in c) + " (0 for an equilibrium; not in the "
                "reported residual)")
        return SteadyResult(X, p, self.full_residual(res.z, p), res, notes)

    def explain(self) -> str:
        g = "" if self.group is None else "\n" + self.group.explain()
        ph = (f" + {self.m} speeds (phase conditions: "
              + ", ".join(D.explain() for D in self.translations) + ")") if self.m else ""
        return (f"AffineSteadyProblem: n = {self.n}, {self.act.size} unknowns{ph}, L(p) = A + B/p "
                f"(affine check {self.affine_error:.1e}), left preconditioner {self.P_name}, p_ref = "
                f"{self.p_ref:g}, {'GPU' if self.gpu else 'CPU'}; " + self.lin.explain() + g)


def isotypic_spectrum(solver, X, group, *, T: float, nsteps: int, k: int, tol: float,
                      ncv: int | None = None, branch_steps: int | None = None,
                      tangent: str = "exact", fd_eps: float | None = None,
                      advance=None) -> dict:
    """Leading eigenvalues of the linearization about a steady state X in **each isotypic
    component** of a diagonal symmetry group (D7): Arnoldi (ARPACK, largest modulus) on the exact
    tangent of the discrete time-T map (FlowMap.jvp, dt = T / nsteps) restricted to the component's
    field entries; |μ| gives Re λ = log|μ| / T. The imaginary part of log μ / T is only known
    modulo 2π/T: with ``branch_steps`` (declared), each Ritz vector v is also pushed through the
    short map of ``branch_steps`` steps (δ = branch_steps · dt) and λ = log(vᴴ J_δ v / vᴴ v) / δ
    fixes the branch (|Im λ| δ < π); both estimates are returned.

    ``tangent``: ``"exact"`` (FlowMap.jvp, the discrete tangent) or ``"central_difference"``
    (declared: (Φ(X + εV) − Φ(X − εV)) / 2ε with ε = ``fd_eps`` · ‖X‖/‖V‖, through the caller's
    ``advance(X, nsteps)`` — e.g. CUDA-graph steps; error O(ε²) + rounding/ε). The central
    difference is compared with the exact tangent on one vector and the deviation is reported.
    The steady state being
    equivariant, the tangent map commutes with the group and preserves each component (the leak
    out of the component is measured and reported). ARPACK needs k < n - 1: a component of n
    field entries gets min(k, n - 2) eigenvalues, the number used is returned in ``k`` per
    character. The solver's step is set to T / nsteps for the computation and restored after.
    Returns {"eigenvalues": {character: eigenvalues}, "short_map", "leak", "matvecs", "k", ...}."""
    if tangent not in ("exact", "central_difference"):
        raise OptionError(f"tangent must be 'exact' or 'central_difference', got {tangent!r}",
                          fix="tangent='exact' (linearized steps) or tangent='central_difference' "
                              "(with fd_eps= and advance=)")
    if tangent == "central_difference" and (fd_eps is None or advance is None):
        raise OptionError("central_difference needs fd_eps and advance (declared)",
                          fix="pass fd_eps= (e.g. 1e-6) and advance=(X, nsteps) -> X, or use "
                              "tangent='exact'")
    dt, dt_caller = T / nsteps, solver.dt
    solver.set_dt(dt)
    try:
        return _isotypic_spectrum(solver, X, group, dt, T=T, nsteps=nsteps, k=k, tol=tol, ncv=ncv,
                                  branch_steps=branch_steps, tangent=tangent, fd_eps=fd_eps,
                                  advance=advance)
    finally:
        if solver.dt != dt_caller:
            solver.set_dt(dt_caller)


def _isotypic_spectrum(solver, X, group, dt, *, T, nsteps, k, tol, ncv, branch_steps, tangent,
                       fd_eps, advance) -> dict:
    """Body of :func:`isotypic_spectrum` (the solver's step already set to dt = T / nsteps)."""
    import scipy.sparse.linalg as spla

    from ..adjoint import FlowMap
    xp = solver.xp
    fm = FlowMap(solver, nsteps)
    fs = FlowMap(solver, branch_steps) if branch_steps else None
    Xd = xp.asarray(X)
    nX = float(xp.linalg.norm(Xd))

    def tan(V, n, exact_map):
        if tangent == "exact":
            return exact_map.jvp(Xd, V)
        e = fd_eps * nX / max(float(xp.linalg.norm(V)), 1e-300)
        return (advance(Xd + e * V, n) - advance(Xd - e * V, n)) / (2 * e)
    fd_check = None
    if tangent == "central_difference":
        rng = np.random.default_rng(0)
        V = xp.asarray(rng.standard_normal(solver.system.n)) * xp.asarray(group.field_entry.astype(float))
        a, b = fm.jvp(Xd, V), tan(V, nsteps, fm)
        fd_check = float(xp.linalg.norm(a - b) / xp.linalg.norm(a))
    out, leaks, short, nmv, k_used = {}, {}, {}, {}, {}
    host = (lambda a: a) if xp is np else (lambda a: a.get())
    for ch, m in group.isotypic.items():
        idx = np.flatnonzero(m)
        idx_d = xp.asarray(idx)
        other = xp.asarray((group.field_entry & ~m).astype(float))
        leak, count = [0.0], [0]

        def embed(v, idx_d=idx_d):
            V = xp.zeros(solver.system.n, dtype=complex if np.iscomplexobj(v) else float)
            V[idx_d] = xp.asarray(v)
            return V

        def mv(v, idx_d=idx_d, other=other, leak=leak, count=count):
            W = tan(embed(v), nsteps, fm)
            count[0] += 1
            nv = float(xp.linalg.norm(W))
            if nv:
                leak[0] = max(leak[0], float(xp.linalg.norm(W * other)) / nv)
            return np.asarray(host(W[idx_d]))
        op = spla.LinearOperator((idx.size, idx.size), matvec=mv, dtype=float)
        k_used[ch] = min(k, idx.size - 2)              # ARPACK: k < n - 1 (reported)
        mu, vec = spla.eigs(op, k=k_used[ch], which="LM", tol=tol, ncv=ncv)
        lam = np.log(mu.astype(complex)) / T
        order = np.argsort(-lam.real)
        out[ch] = lam[order]
        if fs is not None:
            est = []
            for j in order:
                v = vec[:, j]
                Jv = (np.asarray(host(tan(embed(v.real), branch_steps, fs))[idx])
                      + 1j * np.asarray(host(tan(embed(v.imag), branch_steps, fs))[idx]))
                est.append(np.log(np.vdot(v, Jv) / np.vdot(v, v)) / (branch_steps * dt))
            short[ch] = np.array(est)
        leaks[ch], nmv[ch] = leak[0], count[0]
    return {"eigenvalues": out, "short_map": short, "leak": leaks, "matvecs": nmv, "k": k_used, "T": T,
            "nsteps": nsteps, "branch_steps": branch_steps, "tangent": tangent,
            "fd_eps": fd_eps, "fd_vs_exact": fd_check}
