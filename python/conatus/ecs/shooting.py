"""Relative periodic orbits of large spectral states by shooting, and their Floquet multipliers by
Arnoldi on the monodromy. §"Channel orbits".

The 1D tools of :mod:`conatus.ecs.solutions` (KS) work on a FourierOps state with a dense mean
mask. Here the state is that of any Runge–Kutta :class:`IVPSolver` whose periodic axes are
RealFourier (channel, Couette, ...): the unknowns are the field coefficients of a **diagonal
symmetry subspace** (:class:`~conatus.ecs.subspace.DiagonalSymmetryGroup` mask), the period T and
one shift per declared continuous translation (RealFourier axes that the subspace leaves free).

Problem (standard shooting, Viswanath 2007; Channelflow convention Φ_T(u) = τ(a) u):
    R(x, T, a) = [τ(−a) Φ_T(X) − X]_S = 0,   τ(a) u(x) = u(x + a),
with Φ_T the **exact discrete map** (FlowMap, RK, dt = T / nsteps: nsteps is fixed and declared,
dt follows T) and two kinds of phase conditions (pinning the neutral directions):
    ⟨t₀, δx⟩ = 0 (time; t₀ = (Φ_dt(X₀) − X₀)/dt, one step of the discrete map),
    ⟨∂_a X₀, δx⟩ = 0 for each free translation.
Jacobian: J_X δx from FlowMap.jvp (exact tangent of the discrete map); ∂R/∂T by a central
difference of discrete maps, (Φ_{T+ε} − Φ_{T−ε}) X / 2ε with ε declared (``dT_rel``), once per
Newton iteration (one vector, not one per GMRES iteration); ∂R/∂a = −∂_x(τ(−a) Φ_T X), exact.
The reported residual is ‖R‖ of the discrete problem, relative to ‖X‖ and to ‖X − X_base‖
(``base``: e.g. the laminar state, declared), never a preconditioned or approximate number.

Tau and algebraic entries (taus, mean pressure gradient) are not unknowns: they are slaved to the
fields in every solve (the first stage of the IMEX RK schemes used here is explicit, so Φ does not
read them; :meth:`ShootingRPO.tau_independence` measures it instead of assuming it).

Floquet multipliers: Arnoldi (modified Gram–Schmidt + one reorthogonalisation) on the monodromy
A = τ(−a) J_T restricted to one isotypic component of the group (the subspace itself by default;
A commutes with the group because the translations commute with it). Each Ritz pair (Λ, w) comes
with two residuals: the Arnoldi estimate |h_{m+1,m} e_mᵀ y| and the true ‖A w − Λ w‖/‖w‖
(two extra real tangents per pair). Neutral directions are identified by the overlap of w with t
(time) and ∂_a X (translation), threshold declared.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..adjoint import FlowMap
from .newton import NewtonKrylovHookstep, NewtonKrylovResult
from .subspace import StateTranslation, _xp
from ..errors import OptionError, SymmetryError
from .._repr import compact_repr

__all__ = ["StateShift", "ShootingRPO", "ShootingRPOResult", "arnoldi_eigs", "resample_state",
           "span_fractions"]


class StateShift(StateTranslation):
    """Finite translation τ(a) u(x) = u(x + a) along a RealFourier axis: exp(a ∂x) on each
    cos/−sin pair, i.e. a rotation by k a (k = 2πn/L). Exact for every a (no interpolation)."""

    def __init__(self, solver, coord: str) -> None:
        super().__init__(solver, coord)
        # wavenumber of each paired entry (0 where unpaired: those entries are left unchanged)
        self.k = np.abs(self.fac)
        self.sgn = np.sign(self.fac)

    def shift(self, X, a: float):
        """τ(a) X (NumPy or CuPy state); tau entries are multiplied by 1 (left as they are)."""
        xp = _xp(X)
        key = ("shift", id(xp))
        if key not in self._dev:
            self._dev[key] = (xp.asarray(self.src), xp.asarray(self.k), xp.asarray(self.sgn))
        src, k, sgn = self._dev[key]
        return xp.cos(k * a) * X + xp.sin(k * a) * sgn * X[src]


class _StageCachedFlowMap(FlowMap):
    """FlowMap whose RK stages at one X₀ are kept, so that the many tangents of one GMRES (or
    Arnoldi) at the same X₀ replay the stored stages instead of recomputing the forward pass.
    Same arithmetic as FlowMap.jvp (it calls the same ``_rk_jvp``); declared memory cost:
    nsteps × (stages + 1) states."""

    def _record(self, X0):
        key = (np.asarray(X0).tobytes(), self.solver.dt)
        if getattr(self, "_key", None) != key:
            self._subs, X = [], self.xp.asarray(X0, dtype=self.dtype)
            for st in self._steps:
                if st.kind != "rk":
                    raise OptionError("stage cache: Runge–Kutta schemes only",
                                      fix="cache_stages=False, or a Runge-Kutta scheme")
                subs = self._rk_stages(X, st)
                self._subs.append(subs)
                X = subs[-1][-1]
            self._end, self._key = X, key
        return self._subs

    def end(self, X0):
        self._record(X0)
        return self._end

    def jvp(self, X0, V):
        subs = self._record(X0)
        dX = self.xp.asarray(V, dtype=self.dtype)
        for st, s in zip(self._steps, subs, strict=True):
            dX = self._rk_jvp(s, dX, st)
        return dX


@dataclass
class ShootingRPOResult:
    """A converged (or best) relative periodic orbit of the discrete map."""
    __repr__ = compact_repr   # one line, arrays by shape
    X: np.ndarray
    T: float
    shifts: dict
    residual: float                     # ‖R‖ / ‖X_S‖ (field entries of the subspace)
    residual_vs_base: float | None      # ‖R‖ / ‖(X − base)_S‖
    newton: NewtonKrylovResult | None
    nsteps: int
    notes: list = field(default_factory=list)
    # hunt 5A (r031f): a steady state solves the shooting equations for a continuum of (T, a);
    # the solve classifies its result as the 1-D finder does (ecs.solutions._guard_steady)
    kind: str = "relative_periodic_orbit"     # or "equilibrium", "travelling_wave", "trivial"
    status: str = "converged"                 # "retracted": not an orbit; "repeated" (see notes)
    repetitions: int = 1                      # k > 1: k laps of an orbit of period T / k (K4)

    @property
    def dt(self) -> float:
        """Time step of the discrete map, T / nsteps."""
        return self.T / self.nsteps

    def explain(self) -> str:
        """Period, shifts, discrete map and reported residuals, then the Newton history."""
        sh = ", ".join(f"a_{c} = {v:.10g}" for c, v in self.shifts.items()) or "no shift"
        head = {"relative_periodic_orbit": "Relative periodic orbit",
                "equilibrium": "RETRACTED, an equilibrium, not an orbit",
                "travelling_wave": "RETRACTED, a travelling wave (relative equilibrium), not an orbit",
                "trivial": "RETRACTED, the trivial state, not an orbit"}.get(self.kind, self.kind)
        lines = [f"{head}: T = {self.T:.10g}, {sh}; discrete map of {self.nsteps} "
                 f"steps (dt = {self.dt:.6g}); residual ‖τ(−a)Φ_T(X) − X‖/‖X‖ = {self.residual:.3e}"
                 + (f", /‖X − base‖ = {self.residual_vs_base:.3e}"
                    if self.residual_vs_base is not None else "")]
        if self.newton is not None:
            lines.append(self.newton.explain())
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


class ShootingRPO:
    """Shooting problem τ(−a) Φ_T(X) = X in a diagonal symmetry subspace (module docstring).

    ``group``: :class:`DiagonalSymmetryGroup` whose fixed subspace holds the orbit (declared);
    ``shifts``: names of the RealFourier axes along which the orbit may drift (each adds an
    unknown and a phase condition; an axis whose translations break the subspace must not be
    listed); ``nsteps``: steps per period (dt = T / nsteps); ``dT_rel``: relative step of the
    central difference giving ∂Φ_T/∂T; ``cache_stages``: keep the RK stages of the current X
    (memory nsteps × (s + 1) states) — both declared, no default; ``base``: reference state for
    the second reported residual (e.g. laminar), or None."""

    def __init__(self, solver, group, *, shifts: list[str], nsteps: int, dT_rel: float,
                 cache_stages: bool, base=None) -> None:
        if solver.order or getattr(solver, "family", "rk") != "rk":
            raise OptionError("ShootingRPO: use a Runge–Kutta scheme (dt = T/nsteps changes with T)",
                              fix="an IMEX Runge-Kutta scheme (RK222, RK443): dt = T/steps changes with T")
        if solver.backend != "cpu":
            raise OptionError("ShootingRPO: CPU solver (Newton–Krylov vectors are NumPy arrays)",
                              fix="a CPU solver (backend='cpu')")
        if not dT_rel > 0:
            raise OptionError("dT_rel must be > 0 (declared central-difference step)",
                              fix="dT_rel=1e-5 (the relative central-difference step of the period derivative)")
        from .._experimental import experimental
        experimental("orbits")
        self.solver, self.group = solver, group
        self.nsteps, self.dT_rel, self.cache = int(nsteps), float(dT_rel), bool(cache_stages)
        self.idx = np.flatnonzero(group.mask)
        self.n = solver.system.n
        self.shift_ops = {c: StateShift(solver, c) for c in shifts}
        self.base = None if base is None else np.asarray(base, dtype=float)
        self._maps: dict = {}
        self.stats = dict(maps=0, jvps=0, t_maps=0.0, t_jvps=0.0)
        # a declared shift must keep the subspace: check that ∂_c maps S into S on a test vector
        rng = np.random.default_rng(0)
        V = self.embed(rng.standard_normal(self.idx.size))
        for c, op in self.shift_ops.items():
            D = op.apply(V)
            out = np.linalg.norm(D * group.field_entry * ~group.mask)
            if out > 1e-12 * max(np.linalg.norm(D), 1e-300):
                raise SymmetryError(f"translations along {c} leave the symmetry subspace: not a free "
                                    "shift of this group (do not list it in shifts=)",
                                    fix="leave this direction out of shifts= (it is not a free shift of the group)")
        # the flow must keep the subspace and commute with each declared shift (external review
        # of 0.2.3: a finder that assumes a symmetry checks it)
        from .equivariance import LOW_MODE_FRACTION, grid_cell, require_autonomous, require_equivariant
        amp = 1e-3
        probe = self.base + amp * V if self.base is not None else amp * V
        where = "ShootingRPO"
        require_autonomous(solver, [probe], where)
        for c, op in self.shift_ops.items():
            low = np.where(op.k <= LOW_MODE_FRACTION * float(op.k.max()), probe, 0.0)
            cell = grid_cell(solver, c)
            require_equivariant(solver, lambda X, op=op, a=cell: op.shift(X, a), [low], where,
                                f"translation symmetry along {c}",
                                f"leave {c} out of shifts=, or remove the explicit dependence on {c}")

    def explain(self) -> str:
        """The declared shooting problem: unknowns, discrete map, phase conditions, Jacobian."""
        sh = ", ".join(self.shift_ops) or "none"
        return "\n".join([
            f"Shooting for a relative periodic orbit: τ(−a) Φ_T(X) = X in the fixed subspace of "
            f"{self.group.__class__.__name__} ({self.idx.size} field unknowns) + T + shifts "
            f"[{sh}]; τ(a)u(x) = u(x + a), exact rotation of each Fourier pair",
            f"  Φ_T: {self.nsteps} steps of {getattr(self.solver, 'scheme', 'the RK scheme')} with "
            "dt = T / nsteps (the step follows T); taus and algebraic entries slaved",
            "  phase conditions: ⟨t₀, δx⟩ = 0 (t₀ = one step of the discrete map) and ⟨∂_a X₀, δx⟩ = 0 "
            "per shift",
            f"  Jacobian: exact discrete tangent; ∂R/∂T by central difference, ε = {self.dT_rel:g} T; "
            f"∂R/∂a exact; RK stages cached: {self.cache}",
            "  reported residual: ‖τ(−a)Φ_T(X) − X‖/‖X‖ on the subspace entries"
            + (", and relative to ‖X − base‖" if self.base is not None else ""),
        ])

    # ------------------------------------------------------------------ vectors
    def embed(self, x):
        """Full state vector from the subspace unknowns x (zeros elsewhere, taus included)."""
        X = np.zeros(self.n)
        X[self.idx] = x
        return X

    def active(self, X):
        """Subspace unknowns of a full state vector (the entries of the group mask)."""
        return np.asarray(X)[self.idx]

    def flow(self, T: float):
        """FlowMap of nsteps steps with dt = T / nsteps (a few kept, keyed by T)."""
        if T not in self._maps:
            if len(self._maps) >= 3:
                self._maps.pop(next(iter(self._maps)))
                # banded factors are cached per (a, b) = per dt: drop those of evicted periods
                # (rebuilt on demand) so that a long Newton run does not accumulate them
                getattr(self.solver.system, "_factors", {}).clear()
            self.solver.set_dt(T / self.nsteps)
            cls = _StageCachedFlowMap if self.cache else FlowMap
            self._maps[T] = cls(self.solver, self.nsteps)
        fm = self._maps[T]
        self.solver.set_dt(T / self.nsteps)     # the solver's dt is the one this map was built for
        return fm

    def phi(self, X, T: float):
        """Φ_T(X): the discrete map (stage-cached map: same arithmetic as FlowMap.__call__)."""
        t0 = time.time()
        fm = self.flow(T)
        Y = fm.end(X) if self.cache else fm(X)
        self.stats["maps"] += 1
        self.stats["t_maps"] += time.time() - t0
        return Y

    def jvp(self, X, T: float, V):
        """Exact discrete tangent J_T V of the map Φ_T at X (FlowMap.jvp, timed)."""
        t0 = time.time()
        Y = self.flow(T).jvp(X, V)
        self.stats["jvps"] += 1
        self.stats["t_jvps"] += time.time() - t0
        return Y

    def unshift(self, Y, a: dict):
        """τ(−a) Y: undo the declared shifts a = {axis: shift}."""
        for c, op in self.shift_ops.items():
            Y = op.shift(Y, -a[c])
        return Y

    def residual_full(self, X, T: float, a: dict):
        """τ(−a) Φ_T(X) − X on the field entries of the subspace (full-length vector)."""
        return (self.unshift(self.phi(X, T), a) - X) * self.group.mask

    def time_direction(self, X, T: float):
        """(Φ_dt(X) − X)/dt with dt = T / nsteps: one step of the same discrete scheme."""
        s = self.solver
        s.set_dt(T / self.nsteps)
        Y = s.rk_stages(np.asarray(X, dtype=float), 0.0, s.dt)[-1]
        return (Y - X) / s.dt * self.group.mask

    def tau_independence(self, X, T: float) -> float:
        """‖Φ_T(X) − Φ_T(X')‖/‖Φ_T(X)‖ with X' = X with random tau/algebraic entries: must be at
        rounding level for the field-only unknowns to be legitimate (reported)."""
        rng = np.random.default_rng(1)
        Xp = np.array(X, dtype=float)
        tau = ~self.group.field_entry
        Xp[tau] = rng.standard_normal(int(tau.sum())) * max(1.0, float(np.abs(X).max()))
        a, b = self.phi(X, T), self.phi(Xp, T)
        f = self.group.field_entry
        return float(np.linalg.norm((a - b)[f]) / np.linalg.norm(a[f]))

    # ------------------------------------------------------------------ Newton
    def solve(self, X0, T0: float, a0: dict, newton: NewtonKrylovHookstep, *, log=None,
              on_iteration=None, on_improve=None) -> ShootingRPOResult:
        """Newton–Krylov–hookstep from (X0, T0, a0). X0 is projected on the subspace (its defect
        is reported); tau entries of X0 are dropped (slaved). ``on_improve(X, T, a, rel)``
        (optional) is called whenever a residual evaluation beats the best so far: a restart
        point for runs longer than one job (it never changes the iteration)."""
        names = list(self.shift_ops)
        if set(a0) != set(names):
            raise OptionError(f"a0 must give the shifts {names}",
                              fix="a0 = {name: shift} for every name of shifts=")
        defect = float(self.group.defect(np.asarray(X0, dtype=float)))
        X0 = self.embed(self.active(X0))
        t_ref = self.active(self.time_direction(X0, T0))
        t_ref = t_ref / np.linalg.norm(t_ref)
        d_ref = []
        for c in names:
            d = self.active(self.shift_ops[c].apply(X0))
            d_ref.append(d / np.linalg.norm(d))
        na = len(names)
        cache: dict = {}
        best = [np.inf]

        def unpack(z):
            return self.embed(z[:self.idx.size]), float(z[self.idx.size]), \
                {c: float(z[self.idx.size + 1 + i]) for i, c in enumerate(names)}

        def R(z):
            X, T, a = unpack(z)
            r = self.active(self.residual_full(X, T, a))
            nr = float(np.linalg.norm(r))
            if on_improve is not None and nr < best[0]:
                best[0] = nr
                on_improve(X, T, a, nr / float(np.linalg.norm(self.active(X))))
            return np.concatenate([r, np.zeros(1 + na)])

        def columns(z):
            """∂R/∂T (central difference of discrete maps) and ∂R/∂a (exact), once per z."""
            key = z.tobytes()
            if cache.get("key") != key:
                X, T, a = unpack(z)
                e = self.dT_rel * T
                gT = (self.unshift(self.phi(X, T + e), a) - self.unshift(self.phi(X, T - e), a)) / (2 * e)
                Y = self.unshift(self.phi(X, T), a)
                ga = [-self.active(self.shift_ops[c].apply(Y)) for c in names]
                cache.update(key=key, gT=self.active(gT), ga=ga)
            return cache["gT"], cache["ga"]

        def J(z, d):
            X, T, a = unpack(z)
            dx = d[:self.idx.size]
            gT, ga = columns(z)
            body = self.active(self.unshift(self.jvp(X, T, self.embed(dx)), a)) - dx
            body = body + d[self.idx.size] * gT
            for i in range(na):
                body = body + d[self.idx.size + 1 + i] * ga[i]
            rows = [t_ref @ dx] + [dr @ dx for dr in d_ref]
            return np.concatenate([body, rows])

        z0 = np.concatenate([self.active(X0), [T0], [a0[c] for c in names]])
        scale = float(np.linalg.norm(self.active(X0)))
        res = newton.solve(R, J, z0, scale=scale, log=log, on_iteration=on_iteration)
        X, T, a = unpack(res.z)
        sol = self.result(X, T, a, res)
        sol.notes.append(f"initial symmetry defect of X0 (removed by projection): {defect:.2e}")
        return self.classify(sol, scale)

    def classify(self, sol: ShootingRPOResult, scale: float) -> ShootingRPOResult:
        """Retract a solve that converged onto the trivial state or a steady state.

        Hunt 5A (r031f): a travelling wave and an equilibrium (KS E2) solve τ(−a) Φ_T(X) = X for
        a continuum of (T, a) and were returned as relative periodic orbits with residual 1e-13,
        while the 1-D finder retracts them. Same criteria as ``conatus.ecs.solutions``: trivial
        when ‖X_S‖ < TRIVIAL_TOL x scale; steady when the state keeps its shape along the period,
        A = max_j ‖τ(−j δ) X_j − X‖_S / ‖X‖_S < STEADY_TOL x the accuracy of the solve, with δ the
        shift of one step (Gauss–Newton on τ(−δ) Φ_dt(X) = X; on the discrete map a steady state
        moves by the same shift every step). Kind ``equilibrium`` when |δ| ‖∂X‖ is below that
        accuracy too, ``travelling_wave`` otherwise; ``status = "retracted"`` in both cases."""
        from .solutions import STEADY_SAMPLES, STEADY_TOL, TRIVIAL_TOL
        X = np.asarray(sol.X)
        nX = float(np.linalg.norm(self.active(X)))
        sol.notes.append(f"trivial-state threshold: ‖X_S‖ < {TRIVIAL_TOL:g} x {scale:.3g} ⇒ retracted")
        if nX < TRIVIAL_TOL * scale:
            sol.kind, sol.status = "trivial", "retracted"
            sol.notes.append("converged to the trivial state (not a new solution)")
            return sol
        host = lambda u: np.asarray(u.get() if hasattr(u, "get") else u)   # noqa: E731
        states = [host(u) for u in self.flow(sol.T).trajectory(X)[1:]]
        names = list(self.shift_ops)
        d = {c: 0.0 for c in names}
        for _ in range(4):                  # Gauss–Newton for the shift of one step
            Y = self.unshift(states[0], d)
            F = self.active(Y - X)
            if not names:
                break
            Jc = np.column_stack([-self.active(self.shift_ops[c].apply(Y)) for c in names])
            step = np.linalg.lstsq(Jc, -F, rcond=None)[0]
            d = {c: d[c] + float(step[i]) for i, c in enumerate(names)}
        idx = range(len(states))
        if len(states) > STEADY_SAMPLES:
            idx = np.linspace(0, len(states) - 1, STEADY_SAMPLES).astype(int)
        amp = max(float(np.linalg.norm(self.active(
            self.unshift(states[j], {c: (j + 1) * d[c] for c in names}) - X))) for j in idx) / nX
        newton_rel = getattr(sol.newton, "relative_residual", 0.0) or 0.0
        floor = max(float(sol.residual), float(newton_rel), 1e-15)
        move = max([abs(d[c]) * float(np.linalg.norm(self.active(self.shift_ops[c].apply(X)))) / nX
                    for c in names], default=0.0)
        sol.notes.append(f"steady-state check: shape change over the period max_j ‖τ(−j δ) X_j − "
                         f"X‖/‖X‖ = {amp:.3e} with the one-step shift δ = {d} vs STEADY_TOL x "
                         f"accuracy of the solve = {STEADY_TOL * floor:.1e} (below ⇒ retracted)")
        if amp < STEADY_TOL * floor:
            sol.kind = "equilibrium" if move < STEADY_TOL * floor else "travelling_wave"
            sol.status = "retracted"
            sol.notes.append(f"converged to a steady state ({sol.kind.replace('_', ' ')}), not an "
                             "orbit: it solves the shooting equations for any period")
            return sol
        return self._primitive(sol, states, X, nX, STEADY_TOL * floor)

    def _primitive(self, sol, states, X, nX: float, tol: float) -> ShootingRPOResult:
        """Primitive-period check (hunt 5A K4, r031f): for k = 2..8 dividing nsteps, is
        τ(−s) X_{n/k} = X for some shift s (s from a/k + m L/k, m = 0..k-1, refined by
        Gauss–Newton)? Below ``tol`` the orbit is k laps of an orbit of period T/k: status
        "repeated", ``repetitions = k`` and a note. Only the declared shifts are tried (not the
        other elements of the group)."""
        import itertools
        names = list(self.shift_ops)
        meas, found = [], 1
        for k in [k for k in range(2, 9) if self.nsteps % k == 0]:
            U = states[self.nsteps // k - 1]
            best = np.inf
            for ms in itertools.product(range(k), repeat=len(names)) if len(names) <= 2 else [(0,) * len(names)]:
                d = {c: sol.shifts[c] / k + m * self.shift_ops[c].length / k for c, m in zip(names, ms, strict=True)}
                for _ in range(3):
                    Y = self.unshift(U, d)
                    F = self.active(Y - X)
                    if not names:
                        break
                    Jc = np.column_stack([-self.active(self.shift_ops[c].apply(Y)) for c in names])
                    step = np.linalg.lstsq(Jc, -F, rcond=None)[0]
                    d = {c: d[c] + float(step[i]) for i, c in enumerate(names)}
                best = min(best, float(np.linalg.norm(self.active(self.unshift(U, d) - X))) / nX)
            meas.append(f"k = {k}: {best:.1e}")
            if best < tol:
                found = k
        sol.notes.append("primitive-period check: min_s ‖τ(−s) Φ_{T/k}(X) − X‖/‖X‖ "
                         + (", ".join(meas) if meas else "(no k in 2..8 divides nsteps)")
                         + f" vs {tol:.1e}")
        if found > 1:
            sol.repetitions, sol.status = found, "repeated"
            sol.notes.append(f"REPEATED ORBIT: the state returns at T/{found} = {sol.T / found:.10g}; "
                             f"this solution is {found} laps of it")
        return sol

    def result(self, X, T: float, a: dict, newton=None) -> ShootingRPOResult:
        """Residuals of (X, T, a) as a ShootingRPOResult (also for an unconverged candidate)."""
        r = self.residual_full(X, T, a)
        nr = float(np.linalg.norm(r))
        nX = float(np.linalg.norm(self.active(X)))
        rb = None
        if self.base is not None:
            nb = float(np.linalg.norm(self.active(X - self.base)))
            rb = nr / nb if nb else None
        a = {c: _wrap(v, self.shift_ops[c].length) for c, v in a.items()}
        sol = ShootingRPOResult(np.asarray(X), float(T), a, nr / nX, rb, newton, self.nsteps)
        sol.notes.append("residual of the discrete map (FlowMap, exact); field entries of the "
                         f"subspace ({self.idx.size} unknowns), taus slaved")
        return sol

    # ------------------------------------------------------------------ Floquet
    def floquet(self, sol: ShootingRPOResult, *, krylov_dim: int, k: int, component=None,
                overlap_min: float, seed: int = 0, log=None) -> dict:
        """Leading Floquet multipliers of the RPO: Arnoldi of dimension ``krylov_dim`` on
        A = τ(−a) J_T restricted to ``component`` (a boolean mask over the state: an isotypic
        component of the group; default the fixed subspace), k Ritz pairs with Arnoldi and true
        residuals; neutral directions identified by overlap > ``overlap_min`` (declared) and, since the
        neutral eigenvalue is degenerate, by the fraction of each neutral direction in the span of
        the 1 + len(shifts) Ritz vectors closest to 1 (``neutral_span``, 1 = fully contained)."""
        mask = self.group.mask if component is None else np.asarray(component, bool)
        idx = np.flatnonzero(mask)
        X, T, a = sol.X, sol.T, sol.shifts
        leak = [0.0]

        def mv(v):
            V = np.zeros(self.n)
            V[idx] = v
            W = self.unshift(self.jvp(X, T, V), a)
            nW = float(np.linalg.norm(W * self.group.field_entry))
            if nW:
                leak[0] = max(leak[0], float(np.linalg.norm(W * self.group.field_entry * ~mask)) / nW)
            return W[idx]
        rng = np.random.default_rng(seed)
        t0 = time.time()
        out = arnoldi_eigs(mv, rng.standard_normal(idx.size), m=krylov_dim, k=k, log=log)
        out["leak"] = leak[0]
        out["seconds"] = time.time() - t0
        # neutral directions: time (t = (Φ_dt X − X)/dt) and each declared translation
        dirs = {"time": self.time_direction(X, T)[idx]}
        for c, op in self.shift_ops.items():
            dirs[f"translation_{c}"] = op.apply(X)[idx]
        marg = []
        for i, w in enumerate(out["vectors"].T):
            w = w / np.linalg.norm(w)
            ov = {n: abs(np.vdot(w, d)) / np.linalg.norm(d) for n, d in dirs.items()
                  if np.linalg.norm(d) > 0}
            best = max(ov, key=ov.get) if ov else None
            if best is not None and ov[best] > overlap_min:
                marg.append(dict(index=i, direction=best, overlap=float(ov[best]),
                                 multiplier=complex(out["values"][i])))
        out["marginal"] = marg
        # The neutral eigenvalue is degenerate (time + each translation: all ≈ 1), so single Ritz
        # vectors are arbitrary mixtures and per-vector overlaps can label both "time". The
        # well-posed check: the fraction of each neutral direction lying in the span of the
        # 1 + len(shifts) Ritz vectors closest to 1.
        nn = 1 + len(self.shift_ops)
        near = np.argsort(np.abs(out["values"] - 1.0))[:nn]
        out["neutral_span"] = span_fractions(out["vectors"][:, near], dirs)
        out["neutral_values"] = out["values"][near]
        out["overlap_min"] = float(overlap_min)
        out["exponents"] = np.log(out["values"].astype(complex)) / T
        out["T"] = T
        out["component_size"] = int(idx.size)
        return out


def span_fractions(W, dirs: dict) -> dict:
    """‖P d‖/‖d‖ for each real direction d of ``dirs``, P the orthogonal projector on the real span
    of the columns of W (complex Ritz vectors: real and imaginary parts). 1 means d lies in the
    span; used for the degenerate neutral eigenspace of a relative periodic orbit."""
    A = np.concatenate([np.real(W), np.imag(W)], axis=1)
    U, sv, _ = np.linalg.svd(A, full_matrices=False)
    Q = U[:, sv > 1e-12 * sv.max()]
    return {n: float(np.linalg.norm(Q.T @ d) / np.linalg.norm(d))
            for n, d in dirs.items() if np.linalg.norm(d) > 0}


def _wrap(a: float, L: float) -> float:
    """Shift in (−L/2, L/2]."""
    return float(-((-a + L / 2) % L - L / 2))


def arnoldi_eigs(matvec, v0, *, m: int, k: int, log=None) -> dict:
    """Arnoldi (MGS + one reorthogonalisation) of dimension m from v0 on a real operator; the k
    Ritz values of largest modulus with their Arnoldi residual estimates |h_{m+1,m} y_m| ‖w‖⁻¹
    and their true residuals ‖A w − θ w‖/‖w‖ (computed: 2 real matvecs per complex pair, 1 per
    real value). Returns values, vectors (n × k complex), residual_arnoldi, residual_true,
    matvecs, breakdown (True if the Krylov space became invariant before m)."""
    n = v0.size
    V = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))
    V[:, 0] = v0 / np.linalg.norm(v0)
    nmv, breakdown, j = 0, False, 0
    for j in range(m):
        w = matvec(V[:, j])
        nmv += 1
        for _ in range(2):                       # MGS + one reorthogonalisation
            for i in range(j + 1):
                c = V[:, i] @ w
                H[i, j] += c
                w = w - c * V[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        if log and (j + 1) % 10 == 0:
            log(f"arnoldi {j + 1}/{m}")
        if H[j + 1, j] < 1e-14 * np.abs(H[:j + 2, :j + 1]).max():
            breakdown = True
            break
        V[:, j + 1] = w / H[j + 1, j]
    mm = j + 1
    theta, Y = np.linalg.eig(H[:mm, :mm])
    order = np.argsort(-np.abs(theta))[:k]
    theta, Y = theta[order], Y[:, order]
    vecs = V[:, :mm] @ Y
    est = np.abs(H[mm, mm - 1] * Y[mm - 1, :]) / np.linalg.norm(vecs, axis=0) if mm < m + 1 \
        else np.zeros(len(theta))
    true = np.zeros(len(theta))
    done: dict = {}
    for i, (th, w) in enumerate(zip(theta, vecs.T, strict=True)):
        key = (round(th.real, 12), round(abs(th.imag), 12))
        if key in done:                          # conjugate partner: same residual
            true[i] = done[key]
            continue
        Aw = matvec(w.real) + (1j * matvec(w.imag) if np.iscomplexobj(w) and np.any(w.imag) else 0)
        nmv += 2 if np.iscomplexobj(w) and np.any(w.imag) else 1
        true[i] = np.linalg.norm(Aw - th * w) / np.linalg.norm(w)
        done[key] = true[i]
    return dict(values=theta, vectors=vecs, residual_arnoldi=est, residual_true=true,
                matvecs=nmv, krylov_dim=mm, breakdown=breakdown)


def resample_state(solver_from, X, solver_to):
    """Coefficients of X (fields of solver_from) truncated or zero-padded to the shapes of
    solver_to, axis by axis (RealFourier index i keeps its wavenumber i // 2 and its cos/−sin
    part; Chebyshev keeps T_k): the same continuous field when padding, its spectral truncation
    otherwise. Tau and algebraic entries are set to 0 (slaved). Both solvers must have the same
    variables (names) in the same order."""
    hf = getattr(solver_from.system, "host", solver_from.system)
    ht = getattr(solver_to.system, "host", solver_to.system)
    names_f = [v.name for v in hf.variables]
    names_t = [v.name for v in ht.variables]
    if names_f != names_t:
        raise OptionError("resample_state: the two problems must have the same variables",
                          fix="resample between two problems with the same unknowns")
    arrays = hf.scatter(np.asarray(X))
    out = []
    for a, shp in zip(arrays, ht.var_shapes, strict=True):
        b = np.zeros(shp)
        if a is not None and a.ndim == len(shp):
            sl = tuple(slice(0, min(p, q)) for p, q in zip(a.shape, shp, strict=True))
            b[sl] = a[sl]
        out.append(b)
    return ht.gather_state(out, check_subspace=False)
