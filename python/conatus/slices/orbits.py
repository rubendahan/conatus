"""Invariant solutions seen in the slice: relative periodic orbits become periodic orbits,
travelling waves become equilibria, equilibria stay equilibria (or sit on the slice border).
Notes: §3.

Checks, all measured on the exact discrete map of the solver used to find the solution:

* RPO τ_ℓ Φ_T(u) = u: the reduced trajectory closes, ‖X̂(T) − X̂(0)‖/‖X̂(0)‖ (= the full-space
  residual up to the slice projection), and the net phase ℓ(T) − ℓ(0) equals −ℓ modulo L
  (Φ_T u = τ_{−ℓ} u);
* PPO R τ_ℓ Φ_T(u) = u: periodic of period 2T in full space, hence in the slice (net shift 0 mod L);
* TW u = U(x − ct): in-slice velocity ~ 0 and phase velocity ℓ̇ = −c (u(t) = τ_{−ct} U);
* equilibrium: v(u) = 0, ℓ̇ = 0; if its first Fourier mode vanishes (e.g. KS E2, E3 with period
  L/2, L/3) it lies on the slice border: reported, not an error.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..errors import OptionError
from .._repr import compact_repr

__all__ = ["SliceOrbitCheck", "SliceRPO", "find_rpo_in_slice", "slice_orbit_check",
           "slice_equilibrium_check"]


@dataclass
class SliceOrbitCheck:
    kind: str
    period_in_slice: float
    closure: float                     # ‖X̂(T_s) − X̂(0)‖/‖X̂(0)‖
    net_shift: float                   # ℓ(T_s) − ℓ(0), unwrapped
    expected_shift: float              # −ℓ (RPO) or 0 (PPO), modulo L
    shift_error: float                 # |net − expected| modulo L
    min_border_distance: float
    max_phase_jump: float
    notes: list[str] = field(default_factory=list)

    def explain(self) -> str:
        lines = [f"{self.kind} in the slice: period {self.period_in_slice:.10g}, closure "
                 f"{self.closure:.2e}, net shift {self.net_shift:.10g} (expected "
                 f"{self.expected_shift:.10g} mod L, error {self.shift_error:.2e}), min border "
                 f"distance {self.min_border_distance:.3e}, max phase jump {self.max_phase_jump:.3f} rad"]
        return "\n".join(lines + [f"  note: {n}" for n in self.notes])


def _mod(x: float, L: float) -> float:
    return float((x + L / 2) % L - L / 2)


def slice_orbit_check(slc, solver, sol) -> SliceOrbitCheck:
    """RPO or PPO ``sol`` (conatus.ecs) followed through the slice on its own discrete map
    (same scheme, dt = T/nsteps); see the module doc for what is measured."""
    from ..manifolds.trajectory import SolverIntegrator
    if sol.kind not in ("relative_periodic_orbit", "preperiodic_orbit"):
        raise OptionError(f"slice_orbit_check: {sol.kind} is not an orbit (use slice_equilibrium_check)")
    T, ell, nsteps = sol.params["T"], sol.params["ell"], sol.params["nsteps"]
    ppo = sol.kind == "preperiodic_orbit"
    integ = SolverIntegrator(solver, dt=T / nsteps)
    Ts = 2 * T if ppo else T
    tr = integ.integrate(sol.X, Ts)
    st = slc.reduce_trajectory(tr.t, tr.X, v=integ.v)
    closure = float(np.linalg.norm(st.X_hat[-1] - st.X_hat[0]) / np.linalg.norm(st.X_hat[0]))
    net = float(st.ell[-1] - st.ell[0])
    expected = 0.0 if ppo else -ell
    chk = SliceOrbitCheck(sol.kind, Ts, closure, net, _mod(expected, slc.L),
                          abs(_mod(net - expected, slc.L)), float(st.border_distance.min()),
                          st.max_phase_jump)
    chk.notes.append(f"exact discrete map {solver.scheme}, dt = T/nsteps = {T / nsteps:.6g}; "
                     f"full-space residual of the solution {sol.residual:.2e}")
    chk.notes += st.notes
    if ppo:
        chk.notes.append("pre-periodic orbit: closes after 2T in full space (R τ_ℓ applied twice = I)")
    return chk


def slice_equilibrium_check(slc, solver, sol) -> dict:
    """Equilibrium or travelling wave in the slice: border distance, in-slice speed and phase
    velocity ℓ̇ (−c expected for a TW, 0 for an equilibrium)."""
    from ..ecs.solutions import Flow
    v = Flow(solver)
    d = slc.border_distance(sol.X)
    out = {"kind": sol.kind, "border_distance": d, "on_border": d < slc.border_tol,
           "border_tol": slc.border_tol}
    if d < slc.border_tol:
        out["note"] = ("on the slice border (first Fourier mode ~ 0): phase undefined; the "
                       "solution lives in an invariant subspace of higher periodicity")
        return out
    Xh, ell = slc.reduce(sol.X)
    vh, elld = slc.velocity(Xh, v)
    c = sol.params.get("c", 0.0)
    out.update({"in_slice_speed": float(np.linalg.norm(vh) / np.linalg.norm(Xh)),
                "ell_dot": elld, "expected_ell_dot": -c, "ell_dot_error": abs(elld + c),
                "slice_condition": slc.slice_condition(Xh)})
    return out


@dataclass
class SliceRPO:
    """A relative periodic orbit found as a periodic orbit of the reduced flow."""
    __repr__ = compact_repr   # one line, arrays by shape
    x_hat: np.ndarray                  # point on the slice and on the Poincaré section
    T: float
    theta: float                       # net group shift over one period (unwrapped)
    residual: float                    # ‖reduce(Φ_T(x̂)) − x̂‖ / ‖x̂‖
    multipliers: np.ndarray            # eigenvalues of the reduced monodromy (FD, declared step)
    iterations: int
    vectors: np.ndarray | None = None  # matching eigenvectors (columns)
    notes: list[str] = field(default_factory=list)

    def explain(self) -> str:
        mult = ", ".join(f"{m:.8g}" for m in self.multipliers)
        lines = [f"RPO in the slice: T = {self.T:.10f}, θ = {self.theta:.10f}, residual "
                 f"{self.residual:.2e} ({self.iterations} Gauss-Newton iterations); reduced "
                 f"monodromy eigenvalues [{mult}]"]
        return "\n".join(lines + [f"  note: {n}" for n in self.notes])


def find_rpo_in_slice(slc, integrator, section, x0, T0: float, *, tol: float, fd_step: float,
                      max_iterations: int, steady_tol: float) -> SliceRPO:
    """Periodic orbit of the reduced flow through the Poincaré ``section`` (hyperplane or any g):
    unknowns (x̂, T), equations reduce(Φ_T(x̂)) − x̂ = 0, slice condition ⟨x̂, t'⟩ = 0 and
    g(x̂) = 0, solved by Gauss–Newton (least squares: the reduced residual has one component
    that duplicates the slice condition) with a forward-difference Jacobian of declared step
    ``fd_step``; converged when the relative residual is ≤ ``tol``. ``integrator.flow`` must accept
    any T (ODE integrators). θ is the unwrapped net phase along the converged orbit.

    A relative equilibrium is a fixed point of the reduced flow, hence a "periodic orbit" of any
    period: a solution with in-slice speed ‖v̂(x̂)‖/‖v(x̂)‖ < ``steady_tol`` (declared) is refused
    (ConvergenceError), never returned as an orbit (found by the example, where the closest
    recurrence of a trajectory leaving the relative equilibrium converged onto it)."""
    x = np.asarray(slc.reduce(np.asarray(x0, float))[0], float)
    T = float(T0)

    def F(z):
        xx, TT = z[:-1], z[-1]
        y = slc.reduce(integrator.flow(xx, TT))[0]
        return np.concatenate([y - xx, [xx @ slc.tTp, section.g(xx)]])

    z = np.concatenate([x, [T]])
    it, res = 0, np.inf
    for it in range(1, max_iterations + 1):  # noqa: B007 -- read after the loop
        f = F(z)
        res = float(np.linalg.norm(f) / np.linalg.norm(z[:-1]))
        if res <= tol:
            break
        Jm = np.empty((len(f), len(z)))
        for j in range(len(z)):
            e = np.zeros_like(z)
            e[j] = fd_step * max(1.0, abs(z[j]))
            Jm[:, j] = (F(z + e) - f) / e[j]
        z = z + np.linalg.lstsq(Jm, -f, rcond=None)[0]
    else:
        from ..errors import ConvergenceError
        raise ConvergenceError(f"find_rpo_in_slice: residual {res:.2e} > tol after "
                               f"{max_iterations} iterations")
    x, T = z[:-1], float(z[-1])
    v_full = integrator.vector_field(x)
    speed = float(np.linalg.norm(slc.velocity(x, integrator.vector_field)[0])
                  / max(np.linalg.norm(v_full), 1e-300))
    if speed < steady_tol:
        from ..errors import ConvergenceError
        raise ConvergenceError(f"find_rpo_in_slice: converged to a relative equilibrium (in-slice "
                               f"speed / speed = {speed:.1e} < steady_tol = {steady_tol:g}), not an orbit",
                               fix="seed from a recurrence far from the relative equilibria")
    tr = integrator.integrate(x, T)
    st = slc.reduce_trajectory(tr.t, tr.X)
    theta = float((st.ell[-1] - st.ell[0]) * slc.k1)
    n = len(x)
    Mh = np.empty((n, n))
    for j in range(n):                    # central differences (O(h²) error, declared)
        e = np.zeros(n)
        e[j] = fd_step
        Mh[:, j] = (slc.reduce(integrator.flow(x + e, T))[0]
                    - slc.reduce(integrator.flow(x - e, T))[0]) / (2 * fd_step)
    mult, vecs = np.linalg.eig(Mh)
    order = np.argsort(-np.abs(mult))
    out = SliceRPO(x, T, theta, res, mult[order], it, vecs[:, order])
    out.notes.append(f"in-slice speed / speed = {speed:.3e} (relative equilibria refused below "
                     f"steady_tol = {steady_tol:g})")
    out.notes.append(f"reduced monodromy by central differences, h = {fd_step:g}: contains 1 (time) "
                     "and ~0 (the group direction removed by the slice)")
    out.notes.append(f"max phase jump between samples {st.max_phase_jump:.3f} rad; min border "
                     f"distance {st.border_distance.min():.3e}")
    return out
