"""Heteroclinic and homoclinic connections between equilibria by shooting with projection
boundary conditions. Notes: Beyn, IMA J. Numer. Anal.
10, 379–405 (1990) (projection boundary conditions); Doedel & Friedman (1989).

A connecting orbit x(t) from the equilibrium x₋ to x₊ is approximated on a finite interval [0, T]:

* **left end**: x(0) = P(s₀) on the unstable manifold of x₋, from its parametrization
  (:func:`~conatus.manifolds.unstable_manifold`) at a declared radius ρ (|s₀| = ρ). The unknown
  is the position on the fundamental loop (d = 2: angle φ, s₀ = ρ e^{iφ}; d = 1: branch ±ρ,
  fixed) — every orbit of the manifold crosses that loop once;
* **right end**: x(T) − x₊ has no component along the unstable subspace of x₊ (projection
  condition Ψᵀ M (x(T) − x₊) = 0, Ψ = real basis of the left unstable eigenvectors, strictly
  unstable eigenvalues Re λ > ``unstable_tol``, declared);
* the remaining unknowns are the loop angle and/or system parameters (``Lorenz``: r for the
  homoclinic orbit of the origin). #unknowns = #projection conditions (square system).

Scalar problems are solved by bracketing (sign scan on a declared grid) and Brent; square
systems of larger size by Newton with a forward-difference Jacobian of declared step. The error
of the truncation to [0, T] is not hidden: :class:`Connection` reports the end gap
‖x(T) − x₊‖, the left-end manifold defect and, from :func:`connection_vs_T`, how the unknowns
converge as T grows (they converge exponentially for a transversal connection).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.optimize as so

from ..errors import ConvergenceError, OptionError
from .._repr import compact_repr

__all__ = ["Connection", "unstable_left_basis", "projection_gap", "find_connection", "equilibrium_connection",
           "scan_connection", "connection_vs_T"]


def unstable_left_basis(system, x_plus, *, unstable_tol: float, par=None) -> np.ndarray:
    """Real basis Ψ (n, m) of the left eigenvectors of the strictly unstable eigenvalues of
    (J(x₊), M): Ψᵀ M δx = 0 ⇔ δx has no unstable component."""
    import scipy.linalg as sla
    J = system.J(x_plus, par)
    M = system.M.toarray() if hasattr(system.M, "toarray") else np.asarray(system.M, float)
    w, vl, _ = sla.eig(J, M, left=True, right=True)
    keep = np.isfinite(w) & (w.real > unstable_tol)
    cols = []
    for i in np.flatnonzero(keep):
        if w[i].imag < -1e-12 * max(1.0, abs(w[i])):
            continue                                # conjugate of a kept vector
        cols.append(np.real(vl[:, i]))
        if abs(w[i].imag) > 1e-12 * max(1.0, abs(w[i])):
            cols.append(np.imag(vl[:, i]))
    if not cols:
        return np.zeros((len(x_plus), 0))
    Q, _ = np.linalg.qr(np.array(cols).T)
    return Q


def projection_gap(Psi, M, x_end, x_plus) -> np.ndarray:
    """Ψᵀ M (x(T) − x₊) (the projection boundary condition)."""
    Md = M.toarray() if hasattr(M, "toarray") else np.asarray(M, float)
    return Psi.T @ (Md @ (np.asarray(x_end) - np.asarray(x_plus)))


@dataclass
class Connection:
    """A connecting orbit on [0, T] with its evidence."""
    __repr__ = compact_repr   # one line, arrays by shape
    unknowns: np.ndarray
    T: float
    trajectory: object
    projection: np.ndarray              # Ψᵀ M (x(T) − x₊) at the solution
    end_gap: float                      # ‖x(T) − x₊‖ / max(1, ‖x₊‖)
    min_gap: float                      # min_t ‖x(t) − x₊‖ / max(1, ‖x₊‖)
    start_defect: float                 # invariance defect of the parametrization at x(0)
    method: str
    notes: list[str] = field(default_factory=list)

    def explain(self) -> str:
        u = ", ".join(f"{q:.12g}" for q in self.unknowns)
        lines = [f"Connection on [0, {self.T:g}]: unknowns [{u}] ({self.method}); projection "
                 f"|Ψᵀ M (x(T) − x₊)| = {np.linalg.norm(self.projection):.2e}; end gap "
                 f"{self.end_gap:.2e}; min gap along the orbit {self.min_gap:.2e}; left-end "
                 f"manifold defect {self.start_defect:.2e}"]
        return "\n".join(lines + [f"  note: {n}" for n in self.notes])


def _end_point(info, T):
    """x at the right end: x(T), or (``end_radius`` δ in the setup) the first inward crossing of
    the sphere ‖x − x₊‖ = δ on [0, T] after the orbit has left it (located on the dense output,
    Brent ``end_xtol``), so the projection is taken at a fixed distance from x₊ (declared)."""
    delta = info.get("end_radius")
    if delta is None:
        return info["integrator"].flow(info["x0"], T), T, True
    from .poincare import PoincareSection, poincare_crossings
    traj = info["integrator"].integrate(info["x0"], T)
    xp = np.asarray(info["x_plus"])
    sec = PoincareSection(lambda x: float(np.linalg.norm(np.asarray(x) - xp) - delta), direction=-1,
                          name=f"sphere |x - x+| = {delta:g}")
    d = np.linalg.norm(traj.X - xp, axis=1)
    out = np.flatnonzero(d > delta)
    if not len(out):
        raise ConvergenceError(f"connection: the orbit never leaves the sphere of radius {delta:g}")
    cr = poincare_crossings(traj, sec, xtol=info.get("end_xtol", 1e-12), t_min=traj.t[out[0]])
    if not len(cr.t):
        return traj.X[-1], T, False               # never reached the sphere: flagged, not a root
    return cr.X[0], float(cr.t[0]), True


def _covering_time(integrator, t_end: float) -> float:
    """Time to integrate so that the trajectory covers [0, t_end]. A fixed-step map
    (``SolverIntegrator``, attribute ``dt``) only reaches its step grid and refuses any other end
    time; the first inward crossing of the end sphere is located on the dense
    output, between two steps, so the trajectory runs to the first step at or after it (find_connection with end_radius on a SolverIntegrator was refused)."""
    import math
    dt = getattr(integrator, "dt", None)
    if dt is None:
        return float(t_end)
    return max(1, math.ceil(t_end / dt - 1e-9)) * dt


def _evaluate(setup: Callable, q, T):
    """setup(q) -> dict(x0, integrator, x_plus, Psi, M, start_defect[, end_radius]);
    returns (projection, info) with info['t_end'] the time of the right end."""
    info = setup(np.atleast_1d(np.asarray(q, float)))
    x_end, t_end, reached = _end_point(info, T)
    info["t_end"], info["x_end"], info["reached"] = t_end, x_end, reached
    return projection_gap(info["Psi"], info["M"], x_end, info["x_plus"]), info


def scan_connection(setup: Callable, grid, T: float, *,
                    values: list | None = None) -> list[tuple[float, float]]:
    """Brackets [q_i, q_{i+1}] of a scalar unknown where the (single) projection condition
    changes sign on the declared grid, both ends having reached the end sphere (if one is
    declared: a sign change between an orbit that reached it and one that did not is not a
    root). ``values`` (a list) receives (q, projection, reached, t_end) for every grid point."""
    vals, ok = [], []
    for q in grid:
        p, info = _evaluate(setup, q, T)
        if p.size != 1:
            raise OptionError(f"scan_connection: {p.size} projection conditions for one unknown")
        vals.append(float(p[0]))
        ok.append(bool(info["reached"]))
        if values is not None:
            values.append((float(q), float(p[0]), bool(info["reached"]), float(info["t_end"])))
    return [(float(grid[i]), float(grid[i + 1])) for i in range(len(grid) - 1)
            if ok[i] and ok[i + 1] and np.sign(vals[i]) != np.sign(vals[i + 1])]


def find_connection(setup: Callable, T: float, *, bracket=None, q0=None, xtol: float,
                    fd_step: float | None = None, max_iterations: int = 50) -> Connection:
    """Solve the projection conditions for the unknowns q (see module doc).

    ``bracket=(a, b)`` (scalar q, sign change required): Brent with ``xtol``. ``q0`` (vector):
    Newton with forward differences of step ``fd_step`` (required), stopped at
    ‖Δq‖ ≤ ``xtol``·max(1, ‖q‖)."""
    if bracket is not None:
        f = lambda q: float(_evaluate(setup, q, T)[0][0])        # noqa: E731
        fa, fb = f(bracket[0]), f(bracket[1])
        if np.sign(fa) == np.sign(fb):
            raise ConvergenceError(f"find_connection: no sign change on {bracket} "
                                   f"({fa:.3e}, {fb:.3e})", fix="scan_connection on a finer grid")
        q = np.array([so.brentq(f, bracket[0], bracket[1], xtol=xtol, maxiter=max_iterations)])
        method = f"Brent on {bracket}, xtol = {xtol:g}"
    else:
        if q0 is None or fd_step is None:
            raise OptionError("find_connection: give bracket=(a, b), or q0 with fd_step (Newton)")
        q = np.atleast_1d(np.asarray(q0, float)).copy()
        for _it in range(max_iterations):
            p, _ = _evaluate(setup, q, T)
            if p.size != q.size:
                raise OptionError(f"find_connection: {p.size} conditions for {q.size} unknowns")
            Jm = np.empty((p.size, q.size))
            for j in range(q.size):
                e = np.zeros_like(q)
                e[j] = fd_step
                Jm[:, j] = (_evaluate(setup, q + e, T)[0] - p) / fd_step
            dq = np.linalg.solve(Jm, -p)
            q = q + dq
            if np.linalg.norm(dq) <= xtol * max(1.0, np.linalg.norm(q)):
                break
        else:
            raise ConvergenceError(f"find_connection: Newton not converged in {max_iterations} iterations")
        method = f"Newton, forward differences h = {fd_step:g}, xtol = {xtol:g}"
    p, info = _evaluate(setup, q, T)
    if not info["reached"]:
        raise ConvergenceError(f"find_connection: the orbit at the solution never reaches the end "
                               f"sphere |x - x+| = {info.get('end_radius')} on [0, {T:g}] (no connection)")
    t_end = info["t_end"]
    traj = info["integrator"].integrate(info["x0"], _covering_time(info["integrator"], t_end))
    xp = np.asarray(info["x_plus"])
    sc = max(1.0, float(np.linalg.norm(xp)))
    keep = traj.t <= t_end * (1 + 1e-12) + 1e-300          # samples on [0, t_end] only
    gaps = np.linalg.norm(traj.X[keep] - xp, axis=1) / sc
    con = Connection(q, float(t_end), traj, p,
                     float(np.linalg.norm(info["x_end"] - xp) / sc), float(gaps.min()),
                     float(info.get("start_defect", np.nan)), method)
    if info.get("end_radius") is not None:
        con.notes.append(f"right end: first inward crossing of |x - x+| = {info['end_radius']:g} "
                         f"(t = {t_end:.10g}, searched on [0, {T:g}])")
    con.notes.append("single shooting with projection boundary conditions (Beyn 1990); "
                     "left end on the parametrized unstable manifold")
    if "note" in info:
        con.notes.append(info["note"])
    return con


def connection_vs_T(setup: Callable, Ts, *, bracket_of: Callable, xtol: float) -> list[Connection]:
    """The same scalar connection for increasing T (``bracket_of(T, previous)`` gives the bracket):
    the convergence of the unknown with T is the truncation error, reported."""
    out, prev = [], None
    for T in Ts:
        con = find_connection(setup, T, bracket=bracket_of(T, prev), xtol=xtol)
        out.append(con)
        prev = con
    return out


def equilibrium_connection(system, integrator, x_minus, x_plus, *, order: int, degree: int,
                           rho: float, eigvec_scale: float, unstable_tol: float,
                           branch: int = 1, target=None, end_radius: float | None = None,
                           par=None) -> Callable:
    """Setup for :func:`find_connection` from the equilibrium ``x_minus`` to ``x_plus`` of a
    parameter-free system (the unstable manifold of x₋ is parametrized once, to ``order``).

    Unknowns q: the angle φ on the loop |s| = ρ if W^u(x₋) is 2D (s = ρ(cos φ, sin φ)); none for
    a 1D manifold (``branch`` ±1 picks the side; use a ``target`` shift then). ``target``:
    optional callable ℓ ↦ x₊(ℓ) for a target *group orbit* (e.g. ``lambda l: ops.translate(E, l)``):
    ℓ is then the last unknown, and Ψ is recomputed at x₊(ℓ)."""
    from .parametrization import unstable_manifold
    man = unstable_manifold(system, x_minus, order=order, degree=degree, eigvec_scale=eigvec_scale,
                            unstable_tol=unstable_tol, par=par)
    if man.dim > 2:
        raise OptionError(f"equilibrium_connection: W^u(x-) has dimension {man.dim} > 2")
    Psi_fixed = None if target is not None else unstable_left_basis(
        system, x_plus, unstable_tol=unstable_tol, par=par)

    def setup(q):
        q = np.atleast_1d(q)
        k = 0
        if man.dim == 2:
            s = rho * np.array([np.cos(q[0]), np.sin(q[0])])
            k = 1
        else:
            s = np.array([branch * rho])
        xp = x_plus if target is None else target(float(q[k]))
        Psi = Psi_fixed if target is None else unstable_left_basis(system, xp, unstable_tol=unstable_tol,
                                                                     par=par)
        return dict(x0=man(s), integrator=integrator, x_plus=xp, Psi=Psi, M=system.M,
                    start_defect=man.defect(s), end_radius=end_radius,
                    note=f"left end: order-{order} parametrization, |s| = {rho:g}")
    setup.manifold = man
    return setup
