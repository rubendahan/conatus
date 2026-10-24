"""Periodic orbits by orthogonal collocation, their continuation and Floquet multipliers.

Declared method (the only one here): **orthogonal collocation** as in AUTO (Doedel 1981; Doedel,
Keller & Kernévez 1991): τ = t/T ∈ [0, 1] split into ``ntst`` intervals, on each a Lagrange
polynomial of degree ``ncol`` through ncol + 1 equally spaced nodes, collocated at the ncol
Gauss–Legendre points:

    u'(τ_c) − T G(u(τ_c); p) = 0,   u(1) = u(0),   ∫₀¹ ⟨u(τ), u̇_ref(τ)⟩ dτ = 0,

the last one being the integral phase condition against the previous point of the branch (updated
after every accepted point). Unknowns z = (node values, T, p). The Jacobian is sparse (exact
J from the system at every collocation point). The mesh is uniform unless ``remesh_every`` is set:
then the mesh is re-equidistributed (declared) on the ncol-th derivative every ``remesh_every``
accepted points (the solution and tangent are re-interpolated and re-corrected).

**Floquet multipliers** come from the same discretization: on interval j the linearized
collocation equations give the map from the first node to the last, Φ_j; the monodromy matrix is
Φ_{N−1} ⋯ Φ_0 (condensation of parameters, as in AUTO). Products are renormalized to avoid
overflow; the trivial multiplier, exactly 1 for the continuous orbit, is reported: |μ_triv − 1| is
the accuracy of the multipliers (declared on every point).

Test functions along an orbit branch: ``lpc`` (fold of cycles,
t_p), ``bpc`` (Π s(μ_i − 1) over non-trivial multipliers: a second multiplier at +1 without a
fold, e.g. a symmetry-breaking pitchfork of cycles; a zero at an lpc is reported as lpc only), ``pd`` (Π s(μ_i + 1) over non-trivial multipliers), ``ns`` (Π_{i<j} s(μ_i μ_j − 1); a zero
with a real pair is a neutral saddle cycle and is rejected with that reason).
M must be the identity (ODE-type systems; DAE rows are refused explicitly).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..continuation.core import ContinuationSettings
from ..errors import ConvergenceError, OptionError, UnsupportedError, did_you_mean
from .codim2 import _s
from .driver import ArclengthDriver, Curve, CurvePoint
from .system import DynamicalSystem
from .._repr import compact_repr

__all__ = ["CollocationMesh", "PeriodicOrbitProblem", "continue_orbits", "floquet_from_collocation"]


def _lagrange(nodes: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Values l_k(x_c) and derivatives l_k'(x_c) of the Lagrange basis on ``nodes`` (rows c)."""
    m = nodes.size
    L = np.ones((x.size, m))
    D = np.zeros((x.size, m))
    for k in range(m):
        others = [j for j in range(m) if j != k]
        den = np.prod([nodes[k] - nodes[j] for j in others])
        for c, xc in enumerate(x):
            L[c, k] = np.prod([xc - nodes[j] for j in others]) / den
            s = 0.0
            for i in others:
                s += np.prod([xc - nodes[j] for j in others if j != i])
            D[c, k] = s / den
    return L, D


@dataclass
class CollocationMesh:
    """Mesh in τ = t/T ∈ [0, 1]: ``ntst`` intervals, ``ncol`` Gauss collocation points each."""
    __repr__ = compact_repr   # one line, arrays by shape

    ntst: int
    ncol: int
    tmesh: np.ndarray          # ntst + 1 interval ends in [0, 1]

    @classmethod
    def uniform(cls, ntst: int, ncol: int) -> CollocationMesh:
        """Uniform mesh of ``ntst`` intervals with ``ncol`` collocation points each."""
        from ..errors import check_count       # audit inputs: 4.5 was a bare TypeError of linspace
        ntst = check_count(ntst, "ntst", "CollocationMesh.uniform()")
        ncol = check_count(ncol, "ncol", "CollocationMesh.uniform()")
        if ntst < 2 or not 2 <= ncol <= 7:
            raise OptionError("collocation: ntst >= 2 and 2 <= ncol <= 7 required")
        return cls(ntst, ncol, np.linspace(0.0, 1.0, ntst + 1))

    @property
    def h(self) -> np.ndarray:
        """Interval lengths in τ."""
        return np.diff(self.tmesh)

    @property
    def npts(self) -> int:
        """Number of nodes, ntst·ncol + 1 (the last one repeats τ = 1)."""
        return self.ntst * self.ncol + 1

    def explain(self) -> str:
        """Mesh of the orbit collocation."""
        h = self.h
        kind = "uniform" if np.allclose(h, h[0], rtol=1e-12, atol=0.0) else f"adapted (h in [{h.min():.3g}, {h.max():.3g}])"
        return (f"CollocationMesh: {self.ntst} intervals x {self.ncol} collocation points "
                f"(Gauss–Legendre), {self.npts} nodes, {kind} in τ ∈ [0, 1]")

    def times(self) -> np.ndarray:
        """τ of every node (ntst·ncol + 1 values)."""
        s = np.arange(self.ncol) / self.ncol
        t = (self.tmesh[:-1, None] + self.h[:, None] * s[None, :]).ravel()
        return np.concatenate([t, [1.0]])


class PeriodicOrbitProblem:
    """Collocation equations of periodic orbits of ``system`` in the free parameter ``parameter``."""

    def __init__(self, system: DynamicalSystem, parameter: str, mesh: CollocationMesh, *,
                 par: Mapping[str, float] | None = None) -> None:
        if not system.identity_mass:
            raise UnsupportedError("periodic orbits: collocation needs M = I (DAE rows present)",
                                   fix="eliminate the algebraic unknowns, or use the ECS shooting "
                                       "tools (conatus.ecs) on the discrete flow map")
        system._check_parameter(parameter, "PeriodicOrbitProblem")
        self.sys, self.parameter, self.base = system, parameter, system._full(par)
        self.n = system.n
        self.set_mesh(mesh)
        self.uref_dot = None             # derivative of the reference orbit at collocation points

    def explain(self) -> str:
        """Unknowns and equations of the collocation problem."""
        return (f"PeriodicOrbitProblem: periodic orbits of an n = {self.n} system in "
                f"{self.parameter!r}\n"
                f"  unknowns: {self.nu} node values, the period T and {self.parameter}\n"
                f"  equations: du/dτ = T G(u) at the collocation points, u(1) = u(0), integral "
                f"phase condition against the reference orbit"
                + ("" if self.uref_dot is not None else " (reference not set yet)")
                + "\n  " + self.mesh.explain())

    # --- mesh ------------------------------------------------------------------------------------
    def set_mesh(self, mesh: CollocationMesh) -> None:
        """Install a mesh (Lagrange/Gauss tables, number of node unknowns)."""
        self.mesh = mesh
        m = mesh.ncol
        self.nodes = np.arange(m + 1) / m
        g, w = np.polynomial.legendre.leggauss(m)
        self.gauss = 0.5 * (g + 1.0)
        self.gw = 0.5 * w
        self.Lc, self.Dc = _lagrange(self.nodes, self.gauss)       # (m, m+1)
        self.nu = mesh.npts * self.n                              # node unknowns

    def _node_index(self, j: int, k: int) -> int:
        return j * self.mesh.ncol + k

    def split(self, z: np.ndarray):
        """(node values (npts, n), period T, parameter p) from z."""
        return z[:self.nu].reshape(self.mesh.npts, self.n), float(z[self.nu]), float(z[self.nu + 1])

    def pars(self, p: float) -> dict:
        """Parameter values of the system with the continuation parameter set to p."""
        q = dict(self.base)
        q[self.parameter] = p
        return q

    def _interval_nodes(self, U: np.ndarray) -> np.ndarray:
        m = self.mesh.ncol
        idx = np.arange(self.mesh.ntst)[:, None] * m + np.arange(m + 1)[None, :]
        return U[idx]                                            # (ntst, m+1, n)

    def colloc_values(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """u and du/dτ at the collocation points, (ntst, m, n) each."""
        Ui = self._interval_nodes(U)
        u = np.einsum("ck,jkn->jcn", self.Lc, Ui)
        du = np.einsum("ck,jkn->jcn", self.Dc, Ui) / self.mesh.h[:, None, None]
        return u, du

    def set_reference(self, U: np.ndarray) -> None:
        """Reference orbit of the integral phase condition (its τ-derivative is stored)."""
        self.uref_dot = self.colloc_values(U)[1]

    # --- equations -------------------------------------------------------------------------------
    def residual(self, z: np.ndarray) -> np.ndarray:
        """Collocation equations, periodicity u(1) = u(0) and the phase condition."""
        U, T, p = self.split(z)
        u, du = self.colloc_values(U)
        nt, m, n = u.shape
        g = self.sys.G_many(u.reshape(-1, n), self.pars(p)).reshape(nt, m, n)
        col = (du - T * g).ravel()
        per = U[-1] - U[0]
        phase = np.sum(self.gw[None, :, None] * self.mesh.h[:, None, None] * u * self.uref_dot)
        return np.concatenate([col, per, [phase]])

    def jacobian(self, z: np.ndarray, with_p: bool = True) -> sp.csr_matrix:
        """Sparse Jacobian of ``residual`` in (nodes, T[, p]) (exact J at the collocation points)."""
        U, T, p = self.split(z)
        u, _ = self.colloc_values(U)
        nt, m, n = u.shape
        par = self.pars(p)
        flat = u.reshape(-1, n)
        Jc = self.sys.J_many(flat, par).reshape(nt, m, n, n)
        g = self.sys.G_many(flat, par).reshape(nt, m, n)
        rows, cols, vals = [], [], []
        eye = np.eye(n)
        h = self.mesh.h
        for j in range(nt):
            for c in range(m):
                r0 = (j * m + c) * n
                for k in range(m + 1):
                    blk = self.Dc[c, k] / h[j] * eye - T * self.Lc[c, k] * Jc[j, c]
                    c0 = self._node_index(j, k) * n
                    ri, ci = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
                    rows.append((r0 + ri).ravel())
                    cols.append((c0 + ci).ravel())
                    vals.append(blk.ravel())
        nrow_col = nt * m * n
        # d/dT
        rows.append(np.arange(nrow_col))
        cols.append(np.full(nrow_col, self.nu))
        vals.append(-g.ravel())
        if with_p:
            gp = self.sys.Gp_many(flat, self.parameter, par)
            rows.append(np.arange(nrow_col))
            cols.append(np.full(nrow_col, self.nu + 1))
            vals.append((-T * gp).ravel())
        # periodicity
        last = (self.mesh.npts - 1) * n
        rows += [nrow_col + np.arange(n), nrow_col + np.arange(n)]
        cols += [last + np.arange(n), np.arange(n)]
        vals += [np.ones(n), -np.ones(n)]
        # phase: sum_j sum_c w_c h_j sum_k L[c,k] U_{j,k} . uref_dot_{j,c}
        prow = nrow_col + n
        coef = np.einsum("c,j,ck,jcn->jkn", self.gw, h, self.Lc, self.uref_dot)
        Uidx = self._interval_nodes(np.arange(self.nu).reshape(self.mesh.npts, n))
        rows.append(np.full(coef.size, prow))
        cols.append(Uidx.ravel())
        vals.append(coef.ravel())
        ncol_ = self.nu + (2 if with_p else 1)
        A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(prow + 1, ncol_))
        return A.tocsr()

    # --- Floquet -----------------------------------------------------------------------------------
    def monodromy(self, z: np.ndarray) -> tuple[np.ndarray, float]:
        """Monodromy matrix (renormalized product) and log of the scale factor removed."""
        U, T, p = self.split(z)
        u, _ = self.colloc_values(U)
        nt, m, n = u.shape
        Jc = self.sys.J_many(u.reshape(-1, n), self.pars(p)).reshape(nt, m, n, n)
        P = np.eye(n)
        logscale = 0.0
        eye = np.eye(n)
        for j in range(nt):
            A = np.zeros((m * n, (m + 1) * n))
            for c in range(m):
                for k in range(m + 1):
                    A[c * n:(c + 1) * n, k * n:(k + 1) * n] = (self.Dc[c, k] / self.mesh.h[j] * eye
                                                              - T * self.Lc[c, k] * Jc[j, c])
            Phi_all = -np.linalg.solve(A[:, n:], A[:, :n])           # nodes 1..m from node 0
            Phi = Phi_all[(m - 1) * n:]
            P = Phi @ P
            s = np.abs(P).max()
            if s > 1e100 or s < 1e-100:
                P /= s
                logscale += np.log(s)
        return P, logscale

    def multipliers(self, z: np.ndarray) -> tuple[np.ndarray, float, int]:
        """Floquet multipliers (sorted by decreasing modulus), |μ_triv − 1| and the trivial index."""
        P, ls = self.monodromy(z)
        mu = np.linalg.eigvals(P) * np.exp(ls)
        order = np.argsort(-np.abs(mu))
        mu = mu[order]
        itriv = int(np.argmin(np.abs(mu - 1.0)))
        return mu, float(abs(mu[itriv] - 1.0)), itriv

    def floquet_vector(self, z: np.ndarray, target: complex) -> np.ndarray:
        """Eigenvector of the monodromy for the multiplier closest to ``target``, propagated to
        every node (rows = nodes), real part if the multiplier is real."""
        U, T, p = self.split(z)
        u, _ = self.colloc_values(U)
        nt, m, n = u.shape
        Jc = self.sys.J_many(u.reshape(-1, n), self.pars(p)).reshape(nt, m, n, n)
        P, _ = self.monodromy(z)
        w, V = np.linalg.eig(P)
        v = V[:, np.argmin(np.abs(w - target))]
        v = np.real(v) if abs(np.imag(target)) == 0 else v
        out = np.zeros((self.mesh.npts, n), dtype=v.dtype)
        out[0] = v
        eye = np.eye(n)
        for j in range(nt):
            A = np.zeros((m * n, (m + 1) * n))
            for c in range(m):
                for k in range(m + 1):
                    A[c * n:(c + 1) * n, k * n:(k + 1) * n] = (self.Dc[c, k] / self.mesh.h[j] * eye
                                                              - T * self.Lc[c, k] * Jc[j, c])
            vals = -np.linalg.solve(A[:, n:], A[:, :n] @ out[j * m])
            out[j * m + 1:(j + 1) * m + 1] = vals.reshape(m, n)
            s = np.abs(out[(j + 1) * m]).max()
            if s > 1e50:
                out[:(j + 1) * m + 1] /= s
        return out

    # --- mesh adaptation -------------------------------------------------------------------------
    def interpolate(self, U: np.ndarray, t: np.ndarray) -> np.ndarray:
        """u(τ) at arbitrary τ ∈ [0, 1] from the collocation polynomial."""
        tm = self.mesh.tmesh
        j = np.clip(np.searchsorted(tm, t, side="right") - 1, 0, self.mesh.ntst - 1)
        s = (t - tm[j]) / self.mesh.h[j]
        Ui = self._interval_nodes(U)
        out = np.zeros((t.size, U.shape[1]), dtype=U.dtype)
        for i, (jj, ss) in enumerate(zip(j, s, strict=True)):
            L, _ = _lagrange(self.nodes, np.array([ss]))
            out[i] = L[0] @ Ui[jj]
        return out

    def equidistributed_mesh(self, U: np.ndarray) -> CollocationMesh:
        """New mesh equidistributing |u^(ncol)|^(1/(ncol+1)) (AUTO-style), same ntst, ncol."""
        m = self.mesh.ncol
        Ui = self._interval_nodes(U)
        # ncol-th derivative is constant per interval: m! * leading coefficient / h^m
        from math import factorial
        V = np.vander(self.nodes, m + 1, increasing=True)
        coef = np.linalg.solve(V, Ui)                             # (ntst, m+1, n)
        dm = factorial(m) * np.abs(coef[:, m, :]).max(axis=1) / self.mesh.h ** m
        # density floor (declared): 1e-3 of the maximum, so no interval grows without bound
        dens = dm ** (1.0 / (m + 1))
        dens = np.maximum(dens, 1e-3 * dens.max())
        cum = np.concatenate([[0.0], np.cumsum(dens * self.mesh.h)])
        cum /= cum[-1]
        new = np.interp(np.linspace(0, 1, self.mesh.ntst + 1), cum, self.mesh.tmesh)
        new[0], new[-1] = 0.0, 1.0
        return CollocationMesh(self.mesh.ntst, m, new)

    def remesh(self, z: np.ndarray, mesh: CollocationMesh, t: np.ndarray | None = None):
        """Interpolate z (and a tangent t) onto ``mesh`` and install it; returns (z_new, t_new)."""
        U, T, p = self.split(z)
        taus = CollocationMesh(mesh.ntst, mesh.ncol, mesh.tmesh).times()
        Un = self.interpolate(U, taus)
        tn = None
        if t is not None:
            Ut = t[:self.nu].reshape(self.mesh.npts, self.n)
            tn = np.concatenate([self.interpolate(Ut, taus).ravel(), t[self.nu:]])
        self.set_mesh(mesh)
        return np.concatenate([Un.ravel(), [T, p]]), tn


def floquet_from_collocation(problem: PeriodicOrbitProblem, z: np.ndarray) -> dict:
    """Floquet multipliers of the orbit z, the accuracy |μ_triv − 1| and the trivial index."""
    mu, err, it = problem.multipliers(z)
    return {"multipliers": mu, "trivial_error": err, "trivial_index": it}


#: Multipliers with |μ| < RESOLVED_REL x max(1, |μ|max) are not resolved by the collocation product
#: (counted, left out of the tests; declared in ``conatus.tolerances.DECLARED``).
RESOLVED_REL = 1e-8
#: Stability count of an orbit (declared in ``conatus.tolerances.DECLARED``): a nontrivial
#: multiplier counts as unstable when |μ| > 1 + MARGINAL_MULTIPLIER_TOL; those with
#: ||μ| − 1| <= MARGINAL_MULTIPLIER_TOL are near-marginal (stored per point, listed by explain()).
#: A plain |μ| > 1 counted a multiplier 1 + 1e-12 (round-off at a PD/NS/LPC point) as unstable,
#: while equilibria have ``marginal_tol``. Above the collocation accuracy of the multipliers (the
#: stored |μ_triv − 1| is its measure, typically <= 1e-8 on the meshes of the tests).
MARGINAL_MULTIPLIER_TOL = 1e-6


def continue_orbits(problem: PeriodicOrbitProblem, z0: np.ndarray, settings: ContinuationSettings, *,
                    direction: np.ndarray | None = None, bounds: Mapping[str, tuple] | None = None,
                    tests: tuple[str, ...] = ("lpc", "pd", "ns", "bpc"), remesh_every: int | None = None,
                    period_max: float | None = None, fix_start: bool = True) -> Curve:
    """Pseudo-arclength branch of periodic orbits from z0 = (nodes, T, p).

    ``direction``: initial tangent guess (default: +p). ``fix_start``: first correct z0 at fixed
    p (Newton on the square collocation system, phase against z0 itself) — needed when z0 comes
    from a predictor (Hopf normal form, PD doubling). ``remesh_every``: re-equidistribute the mesh
    every that many points (None: fixed mesh, declared). ``period_max``: stop when T exceeds it
    (homoclinic approach). ``bounds``: {parameter name and/or "T": (lo, hi)}; any other key is
    refused (it used to be dropped without a word, so ``{"p": ...}`` bounded nothing).
    """
    unknown = set(tests) - {"lpc", "pd", "ns", "bpc"}
    if unknown:
        raise OptionError(f"continue_orbits: unknown test(s) {sorted(unknown)}",
                          fix="tests among 'lpc', 'pd', 'ns', 'bpc'")
    pb = problem
    if bounds is not None:
        if not isinstance(bounds, Mapping):
            raise OptionError(f"continue_orbits: bounds={bounds!r} must be a dict",
                              fix=f"bounds={{{pb.parameter!r}: (lo, hi)}} and/or {{'T': (lo, hi)}}")
        bad = [k for k in bounds if k not in (pb.parameter, "T")]
        if bad:
            raise OptionError(f"continue_orbits: bounds on unknown name(s) {bad}",
                              why="only the continuation parameter and the period T can be bounded",
                              fix=f"keys {pb.parameter!r} and/or 'T'."
                                  + did_you_mean(str(bad[0]), (pb.parameter, "T")))
    z0 = np.asarray(z0, float)
    pb.set_reference(pb.split(z0)[0])
    if fix_start:
        z0 = _correct_fixed_p(pb, z0, settings)
    notes = [f"periodic orbits of {pb.sys.origin} in {pb.parameter}: orthogonal collocation, "
             f"ntst = {pb.mesh.ntst}, ncol = {pb.mesh.ncol} (Gauss points), integral phase condition",
             "mesh: " + ("uniform, fixed" if remesh_every is None else
                         f"re-equidistributed every {remesh_every} points on |u^(ncol)|^(1/(ncol+1)) "
                         "(density floor 1e-3 of its maximum), then re-corrected on the arclength "
                         "hyperplane"),
             "arclength norm: nodes weighted 1/npts (discrete L2 in τ), T and p weight 1",
             "Floquet multipliers from the collocation monodromy (condensation); "
             "|μ_triv − 1| stored on every point; multipliers below "
             f"{RESOLVED_REL:g} x max(1, |μ|max) are unresolved (count stored) and left out of the tests",
             f"stability: 'unstable' counts nontrivial |μ| > 1 + {MARGINAL_MULTIPLIER_TOL:g}; "
             f"||μ| − 1| <= {MARGINAL_MULTIPLIER_TOL:g} is near-marginal (stored as 'marginal', listed "
             "below), not counted"]
    all_points, all_events, all_rej, all_ref = [], [], [], []
    t_dir = direction
    remaining = settings.max_points
    cv = None
    while True:
        seg = settings if remesh_every is None else _with_points(settings, min(remesh_every, remaining))
        cv = _run_segment(pb, z0, seg, t_dir, tests, bounds, period_max)
        off = len(all_points) - (1 if all_points else 0)
        pts = cv.points if not all_points else cv.points[1:]
        for p_ in pts:
            p_.s += all_points[-1].s if all_points else 0.0
        all_points += pts
        for e in cv.events:
            e.between = (e.between[0] + off, e.between[1] + off)
        all_events += cv.events
        all_rej += cv.rejected
        all_ref += cv.refused
        remaining -= len(cv.points) - 1
        if remesh_every is None or remaining <= 0 or not cv.stop_reason.startswith("max_points"):
            break
        last = cv.points[-1]
        old_mesh = pb.mesh
        z_new, t_new = pb.remesh(last.z, pb.equidistributed_mesh(pb.split(last.z)[0]), last.tangent)
        drv = ArclengthDriver(pb.residual, pb.jacobian, settings,
                              weights=np.concatenate([np.full(pb.nu, 1.0 / pb.mesh.npts), [1.0, 1.0]]))
        pb.set_reference(pb.split(z_new)[0])
        try:
            # re-correct on the hyperplane through the interpolated point (arclength corrector)
            z0 = drv.correct(z_new, t_new / drv.norm(t_new))[0]
            t_dir = t_new
        except ConvergenceError as e:
            # remeshing refused: keep the previous mesh and point (declared in the notes)
            notes.append(f"remesh refused after point {len(all_points) - 1} "
                         f"({str(e).splitlines()[0]}); mesh kept")
            pb.set_mesh(old_mesh)
            pb.set_reference(pb.split(last.z)[0])
            z0, t_dir = last.z, last.tangent
    near = [(i, q.data["marginal"]) for i, q in enumerate(all_points)
            if q.data is not None and len(q.data.get("marginal", ()))]
    if near:
        shown = "; ".join(f"point {i}: " + ", ".join(f"{m:.10g}" for m in ms) for i, ms in near[:5])
        notes.append(f"near-marginal multipliers on {len(near)} point(s), not counted unstable: {shown}"
                     + (" ..." if len(near) > 5 else ""))
    out = Curve(kind="periodic orbit", names={pb.parameter: pb.nu + 1, "T": pb.nu},
                points=all_points, events=all_events, rejected=all_rej, refused=all_ref,
                stop_reason=cv.stop_reason if remaining > 0 else f"max_points = {settings.max_points} reached",
                settings=settings, notes=notes, context=pb)
    return out


def _with_points(cfg: ContinuationSettings, k: int) -> ContinuationSettings:
    from dataclasses import replace
    return replace(cfg, max_points=max(1, k))


def _correct_fixed_p(pb: PeriodicOrbitProblem, z: np.ndarray, cfg: ContinuationSettings) -> np.ndarray:
    """Newton on the collocation system at fixed p (square: nodes + T)."""
    z = z.copy()
    pb.set_reference(pb.split(z)[0])
    for k in range(cfg.max_newton + 1):
        r = pb.residual(z)
        err = float(np.max(np.abs(r)))
        if err <= cfg.tolerance:
            return z
        if k == cfg.max_newton or not np.isfinite(err):
            break
        A = pb.jacobian(z, with_p=False).tocsc()
        dz = spla.spsolve(A, -r)
        z[:pb.nu + 1] += dz
    raise ConvergenceError(f"periodic orbit: fixed-parameter Newton not converged (max|R| = {err:.2e})",
                           fix="smaller predictor amplitude, finer mesh, or larger max_newton")


def _run_segment(pb, z0, cfg, t_dir, tests, bounds, period_max) -> Curve:
    nu = pb.nu

    def H(z):
        return pb.residual(z)

    def Hz(z):
        return pb.jacobian(z)

    def data(z):
        mu, err, it = pb.multipliers(z)
        nontriv = np.delete(mu, it)
        # multipliers below RESOLVED_REL x max(1, |μ|max) are not resolved by the product of the
        # interval maps (round-off ~ eps |μ|max): excluded from the test functions (declared)
        keep = np.abs(nontriv) >= RESOLVED_REL * max(1.0, float(np.abs(mu).max()))
        dist = np.abs(nontriv) - 1.0
        return {"multipliers": mu, "trivial_error": err, "nontrivial": nontriv[keep],
                "unresolved": int(np.sum(~keep)),
                "unstable": int(np.sum(dist > MARGINAL_MULTIPLIER_TOL)),
                "marginal": nontriv[np.abs(dist) <= MARGINAL_MULTIPLIER_TOL]}

    def t_lpc(pt: CurvePoint) -> float:
        return float(pt.tangent[-1])

    def t_pd(pt: CurvePoint) -> float:
        return float(np.real(np.prod(_s(pt.data["nontrivial"] + 1.0))))

    def t_ns(pt: CurvePoint) -> float:
        mu = pt.data["nontrivial"]
        if mu.size < 2:
            return 1.0
        prod = mu[:, None] * mu[None, :] - 1.0
        iu = np.triu_indices(mu.size, 1)
        return float(np.real(np.prod(_s(prod[iu]))))

    def t_bpc(pt: CurvePoint) -> float:
        return float(np.real(np.prod(_s(pt.data["nontrivial"] - 1.0))))

    all_t = {"lpc": t_lpc, "pd": t_pd, "ns": t_ns, "bpc": t_bpc}

    def classify(name, ev):
        mu = ev.point.data["nontrivial"]
        if name == "lpc":
            return "lpc", None
        if name == "bpc":
            # a fold of cycles also puts a multiplier at +1: reported once, as lpc
            if "tp_sign_change" in ev.data:
                return None, "coincides with a fold of cycles"
            k = int(np.argmin(np.abs(mu - 1.0)))
            ev.detail = f"second multiplier μ = {mu[k].real:+.10f} at +1 (branch point of cycles)"
            return "bpc", None
        if name == "pd":
            k = int(np.argmin(np.abs(mu + 1.0)))
            ev.data["multiplier"] = mu[k]
            ev.detail = f"μ = {mu[k].real:+.10f}{mu[k].imag:+.1e}i, |μ_triv − 1| = {ev.point.data['trivial_error']:.1e}"
            return "pd", None
        prod = np.abs(mu[:, None] * mu[None, :] - 1.0) + np.diag(np.full(mu.size, np.inf))
        i, j = np.unravel_index(np.argmin(prod), prod.shape)
        pair = mu[[i, j]]
        ev.data["classification"] = (f"Neimark-Sacker if |Im μ| > hopf_imag_tol = "
                                     f"{cfg.hopf_imag_tol:g} x max(1, |μ|) for the pair of product "
                                     "closest to 1, else neutral saddle cycle")
        # same relative threshold as the Hopf/neutral-saddle split of equilibria (was an absolute
        # 1e-8, identical for |μ| <= 1)
        if np.all(np.abs(pair.imag) > cfg.hopf_imag_tol * max(1.0, float(np.abs(pair).max()))):
            ang = float(abs(np.angle(pair[0])))
            ev.data.update(multiplier=pair[0], angle=ang)
            ev.detail = (f"μ = exp(±{ang:.10f} i), |μ| = {abs(pair[0]):.12f}, "
                         f"|μ_triv − 1| = {ev.point.data['trivial_error']:.1e}")
            return "ns", None
        return None, f"neutral saddle cycle (real pair {pair[0].real:.4g}, {pair[1].real:.4g})"

    def on_accept(pt):
        pb.set_reference(pb.split(pt.z)[0])

    def stop(pt):
        T = pt.z[nu]
        if period_max is not None and T > period_max:
            return f"T = {T:.6g} > period_max = {period_max:g}"
        return None

    # norm: discrete L2 in τ on the nodes (weight 1/npts), 1 on T and p (declared in the notes)
    w = np.concatenate([np.full(nu, 1.0 / pb.mesh.npts), [1.0, 1.0]])
    drv = ArclengthDriver(H, Hz, cfg, tests={k: all_t[k] for k in tests}, data=data,
                          classify=classify, on_accept=on_accept, weights=w)
    if t_dir is None:
        t_dir = np.zeros(z0.size)
        t_dir[-1] = 1.0
    # bounds keys were checked by continue_orbits (parameter name and/or "T")
    return drv.run(z0, t_dir, kind="periodic orbit", names={pb.parameter: nu + 1, "T": nu},
                   bounds=dict(bounds) if bounds else None, stop=stop)
