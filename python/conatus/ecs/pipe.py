"""Travelling waves of pipe flow in their symmetry subspaces: constant-flux pipe, m-fold and
shift-reflect subspaces, Newton-Krylov-hookstep on the shooting map in the comoving frame.

**Problem** (radius R = 1, z periodic of length Lz = 2 pi / alpha, laminar centreline velocity 1)::

    dt(u) - lap(u)/Re + grad(p) - G e_z = -u.grad(u),   div(u) = 0,   u = 0 at r = 1,
    integ(u_z) = Q = (pi / 2) Lz        (constant flux: G is an unknown of the step),

with Re = U_cl R / nu = U_b D / nu (the bulk Reynolds number of Wedin & Kerswell 2004 and of
Willis, Cvitanovic & Avila 2013: U_b = U_cl / 2, D = 2 R). Laminar flow: u = (1 - r^2) e_z,
G_lam = 4 / Re. Subspace: ``AzimuthalMFold(phi, m0)`` (reduced groups, F on the sector) and
``ShiftReflect(phi, z)`` (declared projection of the step, ``timestep/symmetry_projection.py``).

**Travelling wave** u(r, phi, z, t) = U(r, phi, z - c t). With Phi_T the discrete time-T map of the
IVP solver (SBDF, constant step, the state's velocity entries only: the pressure, taus and G of a
state are recomputed by the first solve), the unknowns (x, c), x the velocity coefficients as real
numbers (Re, Im), solve

    R(x, c) = S_{cT} Phi_T(x) - x = 0,     phase:  <i k_z x_ref, x - x_ref> = 0,

S_a the axial translation by +a (coefficients times e^{i k_z a}), x_ref the seed (fixes the axial
phase: the azimuthal phase is fixed by the shift-reflect subspace). The Jacobian is
dR = S_{cT} D Phi_T dx - dx + i k_z T S_{cT} Phi_T(x) dc, with D Phi_T dx by a **forward finite
difference** of the time stepper (declared: (Phi_T(X + eps dX) - Phi_T(X)) / eps, eps = fd_eps
max(||X||, 1) / ||dX||; error O(eps) + rounding / eps), the rest exact. GMRES vectors stay in the subspace
(the residual is in it and the operator preserves it).

**Stokes-preconditioned steady residual** (``method="stokes"``, the default: one F evaluation and one
banded solve per Krylov vector instead of a time-T map; Tuckerman's Stokes preconditioning, as
``ecs.steady.AffineSteadyProblem``): unknowns (X, c), X the whole state (velocity, pressure, taus, G)
as real numbers, residual

    R(X, c) = P^{-1} ( F(X) - L X + c M dz X ),      P = a M + L   (a = 1 / Delta, declared; a = 0: Stokes),

whose zero is the travelling wave of the **spatial discretization** exactly (no time error), the
phase condition as above; dF V by a **central difference** (F(X + eps V) - F(X - eps V)) / 2 eps,
eps = fd_eps max(||X||, 1) / ||V||, exact for the quadratic right-hand side up to rounding (declared). Both methods report the same
evidence.

**Evidence reported** (``TravellingWaveResult.explain``): the shooting residual ||R|| / ||x||, the
**comoving residual** of the continuous equations at the converged state,
||F(X) - L X + c M dz X|| / ||L X|| over all rows (independent of the shooting map: it measures the
time-discretization error of the converged wave), the phase speed c (in U_cl, and in bulk units
c / U_b = 2 c), the friction G / G_lam = D / D_lam (time-averaged dissipation = energy input at
constant flux; constant for a travelling wave), E / E_lam, and the projection leak.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..discretize.jacobi_family import gauss_legendre
from ..errors import OptionError
from .newton import NewtonKrylovHookstep, NewtonKrylovResult
from .._repr import compact_repr

__all__ = ["pipe_flux_problem", "PipeCase", "PipeTravellingWave", "TravellingWaveResult",
           "NEUTRAL_OVERLAP_MIN"]

NEUTRAL_OVERLAP_MIN = 0.99
"""Least alignment |cos| between d_z x and the eigenvector (or the eigenspace of a cluster of
eigenvalues within the Arnoldi accuracy) taken as the neutral translation mode; below it the mode
is reported as not identified and nothing is excluded from the unstable count (declared)."""


@dataclass
class PipeCase:
    """A discretized constant-flux pipe problem (``pipe_flux_problem``)."""

    plan: object
    u: object
    G: object
    Re: float
    alpha: float
    Lz: float
    Q: float
    notes: list[str] = field(default_factory=list)

    @property
    def G_lam(self) -> float:
        """Laminar pressure gradient 4 / Re."""
        return 4.0 / self.Re

    @property
    def E_lam(self) -> float:
        """Kinetic energy (1/2) int |u|^2 dV of the laminar flow: pi Lz / 6."""
        return np.pi * self.Lz / 6.0


def pipe_flux_problem(*, Nphi: int, Nr: int, Nz: int, alpha: float, Re: float, m0: int | None,
                      shift_reflect: bool, dealias: float) -> PipeCase:
    """Constant-flux pipe IVP (module docstring), optionally in the subspace of
    ``AzimuthalMFold(phi, m0)`` and/or ``ShiftReflect(phi, z)`` (every field even)."""
    from .. import expert as C
    from ..discretize import discretize
    cc = C.CylindricalCoordinates("phi", "r", "z")
    phi, r, z = cc["phi"], cc["r"], cc["z"]
    Lz = 2 * np.pi / alpha
    br = C.DiskBasis(r, Nr, 1.0, dealias=dealias)
    D = C.Domain(cc, [C.RealFourier(phi, Nphi, (0, 2 * np.pi), dealias=dealias), br,
                      C.ComplexFourier(z, Nz, (0, Lz), dealias=dealias)])
    u, p = C.VectorField("u", D), C.Field("p", D)
    tu, tp = C.VectorField("tu", D.without(r)), C.Field("tp", C.Domain(cc, []))
    G = C.Field("G", C.Domain(cc, []))
    gens = ([C.AzimuthalMFold(phi, m0)] if m0 else []) + ([C.ShiftReflect(phi, z)] if shift_reflect else [])
    sym = None
    if gens:
        ev = {g: C.EVEN for g in gens}
        sym = C.SymmetrySubspace(*gens, u=ev, p=ev, tu=ev, tp=ev, G=ev)
    Q = np.pi / 2 * Lz
    pb = C.IVP([u, p, tu, tp, G], symmetry=sym, name="constant-flux pipe flow")
    pb.add_equation((C.dt(u) - C.lap(u) / Re + C.grad(p) - G * C.unit_vector(z)
                     + C.lift(tu, br, -1, lam=2), -C.dot(u, C.grad(u))))
    pb.add_equation((C.div(u) + tp, 0))
    pb.add_equation((C.interp(u, r, "right"), 0))
    pb.add_equation((C.integ(p), 0))
    pb.add_equation((C.integ(C.comp(u, 2)), Q))
    plan = discretize(pb)
    notes = [f"constant flux Q = (pi/2) Lz = {Q:.12g} (bulk velocity 1/2), Re = U_cl R / nu = U_b D / nu = {Re:g}",
             f"Lz = 2 pi / {alpha:g}, (Nphi, Nr, Nz) = ({Nphi}, {Nr}, {Nz}), dealias {dealias:g}",
             "subspace: " + (", ".join(repr(g) for g in gens) if gens else "none (full space)")]
    return PipeCase(plan, u, G, float(Re), float(alpha), float(Lz), float(Q), notes)


@dataclass
class TravellingWaveResult:
    """A converged pipe travelling wave and its evidence."""
    __repr__ = compact_repr   # one line, arrays by shape

    X: np.ndarray
    c: float
    shooting_residual: float
    comoving_residual: float
    G_ratio: float
    E_ratio: float
    leak: float
    T: float
    dt: float
    newton: NewtonKrylovResult | None
    notes: list[str]

    @property
    def c_bulk(self) -> float:
        """Phase speed in units of the bulk velocity U_b = U_cl / 2."""
        return 2.0 * self.c

    def explain(self) -> str:
        """Phase speed, friction, energy, residuals and leak of the wave, with the run notes."""
        it = "" if self.newton is None else f", {self.newton.iterations} Newton iterations"
        lines = [f"travelling wave: c = {self.c:.10g} U_cl = {self.c_bulk:.10g} U_b; D/D_lam = G/G_lam = "
                 f"{self.G_ratio:.10g}; E/E_lam = {self.E_ratio:.10g}",
                 f"  shooting residual ||S_cT Phi_T(x) - x|| / ||x|| = {self.shooting_residual:.3e} "
                 f"(T = {self.T:g}, dt = {self.dt:g}{it})",
                 f"  comoving residual ||F - L X + c M dz X|| / ||L X|| = {self.comoving_residual:.3e} "
                 "(continuous equations; time-discretization error of the converged wave)",
                 f"  projection leak (shift-reflect) per step <= {self.leak:.1e}"]
        return "\n".join(lines + [f"  note: {n}" for n in self.notes])


class PipeTravellingWave:
    """Newton-Krylov-hookstep for a pipe travelling wave (module docstring) with an SBDF
    ``solver`` of a :func:`pipe_flux_problem` plan (its dt is the shooting step)."""

    def __init__(self, case: PipeCase, solver, *, T: float, fd_eps: float, method: str = "stokes",
                 stokes_a: float | None = None) -> None:
        from .._experimental import experimental
        experimental("orbits")
        if method not in ("stokes", "shooting"):
            raise OptionError(f"method must be 'stokes' or 'shooting', got {method!r}",
                              fix="method='stokes' (with stokes_a=) or method='shooting'")
        if method == "stokes" and not (stokes_a is not None and stokes_a > 0):
            # a = 0 (P = L) is singular in the banded part of the axisymmetric group of the
            # constant-flux pipe (measured): the preconditioner a M + L needs a declared a > 0
            raise OptionError(f"method='stokes' needs stokes_a = 1/Delta > 0 (preconditioner a M + L), "
                              f"got {stokes_a!r}",
                              why="a = 0 (P = L) is singular for the constant-flux pipe",
                              fix="declare stokes_a > 0, e.g. stokes_a=1/T for a time-T map")
        self.method, self.stokes_a = method, float(stokes_a or 0.0)
        if not str(solver.scheme).upper().startswith("SBDF"):
            raise OptionError("PipeTravellingWave: use an SBDF scheme (the map depends on the velocity "
                              "entries only, every step restarts from them)",
                              where=f"scheme {solver.scheme!r}",
                              fix="build the solver with scheme='SBDF2' (or 'SBDF1'..'SBDF4')")
        self.case, self.s = case, solver
        self.xp = solver.xp
        host = getattr(solver.system, "host", solver.system)
        names = [v.name for v in host.variables]
        k_u, k_G = names.index(case.u.name), names.index(case.G.name)
        self.iu = np.nonzero(host.cols_obj == k_u)[0]
        self.iG = np.nonzero(host.cols_obj == k_G)[0]
        S = case.plan.space
        shape = host.var_shapes[k_u]
        zi = np.unravel_index(host.cols_flat[self.iu], shape)[-1]
        self.kz = S.kz_all[zi]
        self.nu = self.iu.size
        self.nsteps = _steps_of(T, solver.dt)
        self.T, self.fd_eps = float(T), float(fd_eps)
        self.n = int(host.n)
        self._host = host
        self._cache: dict = {}
        self.x_ref = None
        self.n_maps = 0

    # ------------------------------------------------------------------ vectors
    def to_x(self, X) -> np.ndarray:
        """Real Newton vector (Re, Im of the velocity entries) of state X."""
        Xh = X.get() if hasattr(X, "get") else np.asarray(X)
        a = Xh[self.iu]
        return np.concatenate([a.real, a.imag])

    def to_X(self, x):
        """State with the velocity entries of real vector x (other entries zero: refilled by a solve)."""
        X = np.zeros(self.s.system.n, complex)
        X[self.iu] = x[:self.nu] + 1j * x[self.nu:]
        return self.xp.asarray(X)

    def shift(self, x, a: float) -> np.ndarray:
        """Axial translation by +a of real velocity vector x."""
        w = (x[:self.nu] + 1j * x[self.nu:]) * np.exp(1j * self.kz * a)
        return np.concatenate([w.real, w.imag])

    def dz(self, x) -> np.ndarray:
        """Axial derivative of real velocity vector x (the group tangent of the axial translation)."""
        w = 1j * self.kz * (x[:self.nu] + 1j * x[self.nu:])
        return np.concatenate([w.real, w.imag])

    # ------------------------------------------------------------------ map
    def advance(self, X, nsteps: int):
        """nsteps of the solver from state X (history reset: SBDF start-up from X)."""
        s = self.s
        s.X = self.xp.asarray(X).copy()
        s.t, s.iteration = 0.0, 0
        s._reset_history()
        s.run(nsteps)
        self.n_maps += 1
        return s.X.copy()

    def phi(self, x) -> tuple[np.ndarray, object]:
        """Time-T map of velocity vector x (cached for the last x): (real vector, full state),
        the caller's own copies (returning the cached arrays,
        and the result of ``solve`` keeps the full state, so an in-place edit of ``result.X``
        changed what a later ``phi`` of the same x returned)."""
        key = np.asarray(x).tobytes()
        hit = self._cache.get("phi")
        if hit is None or hit[0] != key:
            Y = self.advance(self.to_X(x), self.nsteps)
            hit = self._cache["phi"] = (key, self.to_x(Y), Y)
        return hit[1].copy(), hit[2].copy()

    def residual(self, z) -> np.ndarray:
        """Shooting residual (S_cT Phi_T(x) - x, phase condition) of z = (x, c)."""
        x, c = z[:-1], z[-1]
        y, _ = self.phi(x)
        r = self.shift(y, c * self.T) - x
        return np.concatenate([r, [self._phase(x)]])

    def _phase(self, x) -> float:
        d = self.dz(self.x_ref)
        return float(d @ (x - self.x_ref)) / float(np.linalg.norm(d))

    def jvp(self, z, v) -> np.ndarray:
        """Jacobian of :meth:`residual` at z applied to v (forward difference of the time stepper)."""
        x, c = z[:-1], z[-1]
        dx, dc = v[:-1], v[-1]
        y, _ = self.phi(x)
        out = -dx.copy()
        ndx = float(np.linalg.norm(dx))
        if ndx > 0:
            eps = self.fd_eps * max(float(np.linalg.norm(x)), 1.0) / ndx
            Y2 = self.advance(self.to_X(x + eps * dx), self.nsteps)
            out += self.shift((self.to_x(Y2) - y) / eps, c * self.T)
        out += dc * self.T * self.dz(self.shift(y, c * self.T))
        d = self.dz(self.x_ref)
        return np.concatenate([out, [float(d @ dx) / float(np.linalg.norm(d))]])

    # ------------------------------------------------------------------ Stokes-preconditioned
    def _cx(self, xa) -> np.ndarray:
        """Complex state of the real unknown vector (all entries)."""
        return xa[:self.n] + 1j * xa[self.n:2 * self.n]

    def _rx(self, X) -> np.ndarray:
        Xh = X.get() if hasattr(X, "get") else np.asarray(X)
        return np.concatenate([Xh.real, Xh.imag])

    def _dzX(self, Xc):
        out = np.zeros_like(Xc)
        out[self.iu] = 1j * self.kz * Xc[self.iu]
        return out

    def _F(self, Xc):
        F = self.s.F(self.xp.asarray(Xc), 0.0)
        return F.get() if hasattr(F, "get") else np.asarray(F)

    def _P(self, r):
        y = self.s.system.solve(self.stokes_a, 1.0, self.xp.asarray(r))
        pr = getattr(self.s, "_m0proj", None)          # rounding only (equivariant operators)
        if pr is not None:
            y = pr.apply(y, False) if hasattr(pr, "s") else pr.apply(y)
        return y.get() if hasattr(y, "get") else np.asarray(y)

    def steady_residual(self, z) -> np.ndarray:
        """Stokes-preconditioned comoving residual P^-1 (F - L X + c M dz X) and phase condition."""
        Xc, c = self._cx(z), z[-1]
        key = z.tobytes()
        hit = self._cache.get("F")
        F = hit[1] if (hit is not None and hit[0] == key) else self._F(Xc)
        self._cache["F"] = (key, F)
        r = F - self._host.L @ Xc + c * (self._host.M @ self._dzX(Xc))
        return np.concatenate([self._rx(self._P(r)), [self._phase_full(Xc)]])

    def _phase_full(self, Xc) -> float:
        x = np.concatenate([Xc[self.iu].real, Xc[self.iu].imag])
        return self._phase(x)

    def steady_jvp(self, z, v) -> np.ndarray:
        """Jacobian of :meth:`steady_residual` at z applied to v (dF by a central difference)."""
        Xc, c = self._cx(z), z[-1]
        V, dc = self._cx(v), v[-1]
        nV = float(np.linalg.norm(V))
        dF = np.zeros(self.n, complex)
        if nV > 0:
            eps = self.fd_eps * max(float(np.linalg.norm(Xc)), 1.0) / nV
            dF = (self._F(Xc + eps * V) - self._F(Xc - eps * V)) / (2 * eps)
        r = dF - self._host.L @ V + c * (self._host.M @ self._dzX(V)) + dc * (self._host.M @ self._dzX(Xc))
        x = np.concatenate([V[self.iu].real, V[self.iu].imag])
        d = self.dz(self.x_ref)
        return np.concatenate([self._rx(self._P(r)), [float(d @ x) / float(np.linalg.norm(d))]])

    # ------------------------------------------------------------------ evidence
    def measures(self, X) -> dict:
        """G / G_lam, E / E_lam, the energy of the k_z != 0 modes E3d / E_lam of state X (a state
        after a step: its G entry is the step's pressure gradient) and the largest velocity *coefficient* modulus
        ``max_coeff_u`` (not a grid velocity: 1/sqrt(2) for the laminar flow)."""
        Xh = X.get() if hasattr(X, "get") else np.asarray(X)
        G = float(Xh[self.iG][0].real)
        w = Xh[self.iu]
        E = self.energy(Xh)
        E3 = self.energy(Xh, three_d=True)
        return {"G_ratio": G / self.case.G_lam, "E_ratio": E / self.case.E_lam, "E3d": E3 / self.case.E_lam,
                "max_coeff_u": float(np.abs(w).max())}

    def energy(self, Xh, three_d: bool = False) -> float:
        """(1/2) int |u|^2 dV on the dealiased grid (Gauss-Legendre radial nodes x uniform phi, z),
        or of the k_z != 0 part only."""
        S = self.case.plan.space
        host = getattr(self.s.system, "host", self.s.system)
        X = np.array(Xh)
        if three_d:
            X[self.iu[self.kz == 0]] = 0.0
        k = [v.name for v in host.variables].index(self.case.u.name)
        c = host.scatter(X)[k]
        g = S.backward(c, self.case.u.domain, 1, True, real_tol=np.inf)
        # disk nodes r_i = sqrt((1 + z_i) / 2), z_i Gauss-Legendre: r dr = dz / 4 (exact for the
        # polynomial |u|^2 on the dealiased grid); uniform phi and z grids: plain means. The weights
        # are the refined ones of the disk grid itself (``DiskOps.grid``; numpy's leggauss weights
        # are off by up to 4e-12 relative at 100 points)
        _, w = gauss_legendre(g.shape[2])
        dens = 0.5 * np.sum(np.abs(g) ** 2, axis=0)            # (Mphi, Mr, Mz)
        return float(2 * np.pi * self.case.Lz * 0.25 * np.sum(dens.mean(axis=(0, 2)) * w))

    def comoving_residual(self, X, c: float) -> float:
        """||F(X) - L X + c M dz X|| / ||L X|| over all rows (continuous steady equations in the
        frame moving at c; X must carry consistent pressure / taus / G: a state after a step)."""
        s = self.s
        host = getattr(s.system, "host", s.system)
        Xh = X.get() if hasattr(X, "get") else np.asarray(X)
        dzX = np.zeros_like(Xh)
        dzX[self.iu] = 1j * self.kz * Xh[self.iu]
        Fx = s.F(s.xp.asarray(Xh), 0.0)
        Fx = Fx.get() if hasattr(Fx, "get") else Fx
        LX = host.L @ Xh
        r = Fx - LX + c * (host.M @ dzX)
        return float(np.linalg.norm(r) / np.linalg.norm(LX))

    def eigenvalues(self, X, c: float, *, k: int, T: float, tol: float, fd_eps: float,
                    ncv: int | None = None) -> dict:
        """Leading eigenvalues of the travelling wave (X, c) **in the solver's subspace**: Arnoldi
        (ARPACK, largest modulus) on the comoving time-T map dx -> S_{cT} D Phi_T dx, D Phi_T by a
        **central difference** of the time stepper (declared ``fd_eps``; the subspace projections
        of every step keep the vectors in it), lambda = log(mu) / T (Im lambda modulo 2 pi / T).
        The velocity entries only (real form). The axial translation stays a symmetry of the
        shift-reflect subspace, so the wave has a neutral eigenvalue (mu = 1, eigenvector d_z x):
        ``neutral[i]`` is |cos| between eigenvector i and d_z x (reported, so that the neutral mode
        is never counted as an instability): one mode at most, the best aligned, if the alignment of
        its eigenspace reaches ``NEUTRAL_OVERLAP_MIN`` (``neutral_index``); ``n_unstable`` counts
        Re lambda > 0 without it (members of an eigenvalue cluster degenerate with it: ``n_marginal``).
        Returns {lambda, mu, neutral, neutral_index, n_unstable, n_marginal, notes, matvecs, T, fd_eps}
        sorted by decreasing Re lambda."""
        import scipy.sparse.linalg as spla
        n = _steps_of(T, self.s.dt)
        x = self.to_x(self.xp.asarray(X))
        nx = float(np.linalg.norm(x))
        count = [0]

        def sym(v):
            """v projected onto the solver's subspace (ARPACK vectors are arbitrary: the map is
            restricted explicitly instead of relying on the step's projection, whose leak check
            would refuse them)."""
            return self.to_x(self.s.symmetrize_state(self.to_X(v)))

        def mv(v):
            v = sym(v)
            nv = float(np.linalg.norm(v))
            if nv == 0:
                return np.zeros_like(v)
            e = fd_eps * nx / nv
            yp = self.to_x(self.advance(self.to_X(x + e * v), n))
            ym = self.to_x(self.advance(self.to_X(x - e * v), n))
            count[0] += 1
            return self.shift((yp - ym) / (2 * e), c * T)
        op = spla.LinearOperator((x.size, x.size), matvec=mv, dtype=float)
        v0 = sym(np.random.default_rng(0).standard_normal(x.size))
        mu, vec = spla.eigs(op, k=k, which="LM", tol=tol, ncv=ncv, v0=v0, return_eigenvectors=True)
        lam = np.log(mu.astype(complex)) / T
        d = self.dz(x)
        nd = float(np.linalg.norm(d))
        d = d / nd if nd > 0 else d                  # an axially uniform state has no translation mode
        neutral = np.array([abs(np.vdot(vec[:, i], d)) / np.linalg.norm(vec[:, i]) for i in range(mu.size)])
        order = np.argsort(-lam.real)
        lam, mu, neutral, vec = lam[order], mu[order], neutral[order], vec[:, order]
        neutral_index, n_unstable, n_marginal, notes = _neutral_mode(lam, mu, vec, d, tol)
        return {"lambda": lam, "mu": mu, "neutral": neutral, "neutral_index": neutral_index,
                "n_unstable": n_unstable, "n_marginal": n_marginal, "notes": notes,
                "matvecs": count[0], "T": T, "fd_eps": fd_eps}

    def solve(self, X0, c0: float, newton: NewtonKrylovHookstep, *, log=None) -> TravellingWaveResult:
        """Newton from state X0 (projected by the solver's declared projections) and speed c0;
        X0 must carry consistent pressure / taus / G for ``method='stokes'`` (a state after a
        step: ``advance(X0, 1)`` gives one)."""
        X0 = self.s.symmetrize_state(self.xp.asarray(X0))
        x0 = self.to_x(X0)
        self.x_ref = x0.copy()
        self.s.symmetry_leak(reset=True)
        if self.method == "shooting":
            z0 = np.concatenate([x0, [c0]])
            res = newton.solve(self.residual, self.jvp, z0, scale=float(np.linalg.norm(x0)), log=log)
            x, c = res.z[:-1], float(res.z[-1])
        else:
            z0 = np.concatenate([self._rx(X0), [c0]])
            res = newton.solve(self.steady_residual, self.steady_jvp, z0,
                               scale=float(np.linalg.norm(z0[:-1])), log=log)
            c = float(res.z[-1])
            x = self.to_x(self.xp.asarray(self._cx(res.z)))
        y, Y = self.phi(x)
        shoot = float(np.linalg.norm(self.shift(y, c * self.T) - x) / np.linalg.norm(x))
        leak = self.s.symmetry_leak()
        Xs = self.xp.asarray(self._cx(res.z)) if self.method == "stokes" else Y
        m = self.measures(Xs)
        jac = ("forward finite difference of the time stepper" if self.method == "shooting" else
               f"Stokes-preconditioned steady residual, P = {self.stokes_a:g} M + L, dF by central "
               "difference (exact for quadratic F up to rounding)")
        notes = list(self.case.notes) + [
            newton.explain(), f"method {self.method}: {jac}, fd_eps = {self.fd_eps:g}",
            f"{self.n_maps} time-T maps ({self.nsteps} steps each; the final one is the shooting check)"]
        return TravellingWaveResult(Xs, c, shoot, self.comoving_residual(Xs, c), m["G_ratio"],
                                    m["E_ratio"], leak, self.T, self.s.dt, res, notes)


def _steps_of(T: float, dt: float) -> int:
    """Number of steps of a time-T map; T must be a multiple of dt (never rounded silently: the
    eigenvalues log(mu) / T of a map over another time would be scaled wrongly)."""
    n = int(round(T / dt))
    from ..timestep.integrator import is_whole
    if n < 1 or not is_whole(T / dt):
        raise OptionError(f"T = {T:g} is not a positive multiple of dt = {dt:g}",
                          why="the time-T map is a whole number of steps; rounding T / dt would "
                              "scale the eigenvalues log(mu) / T wrongly",
                          fix=f"take T = n dt (e.g. T = {max(1, round(T / dt)) * dt:g}) or a dt "
                              "that divides T")
    return n


def _neutral_mode(lam, mu, vec, d, tol: float):
    """(neutral_index, n_unstable, n_marginal, notes) of eigenpairs (lam, mu, vec) of a comoving map
    whose translation mode is d (unit vector; zero vector: no translation mode).

    One neutral mode at most. A per-eigenvector threshold on |cos(vec_i, d)| is not enough:
    non-normal stable eigenvectors of the pipe wave are seen at |cos| = 0.96-0.99, and eigenvalues
    degenerate with mu = 1 (within the Arnoldi accuracy) split d between arbitrary vectors of their
    eigenspace. So each eigenvalue is scored by the alignment of d with the span of its cluster
    (eigenvalues within max(1e3 tol, 1e-10) max(1, |mu|)); the best cluster, if its alignment
    reaches ``NEUTRAL_OVERLAP_MIN``, holds the neutral mode (its best-aligned member) and its other
    members are counted as marginal, not unstable."""
    notes: list[str] = []
    marginal = np.zeros(mu.size, bool)
    neutral_index = None
    if mu.size and float(np.linalg.norm(d)) > 0:
        best = (-1.0, -1, None, 0.0)
        for i in range(mu.size):
            near = max(1e3 * tol, 1e-10) * max(1.0, abs(mu[i]))
            cl = np.nonzero(np.abs(mu - mu[i]) <= near)[0]
            Q, _ = np.linalg.qr(vec[:, cl])
            span = float(np.linalg.norm(Q.conj().T @ d))
            if span > best[0] + 1e-12:
                best = (span, i, cl, near)
        span, _, cl, near = best
        if span >= NEUTRAL_OVERLAP_MIN:
            cos = [abs(np.vdot(vec[:, j], d)) / np.linalg.norm(vec[:, j]) for j in cl]
            neutral_index = int(cl[int(np.argmax(cos))])
            marginal[cl] = True
            marginal[neutral_index] = False
            if cl.size > 1:
                notes.append(f"translation mode degenerate with {cl.size - 1} other eigenvalue(s) "
                             f"(|mu - mu_0| <= {near:.1e}): alignment of their span {span:.4f}; "
                             "the others are counted as marginal, not unstable")
        else:
            notes.append(f"translation mode not identified among the {mu.size} eigenvalues (best "
                         f"alignment {span:.3f} < {NEUTRAL_OVERLAP_MIN}): nothing excluded")
    keep = np.ones(mu.size, bool)
    if neutral_index is not None:
        keep[neutral_index] = False
    n_unstable = int(np.sum((lam.real > 0) & keep & ~marginal))
    return neutral_index, n_unstable, int(marginal.sum()), notes
