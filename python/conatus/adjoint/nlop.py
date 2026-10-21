"""Nonlinear optimal perturbation (NLOP) and minimal seed by the exact discrete adjoint.

Problem (Pringle & Kerswell 2010; Rabin, Caulfield & Kerswell 2012; Kerswell 2018 review)::

    maximise  J(u0) = E(Φ_T(u0))   on the sphere  E(u0) = E0,  u0 admissible,

with Φ_T the discrete flow map (:class:`conatus.adjoint.FlowMap`, time-dependent F allowed) and
E(u) = ½ ⟨u, u⟩ the energy of the **physical fields** only. Three explicit pieces:

* :class:`EnergyInnerProduct` ⟨a, b⟩ = aᵀ W b on the global state vector, W the exact energy Gram
  matrix of the declared velocity fields (Parseval along Fourier axes: L for the mean, L/2 for a
  cos/sin coefficient; exact Chebyshev Gram ∫ T_m T_n along the Chebyshev axis). W is zero on the
  pressure, the tau variables and the gauges (and on the msin_0 slots, which are not modes): it is
  singular on the whole state, so the physical adjoint W⁻¹ Jᵀ W is taken on the velocity
  subspace only (W itself is singular, so the adjoint on the full state is not defined).
* :class:`AdmissibleProjector` P: the W-orthogonal projection of a velocity field onto the
  admissible set {v : C_v v + C_o o = 0 for some o}, C the algebraic rows of the discrete system
  (rows without mass: continuity, boundary conditions, gauges) and o the non-velocity unknowns
  (pressure, taus). Every state produced by a step is admissible (the stage solves impose these
  rows), so P is the discrete Leray projection *with the boundary conditions* of this very
  discretization (per Fourier group, dense; the rank decision is reported).
* :class:`NLOP`: J, its exact gradient in the energy inner product
  g = P W⁻¹ Jᵀ W X_T (Riesz representative of dJ on the admissible space), and the rotation update
  on the sphere (Rabin et al. 2012, eq. 2.18 form)::

      g⊥ = g − (⟨g, u0⟩ / ⟨u0, u0⟩) u0,
      u(α) = cos α u0 + sin α (‖u0‖ / ‖g⊥‖) g⊥,           E(u(α)) = E0 exactly,

  with a declared step rule (accept if J increases, then α ← grow α; otherwise α ← α / 2) and the
  optimality residual r = ‖g⊥‖ / ‖g‖ (0 at a critical point on the sphere) reported per
  iteration. Nothing is chosen silently: T, E0, α0, grow, tolerances and checkpoints are arguments.

Multi-job iterations: ``iterate(..., journal=dir, budget_s=...)`` may stop *inside*
an iteration. The adjoint sweep of the gradient is suspended at a step boundary (FlowMap journal
in ``dir/vjp``, with J of the forward pass, so the resumed job does not repeat it); the line
search is suspended between two forward evaluations (seed, tangent direction, tries and α in
``dir/nlop_state.npz``). ``NLOPState.load(dir)`` + ``iterate(..., journal=dir)`` in the next job
continue the same iteration; the result is bit-identical to an uninterrupted run (tested).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.linalg as sla
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp

from .._deprecation import renames
from ..errors import OptionError
from ..ir.bases import Chebyshev, ComplexFourier, RealFourier
from ..tolerances import REQUIRED, require
from .._repr import compact_repr

__all__ = ["EnergyInnerProduct", "AdmissibleProjector", "NLOP", "NLOPState"]


# ---------------------------------------------------------------------------- energy
def _fourier_weights(b) -> np.ndarray:
    """∫ f g along a Fourier axis for the coefficient layout of the basis (exact Parseval)."""
    if isinstance(b, RealFourier):
        w = np.full(b.size, b.length / 2)
        w[0] = b.length
        w[1] = 0.0                           # msin_0: not a mode (always 0)
        return w
    if isinstance(b, ComplexFourier):
        return np.full(b.size, b.length)
    raise TypeError(f"not a Fourier basis: {b}")


class EnergyInnerProduct:
    """⟨a, b⟩ = aᵀ W b on the global state of an IVPSolver (see module docstring); energy
    E(X) = ½ ⟨X, X⟩. ``fields``: names of the scalar velocity components (explicit)."""

    def __init__(self, solver, fields: tuple[str, ...]) -> None:
        from ..linearize.inner_product import chebyshev_gram
        self.solver, self.fields = solver, tuple(fields)
        self.xp = solver.xp
        sysm = solver.system
        host = sysm.host if hasattr(sysm, "host") else sysm
        if host.complex:
            raise OptionError("EnergyInnerProduct: real state vectors only (RealFourier/Chebyshev)",
                              fix="cn.periodic(L) (real series) instead of complex=True")
        self.n = host.n
        names = [v.name for v in host.variables]
        dom = solver.plan.domain
        self.bases = [dom.basis(c) for c in dom.coordsys.coords]
        cheb = [i for i, b in enumerate(self.bases) if isinstance(b, Chebyshev)]
        if len(cheb) != 1:
            raise OptionError("EnergyInnerProduct: exactly one Chebyshev axis expected",
                              fix="a Fourier x Chebyshev domain (walls(a, b) along one axis); other "
                                  "geometries: the spherical shell has SphereEnergyInnerProduct")
        self.cy = cheb[0]
        by = self.bases[self.cy]
        G = chebyshev_gram(by.size, *by.bounds)
        self.G = G
        self.RG = sla.cholesky(G, lower=False)                 # G = RGᵀ RG
        self.RGinv = sla.solve_triangular(self.RG, np.eye(by.size), lower=False)
        fw = [None if i == self.cy else _fourier_weights(b) for i, b in enumerate(self.bases)]
        self.shapes, self.gidx, self.flat = [], [], []
        for f in self.fields:
            if f not in names:
                raise OptionError(f"unknown field {f!r}", fix=f"one of the variables {names}")
            k = names.index(f)
            shape = host.var_shapes[k]
            if len(shape) != len(self.bases):
                raise OptionError(f"{f}: scalar field on the full domain expected",
                                  fix="the velocity components (scalar unknowns on every axis)")
            sel = host._col_sel[k]
            self.shapes.append(shape)
            self.gidx.append(sel)
            self.flat.append(host.cols_flat[sel])
        # Fourier weight per velocity column (product over Fourier axes), and its y index
        cols, wf, iy, key = [], [], [], []
        for fi, (shape, sel, flat) in enumerate(zip(self.shapes, self.gidx, self.flat)):
            idx = np.unravel_index(flat, shape)
            w = np.ones(flat.size)
            for ax, wa in enumerate(fw):
                if wa is not None:
                    w = w * wa[idx[ax]]
            cols.append(sel)
            wf.append(w)
            iy.append(idx[self.cy])
            key.append(np.stack([np.full(flat.size, fi)] + [idx[a] for a in range(len(shape))
                                                             if a != self.cy], axis=1))
        self.vcols = np.concatenate(cols)                      # global columns of the velocity
        self.wf = np.concatenate(wf)
        self.iy = np.concatenate(iy)
        self.key = np.concatenate(key)                         # (field, Fourier indices)
        self.live = self.wf > 0                                 # msin_0 slots: weight 0
        # per-field arrays for fast device application: (gidx, flat) and the Fourier weight grid
        xp = self.xp
        self._d = []
        for shape, sel, flat in zip(self.shapes, self.gidx, self.flat):
            wgrid = np.ones(shape)
            for ax, wa in enumerate(fw):
                if wa is not None:
                    sh = [1] * len(shape)
                    sh[ax] = shape[ax]
                    wgrid = wgrid * wa.reshape(sh)
            inv_sq = np.where(wgrid > 0, 1 / np.sqrt(np.where(wgrid > 0, wgrid, 1)), 0.0)
            self._d.append((xp.asarray(sel), xp.asarray(flat), shape, xp.asarray(wgrid),
                            xp.asarray(np.sqrt(wgrid)), xp.asarray(inv_sq)))
        self._G, self._RG, self._RGinv = xp.asarray(G), xp.asarray(self.RG), xp.asarray(self.RGinv)

    # -------------------------------------------------------------- field-wise operators
    def _apply(self, X, mat_name: str, wkind: int, transpose: bool = False):
        """Y = (Fourier weight) ⊗ (y matrix) applied to the velocity part of X; 0 elsewhere."""
        xp = self.xp
        A = {"G": self._G, "R": self._RG, "Rinv": self._RGinv}[mat_name]
        if transpose:
            A = A.T
        Y = xp.zeros(self.n, dtype=X.dtype)
        for sel, flat, shape, w, sqw, isq in self._d:
            a = xp.zeros(int(np.prod(shape)), dtype=X.dtype)
            a[flat] = X[sel]
            a = xp.moveaxis(a.reshape(shape), self.cy, -1)
            b = a @ A.T                                         # (.., Ny) · Aᵀ: A along y
            b = xp.moveaxis(b, -1, self.cy)
            b = b * (w, sqw, isq)[wkind]
            Y[sel] = b.reshape(-1)[flat]
        return Y

    def W(self, X):
        """W X (velocity part; zero on pressure, taus, gauges)."""
        return self._apply(X, "G", 0)

    def R(self, X):
        """R X with W = Rᵀ R on the velocity (R = sqrt(w_F) ⊗ RG)."""
        return self._apply(X, "R", 1)

    def Rinv(self, X):
        return self._apply(X, "Rinv", 2)

    def RinvT(self, X):
        return self._apply(X, "Rinv", 2, transpose=True)

    def Winv(self, Y):
        """W⁻¹ Y on the velocity subspace (R⁻¹ R⁻ᵀ); zero elsewhere and on weight-0 slots."""
        return self.Rinv(self.RinvT(Y))

    def restrict(self, X):
        """X with everything but the velocity (live slots) set to zero."""
        xp = self.xp
        Y = xp.zeros_like(X)
        v = xp.asarray(self.vcols[self.live])
        Y[v] = X[v]
        return Y

    def inner(self, a, b) -> float:
        return float(a @ self.W(b))

    def energy(self, X) -> float:
        return 0.5 * self.inner(X, X)

    def explain(self) -> str:
        return (f"EnergyInnerProduct: <a, b> = aᵀ W b, W = exact energy Gram of {self.fields} "
                f"(Parseval along Fourier axes, exact Chebyshev Gram along y); zero on pressure, "
                f"taus, gauges and msin_0 slots ({int((~self.live).sum())} velocity slots of weight "
                f"0); E = ½ <X, X>")


# ---------------------------------------------------------------------------- admissible set
class AdmissibleProjector:
    """W-orthogonal projection of a velocity field onto the admissible set of the discrete
    system (module docstring). Built per group of the constraint graph (dense SVDs);
    ``rank_tol`` (explicit): singular values below rank_tol · max are the null space."""

    def __init__(self, ip: EnergyInnerProduct, rank_tol: float = 1e-10) -> None:
        self.ip, self.xp, self.rank_tol = ip, ip.xp, float(rank_tol)
        sysm = ip.solver.system
        host = sysm.host if hasattr(sysm, "host") else sysm
        M, L = host.M.tocsr(), host.L.tocsr()
        alg = np.nonzero(np.diff(M.indptr) == 0)[0]            # rows without mass
        C = L[alg].tocsc()
        n = host.n
        vmask = np.zeros(n, bool)
        vmask[ip.vcols[ip.live]] = True
        zero_w = ip.vcols[~ip.live]                             # weight-0 velocity slots
        # groups: bipartite graph columns-rows of C, plus velocity columns sharing a Fourier key
        Cc = C.tocoo()
        nr = C.shape[0]
        keys = {tuple(k): i for i, k in enumerate(np.unique(ip.key, axis=0))}
        kid = np.array([keys[tuple(k)] for k in ip.key])
        nk = len(keys)
        rows = np.concatenate([Cc.col, ip.vcols])
        cols = np.concatenate([n + Cc.row, n + nr + kid])
        A = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n + nr + nk,) * 2)
        ncomp, lab = connected_components(A, directed=False)
        pos = {int(c): i for i, c in enumerate(ip.vcols)}
        self.groups = []
        drop_max, keep_min, ranks = 0.0, np.inf, []
        Csr = C.tocsr()
        for g in np.unique(lab[ip.vcols[ip.live]]):
            vc = np.nonzero(vmask & (lab[:n] == g))[0]
            rg = np.nonzero(lab[n:n + nr] == g)[0]
            if rg.size == 0:
                continue                                        # unconstrained group
            Cg = Csr[rg].toarray()
            oc = np.nonzero((~vmask) & (np.abs(Cg).sum(axis=0) > 0))[0]
            oc = oc[~np.isin(oc, zero_w)]
            Cv, Co = Cg[:, vc], Cg[:, oc]
            # R^-1 on the group (block-diagonal over Fourier keys, RG^-1 / sqrt(w) along y)
            ii = np.array([pos[int(c)] for c in vc])
            same = (kid[ii][:, None] == kid[ii][None, :])
            Rinv = np.where(same, ip.RGinv[ip.iy[ii][:, None], ip.iy[ii][None, :]], 0.0)
            Rinv = Rinv / np.sqrt(ip.wf[ii])[None, :]
            K = Cv @ Rinv
            if oc.size:
                Uo, so, _ = np.linalg.svd(Co, full_matrices=False)
                Qo = Uo[:, so > rank_tol * max(so.max(), 1e-300)]
                K = K - Qo @ (Qo.T @ K)
            _, s, Vt = np.linalg.svd(K, full_matrices=False)
            if s.size == 0 or s.max() == 0:
                continue
            keep = s > rank_tol * s.max()
            if (~keep).any():
                drop_max = max(drop_max, float(s[~keep].max() / s.max()))
            keep_min = min(keep_min, float(s[keep].min() / s.max()))
            ranks.append(int(keep.sum()))
            self.groups.append((vc, Vt[keep].T.copy()))
        self.gap = (keep_min, drop_max)
        self.ranks = ranks
        # padded batch for the device: idx (G, m) into an extended vector (slot n = scratch)
        G = len(self.groups)
        m = max(len(vc) for vc, _ in self.groups)
        r = max(U.shape[1] for _, U in self.groups)
        idx = np.full((G, m), n)
        Ub = np.zeros((G, m, r))
        for i, (vc, U) in enumerate(self.groups):
            idx[i, :len(vc)] = vc
            Ub[i, :len(vc), :U.shape[1]] = U
        self._idx, self._U = self.xp.asarray(idx), self.xp.asarray(Ub)
        self.n = n

    def project_primed(self, Xp):
        """Euclidean projection in the primed coordinates v' = R v (velocity only)."""
        xp = self.xp
        ext = xp.concatenate([Xp, xp.zeros(1, dtype=Xp.dtype)])
        Z = ext[self._idx]                                      # (G, m)
        c = xp.einsum("gmr,gm->gr", self._U, Z)
        ext[self._idx] = Z - xp.einsum("gmr,gr->gm", self._U, c)
        ext[-1] = 0
        return ext[:-1]

    def __call__(self, X):
        """P X: W-orthogonal projection of the velocity part of X (other entries: zero)."""
        ip = self.ip
        return ip.Rinv(self.project_primed(ip.R(ip.restrict(X))))

    def riesz(self, Y):
        """P W⁻¹ Y: the admissible Riesz representative of the covector Y (velocity part)."""
        ip = self.ip
        return ip.Rinv(self.project_primed(ip.RinvT(ip.restrict(Y))))

    def explain(self) -> str:
        return (f"AdmissibleProjector: W-orthogonal projection onto {{v : C_v v + C_o o = 0}}, C = "
                f"algebraic rows (no mass) of the discrete system, o = pressure/taus/gauges; "
                f"{len(self.groups)} constrained group(s), constraint ranks {min(self.ranks)}.."
                f"{max(self.ranks)}; rank_tol = {self.rank_tol:g} (smallest kept singular value "
                f"{self.gap[0]:.2e}, largest dropped {self.gap[1]:.2e}, relative)")


# ---------------------------------------------------------------------------- NLOP
@dataclass
class NLOPState:
    """Optimizer state (restartable): current seed, step angle, history."""
    __repr__ = compact_repr   # one line, arrays by shape
    u0: np.ndarray
    alpha: float
    J: float | None = None
    it: int = 0
    history: list = field(default_factory=list)
    pending: dict | None = None      # iteration in progress (multi-job): phase, tries, record
    gt: np.ndarray | None = None     # tangent ascent direction of the pending line search

    def save(self, path: str | Path) -> None:
        """Atomic write (temporary file, then rename); a directory means ``<dir>/nlop_state.npz``."""
        path = Path(path)
        if path.is_dir():
            path = path / "nlop_state.npz"
        tmp = path.with_name(path.name + ".tmp.npz")

        def host(a):
            return a if isinstance(a, np.ndarray) else a.get()
        arrays = dict(u0=host(self.u0))
        if self.gt is not None:
            arrays["gt"] = host(self.gt)
        with open(tmp, "wb") as f:              # synced before the rename (node crash)
            np.savez(f, **arrays, meta=json.dumps(dict(alpha=self.alpha, J=self.J, it=self.it,
                                                       history=self.history,
                                                       pending=self.pending)))
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)

    def explain(self) -> str:
        """One line per iteration: gain J/E0, optimality residual, accepted angle."""
        lines = [f"NLOPState: {self.it} iterations, alpha = {self.alpha:.3g}, J = {self.J}"]
        for r in self.history:
            stop = f"  stop: {r['stop']}" if r.get("stop") else ""
            acc = r.get("accepted_alpha")
            acc = "-" if acc is None else f"{acc:.3g}"
            lines.append(f"  it {r['it']:3d}  gain {r['gain']:.8e}  residual {r['residual']:.2e}  "
                         f"accepted alpha {acc}{stop}")
        return "\n".join(lines)

    @classmethod
    def load(cls, path: str | Path, xp=np) -> NLOPState:
        """State saved by :meth:`save` (a journal directory means ``<dir>/nlop_state.npz``)."""
        path = Path(path)
        if path.is_dir():
            path = path / "nlop_state.npz"
        with np.load(path) as d:
            meta = json.loads(str(d["meta"]))
            return cls(u0=xp.asarray(d["u0"]), alpha=meta["alpha"], J=meta["J"], it=meta["it"],
                       history=meta["history"], pending=meta.get("pending"),
                       gt=xp.asarray(d["gt"]) if "gt" in d else None)


class NLOP:
    """max E(Φ_T(u0)) on E(u0) = E0 over admissible u0 (module docstring).

    Most users build it with :meth:`NLOP.from_solver` (the problem as written on paper: horizon
    T, energy E0, velocity fields), then call :meth:`start` and :meth:`iterate`::

        nl = NLOP.from_solver(solver, T=2.0, E0=1.58e-7, fields=("u", "v", "w"), t0=0.0723)
        st = nl.iterate(nl.start(guess, alpha=0.5), max_iterations=20, rtol=1e-3)
        print(nl.explain()); print(st.explain())
    """

    @classmethod
    def from_solver(cls, solver, *, T: float, E0: float, fields: tuple[str, ...],
                    t0: float | None = None, checkpoints: int | None = None,
                    rank_tol: float = 1e-10, grow: float = 1.5, alpha_min: float = 1e-10,
                    checkpoint_store: str = "device", checkpoint_dir=None) -> NLOP:
        """NLOP of ``solver`` over the horizon ``T`` (a whole number of the solver's steps:
        refused otherwise, never rounded), energy ``E0`` of the declared velocity ``fields``
        (pressure, taus and gauges are not in the energy), start time ``t0`` (required when F
        depends on time). Builds the exact discrete flow map (``FlowMap``, ``checkpoints`` as
        there: None = full storage; ``checkpoint_store``/``checkpoint_dir`` as there), the energy
        inner product and the admissible projection.
        Declared defaults, printed by :meth:`explain`: ``rank_tol`` of the admissible projection,
        step growth ``grow`` after an accepted rotation, stop below ``alpha_min``."""
        from .flowmap import FlowMap
        if not T > 0:
            raise OptionError(f"NLOP.from_solver: T = {T!r}", why="the horizon must be > 0",
                              fix="pass the optimisation horizon in the solver's time units, T > 0")
        n = T / solver.dt
        nsteps = int(round(n))
        from ..timestep.integrator import is_whole    # the one whole-steps rule (WHOLE_STEPS_TOL)
        if nsteps < 1 or not is_whole(n):
            raise OptionError(
                f"NLOP.from_solver: T = {T!r} is not a whole number of steps of dt = {solver.dt!r} "
                f"(T/dt = {n!r})",
                why="the horizon must be a whole number of time steps",
                fix=f"pass T = {max(nsteps, 1) * solver.dt!r} or change the solver's dt")
        fm = FlowMap(solver, nsteps, t0=t0, checkpoints=checkpoints,
                     checkpoint_store=checkpoint_store, checkpoint_dir=checkpoint_dir)
        if getattr(solver.plan, "is_sphere", False):
            # spherical shell: energy ½∫|u|² dV with the r² sin θ weights, real
            # pairing, admissible set = discrete Leray + shell BCs + real m = 0 modes
            from .sphere_energy import SphereAdmissibleProjector, SphereEnergyInnerProduct
            ip = SphereEnergyInnerProduct(solver, tuple(fields))
            return cls(fm, ip, SphereAdmissibleProjector(ip, rank_tol=rank_tol), E0, grow=grow,
                       alpha_min=alpha_min)
        ip = EnergyInnerProduct(solver, tuple(fields))
        return cls(fm, ip, AdmissibleProjector(ip, rank_tol=rank_tol), E0, grow=grow,
                   alpha_min=alpha_min)

    def __init__(self, flowmap, ip: EnergyInnerProduct, projector: AdmissibleProjector,
                 E0: float, *, grow: float, alpha_min: float) -> None:
        if not E0 > 0:
            raise OptionError(f"E0 must be > 0 (got {E0!r}): it is the energy of the seed, "
                              "1/2 int |u|^2 dV of the declared velocity fields",
                              fix="E0=1e-4 (the energy of the seed, > 0)")
        self.fm, self.ip, self.P = flowmap, ip, projector
        self.E0, self.grow, self.alpha_min = float(E0), float(grow), float(alpha_min)
        self.n_forward = self.n_adjoint = 0
        self._t_forward: float | None = None   # wall time of the last forward pass (budgets)

    def start(self, guess, *, alpha: float) -> NLOPState:
        """Optimizer state from any velocity ``guess`` (a global state vector): projected onto
        the admissible set and rescaled to E0; ``alpha`` = first rotation angle (explicit)."""
        if not 0 < alpha <= np.pi / 2:
            raise OptionError(f"NLOP.start: alpha = {alpha!r}",
                              why="alpha is the first rotation angle on the energy sphere, in "
                                  "(0, pi/2]", fix="pass e.g. alpha=0.5")
        return NLOPState(u0=self.to_sphere(guess), alpha=float(alpha))

    # -------------------------------------------------------------- sphere
    def to_sphere(self, v):
        """Admissible projection of v, rescaled to energy E0."""
        u = self.P(v)
        e = self.ip.energy(u)
        if not e > 0:
            raise OptionError("zero admissible part: cannot put this field on the sphere (the guess "
                              "has no velocity, or only components the discrete constraints "
                              "remove; start from a field with a non-zero velocity)",
                              fix="start from a field with a non-zero admissible velocity")
        return u * float(np.sqrt(self.E0 / e))

    def J(self, u0) -> float:
        self.n_forward += 1
        t = time.perf_counter()
        out = self.ip.energy(self.fm(u0))
        self._t_forward = time.perf_counter() - t
        return out

    def gradient(self, u0, *, journal=None, budget_s: float | None = None):
        """(J, g) with g = P W⁻¹ Jᵀ W X_T (exact discrete gradient in the energy inner product).

        ``journal`` (directory): the adjoint sweep is journaled in ``<journal>/vjp`` and, if one
        of this u0 is pending there, resumed without repeating the forward pass (J was saved with
        it); ``budget_s``: the sweep raises ``VJPSuspended`` after that many seconds."""
        if journal is None:
            if budget_s is not None:
                raise OptionError("NLOP.gradient(budget_s=...) without journal=",
                                  fix="gradient(u0, journal='<directory>', budget_s=...)")
            XT = self.fm(u0)
            J = self.ip.energy(XT)
            self.n_forward += 1
            self.n_adjoint += 1
            return J, self.P.riesz(self.fm.vjp(u0, self.ip.W(XT)))
        from .flowmap import journal_extra
        jv = Path(journal) / "vjp"
        ex = journal_extra(jv)
        self.n_adjoint += 1
        if ex is not None and "J" in ex:              # pending sweep (u0 checked by digest)
            y = self.fm.vjp(u0, None, journal=jv, budget_s=budget_s)
            return float(ex["J"]), self.P.riesz(y)
        t = time.perf_counter()
        XT = self.fm(u0)
        J = self.ip.energy(XT)
        self.n_forward += 1
        self._t_forward = time.perf_counter() - t
        left = None if budget_s is None else budget_s - self._t_forward
        y = self.fm.vjp(u0, self.ip.W(XT), journal=jv, budget_s=left, extra={"J": J})
        return J, self.P.riesz(y)

    @renames("conatus.adjoint.nlop:NLOP.linear_gain")
    def linear_gain(self, *, n_modes: int = 1, tol: float = REQUIRED, about=None,
                    max_iterations: int | None = None) -> tuple[np.ndarray, list]:
        """Linear optimal energy gains over the same horizon: the ``n_modes`` largest eigenvalues of
        P W⁻¹ J† W J P (J = tangent of the flow map at the state ``about``, default 0: the
        linearization about the base flow held in the coefficients), by Lanczos (``eigsh``,
        ``tol`` explicit) in the primed coordinates v' = R v where the operator is symmetric.
        Returns (gains in decreasing order, admissible seeds of energy E0). The E0 → 0 limit of
        the NLOP gain (the pairing of FlowMap and of the inner product is the same)."""
        require(tol, "tol", where="NLOP.linear_gain()", suggest="tol=1e-12",
                why="tol is the Lanczos (eigsh) stopping tolerance: it sets the accuracy of the "
                    "returned gains; 1e-12 was the value used before v0.1")
        import scipy.sparse.linalg as spla
        fm, ip, P, xp = self.fm, self.ip, self.P, self.ip.xp
        n = fm.n
        cplx = np.dtype(fm.dtype).kind == "c"
        X0 = xp.zeros(n, dtype=fm.dtype) if about is None else xp.asarray(about, dtype=fm.dtype)

        def to_c(y):
            y = xp.asarray(y)
            return (y[:n] + 1j * y[n:]) if cplx else y

        def to_r(z):
            z = xp.asarray(z)
            out = xp.concatenate([z.real, z.imag]) if cplx else z.real
            return out.get() if hasattr(out, "get") else np.asarray(out)

        def matvec(y):
            v = ip.Rinv(P.project_primed(to_c(np.ravel(y))))
            JV = fm.jvp(X0, v)
            z = fm.vjp(X0, ip.W(JV))
            return to_r(P.project_primed(ip.RinvT(ip.restrict(z))))
        m = 2 * n if cplx else n
        op = spla.LinearOperator((m, m), matvec=matvec, dtype=float)
        rng = np.random.default_rng(0)
        v0 = to_r(P.project_primed(xp.asarray(rng.standard_normal(n) * (1 + 1j if cplx else 1))))
        vals, vecs = spla.eigsh(op, k=n_modes, which="LA", tol=tol, v0=v0, maxiter=max_iterations)
        order = np.argsort(vals)[::-1]
        seeds = []
        for i in order:
            u = ip.Rinv(P.project_primed(to_c(vecs[:, i])))
            seeds.append(u * float(np.sqrt(self.E0 / ip.energy(u))))
        return vals[order], seeds

    def tangent(self, u0, g):
        return g - (self.ip.inner(g, u0) / self.ip.inner(u0, u0)) * u0

    def rotate(self, u0, d, alpha: float):
        """cos α u0 + sin α (‖u0‖/‖d‖) d, d tangent at u0: stays on the sphere exactly."""
        nu = np.sqrt(self.ip.inner(u0, u0))
        nd = np.sqrt(self.ip.inner(d, d))
        return np.cos(alpha) * u0 + np.sin(alpha) * (nu / nd) * d

    def fd_check(self, u0, d, alphas=(1e-2, 1e-3, 1e-4)) -> list[dict]:
        """Directional derivative dJ/dα of J(rotate(u0, d, α)) at 0 from the adjoint gradient,
        against centred differences (error O(α²) expected); d is first made admissible and
        tangent (the gradient is a Riesz representative on the admissible set only)."""
        J0, g = self.gradient(u0)
        d = self.tangent(u0, self.P(d))
        scale = np.sqrt(self.ip.inner(u0, u0) / self.ip.inner(d, d))
        exact = self.ip.inner(g, d) * scale
        out = []
        for a in alphas:
            fd = (self.J(self.rotate(u0, d, a)) - self.J(self.rotate(u0, d, -a))) / (2 * a)
            out.append(dict(alpha=a, adjoint=exact, fd=fd, rel=abs(fd - exact) / abs(exact)))
        return out

    # -------------------------------------------------------------- iterations
    @renames("conatus.adjoint.nlop:NLOP.iterate")
    def iterate(self, st: NLOPState, *, max_iterations: int | None = None, rtol: float, log=None,
                save: str | Path | None = None, budget_s: float | None = None,
                journal: str | Path | None = None) -> NLOPState:
        """Rotation steepest ascent from ``st`` (restartable). Stops at maxit, at optimality
        residual ‖g⊥‖/‖g‖ < rtol, when α < alpha_min, or before ``budget_s`` seconds.

        Without ``journal`` the budget is checked between iterations only (as before). With
        ``journal`` (a directory; ``save`` defaults to ``<journal>/nlop_state.npz``) an iteration
        can stop inside its adjoint sweep or between two forward passes of its line search
        (``st.pending`` says where); the same call on ``NLOPState.load(journal)`` in the next
        job continues it bit for bit (module docstring). ``maxit``: deprecated former name of
        ``max_iterations`` (v0.1)."""
        from .._deprecation import given        # maxit= (v0.1 name) is mapped by @renames
        maxit = given("NLOP.iterate", "max_iterations", max_iterations)
        from .checkpoints import VJPSuspended
        t_start = time.time()
        t_mono = time.perf_counter()                # the clock of the budgets (as FlowMap.vjp)
        ip = self.ip
        if journal is not None:
            Path(journal).mkdir(parents=True, exist_ok=True)
            if save is None:
                save = Path(journal) / "nlop_state.npz"

        def left():
            return None if budget_s is None else budget_s - (time.perf_counter() - t_mono)
        u0 = st.u0
        while st.it < maxit:
            t_it = time.time()
            pend = st.pending
            if (pend is None and journal is not None and budget_s is not None
                    and self._t_forward is not None and left() < 1.2 * self._t_forward):
                # not enough time left for the forward pass of the gradient (not suspendable):
                # stop cleanly here rather than be killed in the middle of it
                st.u0 = u0
                st.save(save)
                return st
            if pend is None or pend["phase"] == "gradient":
                try:
                    J0, g = (self.gradient(u0) if journal is None else
                             self.gradient(u0, journal=journal, budget_s=left()))
                except VJPSuspended:
                    st.pending, st.u0 = {"phase": "gradient"}, u0
                    st.save(save)
                    return st
                gt = self.tangent(u0, g)
                res = float(np.sqrt(ip.inner(gt, gt) / ip.inner(g, g)))
                rec = dict(it=st.it, J=J0, gain=J0 / self.E0, residual=res, alpha=st.alpha)
                if res < rtol:
                    rec["stop"] = "residual"
                    st.history.append(rec)
                    st.J, st.pending, st.gt, st.u0 = J0, None, None, u0
                    if log is not None:
                        log(rec)
                    if save is not None:             # the stop is saved too: a resumed job must
                        st.save(save)                # not recompute this gradient
                    break
                tries = 0
                if journal is not None:             # the gradient is done: never redo it
                    st.pending, st.gt, st.u0 = dict(phase="line_search", J_grad=J0, rec=rec, tries=0), gt, u0
                    st.save(save)
            else:                                   # resume a line search
                J0, rec, tries, gt = pend["J_grad"], dict(pend["rec"]), pend["tries"], st.gt
            while True:
                if (journal is not None and budget_s is not None and self._t_forward is not None
                        and left() < 1.2 * self._t_forward):
                    st.pending = dict(phase="line_search", J_grad=J0, rec=rec, tries=tries)
                    st.gt, st.u0 = gt, u0
                    st.save(save)
                    return st
                cand = self.rotate(u0, gt, st.alpha)
                J1 = self.J(cand)
                tries += 1
                if J1 > J0:
                    u0, st.J = cand, J1
                    rec.update(accepted_alpha=st.alpha, J_new=J1, tries=tries)
                    st.alpha = min(st.alpha * self.grow, np.pi / 2)
                    break
                st.alpha /= 2
                if st.alpha < self.alpha_min:
                    rec["stop"] = "alpha_min"
                    break
                if journal is not None:
                    st.pending = dict(phase="line_search", J_grad=J0, rec=rec, tries=tries)
                    st.save(save)
            rec["wall_s"] = time.time() - t_it
            st.history.append(rec)
            st.it += 1
            st.u0, st.pending, st.gt = u0, None, None
            if log is not None:
                log(rec)
            if save is not None:
                st.save(save)
            if rec.get("stop"):
                break
            if budget_s is not None and time.time() - t_start + 1.5 * rec["wall_s"] > budget_s:
                break
        st.u0 = u0
        return st

    def explain(self) -> str:
        return ("NLOP: max E(Φ_T(u0)) on E(u0) = E0 = "
                f"{self.E0:.6e}; gradient P W⁻¹ Jᵀ W X_T (exact discrete adjoint, energy inner "
                "product on the admissible set); rotation update on the sphere, α accepted if J "
                f"increases then α *= {self.grow:g}, else α /= 2 (stop below {self.alpha_min:g}); "
                f"admissible projection rank_tol = {self.P.rank_tol:g}\n"
                + self.fm.explain() + "\n" + self.ip.explain() + "\n" + self.P.explain())
