"""Energy inner product and admissible projection on the spherical shell (NLOP).

The shell counterparts of :class:`conatus.adjoint.nlop.EnergyInnerProduct` and
:class:`~conatus.adjoint.nlop.AdmissibleProjector`, with the same methods (``W, R, Rinv, RinvT,
Winv, restrict, inner, energy`` / ``__call__, riesz, project_primed``), so that :class:`NLOP`
runs unchanged. Two differences, both forced by the complex state of the shell:

* **real pairing**: ⟨a, b⟩ = Re(a^H W b), the pairing of :class:`FlowMap` on the sphere;
* **reality of the m = 0 modes**: the stored modes m > 0 of a real field are free (their -m
  partners are their conjugates), but at m = 0 the spin components must satisfy
  c[flip(comp)] = conj(c[comp]) (u_+ = conj(u_-), u_r and scalars real). The admissible set
  includes these constraints, so that every seed is a real velocity field.

Energy (exact quadrature): with u = sum over spin components c of the (unitary) spin basis,
stored m >= 0 of a RealFourier azimuth, spin-weighted harmonics orthonormal in cos(theta) and
level-0 Chebyshev coefficients in r,

    ∫ |u|² dV = sum_(c, m, l)  w_m  a_(c,m,l)^H  G_r  a_(c,m,l),     w_0 = 2π, w_(m>0) = 4π,

(4π for m > 0: the mode m and its conjugate partner -m; ComplexFourier: 2π for every m) and
G_r[i, j] = ∫_ri^ro T_i T_j r² dr computed by Gauss-Legendre (N + 4 nodes, exact for the
polynomial integrand). E(u) = ½ ⟨u, u⟩. Checked against the grid quadrature of |u|² r² sin θ in
the development test suite.

Admissible set: {v real (m = 0 rule above) : C_v v + C_o o = 0 for some o}, C the algebraic rows
(rows without mass: continuity, boundary conditions, gauges) of the discrete system and o the
pressure (default; ``free_taus=True``: the tau unknowns too). With o = pressure it is the discrete
Leray projection with the shell boundary conditions of this very discretization: the seed is
exactly divergence-free (as a polynomial) and vanishes on both walls. The W-orthogonal projection is computed per group of the
constraint graph in the primed coordinates v' = R v (W = R^H R), over the reals (Re, Im stacked):
the rows that o can absorb are removed (SVD of C_o), the remaining rows and the reality rows give
the orthogonal complement of the admissible space (SVD, declared ``rank_tol``).
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from ..errors import OptionError
from ..discretize.sphere_eval import _flip
from ..ir.bases import RealFourier
from ..discretize.jacobi_family import gauss_legendre

__all__ = ["SphereEnergyInnerProduct", "SphereAdmissibleProjector", "shell_radial_gram"]


def shell_radial_gram(N: int, ri: float, ro: float) -> np.ndarray:
    """G[i, j] = ∫_ri^ro T_i(t(r)) T_j(t(r)) r² dr (level-0 Chebyshev of the shell, exact)."""
    t, w = gauss_legendre(N + 4)
    c, h = 0.5 * (ri + ro), 0.5 * (ro - ri)
    T = np.cos(np.arange(N)[None, :] * np.arccos(t)[:, None])          # (nq, N)
    r = c + h * t
    return h * (T * (w * r ** 2)[:, None]).T @ T


class SphereEnergyInnerProduct:
    """⟨a, b⟩ = Re(a^H W b) on the global state of a shell IVPSolver (module docstring);
    E(X) = ½ ⟨X, X⟩. ``fields``: names of the velocity unknowns (vector fields, or scalars)."""

    pairing = "real"

    def __init__(self, solver, fields: tuple[str, ...]) -> None:
        self.solver, self.fields = solver, tuple(fields)
        self.xp = solver.xp
        sysm = solver.system
        host = sysm.host if hasattr(sysm, "host") else sysm
        S = solver.plan.space
        if not S.shell:
            raise OptionError("SphereEnergyInnerProduct: spherical shell only (ShellBasis in r)",
                              fix="SphereEnergyInnerProduct needs a ShellBasis in r (spherical shell)")
        self.host, self.S, self.n = host, S, host.n
        names = [v.name for v in host.variables]
        self.Gr = shell_radial_gram(S.N, S.ops.ri, S.ops.ro)
        self.RG = sla.cholesky(self.Gr, lower=False)                    # Gr = RGᵀ RG
        self.RGinv = sla.solve_triangular(self.RG, np.eye(S.N), lower=False)
        wm = (np.where(S.ms == 0, 2 * np.pi, 4 * np.pi) if isinstance(S.bphi, RealFourier)
              else np.full(len(S.ms), 2 * np.pi))
        self.wm = wm
        self.shapes, self.gidx, self.flat, self.ranks = [], [], [], []
        for f in self.fields:
            if f not in names:
                raise OptionError(f"unknown field {f!r}", fix=f"one of the variables {names}")
            k = names.index(f)
            v = host.variables[k]
            shape = host.var_shapes[k]
            if not all(S.has(v.domain)):
                raise OptionError(f"{f}: a field on the full (phi, theta, r) domain is expected",
                                  fix="declare the velocity fields on the full (phi, theta, r) domain")
            sel = host._col_sel[k]
            self.shapes.append(shape)
            self.ranks.append(v.rank)
            self.gidx.append(sel)
            self.flat.append(host.cols_flat[sel])
        self.vcols = np.concatenate(self.gidx)
        self.live = np.ones(self.vcols.size, bool)
        xp = self.xp
        self._d = []
        for shape, sel, flat in zip(self.shapes, self.gidx, self.flat):
            sh = [1] * len(shape)
            sh[-3] = shape[-3]
            w = wm.reshape(sh)
            self._d.append((xp.asarray(sel), xp.asarray(flat), shape, xp.asarray(w),
                            xp.asarray(np.sqrt(w)), xp.asarray(1 / np.sqrt(w))))
        self._G, self._RG, self._RGinv = xp.asarray(self.Gr), xp.asarray(self.RG), xp.asarray(self.RGinv)

    def _apply(self, X, mat_name: str, wkind: int, transpose: bool = False):
        """Y = (w_m factor) ⊗ (radial matrix) on the velocity part of X; 0 elsewhere."""
        xp = self.xp
        A = {"G": self._G, "R": self._RG, "Rinv": self._RGinv}[mat_name]
        if transpose:
            A = A.T                                  # real matrices: the adjoint is the transpose
        Y = xp.zeros(self.n, dtype=X.dtype)
        for sel, flat, shape, w, sqw, isq in self._d:
            a = xp.zeros(int(np.prod(shape)), dtype=X.dtype)
            a[flat] = X[sel]
            b = a.reshape(shape) @ A.T                # radial axis last
            b = b * (w, sqw, isq)[wkind]
            Y[sel] = b.reshape(-1)[flat]
        return Y

    def W(self, X):
        """W X (velocity part; zero on pressure, taus, gauges)."""
        return self._apply(X, "G", 0)

    def R(self, X):
        return self._apply(X, "R", 1)

    def Rinv(self, X):
        return self._apply(X, "Rinv", 2)

    def RinvT(self, X):
        return self._apply(X, "Rinv", 2, transpose=True)

    def Winv(self, Y):
        return self.Rinv(self.RinvT(Y))

    def restrict(self, X):
        xp = self.xp
        Y = xp.zeros_like(X)
        v = xp.asarray(self.vcols)
        Y[v] = X[v]
        return Y

    def inner(self, a, b) -> float:
        return float(self.xp.real(self.xp.vdot(a, self.W(b))))

    def energy(self, X) -> float:
        return 0.5 * self.inner(X, X)

    def explain(self) -> str:
        return (f"SphereEnergyInnerProduct: <a, b> = Re(a^H W b), W = exact energy Gram of "
                f"{self.fields} on the shell (w_m = 2π at m = 0, 4π at m > 0 for the stored "
                f"modes of a real field; spin harmonics orthonormal; Chebyshev Gram ∫ T_i T_j r² dr "
                f"by Gauss-Legendre); zero on pressure, taus and gauges; E = ½ <X, X> = ½ ∫ |u|² dV")


class SphereAdmissibleProjector:
    """W-orthogonal projection of a velocity field onto the admissible set of the discrete shell
    system, real m = 0 modes included (module docstring). ``rank_tol`` (explicit): singular
    values below rank_tol · max are the null space."""

    def __init__(self, ip: SphereEnergyInnerProduct, rank_tol: float = 1e-10,
                 free_taus: bool = False) -> None:
        self.ip, self.xp, self.rank_tol = ip, ip.xp, float(rank_tol)
        self.free_taus = bool(free_taus)
        host, S = ip.host, ip.S
        n = host.n
        # unknowns that may absorb the constraint residuals: the pressure (bulk unknowns) only by
        # default, so that a seed is exactly divergence-free with no slip (its taus are zero);
        # free_taus=True also lets the tau unknowns absorb them (every step output is then
        # admissible, but tau-polluted seeds of non-zero divergence too: measured, they add a
        # spurious O(N^-4) excess to the Stokes optimal gain)
        bulk_var = np.array([S.bulk(v.domain) for v in host.variables])
        o_ok = np.ones(n, bool) if self.free_taus else bulk_var[host.cols_obj]
        M, L = host.M.tocsr(), host.L.tocsr()
        alg = np.nonzero(np.diff(M.indptr) == 0)[0]            # rows without mass
        C = L[alg].tocsr()
        nr = C.shape[0]
        # metadata of every velocity column: (field, comp tuple, m index, l, n)
        info = {}
        for fi, (shape, sel, flat, rank) in enumerate(zip(ip.shapes, ip.gidx, ip.flat, ip.ranks)):
            idx = np.unravel_index(flat, shape)
            for j, col in enumerate(sel):
                comp = tuple(int(idx[a][j]) for a in range(rank))
                info[int(col)] = (fi, comp, int(idx[rank][j]), int(idx[rank + 1][j]), int(idx[rank + 2][j]))
        where = {v: k for k, v in info.items()}
        vmask = np.zeros(n, bool)
        vmask[ip.vcols] = True
        real_m0 = isinstance(S.bphi, RealFourier)
        # graph: velocity columns -- constraint rows; columns of one radial block (field, m, l)
        # together (R acts there); m = 0 spin partners together (reality rows)
        Cc = C.tocoo()
        keys = {}
        kid = np.array([keys.setdefault((f, im, ell), len(keys))
                        for (f, _c, im, ell, _n) in (info[int(c)] for c in ip.vcols)])
        rows = [Cc.col, ip.vcols]
        cols = [n + Cc.row, n + nr + kid]
        if real_m0:
            pa, pb = [], []
            for col, (f, comp, im, ell, nn) in info.items():
                if S.ms[im] == 0:
                    partner = where.get((f, _flip(comp), im, ell, nn))
                    if partner is not None:
                        pa.append(col)
                        pb.append(partner)
            rows.append(np.array(pa, int))
            cols.append(np.array(pb, int))
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        A = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n + nr + len(keys),) * 2)
        _, lab = connected_components(A, directed=False)
        Csr = C.tocsr()
        self.groups, ranks = [], []
        drop_max, keep_min = 0.0, np.inf
        RGinv = ip.RGinv
        for g in np.unique(lab[ip.vcols]):
            vc = np.nonzero(vmask & (lab[:n] == g))[0]
            rg = np.nonzero(lab[n:n + nr] == g)[0]
            k = vc.size
            # R^-1 on the group (block-diagonal over (field, comp, m, l): RG^-1 / sqrt(w_m))
            meta = [info[int(c)] for c in vc]
            Rinv = np.zeros((k, k))
            for a_, (fa, ca, ima, la, na) in enumerate(meta):
                for b_, (fb, cb, imb, lb, nb) in enumerate(meta):
                    if (fa, ca, ima, la) == (fb, cb, imb, lb):
                        Rinv[a_, b_] = RGinv[na, nb] / np.sqrt(ip.wm[imb])
            blocks = []
            if rg.size:
                Cg = Csr[rg].toarray()
                oc = np.nonzero((~vmask) & o_ok & (np.abs(Cg).sum(axis=0) > 0))[0]
                K = _realify(Cg[:, vc]) @ np.kron(np.eye(2), Rinv)
                if oc.size:
                    Co = _realify(Cg[:, oc])
                    Uo, so, _ = np.linalg.svd(Co, full_matrices=False)
                    Qo = Uo[:, so > rank_tol * max(so.max(), 1e-300)]
                    K = K - Qo @ (Qo.T @ K)
                blocks.append(K)
            if real_m0:
                # reality at m = 0: Re v_flip - Re v_c = 0, Im v_flip + Im v_c = 0 (c < flip),
                # Im v_c = 0 (own partner: u_r, scalars); rows in v, then times R^-1
                pos = {int(c): i for i, c in enumerate(vc)}
                rr = []
                for i in range(k):
                    f, comp, im, ell, nn = meta[i]
                    if S.ms[im] != 0:
                        continue
                    j = pos.get(where.get((f, _flip(comp), im, ell, nn), -1))
                    if j is None:
                        continue
                    if j == i:
                        row = np.zeros(2 * k)
                        row[k + i] = 1.0
                        rr.append(row)
                    elif i < j:
                        r1, r2 = np.zeros(2 * k), np.zeros(2 * k)
                        r1[j], r1[i] = 1.0, -1.0
                        r2[k + j], r2[k + i] = 1.0, 1.0
                        rr += [r1, r2]
                if rr:
                    blocks.append(np.array(rr) @ np.kron(np.eye(2), Rinv))
            if not blocks:
                continue                                        # unconstrained group
            K = np.vstack(blocks)
            _, s, Vt = np.linalg.svd(K, full_matrices=False)
            if s.size == 0 or s.max() == 0:
                continue
            keep = s > rank_tol * s.max()
            if (~keep).any():
                drop_max = max(drop_max, float(s[~keep].max() / s.max()))
            keep_min = min(keep_min, float(s[keep].min() / s.max()))
            ranks.append(int(keep.sum()))
            self.groups.append((vc, Vt[keep].T.copy()))
        self.gap, self.ranks = (keep_min, drop_max), ranks
        # padded batch over the realified vector [Re X, Im X, 0] (slot 2n = scratch)
        G = len(self.groups)
        m = max(2 * len(vc) for vc, _ in self.groups)
        r = max(U.shape[1] for _, U in self.groups)
        idx = np.full((G, m), 2 * n)
        Ub = np.zeros((G, m, r))
        for i, (vc, U) in enumerate(self.groups):
            idx[i, :2 * len(vc)] = np.concatenate([vc, n + vc])
            Ub[i, :2 * len(vc), :U.shape[1]] = U
        self._idx, self._U = self.xp.asarray(idx), self.xp.asarray(Ub)
        self.n = n

    def project_primed(self, Xp):
        """Euclidean projection (over the reals) in the primed coordinates v' = R v."""
        xp = self.xp
        ext = xp.concatenate([Xp.real, Xp.imag, xp.zeros(1)])
        Z = ext[self._idx]
        c = xp.einsum("gmr,gm->gr", self._U, Z)
        ext[self._idx] = Z - xp.einsum("gmr,gr->gm", self._U, c)
        n = self.n
        return ext[:n] + 1j * ext[n:2 * n]

    def __call__(self, X):
        """P X: W-orthogonal projection of the velocity part of X (other entries: zero)."""
        ip = self.ip
        return ip.Rinv(self.project_primed(ip.R(ip.restrict(X))))

    def riesz(self, Y):
        """P W⁻¹ Y: the admissible Riesz representative of the covector Y (real pairing)."""
        ip = self.ip
        return ip.Rinv(self.project_primed(ip.RinvT(ip.restrict(Y))))

    def explain(self) -> str:
        o = "pressure and taus" if self.free_taus else "pressure only (seeds exactly "
        o += "" if self.free_taus else "divergence-free with no slip, taus zero)"
        return (f"SphereAdmissibleProjector: W-orthogonal projection onto {{v real at m = 0 : "
                f"C_v v + C_o o = 0}}, C = algebraic rows (no mass) of the discrete shell system "
                f"(continuity, no-slip rows, gauge), o = {o}; "
                f"{len(self.groups)} group(s), constraint ranks {min(self.ranks)}..{max(self.ranks)}; "
                f"rank_tol = {self.rank_tol:g} (smallest kept singular value {self.gap[0]:.2e}, "
                f"largest dropped {self.gap[1]:.2e}, relative)")


def _realify(A: np.ndarray) -> np.ndarray:
    """Real matrix of the complex-linear map A on [Re v; Im v]."""
    return np.block([[A.real, -A.imag], [A.imag, A.real]])
