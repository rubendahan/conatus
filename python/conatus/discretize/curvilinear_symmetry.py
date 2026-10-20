"""Symmetry subspaces of curvilinear plans: disk, annulus, cylinder, S2, shell, ball.

Same contract as the Cartesian subspaces: the problem declares a
``SymmetrySubspace`` whose generators are

- ``AzimuthalMFold(phi, m0)``: rotation by 2 pi / m0 (every geometry). Polar and spherical frame
  components are carried by the rotation, so a field of parity p keeps the azimuthal class
  m = 0 (mod m0) (p = +1) or m = m0/2 (mod m0) (p = -1); ``AzimuthalMFold(phi, 2)`` is the
  even / odd m split;
- ``Reflection(theta)`` = ``EquatorialParity(coords)`` (S2, shell, ball): theta -> pi - theta;
- ``Reflection(z)`` on a cylinder: z -> -z (pairs k_z and -k_z).

Each generator acts on the coefficient vector of every object (unknown or equation row) as a
**signed permutation** J (``object_action``): (J X)[a] = sign[a] X[partner[a]]. The subspace is
the set of coefficient vectors fixed by every J; it is spanned by one *representative* per orbit
of coefficients, the other members of the orbit being +- the representative (``factor``), and
orbits on which a J acts as -1 vanish. With E (representative -> members) and P (pick the
representatives), the reduced group system is P_r A E_c: the equations of a fixed vector are fixed,
so the representative rows imply all the others. P E = I, E P = I on the subspace.

Signs (derived in and polar.md, section "Symmetry subspaces";
all verified at plan construction by the equivariance check below, and on the grid by the tests):

- spin-weighted harmonics: Phi_l^{m,s}(-x) = (-1)^(l - max(|m|,|s|)) Phi_l^{m,-s}(x) (checked to
  4e-15). With f(R x) = p R_* f(x) and R_* e_theta = -e_theta, the component of spins
  (s_1..s_q) of a field of tensor parity p satisfies
  c^{(s_i)}_l = p (-1)^(number of tangential indices) (-1)^(l - max(|m|,|sum s_i|)) c^{(-s_i)}_l;
- ball, regularity components (rank 1): diagonal, from the spin rule through Q (sphere_ops), i.e.
  l-1 and l+1 have the scalar sign p (-1)^(l+m), the toroidal l component the opposite sign
  (r x grad Y is a pseudo-vector); computed by Q^H J_spin Q and refused unless diagonal +-1;
- cylinder axial reflection: c_{m,-k} = p (-1)^(number of z indices) c_{m,k};
- rotation: the component is kept iff m is in the class of p.

Nothing is discarded silently: data outside the subspace is refused (``check``), the operators are
checked equivariant on every assembled group (random fixed vector, relative 1e-10, declared), and
``explain()`` prints the kept fraction of every group and the warning that the subspace hides the
symmetry-breaking instabilities (``isotypic`` classes for ``symmetry_breaking_check``).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from ..errors import OptionError, UnsupportedError
from ..ir.symmetry import AzimuthalMFold, Reflection, ShiftReflect, SymmetryError
from .compiler import DiscretizationError

__all__ = ["CurvilinearReduction", "object_action", "EQUIVARIANCE_TOL"]

EQUIVARIANCE_TOL = 1e-10
"""Relative tolerance of the equivariance check of the assembled operators (declared)."""


# ----------------------------------------------------------------------------------------------
# action of one generator on the coefficient array of one object
# ----------------------------------------------------------------------------------------------

def _geometry(plan) -> str:
    S = plan.space
    if getattr(plan, "is_sphere", False):
        return "s2" if S.s2 else ("ball" if S.ball else "shell")
    return "cylinder" if S.cyl else S.ops.kind          # 'disk' | 'annulus' | 'cylinder'


def _check_generator(plan, R) -> None:
    """Refuse a generator the geometry does not support (explicitly, naming the alternative)."""
    geom = _geometry(plan)
    if isinstance(R, AzimuthalMFold):
        if R.coord != plan.space.phi:
            raise SymmetryError(f"{R!r}: acts on {R.coord.name}, the plan's azimuth is {plan.space.phi.name}")
        R.centre(plan.space.bphi)
        return
    if isinstance(R, Reflection):
        if geom in ("s2", "shell", "ball") and R.coord == plan.space.theta:
            return
        if geom == "cylinder" and R.coord == plan.space.zc:
            # coefficients multiply e^{i k (z - a)}: a reflection about z0 pairs k and -k with the
            # phase e^{2 i k (z0 - a)}, equal to 1 for every k iff z0 - a is a multiple of Lz / 2
            bz = plan.space.bz
            z0 = R.centre(bz)
            h = bz.length / 2
            off = (z0 - bz.bounds[0]) % h
            if min(off, h - off) > 1e-12 * bz.length:
                raise SymmetryError(f"{R!r}: the reflection centre must be z = a or a + Lz/2 on the "
                                    f"period [{bz.bounds[0]:g}, {bz.bounds[1]:g}) (a phase otherwise)")
            return
    if isinstance(R, ShiftReflect) and geom == "cylinder" and R.coord == plan.space.phi \
            and R.shift == plan.space.zc:
        from ..ir.problem import IVP
        if not isinstance(plan.problem, IVP):
            raise UnsupportedError(
                f"{R!r}: real-linear (it pairs (m, k_z) with the conjugate of (m, -k_z)), enforced "
                "by the declared projection of an IVP step; EVP/LBVP groups are not invariant "
                "under it (declare it on the IVP, or use AzimuthalMFold / Reflection(z) here)")
        return
    raise UnsupportedError(
        f"{R!r} on a {geom} plan: supported generators are AzimuthalMFold(phi, m0) (all "
        "curvilinear geometries), EquatorialParity / Reflection(theta) (S2, shell, ball), "
        "Reflection(z) and ShiftReflect(phi, z) (cylinder IVPs)")


def _z_mirror_index(plan) -> np.ndarray:
    """Index j' of the axial coefficient of -k_z for every stored z index j (cylinder; the
    Nyquist entry, never represented, maps to itself)."""
    freq = plan.space.zfreq
    lookup = {int(f): j for j, f in enumerate(freq)}
    return np.array([lookup.get(int(-f), j) for j, f in enumerate(freq)], int)


def object_action(plan, dom, rank: int, R, p, cls: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """(partner, sign) over the flat coefficient array of an object (``dom``, tensor ``rank``) of
    tensor parity ``p`` under generator ``R``: a fixed vector satisfies X[a] = sign[a] X[partner[a]].

    ``cls`` selects another isotypic class (symmetry_breaking_check): for a reflection a factor
    eps = -1 multiplies every sign; for a rotation an integer shift j of the kept residue."""
    S = plan.space
    shape = S.coeff_shape(dom, rank)
    size = int(np.prod(shape))
    idx = np.indices(shape).reshape(len(shape), -1)                    # (ndim, size)
    comp = idx[:rank]                                                  # tensor indices
    geom = _geometry(plan)
    partner = np.arange(size)
    p = int(p)
    if geom in ("s2", "shell", "ball"):
        hp, ht, hr = S.has(dom)
        mi, li = idx[rank], idx[rank + 1]
        m = S.ms[mi] if hp else np.zeros(size, int)
        ell = li if ht else np.zeros(size, int)
    else:
        kind = S.kind(dom)
        mi = idx[rank]
        m = S.ms[mi] if kind in ("full", "edge") else np.zeros(size, int)
    if isinstance(R, ShiftReflect):
        # real-linear: not a signed permutation of the complex coefficients; the reduction keeps
        # every entry (identity orbit) and the IVP step projects (timestep/symmetry_projection.py)
        return partner, np.ones(size, int)
    if isinstance(R, AzimuthalMFold):
        r0 = R.residue(p) + (int(cls) if cls is not None else 0)
        keep = (m - r0) % R.m0 == 0
        return partner, np.where(keep, 1, -1)
    eps = -1 if (cls is not None and int(cls) == -1) else 1
    if geom == "cylinder":                                             # z -> -z
        nz_idx = np.zeros(size, int)
        for i in range(rank):
            nz_idx += (comp[i] == 2)
        sign = eps * p * (-1) ** nz_idx
        if S.hz(dom):
            zj = idx[-1]
            zm = _z_mirror_index(plan)[zj]
            pidx = idx.copy()
            pidx[-1] = zm
            partner = np.ravel_multi_index(tuple(pidx), shape)
        return partner, sign
    # equatorial reflection on S2 / shell / ball
    if geom == "ball" and rank == 1:
        # regularity components: diagonal action, Q^H J_spin Q per (m, l)
        sign = np.empty(size, int)
        for a in range(size):
            sign[a] = eps * _ball_reg_sign(plan, int(m[a]), int(ell[a]), int(comp[0][a]), p)
        return partner, sign
    if geom == "ball" and rank > 1:
        raise UnsupportedError("equatorial parity in the ball: tensors of rank >= 2 are not "
                               "represented (rank <= 1, as the ball plans)")
    ntan = np.zeros(size, int)
    s = np.zeros(size, int)
    for i in range(rank):
        ntan += (comp[i] != 2)
        s += np.where(comp[i] == 0, -1, np.where(comp[i] == 1, 1, 0))
    n = ell - np.maximum(np.abs(m), np.abs(s))
    sign = eps * p * (-1) ** ntan * (-1) ** np.abs(n)
    pc = np.where(comp == 0, 1, np.where(comp == 1, 0, comp)) if rank else comp
    pidx = idx.copy()
    pidx[:rank] = pc
    partner = np.ravel_multi_index(tuple(pidx), shape) if rank else partner
    return partner, sign


def _ball_reg_sign(plan, m: int, ell: int, sigma: int, p: int) -> int:
    """Sign of regularity component sigma (0: l-1, 1: l+1, 2: l) at (m, l): Q^H J_spin Q, which
    must be diagonal +-1 (refused otherwise: a wrong derivation, never a silent guess)."""
    key = ("eqsign", m, ell, p)
    cache = plan._cache
    if key not in cache:
        if ell < abs(m):
            cache[key] = np.ones(3, int)
        else:
            Q = plan.Q(m, ell)
            J = np.zeros((3, 3))
            for a, b in ((0, 1), (1, 0)):     # u_-(l) <-> -u_+(l): theta -> pi - theta flips u_theta
                n = ell - max(abs(m), 1)
                J[a, b] = -p * (-1) ** abs(n) if ell >= 1 else 0.0
            J[2, 2] = p * (-1) ** abs(ell - abs(m))
            Jr = Q.conj().T @ J @ Q
            valid = [s_ for s_ in range(3) if s_ == 1 or ell >= 1]
            d = np.real_if_close(np.diag(Jr))
            off = Jr[np.ix_(valid, valid)] - np.diag(np.diag(Jr)[valid])
            if np.abs(off).max() > 1e-12 or any(abs(abs(d[s_]) - 1) > 1e-12 for s_ in valid):
                raise DiscretizationError(
                    f"internal: equatorial action on the ball regularity components at (m, l) = "
                    f"({m}, {ell}) is not a diagonal +-1 (off-diagonal {np.abs(off).max():.1e})")
            out = np.ones(3, int)
            for s_ in valid:
                out[s_] = int(round(float(np.real(d[s_]))))
            cache[key] = out
    return int(cache[key][sigma])


# ----------------------------------------------------------------------------------------------
# reduction of a whole plan
# ----------------------------------------------------------------------------------------------

def _entries(plan, g, which: str) -> tuple[np.ndarray, np.ndarray]:
    """(object index, flat index into the object's coefficient array) of every entry of the full
    group vector (``which`` = 'col': unknowns; 'row': equations), in the plan's group order."""
    S = plan.space
    if which == "col":
        slots = plan.var_slots(g)
        objs = [(v.domain, v.rank) for v in plan.variables]
    else:
        slots = plan.eq_slots(g)
        objs = [(eq.domain, eq.lhs.rank) for eq in plan.problem.equations]
    ob, fl = [], []
    for oi, c, _off, n in slots:
        if n == 0:
            continue
        dom, rank = objs[oi]
        if getattr(plan, "is_sphere", False):
            f, _ = plan.flat_index(dom, rank, c, g)
        else:
            shape = S.coeff_shape(dom, rank)
            f = np.array([np.ravel_multi_index(c + S.cindex(dom, g, j), shape) for j in range(n)], int)
        ob.append(np.full(len(f), oi, int))
        fl.append(np.asarray(f, int))
    if not ob:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(ob), np.concatenate(fl)


class _Orbits:
    """Orbits of the signed permutations of all generators on one index set (columns or rows):
    ``rep[a]`` (the representative, smallest index), ``factor[a]`` (X[a] = factor X[rep]),
    ``zero[a]`` (the orbit vanishes in the subspace)."""

    def __init__(self, n: int, actions: list[tuple[np.ndarray, np.ndarray]]) -> None:
        rep = np.full(n, -1, int)
        factor = np.zeros(n, int)
        zero = np.zeros(n, bool)
        # adjacency: a -- partner(a) with X[a] = sign X[partner]
        for a0 in range(n):
            if rep[a0] >= 0:
                continue
            members, stack = {a0: 1}, [a0]
            bad = False
            while stack:
                a = stack.pop()
                fa = members[a]
                for partner, sign in actions:
                    b, s = int(partner[a]), int(sign[a])
                    if b < 0:
                        # the mirror image of a represented coefficient must be represented too;
                        # dropping the orbit here would be a silent truncation
                        raise DiscretizationError(
                            f"internal: the mirror partner of entry {a} is not represented in the "
                            "plan (symmetry subspace refused)")
                    fb = s * fa                                  # X[b] = X[a] / s = s X[a]
                    if b in members:
                        if members[b] != fb:
                            bad = True
                    else:
                        members[b] = fb
                        stack.append(b)
            r = min(members)
            fr = members[r]
            for a, fa in members.items():
                rep[a], factor[a], zero[a] = r, fa * fr, bad
        self.rep, self.factor, self.zero = rep, factor, zero

    @property
    def kept(self) -> np.ndarray:
        """Indices of the representatives of the non-vanishing orbits."""
        return np.nonzero((self.rep == np.arange(len(self.rep))) & ~self.zero)[0]


class CurvilinearReduction:
    """Symmetry subspace of a PolarPlan / SpherePlan (see module docstring).

    ``cls``: None for the declared subspace, else {generator: class} for another isotypic
    component (reflection: -1; rotation: residue shift j in 1..m0-1), used by
    ``symmetry_breaking_check``."""

    def __init__(self, plan, cls: dict | None = None) -> None:
        pb = plan.problem
        sym = pb.symmetry
        if sym is None:
            raise SymmetryError("no symmetry subspace declared on the problem")
        for R in sym.generators:
            _check_generator(plan, R)
        self.plan, self.sym, self.cls = plan, sym, dict(cls or {})
        groups = list(plan.groups)
        col_e = [_entries(plan, g, "col") for g in groups]
        row_e = [_entries(plan, g, "row") for g in groups]
        self.col_off = np.cumsum([0] + [len(o) for o, _ in col_e])
        self.row_off = np.cumsum([0] + [len(o) for o, _ in row_e])
        cobj = np.concatenate([o for o, _ in col_e]) if col_e else np.zeros(0, int)
        cflat = np.concatenate([f for _, f in col_e]) if col_e else np.zeros(0, int)
        robj = np.concatenate([o for o, _ in row_e]) if row_e else np.zeros(0, int)
        rflat = np.concatenate([f for _, f in row_e]) if row_e else np.zeros(0, int)
        self.col_obj, self.col_flat, self.row_obj, self.row_flat = cobj, cflat, robj, rflat
        self.col_group = np.repeat(np.arange(len(groups)), np.diff(self.col_off))
        self.row_group = np.repeat(np.arange(len(groups)), np.diff(self.row_off))
        variables, eqs = plan.variables, pb.equations
        c_actions, r_actions = [], []
        for R in sym.generators:
            c = self.cls.get(R)
            c_actions.append(self._global_action(
                cobj, cflat, [(v.domain, v.rank, sym.parity(v, R)) for v in variables], R, c))
            r_actions.append(self._global_action(
                robj, rflat, [(eq.domain, eq.lhs.rank, pb.equation_parities[i][R])
                              for i, eq in enumerate(eqs)], R, c))
        self.cols = _Orbits(len(cobj), c_actions)
        self.rows = _Orbits(len(robj), r_actions)
        self._gm: dict = {}

    def _global_action(self, obj, flat, objs, R, cls):
        """Signed permutation of one generator on the stacked entries of all groups."""
        n = len(obj)
        partner = np.full(n, -1, int)
        sign = np.ones(n, int)
        where = {}
        for a in range(n):
            where[(int(obj[a]), int(flat[a]))] = a
        for oi, (dom, rank, p) in enumerate(objs):
            sel = np.nonzero(obj == oi)[0]
            if not sel.size:
                continue
            if p is None:                                     # identically zero equation (parity None)
                partner[sel] = sel
                sign[sel] = -1
                continue
            pa, sg = object_action(self.plan, dom, rank, R, p, cls)
            for a in sel:
                f = int(flat[a])
                partner[a] = where.get((oi, int(pa[f])), -1)
                sign[a] = int(sg[f])
        return partner, sign

    # ------------------------------------------------------------------ per group
    def group_index(self, g) -> int:
        return list(self.plan.groups).index(g)

    def col_members(self, gi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(local rep column, local member column, factor) of the kept orbits inside group gi."""
        lo, hi = self.col_off[gi], self.col_off[gi + 1]
        a = np.arange(lo, hi)
        rep, fac, zero = self.cols.rep[a], self.cols.factor[a], self.cols.zero[a]
        ok = ~zero & (rep >= lo) & (rep < hi)
        return rep[ok] - lo, a[ok] - lo, fac[ok]

    def kept_cols(self, gi: int) -> np.ndarray:
        lo, hi = self.col_off[gi], self.col_off[gi + 1]
        k = self.cols.kept
        return k[(k >= lo) & (k < hi)] - lo

    def kept_rows(self, gi: int) -> np.ndarray:
        lo, hi = self.row_off[gi], self.row_off[gi + 1]
        k = self.rows.kept
        return k[(k >= lo) & (k < hi)] - lo

    def E(self, gi: int) -> sp.csr_matrix:
        """Injection of group gi: reduced (kept representatives) -> full group vector (members of
        orbits whose representative is in another group are not in this group's E)."""
        lo, hi = self.col_off[gi], self.col_off[gi + 1]
        kc = self.kept_cols(gi)
        pos = {int(c): i for i, c in enumerate(kc)}
        r, m, f = self.col_members(gi)
        keep = np.array([int(x) in pos for x in r], bool)
        return sp.csr_matrix((f[keep].astype(float), (m[keep], [pos[int(x)] for x in r[keep]])),
                             shape=(hi - lo, len(kc)))

    def reduce(self, gm_full, gi: int):
        """Reduced GroupMatrices of group gi (rows: kept representatives; columns: P_r A E)."""
        from .plan import GroupMatrices
        kr, kc = self.kept_rows(gi), self.kept_cols(gi)
        E = self.E(gi)
        L = (gm_full.L[kr] @ E).tocsr()
        M = (gm_full.M[kr] @ E).tocsr()
        gm = GroupMatrices(gm_full.group, gm_full.label + " [symmetry subspace]", L, M,
                           [gm_full.rows[i] for i in kr], [gm_full.cols[j] for j in kc], kr, kc,
                           gm_full.lam, reduced=True)
        if hasattr(gm_full, "row_info"):
            gm.row_info = gm_full.row_info
        gm.E = E
        return gm

    def check_equivariance(self, gm_full, gi: int, seed: int = 0) -> float:
        """Relative violation of (A X) fixed for a random fixed X, rows inside the group: the
        discrete operators must commute with the action (declared tolerance EQUIVARIANCE_TOL)."""
        E = self.E(gi)
        if E.shape[1] == 0 or gm_full.L.shape[0] == 0:
            return 0.0
        rng = np.random.default_rng(seed)
        y = rng.standard_normal(E.shape[1]) + 1j * rng.standard_normal(E.shape[1])
        x = E @ y
        worst = 0.0
        lo, hi = self.row_off[gi], self.row_off[gi + 1]
        a = np.arange(lo, hi)
        rep, fac, zero = self.rows.rep[a], self.rows.factor[a], self.rows.zero[a]
        inside = (rep >= lo) & (rep < hi)
        for A in (gm_full.L, gm_full.M):
            b = A @ x
            scale = max(np.abs(b).max(), 1e-300)
            viol = np.zeros(hi - lo)
            viol[zero] = np.abs(b[zero])
            ok = ~zero & inside
            viol[ok] = np.abs(b[ok] - fac[ok] * b[rep[ok] - lo])
            # np.max keeps a NaN violation (review of the audit "assurances")
            worst = float(np.max([worst, float(viol.max() / scale) if viol.size else 0.0]))
        return worst

    # ------------------------------------------------------------------ whole coefficient arrays
    def scatter_maps(self, var_shapes) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Per variable: (global kept index, flat index, factor) of every member (all groups) of a
        kept orbit, in the reduced global numbering (kept columns in group order)."""
        kept = self.cols.kept
        red = np.full(len(self.cols.rep), -1, int)
        red[kept] = np.arange(len(kept))
        a = np.nonzero(~self.cols.zero)[0]
        out = []
        for k in range(len(var_shapes)):
            s = a[self.col_obj[a] == k]
            out.append((red[self.cols.rep[s]], self.col_flat[s], self.cols.factor[s]))
        return out

    def symmetrize(self, field, coeffs: np.ndarray) -> np.ndarray:
        """Orthogonal projection of a coefficient array of ``field`` (a variable of the problem)
        onto its class: (c + J c) / 2 per reflection, the kept azimuthal class per rotation
        (commuting projections; E P in the coefficient layout). A tool for building symmetric
        data explicitly -- the solvers never apply it to data (they refuse asymmetric data)."""
        c = np.array(coeffs, dtype=complex).reshape(-1)
        for R in self.sym.generators:
            pa, sg = object_action(self.plan, field.domain, field.rank, R, self.sym.parity(field, R),
                                   self.cls.get(R))
            c = 0.5 * (c + sg * c[pa])
        return c.reshape(np.shape(coeffs))

    def explain(self) -> str:
        nfull, nred = len(self.cols.rep), len(self.cols.kept)
        rfull, rred = len(self.rows.rep), len(self.rows.kept)
        kept_groups = sum(1 for gi in range(len(self.plan.groups)) if len(self.kept_cols(gi)))
        cls = "" if not self.cls else " isotypic class " + ", ".join(f"{c} under {R!r}" for R, c in self.cls.items())
        proj = [R for R in self.sym.generators if isinstance(R, ShiftReflect)]
        extra = ("; " + ", ".join(repr(R) for R in proj) + ": not reduced (real-linear), enforced by "
                 "the declared projection of the IVP step") if proj else ""
        return (f"curvilinear symmetry subspace{cls}: {nred} / {nfull} unknowns kept "
                f"(x{nfull / max(nred, 1):.2f}), {rred} / {rfull} equation rows, {kept_groups} / "
                f"{len(self.plan.groups)} groups non-empty; operators checked equivariant on every "
                f"assembled group (relative {EQUIVARIANCE_TOL:g}); data outside the subspace refused"
                + extra)


# ----------------------------------------------------------------------------------------------
# plan-level helpers shared by PolarPlan and SpherePlan
# ----------------------------------------------------------------------------------------------

def plan_matrices(plan, group, reduced: bool | None):
    """``plan.matrices(group, reduced)``: full group matrices, or the symmetry subspace's (default
    when the problem declares one). A group with no kept coefficient is not assembled (0 x 0)."""
    from .plan import GroupMatrices
    g = plan.groups[0] if group is None else group
    if reduced is None:
        reduced = plan.problem.symmetry is not None
    if not reduced:
        return plan._assemble(g)
    sub = plan.subspace
    if sub is None:
        raise SymmetryError("no symmetry subspace declared on the problem")
    key = ("reduced", g)
    if key not in plan._cache:
        gi = sub.group_index(g)
        if not len(sub.kept_cols(gi)) and not len(sub.kept_rows(gi)):
            z = sp.csr_matrix((0, 0), dtype=complex)
            gm = GroupMatrices(g, g.label + " [symmetry subspace: empty]", z, z, [], [],
                               np.zeros(0, int), np.zeros(0, int), {}, reduced=True)
            gm.E = sp.csr_matrix((sub.col_off[gi + 1] - sub.col_off[gi], 0))
        else:
            full = plan._assemble(g)
            err = sub.check_equivariance(full, gi)
            if not err <= EQUIVARIANCE_TOL:             # NaN: no pass
                raise DiscretizationError(
                    f"[{full.label}] the discrete operators are not equivariant under the declared "
                    f"symmetry (relative violation {err:.1e} > {EQUIVARIANCE_TOL:g}): a non-constant "
                    "coefficient's data does not have its declared parity, or an internal sign error")
            gm = sub.reduce(full, gi)
            gm.equivariance = err
        plan._cache[key] = gm
    return plan._cache[key]


def plan_embed(plan, x_red: np.ndarray, group=None) -> np.ndarray:
    """E: subspace group vector -> full group vector (members of the group's kept orbits)."""
    g = plan.groups[0] if group is None else group
    return plan_matrices(plan, g, True).E @ np.asarray(x_red)


def plan_project(plan, x_full: np.ndarray, group=None) -> np.ndarray:
    """P: full group vector -> subspace coefficients (the kept representatives); P E = I."""
    g = plan.groups[0] if group is None else group
    return np.asarray(x_full)[plan_matrices(plan, g, True).col_index]


def check_rows_in_subspace(sub: CurvilinearReduction, gi: int, b: np.ndarray, what: str) -> None:
    """Refuse right-hand-side data outside the subspace (never projected away in silence)."""
    lo, hi = sub.row_off[gi], sub.row_off[gi + 1]
    a = np.arange(lo, hi)
    rep, fac, zero = sub.rows.rep[a], sub.rows.factor[a], sub.rows.zero[a]
    inside = ~zero & (rep >= lo) & (rep < hi)
    scale = max(float(np.abs(b).max()) if b.size else 0.0, 1e-300)
    viol = 0.0
    if zero.any():
        viol = float(np.abs(b[zero]).max())
    if inside.any():
        viol = max(viol, float(np.abs(b[inside] - fac[inside] * b[rep[inside] - lo]).max()))
    if viol > 1e-12 * scale:
        raise SymmetryError(f"{what}: the data has components outside the symmetry subspace "
                            f"(max {viol:.2e}, relative {viol / scale:.1e}): forcing/BC data break "
                            "the symmetry", fix="symmetrize the data explicitly "
                            "(plan.subspace.symmetrize(field, coeffs)) or drop the declared symmetry")


def reduced_lbvp_solve(plan, rhs: list, out: dict) -> list:
    """LBVP in the subspace: one sparse LU per non-empty reduced group; the solution is written into
    every member of each kept orbit (all groups); residuals reported per group."""
    import scipy.sparse.linalg as spla
    sub = plan.subspace
    parts, residuals = [], []
    for gi, g in enumerate(plan.groups):
        gm = plan_matrices(plan, g, True)
        if gm.L.shape[0] == 0:
            parts.append(np.zeros(0, complex))
            continue
        b_full = plan.group_rhs(g, rhs)
        check_rows_in_subspace(sub, gi, b_full, f"[{gm.label}] LBVP data")
        b = b_full[gm.row_index]
        A = gm.L.tocsc()
        x = spla.splu(A).solve(b)
        residuals.append((gm.label, float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300))))
        parts.append(x)
    X = np.concatenate(parts) if parts else np.zeros(0, complex)
    for k, (xi, fl, fac) in enumerate(sub.scatter_maps([None] * len(plan.variables))):
        v = plan.variables[k]
        out[v.name].reshape(-1)[fl] = fac * X[xi]
    return residuals


def isotypic_classes(sym) -> list[dict]:
    """All isotypic classes of the declared group relative to the declared subspace: every
    combination of (reflection: +1 / -1) and (rotation m0: residue shift 0..m0-1); the declared
    subspace is the class {} (all +1 / shift 0)."""
    from itertools import product
    choices = []
    for R in sym.generators:
        if isinstance(R, ShiftReflect):
            raise UnsupportedError(
                f"isotypic classes under {R!r}: the group (m, k_z) of an EVP is not invariant under "
                "it (m-fold / reflection classes only; for a travelling wave in the shift-reflect "
                "subspace the symmetry-breaking check is the time-stepper's, conatus.ecs.pipe)")
        choices.append([(R, c) for c in ((1, -1) if not isinstance(R, AzimuthalMFold) else range(R.m0))])
    out = []
    for combo in product(*choices):
        out.append({R: c for R, c in combo if not (c == 1 and not isinstance(R, AzimuthalMFold))
                    and not (c == 0 and isinstance(R, AzimuthalMFold))})
    return out


def symmetry_breaking_check(plan, group=None, *, method: str = "dense", target: complex | None = None,
                            k: int = 8, unstable_tol: float, matrices=None) -> dict:
    """Leading eigenvalue of an EVP group in **every isotypic class** of the declared group
    ("Safeguard"): the declared subspace sees only its own class; the other
    classes (reflection sign -1; rotation residues shifted by 1..m0-1) are assembled with the same
    plan and the same equivariance check. An eigenvalue with Re > ``unstable_tol`` (explicit) outside
    the declared class is a symmetry-breaking instability, reported in the verdict.

    ``matrices``: optional callable (reduction, group index) -> (L, M) for pencils assembled
    outside the plan (e.g. affine in a parameter); default: the plan's group matrices."""
    from ..solvers.evp import solve_dense, solve_sparse
    sym = plan.problem.symmetry
    if sym is None:
        raise SymmetryError("symmetry_breaking_check: no symmetry subspace declared on the problem")
    g = plan.groups[0] if group is None else group
    full = plan._assemble(g)
    out, lead = {}, {}
    for cls in isotypic_classes(sym):
        red = plan.subspace if not cls else CurvilinearReduction(plan, cls)
        gi = red.group_index(g)
        name = "declared" if not cls else ", ".join(
            f"{'sign -1' if not isinstance(R, AzimuthalMFold) else f'residue +{c}'} under {R!r}"
            for R, c in cls.items())
        if not len(red.kept_cols(gi)):
            out[name] = np.zeros(0, complex)
            continue
        err = red.check_equivariance(full, gi)
        if not err <= EQUIVARIANCE_TOL:             # NaN: no pass
            raise DiscretizationError(f"[{full.label}] class {name}: operators not equivariant ({err:.1e})")
        if matrices is not None:
            L, M = matrices(red, gi)
        else:
            gm = red.reduce(full, gi)
            L, M = gm.L, gm.M
        if method == "dense":
            res = solve_dense(L, M, label=name)
        elif method == "shift-invert":
            if target is None:
                raise OptionError("shift-invert needs a target",
                                  fix="pass target= (a complex shift near the wanted eigenvalues)")
            res = solve_sparse(L, M, target, k, label=name)
        else:
            raise OptionError(f"unknown method {method!r} (dense | shift-invert)",
                              fix="method='dense' or method='shift-invert' with target=")
        vals = np.asarray(res.values)
        out[name] = vals[np.argsort(-vals.real)]
        lead[name] = complex(out[name][0]) if vals.size else None
    breaking = {n: v for n, v in lead.items() if n != "declared" and v is not None and v.real > unstable_tol}
    verdict = (f"symmetry-breaking instability in {sorted(breaking)} (Re sigma > {unstable_tol:g})"
               if breaking else f"no symmetry-breaking instability (Re sigma <= {unstable_tol:g} in "
               f"every other class)")
    return {"classes": out, "leading": lead, "verdict": verdict, "unstable_tol": float(unstable_tol)}


# ----------------------------------------------------------------------------------------------
# folded evaluation of right-hand sides in a subspace (the cost reduction of F)
# ----------------------------------------------------------------------------------------------

class Fold:
    """How one value of a folded evaluation lives on the reduced grid.

    ``half``: tensor parity (+1 / -1) under the equatorial reflection when the colatitude grid is
    folded to the northern hemisphere (rows theta_j, j < ceil(Mtheta / 2); the south is the mirror
    image, u_s(pi - theta) = p (-1)^(tangential indices) u_{-s}(theta)); None: no theta folding.
    ``m0, r0``: the azimuthal grid is the sector [0, 2 pi / m0) and the value is in the class
    m = r0 (mod m0) (f(phi) = e^{i r0 phi} g(phi), g of period 2 pi / m0; values carried by the
    rotation, no component sign); m0 = 1: full azimuth. ``zq``: tensor parity under the axial
    reflection z -> -z of a cylinder when the z grid is folded to z_j, j <= Mz / 2 (component sign
    q = zq (-1)^(z indices); coefficients c_{-k} = q c_k, only k >= 0 computed); None: no z
    folding. ``zero``: the value vanishes identically in the subspace (a parity None)."""

    __slots__ = ("half", "m0", "r0", "zero", "zq")

    def __init__(self, half=None, m0: int = 1, r0: int = 0, zero: bool = False, zq=None) -> None:
        self.half, self.m0, self.r0, self.zero, self.zq = half, int(m0), int(r0), bool(zero), zq

    def __repr__(self) -> str:
        return f"Fold(half={self.half}, m0={self.m0}, r0={self.r0}, zero={self.zero}, zq={self.zq})"


class FoldSpec:
    """Folding of an evaluator for a problem's symmetry subspace: which generator folds the
    colatitude (equatorial reflection) and which folds the azimuth (m-fold rotation), and the
    parity of every expression (``parity_of``, cached). ``Mphi``: dealiased azimuthal size; the
    sector needs Mphi divisible by m0, otherwise the azimuth is not folded (declared in ``explain``)."""

    def __init__(self, problem, Mphi: int, theta: bool, zcoord=None) -> None:
        from ..ir.symmetry import ParityLookup, parity_of
        sym = problem.symmetry
        self.lookup = ParityLookup(sym)
        self._parity_of = parity_of
        self.eq = next((R for R in sym.generators if isinstance(R, Reflection) and getattr(R, "equatorial", False)), None)
        if not theta:
            self.eq = None
        # axial reflection of a cylinder (``zcoord``: the plan's z coordinate)
        self.zr = next((R for R in sym.generators if isinstance(R, Reflection) and zcoord is not None
                        and R.coord == zcoord), None)
        rot = next((R for R in sym.generators if isinstance(R, AzimuthalMFold)), None)
        self.rot = rot if (rot is not None and Mphi % rot.m0 == 0) else None
        self.rot_refused = rot is not None and self.rot is None
        self.Mphi = Mphi
        self._cache: dict = {}

    @property
    def active(self) -> bool:
        return self.eq is not None or self.rot is not None or self.zr is not None

    def fold(self, x) -> Fold:
        """Fold of the value of expression ``x`` (parities from the declared subspace)."""
        f = self._cache.get(x)
        if f is None:
            zero = False
            half = None
            m0, r0 = 1, 0
            if self.eq is not None:
                p = self._parity_of(x, self.eq, self.lookup)
                zero = zero or p is None
                half = int(p) if p is not None else 1
            if self.rot is not None:
                p = self._parity_of(x, self.rot, self.lookup)
                zero = zero or p is None
                m0, r0 = self.rot.m0, (self.rot.residue(p) if p is not None else 0)
            zq = None
            if self.zr is not None:
                p = self._parity_of(x, self.zr, self.lookup)
                zero = zero or p is None
                zq = int(p) if p is not None else 1
            f = self._cache[x] = Fold(half, m0, r0, zero, zq)
        return f

    def derived(self, f: Fold, det_flip: bool = False) -> Fold:
        """Fold of a value derived from one of fold ``f`` by grad / div / lap (same parities) or by
        a curl (``det_flip``: the reflection's parity changes sign, the rotation's does not)."""
        if not det_flip or (f.half is None and f.zq is None):
            return f
        return Fold(None if f.half is None else -f.half, f.m0, f.r0, f.zero,
                    None if f.zq is None else -f.zq)

    def explain(self) -> str:
        parts = []
        if self.eq is not None:
            parts.append("colatitude folded to the northern hemisphere (equatorial parity: half the "
                         "theta points, half the Legendre work)")
        if self.rot is not None:
            parts.append(f"azimuth folded to the sector [0, 2 pi/{self.rot.m0}) (m0-fold class: "
                         f"1/{self.rot.m0} of the phi points)")
        if self.zr is not None:
            parts.append("axial grid folded to z in [0, Lz/2] (reflection z -> -z: half the z points, "
                         "only k_z >= 0 computed)")
        if self.rot_refused:
            parts.append(f"azimuth NOT folded: dealiased Nphi = {self.Mphi} is not a multiple of m0 "
                         "(full phi grid, declared)")
        return "; ".join(parts) or "no folding"


def sector_synth(V, Vflip, ms: np.ndarray, Mp: int, real: bool, fold: Fold, xp):
    """Azimuthal synthesis on the sector of ``fold`` (axis 0 of V = stored wavenumbers ms): values
    at phi_j = 2 pi j / Mp, j < Mp / m0. ``Vflip``: the spin-flipped component (real fields: the
    mode -m is its conjugate). Same result as the full-grid synthesis restricted to the sector."""
    m0, r0 = fold.m0, fold.r0
    Mq = Mp // m0
    keep = ((ms - r0) % m0) == 0
    S = xp.zeros((Mq,) + tuple(V.shape[1:]), dtype=complex)
    idx = np.nonzero(keep)[0]
    if idx.size:
        S[((ms[idx] - r0) // m0) % Mq] = V[idx]
    if real:
        neg = idx[ms[idx] > 0]                              # modes -m, m > 0 in the class
        if neg.size:
            S[((-ms[neg] - r0) // m0) % Mq] = xp.conj(Vflip[neg])
    g = xp.fft.ifft(S, axis=0) * Mq
    if r0:
        tw = np.exp(1j * r0 * 2 * np.pi * np.arange(Mq) / Mp).reshape((Mq,) + (1,) * (V.ndim - 1))
        g = g * xp.asarray(tw)
    return g


def kept_index(ms: np.ndarray, fold: Fold) -> np.ndarray:
    """Positions of the stored wavenumbers in the class of ``fold`` (m = r0 mod m0)."""
    return np.nonzero(((np.asarray(ms) - fold.r0) % fold.m0) == 0)[0]


def sector_analysis(g, ms: np.ndarray, Mp: int, fold: Fold, xp, kept_only: bool = False):
    """Azimuthal projection from the sector of ``fold`` (axis 0 of g, Mp / m0 points) onto the
    stored wavenumbers ms (zero outside the class; ``kept_only``: the rows of the class only, in
    the order of ``kept_index``): equal to the full-grid FFT projection of the class function the
    sector represents."""
    m0, r0 = fold.m0, fold.r0
    Mq = Mp // m0
    if r0:
        tw = np.exp(-1j * r0 * 2 * np.pi * np.arange(Mq) / Mp).reshape((Mq,) + (1,) * (g.ndim - 1))
        g = g * xp.asarray(tw)
    Y = xp.fft.fft(g, axis=0) / Mq
    keep = kept_index(ms, fold)
    if kept_only:
        return Y[((ms[keep] - r0) // m0) % Mq]
    out = xp.zeros((len(ms),) + tuple(g.shape[1:]), dtype=complex)
    if keep.size:
        out[keep] = Y[((ms[keep] - r0) // m0) % Mq]
    return out
