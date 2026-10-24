"""Discretization plans on S2, spherical shells and balls.

``discretize(problem)`` dispatches here for problems on ``S2Coordinates`` / ``SphericalCoordinates``.
The public face is the Cartesian/polar ``Plan``'s: ``groups``, ``matrices(group)``, ``rank_report`` /
``check`` (refused before any factorization), ``solve`` (LBVP, residuals reported), ``explain()``,
``methods()``; EVPs go through ``conatus.solvers.eigenvalues`` and IVPs through ``IVPSolver``.

Groups: on S2 one group per azimuthal wavenumber m (all degrees l; the linear operators are diagonal
in l, axisymmetric angular coefficients such as 2 Omega cos(theta) couple neighbouring l). In 3D one
group per (m, l) (rotation-invariant operators decouple the degrees), or one group per m holding every
l when a coefficient depends on theta (e.g. Coriolis with e_z = cos(theta) e_r - sin(theta) e_theta;
declared in ``explain()``).

Inside a group every tensor component is a spin component (S2, shell, surfaces) or a regularity
component (ball bulk) and the IR nodes compile to:

    S2 and shell (spin components, Leibniz rule, sphere_ops):
      grad    (G_- ., G_+ .) / r (or / R on S2) + d_r (radial index) + frame terms
              (grad u)_{+,-} += u_r / r, (grad u)_{+,r} -= u_+ / r, (grad u)_{-,+} += u_r / r,
              (grad u)_{-,r} -= u_- / r (index by index; none on S2)
      div     metric contraction of grad (e_-.e_+ = 1, e_r.e_r = 1);  lap = div(grad);  curl from
              grad with the spin cross product (3D) or the S2 conventions curl(psi) = grad(psi) x e_r,
              curl(u) = e_r . curl(u)
      shell rows are multiplied by r^p, p = largest r-power of their terms (declared; d_r and 1/r
      raise p by one; a tau lifted at level lam is the function r^-lam C^(lam)_{N+n}, held at
      p = lam), exactly Dedalus's r^-k Jacobi(k-1/2, k-1/2) derivative bases.
    ball (regularity components, rank <= 1): grad, div, curl as xi D_+- (the spin operators conjugated
      by the unitary Q of sphere_ops.regularity_Q; every entry is identified as a single regular
      operator or refused), lap = D_-(k+1) D_+(k) per component. Surface quantities (taus, boundary
      values) stay in spin components.
    interp(., r, r0) and interp(diff(., r), r, r0), integ over the whole domain, lift(tau, basis, n,
    lam) (mode N_l + n of level lam), the radial component (shell, surfaces), dt, the EVP eigenvalue,
    constants, parameters and axisymmetric non-constant coefficients (S2 and shell: any (theta, r)
    dependence, scalar or tensor, in *, dot and cross; ball: spherically symmetric scalars).

Nothing is ever divided by r or sin(theta): regularity at the poles and at the centre is in the bases.

Layout (pure split; every name importable from here as before): ``sphere_blocks`` (groups, block
entries, ball regular operators), ``sphere_compiler`` (IR -> blocks for one group, NCC products),
this module (``SpherePlan``: layouts, NCC data, assembly, solve, explain).
"""
from __future__ import annotations

from typing import Any
from collections.abc import Mapping

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..ir import expr as E
from ..ir.domain import Domain
from ..ir.problem import EVP, IVP, LBVP, Problem
from ..errors import InputTypeError, OptionError
from ..reference.evaluate import FieldData
from ..tolerances import NCC_TAIL_TOL
from . import ultraspherical as us
from .compiler import DiscretizationError
from .curvilinear_symmetry import (CurvilinearReduction, plan_embed, plan_matrices, plan_project,
                                   reduced_lbvp_solve)
from .plan import GroupMatrices, LBVPSolution, Plan, data_snapshot, rebind_for_data
from .sphere_eval import SphereEvaluator, SphereSpace, comps, embed_coeffs, rhs_rpow
from .polar_radial import jacobi_values
from .sphere_ops import log_jacobi_norm, regularity_Q, spin_of
from .sphere_blocks import (  # noqa: F401  (re-exported: public face)
    _SQ2, SphereGroup, _Ent, _sum_blocks, _spin_first_order, ball_map, _relabel)

from .jacobi_family import gauss_legendre
from .sphere_compiler import (  # noqa: F401  (re-exported: public face)
    _SphereCompiler)

__all__ = ["SpherePlan", "SphereGroup"]


# ----------------------------------------------------------------------------------------------
# plan
# ----------------------------------------------------------------------------------------------

class SpherePlan:
    """Discretized problem on S2, a spherical shell or a ball (see module docstring)."""

    is_polar = False
    is_sphere = True

    def __init__(self, problem: Problem, ncc: Mapping[E.Field, Any] | None = None,
                 params: Mapping[str, Any] | None = None, ncc_cutoff: float | None = None,
                 assembly: str = "direct", ncc_tail_tol: float = NCC_TAIL_TOL) -> None:
        if assembly != "direct":
            raise OptionError(f"spherical plans assemble every group directly, got assembly={assembly!r}",
                              fix="assembly='direct' (the only assembly of curvilinear plans)")
        self.problem = problem
        self.params = dict(params or {})
        # constructor data kept to rebuild the plan when solve(data) changes a coefficient
        self._ncc_src = data_snapshot(ncc)
        self._plan_kw = dict(ncc_cutoff=ncc_cutoff, assembly=assembly, ncc_tail_tol=ncc_tail_tol)
        self.ncc_cutoff = us.check_cutoff(ncc_cutoff)
        # resolution of the non-constant coefficients (same semantics as the Cartesian plan:
        # a sampled tail above ncc_tail_tol, or the declared ncc_cutoff if larger, is refused)
        self.ncc_tail_tol = us.check_tail_tol(ncc_tail_tol)
        dom = None
        for v in problem.variables:
            dom = Domain.union(dom, v.domain)
        for eq in problem.equations:
            dom = Domain.union(dom, eq.domain)
        self.domain = dom
        self.space = SphereSpace(dom)
        S = self.space
        self.variables = problem.variables
        self.eigenvalue = problem.eigenvalue if isinstance(problem, EVP) else None
        if S.ball:
            for v in problem.variables:
                if v.rank > 1 and S.bulk(v.domain):
                    raise DiscretizationError(f"{v.name}: ball unknowns of rank >= 2 are not implemented")
        used = frozenset(f for eq in problem.equations for side in (eq.lhs, eq.rhs)
                         for f in side.free_fields())
        stray = [f.name for f in (ncc or {}) if f not in used]
        if stray:
            raise DiscretizationError(
                f"ncc given for field(s) not used by the problem: {', '.join(sorted(stray))}")
        # same rule for parameter values: an unknown key (typo) would be ignored and the value
        # declared on the Parameter used instead (as C7 for the polar plans)
        pnames = {p.name for eq in problem.equations for side in (eq.lhs, eq.rhs)
                  for p in side.free_parameters()}
        stray_p = sorted(set(self.params) - pnames)
        from .param_keys import check_param_values
        check_param_values(self.params, where="discretize")
        if stray_p:
            raise DiscretizationError(
                f"params given for parameter(s) not used by the problem: "
                f"{', '.join(repr(n) for n in stray_p)} (used: {sorted(pnames)})")
        from .param_keys import PlanParams     # later writes checked too (review, inputs)
        self.params = PlanParams(self.params, pnames, "plan.params")
        self._params_assembled: dict = {}   # the values of the first assembly (param_keys)
        self.evaluator = SphereEvaluator(S)
        self._ncc_coeffs = {f: self.coeffs_from_data(f, d) for f, d in (ncc or {}).items()}
        self._ncc_data = self._ncc_coeffs
        self.ncc_reports: dict = {}
        self._cache: dict = {}
        # angular dependence of a coefficient *of the left-hand side* couples the degrees l (3D: one
        # group per m then); coefficients used only on right-hand sides (e.g. explicit Coriolis)
        # are evaluated on the grid and do not change the grouping
        lhs_fields = frozenset(f for eq in problem.equations for f in eq.lhs.free_fields())
        self.coupled = S.s2 or any(self._angular(c) for f, c in self._ncc_coeffs.items() if f in lhs_fields)
        if S.ball and self.coupled and not S.s2:
            raise DiscretizationError("ball: coefficients depending on theta are not implemented "
                                      "(spherically symmetric scalar coefficients only)")
        if self.coupled:
            self.groups = tuple(SphereGroup(int(m), i, tuple(range(abs(int(m)), S.L + 1)))
                                for i, m in enumerate(S.ms))
        else:
            self.groups = tuple(SphereGroup(int(m), i, (ell,), True)
                                for i, m in enumerate(S.ms) for ell in range(abs(int(m)), S.L + 1))
        self.taus = frozenset(f for eq in problem.equations for n in eq.lhs.walk()
                              if isinstance(n, E.Lift) for f in n.free_fields() & frozenset(problem.variables))
        # shell: every bulk equation lives in Dedalus's derivative basis k_eq = max(k(lhs), k(rhs))
        # (sphere_eval.rhs_rpow): rows at level and r-power k_eq, right-hand side projected as
        # r^k_eq F from the dealiased grid
        self.rhs_rpow = [max(rhs_rpow(eq.lhs), rhs_rpow(eq.rhs)) if (S.shell and S.bulk(eq.domain)) else 0
                         for eq in problem.equations]
        self.eq_kind = []
        for eq in problem.equations:
            if S.bulk(eq.domain):
                self.eq_kind.append("bulk")
            elif any(isinstance(n, E.Interpolate) for n in eq.lhs.walk()):
                self.eq_kind.append("boundary")
            else:
                self.eq_kind.append("constraint")
        # symmetry subspace (EquatorialParity, AzimuthalMFold): orbits of the signed permutations
        # on the coefficients, curvilinear_symmetry
        self.subspace = CurvilinearReduction(self) if problem.symmetry is not None else None

    def _angular(self, c: np.ndarray) -> bool:
        """Does a coefficient array have content at l > 0 (3D coupling)?"""
        S = self.space
        if S.s2:
            return True
        a = np.asarray(c)
        sc = max(np.abs(a).max(), 1e-300)
        return a.shape[-2] > 1 and np.abs(a[..., 1:, :]).max() > 1e-14 * sc

    # ---------------------------------------------------------------- layouts
    def obj_layout(self, dom: Domain | None, comp: tuple, group: SphereGroup) -> list[tuple[int, int]]:
        """[(l, n_l)] blocks of an object's component in a group (validity: domain and spin)."""
        S = self.space
        bulk3 = (not S.s2) and S.has(dom)[2]
        return [(ell, S.rsize(ell) if bulk3 else 1) for ell in group.ells
                if S.valid(group.m, comp, ell, dom)]

    def size_of(self, dom: Domain | None, comp: tuple, group: SphereGroup) -> int:
        """Number of coefficients of component ``comp`` of an object on ``dom`` in ``group``."""
        return sum(n for _, n in self.obj_layout(dom, comp, group))

    def flat_index(self, dom: Domain | None, rank: int, comp: tuple, group: SphereGroup) -> tuple:
        """(flat indices into the object's coefficient array, sort key) of every entry of a
        component's group vector, in layout order."""
        S = self.space
        shape = S.coeff_shape(dom, rank)
        hp, ht, hr = S.has(dom)
        mi = group.index if hp else 0
        idx, key = [], []
        for ell, n in self.obj_layout(dom, comp, group):
            li = ell if ht else 0
            for j in range(n):
                pos = comp + ((mi, li) if S.s2 else (mi, li, j))
                idx.append(int(np.ravel_multi_index(pos, shape)))
                key.append(ell * (S.N + 8) + j)
        return np.array(idx, int), np.array(key, float)

    def group(self, m: int, ell: int | None = None) -> SphereGroup:
        """The group of azimuthal wavenumber m (and degree l for separable 3D problems)."""
        for g in self.groups:
            if g.m == m and (not g.single or ell is None or g.ells[0] == ell):
                if g.single and ell is None:
                    raise KeyError("separable 3D plan: give the degree l as well")
                return g
        raise KeyError(f"no group m={m}" + (f", l={ell}" if ell is not None else ""))

    def coeff_shape(self, dom: Domain | None, rank: int) -> tuple:
        """Coefficient array shape of a rank-``rank`` object on domain ``dom``."""
        return self.space.coeff_shape(dom, rank)

    def Q(self, m: int, ell: int) -> np.ndarray:
        """Regularity matrix Q of order ``m`` and degree ``ell`` (cached)."""
        key = ("Q", m, ell)
        if key not in self._cache:
            self._cache[key] = regularity_Q(self.space.H, m, ell)
        return self._cache[key]

    def ball_map(self, kind: str, m: int, ell: int) -> dict:
        """Ball index map of kind ``kind`` for order ``m`` and degree ``ell`` (cached)."""
        key = ("bmap", kind, m, ell)
        if key not in self._cache:
            self._cache[key] = ball_map(self.space.H, kind, m, ell)
        return self._cache[key]

    # ---------------------------------------------------------------- data
    def coeffs_from_data(self, f: E.Field, d: Any) -> np.ndarray:
        """Coefficients (level 0) of a field from data: a callable of the grids (phi, theta[, r])
        returning physical components (sampled on the dealiased grid, projected), or FieldData
        (layout "grid": physical values on the non-dealiased grid; "coeff": the plan's layout)."""
        S = self.space
        dealias = "data" if (callable(d) and not isinstance(d, FieldData)) else False
        shape = S.grid_shape(f.domain, f.rank, dealias)
        if isinstance(d, FieldData):
            if d.layout == "coeff":
                return np.broadcast_to(np.asarray(d.data), S.coeff_shape(f.domain, f.rank)).astype(complex)
            g = np.broadcast_to(np.asarray(d.data), shape)
        elif callable(d):
            grids = S.grids("data")
            hp, ht, hr = S.has(f.domain)
            nd = 2 if S.s2 else 3
            sh = lambda a, ax: a.reshape([-1 if i == ax else 1 for i in range(nd)])  # noqa: E731
            args = [sh(grids[0], 0) if hp else np.zeros((1,) * nd),
                    sh(grids[1], 1) if ht else np.full((1,) * nd, np.pi / 2)]
            if not S.s2:
                args.append(sh(grids[2], 2) if hr else np.full((1,) * nd, S.br.bounds[1]))
            g = np.broadcast_to(np.asarray(d(*args)), shape)
        else:
            raise InputTypeError(f"data for {f.name}: pass a FieldData (explicit layout) or a callable "
                                 "of the grids (phi, theta[, r]); bare arrays are ambiguous",
                                 fix="FieldData(array, layout='grid') or FieldData(array, layout='coeff')")
        if S.real and np.iscomplexobj(g) and np.max(np.abs(np.imag(g))) > 0:
            raise OptionError(f"data for {f.name}: complex values on a RealFourier (real-field) domain",
                              fix="pass the real field (np.real(...) if the imaginary part is "
                                  "rounding), or use a ComplexFourier azimuth")
        return S.forward(g, f.domain, f.rank, dealias)

    def to_grid(self, name: str, coeffs: np.ndarray, exact_state: bool = False) -> np.ndarray:
        """Physical values of a variable from its coefficients (a solver state): imaginary parts
        of real fields checked against space.state_real_tol (sphere_eval.STATE_REAL_TOL), or
        space.real_tol with ``exact_state`` (a projected solver state, verified exactly real)."""
        v = next(v for v in self.variables if v.name == name)
        # exact_state: m = 0 coefficients verified exactly real (projected solver state):
        # the transform bound space.real_tol applies instead of the state bound
        return self.space.backward(coeffs, v.domain, v.rank, False,
                                   real_tol=None if exact_state else self.space.state_real_tol)

    def ncc_data(self, f: E.Expr) -> dict:
        """{spin comp: {l_c: radial coefficients (T series on the shell, a number on S2)}} of an
        m = 0 coefficient expression; any phi dependence is refused; tails reported."""
        key = ("nccd", f)
        if key in self._cache:
            return self._cache[key]
        missing = [x for x in f.free_fields() if x not in self._ncc_coeffs]
        if missing:
            raise DiscretizationError(
                f"non-constant coefficient '{f}': no data for {', '.join(m.name for m in missing)} "
                "(pass ncc={field: data})")
        S = self.space
        c = self.evaluator.evaluate(f, self._ncc_coeffs, self.params)
        hp, ht, hr = S.has(f.domain)
        if not (ht or hr):
            raise DiscretizationError(f"coefficient '{f}' depends on neither theta nor r: use a constant")
        if S.ball and not hr:
            raise DiscretizationError(f"ball: coefficient '{f}' must depend on r (spherically symmetric "
                                      "scalar coefficients only)")
        c3 = c if not S.s2 else c[..., None]
        scale = max(np.abs(c3).max(), 1e-300)
        other = c3.copy()
        other[..., 0, :, :] = 0
        if np.abs(other).max() > 1e-12 * scale:
            raise DiscretizationError(
                f"non-constant coefficient '{f}' depends on phi (|m != 0 part| = {np.abs(other).max():.1e}): "
                "it couples azimuthal wavenumbers; not separable")
        out: dict = {}
        for comp in comps(f.rank, S.nc):
            v = (c3[comp] if comp else c3)[0]                     # (L+1|1, N|1)
            per_l = {}
            for lc in range(v.shape[0]):
                rad = v[lc]
                if np.abs(rad).max() <= 1e-14 * scale:
                    continue
                if np.max(np.abs(rad.imag)) <= 1e-14 * max(np.abs(rad).max(), 1e-300):
                    rad = rad.real
                if S.s2:
                    per_l[lc] = complex(rad[0]) if np.iscomplexobj(rad) else float(rad[0])
                elif not hr:
                    # constant in r: the radial series is the single exact coefficient T_0 (no tail)
                    per_l[lc] = np.asarray(rad[:1])
                else:
                    a, rep = us._ncc_coeffs(rad, f"{f}{list(comp)}[l={lc}]", self.ncc_cutoff, tail_tol=self.ncc_tail_tol, axis="r")
                    self.ncc_reports[(str(f), comp, lc)] = rep
                    per_l[lc] = a
            if per_l:
                out[comp] = per_l
        self._cache[key] = out
        return out

    def surface_tensor(self, f: E.Expr) -> np.ndarray:
        """Spin components (3 x 3, constants) of a rank-2 coefficient given on the sphere (no r):
        only rotation-invariant tensors are accepted (m = 0, l = 0 content in the spin-0
        components: e_r e_r, I - e_r e_r, ...), so a product with a boundary trace stays diagonal
        in (m, l). Used by the curved free-slip condition (model/conditions.py)."""
        key = ("surf", f)
        if key in self._cache:
            return self._cache[key]
        missing = [x for x in f.free_fields() if x not in self._ncc_coeffs]
        if missing:
            raise DiscretizationError(
                f"boundary coefficient '{f}': no data for {', '.join(m.name for m in missing)} "
                "(pass ncc={field: data})")
        S = self.space
        if f.rank != 2 or S.s2 or S.has(f.domain)[2]:
            raise DiscretizationError(f"boundary coefficient '{f}': a rank-2 tensor without radial "
                                      "dependence is expected")
        c = np.asarray(self.evaluator.evaluate(f, self._ncc_coeffs, self.params))
        v = c.reshape(c.shape[:2] + (-1,))
        scale = max(np.abs(v).max(), 1e-300)
        if np.abs(v[..., 1:]).max(initial=0.0) > 1e-12 * scale:
            raise DiscretizationError(
                f"boundary coefficient '{f}' is not rotation invariant (content at m != 0 or l > 0): "
                "its product with a boundary trace would couple degrees")
        out = v[..., 0] / _SQ2                       # Phi_0^{0,0} = 1/sqrt(2) (unit sphere)
        self._cache[key] = out
        return out

    def ball_ncc_legendre(self, f: E.Expr) -> np.ndarray:
        """Legendre coefficients (in z = 2 rho^2 - 1) of a spherically symmetric scalar coefficient on
        the ball (exact projection on Gauss-Legendre nodes; tail reported)."""
        key = ("bleg", f)
        if key in self._cache:
            return self._cache[key]
        S = self.space
        c = self.evaluator.evaluate(f, self._ncc_coeffs, self.params)
        c0 = c.reshape(c.shape[-3:])[0, 0, :S.N]                  # m = 0, l = 0 radial coefficients
        nq = 2 * S.N + 4
        z, w = gauss_legendre(nq)
        s = np.exp([log_jacobi_norm(0, 0.5, n) for n in range(S.N)])
        vals = (jacobi_values(S.N, 0, 0.5, z) / s) @ c0 / _SQ2      # g(z) = c_0(r) Phi_0^{0,0}
        if np.max(np.abs(vals.imag)) <= 1e-14 * max(np.abs(vals).max(), 1e-300):
            vals = vals.real
        deg = S.N
        Pl = jacobi_values(deg, 0, 0, z)
        leg = ((2 * np.arange(deg) + 1) / 2) * (Pl.T @ (w * vals))
        leg, rep = us._ncc_coeffs(leg, str(f), self.ncc_cutoff, tail_tol=self.ncc_tail_tol, axis="r")
        self.ncc_reports[(str(f), (), 0)] = rep
        self._cache[key] = leg
        return leg

    # ---------------------------------------------------------------- group layouts
    def var_slots(self, group: SphereGroup) -> list[tuple]:
        """(variable index, component, offset, size) of every variable component in the group's
        columns."""
        out, off = [], 0
        for vi, v in enumerate(self.variables):
            for c in comps(v.rank, self.space.nc):
                n = self.size_of(v.domain, c, group)
                out.append((vi, c, off, n))
                off += n
        return out

    def eq_slots(self, group: SphereGroup) -> list[tuple]:
        """(equation index, component, offset, size) of every equation component in the group's
        rows."""
        out, off = [], 0
        for i, eq in enumerate(self.problem.equations):
            for c in comps(eq.lhs.rank, self.space.nc):
                n = self.size_of(eq.domain, c, group)
                out.append((i, c, off, n))
                off += n
        return out

    # ---------------------------------------------------------------- assembly
    def _assemble(self, group: SphereGroup) -> GroupMatrices:
        from .param_keys import check_assembled_params
        check_assembled_params(self)          # cached groups hold the first values
        key = ("full", group)
        if key in self._cache:
            return self._cache[key]
        S = self.space
        vs, es = self.var_slots(group), self.eq_slots(group)
        vpos = {(self.variables[vi], c): (off, n) for vi, c, off, n in vs}
        nrow, ncol = sum(s[3] for s in es), sum(s[3] for s in vs)
        comp = _SphereCompiler(self, group)
        pieces: dict = {"L": [], "M": []}
        rows: list = []
        info: dict = {}
        forms: dict = {}
        for i, c, r0, nr in es:
            eq = self.problem.equations[i]
            if i not in forms:
                fL = comp.compile(eq.lhs, "L")
                fM = comp.compile(eq.lhs, "M") if isinstance(self.problem, (IVP, EVP)) else {}
                if (isinstance(self.problem, IVP) and any(isinstance(n, E.TimeDerivative) for n in eq.lhs.walk())
                        and not any(e.blocks for e in fM.values())):
                    raise DiscretizationError(f"equation [{i}] contains a time derivative but its "
                                              "mass operator M is empty: refusing")
                forms[i] = (fL, fM)
            fL, fM = forms[i]
            rows += [(i, c, self.eq_kind[i])] * nr
            if nr == 0:
                continue
            bulk = S.bulk(eq.domain)
            ents = [f[c] for f in (fL, fM) if c in f]
            level = p = None
            if bulk:
                level = max([e.level for e in ents if e.bulk] + [self.rhs_rpow[i]])
                p = max([e.p for e in ents if e.bulk and e.p is not None] + [self.rhs_rpow[i]])
            # rows of the equation: entry layout restricted to the equation's domain
            elay = comp.lay(c, bulk)
            olay = dict(self.obj_layout(eq.domain, c, group))
            sel, off = [], 0
            for ell, n in elay:
                if ell in olay:
                    sel.extend(range(off, off + n))
                off += n
            info[(i, c)] = (r0, nr, level, p)
            for f, target in ((fL, "L"), (fM, "M")):
                if c not in f:
                    continue
                e = f[c]
                if bulk:
                    e = comp.normalize(e, c, level, p)
                elif e.bulk:
                    raise DiscretizationError(f"equation [{i}] lives on a surface/constant domain but "
                                              "its left-hand side depends on the innermost coordinate "
                                              "(missing interp?)")
                for (v, vc), blk in e.blocks.items():
                    c0, nv = vpos[(v, vc)]
                    blk = blk.tocsr()[sel]
                    if blk.shape != (nr, nv):
                        raise DiscretizationError(
                            f"internal shape mismatch in equation [{i}]{list(c)} for {v.name}{list(vc)}"
                            f" at {group.label}: {blk.shape} vs {(nr, nv)}")
                    if blk.nnz:
                        pieces[target].append((r0, c0, blk.tocoo()))

        def build(ps):
            if not ps:
                return sp.csr_matrix((nrow, ncol), dtype=complex)
            r = np.concatenate([b.row + r0 for r0, _, b in ps])
            cc = np.concatenate([b.col + c0 for _, c0, b in ps])
            d = np.concatenate([b.data.astype(complex) for _, _, b in ps])
            A = sp.coo_matrix((d, (r, cc)), shape=(nrow, ncol)).tocsr()
            A.sum_duplicates()
            return A
        L, M = build(pieces["L"]), build(pieces["M"])
        cols = []
        for vi, c, _off, n in vs:
            v = self.variables[vi]
            kd = "tau" if v in self.taus else ("field" if S.bulk(v.domain) else "constant")
            cols += [(v.name, c, kd)] * n
        lams = {k: val[2] for k, val in info.items()}
        gm = GroupMatrices(group, group.label, L, M, rows, cols, np.arange(nrow), np.arange(ncol), lams)
        gm.row_info = info
        self._cache[key] = gm
        return gm

    def matrices(self, group: SphereGroup | None = None, reduced: bool | None = None) -> GroupMatrices:
        """Public L, M, boundary rows and tau columns of one group (default: the first); in the
        declared symmetry subspace by default when the problem has one (``reduced``)."""
        return plan_matrices(self, group, reduced)

    def embed(self, x_red: np.ndarray, group: SphereGroup | None = None) -> np.ndarray:
        """E: symmetry-subspace group vector -> full group vector."""
        return plan_embed(self, x_red, group)

    def project(self, x_full: np.ndarray, group: SphereGroup | None = None) -> np.ndarray:
        """P: full group vector -> subspace coefficients (P E = I)."""
        return plan_project(self, x_full, group)

    rank_report = Plan.rank_report
    check = Plan.check

    # ---------------------------------------------------------------- right-hand sides
    def rhs_map(self, group: SphereGroup, i: int) -> sp.csr_matrix:
        """Sparse map from equation i's right-hand-side coefficient array (level 0, flattened) to
        the group's rows: conversion to the row level, r^p multiplication on the shell, identity
        for point rows."""
        key = ("rhs", group, i)
        if key in self._cache:
            return self._cache[key]
        gm = self._assemble(group)
        S = self.space
        eq = self.problem.equations[i]
        ncoef = int(np.prod(S.coeff_shape(eq.domain, eq.lhs.rank)))
        rr, cc, vv = [], [], []
        for (ii, c), (r0, nr, level, p) in gm.row_info.items():
            if ii != i or nr == 0:
                continue
            lay = self.obj_layout(eq.domain, c, group)
            if S.ball and level is None and len(c) == 1:
                # regularity-labelled surface rows, spin right-hand side: row sigma = Q^H[sigma, s] b_s
                hp, ht, _ = S.has(eq.domain)
                shape = S.coeff_shape(eq.domain, 1)
                off = 0
                for ell, n in lay:
                    Qm = self.Q(group.m, ell)
                    for sidx in range(3):
                        w = np.conj(Qm[sidx, c[0]])
                        if w == 0:
                            continue
                        pos = (sidx, group.index if hp else 0, ell if ht else 0, 0)
                        rr.append(np.array([r0 + off]))
                        cc.append(np.array([np.ravel_multi_index(pos, shape)]))
                        vv.append(np.array([w], dtype=complex))
                    off += n
                continue
            flat, _ = self.flat_index(eq.domain, eq.lhs.rank, c, group)
            off = 0
            for ell, n in lay:
                if level is None or S.s2:
                    T = sp.identity(n, format="coo")
                elif S.ball:
                    T = self.space.ops.convert(ell + spin_of(c), 0, level, n).tocoo()
                else:        # coefficients of r^k F (k = rhs_rpow) -> C^(level), times r^(p - k)
                    T = (S.ops.multiply_r(level, p - self.rhs_rpow[i], n) @ S.ops.convert(0, level, n)).tocoo()
                rr.append(r0 + off + T.row)
                cc.append(flat[off + T.col])
                vv.append(T.data.astype(complex))
                off += n
        if rr:
            A = sp.csr_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                              shape=(gm.L.shape[0], ncoef))
        else:
            A = sp.csr_matrix((gm.L.shape[0], ncoef), dtype=complex)
        self._cache[key] = A
        return A

    def rhs_coeffs(self, coeffs: Mapping[E.Field, np.ndarray]) -> list:
        """Coefficients of every right-hand side evaluated with the field data ``coeffs`` and
        the NCCs (None for a zero right-hand side)."""
        data = dict(self._ncc_coeffs)
        data.update(coeffs)
        S = self.space
        return [None if eq.rhs.is_zero() else embed_coeffs(
                    self.evaluator.evaluate(eq.rhs, data, self.params, k),
                    S.coeff_shape(eq.domain, eq.lhs.rank), f"right-hand side of equation {i}")
                for i, (eq, k) in enumerate(zip(self.problem.equations, self.rhs_rpow))]

    def group_rhs(self, group: SphereGroup, rhs: list) -> np.ndarray:
        """Right-hand side vector of ``group`` from the per-equation coefficient arrays ``rhs``."""
        b = np.zeros(self._assemble(group).L.shape[0], dtype=complex)
        for i, r in enumerate(rhs):
            if r is not None:
                b += self.rhs_map(group, i) @ np.asarray(r).reshape(-1)
        return b

    def solve(self, data: Mapping[E.Field, Any] | None = None, reduced: bool | None = None) -> LBVPSolution:
        """LBVP: one sparse LU per group, relative residual of every solve reported."""
        if not isinstance(self.problem, LBVP):
            raise DiscretizationError("solve() is for LBVPs (EVP: solvers.eigenvalues, IVP: IVPSolver)")
        fresh = rebind_for_data(self, data)       # a new operator coefficient: rebuilt plan
        if fresh is not None:
            return fresh.solve(data, reduced)
        if reduced is None:
            reduced = self.problem.symmetry is not None
        self.check(reduced)
        coeffs = {f: self.coeffs_from_data(f, d) for f, d in (data or {}).items()}
        rhs = self.rhs_coeffs(coeffs)
        out = {v.name: np.zeros(self.space.coeff_shape(v.domain, v.rank), dtype=complex)
               for v in self.variables}
        if reduced:
            return LBVPSolution(out, reduced_lbvp_solve(self, rhs, out), self, True)
        residuals = []
        for g in self.groups:
            gm = self._assemble(g)
            if gm.L.shape[0] == 0:
                continue
            b = self.group_rhs(g, rhs)
            A = gm.L.tocsc()
            x = spla.splu(A).solve(b)
            residuals.append((gm.label, float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300))))
            self.scatter_group(g, x, out)
        return LBVPSolution(out, residuals, self, False)

    def scatter_group(self, group: SphereGroup, x: np.ndarray, out: dict) -> None:
        """Write the group solution ``x`` into the per-variable coefficient arrays of ``out``."""
        for vi, c, off, n in self.var_slots(group):
            v = self.variables[vi]
            if n == 0:
                continue
            flat, _ = self.flat_index(v.domain, v.rank, c, group)
            out[v.name].reshape(-1)[flat] = x[off:off + n]

    # ---------------------------------------------------------------- explain / methods
    def _repr_html_(self) -> str:       # Jupyter display (conatus.notebook)
        from ..notebook import explain_html; return explain_html(self)  # noqa: E702

    def explain(self, group: SphereGroup | None = None) -> str:
        """Bases, groups, equation spaces and rank report of ``group`` (default the first one)."""
        S = self.space
        g0 = self.groups[0] if group is None else group
        gm = self._assemble(g0)
        geo = "S2" if S.s2 else ("ball" if S.ball else "shell")
        lines = [f"Discretization plan for {self.problem.kind} '{self.problem.name}' ({geo})",
                 f"  domain: {self.domain!r}",
                 f"  {S.phi.name}: {type(S.bphi).__name__}({S.bphi.size}); {S.theta.name}: "
                 f"{S.H.explain()}",
                 f"  groups: {len(self.groups)} "
                 + ("(one per m, all degrees l)" if self.coupled else "(one per (m, l), separable)")
                 + (" ; real fields: m >= 0 stored, f_-m = conj(f_m) with spins flipped" if S.real else "")]
        if not S.s2:
            lines.append(f"  {S.r.name}: {S.ops.explain()}")
        lines.append("  tensors: spin components (u_-, u_+" + ("" if S.s2 else ", u_r")
                     + "), u_+- = (u_theta +- i u_phi)/sqrt(2)" + ("; ball bulk: regularity components "
                                                                     "(l-1, l+1, l)" if S.ball else "")
                     + "; nothing divided by r or sin(theta)")
        if S.shell:
            lines.append("  shell rows: multiplied by r^p (p = largest r-power of the row's terms, "
                         "declared); right-hand sides multiplied identically; lifted taus are not")
        lines.append(f"  group [{gm.label}]: {gm.shape[0]} x {gm.shape[1]}, nnz(L) = {gm.L.nnz}, "
                     f"nnz(M) = {gm.M.nnz}")
        for (i, c), (_r0, nr, level, p) in sorted(gm.row_info.items(), key=lambda t: (t[0][0], t[0][1])):
            space = ("point rows" if level is None else
                     (f"level {level}" + (f", x r^{p}" if S.shell else "")))
            lines.append(f"  equation [{i}]{list(c) if c else ''}: {self.eq_kind[i]}, {nr} rows, {space}")
        lines.append("  NCC resolution: " + (
            us.ncc_policy(self.ncc_tail_tol, self.ncc_cutoff)))
        if self.ncc_reports:
            lines.append("  non-constant coefficients (axisymmetric):")
            for rep in self.ncc_reports.values():
                lines.append("    " + rep.explain())
        Ms = S.sizes(True)
        lines.append(f"  nonlinear terms: dealiased grid {' x '.join(str(v) for v in Ms)} = floor(N * dealias); "
                     "angular derivatives diagonal in l, radial derivatives on the nodes, grid products, "
                     "projection back (Gauss quadratures)")
        if S.real:
            lines.append(f"  real fields: imaginary parts on the grid refused above {S.real_tol:g} x max|value| "
                         "(plan.space.real_tol, declared; never dropped silently); a pointwise function "
                         f"in the RHS (T**2, exp) acts on Re of its argument, refused above "
                         f"{S.func_real_tol:g} relative (plan.space.func_real_tol: the non-real m = 0 part "
                         "of a state, solve rounding, grows relative to a decaying field); state outputs "
                         f"(to_grid) refuse above {S.state_real_tol:g} relative")
        if self.subspace is not None:          # declared symmetry subspace: reduction and warning
            lines.append("  " + self.subspace.explain())
            lines += ["  " + ln for ln in self.problem.symmetry.explain().splitlines()]
        lines.append(Plan.rank_report(self, g0).explain())
        return "\n".join(lines)

    def methods(self, solver: Any = None, **kw):
        """Numerical-methods summary (conatus.methods) of this plan."""
        from ..methods import methods
        return methods(self, solver, **kw)
