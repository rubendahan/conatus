"""IVPs on polar domains: stacked azimuthal groups, batched banded solves, polar F.

``IVPSolver(plan, ...)`` with a ``PolarPlan`` returns a ``PolarIVPSolver`` (dispatch installed at
the end of this module, imported by ``conatus.timestep``): the time-stepping arithmetic (SBDF,
IMEX RK, start-up, checkpoints) is the Cartesian ``IVPSolver``'s, unchanged; only the global system
(groups m, spin components) and the right-hand side (``PolarEvaluator``: spectral derivatives,
products on the dealiased grid) are polar. CPU only for now (declared; GPU is a later step).
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import structural_rank

from ..discretize.plan import RankError
from ..discretize.polar_plan import PolarPlan
from ..ir import expr as E
from ..runtime import BorderedBandedBatch
from .._deprecation import renames
from ..errors import InputTypeError, OptionError, SymmetryError, UnsupportedError, did_you_mean
from .integrator import IVPSolver, check_dt_backend, check_substeps, finite_cadence, setup_scheme
from .real_projection import m0_projection, m0_projection_text
from .system import ORDERINGS, FactorCache, GlobalSystem, GroupSlot

__all__ = ["PolarGlobalSystem", "PolarIVPSolver"]


FACTOR_COND_MAX = 1e13
"""Declared bound on the condition estimate ||A|| ||x|| / ||b|| (infinity norms, random b) of every
factorized group system a M + b L, measured once per factorization: the structural
rank check does not see a numerically singular group (e.g. a formulation singular at m = 0 or
l = 0), whose solves are then garbage along the null space, silently. A backward-stable solve of
a singular system has a small backward error but an enormous solution: the estimate sees it."""
# backward error allowed for a column-border factorization (checked once per (a, b))
COLBORDER_BERR_MAX = 1e-12


def _check_conditioning(batch, mats, labels, key) -> None:
    """Solve every system of a bucket once with a random right-hand side; refuse the factorization
    if the condition estimate exceeds FACTOR_COND_MAX (singular or numerically singular group)."""
    n = mats[0].shape[0]
    rng = np.random.default_rng(12345)
    b = rng.standard_normal((len(mats), n))
    if np.dtype(batch.dtype).kind == "c":
        b = b + 1j * rng.standard_normal((len(mats), n))
    x = batch.solve(b.astype(batch.dtype))
    colb = getattr(batch, "col_border", False)
    for k, A in enumerate(mats):
        nA = float(abs(A).sum(axis=1).max())
        est = nA * float(np.abs(x[k]).max()) / float(np.abs(b[k]).max())
        if colb:
            # column border (timestep/colborder.py): its rows were chosen on one group at one
            # pencil; the backward error of every group at this (a, b) is checked here
            berr = float(np.abs(A @ x[k] - b[k]).max()) / (nA * float(np.abs(x[k]).max())
                                                           + float(np.abs(b[k]).max()))
            if not berr <= COLBORDER_BERR_MAX:
                raise RankError(
                    f"[{labels[k]}] the column-border layout of this group is inaccurate at "
                    f"(a, b) = {key}: backward error {berr:.1e} > {COLBORDER_BERR_MAX:g}",
                    fix="pass border='rows' to the solver (boundary rows only, the layout before "
                        "the column border; slower when the group has dense tau columns)")
        if not est <= FACTOR_COND_MAX:
            raise RankError(
                f"[{labels[k]}] the IVP system a M + b L (a, b = {key}) is numerically singular: "
                f"condition estimate {est:.1e} > {FACTOR_COND_MAX:g} (structural rank was full). Check "
                "plan.rank_report(group): a missing gauge condition, or a tau formulation singular in "
                "this group (second-order form in an annulus/shell at m = 0 / l = 0: use the "
                "first-order form)")


class PolarGlobalSystem:
    """All azimuthal groups of a polar IVP stacked in one vector X; gather/scatter and right-hand
    side maps are precomputed sparse/integer maps (no Python loop over groups at run time)."""

    def __init__(self, plan: PolarPlan, ordering: str = "interleaved", reduced: bool = False) -> None:
        if ordering not in ORDERINGS:
            raise OptionError(f"ordering must be one of {ORDERINGS}, got {ordering!r}",
                              fix="ordering='interleaved' (default)" + did_you_mean(str(ordering), ORDERINGS),
                              doc="docs/site/guide/timesteppers.md")
        # a switch, never the truth value of another object (review of audit inputs:
        # bool(reduced) read reduced="no" as True on the polar and spherical systems)
        from ..errors import check_flag
        reduced = check_flag(reduced, "reduced", "PolarGlobalSystem")
        if reduced and plan.subspace is None:
            raise OptionError("reduced=True needs a symmetry subspace declared on the problem",
                              fix="declare symmetry=SymmetrySubspace(...) on the problem, or drop "
                                  "reduced=", doc="docs/site/guide/timesteppers.md")
        self.plan, self.ordering, self.reduced, self.complexify = plan, ordering, reduced, False
        S = plan.space
        pb = plan.problem
        self.variables = plan.variables
        self.var_shapes = [S.coeff_shape(v.domain, v.rank) for v in self.variables]
        self.eq_shapes = [S.coeff_shape(eq.domain, eq.lhs.rank) for eq in pb.equations]
        self.slots: list[GroupSlot] = []
        cols_obj, cols_flat = [], []
        rmap = [([], [], []) for _ in pb.equations]             # per equation: (row, flat, weight)
        ops, off = S.ops, 0
        for g in plan.groups:
            gm = plan.matrices(g, self.reduced)
            n = gm.L.shape[0]
            if n == 0:
                continue
            if n != gm.L.shape[1]:
                raise RankError(f"[{gm.label}] non-square group system {gm.L.shape}")
            cm = self._meta(plan.var_slots(g), [v.domain for v in self.variables], self.var_shapes, g)
            rm = self._meta(plan.eq_slots(g), [eq.domain for eq in pb.equations], self.eq_shapes, g)
            # symmetry subspace: keep the representative entries (gm.col_index / row_index of the
            # full group); full-group row r -> reduced row rloc[r] (-1: not a representative)
            cm = [cm[j] for j in gm.col_index]
            rm = [rm[i] for i in gm.row_index]
            rloc = np.full(sum(sl[3] for sl in plan.eq_slots(g)), -1, int)
            rloc[gm.row_index] = np.arange(n)
            for k, fl, _ in cm:
                cols_obj.append(k)
                cols_flat.append(fl)
            kinds = [r[2] for r in gm.rows]
            rperm, cperm, nb = GlobalSystem._ordering(self, gm, cm, rm, kinds)
            self.slots.append(GroupSlot(g, gm.label, off, n, gm.M.tocsr(), gm.L.tocsr(), kinds,
                                        rperm, cperm, nb))
            # right-hand side map: level-0 coefficients -> converted, r^p-multiplied rows
            for (i, c), (r0, nr, k, level, p) in gm.row_info.items():
                idx0 = S.cindex(pb.equations[i].domain, g, 0)
                rows, flats, wts = rmap[i]
                if level is None:
                    if rloc[r0] >= 0:
                        rows.append(off + rloc[r0])
                        flats.append(int(np.ravel_multi_index(c + idx0, self.eq_shapes[i])))
                        wts.append(1.0)
                    continue
                T = (ops.multiply_r(k, level, p, nr) @ ops.convert(k, 0, level, nr)).tocoo()
                base = int(np.ravel_multi_index(c + idx0, self.eq_shapes[i]))
                # radial index j sits at stride (Nz on a cylinder, 1 on a disk/annulus)
                stride = self.eq_shapes[i][-1] if S.cyl else 1
                keep = rloc[r0 + T.row] >= 0
                rows.extend(off + rloc[r0 + T.row[keep]])
                flats.extend(base + T.col[keep] * stride)
                wts.extend(T.data[keep])
            off += n
        self.n = off
        self.complex, self.dtype = True, np.complex128
        self.cols_obj, self.cols_flat = np.array(cols_obj, int), np.array(cols_flat, int)
        self._col_sel = [np.nonzero(self.cols_obj == k)[0] for k in range(len(self.variables))]
        # symmetry subspace: every member of a kept orbit (possibly in a dropped group) is
        # +- its representative; scatter writes them, gather_state checks them
        self.scatter_extra = (plan.subspace.scatter_maps(self.var_shapes) if self.reduced
                              else [None] * len(self.variables))
        self.M = sp.block_diag([s.M for s in self.slots], format="csr").astype(self.dtype)
        self.L = sp.block_diag([s.L for s in self.slots], format="csr").astype(self.dtype)
        self._rhs_maps = [sp.csr_matrix((w, (r, f)), shape=(self.n, int(np.prod(shape))), dtype=complex)
                          for (r, f, w), shape in zip(rmap, self.eq_shapes, strict=True)]
        for s in self.slots:
            pat = sp.csr_matrix((abs(s.M) + abs(s.L)) > 0, dtype=float)
            if structural_rank(pat) != s.size:
                raise RankError(f"[{s.label}] structurally singular IVP pencil M + dt L: refused")
        buckets: dict = {}
        for i, s in enumerate(self.slots):
            pat = sp.csr_matrix((abs(s.M) + abs(s.L)) > 0, dtype=np.int8)[s.rperm][:, s.cperm]
            buckets.setdefault((s.size, s.n_border, pat.indptr.tobytes(), pat.indices.tobytes()), []).append(i)
        self._buckets = []
        for idx in buckets.values():
            sl = [self.slots[i] for i in idx]
            self._buckets.append((idx, np.stack([s.offset + s.rperm for s in sl]),
                                  np.stack([s.offset + s.cperm for s in sl])))
        self._factors = FactorCache()      # bounded LRU (audit state)

    def _meta(self, slots, domains, shapes, g) -> list[tuple]:
        """(object index, flat index in its coefficient array, radial index or None) per entry."""
        S = self.plan.space
        out = []
        for oi, c, _off, n in slots:
            bulk = S.kind(domains[oi]) in ("full", "radial")
            for j in range(n):
                fl = int(np.ravel_multi_index(c + S.cindex(domains[oi], g, j), shapes[oi]))
                out.append((oi, fl, j if bulk else None))
        return out

    # ------------------------------------------------------------------ solves
    def factor(self, a: float, b: float) -> list:
        """Batched factorizations of a M + b L per solver bucket (cached by (a, b))."""
        key = (float(a), float(b))
        if self._factors.get(key) is None:       # get(): a hit is a use of the LRU store
            f = []
            for idx, _, _ in self._buckets:
                mats = [(a * self.slots[i].M + b * self.slots[i].L).tocsr()[self.slots[i].rperm][:, self.slots[i].cperm]
                        for i in idx]
                s0 = self.slots[idx[0]]
                batch = BorderedBandedBatch(mats, list(range(s0.n_border)), dtype=self.dtype,
                                            col_border=bool(s0.col_border))
                _check_conditioning(batch, mats, [self.slots[i].label for i in idx], key)
                f.append(batch)
            self._factors[key] = f
        return self._factors[key]

    def solve(self, a: float, b: float, rhs: np.ndarray) -> np.ndarray:
        """X with (a M + b L) X = rhs, all groups (C++ bordered banded, batched per bucket)."""
        x = np.empty(self.n, dtype=self.dtype)
        for (_idx, rows, cols), batch in zip(self._buckets, self.factor(a, b), strict=True):
            x[cols] = batch.solve(rhs[rows])
        return x

    def solve_transpose(self, a: float, b: float, rhs: np.ndarray) -> np.ndarray:
        """Solve (a M + b L)^T x = rhs with the cached factors (adjoint runs)."""
        x = np.empty(self.n, dtype=self.dtype)
        for (_idx, rows, cols), batch in zip(self._buckets, self.factor(a, b), strict=True):
            x[rows] = batch.solve(rhs[cols], transpose=True)
        return x

    @property
    def n_factorizations(self) -> int:
        """Number of distinct (a, b) factorizations held."""
        return len(self._factors)

    # ------------------------------------------------------------------ gather / scatter
    def scatter(self, X: np.ndarray, only: frozenset | None = None) -> list:
        """Global vector -> coefficient arrays per variable (``only``: names to fill, others None)."""
        out = []
        for k, shape in enumerate(self.var_shapes):
            if only is not None and self.variables[k].name not in only:
                out.append(None)
                continue
            a = np.zeros(int(np.prod(shape)), dtype=self.dtype)
            sel = self._col_sel[k]
            a[self.cols_flat[sel]] = X[sel]
            extra = self.scatter_extra[k] if getattr(self, "scatter_extra", None) else None
            if extra is not None:                        # mirror members of the kept orbits
                xi, fl, fac = extra
                a[fl] = fac * X[xi]
            out.append(a.reshape(shape))
        return out

    def gather_state(self, coeffs, check_subspace: bool = True) -> np.ndarray:
        """Coefficient arrays -> global vector. With ``check_subspace`` (default), data in slots
        the system does not represent (per-m radial truncation N_m) is refused instead of
        being dropped; the argument is never ignored. The polar forward transform
        leaves those slots exactly 0, so only hand-built coefficients can trip the check."""
        X = np.zeros(self.n, dtype=self.dtype)
        flat_in = []
        for k, a in enumerate(coeffs):
            a = np.asarray(a).reshape(-1)
            flat_in.append(a)
            sel = self._col_sel[k]
            X[sel] = a[self.cols_flat[sel]]
        if check_subspace:
            # what the system represents (per-m radial truncation; symmetry subspace: every
            # orbit member +- its representative) must reproduce the data
            for k, (a, back) in enumerate(zip(flat_in, self.scatter(X), strict=True)):
                lost = np.abs(a - back.reshape(-1))
                # relative to the data (critic review: max(1, |a|) silently dropped the
                # unrepresented part of data of amplitude below 1)
                if lost.size and lost.max() > 1e-12 * np.abs(a).max():
                    where = ("the symmetry subspace or the represented modes" if self.reduced
                             else "the represented modes (per-m radial truncation)")
                    raise SymmetryError(
                        f"initial data for {self.variables[k].name} has components outside "
                        f"{where} (max {lost.max():.2e})",
                        why="the solver runs in the declared subspace; a component outside it "
                            "would be dropped, which would change the initial state",
                        fix="give initial data with the declared symmetry, or solve in the full "
                            "space (reduced=False)", doc="docs/site/guide/timesteppers.md")
        return X

    def gather_rhs(self, eq_coeffs) -> np.ndarray:
        """Equation right-hand sides (level-0 spin coefficients) -> rows: one sparse product per
        equation (conversion to the row level and declared r^p multiplication included)."""
        b = np.zeros(self.n, dtype=self.dtype)
        for R, c in zip(self._rhs_maps, eq_coeffs, strict=True):
            if c is not None:
                b += R @ np.asarray(c).reshape(-1)
        return b

    def explain(self) -> str:
        """Groups, unknowns, ordering and solver buckets of the polar system."""
        from ..runtime import BorderedBanded
        lines = [f"PolarGlobalSystem: {len(self.slots)} azimuthal group(s), {self.n} unknowns, complex128, "
                 f"ordering={self.ordering}; {len(self._buckets)} solver bucket(s) (groups sharing one "
                 "sparsity pattern, batched C++ bordered banded solves)"]
        if self.reduced:
            lines.append("  " + self.plan.subspace.explain())
        for idx, _, _ in self._buckets[:4]:
            s = self.slots[idx[0]]
            bb = BorderedBanded((s.M + s.L).tocsr()[s.rperm][:, s.cperm], list(range(s.n_border)),
                                dtype=self.dtype, col_border=bool(s.col_border))
            lines.append(f"  [{s.label}] n={s.size}, border rows={s.n_border}"
                         + (f" (incl. {s.col_border} for the column border)" if s.col_border else "")
                         + f", band kl={bb.kl} ku={bb.ku}"
                         + (f" (+{len(idx) - 1} group(s) with the same pattern)" if len(idx) > 1 else ""))
        return "\n".join(lines)


class PolarIVPSolver(IVPSolver):
    """IVPSolver on a PolarPlan: same schemes and step arithmetic (inherited), polar system and F."""

    def __init__(self, plan: PolarPlan, scheme: str, dt: float, *, reduced: bool | None = None,
                 ordering: str = "interleaved", startup: str = "lower_order",
                 startup_substeps: int = 2, time_parameter: str | None = None,
                 backend: str = "cpu", f_backend: str = "evaluator", complexify: bool = False,
                 bc_time: str = "auto", project_m0: bool = True, check_finite: bool | int = True, **kw: Any) -> None:
        setup_scheme(self, scheme, startup, where=f"PolarIVPSolver(plan, {scheme!r}, ...)",
                     bc_time=bc_time)
        check_dt_backend(dt, backend, f"PolarIVPSolver(plan, {scheme!r}, ...)")
        self._finite_every = finite_cadence(check_finite, f"PolarIVPSolver(plan, {scheme!r}, ...)")
        if f_backend != "evaluator":
            raise UnsupportedError(f"polar IVPs: f_backend={f_backend!r}",
                                   fix="f_backend='evaluator' (the only evaluator of F on "
                                       "polar domains)")
        if complexify:
            raise UnsupportedError("polar IVPs: complexify=True", why="the groups are already "
                                   "complex", fix="drop complexify=")
        if reduced is None:
            reduced = plan.problem.symmetry is not None     # declared subspace: run in it (explain)
        from ..errors import check_flag      # reduced="no" read as True (audit inputs review)
        reduced = check_flag(reduced, "reduced", "PolarIVPSolver")
        if reduced and plan.problem.symmetry is None:
            raise OptionError("reduced=True needs a symmetry subspace declared on the problem",
                              fix="declare symmetry=SymmetrySubspace(...) on the problem, or drop "
                                  "reduced=")
        if kw:
            raise InputTypeError(f"unexpected arguments for a polar IVP: {sorted(kw)}",
                                 fix="arguments of PolarIVPSolver: scheme, dt, reduced, ordering, startup, "
                                     "startup_substeps, time_parameter, backend, f_backend, "
                                     "bc_time, project_m0, check_finite")
        self.plan = plan
        self._params0 = dict(plan.params)   # matrices built with these (check_params_unchanged)
        self.scheme, self.dt = scheme, float(dt)
        self.startup = startup
        self.startup_substeps = check_substeps(startup_substeps, "PolarIVPSolver")
        from .integrator import check_time_parameter    # a name the equations use (audit inputs)
        check_time_parameter(plan, time_parameter, "PolarIVPSolver")
        self.time_parameter = time_parameter
        self.backend, self.f_backend, self.xp = "cpu", "evaluator", np
        self.system = PolarGlobalSystem(plan, ordering=ordering, reduced=reduced)
        # m = 0 coefficients projected onto real fields after every solve of the step
        # (declared in explain(); project_m0=False keeps the unprojected step)
        self._m0proj = m0_projection(self.system, project_m0)
        # ShiftReflect(phi, z) (pipe travelling waves): real-linear, enforced by a declared
        # projection chained with the m = 0 one (timestep/symmetry_projection.py)
        from .symmetry_projection import shift_reflect_projection
        self._m0proj = shift_reflect_projection(self, self.system, self._m0proj)
        self.evaluator = plan.evaluator
        if reduced:
            # symmetry subspace: F evaluated on the azimuthal sector of an m0-fold class and / or
            # the half z period of an axial reflection (declared in explain())
            from ..discretize.curvilinear_symmetry import FoldSpec
            from ..discretize.polar_eval import PolarEvaluator
            self.evaluator = PolarEvaluator(plan.space, fold=FoldSpec(
                plan.problem, plan.space.sizes(True)[0], False, plan.space.zc if plan.space.cyl else None))
        self.X = np.zeros(self.system.n, dtype=self.system.dtype)
        self.t, self.iteration = 0.0, 0
        self._MXh: deque = deque()
        self._Fh: deque = deque()
        self.n_F = 0
        self._ncc = dict(plan._ncc_coeffs)
        pb = plan.problem
        self._rhs_exprs = [None if eq.rhs.is_zero() else eq.rhs for eq in pb.equations]
        self._rhs_vars = frozenset(f.name for e in self._rhs_exprs if e is not None for f in e.free_fields())
        self._fused, self._fused_note = None, ""
        if backend == "cuda":      # GPU mirror of the same host system (timestep/curvilinear_device.py)
            from .curvilinear_device import attach_device
            attach_device(self, self.system)

    def set_state(self, data: Mapping[E.Field, Any], t: float = 0.0) -> None:
        """Initial condition: callables of the grids (phi, r) returning physical components, or
        FieldData. Variables not given start at zero. Resets the multistep history."""
        coeffs = [self.plan.coeffs_from_data(v, data[v]) if v in data else np.zeros(shape, complex)
                  for v, shape in zip(self.system.variables, self.system.var_shapes, strict=True)]
        self.X = self.system.gather_state(coeffs)
        self._check_projected_subspace(self.X, "initial data")
        self.t, self.iteration = float(t), 0
        self._t_origin, self._it_origin = float(t), 0
        self._t_initial = float(t)                  # start time of the chain (checkpoints)
        self._reset_history()

    def _check_projected_subspace(self, X, what: str) -> None:
        """Refuse a state outside a projected (real-linear) symmetry subspace (ShiftReflect)."""
        sp_ = getattr(getattr(self, "_m0proj", None), "s", None)
        if sp_ is None:
            return
        d = sp_.distance(X)
        # same bound as the verification of the inferred map: max(1e-12, 100 x the transforms'
        # round-trip rounding), declared by explain() (the disk transforms round at ~1e-13 from
        # N_r = 32 on: a fixed 1e-12 left a factor 10 over the rounding of symmetric grid data)
        if d > sp_.tol:
            raise SymmetryError(
                f"{what} has components outside the {sp_.R!r} subspace (relative {d:.1e} > {sp_.tol:.1e})",
                why="the solver projects every step on the declared subspace; a component outside "
                    "it would be removed, which would change the initial state",
                fix="give data with the declared symmetry (e.g. solver.symmetrize_state(X)), or drop "
                    "the generator", doc="docs/site/tutorials/03_symmetries.md")

    def symmetry_leak(self, reset: bool = False) -> float:
        """Largest relative part removed by the projection onto a projected symmetry subspace
        (ShiftReflect) in one step since the last reset (0 without one); refused above
        ``symmetry_projection.LEAK_MAX`` (equivariant operators: rounding)."""
        sp_ = getattr(getattr(self, "_m0proj", None), "s", None)
        if sp_ is None:
            return 0.0
        from .symmetry_projection import LEAK_MAX
        out = sp_.leak(reset)
        if out > LEAK_MAX:
            raise SymmetryError(
                f"{sp_.R!r}: the step left the subspace by {out:.1e} > {LEAK_MAX:g} (relative)",
                why="the operators, forcing or data of the problem are not equivariant under the "
                    "declared symmetry: projecting them back every step would change the equations",
                fix="check that every term (forcing, NCC, boundary data) has the declared symmetry and "
                    "parities, or drop the ShiftReflect generator (full space)",
                doc="docs/site/tutorials/03_symmetries.md")
        return out

    @renames("conatus.timestep.polar:PolarIVPSolver.run")
    def run(self, steps: int, *args, **kwargs):
        """``IVPSolver.run``, then the leak of a projected symmetry subspace (ShiftReflect) is
        checked: a step that left the subspace by more than ``LEAK_MAX`` (operators or forcing not
        equivariant) is refused instead of being projected in silence. One
        device-to-host read per call (no synchronisation per step)."""
        out = super().run(steps, *args, **kwargs)
        self.symmetry_leak()
        return out

    @renames("conatus.timestep.polar:PolarIVPSolver.run_graph")
    def run_graph(self, steps: int) -> None:
        """``IVPSolver.run_graph``, then the leak check of :meth:`run`."""
        super().run_graph(steps)
        self.symmetry_leak()

    def symmetrize_state(self, X):
        """The declared projections of the step applied to a copy of state X (m = 0 real part,
        projected symmetry subspace): a tool to build symmetric data explicitly."""
        Y = X.copy()
        pr = getattr(self, "_m0proj", None)
        if pr is None:
            return Y
        return pr.apply(Y, False) if hasattr(pr, "s") else pr.apply(Y)

    def coeffs(self, name: str) -> np.ndarray:
        """Current coefficients of a variable."""
        k = [v.name for v in self.system.variables].index(name)
        c = self.system.scatter(self.X)[k]
        return c if self.backend == "cpu" else self.xp.asnumpy(c)      # host copy (GPU backend)

    def grid(self, name: str) -> np.ndarray:
        """Current grid values of a variable (non-dealiased grid). Imaginary parts of real fields
        are refused above space.real_tol when the m = 0 coefficients of the state are exactly real
        (projected step: every state after a step), above space.state_real_tol otherwise
        (project_m0=False, a state set by hand or read from an unprojected checkpoint)."""
        pr = getattr(self, "_m0proj", None)
        exact = bool(getattr(pr, "active", False)) and pr.non_real(self.X) == 0.0
        return self.plan.to_grid(name, self.coeffs(name), exact_state=exact)

    def F(self, X: np.ndarray, t: float) -> np.ndarray:
        """Explicit right-hand side F(X, t) (polar pseudospectral evaluator), as row vector."""
        self.n_F += 1
        data = dict(self._ncc)
        data.update((v, c) for v, c in zip(self.system.variables,
                                           self.system.scatter(X, only=self._rhs_vars), strict=True) if c is not None)
        params = dict(self.plan.params)
        if self.time_parameter:
            params[self.time_parameter] = t
        S, eqs = self.plan.space, self.plan.problem.equations
        out = [None if e is None else S.embed(self.evaluator.evaluate(e, data, params), eq.domain,
                                              eq.lhs.rank, f"right-hand side of equation {k}")
               for k, (e, eq) in enumerate(zip(self._rhs_exprs, eqs, strict=True))]
        return self.system.gather_rhs(out)

    def _data_rhs_eq(self, k: int, params: dict, e=None):
        """State-independent right-hand side of equation k (time-dependent boundary data;
        ``e``: its time derivative in its place)."""
        eq = self.plan.problem.equations[k]
        e = self._rhs_exprs[k] if e is None else e
        return self.plan.space.embed(self.evaluator.evaluate(e, dict(self._ncc), params),
                                     eq.domain, eq.lhs.rank, f"right-hand side of equation {k}")

    def _short_notes(self) -> list[str]:
        """The m = 0 real projection changes the solution: kept in the short explain()."""
        return super()._short_notes() + [m0_projection_text(self).strip()]

    def _explain_full(self) -> str:
        st = self._startup_text()
        S = self.plan.space
        grid = " x ".join(str(v) for v in S.sizes(True))      # (Mphi, Mr[, Mz]): cylinders too
        gpu = self.backend == "cuda"
        return (f"IVPSolver[polar](scheme={self.scheme}, dt={self.dt:g}, constant step{st}, backend={self.backend})\n"
                + self.system.explain()
                + f"\n  F: polar pseudospectral {'CuPy (GPU: cuFFT, batched cuBLAS)' if gpu else 'NumPy'} "
                  f"evaluator (spectral spin derivatives, products on the dealiased {grid} grid, projection "
                  f"back" + (f"; {self.evaluator.fold.explain()}" if getattr(self.evaluator, "fold", None) else "")
                + f"); linear solves: {'GPU' if gpu else 'C++'} bordered banded"
                + self._bc_text() + m0_projection_text(self) + "\n" + self._finite_text())


def _dispatch_new(cls, plan=None, *args, **kwargs):
    """``IVPSolver(polar_plan, ...)`` builds a PolarIVPSolver (same public API)."""
    if cls is IVPSolver and getattr(plan, "is_polar", False):
        return object.__new__(PolarIVPSolver)
    return object.__new__(cls)


IVPSolver.__new__ = _dispatch_new
