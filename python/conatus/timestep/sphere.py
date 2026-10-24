"""IVPs on S2, spherical shells and balls: stacked groups, batched banded solves, spherical F.

``IVPSolver(plan, ...)`` with a ``SpherePlan`` returns a ``SphereIVPSolver`` (dispatch installed at the
end of this module, imported by ``conatus.timestep``): the time-stepping arithmetic (SBDF, IMEX RK,
start-up) is the Cartesian ``IVPSolver``'s, unchanged, through ``PolarIVPSolver``; only the global
system (groups (m[, l]), spin/regularity components) and the right-hand side (``SphereEvaluator``)
are spherical. ``backend="cuda"``: curvilinear_device.py; ``f_backend="fused"``: sphere_fused.py.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import structural_rank

from ..discretize.plan import RankError
from ..discretize.sphere_eval import embed_coeffs
from ..discretize.sphere_plan import SpherePlan
from ..errors import InputTypeError, OptionError, UnsupportedError, did_you_mean
from .integrator import IVPSolver, check_dt_backend, check_substeps, finite_cadence, setup_scheme
from .polar import PolarGlobalSystem, PolarIVPSolver
from .real_projection import m0_projection, m0_projection_text
from .colborder import BORDER_MODES, plan_column_border
from .system import ORDERINGS, FactorCache, GlobalSystem, GroupSlot

__all__ = ["SphereGlobalSystem", "SphereIVPSolver"]


class SphereGlobalSystem(PolarGlobalSystem):
    """All groups of a spherical IVP stacked in one vector X; gather/scatter and right-hand-side maps
    are precomputed sparse/integer maps (no Python loop over groups at run time). Solves, gather and
    scatter are PolarGlobalSystem's (same data layout: flat coefficient indices per column)."""

    def __init__(self, plan: SpherePlan, ordering: str = "interleaved", reduced: bool = False,
                 border: str = "auto") -> None:
        if border not in BORDER_MODES:
            raise OptionError(f"border must be one of {BORDER_MODES}, got {border!r}",
                              fix="border='auto' (column border where a group has dense tau columns) "
                                  "or 'rows' (boundary rows only)" + did_you_mean(str(border), BORDER_MODES),
                              doc="docs/site/guide/timesteppers.md")
        if ordering not in ORDERINGS:
            raise OptionError(f"ordering must be one of {ORDERINGS}, got {ordering!r}",
                              fix="ordering='interleaved' (default)" + did_you_mean(str(ordering), ORDERINGS),
                              doc="docs/site/guide/timesteppers.md")
        # a switch, never the truth value of another object (review of audit inputs:
        # bool(reduced) read reduced="no" as True on the polar and spherical systems)
        from ..errors import check_flag
        reduced = check_flag(reduced, "reduced", "SphereGlobalSystem")
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
        rhs_blocks: list[list] = [[] for _ in pb.equations]
        off = 0
        for g in plan.groups:
            gm = plan.matrices(g, self.reduced)
            n = gm.L.shape[0]
            if n == 0:
                continue
            if n != gm.L.shape[1]:
                raise RankError(f"[{gm.label}] non-square group system {gm.L.shape}")
            cm = self._smeta(plan.var_slots(g), [(v.domain, v.rank) for v in self.variables], g)
            rm = self._smeta(plan.eq_slots(g), [(eq.domain, eq.lhs.rank) for eq in pb.equations], g)
            # symmetry subspace: keep the representative entries of the full group
            cm = [cm[j] for j in gm.col_index]
            rm = [rm[i] for i in gm.row_index]
            for k, fl, _ in cm:
                cols_obj.append(k)
                cols_flat.append(fl)
            kinds = [r[2] for r in gm.rows]
            rperm, cperm, nb = GlobalSystem._ordering(self, gm, cm, rm, kinds)
            self.slots.append(GroupSlot(g, gm.label, off, n, gm.M.tocsr(), gm.L.tocsr(), kinds,
                                        rperm, cperm, nb))
            for i in range(len(pb.equations)):
                A = plan.rhs_map(g, i)
                A = (A[gm.row_index] if self.reduced else A).tocoo()     # representative rows
                if A.nnz:
                    rhs_blocks[i].append((A.row + off, A.col, A.data))
            off += n
        self.n = off
        self.complex, self.dtype = True, np.complex128
        self.cols_obj, self.cols_flat = np.array(cols_obj, int), np.array(cols_flat, int)
        self._col_sel = [np.nonzero(self.cols_obj == k)[0] for k in range(len(self.variables))]
        # symmetry subspace: mirror members of the kept orbits (PolarGlobalSystem.scatter)
        self.scatter_extra = (plan.subspace.scatter_maps(self.var_shapes) if self.reduced
                              else [None] * len(self.variables))
        self.M = sp.block_diag([s.M for s in self.slots], format="csr").astype(self.dtype)
        self.L = sp.block_diag([s.L for s in self.slots], format="csr").astype(self.dtype)
        self._rhs_maps = []
        for blocks, shape in zip(rhs_blocks, self.eq_shapes):
            size = int(np.prod(shape))
            if blocks:
                r = np.concatenate([b[0] for b in blocks])
                c = np.concatenate([b[1] for b in blocks])
                d = np.concatenate([b[2] for b in blocks])
                self._rhs_maps.append(sp.csr_matrix((d, (r, c)), shape=(self.n, size), dtype=complex))
            else:
                self._rhs_maps.append(sp.csr_matrix((self.n, size), dtype=complex))
        for s in self.slots:
            pat = sp.csr_matrix((abs(s.M) + abs(s.L)) > 0, dtype=float)
            if structural_rank(pat) != s.size:
                raise RankError(f"[{s.label}] structurally singular IVP pencil M + dt L: refused")
        self.border = border
        self._border_notes = self._column_border() if border == "auto" else []
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

    def _column_border(self) -> list[tuple[str, int, str]]:
        """Column border per sparsity bucket (timestep/colborder.py): where a group
        has a few dense columns (lifted gradient tau of the shell), move them and as many bulk rows
        to the border, so that the core is banded again. Decided on the first group of each bucket,
        applied to all its groups (same pattern). Returns (label, groups, note) per bucket."""
        groups: dict = {}
        for i, s in enumerate(self.slots):
            pat = sp.csr_matrix((abs(s.M) + abs(s.L)) > 0, dtype=np.int8)[s.rperm][:, s.cperm]
            groups.setdefault((s.size, s.n_border, pat.indptr.tobytes(), pat.indices.tobytes()), []).append(i)
        notes = []
        for idx in groups.values():
            s0 = self.slots[idx[0]]
            plan, why = plan_column_border(s0.M, s0.L, s0.rperm, s0.cperm, s0.n_border)
            if plan is None:
                notes.append((s0.label, len(idx), f"row border only ({why})"))
                continue
            for i in idx:
                s = self.slots[i]
                s.rperm, s.cperm = s.rperm[plan.rows], s.cperm[plan.cols]
                s.n_border, s.col_border = plan.n_border, plan.n_cols
            notes.append((s0.label, len(idx), plan.explain()))
        return notes

    def _smeta(self, slots, objs, g) -> list[tuple]:
        """(object index, flat index in its coefficient array, sort key (l, n) or None)."""
        S = self.plan.space
        out = []
        for oi, c, _off, n in slots:
            if n == 0:
                continue
            dom, rank = objs[oi]
            flat, key = self.plan.flat_index(dom, rank, c, g)
            bulk = S.bulk(dom)
            for fl, k in zip(flat, key):
                out.append((oi, int(fl), float(k) if bulk else None))
        return out

    def explain(self) -> str:
        """The polar system's explain() for spherical groups, with the border mode."""
        text = super().explain().replace("PolarGlobalSystem", "SphereGlobalSystem").replace(
            "azimuthal group(s)", "group(s)")
        if self.border == "rows":
            return text + "\n  border='rows': boundary rows only (no column border, as requested)"
        # border='auto': every bucket is declared, moved or not, with the reason when it is kept
        # on the row border (no dense column worth moving, gain too small, singular core). The
        # old text said "none needed (no group with dense columns)" whenever nothing moved, also
        # when dense columns were there and the plan was refused.
        moved = [n for n in self._border_notes if n[2].startswith("column border")]
        kept = [n for n in self._border_notes if not n[2].startswith("column border")]
        lines = [text, f"  border='auto': column border in {len(moved)} of {len(self._border_notes)} "
                       "sparsity bucket(s)"]
        for group, what in ((moved, "with a column border"), (kept, "on the row border")):
            for label, k, note in group[:3]:
                lines.append(f"  [{label}] ({k} group(s)) {note}")
            if len(group) > 3:
                lines.append(f"  ... {len(group) - 3} more bucket(s) {what}")
        return "\n".join(lines)


class SphereIVPSolver(PolarIVPSolver):
    """IVPSolver on a SpherePlan: same schemes and step arithmetic (inherited), spherical system and F."""

    def __init__(self, plan: SpherePlan, scheme: str, dt: float, *, reduced: bool | None = None,
                 ordering: str = "interleaved", startup: str = "lower_order",
                 startup_substeps: int = 2, time_parameter: str | None = None,
                 backend: str = "cpu", f_backend: str = "evaluator", complexify: bool = False,
                 bc_time: str = "auto", border: str = "auto", project_m0: bool = True, check_finite: bool | int = True,
                 **kw: Any) -> None:
        setup_scheme(self, scheme, startup, where=f"SphereIVPSolver(plan, {scheme!r}, ...)",
                     bc_time=bc_time)
        check_dt_backend(dt, backend, f"SphereIVPSolver(plan, {scheme!r}, ...)")
        self._finite_every = finite_cadence(check_finite, f"SphereIVPSolver(plan, {scheme!r}, ...)")
        if f_backend not in ("evaluator", "fused"):
            raise UnsupportedError(f"spherical IVPs: f_backend={f_backend!r}",
                                   fix="f_backend='evaluator' (the reference evaluator of F) or "
                                       "'fused' (planned evaluation, same numbers up to rounding)"
                                       + did_you_mean(str(f_backend), ("evaluator", "fused")))
        if complexify:
            raise UnsupportedError("spherical IVPs: complexify=True", why="the groups are already "
                                   "complex", fix="drop complexify=")
        if reduced is None:
            reduced = plan.problem.symmetry is not None     # declared subspace: run in it (explain)
        from ..errors import check_flag      # reduced="no" read as True (audit inputs review)
        reduced = check_flag(reduced, "reduced", "SphereIVPSolver")
        if reduced and plan.problem.symmetry is None:
            raise OptionError("reduced=True needs a symmetry subspace declared on the problem",
                              fix="declare symmetry=SymmetrySubspace(...) on the problem, or drop "
                                  "reduced=")
        if kw:
            raise InputTypeError(f"unexpected arguments for a spherical IVP: {sorted(kw)}",
                                 fix="arguments of SphereIVPSolver: scheme, dt, reduced, ordering, startup, "
                                     "startup_substeps, time_parameter, backend, f_backend, "
                                     "bc_time, border, project_m0, check_finite")
        self.plan = plan
        self._params0 = dict(plan.params)   # matrices built with these (check_params_unchanged)
        if plan.space.shell:                      # beyond the tested shell grid: experimental
            from .._experimental import check_shell_size
            check_shell_size(plan.space.sizes(False))
        self.scheme, self.dt = scheme, float(dt)
        self.startup = startup
        self.startup_substeps = check_substeps(startup_substeps, "SphereIVPSolver")
        from .integrator import check_time_parameter    # a name the equations use (audit inputs)
        check_time_parameter(plan, time_parameter, "SphereIVPSolver")
        self.time_parameter = time_parameter
        self.backend, self.f_backend, self.xp = "cpu", "evaluator", np
        # border='auto': column border where groups have dense tau columns (timestep/colborder.py);
        # 'rows': boundary rows only (the layout before), kept as the declared fallback
        self.system = SphereGlobalSystem(plan, ordering=ordering, reduced=reduced, border=border)
        # m = 0 coefficients projected onto real fields after every solve of the step
        # (declared in explain(); project_m0=False keeps the unprojected step)
        self._m0proj = m0_projection(self.system, project_m0)
        self.evaluator = plan.evaluator
        if reduced:
            # symmetry subspace: F evaluated on the folded grid (northern hemisphere for the
            # equatorial parity, azimuthal sector for an m-fold class), declared in explain()
            from ..discretize.curvilinear_symmetry import FoldSpec
            from ..discretize.sphere_eval import SphereEvaluator
            self.evaluator = SphereEvaluator(plan.space, fold=FoldSpec(plan.problem, plan.space.sizes(True)[0], True))
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
        # a spurious growing tau mode of the implicit operator (shell) is refused
        # when the step amplifies it, reported by explain() otherwise (timestep/tau_modes.py)
        self._tau_note = ""
        if plan.space.shell:
            from .tau_modes import check_tau_modes
            self._tau_note = check_tau_modes(self, f"SphereIVPSolver(plan, {scheme!r}, dt={dt!r})")
        # f_backend='fused': planned evaluation of the same expressions (common
        # sub-expressions, cached constants, real-field transforms); the reference evaluator
        # stays self.evaluator (tangent / adjoint, data right-hand sides)
        self.f_backend = f_backend
        self._sfused = None
        if f_backend == "fused":
            from .sphere_fused import SphereFusedRHS
            self._sfused = SphereFusedRHS(self)

    def _graph_time_support(self):
        """run_graph with a time-dependent F: the planned F reads its time factors from
        device slots refreshed before each replay (sphere_fused.GraphScalars). None (refused)
        without f_backend='fused' on the GPU or with time-dependent boundary data."""
        if self.backend != "cuda" or self._sfused is None or self._bc_data() is not None:
            return None
        if self._sfused.ev.products != "kernel":
            # only the generated product kernels read the device slots: array operations would
            # take a_g(t) as host numbers frozen by the capture
            return None
        from ..discretize._cublas import available
        if not available():
            return None                  # CuPy's cuBLAS cannot be captured: refused
        if getattr(self, "_gscalars", None) is None:
            from .sphere_fused import GraphScalars
            self._gscalars = GraphScalars(self, self._sfused.ev)
        return self._gscalars

    def set_dt(self, dt: float) -> None:
        """Change the step (RK schemes); the spurious-tau-mode check is redone for the new dt."""
        old = self.dt
        super().set_dt(dt)
        if self.plan.space.shell:
            from .tau_modes import check_tau_modes
            try:
                self._tau_note = check_tau_modes(self, f"SphereIVPSolver.set_dt({dt!r})")
            except Exception:
                self.dt = old                            # refused: the solver keeps its step
                raise

    def F(self, X: np.ndarray, t: float) -> np.ndarray:
        """Right-hand sides at state X (shell: projected as r^k F, plan.rhs_rpow)."""
        self.n_F += 1
        data = dict(self._ncc)
        data.update((v, c) for v, c in zip(self.system.variables,
                                           self.system.scatter(X, only=self._rhs_vars)) if c is not None)
        params = dict(self.plan.params)
        if self.time_parameter:
            params[self.time_parameter] = t
        if self._sfused is not None:
            return self.system.gather_rhs(self._sfused.evaluate(data, params))
        # an expression on fewer coordinates than its equation (e.g. a boundary value a(t) * uw
        # with uw on [theta] only) is embedded exactly in the equation's layout (B4, embed_coeffs)
        out = [None if e is None else embed_coeffs(self.evaluator.evaluate(e, data, params, k), shape,
                                                   f"right-hand side of equation {i}")
               for i, (e, k, shape) in enumerate(zip(self._rhs_exprs, self.plan.rhs_rpow,
                                                     self.system.eq_shapes))]
        return self.system.gather_rhs(out)

    def _data_rhs_eq(self, k: int, params: dict, e=None):
        """State-independent right-hand side of equation k (time-dependent boundary data;
        ``e``: its time derivative in its place), embedded in the equation's layout like
        :meth:`F`."""
        e = self._rhs_exprs[k] if e is None else e
        c = self.evaluator.evaluate(e, dict(self._ncc), params, self.plan.rhs_rpow[k])
        return embed_coeffs(c, self.system.eq_shapes[k], f"right-hand side of equation {k}")

    def _short_notes(self) -> list[str]:
        """The m = 0 projection and the spurious tau-mode check (a warning) of the shell."""
        return super()._short_notes() + ([self._tau_note] if self._tau_note else [])

    def _explain_full(self) -> str:
        st = self._startup_text()
        M = self.plan.space.sizes(True)
        gpu = self.backend == "cuda"
        if self._sfused is not None:
            return (f"IVPSolver[sphere](scheme={self.scheme}, dt={self.dt:g}, constant step{st}, backend={self.backend})\n"
                    + self.system.explain() + "\n" + self._sfused.explain()
                    + f"\n  linear solves: {'GPU' if gpu else 'C++'} bordered banded" + self._bc_text() + m0_projection_text(self)
                    + (f"\n  {self._tau_note}" if self._tau_note else "") + "\n" + self._finite_text())
        return (f"IVPSolver[sphere](scheme={self.scheme}, dt={self.dt:g}, constant step{st}, backend={self.backend})\n"
                + self.system.explain()
                + f"\n  F: spherical pseudospectral {'CuPy (GPU: cuFFT, batched cuBLAS)' if gpu else 'NumPy'} "
                  f"evaluator (angular derivatives diagonal in l, radial derivatives on the nodes, products on "
                  f"the dealiased {' x '.join(str(v) for v in M)} grid, projection back"
                  + (f"; {self.evaluator.fold.explain()}" if getattr(self.evaluator, "fold", None) else "")
                  + "); linear solves: "
                  f"{'GPU' if gpu else 'C++'} bordered banded" + self._bc_text() + m0_projection_text(self)
                  + (f"\n  {self._tau_note}" if self._tau_note else "") + "\n" + self._finite_text())


_prev_new = IVPSolver.__new__


def _dispatch_new(cls, plan=None, *args, **kwargs):
    """``IVPSolver(sphere_plan, ...)`` builds a SphereIVPSolver (same public API)."""
    if cls is IVPSolver and getattr(plan, "is_sphere", False):
        return object.__new__(SphereIVPSolver)
    return _prev_new(cls, plan, *args, **kwargs)


IVPSolver.__new__ = _dispatch_new
