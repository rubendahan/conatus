"""GPU backend of the curvilinear IVPs (polar, cylindrical, spherical): ``backend="cuda"``.

``IVPSolver(polar_or_sphere_plan, ..., backend="cuda")`` keeps the host global system (groups,
orderings, buckets, gather/scatter and right-hand-side maps: the single source of the structure)
and mirrors it on the device, exactly as ``device.DeviceSystem`` does for Cartesian plans:

- state X, stage vectors and histories are CuPy arrays; M, L and the right-hand-side maps
  (level conversion and declared r^p multiplication, per equation) are CSR matrices applied by a
  plain CUDA kernel (``DeviceCsr``, capturable in CUDA graphs);
- linear solves: **banded** (declared choice) -- the batched bordered banded GPU solver of the
  core (``runtime.cuda.BandedBatch``, one warp per group, banded LU with partial pivoting in the
  band + Woodbury/Schur on the boundary rows), one batch per bucket of groups sharing a sparsity
  pattern. Small dense batched solves are not used: the groups are banded after the interleaved
  ordering and the banded factorization is the same arithmetic as the CPU core;
- nonlinear terms: the polar / spherical pseudospectral evaluators run unchanged on CuPy arrays
  (``PolarEvaluator(space, xp=cupy)``): FFTs in phi (and z) by cuFFT, radial / Zernike / spin-
  weighted Legendre transforms as batched matrix products (cuBLAS), products on the dealiased grid
  on the device.

Host <-> device traffic inside a step: none for the arithmetic; a scalar read back only where a
real field goes through a pointwise function (the declared check of the imaginary part, R9). The
system is built from any host global system exposing ``M, L, slots, _buckets, _col_sel,
cols_flat, _rhs_maps`` (full or symmetry-reduced).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..runtime.banded import solve_operand
from .device import DeviceCsr

__all__ = ["CurvilinearDeviceSystem", "attach_device"]


class CurvilinearDeviceSystem:
    """Device mirror of a PolarGlobalSystem / SphereGlobalSystem (same ordering, buckets, dtype)."""

    solver_kind = "banded"

    def __init__(self, host, pivot: bool = True) -> None:
        from ..runtime import cuda
        if not cuda.available():
            from ..errors import BackendError
            raise BackendError(
                "backend='cuda' requested but no CUDA device is available",
                where="IVPSolver(curvilinear plan, ..., backend='cuda')",
                why=cuda.unavailable_reason(),
                fix="run in a job on a GPU node of your cluster, build the core with CUDA "
                    "(docs/site/install.md#gpu-cuda-install: CONATUS_CUDA=ON, CuPy from "
                    "conda-forge or the gpu extra of a wheel), or use backend='cpu'")
        import cupy as cp
        self.cp, self._cuda, self.host, self.pivot = cp, cuda, host, bool(pivot)
        for name in ("plan", "reduced", "ordering", "variables", "var_shapes", "eq_shapes", "n",
                     "complex", "dtype", "slots"):
            setattr(self, name, getattr(host, name))
        self.M = DeviceCsr(host.M, host.dtype)
        self.L = DeviceCsr(host.L, host.dtype)
        self._buckets = [(idx, cp.asarray(rows), cp.asarray(cols)) for idx, rows, cols in host._buckets]
        self._col_sel = [(cp.asarray(sel), cp.asarray(host.cols_flat[sel])) for sel in host._col_sel]
        # one CSR per equation: level-0 coefficients (flattened) -> rows of the stacked system
        self._rhs_maps = [DeviceCsr(R, host.dtype) if R.nnz else None for R in host._rhs_maps]
        # symmetry-reduced systems: mirror members of the kept coefficients, a[flat] = f X[x]
        # (factors +-1, possibly in dropped groups); None for full systems (no attribute either)
        extra = getattr(host, "scatter_extra", None) or [None] * len(self.variables)
        self._extra = [None if e is None else (cp.asarray(e[0]), cp.asarray(e[1]),
                                               cp.asarray(e[2], dtype=host.dtype)) for e in extra]
        self._factors: dict = {}

    # ------------------------------------------------------------------ solves
    def factor(self, a: float, b: float) -> list:
        """Batched GPU factorizations of a M + b L, one per bucket (cached by (a, b))."""
        key = (float(a), float(b))
        f = self._factors.get(key)
        if f is None:
            # declared conditioning guard (timestep/polar.py, FACTOR_COND_MAX): the host factorizes
            # once per (a, b) and refuses a numerically singular group before any device solve
            # (one CPU factorization per distinct (a, b), outside the step loop)
            if hasattr(self.host, "factor"):
                self.host.factor(a, b)
            f = []
            for idx, _, _ in self.host._buckets:
                mats = [(a * self.slots[i].M + b * self.slots[i].L).tocsr()[self.slots[i].rperm][:, self.slots[i].cperm]
                        .astype(self.dtype) for i in idx]
                s0 = self.slots[idx[0]]
                f.append(self._cuda.BandedBatch(mats, list(range(s0.n_border)),
                                                pivot=self.pivot, dtype=self.dtype,
                                                col_border=bool(getattr(s0, "col_border", 0))))
            self._factors[key] = f
        return f

    def solve(self, a: float, b: float, rhs):
        """X with (a M + b L) X = rhs, all groups, in place on device buffers (same stream)."""
        cp = self.cp
        rhs = solve_operand(rhs, self.dtype, "solve (curvilinear, GPU)", cp)
        x = cp.empty(self.n, dtype=self.dtype)
        stream = int(cp.cuda.get_current_stream().ptr)
        for (_idx, rows, cols), batch in zip(self._buckets, self.factor(a, b)):
            B = cp.ascontiguousarray(rhs[rows])
            batch._s.solve_ptr(int(B.data.ptr), int(B.shape[0]), int(B.shape[1]), stream)
            x[cols] = B
        return x

    def solve_transpose(self, a: float, b: float, rhs):
        """(a M + b L)^T x = rhs with the cached factors (plain transpose, no conjugation)."""
        cp = self.cp
        rhs = solve_operand(rhs, self.dtype, "solve_transpose (curvilinear, GPU)", cp)
        x = cp.empty(self.n, dtype=self.dtype)
        stream = int(cp.cuda.get_current_stream().ptr)
        for (_idx, rows, cols), batch in zip(self._buckets, self.factor(a, b)):
            B = cp.ascontiguousarray(rhs[cols])
            batch._s.solve_transpose_ptr(int(B.data.ptr), int(B.shape[0]), int(B.shape[1]), stream)
            x[rows] = B
        return x

    @property
    def n_factorizations(self) -> int:
        """Number of distinct (a, b) factorizations held on the device."""
        return len(self._factors)

    # ------------------------------------------------------------------ gather / scatter
    def scatter(self, X, only: frozenset | None = None) -> list:
        """Global device vector -> device coefficient arrays per variable (``only``: names)."""
        cp = self.cp
        out = []
        for v, (sel, flat), shape, ex in zip(self.variables, self._col_sel, self.var_shapes, self._extra):
            if only is not None and v.name not in only:
                out.append(None)
                continue
            a = cp.zeros(int(np.prod(shape)), dtype=self.dtype)
            a[flat] = X[sel]
            if ex is not None:                        # mirror members (symmetry subspace)
                a[ex[1]] = ex[2] * X[ex[0]]
            out.append(a.reshape(shape))
        return out

    def gather_state(self, coeffs: Sequence, check_subspace: bool = True):
        """Coefficient arrays (host or device) -> device global vector (host checks kept)."""
        host = [self.cp.asnumpy(c) if hasattr(c, "device") else np.asarray(c) for c in coeffs]
        return self.cp.asarray(self.host.gather_state(host, check_subspace))

    def gather_rhs(self, eq_coeffs: Sequence):
        """Equation right-hand sides (device level-0 coefficients) -> rows: one CSR product each."""
        cp = self.cp
        b = cp.zeros(self.n, dtype=self.dtype)
        for R, c in zip(self._rhs_maps, eq_coeffs):
            if c is not None and R is not None:
                b += R @ cp.asarray(c, dtype=self.dtype).reshape(-1)
        return b

    def explain(self) -> str:
        """The host system's explain() with the device solves and the first buckets."""
        lines = [self.host.explain().replace("batched C++ bordered banded solves",
                                             "batched GPU bordered banded solves, one warp per group")]
        for (_idx, _, _), batch in zip(self.host._buckets[:4], self._factors.get(next(iter(self._factors), None), [])):
            lines.append("  " + batch.explain())
        lines.append("  state resident on the device (CuPy); linear solves: banded (declared choice, "
                     f"not dense batched), pivot={'partial' if self.pivot else 'none (declared)'}; "
                     "M, L, right-hand-side maps: device CSR (CUDA kernel)")
        return "\n".join(lines)


def attach_device(solver, host) -> None:
    """Switch a Polar/SphereIVPSolver to the GPU: device system, CuPy evaluator, device NCC data.
    Called by the solvers' constructors for ``backend="cuda"`` (isolated hook)."""
    from ..errors import require_cupy
    cp = require_cupy("IVPSolver(curvilinear plan, ..., backend='cuda')")
    solver.backend, solver.xp = "cuda", cp
    solver.system = CurvilinearDeviceSystem(host)
    ev_cls = type(solver.plan.evaluator)
    fold = getattr(solver.evaluator, "fold", None)     # symmetry subspace: same folded grid on the GPU
    solver.evaluator = ev_cls(solver.plan.space, xp=cp, **({"fold": fold} if fold is not None else {}))
    solver.X = cp.zeros(solver.system.n, dtype=solver.system.dtype)
    solver._ncc = {f: cp.asarray(c) for f, c in solver._ncc.items()}
