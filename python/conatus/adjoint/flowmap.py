"""Flow map of a discretized IVP with its exact discrete tangent (jvp) and adjoint (vjp).

The flow map Φ: X_0 ↦ X_N is the composition of the constant-step IMEX steps of
:class:`conatus.timestep.IVPSolver` (same schemes, same startup). Differentiating the
*discrete* map (not the PDE) makes the derivative exact by construction:

* tangent: every stage is re-solved with the same operators (a M + b L) applied to the
  linearized right-hand side F'(X_k) δX_k;
* adjoint: the same relations transposed (bilinear pairing ⟨a, b⟩ = Σ a b, no conjugation,
  decision D8), stages and steps in reverse order.

Choices (explicit arguments, shown by ``explain()``):

* ``f_transpose="polynomial"`` (default): F' and F'ᵀ matrix-free from the polynomial form of the
  RHS (:class:`LinearizedRHS`: explicit transposes of the dealiased transforms and derivatives);
  ``"dense"``: F' from the IR linearization (evaluator) and F'ᵀ from its dense Jacobian (n JVPs,
  refused above ``max_dense``) — the phase-1 oracle, needed for non-polynomial RHS;
* every scheme of ``IVPSolver``: IMEX Runge–Kutta (RK111/222/443, RKSMR, RKGFY) through
  ``rk_stages``; SBDF1–4 and the CN multistep family (CNAB1/2, MCNAB2, CNLF2) on the augmented
  state, with the CN relation (a_0/dt M + b_0 L) X^{n+1} = sum_j c_j F^{n+1-j} - (a_j/dt) M X^{n+1-j}
  - b_j L X^{n+1-j} differentiated and transposed term by term;
* ``solves="banded"`` (default): the batched bordered-banded factors of the solver's own system
  (``GlobalSystem`` on CPU, ``DeviceSystem`` on the GPU): ``solve`` for J, ``solve_transpose`` for
  Jᵀ, the **same factors** as the forward map; ``"superlu"``: SuperLU of the global a M + b L
  (phase-1 oracle, CPU);
* RK stage states come from ``IVPSolver.rk_stages`` (the single implementation of the RK
  arithmetic), so the tangent and the adjoint replay exactly the forward stages;
* ``checkpoints=None`` stores every state; an integer is the Revolve budget. Multistep (SBDF)
  schemes are reversed on the augmented state (X_n, …, X_{n-q+1}), so Revolve covers them too;
  the binomial schedule is run as an explicit action list (:func:`revolve_actions`);
* ``checkpoint_store``: where the Revolve checkpoints live, ``"device"``
  (default), ``"host"`` (pinned host memory) or ``"disk"`` (``checkpoint_dir``); bit-identical
  results, only the place and the copy cost change (:mod:`conatus.adjoint.checkpoints`);
  ``vjp(..., journal=dir, budget_s=...)`` suspends a long reverse sweep at a step boundary and a
  later job resumes it bit for bit;
* **time-dependent F** (``IVPSolver(..., time_parameter=...)``): the map starts at the explicit
  time ``t0`` (required, no default) and every F, F' and F'ᵀ is evaluated at the stage time of
  the forward step (t_n + c_i dt, the same floating-point expression as ``IVPSolver``). The
  terms with explicit t are split exactly as F_0(X) + sum_g a_g(t) F_g(X) (``split_time``):
  the tangent evaluates each a_g at the stage time; state-independent terms (forcing, boundary
  values) drop out. Time-dependent boundary rows are constant RHS: derivative 0.
* **spherical shell** (``ShellBasis × SphereBasis × Fourier``): the state is
  complex (stored m >= 0 of real fields) and the step is only real-linear, so the adjoint is
  taken for the **real pairing** <a, b> = Re(a^H b) (``FlowMap.pairing == "real"``): the
  transposed solves and products use the conjugate transposes (a M + b L)^H, M^H, L^H, and F'†
  comes from :class:`conatus.adjoint.sphere_rhs.SphereLinearizedRHS` (explicit adjoints of the
  spherical transforms). Same API, same schemes; ``FlowMap.pair(a, b)`` is the pairing in use.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..errors import OptionError
from ..ir import expr as E
from ..timestep.integrator import IVPSolver, cn_coefficients, rk_tableau, sbdf_coefficients
from .rhs_transpose import LinearizedRHS
from .._deprecation import renames

__all__ = ["FlowMap", "revolve_schedule", "revolve_actions", "revolve_cost", "flowmap_memory_estimate",
           "journal_extra"]


# ---------------------------------------------------------------------------- Revolve
def _beta(c: int, r: int) -> int:
    """Number of steps reversible with c checkpoints and r forward sweeps (Griewank)."""
    return comb(c + r, c)


#: Review of the audit "inputs": a budget of 0 slots was accepted ("no slot:
#: recompute from the start") but its schedule was exactly that of checkpoints=1 (split at
#: steps - 1, one stored state at a time): the run held a checkpoint the budget said it would not
#: (revolve_cost(n, 0)["peak_stored"] == 1). 0 is refused rather than renamed.
_ZERO_SLOTS = "{where}: checkpoints=0 is not a Revolve budget"
_ZERO_WHY = ("reversing a step needs its input state: with 0 slots the schedule was that of "
             "checkpoints=1 and stored one state at a time, a memory the budget said was not used")
_ZERO_FIX = ("checkpoints=1 (the least memory: one stored state besides the input), a larger "
             "budget, or checkpoints=None (store every step)")


def _check_revolve(steps: object, checkpoints: object, where: str) -> None:
    """Integer counts (audit inputs: revolve_schedule(10.5, 2) returned a float
    split point, a float checkpoint budget looped on non-integer binomials)."""
    from ..errors import check_count
    check_count(steps, "steps", f"{where}()")
    check_count(checkpoints, "checkpoints", f"{where}()", minimum=0)
    if checkpoints == 0:
        raise OptionError(_ZERO_SLOTS.format(where=f"{where}({steps}, 0)"),
                          why=_ZERO_WHY, fix=_ZERO_FIX)


@renames("conatus.adjoint.flowmap:revolve_schedule")
def revolve_schedule(steps: int, checkpoints: int) -> int:
    """Binomial split point for reversing ``steps`` steps with ``checkpoints`` free slots.

    Returns m (1 <= m < steps): advance m steps, store a checkpoint, reverse the tail, then
    reverse the head with one slot less. Any 1 <= m < steps is correct; the binomial choice
    minimizes recomputation (Griewank & Walther 2000)."""
    _check_revolve(steps, checkpoints, "revolve_schedule")
    if steps < 2:
        raise OptionError(f"revolve_schedule({steps}, {checkpoints}): nothing to split",
                          fix="a split point needs steps >= 2 (one step is reversed directly)")
    r = 0
    while _beta(checkpoints, r) < steps:
        r += 1
    m = steps - _beta(checkpoints - 1, r) if r > 0 else 1
    return min(max(m, 1), steps - 1)


@renames("conatus.adjoint.flowmap:revolve_actions")
def revolve_actions(steps: int, checkpoints: int) -> list[tuple]:
    """The binomial Revolve schedule of :meth:`FlowMap.vjp`, flattened into a list of actions
    (the recursion of ``_revolve`` written iteratively, same order, same split points):

    * ``("advance", a, b)``: from the checkpoint at step a, take steps a .. b-1 and store a
      checkpoint at b;
    * ``("reverse", n)``: adjoint of step n from the checkpoint at n, which is then released.

    The checkpoint at 0 is the input X_0. Checkpoints are released in last-in first-out order,
    at most ``checkpoints`` (+ the input) are held at once. An explicit list (not a recursion)
    has no depth limit and gives a position (an index) at which a reverse sweep can be suspended
    and resumed by another job."""
    _check_revolve(steps, checkpoints, "revolve_actions")
    out: list[tuple] = []
    stack = [(0, steps, checkpoints)]
    while stack:
        start, stop, c = stack.pop()
        if stop - start == 1:
            out.append(("reverse", start))
            continue
        m = revolve_schedule(stop - start, c)
        out.append(("advance", start, start + m))
        stack.append((start, start + m, c))            # head: done after the tail
        stack.append((start + m, stop, c - 1))
    return out


@renames("conatus.adjoint.flowmap:revolve_cost")
def revolve_cost(steps: int, checkpoints: int | None) -> dict:
    """Forward steps recomputed by the reverse sweep of ``FlowMap.vjp`` (excluding the stage
    recomputation inside each step's adjoint), its ratio to ``steps``, the peak number of stored
    checkpoints and the number of checkpoint stores. ``checkpoints=None``: full storage."""
    if checkpoints is None:
        return dict(advances=steps, sweeps=1.0, peak_stored=steps, stores=steps)
    adv = stores = held = peak = 0
    for act in revolve_actions(steps, checkpoints):
        if act[0] == "advance":
            adv += act[2] - act[1]
            stores += 1
            held += 1
            peak = max(peak, held)
        elif act[1] > 0:
            held -= 1
    return dict(advances=adv, sweeps=adv / steps, peak_stored=peak, stores=stores)


def flowmap_memory_estimate(fm: FlowMap) -> dict:
    """Declared accounting (bytes, v0: counted from the code, not a liveness plan) of the device
    memory that ``jvp``/``vjp`` add to a forward step, for one RK step of the map:

    * ``rk_stage_states``: the s + 1 stage states of a step recomputed from its checkpoint;
    * ``vjp_vectors``: the s + 1 stage cotangents, the accumulated F'ᵀ inputs (s), mu, M^T mu and
      L^T mu; ``jvp_vectors``: the s + 1 stage tangents, F' (S_k) dS_k (s) and the right-hand side;
    * ``factor_grid_cache``: the passive factor grids G(D^a f) of the F'ᵀ being applied, full-size
      factors on the dealiased grid: one stage at a time for RK (``_rk_vjp`` drops them once used), over the history levels of a multistep step;
    * ``transpose_transients``: grid cotangents of every factor during one F'ᵀ, plus the
      transposed dealiasing projection of the output (one grid per equation component);
    * ``transposed_operators``: M^T and L^T as device CSR (built at the first vjp);
    * ``checkpoint``: one augmented state (what each Revolve slot holds).

    Not counted: the forward step itself (see ``conatus.runtime.memory``), allocator rounding,
    the CUDA context."""
    it = np.dtype(fm.dtype).itemsize
    n = fm.n
    s = fm.solver
    st = next((x for x in fm._steps if x.kind == "rk"), None)
    ns = (len(st.tableau[2]) - 1) if st is not None else 1
    A = st.tableau[0] if st is not None else [[0]]
    nF = sum(1 for k in range(ns) if any(A[i][k] != 0 for i in range(k + 1, ns + 1)))
    out = dict(rk_stage_states=(ns + 1) * n * it,
               vjp_vectors=(2 * ns + 4) * n * it,
               jvp_vectors=(2 * ns + 2) * n * it,
               checkpoint=fm.Q * n * it)
    rl = fm.rhs_lin
    if isinstance(rl, LinearizedRHS):
        nd = len(rl.bases)
        P = int(np.prod([getattr(b, "grid_size", b.size) for b in rl.bases]))
        full = set()
        ncomp = 0
        for Pk in rl.polys:
            if Pk is None:
                continue
            ncomp += len(Pk)
            for monos in Pk.values():
                for _c, fs, _g in monos:
                    for f in fs:
                        sh = s.system.var_shapes[rl._vidx[f[0]]] if f[0] in rl._vidx else ()
                        if len(sh) >= nd:          # full-dimensional factor (NCC profiles: 1D)
                            full.add(f)
        nfac = len(full)
        out["factor_grids_per_stage"] = nfac
        # one stage's grids at a time (_rk_vjp drops them once its F'ᵀ is applied); multistep
        # steps keep them over the step's history levels (q stages)
        out["factor_grid_cache"] = (1 if st is not None else nF) * nfac * P * 8
        out["transpose_transients"] = (nfac + ncomp) * P * 8
    host = s.system.host if fm.gpu else s.system
    out["transposed_operators"] = int(sum(M.nnz * (it + 4) + (M.shape[0] + 1) * 4
                                          for M in (host.M, host.L)))
    out["vjp_extra_total"] = int(out["rk_stage_states"] + out["vjp_vectors"]
                                 + out.get("factor_grid_cache", 0) + out.get("transpose_transients", 0)
                                 + out["transposed_operators"])
    out["jvp_extra_total"] = int(out["rk_stage_states"] + out["jvp_vectors"]
                                 + out.get("transpose_transients", 0))
    return out


# ---------------------------------------------------------------------------- records
@dataclass(frozen=True)
class _Step:
    """One step of the trajectory: RK (from X_n only), SBDF or CN (from X_n … X_{n-q+1})."""
    kind: str                     # "rk" | "sbdf" | "cn"
    dt: float
    tableau: tuple | None = None  # RK: (A, H, c)
    q: int = 0                    # SBDF order / CN history levels used at this step
    coefs: tuple | None = None    # CN: (a, b, c) of this step (cn_coefficients)
    substeps: int = 1             # RK: number of substeps of size dt / substeps
    t: float = 0.0                # time at the start of the step (IVPSolver arithmetic)
    n: int = 0                    # index of the step


def _lever_keys(s) -> dict:
    """Fingerprint keys of the levers that change the arithmetic in the last bits:
    complex modes, the level-scheduled solve variants, the F transform options. Only their
    non-default values are added, so the journals of maps without them keep their fingerprint;
    a journal written without a lever is refused by a map with it and conversely (before, a sweep resumed with complexify=True or compact factors mixed two
    trajectories silently). Shared factors and the fused complex-mode maps give the same bits:
    not part of it."""
    out: dict = {}
    sysm = s.system
    host = getattr(sysm, "host", sysm)
    if getattr(host, "complexify", False):
        out["complexify"] = True
    if getattr(sysm, "host", None) is not None:          # DeviceSystem
        if not getattr(sysm, "level_schedule", True):
            out["level_schedule"] = False
        elif not getattr(sysm, "level_merge", True):
            out["level_merge"] = False
        if getattr(sysm, "compact_factors", False):
            out["compact_factors"] = True
    fu = getattr(s, "_fused", None)
    fu = getattr(fu, "inner", fu)
    if fu is not None:
        if getattr(fu, "transform_batch", None):
            out["transform_batch"] = int(fu.transform_batch)
        if getattr(fu, "_ffc", None) is not None:
            out["f_ffc"] = True
        if getattr(fu, "_low_on", False):
            out["f_low_factors"] = True
    return out


_ABSENT = object()
_STATE_ATTRS = ("t", "iteration", "_t_origin", "_it_origin")
_HISTORY_ATTRS = ("_MXh", "_Fh", "_LXh")


def _solver_state(s) -> tuple:
    """A copy of what ``IVPSolver.run`` changes: state, clock, time origin, multistep history."""
    from collections import deque
    hist = {}
    for name in _HISTORY_ATTRS:
        h = getattr(s, name, None)
        hist[name] = None if h is None else deque((x.copy() for x in h), maxlen=h.maxlen)
    return (s.X.copy(), {a: s.__dict__.get(a, _ABSENT) for a in _STATE_ATTRS}, hist)


def _set_solver_state(s, saved: tuple) -> None:
    """Put back a state taken by :func:`_solver_state` (new history deques: the run cleared and
    refilled the old ones in place)."""
    X, clock, hist = saved
    s.X = X
    for a, v in clock.items():
        if v is _ABSENT:
            s.__dict__.pop(a, None)
        else:
            setattr(s, a, v)
    for name, h in hist.items():
        if h is not None:
            setattr(s, name, h)


class FlowMap:
    """Φ(X_0) = X_N for an :class:`IVPSolver` (CPU or GPU), with exact discrete ``jvp`` and ``vjp``
    (see the module docstring for the explicit choices ``f_transpose``, ``solves``, ``checkpoints``)."""

    #: True: Φ(X_0) puts the solver's state, clock and multistep history back after its run
    #: (set by ``Simulation.flowmap``; review of the state audit, F6: ``sim.flowmap(T)(X0)``
    #: moved the simulation to X_N). False (the expert default): they are left at X_N.
    restore: bool = False

    @renames("conatus.adjoint.flowmap:FlowMap.__init__")
    def __init__(self, solver: IVPSolver, steps: int, *, checkpoints: int | None = None,
                 f_transpose: str = "polynomial", solves: str = "banded",
                 max_dense: int = 3000, t0: float | None = None,
                 checkpoint_store: str = "device", checkpoint_dir=None) -> None:
        if solver.backend != "cpu" and (f_transpose != "polynomial" or solves != "banded"):  # oracles: CPU
            raise OptionError("FlowMap on the GPU backend: f_transpose='polynomial' and "
                              "solves='banded' only (the dense and SuperLU oracles are CPU tools)",
                              fix="FlowMap(solver, n) with the defaults, or a CPU solver for the "
                                  "oracles", doc="docs/site/guide/dynamics/adjoint.md")
        self.tname = solver.time_parameter or None
        if self.tname and t0 is None:
            raise OptionError(f"FlowMap: F depends on {self.tname} (time_parameter): the start time "
                              "t0 is part of the map and must be given explicitly",
                              fix=f"FlowMap(solver, {steps}, t0=solver.t)", doc="docs/site/guide/dynamics/adjoint.md")
        self.t0 = float(t0) if t0 is not None else 0.0
        # a float is refused, never truncated (FlowMap(solver, 2.7)
        # was a 2-step map); same check as every run()
        from ..timestep.integrator import check_steps
        steps = check_steps(steps, "FlowMap(solver, steps)", "steps")
        if steps < 1:
            raise OptionError(f"FlowMap: steps must be >= 1, got {steps!r}",
                              fix="the number of steps of the map, e.g. FlowMap(solver, 100)", doc="docs/site/guide/dynamics/adjoint.md")
        if f_transpose not in ("polynomial", "dense"):
            raise OptionError(f"f_transpose must be 'polynomial' or 'dense', got {f_transpose!r}",
                              fix="f_transpose='polynomial' (matrix-free, default) or 'dense' "
                                  "(oracle, small n)", doc="docs/site/guide/dynamics/adjoint.md")
        if solves not in ("banded", "superlu"):
            raise OptionError(f"solves must be 'banded' or 'superlu', got {solves!r}",
                              fix="solves='banded' (the solver's factors, default) or 'superlu' "
                                  "(oracle)", doc="docs/site/guide/dynamics/adjoint.md")
        if checkpoints is not None and (isinstance(checkpoints, bool) or not isinstance(
                checkpoints, (int, np.integer)) or checkpoints < 0):
            raise OptionError(f"checkpoints must be None (full storage) or >= 1, got {checkpoints!r}",
                              fix="checkpoints=None (store every step) or checkpoints=k (Revolve "
                                  "with k slots)", doc="docs/site/guide/dynamics/adjoint.md")
        if checkpoints is not None and checkpoints == 0:       # not a bound (review, inputs)
            raise OptionError(_ZERO_SLOTS.format(where="FlowMap(..., checkpoints=0)"),
                              why=_ZERO_WHY, fix=_ZERO_FIX, doc="docs/site/guide/dynamics/adjoint.md")
        from .checkpoints import STORES
        if checkpoint_store not in STORES:
            raise OptionError(f"checkpoint_store must be one of {STORES}, got {checkpoint_store!r}",
                              fix="checkpoint_store='device' (default), 'host' (pinned host "
                                  "memory) or 'disk' (with checkpoint_dir=...)",
                              doc="docs/site/guide/dynamics/adjoint.md")
        if checkpoint_store != "device" and checkpoints is None:
            raise OptionError(f"checkpoint_store={checkpoint_store!r} with checkpoints=None",
                              why="full storage keeps every RK stage of the trajectory on the "
                                  "device; the stores apply to Revolve checkpoints",
                              fix="checkpoints=k (Revolve with k slots), e.g. checkpoints=400 "
                                  "with checkpoint_store='host'",
                              doc="docs/site/guide/dynamics/adjoint.md")
        if checkpoint_store == "disk" and checkpoint_dir is None:
            raise OptionError("checkpoint_store='disk' needs checkpoint_dir",
                              fix="FlowMap(..., checkpoint_store='disk', checkpoint_dir='<a "
                                  "directory on the project store>')",
                              doc="docs/site/guide/dynamics/adjoint.md")
        self.checkpoint_store, self.checkpoint_dir = checkpoint_store, checkpoint_dir
        self.last_vjp: dict = {}                 # statistics of the last vjp (store copies, time)
        self.solver, self.nsteps = solver, int(steps)
        self.xp = solver.xp
        self.gpu = self.xp is not np
        self.checkpoints, self.max_dense = checkpoints, int(max_dense)
        self.f_transpose, self.solves = f_transpose, solves
        self.system = solver.system
        self.n = solver.system.n
        self.dtype = solver.system.dtype
        # length of the augmented state (multistep history: SBDF order or CN levels)
        self.Q = max(solver.order, getattr(solver, "_nhist", 0), 1)
        # rows of time-dependent boundary data imposed at stage times (None otherwise)
        self._bc_rows = solver.bc_rows() if hasattr(solver, "bc_rows") else None
        plan = solver.plan
        # complex curvilinear states of real fields: real pairing Re(a^H b) (module docstring)
        self.pairing = "real" if getattr(plan, "is_sphere", False) else "bilinear"
        if getattr(plan, "is_polar", False):
            from ..errors import UnsupportedError
            raise UnsupportedError(
                "FlowMap on a polar/cylindrical plan: the transposed polar evaluator is not written "
                "yet (the spherical shell and Cartesian plans are supported)")
        if self.pairing == "real":
            if f_transpose != "polynomial":
                raise OptionError("FlowMap on the sphere: f_transpose='polynomial' (the matrix-free "
                                  "adjoint of the spherical evaluator) only; no dense oracle",
                                  fix="drop f_transpose= (default 'polynomial')", doc="docs/site/guide/dynamics/adjoint.md")
            from .sphere_rhs import SphereLinearizedRHS
            self.rhs_lin = SphereLinearizedRHS(solver)
        elif f_transpose == "polynomial":
            self.rhs_lin = LinearizedRHS(solver)   # SpectralError if the RHS is not polynomial
        else:
            self.rhs_lin = None
            self._dvars = [E.Field("d" + v.name, v.domain, v.tensorsig, v.dtype)
                           for v in self.system.variables]
            direction = dict(zip(self.system.variables, self._dvars))
            self._lin = []
            for e in solver._rhs_exprs:
                if e is None or not (set(e.free_fields()) & set(self.system.variables)):
                    self._lin.append(None)        # constant RHS (BCs, gauges): no derivative
                else:
                    le = e.linearize(None, direction)
                    self._lin.append(None if le.is_zero() else le)
        # the m = 0 real projection of the solver's step (None elsewhere), applied after
        # the tangent's solves and before the adjoint's transposed solves
        pr = getattr(solver, "_m0proj", None)
        self._m0proj = pr if getattr(pr, "active", False) else None
        self._fac: dict = {}
        self.n_recompute = 0                     # forward steps recomputed by the reverse sweep
        self._steps = self._schedule()

    # ------------------------------------------------------------------ schedule
    def _schedule(self) -> list[_Step]:
        """The exact sequence of steps IVPSolver.run(nsteps) performs from iteration 0."""
        s = self.solver
        out = []
        # start time of step n: t0 + n dt, the expression of IVPSolver.step (t0 exactly at n = 0)
        self._times = [self.t0] + [self.t0 + n * s.dt for n in range(1, self.nsteps + 1)]
        for it in range(self.nsteps):
            t = self._times[it]
            if s.order:
                if s.startup == "rk443" and it < s.order - 1:
                    out.append(_Step("rk", s.dt / s.startup_substeps, rk_tableau("RK443"),
                                     substeps=s.startup_substeps, t=t, n=it))
                else:
                    out.append(_Step("sbdf", s.dt, q=min(s.order, it + 1), t=t, n=it))
            elif getattr(s, "family", "") == "cn":
                if s.startup == "rk443" and it < s._nhist - 1:
                    out.append(_Step("rk", s.dt / s.startup_substeps, rk_tableau("RK443"),
                                     substeps=s.startup_substeps, t=t, n=it))
                else:    # lower-order start-up: CNAB1 until the history is full (IVPSolver._cn_step)
                    co = s._cn if it + 1 >= s._nhist else cn_coefficients("CNAB1")
                    out.append(_Step("cn", s.dt, q=len(co[0]) - 1, coefs=co, t=t, n=it))
            else:
                out.append(_Step("rk", s.dt, s._rk, t=t, n=it))
        return out

    def _hist_time(self, st: _Step, j: int) -> float:
        """Time of hist[j - 1] = X_{n-j+1} for SBDF step n (the time IVPSolver stored with F)."""
        return self._times[st.n - j + 1]

    @staticmethod
    def _sub_times(st: _Step) -> list[float]:
        """Start times of the RK substeps (t += h, as IVPSolver.step accumulates them)."""
        out, t = [], st.t
        for _ in range(st.substeps):
            out.append(t)
            t += st.dt
        return out

    # ------------------------------------------------------------------ primitives
    def _F(self, X: np.ndarray, t: float) -> np.ndarray:
        return self.solver.F(X, t)

    def _JF(self, X: np.ndarray, V: np.ndarray, t: float | None = None) -> np.ndarray:
        """F'(X, t) V (polynomial form, or IR Fréchet derivative with the same evaluator)."""
        if self.tname and t is None:
            raise OptionError(f"time-dependent F ({self.tname}): the stage time t is required",
                              fix="pass the stage time t= (F depends on the time parameter)")
        if self.rhs_lin is not None:
            return self.rhs_lin.apply(X, V, t)
        s, sysm = self.solver, self.system
        data = dict(s._ncc)
        data.update(zip(sysm.variables, sysm.scatter(X)))
        data.update(zip(self._dvars, sysm.scatter(V)))
        params = dict(s.plan.params)
        if self.tname:
            params[self.tname] = t
        out = []
        for le, shape, eq in zip(self._lin, sysm.eq_shapes, s.plan.problem.equations):
            if le is None:
                out.append(None)
                continue
            c = s.evaluator.evaluate(le, data, params)
            out.append(s.evaluator._embed(c, eq.lhs.rank, shape))
        return sysm.gather_rhs(out)

    def _JF_dense(self, X: np.ndarray, t: float | None = None) -> np.ndarray:
        """Dense Jacobian of F at X (columns J e_i); refused above ``max_dense`` unknowns."""
        if self.n > self.max_dense:
            raise OptionError(f"dense F-Jacobian refused: n = {self.n} > max_dense = "
                              f"{self.max_dense}",
                              fix="f_transpose='polynomial' (matrix-free), or raise max_dense= "
                                  "knowingly (memory n^2)", doc="docs/site/guide/dynamics/adjoint.md")
        J = np.empty((self.n, self.n), dtype=self.dtype)
        e = np.zeros(self.n, dtype=self.dtype)
        for i in range(self.n):
            e[i] = 1
            J[:, i] = self._JF(X, e, t)
            e[i] = 0
        return J

    def _JFT(self, X: np.ndarray, W: np.ndarray, t: float | None = None,
             cache: dict | None = None) -> np.ndarray:
        """F'(X, t)ᵀ W."""
        if self.rhs_lin is not None:               # per-step cache of the passive data
            return self.rhs_lin.transpose(X, W, t, cache=cache)
        key = (id(X), t)
        if cache is not None and key in cache:
            JT = cache[key][1]
        else:
            JT = self._JF_dense(X, t).T
            if cache is not None:
                cache[key] = (X, JT)
        return JT @ W

    def _bc(self, rhs, t: float, beta: float):
        """Forward multistep step: time-dependent boundary data imposed at t (the
        solver's own rule, no-op without such data or with bc_time='step')."""
        imp = getattr(self.solver, "impose_bc", None)
        return rhs if imp is None else imp(rhs, t, beta)

    def _zero_bc(self, v):
        """Rows of time-dependent boundary data imposed at stage times: their right-hand
        side beta g(t) does not depend on the state, so they are zero in the tangent's right-hand
        side and the adjoint's multiplier (the transpose of that projection)."""
        rows = self._bc_rows
        if rows is not None:
            v[rows] = 0
        return v

    def pair(self, a, b) -> float:
        """The pairing for which ``vjp`` is the adjoint of ``jvp``: Σ a b (bilinear, real states)
        or Re Σ conj(a) b (real pairing, complex states of real fields on the sphere)."""
        xp = self.xp
        if self.pairing == "real":
            return float(xp.real(xp.vdot(a, b)))
        return float(xp.real(xp.sum(a * b)))

    def _solve(self, a, b, rhs, transpose=False):
        """(a M + b L) x = rhs, or its adjoint for the pairing (plain transpose for the bilinear
        pairing, conjugate transpose for the real pairing), with the solver's own factors
        (banded) or the SuperLU oracle."""
        conj = transpose and self.pairing == "real"
        proj = self._m0proj
        if transpose and proj is not None:
            # the step is P (a M + b L)^-1 with P the m = 0 real projection (real-linear,
            # self-adjoint for Re(a^H b)): its adjoint applies P to the multiplier first
            rhs = proj.apply(rhs.copy())
        if conj:
            rhs = rhs.conj()                       # A^H x = r  <=>  A^T conj(x) = conj(r)
        if self.solves == "banded":
            sysm = self.system
            x = sysm.solve_transpose(a, b, rhs) if transpose else sysm.solve(a, b, rhs)
        else:
            key = (float(a), float(b))
            f = self._fac.get(key)
            if f is None:
                A = (a * self.system.M + b * self.system.L).tocsc().astype(self.dtype)
                f = self._fac[key] = spla.splu(A)
            x = f.solve(np.asarray(rhs, dtype=self.dtype), trans="T" if transpose else "N")
        if not transpose and proj is not None:
            x = proj.apply(x)                      # tangent of P S: P (dS), P real-linear
        return x.conj() if conj else x

    # ------------------------------------------------------------------ forward
    def _rk_stages(self, X0, st: _Step) -> list[list]:
        """Stage states of every substep, from ``IVPSolver.rk_stages`` (same arithmetic)."""
        subs, X = [], X0
        for t in self._sub_times(st):
            stages = self.solver.rk_stages(X, t, st.dt, st.tableau)
            subs.append(stages)
            X = stages[-1]
        return subs

    def _advance(self, hist: list[np.ndarray], st: _Step) -> np.ndarray:
        """X_{n+1} from the history (hist[0] = X_n, hist[1] = X_{n-1}, …)."""
        sysm, X = self.system, hist[0]
        self.n_recompute += 1
        if st.kind == "rk":
            for t in self._sub_times(st):
                X = self.solver.rk_stages(X, t, st.dt, st.tableau)[-1]
            return X
        if st.kind == "cn":
            a, b, c = st.coefs
            rhs = self.xp.zeros(self.n, dtype=self.dtype)
            for j in range(1, st.q + 1):
                if c[j] != 0:
                    rhs += c[j] * self._F(hist[j - 1], self._hist_time(st, j))
                if a[j] != 0:
                    rhs -= (a[j] / st.dt) * (sysm.M @ hist[j - 1])
                if b[j] != 0:
                    rhs -= b[j] * (sysm.L @ hist[j - 1])
            return self.solver._solve_state(a[0] / st.dt, b[0], self._bc(rhs, st.t + st.dt, b[0]))
        a, c = sbdf_coefficients(st.q)
        rhs = self.xp.zeros(self.n, dtype=self.dtype)
        for j in range(1, st.q + 1):
            rhs += (c[j] * self._F(hist[j - 1], self._hist_time(st, j))
                    - (a[j] / st.dt) * (sysm.M @ hist[j - 1]))
        return self.solver._solve_state(a[0] / st.dt, 1.0, self._bc(rhs, st.t + st.dt, 1.0))

    @renames("conatus.adjoint.flowmap:FlowMap.trajectory")
    def trajectory(self, x0: np.ndarray) -> list[np.ndarray]:
        """All states X_0 … X_N."""
        states = [self.xp.asarray(x0, dtype=self.dtype)]
        for st in self._steps:
            states.append(self._advance(states[::-1], st))
        return states

    def __call__(self, X0: np.ndarray) -> np.ndarray:
        """Φ(X_0): the solver run ``nsteps`` steps from X_0 at t0. Its state, clock and
        multistep history are overwritten, unless ``restore`` is True (the maps of
        ``Simulation.flowmap``): they are then put back as they were, also on an error."""
        if not self.restore:
            return self._run_from(X0)
        saved = _solver_state(self.solver)
        try:
            return self._run_from(X0)
        finally:
            _set_solver_state(self.solver, saved)

    def _run_from(self, X0: np.ndarray) -> np.ndarray:
        """Φ(X_0) on the solver itself (its state, clock and history left at the end)."""
        s = self.solver
        s.X = self.xp.array(X0, dtype=self.dtype)
        s.t, s.iteration, s._t_origin, s._it_origin = self.t0, 0, self.t0, 0
        s._MXh.clear()
        s._Fh.clear()
        if getattr(s, "_LXh", None) is not None:
            s._LXh.clear()
        s.run(self.nsteps)
        return s.X.copy()

    # ------------------------------------------------------------------ tangent
    def _rk_jvp(self, subs, dX0, st: _Step):
        """Tangent of one RK step from its recorded stage states ``subs``."""
        A, H, c = st.tableau
        sysm, dX = self.system, dX0
        for stages, ts in zip(subs, self._sub_times(st)):
            dS = [dX]
            JF: dict = {}                  # F'(S_k) dS_k, computed once per stage k (it does not
            MdX0 = sysm.M @ dX             # depend on i: same numbers, s instead of s(s+1)/2 calls)
            for i in range(1, len(c)):
                rhs = MdX0.copy()
                for k in range(i):
                    if A[i][k] != 0:
                        if k not in JF:
                            JF[k] = self._JF(stages[k], dS[k], ts + c[k] * st.dt)
                        rhs += (st.dt * A[i][k]) * JF[k]
                    if H[i][k] != 0:
                        rhs -= (st.dt * H[i][k]) * (sysm.L @ dS[k])
                dS.append(self._solve(1.0, st.dt * H[i][i], self._zero_bc(rhs)))
            dX = dS[-1]
        return dX

    def _forward(self, hist: list, n: int):
        """Step n from the augmented state: (next augmented state, RK stage record or None).
        The RK stages are those of IVPSolver._rk_step (same arithmetic, identical last stage)."""
        st = self._steps[n]
        if st.kind == "rk":
            self.n_recompute += 1
            subs = self._rk_stages(hist[0], st)
            return [subs[-1][-1]] + hist[:self.Q - 1], subs
        return self._next_hist(hist, n), None

    @renames("conatus.adjoint.flowmap:FlowMap.jvp")
    def jvp(self, x0: np.ndarray, V: np.ndarray) -> np.ndarray:
        """J V with J = dΦ/dX_0 at X_0 (exact discrete tangent), one forward pass."""
        xp = self.xp
        hist = [xp.asarray(x0, dtype=self.dtype)] + [None] * (self.Q - 1)
        dhist = [xp.asarray(V, dtype=self.dtype)] + [None] * (self.Q - 1)
        sysm = self.system
        for n, st in enumerate(self._steps):
            nxt, subs = self._forward(hist, n)
            if st.kind == "rk":
                dX = self._rk_jvp(subs, dhist[0], st)
            elif st.kind == "cn":
                a, b, c = st.coefs
                rhs = xp.zeros(self.n, dtype=self.dtype)
                for j in range(1, st.q + 1):
                    if c[j] != 0:
                        rhs += c[j] * self._JF(hist[j - 1], dhist[j - 1], self._hist_time(st, j))
                    if a[j] != 0:
                        rhs -= (a[j] / st.dt) * (sysm.M @ dhist[j - 1])
                    if b[j] != 0:
                        rhs -= b[j] * (sysm.L @ dhist[j - 1])
                dX = self._solve(a[0] / st.dt, b[0], self._zero_bc(rhs))
            else:
                a, c = sbdf_coefficients(st.q)
                rhs = xp.zeros(self.n, dtype=self.dtype)
                for j in range(1, st.q + 1):
                    rhs += (c[j] * self._JF(hist[j - 1], dhist[j - 1], self._hist_time(st, j))
                            - (a[j] / st.dt) * (sysm.M @ dhist[j - 1]))
                dX = self._solve(a[0] / st.dt, 1.0, self._zero_bc(rhs))
            hist, dhist = nxt, [dX] + dhist[:self.Q - 1]
        return dhist[0]

    # ------------------------------------------------------------------ adjoint
    def _rk_vjp(self, X0, W, st: _Step, subs=None):
        """Adjoint of one RK step (all substeps) at X0 applied to the cotangent W; the stage
        states are recomputed unless given (full storage records them in the forward pass)."""
        A, H, c = st.tableau
        MT, LT = self._MT, self._LT
        if subs is None:
            subs = self._rk_stages(X0, st)
        lam = W
        for stages, ts in zip(reversed(subs), reversed(self._sub_times(st))):
            s = len(c) - 1
            bar = [self.xp.zeros(self.n, dtype=self.dtype) for _ in range(s + 1)]
            bar[s] = bar[s] + lam
            # the cotangents sent to stage k through F'(S_k)ᵀ are summed first (acc[k] =
            # Σ_i dt A[i][k] μ_i) and transposed once when stage k is reached: F'ᵀ is linear, so
            # this equals Σ_i F'ᵀ(dt A[i][k] μ_i) up to rounding, with s instead of s(s+1)/2 calls
            acc: dict = {}

            def flush(k, bar=bar, acc=acc, stages=stages, ts=ts):
                # each stage is transposed exactly once: its factor grids are dropped as soon as
                # F'(S_k)ᵀ is applied (a per-call cache), instead of being kept for every stage of
                # the step (same arithmetic, bit-identical: 2/3 of that memory at RK443)
                if k in acc:
                    bar[k] += self._JFT(stages[k], acc.pop(k), ts + c[k] * st.dt, {})
            for i in range(s, 0, -1):
                flush(i)
                mu = self._zero_bc(self._solve(1.0, st.dt * H[i][i], bar[i], transpose=True))
                bar[0] += MT @ mu
                for k in range(i):
                    if A[i][k] != 0:
                        a = (st.dt * A[i][k]) * mu
                        acc[k] = a if k not in acc else acc[k] + a
                    if H[i][k] != 0:
                        bar[k] -= (st.dt * H[i][k]) * (LT @ mu)
            flush(0)
            lam = bar[0]
        return lam

    def _step_vjp(self, hist: list, bars: list, n: int, subs=None) -> list:
        """Adjoint of step n on the augmented state.

        hist = (X_n, X_{n-1}, …) (length Q, None before X_0); bars = cotangents of
        (X_{n+1}, X_n, …, X_{n+2-Q}). Returns the cotangents of (X_n, …, X_{n+1-Q})."""
        st = self._steps[n]
        lam = bars[0]
        out = [bars[i + 1] if i + 1 < self.Q else self.xp.zeros(self.n, dtype=self.dtype)
               for i in range(self.Q)]
        out = [o.copy() for o in out]
        if st.kind == "rk":
            out[0] += self._rk_vjp(hist[0], lam, st, subs)
            return out
        if st.kind == "cn":
            a, b, c = st.coefs
            mu = self._zero_bc(self._solve(a[0] / st.dt, b[0], lam, transpose=True))
            cache: dict = {}
            for j in range(1, st.q + 1):
                if c[j] != 0:
                    out[j - 1] += c[j] * self._JFT(hist[j - 1], mu, self._hist_time(st, j), cache)
                if a[j] != 0:
                    out[j - 1] -= (a[j] / st.dt) * (self._MT @ mu)
                if b[j] != 0:
                    out[j - 1] -= b[j] * (self._LT @ mu)
            return out
        a, c = sbdf_coefficients(st.q)
        mu = self._zero_bc(self._solve(a[0] / st.dt, 1.0, lam, transpose=True))
        MTmu = self._MT @ mu
        cache: dict = {}
        for j in range(1, st.q + 1):
            out[j - 1] += (c[j] * self._JFT(hist[j - 1], mu, self._hist_time(st, j), cache)
                           - (a[j] / st.dt) * MTmu)
        return out

    def _next_hist(self, hist: list, n: int) -> list:
        """Augmented state after step n."""
        return [self._advance(hist, self._steps[n])] + hist[:self.Q - 1]

    @renames("conatus.adjoint.flowmap:FlowMap.vjp")
    def vjp(self, x0: np.ndarray, W: np.ndarray | None, *, journal=None,
            budget_s: float | None = None, extra: dict | None = None) -> np.ndarray:
        """Jᵀ W (bilinear transpose, exact discrete adjoint), states from checkpoints.

        ``journal`` (a directory, Revolve only): if it holds a suspended sweep of this very
        computation (same x0, W and map, checked by digest), the sweep resumes from it (``W`` may
        then be None); ``budget_s``: seconds after which the sweep stops at a step boundary,
        writes the journal and raises :class:`conatus.adjoint.checkpoints.VJPSuspended`.
        ``extra``: JSON-ready data saved with the journal (``journal_extra(journal)`` reads it).
        Without ``journal`` nothing is written and ``budget_s`` is refused."""
        xp = self.xp
        if (journal is not None or budget_s is not None) and self.checkpoints is None:
            raise OptionError("vjp(journal=/budget_s=) needs Revolve (checkpoints=k)",
                              why="with full storage the whole trajectory is in device memory: "
                                  "there is no small state to write and resume",
                              fix="FlowMap(..., checkpoints=k, checkpoint_store='disk' or 'host')",
                              doc="docs/site/guide/dynamics/adjoint.md")
        if budget_s is not None and journal is None:
            raise OptionError("vjp(budget_s=...) without journal=",
                              why="a suspended sweep must be written somewhere to be resumed",
                              fix="vjp(x0, W, journal='<directory>', budget_s=...)",
                              doc="docs/site/guide/dynamics/adjoint.md")
        if W is None and journal is None:
            raise OptionError("vjp: W is None", fix="pass the cotangent W (W may be None only "
                              "when resuming from a journal)", doc="docs/site/guide/dynamics/adjoint.md")
        if W is not None:
            W = xp.asarray(W, dtype=self.dtype)
        if not hasattr(self, "_MT"):
            host = self.system.host if self.gpu else self.system
            MT, LT = host.M.T.tocsr(), host.L.T.tocsr()
            if self.pairing == "real":             # adjoints for Re(a^H b): conjugate transposes
                MT, LT = MT.conj().tocsr(), LT.conj().tocsr()
            if self.gpu:
                from ..timestep.device import DeviceCsr
                MT, LT = DeviceCsr(MT, self.dtype), DeviceCsr(LT, self.dtype)
            self._MT, self._LT = MT, LT
        x0 = xp.asarray(x0, dtype=self.dtype)
        hist0 = [x0] + [None] * (self.Q - 1)
        self.n_recompute = 0
        if self.checkpoints is not None:
            return self._revolve_run(hist0, W, journal, budget_s, extra)[0]
        bars = [W.copy()] + [xp.zeros(self.n, dtype=self.dtype) for _ in range(self.Q - 1)]
        hists, recs = [hist0], []
        for n in range(self.nsteps):                   # one forward pass, RK stages recorded
            nxt, subs = self._forward(hists[-1], n)
            recs.append(subs)
            if n < self.nsteps - 1:
                hists.append(nxt)
        for n in range(self.nsteps - 1, -1, -1):
            bars = self._step_vjp(hists[n], bars, n, recs[n])
        return bars[0]

    def _revolve_recursive(self, hist, start, stop, bars, c):
        """Reverse steps [start, stop) from the stored augmented state with c free checkpoints:
        the recursive form of the schedule, kept as the oracle of :meth:`_revolve_run` (tests
        check that both give the same bits)."""
        if stop - start == 1:
            return self._step_vjp(hist, bars, start)
        m = revolve_schedule(stop - start, c)
        Y = hist
        for n in range(start, start + m):             # recompute forward to the split
            Y = self._next_hist(Y, n)
        bars = self._revolve_recursive(Y, start + m, stop, bars, c - 1)
        return self._revolve_recursive(hist, start, start + m, bars, c)

    def _fingerprint(self, X0, W) -> dict:
        """What identifies a reverse sweep (journal): the map, the backend, a digest of the
        operators and parameters, and digests of X0 and W. The store is not part of it (the stores
        give the same bits: a sweep may change store on resume). The operator digest matters
        because a resumed sweep gets W = None: without it, a job resumed with another Re (same
        grid, same X0) would mix two different adjoints silently; the backend,
        because CPU and GPU arithmetic differ in the last bits."""
        from .checkpoints import array_digest
        s = self.solver
        fp = dict(nsteps=self.nsteps, checkpoints=self.checkpoints, dt=float(s.dt), t0=self.t0,
                  scheme=s.scheme, n=int(self.n), dtype=str(np.dtype(self.dtype)), Q=self.Q,
                  backend=str(s.backend), operators=self._operator_digest(),
                  X0=array_digest(X0), W=None if W is None else array_digest(W))
        sf = getattr(s, "_sfused", None)
        if sf is not None:
            # planned spherical F (B2): its F and tangent differ from the reference evaluator's
            # in the last bits, and so does each m-chunking of its GEMMs; a sweep resumed with
            # another f_backend would mix them. Only on fused solvers, so the
            # journals of every other map keep their fingerprint.
            fp.update(f_backend="fused", ang_chunks=int(sf.ev.S.nchunk))
            pl = getattr(self.rhs_lin, "_planned", None)
            if pl is not None and pl.planned:
                # planned adjoint: same adjoint as the explicit walk up to rounding, not
                # to the bit; a sweep journaled by the explicit walk is refused, not mixed
                fp.update(adjoint="planned")
        if self._m0proj is not None:
            # the projected step differs from the unprojected one in the last bits; a
            # journal of the unprojected map is refused (only curvilinear maps carry the key)
            fp.update(m0_projection=True)
        fp.update(_lever_keys(s))
        return fp

    def _operator_digest(self) -> str:
        """SHA-256 of M and L (host CSR) and of the plan parameters (computed once per map)."""
        if getattr(self, "_opdigest", None) is None:
            import hashlib
            h = hashlib.sha256()
            host = self.system.host if self.gpu else self.system
            for A in (host.M, host.L):
                A = A.tocsr()
                for a in (A.indptr, A.indices, A.data):
                    h.update(np.ascontiguousarray(a).tobytes())
            for k, v in sorted(dict(getattr(self.solver.plan, "params", {}) or {}).items()):
                h.update(repr(k).encode())
                h.update(np.ascontiguousarray(np.asarray(v)).tobytes()
                         if isinstance(v, np.ndarray) else repr(v).encode())
            self._opdigest = h.hexdigest()
        return self._opdigest

    def _revolve_run(self, hist0, W, journal, budget_s, extra):
        """Binomial Revolve as an explicit action list (:func:`revolve_actions`, the same order
        and arithmetic as the recursion), checkpoints in the chosen store, optional journal."""
        from .checkpoints import Journal, make_store
        import time as _time
        xp = self.xp
        t_start = _time.perf_counter()
        deadline = None if budget_s is None else t_start + float(budget_s)
        acts = revolve_actions(self.nsteps, self.checkpoints)
        store = make_store(self.checkpoint_store, xp, self.checkpoint_dir)
        jr = None if journal is None else Journal(journal, self._fingerprint(hist0[0], W))
        st = None if jr is None else jr.load()
        cursor, progress, work = 0, None, None
        if st is None:
            if W is None:
                raise OptionError(f"vjp: W is None and the journal {journal} holds no suspended "
                                  "sweep", fix="pass the cotangent W", doc="docs/site/guide/dynamics/adjoint.md")
            bars = [W.copy()] + [xp.zeros(self.n, dtype=self.dtype) for _ in range(self.Q - 1)]
        try:
            if st is not None:                         # resume: exact bits from the journal
                cursor, progress = st["cursor"], st["progress"]
                bars = [None if b is None else xp.asarray(b) for b in jr.read_arrays(st["bars"])]
                if st["work"] is not None:
                    work = [None if a is None else xp.asarray(a)
                            for a in jr.read_arrays(st["work"])]
                for k, names in st["checkpoints"].items():
                    if store.kind == "disk":           # hard links back, no rewrite
                        store.adopt(int(k), [None if m is None else jr.dir / m for m in names])
                    else:
                        store.put_host(int(k), jr.read_arrays(names))
                if extra is None:
                    extra = st["extra"]
            return self._revolve_loop(acts, store, jr, hist0, bars, cursor, progress, work,
                                      deadline, budget_s, journal, extra, st, t_start)
        finally:
            # the store is never needed after this call: a suspended sweep lives in its journal
            # (hard links for the disk store, copies otherwise). Clearing it here releases pinned
            # host buffers and removes disk-store files also on a suspension or an error (they
            # used to be left behind in checkpoint_dir)
            store.clear()

    def _revolve_loop(self, acts, store, jr, hist0, bars, cursor, progress, work, deadline,
                      budget_s, journal, extra, st, t_start):
        """The action loop of :meth:`_revolve_run` (the store is cleared by the caller)."""
        from .checkpoints import VJPSuspended
        import time as _time
        store.put(0, hist0)                            # the input X_0 (from the caller)
        hot = (0, hist0)                               # last state produced on the device
        done = 0                                       # steps taken by this call: the budget is
        #   checked only once one step is done, so every call progresses (a budget already spent,
        #   e.g. after the forward pass of NLOP.gradient, cannot loop on empty suspensions)

        def late():
            return deadline is not None and done > 0 and _time.perf_counter() > deadline

        def fetch(k):
            return hot[1] if hot[0] == k else store.get(k)

        def suspend(i, prog, Y):
            pos = dict(cursor=i, actions=len(acts), progress=prog,
                       held=[k for k in store.held() if k != 0])
            jr.save(cursor=i, progress=prog, bars=bars, work=Y, store=_Without0(store),
                    extra=extra)
            self.last_vjp = dict(store=store.stats, suspended=pos,
                                 seconds=_time.perf_counter() - t_start)
            raise VJPSuspended(f"vjp suspended at action {i}/{len(acts)} (budget {budget_s} s); "
                               f"journal written to {journal}", journal=str(journal), position=pos)

        for i in range(cursor, len(acts)):
            act = acts[i]
            if act[0] == "advance":
                _, a, b = act
                if i == cursor and work is not None:   # resume inside this forward recomputation
                    Y, k0 = work, progress
                else:
                    Y, k0 = fetch(a), a
                for n in range(k0, b):
                    if late():
                        suspend(i, n, Y)
                    Y = self._next_hist(Y, n)
                    done += 1
                store.put(b, Y)
                hot = (b, Y)
            else:
                n = act[1]
                if late():
                    suspend(i, None, None)
                bars = self._step_vjp(fetch(n), bars, n)
                done += 1
                if n > 0:
                    store.drop(n)
                if hot[0] == n:
                    hot = (None, None)
        if jr is not None:
            jr.clear()
        self.last_vjp = dict(store=store.stats, seconds=_time.perf_counter() - t_start,
                             resumed_from=cursor if st is not None else None)
        return bars

    # ------------------------------------------------------------------ physical adjoint
    @renames("conatus.adjoint.flowmap:FlowMap.vjp_physical")
    def vjp_physical(self, x0: np.ndarray, W: np.ndarray, weight) -> np.ndarray:
        """Adjoint for the inner product ⟨a, b⟩_G = aᵀ G b: Φ† w = G⁻¹ Jᵀ G w.
        Refused if G is singular (e.g. tau rows carry no weight). CPU backend only."""
        if self.gpu:
            raise OptionError("vjp_physical: CPU backend only for now",
                              fix="build the FlowMap on a CPU solver for the physical adjoint", doc="docs/site/guide/dynamics/adjoint.md")
        G = sp.csc_matrix(weight, dtype=self.dtype)
        try:
            lu = spla.splu(G)
        except RuntimeError as err:
            raise OptionError(f"weight matrix singular: physical adjoint undefined ({err})",
                              fix="W must be the (symmetric positive definite) Gram matrix of the inner product") from err
        d = lu.U.diagonal()
        if np.min(np.abs(d)) <= 1e-14 * np.max(np.abs(d)):
            raise OptionError("weight matrix numerically singular: physical adjoint refused",
                              fix="W must be the (symmetric positive definite) Gram matrix of the inner product")
        return lu.solve(self.vjp(x0, G @ np.asarray(W, dtype=self.dtype)))

    # ------------------------------------------------------------------ explain
    def explain(self) -> str:
        s = self.solver
        if self.checkpoints is None:
            ck = "full storage of the trajectory (device)"
        else:
            cost = revolve_cost(self.nsteps, self.checkpoints)
            where = {"device": "device memory (references to the state arrays; default)",
                     "host": "host memory (pinned buffers on the GPU, copies at every store and "
                             "fetch)",
                     "disk": f"disk, one .npy file per array in {self.checkpoint_dir} (atomic "
                             "writes)"}[self.checkpoint_store]   # no store built: explain() has
            #   no side effect (make_store('disk') creates the directory)
            ck = (f"Revolve (binomial, explicit action list), {self.checkpoints} checkpoint "
                  f"slot(s), augmented state of length {self.Q}; {cost['advances']} forward steps "
                  f"recomputed ({cost['sweeps']:.2f} x nsteps, plus the stages of each reversed "
                  f"step); checkpoints in {where}; vjp(journal=, budget_s=) suspends and resumes "
                  "the sweep bit for bit")
        fl = (self.rhs_lin.explain() if self.pairing == "real" else
              "polynomial form, F' and F'ᵀ matrix-free (explicit transposes of the dealiased "
              "transforms and derivatives)" if self.rhs_lin is not None else
              f"IR Fréchet derivative; transpose from dense Jacobian (n = {self.n}, max_dense = "
              f"{self.max_dense})")
        sv = ("batched bordered-banded factors of IVPSolver (C++), solve for J and solve_transpose "
              "for Jᵀ (same factors)"
              if self.solves == "banded" else "SuperLU of a M + b L (same factor for J and Jᵀ)")
        if self.gpu:
            sv = ("GPU: batched device factors of IVPSolver, solve for J and solve_transpose "
                  "for Jᵀ (same factors, no host copy)")
        tm = (f", from t0 = {self.t0!r} ({self.tname}: F evaluated at the stage times"
              + (f"; time-dependent boundary data imposed at the stage times "
                 f"(bc_time={s.bc_time!r}), zero rows in the tangent and the adjoint"
                 if self._bc_rows is not None else "") + ")"
              if self.tname else "")
        return (f"FlowMap(Φ: X_0 -> X_{self.nsteps}, T = {self.nsteps * s.dt:g}{tm})\n"
                f"  steps: {s.scheme}, dt = {s.dt:g}"
                + (f", startup = {s.startup}" if s.order or getattr(s, "family", "") == "cn" else "")
                + "\n"
                + ("  tangent/adjoint: exact discrete (same schemes, same operators), real pairing "
                   "⟨a, b⟩ = Re Σ conj(a) b (complex state of real fields; conjugate-transposed "
                   "solves); physical adjoint via vjp_physical(G)\n" if self.pairing == "real" else
                   "  tangent/adjoint: exact discrete (same schemes, same operators), bilinear "
                   "pairing ⟨a, b⟩ = Σ a b (D8); physical adjoint via vjp_physical(G)\n") +
                f"  linear solves in jvp/vjp: {sv}\n"
                + ("  m = 0 real projection P after every solve of the step: P after the "
                   "tangent's solves, P before the adjoint's transposed solves (self-adjoint for "
                   "the real pairing)\n" if self._m0proj is not None else "")
                + f"  F linearization: {fl}\n"
                f"  checkpointing: {ck}\n"
                + ("  Φ(X_0) runs the solver and puts its state, clock and history back"
                   if self.restore else
                   "  Φ(X_0) runs the solver: its state, clock and history are left at X_N "
                   "(restore=True keeps them)"))


class _Without0:
    """View of a checkpoint store without the input checkpoint 0 (the journal does not copy X_0:
    the caller passes it again, checked by digest)."""

    def __init__(self, store) -> None:
        self.s = store

    def held(self) -> list[int]:
        return [k for k in self.s.held() if k != 0]

    def files(self, k: int):
        return self.s.files(k)

    def to_host(self, k: int) -> list:
        return self.s.to_host(k)


def journal_extra(journal) -> dict | None:
    """The ``extra`` data saved with a suspended vjp in ``journal`` (None if none is pending)."""
    import json
    from pathlib import Path
    p = Path(journal) / "state.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("extra") or {}
