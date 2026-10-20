"""Lyapunov spectrum (Benettin / QR), covariant Lyapunov vectors (Ginelli et al. 2007),
Kaplan–Yorke dimension and finite-time exponents of a discretized IVP.
Notes:

What is computed is the spectrum of the **discrete map** of the solver (same scheme, same dt):
the tangent is the exact derivative of the step (``chaos.tangent``), not a separate discretization
of the variational equation. The exponents of the map converge to those of the flow as dt -> 0
(RK443: observed order reported by the caller from two runs at dt and dt/2).

Algorithm (Benettin et al. 1980; Shimada & Nagashima 1979): k orthonormal tangent vectors per
trajectory are advanced with the trajectory; every ``reortho_every`` steps V = Q R (QR per
trajectory, diag R > 0) and log R_ii is accumulated. λ_i = (Σ log R_ii) / T. B independent
trajectories run in one batch (``conatus.ensembles``), so the statistical error is estimated from
B x ``n_blocks`` block means (standard error of the mean, blocks assumed longer than the
correlation time: the lag-1 correlation of the block means is reported to check it).

Covariant vectors (Ginelli, Poggi, Turchi, Chaté, Livi & Politi, PRL 99, 130601, 2007): with
J_j Q_j = Q_{j+1} R_{j+1} stored over a window, C_j = R_{j+1}^{-1} C_{j+1} (columns normalized),
backwards from a random upper-triangular C, and γ_j = Q_j C_j. The first
``clv_backward_transient`` intervals of the backward pass (the end of the window) are discarded.

Every choice is an argument (no defaults for lengths, vector count, inner product or seed):
``explain()`` of the result lists them with the statistical error of every exponent.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..ensembles import Ensemble
from ..errors import ConvergenceError, OptionError
from .tangent import require_rk, rk_tangent_step
from .._deprecation import renames
from .._repr import compact_repr

__all__ = ["lyapunov_spectrum", "LyapunovResult", "Domination", "kaplan_yorke",
           "clv_subspace_angles"]


def kaplan_yorke(exponents) -> float:
    """D_KY = j + (λ_1 + … + λ_j) / |λ_{j+1}|, j the largest index with a non-negative partial
    sum (exponents sorted in decreasing order). Refused (ConvergenceError) if every partial sum of
    the given exponents is non-negative: more exponents are needed, never extrapolated."""
    lam = np.asarray(exponents, float)
    if lam.ndim != 1 or lam.size == 0 or not np.all(np.isfinite(lam)):
        # audit inputs: a NaN exponent was sorted last and ended in a bare IndexError,
        # a 2D array was sorted along its last axis only
        from ..errors import InputTypeError
        raise InputTypeError(f"kaplan_yorke: a 1D array of finite exponents, got shape {lam.shape}"
                             + ("" if lam.ndim != 1 or lam.size == 0 else " with NaN/inf"),
                             fix="the exponents of a LyapunovResult (res.exponents)")
    lam = np.sort(lam)[::-1]
    S = np.cumsum(lam)
    if S[-1] >= 0:
        raise ConvergenceError(
            f"Kaplan–Yorke dimension undefined with these {lam.size} exponents: every partial "
            f"sum is >= 0 (last {S[-1]:.4g})",
            fix="compute more Lyapunov vectors (n_vectors) until the partial sum turns negative")
    if S[0] < 0:
        return 0.0
    j = int(np.nonzero(S >= 0)[0][-1]) + 1            # number of exponents in the positive sum
    return j + S[j - 1] / abs(lam[j])


def clv_subspace_angles(clv, j: int, weight=None) -> np.ndarray:
    """Smallest principal angle between span(γ_1..γ_j) and span(γ_{j+1}..γ_k) for every sample.
    ``clv``: (..., k, n) covariant vectors; ``weight``: diagonal inner-product weight (n,) or None
    (Euclidean). A vanishing angle is a tangency (no domination between the two groups)."""
    V = np.asarray(clv)
    if weight is not None:
        V = V * np.sqrt(np.asarray(weight))
    k = V.shape[-2]
    if not 1 <= j < k:
        raise OptionError(f"clv_subspace_angles: need 1 <= j < k = {k}, got {j}")
    A = np.swapaxes(V[..., :j, :], -1, -2)
    Bm = np.swapaxes(V[..., j:, :], -1, -2)
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(Bm)
    s = np.linalg.svd(np.swapaxes(Qa, -1, -2) @ Qb, compute_uv=False)
    return np.arccos(np.clip(s[..., 0], -1.0, 1.0))


@dataclass
class LyapunovResult:
    """Exponents, their statistical errors, the log-growth record and (optionally) the CLVs."""
    __repr__ = compact_repr   # one line, arrays by shape
    exponents: np.ndarray                 # (k,) mean over trajectories and time
    stderr: np.ndarray                    # (k,) standard error from block means
    per_trajectory: np.ndarray            # (B, k)
    block_means: np.ndarray               # (B * n_blocks, k)
    block_lag1: np.ndarray                # (k,) lag-1 correlation of consecutive block means
    logs: np.ndarray                      # (n_intervals, B, k): log R_ii per interval
    interval: float                       # time between reorthonormalizations
    settings: dict
    clv: np.ndarray | None = None         # (B, n_kept, k, n)
    clv_states: np.ndarray | None = None  # (B, n_kept, n) trajectory states at the CLV times
    clv_times: np.ndarray | None = None   # (n_kept,)
    clv_exponents: np.ndarray | None = None  # (k,) mean growth rates of the CLVs over the window
    notes: list = field(default_factory=list)
    # (additions): per-interval log growth of every CLV over the window (forward time
    # order; local exponents for the domination diagnostics), the memory used by the covariant
    # pass, the recomputation check of the checkpointed pass, and the covariance residual
    clv_logs: np.ndarray | None = None    # (n_window, B, k)
    clv_memory: str | None = None
    clv_recompute_deviation: float | None = None
    clv_covariance_residual: np.ndarray | None = None   # (k,) largest sine, see explain()
    clv_backward_convergence: np.ndarray | None = None  # (k,) largest sine between two starts
    clv_covariance_floor: np.ndarray | None = None     # (k,) rounding floor of the residual
    clv_split_convergence: np.ndarray | None = None    # (k-1,) subspace convergence per split
    inner_weight: np.ndarray | None = field(default=None, repr=False)   # diagonal weight or None
    _grid: Any = field(default=None, repr=False)   # vector -> grid values (native sim.lyapunov)

    @property
    def T(self) -> float:
        """Averaging time per trajectory."""
        return self.logs.shape[0] * self.interval

    def kaplan_yorke(self) -> tuple[float, float]:
        """(D_KY of the mean exponents, standard error from the block estimates)."""
        D = kaplan_yorke(self.exponents)
        blocks = []
        for lam in self.block_means:
            try:
                blocks.append(kaplan_yorke(lam))
            except ConvergenceError:
                blocks.append(np.nan)
        blocks = np.asarray(blocks)
        ok = blocks[np.isfinite(blocks)]
        err = float(np.std(ok, ddof=1) / np.sqrt(ok.size)) if ok.size > 1 else float("nan")
        return D, err

    def ftle(self, window: int) -> np.ndarray:
        """Finite-time exponents over ``window`` consecutive intervals (non-overlapping), all
        trajectories: array (samples, k)."""
        n = self.logs.shape[0] // window
        if n < 1:
            raise OptionError(f"ftle: window {window} longer than the record ({self.logs.shape[0]})")
        L = self.logs[:n * window].reshape(n, window, *self.logs.shape[1:]).sum(axis=1)
        return (L / (window * self.interval)).reshape(-1, self.logs.shape[-1])

    def _need_clv(self, what: str) -> None:
        if self.clv is None:
            raise OptionError(f"{what}: no covariant vectors in this result",
                              why="the CLVs are computed only over a declared window",
                              fix="clv_window=... and clv_backward_transient=... in "
                                  "lyapunov_spectrum (covariant=True with clv_window/clv_backward "
                                  "in sim.lyapunov)")

    def angles(self, j: int) -> np.ndarray:
        """Smallest principal angle between span(γ_1..γ_j) and span(γ_j+1..γ_k) at every kept
        time, (B, n_kept), in the declared inner product (``clv_subspace_angles``): a vanishing
        angle is a tangency between the two groups (no hyperbolic splitting there)."""
        self._need_clv("angles")
        return clv_subspace_angles(self.clv, j, self.inner_weight)

    def pair_angle(self, i: int, j: int) -> np.ndarray:
        """Angle in [0, pi/2] between the covariant vectors γ_i and γ_j (1-based) at every kept
        time, (B, n_kept), in the declared inner product."""
        self._need_clv("pair_angle")
        k = self.clv.shape[2]
        if not (1 <= i <= k and 1 <= j <= k) or i == j:
            raise OptionError(f"pair_angle({i}, {j}): two different indices in 1..{k}")
        a, b = self.clv[:, :, i - 1], self.clv[:, :, j - 1]
        if self.inner_weight is not None:
            sq = np.sqrt(self.inner_weight)
            a, b = a * sq, b * sq
        c = np.abs((a * b).sum(-1)) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
        return np.arctan2(_sine(np, a, b, axis=-1), c)       # accurate near 0 and pi/2

    def clv_grid(self, name: str, i: int, sample: int = 0, trajectory: int = 0) -> np.ndarray:
        """Grid values of unknown ``name`` in the covariant vector γ_i (1-based) at kept time
        ``sample`` of trajectory ``trajectory``, in the layout of ``sim.grid(name)``: results of
        the native ``sim.lyapunov`` (an expert result holds state vectors only: ``clv``)."""
        self._need_clv("clv_grid")
        if self._grid is None:
            raise OptionError("clv_grid: this result comes from the expert lyapunov_spectrum",
                              why="the layout of the state vector is known to the simulation",
                              fix="sim.lyapunov(...) on a native simulation, or scatter "
                                  "res.clv[b, s, i - 1] with solver.system.scatter")
        k = self.clv.shape[2]
        if not 1 <= i <= k:
            raise OptionError(f"clv_grid({name!r}, {i}): i in 1..{k}")
        return self._grid(name, self.clv[trajectory, sample, i - 1])

    def domination(self, tau: float) -> Domination:
        """Domination of the Oseledets splitting over time windows of length ``tau``: the local exponents Λ_i(t, τ) (mean growth rate of γ_i over [t, t + τ], from the
        backward pass) on non-overlapping windows of all trajectories, and the fraction
        ν_ij = P(Λ_i < Λ_j), i < j, of windows where the ordering of the exponents is violated
        (Pazó, Szendro, López & Rodríguez 2008; Yang, Takeuchi, Ginelli, Chaté & Radons 2009;
        Takeuchi, Yang, Ginelli, Radons & Chaté 2011). ``tau`` is a whole number of intervals."""
        self._need_clv("domination")
        from ..timestep.integrator import is_whole
        m = tau / self.interval
        if not (m >= 1 and is_whole(m)):
            raise OptionError(f"domination(tau={tau!r}): tau must be a whole number (>= 1) of "
                              f"reorthonormalization intervals ({self.interval:g})",
                              fix=f"tau = {max(1, int(round(m))) * self.interval!r}")
        m = int(round(m))
        L = self.clv_logs
        nw = L.shape[0] // m
        if nw < 1:
            raise OptionError(f"domination(tau={tau!r}): longer than the CLV window "
                              f"({L.shape[0] * self.interval:g})")
        loc = (L[:nw * m].reshape(nw, m, *L.shape[1:]).sum(axis=1) / (m * self.interval))
        loc = loc.reshape(-1, L.shape[-1])                    # (windows x B, k)
        d = Domination.from_local(loc, float(tau), self.clv, self.inner_weight)
        d.split_convergence = self.clv_split_convergence
        return d

    def explain(self) -> str:
        s = self.settings
        lines = [f"Lyapunov spectrum of the discrete map ({s['scheme']}, dt = {s['dt']:g}, "
                 f"backend = {s['backend']}): {s['n_vectors']} vector(s), {s['trajectories']} "
                 f"trajectory(ies)",
                 f"  transient {s['transient_steps']} steps, tangent transient "
                 f"{s['tangent_transient_steps']} steps, average over T = {self.T:g} per trajectory "
                 f"({self.logs.shape[0]} QR every {s['reortho_every']} steps = {self.interval:g})",
                 f"  inner product: {s['inner']}; projection: {s['project']}; seed {s['seed']}",
                 f"  error: standard error of {self.block_means.shape[0]} block means "
                 f"({s['n_blocks']} per trajectory), max |lag-1 corr| = "
                 f"{np.nanmax(np.abs(self.block_lag1)):.2f}"]
        for i, (l, e) in enumerate(zip(self.exponents, self.stderr, strict=True)):
            extra = "" if self.clv_exponents is None else f"   (CLV growth {self.clv_exponents[i]:+.5f})"
            lines.append(f"  λ{i + 1:<3d} = {l:+.5f} ± {e:.5f}{extra}")
        lines.append(f"  sum = {self.exponents.sum():+.5f}")
        try:
            D, dD = self.kaplan_yorke()
            lines.append(f"  Kaplan–Yorke dimension = {D:.4f} ± {dD:.4f}")
        except ConvergenceError as err:
            lines.append(f"  Kaplan–Yorke dimension: undefined ({err.message})")
        if self.clv is not None:
            lines.append(f"  covariant vectors (Ginelli et al. 2007): {self.clv.shape[1]} kept "
                         f"time(s) x {self.clv.shape[0]} trajectory(ies), window "
                         f"{s['clv_window']} interval(s) after a backward transient of "
                         f"{s['clv_backward_transient']}, kept every "
                         f"{s.get('clv_keep_every', 1)}")
            if self.clv_memory:
                lines.append(f"  memory: {self.clv_memory}")
            if self.clv_recompute_deviation is not None:
                lines.append(f"  recomputed R vs stored R: max relative difference "
                             f"{self.clv_recompute_deviation:.2e}")
            if self.clv_backward_convergence is not None:
                c = self.clv_backward_convergence
                lines.append(f"  backward convergence (sine of the angle between the CLVs of two "
                             f"independent random backward starts): max {c.max():.2e} (worst "
                             f"vector {int(c.argmax()) + 1}); slow for nearly equal exponents "
                             "(rate λ_i-1 - λ_i): lengthen clv_backward if it matters")
                sc = self.clv_split_convergence
                if sc is not None and sc.size:
                    lines.append("  subspace convergence per split j (span(γ_j+1..γ_k), two "
                                 "starts, largest sine; rate λ_j - λ_j+1): "
                                 + ", ".join(f"{v:.1e}" for v in sc))
            if self.clv_covariance_residual is not None:
                r, fl = self.clv_covariance_residual, self.clv_covariance_floor
                ratio = r / fl
                lines.append("  covariance residual (sine of the angle between J γ_i(t_j) and "
                             "γ_i(t_j+1) over one interval; consistency of the stored factors "
                             f"with the tangent): max {r.max():.2e} (vector "
                             f"{int(r.argmax()) + 1}); largest ratio to its rounding floor "
                             f"eps exp((λ_1 - λ_i) Δt): {ratio.max():.3g} (vector "
                             f"{int(ratio.argmax()) + 1})")
            k = self.clv.shape[2]
            if k > 1:
                mins = [float(self.angles(j).min()) for j in range(1, k)]
                lines.append("  smallest principal angle between span(γ_1..γ_j) and the rest, "
                             "j = 1.." + str(k - 1) + ": "
                             + ", ".join(f"{a:.3g}" for a in mins) + " rad")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


@dataclass
class Domination:
    """Domination of the Oseledets splitting at one window length τ (``LyapunovResult.domination``).

    Violation fractions, entangled clusters and the physical-mode count. (The reference to the
    method stays on one line: the API summary cuts the first sentence at a full stop followed by
    a space, which split the inline literal and failed the strict docs build.)"""
    __repr__ = compact_repr
    tau: float
    nu: np.ndarray               # (k, k): nu[i, j] = P(Λ_i < Λ_j) for i < j (0-based), 0 below
    samples: int                 # number of (window, trajectory) samples
    clusters: list               # [(first, last)] 1-based, contiguous, covering 1..k
    split_angles: np.ndarray     # (k - 1,): smallest principal angle at each split j = 1..k-1
    split_convergence: np.ndarray | None = None   # (k - 1,): from LyapunovResult (two starts)

    @classmethod
    def from_local(cls, loc: np.ndarray, tau: float, clv, weight) -> Domination:
        """From local exponents (samples, k)."""
        k = loc.shape[1]
        nu = np.zeros((k, k))
        for i in range(k - 1):
            nu[i, i + 1:] = (loc[:, i:i + 1] < loc[:, i + 1:]).mean(axis=0)
        # entangled clusters: i < j with an observed violation join every mode between them
        link = np.zeros(k - 1, bool)                          # link[m]: modes m and m+1 joined
        for i in range(k - 1):
            js = np.nonzero(nu[i, i + 1:] > 0)[0]
            if js.size:
                link[i:i + 1 + js.max()] = True
        clusters, start = [], 0
        for m in range(k - 1):
            if not link[m]:
                clusters.append((start + 1, m + 1))
                start = m + 1
        clusters.append((start + 1, k))
        ang = np.array([float(clv_subspace_angles(clv, j, weight).min()) for j in range(1, k)])
        return cls(tau, nu, loc.shape[0], clusters, ang)

    @property
    def n_physical(self) -> int:
        """Size of the leading entangled cluster: the modes joined to γ_1 by a chain of observed
        violations at this τ ("physical" modes of Yang et al. 2009 / Takeuchi et al. 2011; the
        following clusters are "isolated" modes, single or in degenerate pairs)."""
        return self.clusters[0][1]

    def violation(self, j: int) -> float:
        """Largest ν_il with i <= j < l (1-based split after mode ``j``): 0 means the splitting
        {1..j} | {j+1..k} was dominated in every sample at this τ."""
        k = self.nu.shape[0]
        if not 1 <= j < k:
            raise OptionError(f"violation({j}): a split j in 1..{k - 1}")
        return float(self.nu[:j, j:].max())

    def explain(self) -> str:
        """Violations per split, clusters, the physical-mode count and the sample size."""
        k = self.nu.shape[0]
        lines = [f"Domination of the Oseledets splitting at tau = {self.tau:g}: {self.samples} "
                 f"samples (windows x trajectories), {k} covariant vectors",
                 "  violation fraction across each split j | j+1 (largest nu_ij, i <= j < l): "
                 + ", ".join(f"{j}:{self.violation(j):.3g}" for j in range(1, k)),
                 "  clusters of entangled modes (any observed violation joins them): "
                 + ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in self.clusters),
                 f"  physical modes (leading entangled cluster): {self.n_physical}; the split "
                 f"after it has violation {self.violation(self.n_physical):.3g} and smallest "
                 f"principal angle {self.split_angles[self.n_physical - 1]:.3g} rad"
                 + ("" if self.split_convergence is None else
                    f" (its subspaces converged to {self.split_convergence[self.n_physical - 1]:.1e})")
                 if self.n_physical < k else
                 "  every computed mode is entangled with γ_1 at this tau: compute more vectors "
                 "or lengthen tau before counting physical modes",
                 "  a violation fraction is a sample estimate: 0 means none observed in "
                 f"{self.samples} samples; modes near k may entangle with modes not computed"]
        return "\n".join(lines)

    __str__ = explain


def _qr(xp, W, sq):
    """Batched QR of the directions W (B, k, n) in the weighted inner product (sq = sqrt(weight)
    or None): returns (Q (B, k, n) orthonormal, R (B, k, k)) with diag R > 0."""
    U = W if sq is None else W * sq
    Q, R = xp.linalg.qr(xp.swapaxes(U, 1, 2))                  # (B, n, k), (B, k, k)
    d = xp.diagonal(R, axis1=1, axis2=2)
    sgn = xp.where(d < 0, -1.0, 1.0)
    Q = Q * sgn[:, None, :]
    R = R * sgn[:, :, None]
    Q = xp.swapaxes(Q, 1, 2)
    if sq is not None:
        Q = Q / sq
    return xp.ascontiguousarray(Q), R


@renames("conatus.chaos.lyapunov:lyapunov_spectrum")
def lyapunov_spectrum(solver, x0, *, n_vectors: int, transient_steps: int,
                      tangent_transient_steps: int, steps: int, reortho_every: int, inner,
                      n_blocks: int, seed: int, project=None, clv_window: int = 0,
                      clv_backward_transient: int = 0, clv_keep_every: int = 1,
                      rhs: str = "batched", progress=None,
                      clv_checkpoints: int | None = None) -> LyapunovResult:
    """Lyapunov exponents (and optionally CLVs) of ``solver``'s discrete map.

    ``x0``: one state (n,) or B states (B, n) (independent trajectories, one batch).
    ``n_vectors``: k tangent vectors per trajectory (the k leading exponents).
    ``transient_steps``: trajectory only; ``tangent_transient_steps``: trajectory + tangent, no
    averaging (aligns the vectors); ``steps``: averaging (multiple of ``reortho_every``).
    ``seed``: the initial tangent vectors of each trajectory are drawn from (seed, its start
    state): a trajectory's exponents do not depend on its place in the batch.
    ``inner``: ``"euclidean"`` (on the coefficient vector) or a positive diagonal weight (n,);
    the exponents do not depend on it, the finite-time values and the CLV angles do.
    ``project``: optional callable V -> P V (tangent vectors restricted to an invariant subspace,
    e.g. zero mean for KS), applied before every QR; declared in ``explain()``.
    ``clv_window`` intervals (> 0: covariant vectors over the last window of the run, after
    ``clv_backward_transient`` more intervals), kept every ``clv_keep_every`` interval.
    ``progress``: optional callable(interval index, n_intervals) for logging.
    ``clv_checkpoints``: memory of the covariant pass. ``None`` stores every
    (Q_j, X_j) of the window and backward transient (the former behaviour); an integer s stores
    only the R matrices (k x k per interval) and s checkpoints (Q, X) at evenly spaced intervals:
    the backward pass recomputes each segment's Q_j forward from its checkpoint (one extra
    forward pass over the stored intervals, memory s + ceil(n_store / s) sets of k vectors instead
    of n_store). The recomputed R are compared to the stored ones
    (``res.clv_recompute_deviation``, 0 when the arithmetic is deterministic); the stored R are
    the ones used."""
    from ..errors import check_count
    from ..timestep.integrator import check_steps
    # every count is an integer, never truncated (n_vectors=2.5 ran 2 vectors,
    # a float steps failed deep in range(), reortho_every=0 divided by zero)
    for name, v in (("transient_steps", transient_steps), ("tangent_transient_steps",
                    tangent_transient_steps), ("steps", steps), ("clv_window", clv_window),
                    ("clv_backward_transient", clv_backward_transient)):
        check_steps(v, "lyapunov_spectrum", name)
    for name, v in (("n_vectors", n_vectors), ("reortho_every", reortho_every),
                    ("n_blocks", n_blocks), ("clv_keep_every", clv_keep_every)):
        check_count(v, name, "lyapunov_spectrum")
    if clv_checkpoints is not None:
        check_count(clv_checkpoints, "clv_checkpoints", "lyapunov_spectrum")
        if not clv_window:
            raise OptionError("clv_checkpoints without clv_window",
                              why="checkpoints set the memory of the covariant (backward) pass",
                              fix="give clv_window (and clv_backward_transient), or drop "
                                  "clv_checkpoints")
    if steps % reortho_every:
        raise OptionError(f"steps = {steps} must be a multiple of reortho_every = {reortho_every}")
    nint = steps // reortho_every
    if n_blocks < 1 or nint % n_blocks:
        raise OptionError(f"n_blocks = {n_blocks} must divide the number of intervals ({nint})")
    if clv_window + clv_backward_transient > nint:
        raise OptionError("clv_window + clv_backward_transient exceeds the number of intervals")
    xp = solver.xp
    x0 = np.atleast_2d(np.asarray(x0 if xp is np else xp.asnumpy(xp.asarray(x0))))
    B, n = x0.shape
    k = n_vectors
    if not 1 <= k <= n:
        raise OptionError(f"n_vectors must be in [1, n = {n}]")
    ens = Ensemble(solver, B, rhs=rhs)
    require_rk(ens)
    if isinstance(inner, str):
        if inner != "euclidean":
            raise OptionError(f"inner must be 'euclidean' or a diagonal weight array, got {inner!r}")
        sq, inner_s = None, "Euclidean on the coefficient vector"
    else:
        w = np.asarray(inner, float)
        if w.shape != (n,) or np.any(w <= 0):
            raise OptionError("inner: diagonal weight must be a positive array of length n")
        sq, inner_s = xp.asarray(np.sqrt(w)), "diagonal weight (user)"
    ens.set_states(x0)
    dt, tab = ens.dt, ens._rk
    ens.run(transient_steps)
    # the initial tangent vectors of a trajectory come from (seed, its start state), not from its
    # place in the batch (audit invariance): one stream of B*k draws gave a trajectory
    # other finite-time exponents when the starts were reordered or when it was run alone
    rngs = [np.random.default_rng([int(seed), zlib.crc32(np.ascontiguousarray(x0[b]).tobytes())])
            for b in range(B)]
    V = xp.asarray(np.stack([r.standard_normal((k, n)) for r in rngs]))
    if project is not None:
        V = project(V)
    V, _ = _qr(xp, V, sq)

    def advance(V, nsteps):
        X = ens.X
        for _ in range(nsteps):
            st = ens.rk_stages(X, dt, tab)
            V = rk_tangent_step(ens, st, V, dt, tab)
            X = st[-1]
        ens.X = X
        ens.iteration += nsteps
        ens.t = ens._t0 + ens.iteration * dt
        return V

    def interval(V):
        """One reorthonormalization interval: advance, project, QR (forward pass and the
        recomputation of the checkpointed covariant pass use this same function)."""
        V = advance(V, reortho_every)
        if project is not None:
            V = project(V)
        return _qr(xp, V, sq)

    for _ in range(tangent_transient_steps // reortho_every):
        V, _ = interval(V)
    rem = tangent_transient_steps % reortho_every
    if rem:
        V, _ = _qr(xp, advance(V, rem) if project is None else project(advance(V, rem)), sq)

    logs = np.empty((nint, B, k))
    nstore = clv_window + clv_backward_transient
    first = nint - nstore                         # first stored interval
    store: list = []                              # (Q_j, X_j, R_{j+1}) per stored interval
    Rs: list = []                                 # R_{j+1} only (checkpointed mode)
    seg = -(-nstore // clv_checkpoints) if clv_checkpoints and nstore else 0
    ckpts: list = []                              # (Q, X, iteration) at the segment starts
    for j in range(nint):
        keep = clv_window and j >= first
        if keep and not seg:
            Qj, Xj = V, ens.X.copy()
        elif keep and (j - first) % seg == 0:
            ckpts.append((V, ens.X.copy(), ens.iteration))
        V, R = interval(V)
        d = xp.diagonal(R, axis1=1, axis2=2)
        logs[j] = np.log(d if xp is np else xp.asnumpy(d))
        if keep and not seg:
            store.append((Qj, Xj, R))
        elif keep:
            Rs.append(R)
        if progress is not None:
            progress(j, nint)

    interval_t = reortho_every * dt
    T = nint * interval_t
    per_traj = logs.sum(axis=0) / T                               # (B, k)
    nb = nint // n_blocks
    blk = logs.reshape(n_blocks, nb, B, k).sum(axis=1) / (nb * interval_t)  # (n_blocks, B, k)
    bm = blk.transpose(1, 0, 2).reshape(B * n_blocks, k)
    lam = per_traj.mean(axis=0)
    err = bm.std(axis=0, ddof=1) / np.sqrt(bm.shape[0]) if bm.shape[0] > 1 else np.full(k, np.nan)
    if n_blocks > 2:
        a, b_ = blk[:-1] - blk.mean(axis=0), blk[1:] - blk.mean(axis=0)
        lag1 = (a * b_).sum(axis=(0, 1)) / np.maximum((blk - blk.mean(axis=0)) ** 2, 1e-300).sum(axis=(0, 1))
    else:
        lag1 = np.full(k, np.nan)
    settings = dict(scheme=ens.scheme, dt=dt, backend=ens.backend, n_vectors=k, trajectories=B,
                    transient_steps=transient_steps, tangent_transient_steps=tangent_transient_steps,
                    steps=steps, reortho_every=reortho_every, inner=inner_s,
                    project="none" if project is None else getattr(project, "__doc__", None)
                    or "user projection", n_blocks=n_blocks, seed=seed, rhs=rhs,
                    clv_window=clv_window, clv_backward_transient=clv_backward_transient,
                    clv_keep_every=clv_keep_every, clv_checkpoints=clv_checkpoints)
    res = LyapunovResult(lam, err, per_traj, bm, lag1, logs, interval_t, settings)
    res.inner_weight = None if sq is None else np.asarray(inner, float)
    if k < n:
        res.notes.append(f"{k} of {n} exponents computed: sum and Kaplan–Yorke use these only")
    if clv_window:
        itemsize = np.dtype(ens.dtype).itemsize
        qset = (B * k * n + B * n) * itemsize                 # one (Q, X) pair
        rset = B * k * k * itemsize
        t_first = ens.t - nstore * interval_t
        full = nstore * (qset + rset)
        if not seg:
            items = ((jj, *store[jj]) for jj in range(nstore - 1, -1, -1))
            res.clv_memory = (f"all {nstore} stored intervals kept in memory (Q_j, X_j, R_j+1): "
                              f"{full / 2**20:.3g} MiB")
        else:
            dev = [0.0]
            items = _recompute(xp, ens, interval, ckpts, Rs, seg, nstore, dev)
            mem = (len(ckpts) + seg) * qset + nstore * rset
            res.clv_memory = (f"checkpointed: {len(ckpts)} checkpoint(s) (Q, X) every {seg} "
                              f"interval(s), {nstore} R matrices, one segment of at most {seg} "
                              f"Q_j recomputed at a time: {mem / 2**20:.3g} MiB instead of "
                              f"{full / 2**20:.3g} MiB, for one extra forward pass over the "
                              f"{nstore} stored intervals")
        _ginelli(xp, res, items, nstore, clv_backward_transient, clv_keep_every, rngs, interval_t,
                 t_first, B, k)
        if seg:
            res.clv_recompute_deviation = dev[0]
        res.clv_covariance_residual, res.clv_covariance_floor = _covariance_residual(
            xp, ens, advance, project, res, reortho_every, sq)
    return res


def _recompute(xp, ens, interval, ckpts: list, Rs: list, seg: int, nstore: int, dev: list):
    """Stored intervals in reverse order, recomputed segment by segment from the checkpoints
    (checkpointed covariant pass): yields (j, Q_j, X_j, R_j+1) with the stored R; the largest
    relative difference between a recomputed and a stored R goes into ``dev[0]``."""
    for s in range(len(ckpts) - 1, -1, -1):
        V, X, it = ckpts[s]
        ens.X = X.copy()
        ens.iteration = it
        ens.t = ens._t0 + it * ens.dt
        j0, j1 = s * seg, min((s + 1) * seg, nstore)
        part = []
        for jj in range(j0, j1):
            Qj, Xj = V, ens.X.copy()
            V, R = interval(V)
            dev[0] = max(dev[0], float(xp.abs(R - Rs[jj]).max() / xp.abs(Rs[jj]).max()))
            part.append((Qj, Xj))
        for jj in range(j1 - 1, j0 - 1, -1):
            yield (jj, *part[jj - j0], Rs[jj])


def _covariance_residual(xp, ens, advance, project, res: LyapunovResult, nsteps: int, sq):
    """Covariance check of the CLVs: γ_i(t_j) advanced by the exact tangent over
    one reorthonormalization interval must be parallel to γ_i(t_j+1), for up to three evenly
    spaced kept times (pairs recorded by the backward pass) and every trajectory. Returns the
    largest sine of the angle (declared inner product) per vector, and its rounding floor
    eps * exp((λ_1 - λ_i) * interval): a strongly contracted vector is swamped by the rounding of
    the expanding directions over the interval, so a residual at its floor is all that can be
    measured. Every backward iterate is covariant by construction: this checks the consistency
    of the stored (or recomputed) factors with the tangent; the convergence of the backward
    iteration is ``clv_backward_convergence``."""
    pairs = res.__dict__.pop("_cov_pairs", [])
    k = res.clv.shape[2]
    floor = np.finfo(float).eps * np.exp((res.exponents[0] - res.exponents) * res.interval)
    if not pairs:
        return None, floor
    worst = np.zeros(k)
    for X, G0, G1 in pairs:
        ens.X = X.copy()
        W = advance(G0, nsteps)
        if project is not None:
            W = project(W)
        G = G1
        if sq is not None:
            W, G = W * sq, G * sq
        sin = _sine(xp, W, G, axis=-1)
        worst = np.maximum(worst, (sin if xp is np else xp.asnumpy(sin)).max(axis=0))
    return worst, floor


def _split_sines(xp, C, C2) -> np.ndarray:
    """For each split j = 1..k-1: the largest sine of the principal angles between
    span(columns j+1..k) of C and of C2 (B, k, k), maximum over the batch: the convergence of the
    subspace span(γ_j+1..γ_k), which needs only the gap λ_j - λ_j+1 (span(γ_1..γ_j) is span(q_1..
    q_j) for both starts, exactly)."""
    k = C.shape[-1]
    out = np.zeros(k - 1)
    for j in range(1, k):
        A, Bm = xp.linalg.qr(C[:, :, j:])[0], xp.linalg.qr(C2[:, :, j:])[0]
        D = A - Bm @ (xp.swapaxes(Bm, 1, 2) @ A)                 # part of A outside span(Bm)
        sv = xp.linalg.norm(D, axis=(1, 2), ord=2)
        out[j - 1] = float(sv.max())
    return out


def _sine(xp, a, b, axis: int):
    """Sine of the angle between a and b along ``axis``, from the component of a orthogonal to b
    (accurate for small angles, unlike sqrt(1 - cos^2), which bottoms out at sqrt(eps))."""
    bh = b / xp.linalg.norm(b, axis=axis, keepdims=True)
    ah = a / xp.linalg.norm(a, axis=axis, keepdims=True)
    return xp.linalg.norm(ah - (ah * bh).sum(axis=axis, keepdims=True) * bh, axis=axis)


def _ginelli(xp, res: LyapunovResult, items, nstore: int, nback: int, every: int, rngs,
             interval: float, t_first: float, B: int, k: int) -> None:
    """Backward pass of Ginelli et al. over the stored intervals, given in reverse order as
    (j, Q_j, X_j, R_{j+1}). Also records the per-interval log growth of every CLV
    (``res.clv_logs``, forward time order): the local exponents of the domination diagnostics.

    Convergence of the backward iteration: a second, independent random
    upper-triangular start is iterated alongside (k x k work only); at every kept time the sine of
    the angle between the two estimates of each γ_i is measured, and its largest value is
    ``res.clv_backward_convergence`` (k,): the error left by the backward transient. (The
    covariance residual cannot see it: every backward iterate is exactly covariant.)"""
    def start():
        C = np.triu(np.stack([r.standard_normal((k, k)) for r in rngs]))   # per trajectory
        return xp.asarray(C / np.linalg.norm(C, axis=1, keepdims=True))
    C, C2 = start(), start()
    kept, states, times, growth = [], [], [], []
    conv = np.zeros(k)
    split = np.zeros(max(k - 1, 0))
    n_kept = (nstore - nback - 1) // every + 1
    probe = {int(round(v)) for v in np.linspace(0, n_kept - 1, min(3, n_kept))}
    pairs, prev = [], None                                       # prev: γ at t_{j+1}
    for jj, Qj, Xj, R in items:
        C_next = C
        C = xp.linalg.solve(R, C)                                # C_j = R_{j+1}^{-1} C_{j+1}
        nrm = xp.linalg.norm(C, axis=1)                          # (B, k)
        C = C / nrm[:, None, :]
        C2 = xp.linalg.solve(R, C2)
        C2 = C2 / xp.linalg.norm(C2, axis=1)[:, None, :]
        if jj < nstore - nback:
            growth.append(-np.log(nrm if xp is np else xp.asnumpy(nrm)))
            if (nstore - nback - 1 - jj) % every == 0:
                G = xp.einsum("bkn,bki->bin", Qj, C)             # γ_i = Σ_m C_mi q_m
                if len(kept) in probe and prev is not None:      # covariance-check pair
                    pairs.append((Xj.copy(), G, xp.einsum("bkn,bki->bin", prev, C_next)))
                kept.append(G if xp is np else xp.asnumpy(G))
                states.append(Xj if xp is np else xp.asnumpy(Xj))
                times.append(t_first + jj * interval)
                # Q_j orthonormal in the declared inner product: angles between γ estimates are
                # angles between the columns of C and C2 (unit columns)
                sin = _sine(xp, C, C2, axis=1)                   # (B, k)
                conv = np.maximum(conv, (sin if xp is np else xp.asnumpy(sin)).max(axis=0))
                split = np.maximum(split, _split_sines(xp, C, C2))
        prev = Qj
    res.clv = np.stack(kept[::-1], axis=1)
    res.clv_states = np.stack(states[::-1], axis=1)
    res.clv_times = np.asarray(times[::-1])
    res.clv_logs = np.stack(growth[::-1])                        # (n_window, B, k)
    res.clv_exponents = res.clv_logs.mean(axis=(0, 1)) / interval
    res.clv_backward_convergence = conv
    res.clv_split_convergence = split
    res._cov_pairs = pairs
