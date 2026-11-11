"""Cost estimate of an initial-value run before it starts: predicted time per step,
run time and peak memory on the CPU, from the compiled problem, with a stated error.

    from conatus.cost import estimate
    est = estimate(m, N=(256, 64), scheme="RK443", dt=5e-4, steps=2000)
    print(est.explain())

What is computed. ``estimate`` builds the simulation (discretisation and stacked system: the setup a
run pays anyway, measured and reported as ``setup_s``) and takes no step. From the stacked system it
reads, without factorising anything:

* ``solve_work``: band entries touched by one implicit solve, the sum over solver buckets of
  systems x n x (2 kl + ku + 1 + 2 r) (r border rows);
* ``fft_work``: grid points P of the dealiased grid times log2 P, times the number of unknowns
  that enter a right-hand side and the number of equations with one (their transforms);
* ``solves``: implicit solves (and right-hand-side evaluations) per step: the number of stages of
  an IMEX Runge-Kutta scheme, 1 for a multistep scheme;
* ``state_bytes``: the size of the state vector.

The model is linear in these features, time per step
``t = solves * (a solve_work + b fft_work + c) + d`` and memory above the imported package
``mem = e factor_bytes + f state_bytes * solves + g grid_bytes + h``, with coefficients fitted on
one thread of an AMD EPYC 9004-series node and checked on cases
and resolutions that were not used in the fit (``CALIBRATION``). The stated error is the largest
ratio between measured and predicted values on that check. On another machine the coefficients
scale with its speed: ``estimate(..., calibrate=True)`` times a banded solve and an FFT of the
problem's own sizes for a few milliseconds and rescales ``a`` and ``b``.

Declared limits: the CPU backend and one thread (more threads: the time is an upper bound); the
fitted cases are Cartesian (Fourier and Chebyshev bases); a curvilinear geometry is estimated
with the same coefficients and its error is not checked (said in ``explain()``).
"""
from __future__ import annotations

import math
import resource
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import OptionError

__all__ = ["CostEstimate", "estimate", "features", "predict", "micro_times", "CALIBRATION"]

#: Fitted coefficients (seconds per unit, bytes per byte) and their provenance (the run named in
#: "runs"). "time_ratio"/"memory_ratio": the
#: largest measured/predicted ratio (or its inverse) on the held-out configurations.
CALIBRATION: dict[str, Any] = {
    "machine": "AMD EPYC 9004-series node, one thread",
    "threads": 1,
    "time": {
        "a": 1.8861917126080083e-09,
        "b": 2.563661984152015e-09,
        "c": 0.00022673976437225527,
        "d": 0.0
    },
    "memory": {
        "e": 0.723734387788443,
        "f": 59.40025760274621,
        "g": 0.5837713124208719,
        "h": 21884777.109660517
    },
    "time_ratio": 1.382,
    "memory_ratio": 2.216,
    "fit_time_ratio": 1.42,
    "fit_memory_ratio": 1.451,
    "micro": {
        "solve_s": 3.0291376726310166e-05,
        "fft_s": 3.672657062410625e-05
    },
    "runs": [
        "v04_cost_c (srun job 17461, tree v04c, core build_v04a)"
    ],
    "checked_on": [
        "burgers 1024",
        "burgers 16384",
        "tg2d 64x64",
        "tg2d 256x256",
        "rb2d 64x32",
        "rb2d 128x64",
        "rb2d 256x128",
        "rb3d 16x16x24",
        "rb3d 32x32x24"
    ]
}


def _stages(scheme: str) -> int:
    from .timestep.schemes import RK_SCHEMES, rk_tableau
    if scheme in RK_SCHEMES:
        return len(rk_tableau(scheme)[2]) - 1
    return 1


def features(sim) -> dict:
    """Cost features of a built (not stepped) native simulation or ``IVPSolver``."""
    solver = getattr(sim, "solver", sim)
    sysm = solver.system
    from .timestep.system import _band_of
    solve_work, factor_entries = 0, 0
    for idx, _, _ in sysm._buckets:
        s = sysm.slots[idx[0]]
        kl, ku = _band_of(s.M, s.L, s.rperm, s.cperm, s.n_border)
        w = len(idx) * s.size * (2 * kl + ku + 1 + 2 * s.n_border)
        solve_work += w
        factor_entries += w
    for idx, *_ in sysm._cbuckets:
        s = sysm.slots[idx[0]]
        cz = s.cz
        kl, ku = _band_of(cz.M, cz.L, cz.rperm, cz.cperm, cz.n_border)
        w = 4 * len(idx) * cz.size * (2 * kl + ku + 1 + 2 * cz.n_border)   # complex arithmetic
        solve_work += w
        factor_entries += w // 2
    for keys, *_ in sysm._zbuckets:
        s = sysm.slots[keys[0][0]]
        v = s.zs[keys[0][1]]
        kl, ku = _band_of(v.M, v.L, v.rperm, v.cperm, v.n_border)
        w = 4 * len(keys) * v.size * (2 * kl + ku + 1 + 2 * v.n_border)
        solve_work += w
        factor_entries += w // 2
    P = 1
    for b in solver.plan.domain.bases:
        P *= max(1, int(b.grid_size))
    n_rhs_vars = len(getattr(solver, "_rhs_vars", ())) or 1
    n_rhs_eqs = sum(e is not None for e in getattr(solver, "_rhs_exprs", [])) or 1
    transforms = n_rhs_vars * (1 + len(solver.plan.domain.bases)) + n_rhs_eqs
    item = np.dtype(sysm.dtype).itemsize
    return dict(unknowns=int(sysm.n), solve_work=int(solve_work), grid_points=int(P),
                fft_work=float(transforms * P * math.log2(max(P, 2))), transforms=int(transforms),
                solves=_stages(solver.scheme), state_bytes=int(sysm.n * item),
                factor_bytes=int(factor_entries * item), grid_bytes=int(transforms * P * 8),
                buckets=len(sysm._buckets) + len(sysm._cbuckets) + len(sysm._zbuckets))


def predict(feat: dict, cal: dict | None = None, scale: tuple[float, float] = (1.0, 1.0)) -> tuple[float, float]:
    """(seconds per step, bytes above the imported package) from features and coefficients."""
    cal = CALIBRATION if cal is None else cal
    t, m = cal["time"], cal["memory"]
    s = feat["solves"]
    sec = s * (scale[0] * t["a"] * feat["solve_work"] + scale[1] * t["b"] * feat["fft_work"] + t["c"]) + t["d"]
    mem = (m["e"] * feat["factor_bytes"] + m["f"] * feat["state_bytes"] * s + m["g"] * feat["grid_bytes"]
           + m["h"])
    return float(sec), float(mem)


def micro_times() -> tuple[float, float]:
    """Seconds of one banded LU solve (n = 512, kl = ku = 8, LAPACK dgbtrs) and of one real FFT of
    length 4096 (numpy.fft), each timed for about 20 ms."""
    import scipy.linalg as sl
    n, kl = 512, 8
    ab = np.random.default_rng(0).standard_normal((3 * kl + 1, n))
    ab[2 * kl] += 10 * kl
    lu, piv, _ = sl.lapack.dgbtrf(ab, kl, kl)
    rhs = np.ones(n)

    def bench(fn) -> float:
        fn()
        reps, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < 0.02:
            fn()
            reps += 1
        return (time.perf_counter() - t0) / reps

    ts = bench(lambda: sl.lapack.dgbtrs(lu, kl, kl, rhs, piv))
    x = np.random.default_rng(1).standard_normal(4096)
    return ts, bench(lambda: np.fft.rfft(x))


def _micro_scale() -> tuple[float, float]:
    """Speed of this machine relative to the calibration machine on the two kernels of
    :func:`micro_times` (reference times in CALIBRATION["micro"])."""
    ref = CALIBRATION.get("micro")
    if not ref:
        return 1.0, 1.0
    ts, tf = micro_times()
    return ts / ref["solve_s"], tf / ref["fft_s"]


@dataclass
class CostEstimate:
    """Predicted cost of a run (see the module docstring for the model and its error)."""
    N: Any
    scheme: str
    steps: int | None
    features: dict
    s_per_step: float
    memory_bytes: float
    setup_s: float
    setup_rss_bytes: float
    time_ratio: float
    memory_ratio: float
    scale: tuple[float, float] = (1.0, 1.0)
    notes: list[str] = field(default_factory=list)

    @property
    def run_s(self) -> float | None:
        """Predicted run time: time per step times the steps (None without steps or until)."""
        return None if self.steps is None else self.s_per_step * self.steps

    def explain(self) -> str:
        """The prediction, its stated error, the features it comes from and the notes."""
        f, r = self.features, self.time_ratio
        lines = [f"Cost estimate, N = {self.N}, {self.scheme}, CPU, one thread "
                 f"(coefficients: {CALIBRATION['machine']}):",
                 f"  {f['unknowns']} unknowns, {f['grid_points']} grid points, {f['solves']} implicit "
                 f"solve(s) per step, {f['buckets']} solver bucket(s)",
                 f"  time per step {self.s_per_step * 1e3:.3g} ms (measured/predicted within a factor "
                 f"{r:.2f} on the checked configurations)"]
        if self.steps is not None:
            lines.append(f"  {self.steps} steps: {self.run_s:.3g} s, plus the setup already paid "
                         f"here: {self.setup_s:.3g} s")
        lines.append(f"  memory of the run above the imported package: {self.memory_bytes / 2**20:.4g} MiB "
                     f"(within a factor {self.memory_ratio:.2f}); building it took "
                     f"{self.setup_rss_bytes / 2**20:.4g} MiB")
        if self.scale != (1.0, 1.0):
            lines.append(f"  rescaled to this machine: solves x{self.scale[0]:.2f}, transforms "
                         f"x{self.scale[1]:.2f} (micro-benchmark)")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def estimate(model, N: Any, *, scheme: str, dt: float, steps: int | None = None,
             until: float | None = None, calibrate: bool = False, **simulate_options: Any) -> CostEstimate:
    """Predicted time per step, run time and memory of ``model.simulate(N, scheme=, dt=)`` (a
    native ``cn.Model`` with ``dt()``); ``steps`` or ``until`` (a whole number of steps) gives the
    run time. ``calibrate=True`` rescales the time coefficients to this machine (about 40 ms)."""
    if steps is not None and until is not None:
        raise OptionError("estimate(): steps= or until=, not both", fix="estimate(..., steps=1000)")
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    t0 = time.perf_counter()
    sim = model.simulate(N=N, scheme=scheme, dt=dt, **simulate_options)
    setup = time.perf_counter() - t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if steps is None and until is not None:
        steps = sim._steps_of(until, "estimate")    # the run's own whole-number-of-steps rule
    if getattr(sim.solver, "backend", "cpu") != "cpu":
        raise OptionError("estimate(): only the CPU backend is modelled", fix="backend='cpu'")
    feat = features(sim)
    scale = _micro_scale() if calibrate else (1.0, 1.0)
    sec, mem = predict(feat, scale=scale)
    notes = []
    if CALIBRATION["machine"] == "not calibrated":
        notes.append("no calibration recorded: the numbers are zero")
    kinds = {type(b).__name__ for b in sim.solver.plan.domain.bases}
    if not kinds <= {"RealFourier", "ComplexFourier", "Chebyshev"}:
        notes.append(f"bases {sorted(kinds)}: outside the checked Fourier/Chebyshev cases, error unknown")
    return CostEstimate(N=N, scheme=scheme, steps=steps, features=feat, s_per_step=sec,
                        memory_bytes=mem, setup_s=setup, setup_rss_bytes=float(max(0, rss1 - rss0)),
                        time_ratio=float(CALIBRATION["time_ratio"]),
                        memory_ratio=float(CALIBRATION["memory_ratio"]), scale=scale, notes=notes)
