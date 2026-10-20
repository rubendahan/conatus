"""``sim.lyapunov(...)``: Lyapunov exponents and covariant Lyapunov vectors of a native simulation. A thin, declared wrapper over :func:`conatus.chaos.lyapunov_spectrum`: lengths are
given as durations of the simulation (whole numbers of steps and of reorthonormalization
intervals, never rounded), the tangent is the exact derivative of the simulation's own step
(same scheme, same dt, same factors), and the restriction to zero-mean tangent vectors of
periodic unknowns is a named option instead of an index computed by hand.

What it computes, the algorithms and the references: and the
guide page ``docs/site/guide/dynamics/chaos.md``.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ..errors import InputTypeError, OptionError, ResolutionError

__all__ = ["simulation_lyapunov"]

_DOC = "docs/site/guide/dynamics/chaos.md"


def _whole(sim: Any, T: float, what: str, unit: float, unit_name: str, *, zero: bool) -> int:
    """Number of units in the duration ``T`` (refused unless whole; 0 only when ``zero``)."""
    from ..timestep.integrator import is_whole
    import math
    # a finite number (review of the inputs audit: T=nan was a bare ValueError of
    # int(round(nan)), inf an OverflowError)
    if isinstance(T, bool) or not isinstance(T, (int, float, np.integer, np.floating)) \
            or not math.isfinite(float(T)):
        raise InputTypeError(f"lyapunov({what}={T!r}): a duration (a finite number)", doc=_DOC)
    n = float(T) / unit
    k = int(round(n))
    if k < (0 if zero else 1) or not is_whole(n):
        raise ResolutionError(
            f"lyapunov({what}={T!r}): not a whole number of {unit_name} ({unit:g}): {n:.6f}",
            why="durations are never rounded: the time step and the reorthonormalization "
                "interval stay what was declared",
            fix=f"{what} = {max(k, 0 if zero else 1) * unit!r}", doc=_DOC)
    return k


def _zero_mean_projection(sim: Any, names: Sequence[str]) -> Callable:
    """Projection of tangent vectors (B, k, n) onto zero mean for the listed unknowns: their
    coefficient of wavenumber zero along every axis (all axes periodic) is set to 0."""
    comp = sim.compiled
    from ..ir.bases import ComplexFourier, RealFourier
    bad = [c for c, b in comp.bases.items() if not isinstance(b, (RealFourier, ComplexFourier))]
    if bad:
        raise OptionError(f"lyapunov(zero_mean={tuple(names)!r}) on a non-periodic axis "
                          f"({', '.join(bad)})",
                          why="the mean is one coefficient only when every axis is periodic",
                          fix="project=f with your own projection V -> P V, or drop zero_mean",
                          doc=_DOC)
    host = getattr(sim.solver.system, "host", sim.solver.system)
    known = [v.name for v in host.variables]
    coeffs = [np.zeros(shape) for shape in host.var_shapes]
    for name in names:
        if name not in known or name.startswith("tau_"):
            sim._check_unknown(name, "lyapunov(zero_mean=...)")
        a = coeffs[known.index(name)]
        rank = comp.fields[name].rank
        idx = (slice(None),) * rank + (0,) * (a.ndim - rank)
        a[idx] = 1.0
    mask = np.abs(host.gather_state(coeffs, check_subspace=False)) > 0
    idx = np.nonzero(mask)[0]
    xp = sim.solver.xp
    idx_d = xp.asarray(idx)

    def project(V):
        V = V.copy()
        V[..., idx_d] = 0.0
        return V

    project.__doc__ = ("zero mean: the wavenumber-zero coefficient of "
                       + ", ".join(names) + f" set to 0 in every tangent vector ({idx.size} "
                       "component(s) of the state vector)")
    return project


def _starts(sim: Any, starts: Any) -> np.ndarray:
    """Initial states (B, n): the current state, an array of solver states, or a list of
    ``sim.initial`` keyword mappings (the simulation's own state is restored afterwards)."""
    sv = sim.solver
    xp = sv.xp

    def host(a):
        return np.asarray(a if xp is np else xp.asnumpy(a))

    if starts is None:
        return host(sv.X)[None].copy()
    if isinstance(starts, np.ndarray) or (hasattr(starts, "shape") and hasattr(starts, "dtype")):
        X = np.atleast_2d(host(starts))
        if X.ndim != 2 or X.shape[1] != sv.system.n:
            raise InputTypeError(f"lyapunov(starts=<array of shape {X.shape}>): states of shape "
                                 f"(B, {sv.system.n}) (sim.solver.X of each start)", doc=_DOC)
        return X
    if isinstance(starts, Sequence) and starts and all(isinstance(d, Mapping) for d in starts):
        # every attribute sim.initial and set_state may rebind or clear is restored (a fixed
        # list missed the start time of the run, _t_initial / _t_given,
        # added by sim.initial(t=), and the multistep history lists)
        saved_sv = _snapshot(sv)
        saved_sim = _snapshot(sim)
        out = []
        try:
            for d in starts:
                sim.initial(**d)
                out.append(host(sv.X).copy())
        finally:
            _restore(sv, saved_sv)
            _restore(sim, saved_sim)
        return np.stack(out)
    raise InputTypeError("lyapunov(starts=...): None (the current state), an array (B, n) of "
                         "solver states, or a list of sim.initial(...) keyword dicts",
                         fix="starts=[{'u': f1}, {'u': f2}]", doc=_DOC)


def _snapshot(obj: Any) -> dict:
    """Shallow copy of an object's attributes: lists copied (set_state clears the multistep
    history in place), the state vector ``X`` copied, every other value by reference (they are
    rebound, not modified, by sim.initial; operators and factors are not duplicated)."""
    out = {k: list(v) if isinstance(v, list) else v for k, v in vars(obj).items()}
    if "X" in out and hasattr(out["X"], "copy"):
        out["X"] = out["X"].copy()
    return out


def _restore(obj: Any, saved: dict) -> None:
    """Put back the attributes of :func:`_snapshot` (lists refilled in place, so references held
    elsewhere stay valid) and drop attributes added since."""
    for k in [k for k in vars(obj) if k not in saved]:
        delattr(obj, k)
    for k, v in saved.items():
        cur = vars(obj).get(k)
        if isinstance(v, list) and isinstance(cur, list):
            cur[:] = v
        else:
            setattr(obj, k, v)


def simulation_lyapunov(sim: Any, n_vectors: int, *, transient: float, align: float, T: float,
                        reortho_every: int, inner: Any, n_blocks: int, seed: int,
                        starts: Any = None, zero_mean: Sequence[str] = (),
                        project: Callable | None = None, covariant: bool = False,
                        clv_window: float | None = None, clv_backward: float | None = None,
                        clv_every: int = 1, clv_checkpoints: int | None = None,
                        progress: Callable | None = None):
    """See :meth:`conatus.model.run.Simulation.lyapunov`."""
    from ..chaos import lyapunov_spectrum
    if not isinstance(covariant, bool):
        raise OptionError(f"lyapunov(covariant={covariant!r}): True or False", doc=_DOC)
    if isinstance(zero_mean, str):
        zero_mean = (zero_mean,)
    if zero_mean and project is not None:
        raise OptionError("lyapunov(zero_mean=..., project=...): one projection",
                          fix="fold the zero mean into your project function, or drop one",
                          doc=_DOC)
    dt = sim.dt
    from ..errors import check_count
    reortho_every = check_count(reortho_every, "reortho_every", "lyapunov")
    # the seed of the random tangent vectors: an integer >= 0 (review of the inputs audit:
    # seed=0.5 was a bare TypeError of NumPy's SeedSequence)
    check_count(seed, "seed", "lyapunov", minimum=0)
    unit = reortho_every * dt
    transient_steps = _whole(sim, transient, "transient", dt, "steps", zero=True)
    align_steps = _whole(sim, align, "align", dt, "steps", zero=True)
    steps = _whole(sim, T, "T", unit, "intervals reortho_every * dt", zero=False) * reortho_every
    if covariant:
        if clv_window is None or clv_backward is None:
            raise OptionError("lyapunov(covariant=True) needs clv_window= and clv_backward=",
                              why="the covariant vectors are computed over a window that ends "
                                  "clv_backward before the end of the run (the backward "
                                  "iteration converges over that time): two durations of the "
                                  "problem, never defaulted",
                              fix=f"clv_window={20 * unit!r}, clv_backward={20 * unit!r} (whole "
                                  "intervals; longer for close exponents)", doc=_DOC)
        win = _whole(sim, clv_window, "clv_window", unit, "intervals", zero=False)
        back = _whole(sim, clv_backward, "clv_backward", unit, "intervals", zero=True)
    else:
        if clv_window is not None or clv_backward is not None or clv_checkpoints is not None:
            raise OptionError("lyapunov(clv_window=/clv_backward=/clv_checkpoints=) without "
                              "covariant=True", fix="covariant=True", doc=_DOC)
        win = back = 0
    proj = _zero_mean_projection(sim, tuple(zero_mean)) if zero_mean else project
    X0 = _starts(sim, starts)
    res = lyapunov_spectrum(sim.solver, X0, n_vectors=n_vectors, transient_steps=transient_steps,
                            tangent_transient_steps=align_steps, steps=steps,
                            reortho_every=reortho_every, inner=inner, n_blocks=n_blocks,
                            seed=seed, project=proj, clv_window=win, clv_backward_transient=back,
                            clv_keep_every=clv_every, rhs="batched", progress=progress,
                            clv_checkpoints=clv_checkpoints)
    res.notes.append(f"native: sim.lyapunov on '{sim.compiled.problem.name}' "
                     f"(transient {transient:g}, align {align:g}, T {T:g} per trajectory"
                     + (f", CLV window {clv_window:g} ending {clv_backward:g} before the end"
                        if covariant else "") + "); the simulation's own state is not advanced")
    _attach_fields(sim, res)
    return res


def _attach_fields(sim: Any, res: Any) -> None:
    """Make ``res.clv_grid(name, i, ...)`` return grid values of one unknown (the layout of
    ``sim.grid(name)``) for this simulation."""
    host = getattr(sim.solver.system, "host", sim.solver.system)
    plan = sim.solver.plan
    names = [v.name for v in host.variables]

    def grid(name: str, vec: np.ndarray) -> np.ndarray:
        sim._check_unknown(name, "clv_grid")
        c = host.scatter(np.asarray(vec))[names.index(name)]
        g = np.asarray(plan.to_grid(name, c))
        if sim.compiled.symmetry is not None:
            from .sincos import physical_slice
            ax, sl = physical_slice(sim.compiled)
            rank = sim.compiled.fields[name].rank
            g = np.take(g, np.arange(g.shape[rank + ax])[sl], axis=rank + ax)
        return g

    res._grid = grid
