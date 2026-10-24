"""Argument and state checks shared by every ``run`` of the time steppers: the finite-state
check and its cadence, the whole-number rule for a horizon in steps, and the checks of steps,
substeps, dt and backend, parameters, checkpoint suffixes, the time parameter and callbacks.
Split out of :mod:`conatus.timestep.integrator`, which re-exports every name."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..errors import OptionError, did_you_mean


#: Finite-state check: with ``check_finite=True`` (the default) the state is
#: tested for NaN/inf every FINITE_CHECK_EVERY steps, at the end of every ``run``/``run_graph``
#: call and before every checkpoint or analysis write; a blown-up run stops with a NumericalError
#: naming the step, the time and the field. One reduction over the state (one host sync on the
#: GPU) per check: measured below 1 % of the step cost (docs/site/guide/timesteppers.md).
FINITE_CHECK_EVERY = 100


#: One whole-number rule for "a horizon is a whole number of steps" (
#: P4.4: run(until=), the watch's dt setter, flowmap(T), shell Floquet, pipe orbits and edge
#: tracking used 1e-9 or 1e-12): n = T/dt is whole when |n - round(n)| <= WHOLE_STEPS_TOL x
#: max(1, |n|). The quotient of two decimals written with ~10 significant digits (T = 0.3999,
#: dt = T/4000) stays within it; a step that does not divide the horizon misses it by >= 1/n.
WHOLE_STEPS_TOL = 1e-9


def is_whole(n: float) -> bool:
    """``n`` (a number of steps T/dt) is whole within :data:`WHOLE_STEPS_TOL`."""
    return abs(n - round(n)) <= WHOLE_STEPS_TOL * max(1.0, abs(n))


def finite_cadence(check_finite: Any, where: str) -> int:
    """Steps between two finite-state checks: ``True`` -> FINITE_CHECK_EVERY, ``False`` -> 0
    (off, declared by explain()), a positive int -> that cadence."""
    if check_finite is True:
        return FINITE_CHECK_EVERY
    if check_finite is False:
        return 0
    if isinstance(check_finite, (int, np.integer)) and not isinstance(check_finite, bool) \
            and check_finite > 0:
        return int(check_finite)
    raise OptionError(f"{where}: check_finite={check_finite!r}",
                      why="the finite-state check is on (True: every "
                          f"{FINITE_CHECK_EVERY} steps), off (False) or every n steps (n > 0)",
                      fix="check_finite=True (default), check_finite=10 or check_finite=False",
                      doc="docs/site/guide/timesteppers.md")


def solver_finite_check(solver: Any, where: str) -> None:
    """The finite-state check of ``solver`` from outside its step loop (outputs, checkpoints,
    error control): its own ``_check_finite`` when it has one (collective in the distributed
    solver: every rank must call it at the same iteration), else :func:`check_state_finite`."""
    own = getattr(solver, "_check_finite", None)
    if callable(own):
        own(where)
    else:
        check_state_finite(solver, where)


def check_state_finite(solver: Any, where: str) -> None:
    """Raise NumericalError if the state ``solver.X`` holds a NaN or an inf (one reduction; on
    the GPU one host synchronisation). Names the first non-finite unknown, the iteration and the
    time, and the last iteration at which the state was checked finite."""
    xp = getattr(solver, "xp", np)
    X = solver.X
    if bool(xp.isfinite(X).all()):
        solver._finite_ok = (solver.iteration, solver.t)
        return
    from ..errors import NumericalError
    bad, count = "?", 0
    try:
        system = solver.system
        for v, c in zip(system.variables, system.scatter(X), strict=True):
            n = int(xp.count_nonzero(~xp.isfinite(c)))
            if n:
                bad, count = v.name, n
                break
    except Exception:  # noqa: BLE001 -- naming the field is a courtesy; the error is raised anyway
        pass
    last = getattr(solver, "_finite_ok", None)
    since = (f"; last checked finite at iteration {last[0]} (t = {last[1]:.10g})" if last
             else "")
    raise NumericalError(
        f"non-finite state (NaN or inf) at iteration {solver.iteration}, t = {solver.t:.10g}: "
        f"unknown {bad!r} ({count} non-finite coefficient(s)){since}",
        where=where,
        why="the run has blown up (a numerical instability, or a physical one the resolution "
            "cannot follow); stepping further would only produce NaN outputs and checkpoints",
        fix=f"reduce dt (now {solver.dt:g}; sim.cfl('u', safety=0.5) gives the advective limit), "
            "raise the resolution N, check the initial state and the parameters, or use a more "
            "stable scheme; check_finite=False turns this check off (declared by explain())",
        doc="docs/site/guide/timesteppers.md")


def check_steps(nsteps: Any, where: str, name: str = "nsteps") -> int:
    """A number of steps is a non-negative integer (a float or a bool is refused, never truncated:
    ``run(2.7)`` used to take 2 steps without a word). Shared by every ``run``."""
    import numbers
    if isinstance(nsteps, bool) or not isinstance(nsteps, numbers.Integral) or nsteps < 0:
        raise OptionError(f"{where}: {name}={nsteps!r} is not a number of steps",
                          why="steps must be an integer greater than or equal to zero",
                          fix=f"an integer >= 0, e.g. {name}=100"
                              + (f" (or {name}=round({nsteps!r}) if that is meant)"
                                 if isinstance(nsteps, float) and nsteps >= 0 else ""))
    return int(nsteps)


def check_substeps(n: Any, where: str) -> int:
    """``startup_substeps``: an integer >= 1 (the RK443 startup step is dt / n; a float used to be
    truncated, 2.5 -> 2, and 0 divided dt by zero at the first step). Shared by every IVP solver."""
    from ..errors import check_count
    return check_count(n, "startup_substeps", where)


def backend_hint(backend: Any) -> str:
    """The suggestion for a refused ``backend=`` value (``"CUDA"`` gets a hint, and
    the hint quotes only words the user wrote)."""
    b = str(backend)
    low = b.strip().lower()
    if low in ("cpu", "cuda"):
        return f" Did you mean {low!r}? (backend names are lower case)"
    if low in ("gpu", "cupy", "nvidia", "device", "cuda12", "cuda13"):
        return f" A GPU backend is spelled 'cuda' here (not {b!r})."
    if low in ("numpy", "host", "scipy"):
        return f" The NumPy (host) backend is spelled 'cpu' here (not {b!r})."
    return did_you_mean(low, ("cpu", "cuda"))


def check_dt_backend(dt: Any, backend: Any, where: str) -> None:
    """The time step and the backend of every IVP solver (Cartesian, polar, spherical,
    distributed, ensemble): one check, one message (the polar and spherical
    solvers raised bare ValueError without the fix, the Cartesian one an OptionError)."""
    if not (isinstance(dt, (int, float, np.floating, np.integer)) and not isinstance(dt, bool)
            and dt > 0 and np.isfinite(dt)):
        raise OptionError(f"{where}: dt={dt!r} must be a finite number > 0",
                          why="the time step is part of the problem's numerics",
                          fix="dt=1e-3 (a CFL estimate is a stability indicator: sim.cfl('u', safety=0.5))")
    if backend not in ("cpu", "cuda"):
        raise OptionError("backend must be 'cpu' or 'cuda' (explicit choice)",
                          where=f"{where}, backend={backend!r}",
                          why="the backend is an explicit choice",
                          fix="backend='cpu', or backend='cuda' for an NVIDIA GPU (a job on "
                              "a GPU node)." + backend_hint(backend))


def check_params_unchanged(solver: Any, where: str) -> None:
    """``plan.params`` is read at every evaluation of F, but L and M were assembled and factored
    with the values at construction: a later change (``solver.plan.params["Ra"] = 4000``) was
    ignored by the implicit part and used by the explicit part, silently.
    Refused at the next step."""
    p0 = solver.__dict__.get("_params0")
    if p0 is None or dict(solver.plan.params) == p0:
        return
    now = dict(solver.plan.params)
    diff = ", ".join(f"{k}: {p0.get(k)!r} -> {now.get(k)!r}" for k in sorted(set(p0) | set(now))
                     if p0.get(k) != now.get(k))
    raise OptionError(f"{where}: plan.params changed after the solver was built ({diff})",
                      why="the factored L and M keep the values they were built with, while F "
                          "reads plan.params at every stage: the two sides would disagree",
                      fix="build a new solver: IVPSolver(discretize(problem, params={...}), ...), "
                          "restarting from the current state (solver.X)")


#: Suffixes that name another file format: ``save("state.hdf")`` wrote ``state.hdf.npz`` (a NumPy
#: archive under a name that reads as HDF5), ``save("state.nc")`` ``state.nc.npz`` (review of the
#: audit "inputs"). Refused with the two formats a checkpoint has; any other suffix
#: (``state_t1.5`` has the suffix ".5") keeps the historical rule: ``.npz`` appended.
_OTHER_FORMATS = (".hdf", ".hd5", ".he5", ".h4", ".hdf4", ".nc", ".nc4", ".cdf", ".zarr", ".mat",
                  ".npy", ".pkl", ".pickle", ".json", ".txt", ".csv", ".dat", ".bin", ".xmf",
                  ".xdmf", ".vtk", ".vtu", ".vti")


def check_checkpoint_suffix(path: Path, where: str) -> None:
    """Refuse a checkpoint path whose suffix names a format other than HDF5 (``.h5``/``.hdf5``)
    and NumPy (``.npz``): the file would be a NumPy archive with ``.npz`` appended."""
    suf = path.suffix.lower()
    if suf in _OTHER_FORMATS:
        stem = path.name[: -len(path.suffix)]
        hdf = suf in (".hdf", ".hd5", ".he5", ".h4", ".hdf4")
        raise OptionError(
            f"{where}({str(path)!r}): {path.suffix!r} is not a checkpoint format",
            why="a checkpoint is HDF5 ('.h5' or '.hdf5', readable without Conatus) or NumPy "
                f"('.npz'); this path would have been written as {path.name}.npz"
                + (" (a NumPy file, not HDF5)" if hdf else ""),
            fix=f"{where}('{stem}.h5') for HDF5, or {where}('{stem}.npz') for NumPy",
            doc="docs/site/guide/io.md")


def check_time_parameter(plan: Any, time_parameter: Any, where: str) -> None:
    """``time_parameter`` is None or the name of a Parameter of the problem's equations, the one
    check of every IVP solver (flat, polar, spherical, distributed). Audit inputs:
    ``time_parameter="tt"`` was kept and never set, so a forcing in the real time parameter
    stayed at its declared value without a word; the polar, spherical and distributed solvers
    kept it too (review of the audit)."""
    if time_parameter is None:
        return
    from ..discretize.param_keys import problem_parameter_names
    used = problem_parameter_names(plan.problem)
    if not isinstance(time_parameter, str) or time_parameter not in used:
        raise OptionError(
            f"{where}: time_parameter={time_parameter!r} is not a parameter of the problem."
            + did_you_mean(str(time_parameter), sorted(used)),
            why="the time parameter is set to the stage time at every evaluation of F; "
                "a name the equations do not use would leave the forcing frozen",
            fix="the name of the time Parameter of the equations (parameters: "
                + (", ".join(sorted(used)) or "none") + "), or drop time_parameter=",
            doc="docs/site/guide/timesteppers.md")


def check_callback(callback: Any, every: Any, where: str) -> None:
    """``callback`` and ``every`` go together and are both explicit; ``every`` is an
    integer >= 1 (a float or a bool is refused, not truncated). Shared by every ``run``."""
    import numbers
    if callback is not None and every is None:
        raise OptionError(
            f"{where}: callback given without every= (callback and every go together)",
            why="the call frequency is part of the run (cost, what a monitor records); "
                "it has no default",
            fix="pass every=1 to call it after each step, or every=k for every k-th step")
    if callback is None and every is not None:
        raise OptionError(f"{where}: every= given without a callback (they go together)",
                          fix="pass callback=cb, or drop every=")
    if every is not None and (isinstance(every, bool) or not isinstance(every, numbers.Integral)
                              or every < 1):
        raise OptionError(f"{where}: every={every!r} must be an integer >= 1",
                          fix="every=1 calls the callback after each step")
    if callback is not None and not callable(callback):
        # refused before the first step (audit inputs: a bare TypeError at the first
        # call, after `every` steps had run)
        raise OptionError(f"{where}: callback={callback!r} is not callable",
                          fix="callback=lambda sim: ..., a function of the simulation")
