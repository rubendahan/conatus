"""One entry point for AI agents that use Conatus live: ``python -m conatus agent``.

Every command prints one JSON object on stdout (keys sorted, ``"api": 1``; within an API version
fields are only ever added, never renamed or removed) and says what happened by its exit code, so
an agent never has to parse prose. The contract is documented in ``AGENTS.md`` and in
``docs/site/guide/agents.md``; the development test suite freezes the keys.

    python -m conatus agent                       # = info: versions, CPU/GPU, user layer, runs
    python -m conatus agent info --no-probe       # skip the GPU probe (a few seconds)
    python -m conatus agent explain cn.Model.simulate     # signature, required arguments, doc
    python -m conatus agent check script.py       # dry run: refusals, missing tolerances, memory
    python -m conatus agent check run.py --memory -- --Ra 1e6   # script arguments after --
    python -m conatus agent explain script.py     # refused (never runs a script): use check

``check`` executes the script for real up to its **first expensive call** (``simulate``,
``eigenvalues``, ``solve``, a Newton search, a continuation, ...), where it stops: the call's
arguments are checked against its signature (missing arguments and tolerances that have no
default), the model is discretized at the requested resolution (so the refusals of the
compiler and of the plan, an under-resolved coefficient for instance, appear), and the
``explain()`` texts and, for ``backend="cuda"``, the predicted device memory are reported. No
Conatus call in ``EXPENSIVE`` time-steps or solves, and nothing is changed: a refusal is reported
with its ``fix`` as written by Conatus, never applied.

What ``check`` does not sandbox: the script runs in this process with the user's privileges, so
its own code (``open(..., "w")``, ``np.save``, subprocesses, network) runs as written; only the
Conatus calls listed in ``EXPENSIVE`` are intercepted, and only in this process (a child process
or an MPI launch is not). The interception is process-wide while ``check`` runs (one ``check`` at
a time, serialized by a lock; another thread calling Conatus meanwhile is stopped too), and the
wrappers fall back to the original functions once it returns. A script that catches the stop
(bare ``except:`` or ``except BaseException``) and goes on is reported as an error. The timeout
uses ``SIGALRM``: only in the main thread, and it cannot interrupt a long call inside compiled code.

Exit codes: 0 ok; 1 ``check`` found a refusal, a missing argument or a script error; 2 usage;
4 ``explain`` of a name that does not exist.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata as md
import inspect
import json
import os
import platform
import runpy
import signal
import socket
import sys
import threading
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

from ._explain import full_explain   # the complete explain() text kept by reports

# Documented names only (M5): ``API`` (the "api" field of every answer) and ``main`` (the
# command line, python -m conatus agent) stay importable.
__all__ = ["EXPENSIVE", "info", "describe", "check"]

API = 1

# Calls ``check`` intercepts: (module, owner class or None, attribute). It stops at the first
# one that would integrate, solve or iterate; everything before it runs for real, except the
# side effects below (skipped) and the constructors of PASS_THROUGH (run, then inspected).
EXPENSIVE = (
    ("conatus.model.model", "Model", "simulate"),
    ("conatus.model.model", "Model", "eigenvalues"),
    ("conatus.model.model", "Model", "solve"),
    ("conatus.model.run", "Simulation", "run"),
    ("conatus.model.run", "Linearization", "eigenvalues"),
    ("conatus.model.run", "Linearization", "floquet"),
    ("conatus.model.run", "Linearization", "optimal_gain"),
    ("conatus.model.run", "Linearization", "resolvent"),
    ("conatus.model.run", "Linearization", "hinf_norm"),
    ("conatus.model.run", "Linearization", "h2_norm"),
    ("conatus.timestep.integrator", "IVPSolver", "run"),
    ("conatus.timestep.integrator", "IVPSolver", "step"),
    ("conatus.solvers.plan_evp", None, "eigenvalues"),
    ("conatus.solvers.parametric", None, "critical"),
    ("conatus.ecs.solutions", None, "find_equilibrium"),
    ("conatus.ecs.solutions", None, "find_travelling_wave"),
    ("conatus.ecs.solutions", None, "find_relative_periodic_orbit"),
    ("conatus.bifurcation.equilibria", None, "continue_equilibria"),
    ("conatus.bifurcation.orbits", None, "continue_orbits"),
    ("conatus.bifurcation.codim2", None, "continue_codim2"),
    ("conatus.bifurcation.branching", None, "branch_switch"),
    ("conatus.manifolds.parametrization", None, "unstable_manifold"),
    ("conatus.chaos.lyapunov", None, "lyapunov_spectrum"),
    ("conatus.ensembles.fate", None, "ensemble_fate"),
    ("conatus.floquet.shell", None, "shell_floquet"),
    ("conatus.growth.optimize", None, "optimal_growth"),
    ("conatus.growth.optimize", None, "growth_at"),
    ("conatus.solvers.nlbvp", "NewtonSolver", "solve"),
    ("conatus.continuation.nlbvp", "NLBVPContinuation", "run"),
    ("conatus.ecs.solutions", None, "equilibrium_eigenvalues"),
    ("conatus.ecs.solutions", None, "floquet_multipliers"),
    ("conatus.ecs.edge", "EdgeTracker", "run"),
    ("conatus.ecs.edge", "EdgeTracker", "bisect"),
    ("conatus.ecs.edge", "EdgeTracker", "fate"),
    ("conatus.ecs.edge", "ThresholdBisection", "run"),
    ("conatus.adjoint.flowmap", "FlowMap", "jvp"),
    ("conatus.adjoint.flowmap", "FlowMap", "vjp"),
    ("conatus.adjoint.flowmap", "FlowMap", "vjp_physical"),
    ("conatus.adjoint.flowmap", "FlowMap", "trajectory"),
    ("conatus.adjoint.nlop", "NLOP", "J"),
    ("conatus.adjoint.nlop", "NLOP", "gradient"),
    ("conatus.adjoint.nlop", "NLOP", "linear_gain"),
    ("conatus.adjoint.nlop", "NLOP", "iterate"),
    ("conatus.adjoint.nlop", "NLOP", "fd_check"),
    ("conatus.ensembles.ensemble", "Ensemble", "run"),
    ("conatus.nonmodal.pseudospectra", None, "max_growth"),
    ("conatus.nonmodal.pseudospectra", None, "pseudospectra"),
    ("conatus.nonmodal.pseudospectra", None, "kreiss_constant"),
    ("conatus.model.run", "Simulation", "watch"),
    ("conatus.model.run", "Simulation", "output"),
    ("conatus.model.run", "Simulation", "checkpoints"),
    ("conatus.model.run", "Simulation", "save"),
    ("conatus.model.run", "Simulation", "resume_or_start"),
)

# Run for real (cheap: compile, discretize, factorize on the CPU), then inspected: an expensive
# search that takes the solver (``find_equilibrium(sim.solver, ...)``) is reached and checked.
# Only with ``backend="cpu"``: a CUDA solver would allocate device memory, so ``check`` stops at
# ``simulate`` instead and predicts that memory.
PASS_THROUGH = ("Model.simulate",)

# Skipped by ``check``: they write files or announce a Studio run.
SIDE_EFFECTS = ("Simulation.watch", "Simulation.output", "Simulation.checkpoints",
                "Simulation.save")
# Run by ``check`` without their file side effect (the dry
# run of resume_or_start created the checkpoint directory): an existing directory is read as in
# the real run (a restart reads, never writes); a missing one is not created, and the initial
# state is set as on a fresh start.
NO_WRITE = ("Simulation.resume_or_start",)

# Calls whose argument VALUES ``check`` validates at the stop, by the side-effect-free method
# the call itself runs first (review 0.1.1, F13: run(steps=-1) was reported ok). For every other
# stop only the signature, the conditional tolerances and the discretization are checked; the
# report's ``validated`` field says which applied.
PREFLIGHT = {"Simulation.run": "_run_arguments", "IVPSolver.run": "_run_arguments"}

# Arguments that a call requires only in some cases (tolerances): they hide behind
# ``**kw`` or are required by one option, so a signature does not show them. (call, condition as
# text, condition, argument, value Conatus suggests in its refusal or None).
_CONDITIONAL = (
    ("Model.eigenvalues", "drift=N2", lambda kw: kw.get("drift") is not None, "drift_tol",
     "1e-6"),
    ("Linearization.eigenvalues", "drift=N2", lambda kw: kw.get("drift") is not None,
     "drift_tol", "1e-6"),
    ("eigenvalues", "drift_plan=...", lambda kw: kw.get("drift_plan") is not None, "drift_tol",
     "1e-6"),
    ("Linearization.floquet", "method='hill'", lambda kw: kw.get("method") == "hill", "tol",
     "1e-12"),
    ("Linearization.floquet", "method='hill'", lambda kw: kw.get("method") == "hill", "Nh", None),
    ("Linearization.floquet", "method='hill'", lambda kw: kw.get("method") == "hill", "targets",
     None),
    ("Linearization.floquet", "method='monodromy'", lambda kw: kw.get("method") == "monodromy",
     "nsteps", None),
    ("Linearization.floquet", "method='monodromy'", lambda kw: kw.get("method") == "monodromy",
     "scheme", None),
)

DOCS = {"agents": "AGENTS.md", "guide": "docs/site/guide/agents.md",
        "live_runs": "docs/site/guide/studio_agents.md", "errors": "docs/site/guide/errors.md"}

COMMANDS = {
    "python -m conatus agent [info]": "versions, CPU/GPU, user layer, watched runs (JSON)",
    "python -m conatus agent explain <name>": "signature, required arguments, doc of a public name",
    "python -m conatus agent check <script.py>": "dry run up to the first expensive call (JSON)",
    "python -m conatus doctor --json": "full installation check (slower), exit 1 if required is missing",
    "python -m conatus studio status|watch|send|frame|log <run>": "follow and pilot a watched run",
}


def _dump(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, indent=1, default=_jsonable)


def _jsonable(v: Any) -> Any:
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist() if v.size <= 64 else f"<array shape={v.shape} dtype={v.dtype}>"
    except ImportError:                                   # numpy is a hard dependency; be safe
        pass
    if isinstance(v, complex):
        return {"re": v.real, "im": v.imag}
    if isinstance(v, (set, tuple)):
        return list(v)
    if isinstance(v, Path):
        return str(v)
    return repr(v)


# =========================================================================================== info
def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def _gpu(probe: bool) -> dict:
    if not probe:
        return {"probed": False, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "how": "python -m conatus agent info (without --no-probe)"}
    from .gpu import _core_status, _cupy_status
    cu, cu_why = _cupy_status()
    co, co_why = _core_status()
    return {"probed": True, "available": bool(cu and co), "cupy": cu_why, "core": co_why,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "backend": "cuda" if cu and co else "cpu"}


def _runs(limit: int = 20) -> dict:
    try:
        from .studio.client import SpoolClient
        from .telemetry import default_spool
        rows = SpoolClient(default_spool()).runs().get("runs", [])
    except Exception as err:  # noqa: BLE001 -- the run list is optional context, report why not
        return {"available": False, "error": f"{type(err).__name__}: {err}"}
    rows = sorted(rows, key=lambda r: r.get("age_s") or 0.0)[:limit]
    keep = ("run_id", "state", "t", "iteration", "dt", "age_s", "commands", "settable")
    return {"available": True, "runs": [
        {**{k: r.get(k) for k in keep}, "alerts": [a.get("code") for a in r.get("alerts", [])]}
        for r in rows], "more": "python -m conatus studio status <run_id>"}


def _user_layer() -> dict:
    import conatus as cn
    from . import _surface
    mods = {}
    for m in _surface.USER_MODULES:
        mod = importlib.import_module(f"conatus.{m}")
        mods[f"cn.{m}"] = sorted(getattr(mod, "__all__", []))
    return {"cn": sorted(n for n in cn.__all__ if not n.startswith("_")), "modules": mods,
            "expert": "conatus.expert (python -m conatus._surface prints every signature)"}


def info(*, probe: bool = True, runs: bool = True) -> dict:
    """What an agent needs before writing a line: versions, CPU/GPU, the user layer, live runs."""
    import conatus as cn
    out = {"api": API, "command": "info", "conatus": cn.__version__,
           "python": platform.python_version(), "platform": platform.platform(),
           "host": socket.gethostname(), "cpus": os.cpu_count(),
           "packages": {p: _version(p) for p in ("numpy", "scipy", "h5py", "cupy", "cupy-cuda12x",
                                                   "mpi4py", "matplotlib")},
           "gpu": _gpu(probe), "surface": _user_layer(), "commands": COMMANDS, "docs": DOCS,
           "rules": ["read every refusal: its 'fix' is a suggestion for you to apply",
                     "tolerances that decide digits have no default: write them",
                     "validate by residuals and explain(), never by a bare value",
                     "check a script first: python -m conatus agent check script.py"]}
    if runs:
        out["runs"] = _runs()
    return out


# ======================================================================================= describe
def _resolve(name: str) -> tuple[str, Any]:
    """``cn.X.y`` / ``expert.X`` / ``conatus.mod.X`` -> object (raises LookupError)."""
    parts = name.split(".")
    if parts[0] == "cn":
        parts[0] = "conatus"
    elif parts[0] == "expert":
        parts = ["conatus", "expert"] + parts[1:]
    if parts[0] != "conatus":
        parts = ["conatus"] + parts
    obj, i = None, len(parts)
    while i > 0:                                         # longest importable module prefix
        try:
            obj = importlib.import_module(".".join(parts[:i]))
            break
        except ImportError:
            i -= 1
    if obj is None:
        raise LookupError(name)
    for p in parts[i:]:
        try:
            obj = inspect.getattr_static(obj, p) if inspect.isclass(obj) else getattr(obj, p)
        except AttributeError:
            raise LookupError(name) from None
        if isinstance(obj, (staticmethod, classmethod)):
            obj = obj.__func__
    return ".".join(parts), obj


# Model methods that declare an unknown, and the default name of those that have one.
_DECLARE = {"scalar": None, "vector": None, "velocity": "u", "pressure": "p", "dye": None}


def declared_unknowns(func: Any) -> list[dict]:
    """Unknowns a model factory declares (``m.velocity("u")``, ``m.scalar("T")``, ...), read
    from its source and from the module-level functions of the same module it calls, in the
    order written; nothing is called, so ``agent explain`` of a ready model lists its unknowns, not only its
    arguments. ``[]`` when there is no source or none found."""
    import ast
    func = inspect.unwrap(func)
    mod = inspect.getmodule(func)
    try:
        tree = ast.parse(inspect.getsource(mod))
    except (OSError, TypeError, SyntaxError):
        return []
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    out: list[dict] = []
    seen: set[str] = set()

    def walk(fname: str) -> None:
        if fname in seen or fname not in defs:
            return
        seen.add(fname)
        for node in ast.walk(defs[fname]):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr in _DECLARE:
                arg = node.args[0] if node.args else None
                name = arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                    else (_DECLARE[f.attr] if not node.args else None)
                if name and all(u["name"] != name for u in out):
                    out.append({"name": name, "kind": f.attr})
            elif isinstance(f, ast.Name):
                walk(f.id)
    walk(func.__name__)
    return out


def describe(name: str) -> dict:
    """Signature, required arguments (no default, or a tolerance ``REQUIRED``), first paragraph
    of the docstring and, when the object has an argument-free ``explain()``, its text."""
    from .tolerances import is_required
    qual, obj = _resolve(name)
    kind = ("module" if inspect.ismodule(obj) else "class" if inspect.isclass(obj)
            else "property" if isinstance(obj, property) else "function" if callable(obj)
            else type(obj).__name__)
    out: dict[str, Any] = {"api": API, "command": "explain", "name": name, "qualified": qual,
                           "kind": kind}
    doc = inspect.getdoc(obj) if (callable(obj) or inspect.ismodule(obj)
                                  or isinstance(obj, property)) else None
    out["summary"] = doc.split("\n\n")[0].strip() if doc else None
    if not (callable(obj) or inspect.ismodule(obj) or isinstance(obj, property)):
        out["value"] = repr(obj) if len(repr(obj)) < 200 else f"<{type(obj).__name__}>"
    if callable(obj) and kind != "module":
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            sig = None
        if sig is not None:
            out["signature"] = str(sig)
            req, tol = [], []
            for p in sig.parameters.values():
                if p.name in ("self", "cls") or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                    continue
                if p.default is p.empty:
                    req.append(p.name)
                elif is_required(p.default):
                    tol.append(p.name)
            out["required"], out["required_tolerances"] = req, tol
        from ._deprecation import aliases_of
        try:
            dep = aliases_of(obj)
        except Exception:  # noqa: BLE001 -- an object without aliases
            dep = {}
        if dep:            # former argument names still accepted (DeprecationWarning)
            out["deprecated"] = dep
        cond = [{"argument": a, "when": w, "suggest": s, "kind": "tolerance" if s else "required"}
                for c, w, _, a, s in _CONDITIONAL if qual.endswith("." + c)]
        if cond:
            out["conditional"] = cond
        ret = str(getattr(sig, "return_annotation", "")) if sig is not None else ""
        if inspect.isfunction(inspect.unwrap(obj)) and ("Model" in ret or "Linearization" in ret):
            unknowns = declared_unknowns(obj)
            if unknowns:                    # a ready model: what it evolves, as m.explain() says
                out["unknowns"] = unknowns
                out["unknowns_text"] = ", ".join(f"{u['name']} ({u['kind']})" for u in unknowns)
    if inspect.isclass(obj):
        out["methods"] = sorted(k for k, v in vars(obj).items()
                                if not k.startswith("_") and callable(v))
    if inspect.ismodule(obj):
        out["names"] = sorted(getattr(obj, "__all__", []))
    ex = getattr(obj, "explain", None)
    if ex is not None and not inspect.isclass(obj) and not inspect.isfunction(obj):
        try:
            if not inspect.signature(ex).parameters:
                out["explain"] = ex()
        except (TypeError, ValueError):
            pass
    return out


# ========================================================================================== check
class _Stop(BaseException):
    """Raised at the first expensive call (BaseException: a script's ``except Exception`` must
    not swallow it)."""


class _Timeout(BaseException):
    pass


def _script_line(path: str) -> int | None:
    for fr in inspect.stack()[2:]:
        if os.path.abspath(fr.filename) == path:
            return fr.lineno
    return None


def _error(err: BaseException, path: str) -> dict:
    tb = traceback.extract_tb(err.__traceback__)
    line = next((f.lineno for f in reversed(tb) if os.path.abspath(f.filename) == path), None)
    from .errors import ConatusError
    d = {"type": type(err).__name__, "line": line,
         "conatus": isinstance(err, ConatusError)}
    if isinstance(err, ConatusError):
        d.update(message=err.message, where=err.where, why=err.why, fix=err.fix, doc=err.doc)
    else:
        d["message"] = str(err)
        d["at"] = f"{tb[-1].filename}:{tb[-1].lineno}" if tb else None
    return d


_LOCK = threading.Lock()                                # one check at a time (patches are global)



class _Checker:
    """Patches the expensive calls for the duration of one ``check``."""

    def __init__(self, path: str, memory: bool) -> None:
        self.path, self.memory = path, memory
        self.report: dict[str, Any] = {"missing": [], "refusals": [], "explain": {},
                                       "memory": None, "stopped_at": None, "plan": None,
                                       "calls": [], "notes": [], "validated": []}
        self._undo: list[tuple[Any, str, Any]] = []
        self._wrappers: dict[int, tuple[Any, Any]] = {}
        self.active = False            # wrappers that outlive the check call the original
        self.stops = 0                 # > 1: the script caught a stop and went on

    # -- patching ---------------------------------------------------------------------------
    def install(self) -> None:
        for mod, owner, attr in EXPENSIVE:
            label = f"{owner}.{attr}" if owner else attr
            try:
                m = importlib.import_module(mod)
            except ImportError:
                # a module this installation does not have (cut from the public package, or an
                # optional dependency missing): the script cannot call it either, so no note
                # (0.1.1 leftover: one note per absent module on every check was noise)
                continue
            target = getattr(m, owner, None) if owner else m
            orig = vars(target).get(attr) if target is not None else None
            if orig is None:                       # the table is out of date: say so, never hide it
                self.report["notes"].append(f"{label} ({mod}) not found: a call to it would run "
                                            "for real (stale table in conatus.agent)")
                continue
            new = self._wrap(label, orig)
            self._patch(target, attr, new)
            self.active = True
            if owner is None:                      # re-exports already bound (packages, expert)
                for name, other in list(sys.modules.items()):
                    if (name.startswith("conatus") and other is not None and other is not m
                            and vars(other).get(attr) is orig):
                        self._patch(other, attr, new)

    def _patch(self, target: Any, attr: str, new: Any) -> None:
        self._undo.append((target, attr, target.__dict__[attr] if attr in target.__dict__
                           else None))
        setattr(target, attr, new)

    def uninstall(self) -> None:
        self.active = False
        for target, attr, old in reversed(self._undo):
            if old is None:
                with contextlib.suppress(AttributeError, TypeError):
                    delattr(target, attr)
            else:
                setattr(target, attr, old)
        # modules the script imported while patched bound the wrappers: give them the originals
        names = {attr for _, _, attr in EXPENSIVE}
        for mod in list(sys.modules.values()):    # conatus and the script's own modules
            try:
                attrs = names & set(vars(mod))
            except (TypeError, AttributeError):
                continue
            for attr in attrs:
                hit = self._wrappers.get(id(vars(mod)[attr]))
                if hit is not None and hit[0] is vars(mod)[attr]:
                    setattr(mod, attr, hit[1])

    def _resume_or_start(self, label: str, line: Any, orig: Any, args: tuple,
                         kwargs: dict) -> Any:
        """``resume_or_start`` without creating its directory (see ``NO_WRITE``)."""
        from pathlib import Path
        sim, rest = (args[0], args[1:]) if args else (None, ())
        directory = rest[0] if rest else kwargs.get("directory")
        if sim is None or directory is None or Path(directory).exists():
            out = orig(*args, **kwargs)
            self.report["calls"].append({"call": label, "line": line, "action": "ran",
                                         "why": "the directory exists: read as in the real run"})
            return out
        initial = {k: v for k, v in kwargs.items() if k not in ("directory", "allow", "check")}
        if not initial:                    # the real call's refusal, without the mkdir
            return orig(sim, directory, **kwargs)
        sim.initial(check=kwargs.get("check", True), **initial)
        sim._chain = ("fresh", f"resume_or_start({str(directory)!r}): no checkpoint there "
                      "(agent check: directory not created): started from the initial state")
        self.report["calls"].append({"call": label, "line": line, "action": "ran without writing",
                                     "why": f"{directory} does not exist: initial state set, "
                                            "directory not created by check"})
        return sim

    def _wrap(self, label: str, orig: Any):
        checker = self

        def stop(*args: Any, **kwargs: Any) -> Any:
            if not checker.active:             # bound by someone during the check: now inert
                return orig(*args, **kwargs)
            return checker.at_call(label, orig, args, kwargs)
        stop.__wrapped__ = orig
        self._wrappers[id(stop)] = (stop, orig)
        return stop

    # -- the calls --------------------------------------------------------------------------
    def at_call(self, label: str, orig: Any, args: tuple, kwargs: dict) -> Any:
        """Skip a side effect, run a cheap constructor for real, or stop at an expensive call."""
        line = _script_line(self.path)
        rep = self.report
        if label in SIDE_EFFECTS:
            rep["calls"].append({"call": label, "line": line, "action": "skipped",
                                 "why": "writes files or a Studio run; not done by check"})
            return None
        if label in NO_WRITE:
            return self._resume_or_start(label, line, orig, args, kwargs)
        if rep["stopped_at"] is not None:      # the script caught the first stop: stop again
            self.stops += 1
            raise _Stop(label)
        given = self._arguments(label, orig, args, kwargs)
        if label in PASS_THROUGH and given.get("backend", "cpu") != "cpu":
            rep["notes"].append(f"{label} at line {line}: backend={given['backend']!r} not built "
                                "(it would allocate device memory); stopped after discretizing")
        elif label in PASS_THROUGH:
            try:
                sim = orig(*args, **kwargs)
            except Exception as err:  # noqa: BLE001 -- inspected below, then stop or re-raise
                from .errors import BackendError
                if not isinstance(err, BackendError):
                    raise
                rep["notes"].append(f"{label} at line {line}: {type(err).__name__}: "
                                    f"{str(err).splitlines()[0]}; stopped after discretizing")
            else:
                rep["calls"].append({"call": label, "line": line, "action": "ran"})
                self._collect_simulation(sim, given)
                return sim
        rep["calls"].append({"call": label, "line": line, "action": "stopped"})
        rep["stopped_at"] = {"call": label, "line": line}
        self._preflight(label, line, args, kwargs)
        self._discretize(label, args, given)
        self.stops += 1
        raise _Stop(label)

    def _preflight(self, label: str, line: Any, args: tuple, kwargs: dict) -> None:
        """The value checks of the stopped call (:data:`PREFLIGHT`), recorded in ``validated``;
        a refusal is reported like any other."""
        from .errors import ConatusError
        attr = PREFLIGHT.get(label)
        fn = getattr(args[0], attr, None) if (attr and args) else None
        if fn is None:
            self.report["validated"].append(
                {"call": label, "values": "not checked", "checked": "signature, conditional "
                 "tolerances and discretization; the argument values are checked only when the "
                 "call runs"})
            return
        self.report["validated"].append({"call": label, "values": "checked",
                                         "checked": f"{label} argument checks ({attr})"})
        try:
            fn(*args[1:], **kwargs)
        except ConatusError as err:
            e = _error(err, self.path)
            e["line"] = line
            self.report["refusals"].append(e)
        except TypeError:          # an argument the signature lacks: already in "missing"
            pass

    def _arguments(self, label: str, orig: Any, args: tuple, kwargs: dict) -> dict:
        """Arguments as given; record missing required arguments and tolerances."""
        from .tolerances import is_required
        rep = self.report
        try:
            sig = inspect.signature(orig)
            bound = sig.bind_partial(*args, **kwargs)
            for p in sig.parameters.values():
                if p.name in bound.arguments or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                    continue
                if p.default is p.empty:
                    rep["missing"].append({"call": label, "argument": p.name,
                                           "kind": "required"})
                elif is_required(p.default):
                    rep["missing"].append({"call": label, "argument": p.name,
                                           "kind": "tolerance"})
            given = dict(bound.arguments)
            for p in sig.parameters.values():
                if p.kind == p.VAR_KEYWORD:
                    given.update(given.pop(p.name, {}) or {})
        except TypeError as err:
            rep["missing"].append({"call": label, "argument": None, "kind": "signature",
                                   "message": str(err)})
            given = dict(kwargs)
        for call, when, cond, arg, suggest in _CONDITIONAL:
            if call == label and cond(given) and arg not in given:
                rep["missing"].append({"call": label, "argument": arg, "kind": "tolerance"
                                       if suggest else "required", "suggest": suggest,
                                       "when": when})
        return given

    def _collect_simulation(self, sim: Any, given: dict) -> None:
        rep = self.report
        with contextlib.suppress(Exception):
            rep["explain"]["simulation"] = full_explain(sim)
        plan = getattr(sim, "plan", None)
        if plan is not None and hasattr(plan, "explain"):
            rep["explain"]["plan"] = plan.explain()
            groups = getattr(plan, "groups", None)
            rep["plan"] = {"type": type(plan).__name__,
                           "groups": len(groups) if groups is not None else None}
            if given.get("backend", "cpu") == "cuda" or self.memory:
                self._memory(plan, given.get("scheme"))

    def _discretize(self, label: str, args: tuple, given: dict) -> None:
        """Compile and discretize what the call would (refusals of compiler and plan surface
        here), collect explain() texts and, for backend='cuda', the predicted device memory."""
        rep = self.report
        obj = args[0] if args else None
        try:
            plan = None
            if label.startswith("Model."):
                N, dealias = given.get("N"), given.get("dealias")
                rep["explain"]["model"] = full_explain(obj, N, dealias)
                ncc = {k: given[k] for k in ("ncc_cutoff", "ncc_tail_tol") if k in given}
                plan = obj.compile(N, dealias).discretize(**ncc)
            elif label.startswith("Linearization."):
                rep["explain"]["linearization"] = full_explain(obj, given.get("N"))
                if label != "Linearization.floquet":
                    plan = obj.discretize(given.get("N"), given.get("dealias"), k=given.get("k"))
            elif label.startswith("Simulation."):
                rep["explain"]["simulation"] = full_explain(obj)
                plan = getattr(obj, "plan", None)
            elif label.startswith("IVPSolver."):
                rep["explain"]["solver"] = full_explain(obj)
                plan = getattr(obj, "plan", None)
            elif label == "eigenvalues" and obj is not None:
                plan = obj
            elif "." in label and obj is not None and hasattr(obj, "explain"):
                with contextlib.suppress(Exception):      # NewtonSolver, FlowMap, NLOP, ...
                    rep["explain"][label.split(".")[0].lower()] = obj.explain()
            if plan is not None and hasattr(plan, "explain"):
                rep["explain"]["plan"] = plan.explain()
                groups = getattr(plan, "groups", None)
                rep["plan"] = {"type": type(plan).__name__,
                               "groups": len(groups) if groups is not None else None}
            if label == "Model.simulate":
                scheme, backend = given.get("scheme"), given.get("backend", "cpu")
                from .timestep.integrator import SCHEMES
                if scheme is not None and scheme not in SCHEMES:
                    rep["refusals"].append({"type": "OptionError", "conatus": True, "line":
                                            rep["stopped_at"]["line"], "message":
                                            f"unknown scheme {scheme!r}", "fix":
                                            f"one of {', '.join(SCHEMES)}", "where": label,
                                            "why": None, "doc": None})
                if plan is not None and (backend == "cuda" or self.memory) and scheme in SCHEMES:
                    self._memory(plan, scheme)
            elif (label.startswith(("Simulation.", "IVPSolver.")) and plan is not None
                  and rep["memory"] is None):
                solver = getattr(obj, "solver", obj)
                if getattr(solver, "backend", "cpu") == "cuda" or self.memory:
                    self._memory(plan, getattr(solver, "scheme", "RK443"))
        except _Timeout:
            raise
        except Exception as err:  # noqa: BLE001 -- a refusal of the discretization is the result
            rep["refusals"].append(_error(err, self.path))

    def _memory(self, plan: Any, scheme: str) -> None:
        try:
            from .runtime.memory import plan_memory
            m = plan_memory(plan, scheme, cufft_query=False)
            d = m.to_dict()
            self.report["memory"] = {"model": "CUDA backend, liveness planner "
                                              "(conatus.runtime.memory.plan_memory); declared, "
                                              "not measured", "scheme": scheme,
                                     "peak_bytes": d["predicted"]["peak_bound_bytes"],
                                     # + the declared CUDA process floor (context, kernels)
                                     "peak_with_process_floor_bytes":
                                         d["predicted"]["peak_with_floor_bytes"],
                                     "report": d}
        except Exception as err:  # noqa: BLE001 -- not every plan is modelled: say which
            self.report["memory"] = {"model": None,
                                     "not_modelled": f"{type(err).__name__}: {err}"}


def check(path: str | os.PathLike, *, argv: list[str] | None = None, timeout: float = 300.0,
          memory: bool = False) -> dict:
    """Dry run of ``path`` (see the module docstring). ``memory=True`` also predicts the device
    memory of a CPU simulation as if it ran with ``backend='cuda'``."""
    import conatus  # noqa: F401  -- the package must import before the script runs
    path = os.path.abspath(os.fspath(path))
    t0 = time.perf_counter()
    out: dict[str, Any] = {"api": API, "command": "check", "script": path}
    if not os.path.isfile(path):                  # same keys as any other report
        out.update(ok=False, calls=[], stopped_at=None, missing=[], refusals=[], warnings=[],
                   validated=[],
                   explain={}, plan=None, memory=None, notes=[],
                   errors=[{"type": "FileNotFoundError", "message": path, "line": None}],
                   elapsed_s=round(time.perf_counter() - t0, 3))
        return out
    with _LOCK:
        return _check_locked(path, argv, timeout, memory, out, t0)


def _check_locked(path: str, argv: list[str] | None, timeout: float, memory: bool, out: dict,
                  t0: float) -> dict:
    ck = _Checker(path, memory)
    caught: list[warnings.WarningMessage] = []
    old_argv, old_path = sys.argv, list(sys.path)
    old_alarm = None
    script_error = None
    stopped = False
    alarm = (bool(timeout) and hasattr(signal, "SIGALRM")
             and threading.current_thread() is threading.main_thread())
    if timeout and not alarm:
        ck.report["notes"].append("no timeout: check is not in the main thread (SIGALRM)")
    try:
        ck.install()
        sys.argv = [path] + list(argv or [])
        sys.path.insert(0, os.path.dirname(path))
        if alarm:
            def _on_alarm(signum, frame):                 # noqa: ARG001 -- signal signature
                raise _Timeout()
            old_alarm = signal.signal(signal.SIGALRM, _on_alarm)
            signal.setitimer(signal.ITIMER_REAL, timeout)
        with warnings.catch_warnings(record=True) as caught, \
                contextlib.redirect_stdout(sys.stderr):
            warnings.simplefilter("always")
            try:
                try:
                    runpy.run_path(path, run_name="__main__")
                finally:                                # no alarm while reporting
                    if old_alarm is not None:
                        signal.setitimer(signal.ITIMER_REAL, 0)
            except _Stop:
                stopped = True
            except _Timeout:
                script_error = {"type": "Timeout", "message": f"no expensive call reached "
                                f"within {timeout:g} s (the script itself is slow)"}
            except SystemExit as err:
                if err.code not in (None, 0):
                    script_error = {"type": "SystemExit", "message": str(err.code)}
            except Exception as err:  # noqa: BLE001 -- every failure of the script is reported
                script_error = _error(err, path)
    finally:
        if old_alarm is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_alarm)
        ck.uninstall()
        sys.argv, sys.path[:] = old_argv, old_path
    rep = ck.report
    if rep["stopped_at"] is not None and (not stopped or ck.stops > 1):
        rep.setdefault("errors", []).append(
            {"type": "StopCaught", "line": rep["stopped_at"]["line"], "message":
             "the script caught the stop of check (bare except: or except BaseException) and "
             "went on: the lines after it ran for real; let the stop propagate"})
    if script_error is not None:
        (rep["refusals"] if script_error.get("conatus") else rep.setdefault("errors", [])
         ).append(script_error)
    out.update(rep)
    out.setdefault("errors", [])
    out["warnings"] = [{"category": w.category.__name__, "message": str(w.message),
                        "line": w.lineno if os.path.abspath(w.filename) == path else None}
                       for w in caught]
    out["ok"] = not (rep["missing"] or rep["refusals"] or out["errors"])
    out["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return out


# ============================================================================================ CLI
def main(argv: list[str] | None = None) -> int:
    """``python -m conatus agent ...``; returns the exit code (module docstring)."""
    argv = sys.argv[1:] if argv is None else list(argv)
    p = argparse.ArgumentParser(prog="python -m conatus agent",
                                description="JSON interface for AI agents (api 1).")
    sub = p.add_subparsers(dest="cmd")
    i = sub.add_parser("info", help="versions, CPU/GPU, user layer, watched runs")
    i.add_argument("--no-probe", action="store_true", help="do not probe the GPU")
    i.add_argument("--no-runs", action="store_true", help="do not list the Studio runs")
    e = sub.add_parser("explain", help="a public name (cn.Model.simulate); never runs a script")
    e.add_argument("target")
    c = sub.add_parser("check", help="dry run of a script up to its first expensive call")
    c.add_argument("script")
    c.add_argument("--timeout", type=float, default=300.0,
                   help="seconds before giving up (default 300)")
    c.add_argument("--memory", action="store_true",
                   help="predict device memory as for backend='cuda' even on a CPU simulation")
    script_args: list[str] = []
    if "--" in argv:                                   # check script.py [options] -- script args
        k = argv.index("--")
        argv, script_args = argv[:k], argv[k + 1:]
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help")):
        # "agent --no-probe" was "unrecognized arguments"; the
        # default command takes its options too
        argv = ["info"] + argv
    try:
        a = p.parse_args(argv)
    except SystemExit as err:
        return 2 if err.code else 0
    if a.cmd == "info":
        print(_dump(info(probe=not a.no_probe, runs=not a.no_runs)))
        return 0
    if a.cmd == "explain" and a.target.endswith(".py"):
        # "explain FILE.py" ran the script (= check); explain only
        # reads names, so a script is refused with the command that runs it, declared
        print(_dump({"api": API, "command": "explain", "name": a.target, "error":
                     "explain reads public names and never runs a script",
                     "hint": f"python -m conatus agent check {a.target} (dry run: executes the "
                             "script up to its first expensive call)"}))
        return 2
    if a.cmd == "explain":
        try:
            print(_dump(describe(a.target)))
        except LookupError:
            print(_dump({"api": API, "command": "explain", "name": a.target, "error":
                         "no such public name", "hint": "cn.<name>, expert.<name> or "
                         "conatus.<module>.<name>; python -m conatus agent info lists cn.*"}))
            return 4
        return 0
    rep = check(a.script, argv=script_args, timeout=a.timeout, memory=a.memory)
    print(_dump(rep))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
