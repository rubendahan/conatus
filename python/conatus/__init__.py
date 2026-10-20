"""Conatus: a high-performance mathematical laboratory for PDEs and dynamical systems."""
__version__ = "0.1.1"  # single source of the version (pyproject reads it)
# The content of every conatus module this process imports is recorded (sha256 of its source):
# a module edited on disk afterwards is not the code that runs (conatus.provenance.code_state,
# changed_since_import, audit state).
from . import _import_record  # noqa: E402
_import_record.install(__file__, _import_record.__file__)


# Two layers: ``import conatus as cn`` is the user layer (native models
# cn.Box, cn.Model, ... from conatus.model, and the utilities below);
# ``conatus.expert`` is the expert layer (the IR, flattened). A name in both means the same thing
# in both (the development test suite); ``conatus.api`` is the deprecated former name of the expert layer.
_NATIVE = ("Box", "Channel", "Disk", "Annulus", "Pipe", "Cylinder", "TaylorCouette", "Sphere", "Ball", "Shell",
           "periodic", "walls", "line", "half_line", "MappedBox", "between", "Model", "free_slip", "no_slip", "navier_slip", "view",
           "spectrum", "profile", "series")


# The user layer, frozen in v0.1 (snapshot the development test suite):
# the native layer, the utility entry points and modules below, and the four packages shared with
# conatus.expert whose call is the expert function (cn.discretize(pb), cn.linearize, cn.methods,
# cn.verify). ``cn.api`` (deprecated, FutureWarning) is reachable but not listed.
_LAZY = {"from_dedalus": "conatus.translate", "doctor": "conatus._doctor",
         "parse_equation": "conatus.ir.parse", "parse_expr": "conatus.ir.parse",
         "StopRun": "conatus.telemetry", "uncertainty": "conatus.verify.uncertainty",
         "time_average": "conatus.verify.statistics", "box": "conatus.contour"}
_MODULES = ("expert", "runs", "problems", "notebook", "translate", "resolution", "model", "io",
            "tolerances", "errors", "discretize", "linearize", "methods", "verify", "contour")
__all__ = sorted(["__version__", *_NATIVE, *_LAZY, *_MODULES])


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


def __getattr__(name: str):
    """Lazy ``conatus.expert`` (PEP 562): ``import conatus`` stays light (no solver, no compiled core
    loaded); ``conatus.expert`` is imported on first access and then cached as a normal submodule
    (``cn.api``, its deprecated former name, likewise, with a FutureWarning).
    ``runs`` (the run registry, ``cn.runs.find(...)``) and ``io`` (HDF5 analysis files and
    checkpoints, ``cn.io.read(...)``) are lazy the same way. Any other missing
    name is a plain AttributeError.

    v0.1 UX entry points, lazy the same way: ``cn.problems`` (ready problems), ``cn.notebook``,
    ``cn.resolution``,
    ``cn.from_dedalus(script)`` (conatus.translate), ``cn.doctor()`` (conatus._doctor),
    ``cn.parse_equation`` / ``cn.parse_expr`` (string equations, conatus.ir.parse),
    ``cn.uncertainty`` (numerical-uncertainty report, conatus.verify.uncertainty),
    ``cn.time_average`` (mean and statistical error of a correlated series, conatus.verify.statistics),
    ``cn.tolerances`` (which tolerances are required, the declared defaults: ``cn.tolerances.explain()``).
    ``cn.box(re=, im=)`` (a region of the complex plane: ``m.eigenvalues(..., region=)``) and
    ``cn.contour`` (eigenvalue counts by the argument principle, Beyn's method).
    Native layer: ``cn.Box``, ``cn.Channel``, ``cn.Disk``, ..., ``cn.Model`` (conatus.model);
    watching a run in the Studio: ``cn.view``, ``cn.spectrum``, ``cn.profile``, ``cn.series``
    (``sim.watch``), ``cn.StopRun`` (a stop sent from the Studio)."""
    import importlib
    if name in _MODULES or name == "api":
        return importlib.import_module(f"conatus.{name}")
    lazy = _LAZY
    if name in _NATIVE:                       # native layer
        return getattr(importlib.import_module("conatus.model"), name)
    if name in lazy:
        return getattr(importlib.import_module(lazy[name]), name)
    raise AttributeError(f"module 'conatus' has no attribute '{name}'")
