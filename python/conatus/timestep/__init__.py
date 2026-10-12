"""Time integration of IVPs: pseudospectral F, stacked groups, IMEX schemes."""
from . import polar as _polar  # noqa: F401,E402  (IVPSolver(polar_plan) -> PolarIVPSolver)
from . import sphere as _sphere  # noqa: F401,E402  (IVPSolver(sphere_plan) -> SphereIVPSolver)
from .control import CFL, EmbeddedError, TauWatch
from .integrator import (CN_SCHEMES, RK_SCHEMES, SBDF_SCHEMES, SCHEMES, STARTUPS, IVPSolver,
                         cn_coefficients, rk_tableau, sbdf_coefficients, scheme_description)
from .spectral import SpectralError, SpectralEvaluator
from .system import ORDERINGS, GlobalSystem

__all__ = ["CFL", "EmbeddedError", "TauWatch", "IVPSolver", "SCHEMES", "SBDF_SCHEMES", "CN_SCHEMES",
           "RK_SCHEMES", "STARTUPS", "SpectralEvaluator", "SpectralError", "GlobalSystem", "ORDERINGS",
           "rk_tableau", "sbdf_coefficients", "cn_coefficients", "scheme_description"]


# memory planner and profiler as IVPSolver methods. They live in conatus.runtime (memory.py,
# profile.py) and are attached here so that integrator.py stays untouched while another work
# package edits it (decision); the minimal in-class hook is two delegating methods.
def _memory_report(self, steps: int | None = None, cufft_query: bool | None = None):
    """Declared device memory of this solver (liveness planner): ``conatus.runtime.memory``."""
    from ..runtime.memory import memory_report
    return memory_report(self, steps=steps, cufft_query=cufft_query)


def _profile(self, steps: int = 10, *, warmup: int = 2, peaks=None, restore: bool = True):
    """Per-phase timing with GPU synchronisation and a simple roofline: ``conatus.runtime.profile``."""
    from ..runtime.profile import profile
    return profile(self, steps, warmup=warmup, peaks=peaks, restore=restore)


IVPSolver.memory_report = _memory_report
IVPSolver.profile = _profile


def _public_signature() -> None:
    """``inspect.signature(IVPSolver)`` showed ``(plan=None, *args, **kwargs)`` (the dispatching
    ``__new__`` of the polar/spherical solvers) instead of the real arguments (BACKLOG, UX of the
    API reference): declare the signature of ``__init__`` (without self)."""
    import inspect
    sig = inspect.signature(IVPSolver.__init__)
    IVPSolver.__signature__ = sig.replace(parameters=list(sig.parameters.values())[1:])


_public_signature()
