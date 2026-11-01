"""Former names of the v0.1 API, removed in 0.3.

A renamed argument kept its former name until 0.3 with a ``DeprecationWarning``. Since 0.3 the
former name is refused with an :class:`~conatus.errors.OptionError` that names the new one. Every
former name stays listed in :data:`RENAMES`, so that the refusal can say what to write
(the development test suite checks each one).
"""
from __future__ import annotations

from typing import Any

#: The release that narrowed the public surface (``conatus.expert`` lists documented names only,
#: some modules hide former names) and the release that removes the former names. Every warning
#: about those names reads its versions here.
SURFACE_NARROWED_IN = "0.1.1"
SURFACE_REMOVED_IN = "0.3"
ARGUMENTS_REMOVED_IN = "0.3"

__all__ = ["DEPRECATED", "RENAMES", "RENAME_NOTES", "VOCABULARY", "renamed", "renames",
           "aliases_of", "given"]

#: Deprecations that are not argument renames (those are in :data:`RENAMES`): where -> (former
#: name, new name, category); ``conatus.api`` predates v0.1 (FutureWarning, shown to end users by
#: default) and is kept for 0.1.
DEPRECATED: dict[str, tuple[str, str, str]] = {
    "conatus.api": ("conatus.api", "conatus.expert", "FutureWarning"),
    "NCCResolutionWarning": ("NCCResolutionWarning", "DiscretizationError (refusal)",
                             "no longer emitted"),
}


#: The one table of renamed arguments: qualified site (module:
#: qualname) -> {former name: new name}. Applied by :func:`renames`; listed by the frozen
#: surface (``[renamed: ...]``);
#: the development test suite checks that every entry works, warns, refuses both names
#: together and is documented in CHANGELOG.md. The old names are refused since 0.3.
RENAMES: dict[str, dict[str, str]] = {
    # v0.1 freeze: one budget name, the step, the continuation parameter
    "conatus.solvers.parametric:ParametricEVP.track": {"maxit": "max_iterations"},
    "conatus.adjoint.nlop:NLOP.iterate": {"maxit": "max_iterations"},
    # a count is n_modes, a budget max_iterations
    "conatus.adjoint.nlop:NLOP.linear_gain": {"k": "n_modes", "maxiter": "max_iterations"},
    "conatus.ecs.ks:ks_solver": {"dt_": "timestep"},
    "conatus.ecs.kolmogorov:kolmogorov_solver": {"dt_": "timestep"},
    "conatus.continuation.nlbvp:NLBVPContinuation.__init__": {"param": "parameter"},
    # k is a wavenumber everywhere, a count of eigenvalues is n_modes
    "conatus.ecs.solutions:equilibrium_eigenvalues": {"k": "n_modes"},
    "conatus.ecs.solutions:floquet_multipliers": {"k": "n_modes"},
    "conatus.datadriven.spod:spod": {"nmodes": "n_modes"},
    "conatus.growth.periodic:operator_floquet": {"nev": "n_modes", "nsteps": "steps"},
    # the rename of growth/periodic.py deferred by
    "conatus.growth.periodic:operator_gain": {"nsteps": "steps", "k": "n_modes"},
    "conatus.growth.periodic:optimal_gain": {"nsteps": "steps"},
    # the counts still spelled k (k is a wavenumber everywhere else)
    "conatus.solvers.evp:solve_sparse": {"k": "n_modes"},
    # the count of shift-invert modes reached from m.eigenvalues
    "conatus.solvers.plan_evp:eigenvalues": {"k": "n_modes"},
    # accuracy: tol (relative residual of a solve); xtol (precision of a located value)
    "conatus.solvers.nlbvp:NewtonSolver.__init__": {"tolerance": "tol"},
    "conatus.solvers.periodic:PeriodicNewtonSolver.__init__": {"tolerance": "tol"},
    "conatus.nonmodal.energy:max_growth": {"xatol": "xtol"},
    "conatus.growth.optimize:optimal_growth": {"xatol": "xtol"},
    # iteration budgets: max_iterations
    "conatus.bifurcation.equilibria:newton_equilibrium": {"max_iter": "max_iterations"},
    "conatus.chaos.poincare:poincare_section": {"max_iter": "max_iterations", "X0": "x0"},
    # a starting state is x0; a scalar start is start (seed is only a random seed)
    "conatus.ecs.solutions:find_equilibrium": {"X0": "x0"},
    "conatus.ecs.solutions:find_travelling_wave": {"X0": "x0"},
    "conatus.ecs.solutions:find_relative_periodic_orbit": {"X0": "x0", "nsteps": "steps"},
    "conatus.chaos.lyapunov:lyapunov_spectrum": {"X0": "x0"},
    "conatus.adjoint.flowmap:FlowMap.jvp": {"X0": "x0"},
    "conatus.adjoint.flowmap:FlowMap.trajectory": {"X0": "x0"},
    "conatus.adjoint.flowmap:FlowMap.vjp": {"X0": "x0"},
    "conatus.adjoint.flowmap:FlowMap.vjp_physical": {"X0": "x0"},
    "conatus.timestep.integrator:IVPSolver.rk_stages": {"X0": "x0"},
    "conatus.ensembles.ensemble:Ensemble.rk_stages": {"X0": "x0"},
    "conatus.continuation.ks:KSDomainContinuation.run": {"X0": "x0"},
    "conatus.floquet.hill_general:secant_root": {"seed": "start", "maxit": "max_iterations"},
    # end time of an integration from t = 0: until
    "conatus.manifolds.trajectory:ODEIntegrator.integrate": {"t_end": "until"},
    "conatus.manifolds.trajectory:SolverIntegrator.integrate": {"t_end": "until"},
    "conatus.manifolds.parametrization:globalize": {"t_end": "until"},
    # one word per notion. A wavenumber keyword is k (a problem
    # kx/kz
    # pairs name directions); a radius ratio is eta; a number of time steps is steps.
    "conatus.problems.native:pipe_evp": {"alpha": "k"},
    "conatus.problems.cylinders:pipe_evp": {"alpha": "k"},
    "conatus.problems.native:rotating_shell_convection": {"radius_ratio": "eta"},
    "conatus.problems.native:rotating_shell_convection_evp": {"radius_ratio": "eta"},
    "conatus.problems.convection:rotating_shell_convection": {"radius_ratio": "eta"},
    "conatus.problems.convection:rotating_shell_convection_evp": {"radius_ratio": "eta"},
    "conatus.timestep.integrator:IVPSolver.run": {"nsteps": "steps"},
    "conatus.timestep.integrator:IVPSolver.run_graph": {"nsteps": "steps"},
    "conatus.timestep.polar:PolarIVPSolver.run": {"nsteps": "steps"},
    "conatus.timestep.polar:PolarIVPSolver.run_graph": {"nsteps": "steps"},
    "conatus.ensembles.ensemble:Ensemble.run": {"nsteps": "steps"},
    "conatus.distributed.imex:DistributedIVPSolver.run_graph": {"nsteps": "steps"},
    "conatus.distributed.imex:DistributedIVPSolver.run_checkpointed": {"nsteps": "steps"},
    "conatus.model.run:Linearization.optimal_gain": {"nsteps": "steps"},
    "conatus.model.run:Linearization.floquet": {"nsteps": "steps"},
    "conatus.adjoint.flowmap:FlowMap.__init__": {"nsteps": "steps"},
    "conatus.adjoint.flowmap:revolve_schedule": {"nsteps": "steps"},
    "conatus.adjoint.flowmap:revolve_actions": {"nsteps": "steps"},
    "conatus.adjoint.flowmap:revolve_cost": {"nsteps": "steps"},
    "conatus.growth.propagator:propagator": {"nsteps": "steps"},
    "conatus.nonmodal.transient:propagator_growth": {"nsteps": "steps"},
    "conatus.growth.optimize:growth_at": {"nsteps": "steps"},
    "conatus.ecs.newton:arnoldi_gmres": {"maxiter": "max_iterations"},
    "conatus.ecs.newton:NewtonKrylovHookstep.__init__": {"gmres_maxiter": "gmres_max_iterations"},
    # the wall-clock periods of a watch carry their unit (every= counts steps everywhere else)
    "conatus.model.run:Simulation.watch": {"every": "every_s", "scalars_every": "scalars_every_s",
                                           "controls_every": "controls_every_s"},
    "conatus.model.watch:Watch.__init__": {"every": "every_s", "scalars_every": "scalars_every_s",
                                           "controls_every": "controls_every_s"},
    "conatus.floquet.shell:shell_floquet": {"nev": "n_modes", "maxiter": "max_iterations",
                                            "nsteps": "steps"},
}

#: What a former name's warning adds to "same meaning": the unit the new name carries.
RENAME_NOTES: dict[str, str] = {
    "every_s": "wall-clock seconds, not steps",
    "scalars_every_s": "wall-clock seconds, not steps",
    "controls_every_s": "wall-clock seconds, not steps",
}

#: Names that the surface may carry only as declared aliases or with a stated reason (the
#: vocabulary rule of the renaming, enforced by the development test suite): former name ->
#: the name to use instead.
VOCABULARY: dict[str, str] = {
    "maxit": "max_iterations", "maxiter": "max_iterations", "max_iter": "max_iterations",
    "gmres_maxiter": "gmres_max_iterations", "nev": "n_modes", "nmodes": "n_modes",
    "X0": "x0", "xatol": "xtol", "t_end": "until", "tolerance": "tol",

    "nsteps": "steps", "max_newton": "max_iterations", "radius_ratio": "eta",
    "alpha": "k", "kx": "k", "R": "Re",
}

#: Surface entries that keep a former name of :data:`VOCABULARY`, with the reason (the rule is
#: that nothing else may): the development test suite enforces both.
VOCABULARY_KEPT: dict[str, str] = {
    "expert.ContinuationSettings(tolerance)": "a dataclass field that scripts read back "
    "(cfg.tolerance); its rename is still open",
    "expert.KrylovContinuationSettings(tolerance)": "as ContinuationSettings",
    "expert.ContinuationSettings(max_newton)": "a dataclass field read back (cfg.max_newton); "
    "the property cfg.max_iterations reads it under the API-wide name",
    # problem PARAMETERS keep the name of their literature (they are the names of the
    # problem's Parameter objects, of m.params and of critical(over=...)); keywords that are not
    # parameters use k.
    "cn.problems.channel_evp(alpha)": "the Orr-Sommerfeld parameter alpha of the literature "
    "(Orszag 1971), a Parameter of the problem; the wavenumber of the native layer is k",
    "cn.problems.plane_couette_evp(alpha)": "as channel_evp",
    "cn.problems.native.orr_sommerfeld(alpha)": "as channel_evp: alpha is a model parameter "
    "(m.params['alpha'], critical(over={'alpha': ...}))",
    # kx and kz name the two wavenumber DIRECTIONS of a 3D problem (not one wavenumber k)
    "expert.GrowthPoint(kx)": "a result field read back (pt.kx), streamwise of the (kx, kz) pair "
    "of growth_at",
    "expert.growth_at(kx)": "streamwise wavenumber, paired with kz",
    "expert.floquet_exponents(kx)": "paired with kz (the build(p) callables read p['kx'], p['kz'])",
    "expert.optimal_gain(kx)": "as floquet_exponents",
    "expert.periodic_operator(kx)": "as floquet_exponents",
    "expert.parallel_shear_flow(kx)": "paired with kz",
    # other meanings of the same letters
    "expert.SPOD.confidence(alpha)": "a significance level, not a wavenumber",
    "expert.SymmetrySubspace.parity(R)": "a reflection generator R, not a Reynolds number",
    # growth/periodic.py: the arguments are steps= since (nsteps= deprecated)
    "expert.OptimalGain(nsteps)": "a result field read back (g.nsteps), in growth/periodic.py",
}


def aliases_of(func: Any) -> dict[str, str]:
    """{former name: new name} of a callable (or of a class through its ``__init__``)."""
    f = func.__init__ if isinstance(func, type) else func
    return dict(getattr(f, "__conatus_renamed__", {}))


def _refuse(where: str, old: str, new: str) -> None:
    from .errors import OptionError
    note = RENAME_NOTES.get(new)
    raise OptionError(f"{where}: {old}= was removed in {ARGUMENTS_REMOVED_IN}, write {new}=",
                      why=f"{new} is the name with the same meaning" + (f": {note}" if note else ""),
                      fix=f"replace {old}= by {new}=")


def renames(site: str):
    """Decorator for the function or method ``site`` of :data:`RENAMES`: each former name is refused
    with the new one, and listed by :func:`aliases_of`; the signature shows the new names only."""
    import functools
    table = RENAMES[site]

    def deco(func):
        where = site.split(":")[1].replace(".__init__", "")

        @functools.wraps(func)
        def wrapper(*args: Any, **kw: Any):
            for old, new in table.items():
                if old in kw:
                    _refuse(where, old, new)
            return func(*args, **kw)
        wrapper.__conatus_renamed__ = dict(table)
        return wrapper
    return deco


def renamed(where: str, old: str, new: str, old_value: Any, new_value: Any, *,
            required: bool = True, note: str | None = None) -> Any:
    """The value of the argument ``new``. ``old_value`` given: refused with the new name. Neither
    given and ``required``: refused with what to write."""
    if old_value is not None:
        _refuse(where, old, new)
    return given(where, new, new_value) if required else new_value


def given(where: str, name: str, value: Any) -> Any:
    """``value`` of an argument that must be given although its signature default is ``None``
    (the default stood for "given under the former name", which :func:`renames` now maps before
    the call): refused when ``None``, with what to write."""
    if value is None:
        from .errors import OptionError
        raise OptionError(f"{where}: {name} must be given explicitly (no default)",
                          fix=f"{where}(..., {name}=...)")
    return value


def former_names(module: str, names: dict[str, tuple[str, Any]]):
    """A module ``__getattr__`` (PEP 562) for names a module showed in 0.1 and no longer does
    (release 0.1.1): ``names`` maps the former name to (its canonical path, the object). The
    name resolves to the same object with a DeprecationWarning at the user's line; anything else
    raises AttributeError as usual. Example: ``renames`` of the problem modules, imported there as
    ``_renames`` since 0.1.1 (V5)."""
    def __getattr__(name: str) -> Any:
        if name in names:
            from .errors import _warn_user
            path, obj = names[name]
            _warn_user(f"{module}.{name} is not a public name of this module since "
                       f"{SURFACE_NARROWED_IN}: use {path}, the same object", DeprecationWarning)
            return obj
        raise AttributeError(f"module {module!r} has no attribute {name!r}")
    return __getattr__
