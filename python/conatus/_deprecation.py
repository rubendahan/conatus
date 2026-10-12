"""Declared deprecations of the v0.1 API.

A renamed argument keeps its former name for one cycle: the old name gives the same value (never
another behaviour), with a ``DeprecationWarning`` that names the new one; giving both is refused.
Every alias is listed in :data:`DEPRECATED` (and in CHANGELOG.md);
the development test suite checks each one. Removal: v0.2.
"""
from __future__ import annotations

import warnings
from typing import Any

__all__ = ["DEPRECATED", "RENAMES", "RENAME_NOTES", "VOCABULARY", "renamed", "renames",
           "positional_order", "aliases_of", "given"]

#: Deprecations that are not argument renames (those are in :data:`RENAMES`): where -> (former
#: name, new name, category); ``conatus.api`` predates v0.1 (FutureWarning, shown to end users by
#: default) and is kept for 0.1.
DEPRECATED: dict[str, tuple[str, str, str]] = {
    "conatus.api": ("conatus.api", "conatus.expert", "FutureWarning"),
    "NCCResolutionWarning": ("NCCResolutionWarning", "DiscretizationError (refusal)",
                             "no longer emitted"),
    # the settings became keyword-only when their algorithm controls got defaults;
    # positional construction (in the v0.1 field order) still works for one cycle
    "ContinuationSettings": ("positional arguments", "keyword arguments", "DeprecationWarning"),
    "KrylovContinuationSettings": ("positional arguments", "keyword arguments",
                                   "DeprecationWarning"),
    # one name for the radius ratio (taylor_couette(eta=), rotating_shell_*(eta=))
    "TaylorCouette.radius_ratio": ("TaylorCouette.radius_ratio", "TaylorCouette.eta",
                                   "DeprecationWarning"),
}


#: The one table of renamed arguments: qualified site (module:
#: qualname) -> {former name: new name}. Applied by :func:`renames`; listed by the frozen
#: surface (``[renamed: ...]``);
#: the development test suite checks that every entry works, warns, refuses both names
#: together and is documented in CHANGELOG.md. Removal of the old names: v0.2.
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
    "(cfg.tolerance); renamed with an attribute alias in v0.2",
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


def renames(site: str):
    """Decorator for the function or method ``site`` of :data:`RENAMES`: each former name still
    works (DeprecationWarning naming the new one, the same value), giving both is refused; the
    signature shows the new names only, the former ones are listed by :func:`aliases_of`."""
    import functools
    table = RENAMES[site]

    def deco(func):
        where = site.split(":")[1].replace(".__init__", "")

        @functools.wraps(func)
        def wrapper(*args: Any, **kw: Any):
            for old, new in table.items():
                if old not in kw:
                    continue
                if new in kw:
                    # refused whatever the values (renamed() treats None as "not
                    # given", so f(k=None, n_modes=8) used to pick one name silently)
                    from .errors import OptionError
                    raise OptionError(f"{where}: {old}= and {new}= given together",
                                      why=f"{old} is the deprecated name of {new} (same meaning)",
                                      fix=f"keep {new}= only")
                value = renamed(where, old, new, kw.pop(old), None, required=False,
                                note=RENAME_NOTES.get(new))
                if value is not None:
                    # an explicit old=None leaves the new argument's default in place (it
                    # meant "not given" for the former names), never None over a default
                    kw[new] = value
            return func(*args, **kw)
        wrapper.__conatus_renamed__ = dict(table)
        return wrapper
    return deco


def positional_order(cls: type, order: tuple[str, ...]) -> type:
    """Let the keyword-only dataclass ``cls`` still be built positionally in the v0.1 field
    ``order`` (DeprecationWarning naming the keywords); the signature shown stays keyword-only."""
    import functools
    import inspect
    init = cls.__init__

    @functools.wraps(init)
    def __init__(self, *args: Any, **kw: Any) -> None:
        if args:
            if len(args) > len(order):
                raise TypeError(f"{cls.__name__}() takes at most {len(order)} positional "
                                f"arguments ({len(args)} given)")
            names = order[:len(args)]
            both = [n for n in names if n in kw]
            if both:
                raise TypeError(f"{cls.__name__}() got multiple values for {both}")
            warnings.warn(f"{cls.__name__}: positional arguments are deprecated since v0.1 "
                          f", write keywords: {', '.join(n + '=' for n in names)}",
                          DeprecationWarning, stacklevel=2)
            kw.update(zip(names, args, strict=True))
        init(self, **kw)
    __init__.__signature__ = inspect.signature(init)
    cls.__init__ = __init__
    return cls


def renamed(where: str, old: str, new: str, old_value: Any, new_value: Any, *,
            required: bool = True, note: str | None = None) -> Any:
    """The value of the argument ``new`` given under either name (``None`` = not given).

    ``old_value`` given: DeprecationWarning, same value. Both given: refused. Neither and
    ``required``: refused with what to write."""
    from .errors import OptionError
    if old_value is not None:
        if new_value is not None:
            raise OptionError(f"{where}: {old}= and {new}= given together",
                              why=f"{old} is the deprecated name of {new} (same meaning)",
                              fix=f"keep {new}= only")
        meaning = f"same meaning: {note}" if note else "same meaning"
        # attributed to the user's line whatever the depth (a fixed
        # stacklevel=3 pointed into the library when a conatus function forwards **kw to a
        # renamed one, e.g. floquet_exponents(nsteps=...), where the default filter hides it)
        from .errors import _warn_user
        _warn_user(f"{where}: {old}= is deprecated since v0.1, write {new}= ({meaning}; the "
                   "old name is removed in v0.2)", DeprecationWarning)
        return old_value
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
