"""The expert layer (the IR) in one import: ``from conatus.expert import ...``.

Two layers, one story: ``import conatus as cn`` is the user layer
(native models ``cn.Box``, ``cn.Model``, ... and the utilities ``cn.problems``, ``cn.runs``,
``cn.doctor``, ...); ``conatus.expert`` is the expert layer, the IR and every toolkit, flattened.
A name present in both means the same thing in both: the same object, or, for ``discretize``,
``linearize``, ``methods`` and ``verify``, the package ``cn.<name>`` whose call is the function
``conatus.expert.<name>`` (``conatus._callable``). ``conatus.api`` is the deprecated former name of
this module (a FutureWarning points here).

Re-exports, under their own names, the *documented* public objects of the user-facing modules: the
IR, base flows, solvers (EVP, NLBVP, parametric pencils), time stepping (IVPSolver), the dynamics
toolkit (adjoint FlowMap, invariant solutions, continuation, Floquet--Hill, transient growth),
units, study presets, the methods text generator, run provenance, the error classes, the GPU query
and verification. Nothing is added or changed here.

What is exported is a reviewed list (``_PUBLIC``, decision M5 of release 0.2): a name of a
module's ``__all__`` is flattened here only if the documentation shows it
(the development test suite); every exception class of ``conatus.errors`` is (the
error catalogue of the guide documents each). The other names stay importable from their module
(``conatus.contour.box``); ``conatus.expert.<name>`` still resolves them for one release
with a DeprecationWarning that names the module (removed in 0.3). The Studio wire protocol and the
telemetry recorder (``conatus.telemetry``) and the BibTeX strings (``conatus.methods.BIBTEX``) are not
part of the expert layer.

Name clashes are never resolved silently. Three kinds, all listed by ``explain()``:

* two modules export *different* objects under the same name: the first one in ``_MODULES``
  order is kept here and the other stays reachable through its module
  (``conatus.expert.ecs.VectorField``), listed in ``SHADOWED``; except names in ``_WITHHELD``,
  whose meanings are too different to pick one (``record``: run provenance here, a telemetry
  frame in ``conatus.telemetry``): the bare name is refused (``AmbiguousNameError``) and an
  explicit alias is exported (``provenance_record``);
* a module export collides with a name defined here (``explain``, ``verify``...): the one
  defined here wins, the other stays in its module (``SHADOWED`` too);
* an object takes the name of a module (the function ``linearize``): the module stays
  ``conatus.linearize`` (callable as the function, see above); listed in ``MODULE_NAME_TAKEN``.

``sensitivity`` is the ``conatus.sensitivity`` (``eigenvalue_sensitivity``,
``structural_sensitivity``, ...). The low-level ``d sigma / dp`` of a dense EVP result, formerly
flattened here under that same name, is ``conatus.solvers.sensitivity`` (still
``conatus.api.sensitivity`` in the deprecated alias).
"""
from __future__ import annotations

import importlib
import inspect
import types
import warnings

from ._deprecation import SURFACE_NARROWED_IN, SURFACE_REMOVED_IN
from .errors import AmbiguousNameError
from .verify import Check, VerificationReport, verify

# Order matters only for clashes (earlier wins); see SHADOWED.
# "discretize" and "reference": the examples import ``discretize`` from conatus.discretize;
# "reference" gives FieldData (explicit data layout) and grid. "presets", "methods",
# "provenance", "errors" and "gpu": public, documented, imported by users from their modules.
# "bifurcation" (native bifurcation core: branch switching, orbits, codim 2, normal forms) added
# with that package. "slices" and "manifolds" (phase-space geometry: symmetry reduction,
# Poincaré sections, invariant manifolds, connections, snaking). "sensitivity" (wavemaker,
# base-flow/forcing sensitivity, unified dσ/dp) and "datadriven" (POD, DMD, SPOD,
# EDMD), with names distinct from every earlier export. Modules that need more
# than NumPy (the compiled core internals in conatus.runtime, for one) stay out: import them
# explicitly.
# "chaos" (Lyapunov spectrum, Poincaré sections) and "ensembles" (batched members, fates): public
# and documented (tutorial 12).
# "contour" (eigenvalue counts by the argument principle, Beyn's method): its classes are
# flattened, its verbs stay in the module (conatus.expert.contour.count, .beyn, .box).
# "telemetry" left in 0.1.1 (Z7): the Studio wire protocol is not a solver toolkit; its former
# names resolve through _FORMER_MODULES with a DeprecationWarning.
_MODULES = ("ir", "flows", "units", "discretize", "reference", "solvers", "linearize", "timestep",
            "adjoint", "ecs", "continuation", "floquet", "growth", "nonmodal", "presets", "methods",
            "provenance", "errors", "gpu", "bifurcation", "slices", "manifolds",
            "sensitivity", "datadriven", "chaos", "ensembles", "contour")
# Flattened in 0.1.0, no longer in 0.1.1: their names still resolve, with a DeprecationWarning (0.3: gone).
_FORMER_MODULES = ("telemetry",)
# Names 0.1 exported here from a module of _MODULES whose __all__ lists them no longer (L8).
_FORMER_NAMES = {"errors": ("catalogue", "did_you_mean", "doc_reference", "require_cupy")}

# The exported names of each module (M5): the ones the documentation shows, reviewed one by
# one (tests/docs/test_surface_documented.py checks each is documented; the development test suite that
# each is exported by its module). Exception classes of conatus.errors are added by rule (the error
# catalogue documents each). A new public name of a module is not flattened until it is listed here.
_PUBLIC: dict[str, tuple[str, ...]] = {
    "ir": ("AnnulusBasis", "AzimuthalMFold", "BallBasis", "CartesianCoordinates", "Chebyshev",
           "ChebyshevT", "ChebyshevU", "ChebyshevV", "ComplexFourier", "CompoundChebyshev",
           "CylindricalCoordinates", "DiskBasis", "Domain", "EVEN", "EVP", "EquatorialParity", "Field",
           "Hermite", "IVP", "Jacobi", "LBVP", "Laguerre", "Legendre", "MappedChebyshev", "NLBVP", "ODD",
           "Parameter", "PolarCoordinates", "Problem", "RankReport", "RationalMap", "RealFourier",
           "Reflection", "S2Coordinates", "ShellBasis", "ShiftReflect", "SphereBasis",
           "SphericalCoordinates", "SymmetrySubspace", "TanMap", "TanhMap", "TensorField",
           "Ultraspherical", "VectorField", "absolute", "comp", "conj", "cos", "cross", "curl", "diff", "div", "dot", "dt",
           "exp", "grad", "integ", "interp", "lap", "lift", "log", "outer", "parse_equation",
           "parse_expr", "sign", "sin", "sqrt", "tanh", "unit_vector"),
    "flows": ("ChannelBaseFlow", "channel"),
    "units": ("ChannelUnits", "Energy", "UnitSystem", "convert"),
    "discretize": ("LBVPSolution", "Plan", "discretize"),
    "reference": ("FieldData", "grid"),
    "solvers": ("CriticalPoint", "EVPResult", "NLBVPSolution", "NewtonSolver", "ParametricEVP",
                "PeriodicNewtonSolver", "critical", "eigenvalues", "solve_dense", "solve_sparse"),
    "linearize": ("InnerProduct", "linearize"),
    "timestep": ("CFL", "EmbeddedError", "IVPSolver", "SCHEMES", "TauWatch"),
    "adjoint": ("FlowMap", "VJPSuspended", "revolve_cost"),
    "ecs": ("EdgeTracker", "Equilibrium", "FourierOps", "NewtonKrylovHookstep", "ThresholdBisection",
            "equilibrium_eigenvalues", "find_equilibrium", "find_relative_periodic_orbit",
            "find_travelling_wave", "floquet_multipliers", "symmetry_breaking_check"),
    "continuation": ("ContinuationSettings", "KSDomainContinuation", "NLBVPContinuation"),
    "floquet": ("CertifiedRoot", "HillBlocks", "HillGPU", "SecantRoot", "exponents_dense", "hill_from_plan", "secant_root", "shell_floquet"),
    "growth": ("energy_factor", "floquet_exponents", "gain", "optimal_gain", "optimal_growth",
               "periodic_operator", "propagator", ),
    "nonmodal": ("Y", "compare_adjoints", "h2_norm", "hinf_norm", "kreiss_constant", "max_growth",
                 "numerical_abscissa", "parallel_shear_flow", "pseudospectra", "resolvent_modes",
                 "state_space", "z_from_frequency"),
    "presets": ("CHANNEL_SYSTEMS", "laminar_poiseuille"),
    "methods": ("methods",),
    "provenance": ("record",),                      # exported as provenance_record (_WITHHELD)
    "errors": (),                                   # every exception class, by rule (_catalogued)
    "gpu": ("gpu_available", "gpu_report"),
    "bifurcation": ("CollocationMesh", "PeriodicOrbitProblem", "amplitude_equation", "branch_switch",
                    "continue_codim2", "continue_equilibria", "continue_orbits", "cos_mode_observable",
                    "ginzburg_landau", "ir_system", "normal_form", "ode_problem"),
    "slices": ("FirstFourierModeSlice", "find_rpo_in_slice", "slice_equilibrium_check"),
    "manifolds": ("PoincareSection", "ecs_system", "find_connection", "first_return_map", "globalize",
                  "scan_connection", "unstable_manifold"),
    "sensitivity": ("base_flow_sensitivity", "ginzburg_landau_exact", "linear_ginzburg_landau",
                    "structural_sensitivity"),
    "datadriven": ("DMD", "DataInnerProduct", "SPOD", "dmd", "edmd", "linear_process", "monomials", "pod",
                   "rbf", "resolvent_spod", "spod"),
    "chaos": ("kaplan_yorke", "lyapunov_spectrum", "poincare_section"),
    "ensembles": ("Ensemble", "ensemble_fate", "proportion_ci"),
    "contour": ("BeynResult", "CountResult", "RegionEVPResult"),
}

# Modules reachable as conatus.expert.<module> but not flattened: their verbs (find, get, record,
# compare...) would read as unrelated functions in user scripts. ``runs`` = the run registry;
# ``diagnostics`` (post-processing on grids: its Chebyshev, Grid, Periodic, explain are grid
# descriptions that would shadow or be shadowed by the IR bases).
# ``tolerances`` (REQUIRED, require, the DECLARED defaults; its explain() would shadow).
_MODULE_ONLY = ("runs", "diagnostics", "tolerances")

# Names with two unrelated meanings: not re-exported bare; one explicit alias per meaning that the
# expert layer exports. ``record`` is also conatus.telemetry.record (a Studio frame), outside it.
_WITHHELD = {"record": {"provenance": "provenance_record"}}
_WITHHELD_ELSEWHERE = {"record": ("telemetry",)}

# Generic helper names left in their module only (conatus.expert.provenance.write, ...): flattened,
# they would read as unrelated verbs (``write``, ``environment``) in user scripts.
_KEEP_IN_MODULE = {"provenance": {"code_state", "numerics", "environment", "problem_hash", "write",
                                  "parameter_values"},
                   "errors": {"did_you_mean", "require_cupy", "catalogue"},
                   # the name belongs to the package conatus.sensitivity, not to this helper
                   "solvers": {"sensitivity"},
                   # generic verbs: conatus.expert.contour.count / .beyn / .box
                   "contour": {"count", "beyn", "box"}}

# Defined in this module; a module export under one of these names never replaces it.
_OWN = {"Check": Check, "VerificationReport": VerificationReport, "verify": verify}

__all__ = ["Check", "VerificationReport", "verify", "explain"] + list(_MODULE_ONLY)
SHADOWED: dict[str, list[str]] = {}
MODULE_NAME_TAKEN: dict[str, str] = {}
# name -> (module, attribute) of a name exported by conatus.expert in 0.1 and no longer (M5, Z7)
_UNLISTED: dict[str, tuple[str, str]] = {}


_errors = importlib.import_module("conatus.errors")
_CATALOGUE = {id(getattr(_errors, n)) for n in _errors.__all__
              if inspect.isclass(getattr(_errors, n)) and issubclass(getattr(_errors, n), BaseException)}


def _catalogued(obj) -> bool:
    """An exception class listed by conatus.errors: the guide's error catalogue documents each one."""
    return id(obj) in _CATALOGUE


_mods = {n: importlib.import_module(f"conatus.{n}") for n in _MODULES}
for _name, _mod in _mods.items():
    _public = set(_PUBLIC[_name])
    for _n in getattr(_mod, "__all__", [n for n in dir(_mod) if not n.startswith("_")]):
        _o = getattr(_mod, _n)
        if isinstance(_o, types.ModuleType):       # objects only, not submodules
            continue
        if _n in _KEEP_IN_MODULE.get(_name, ()):
            continue
        if _n in _WITHHELD:
            _alias = _WITHHELD[_n].get(_name)
            if _alias is None:
                raise AmbiguousNameError(f"conatus.expert: '{_n}' exported by {_name}, which has no "
                                         "alias in _WITHHELD", fix="add one in expert._WITHHELD")
            globals()[_alias] = _o
            __all__.append(_alias)
            continue
        if _n == "explain" or (_n in globals() and globals()[_n] is not _o):
            SHADOWED.setdefault(_n, []).append(_name)
            continue
        if _n not in _public and not _catalogued(_o):
            _UNLISTED.setdefault(_n, (_name, _n))
            continue
        if _n not in __all__:
            __all__.append(_n)
        globals()[_n] = _o
for _name in _FORMER_MODULES:                      # 0.1 exports of modules no longer flattened
    _mod = importlib.import_module(f"conatus.{_name}")
    for _n in getattr(_mod, "__all__", []):
        if _n in _WITHHELD and _name in _WITHHELD_ELSEWHERE.get(_n, ()):
            _UNLISTED.setdefault(f"{_name}_{_n}", (_name, _n))       # telemetry_record
        elif _n not in globals() and not isinstance(getattr(_mod, _n), types.ModuleType):
            _UNLISTED.setdefault(_n, (_name, _n))
# Names 0.1 flattened from a module whose own __all__ the re-cut trimmed (they stay in the module):
# no longer in that __all__, so the loop above does not see them.
for _name, _names in _FORMER_NAMES.items():
    for _n in _names:
        _UNLISTED.setdefault(_n, (_name, _n))
for _n in list(_UNLISTED):
    if _n in globals():                            # exported under that name by an earlier module
        del _UNLISTED[_n]
for _name in _MODULE_ONLY:
    globals()[_name] = importlib.import_module(f"conatus.{_name}")
# Modules are reachable as conatus.expert.<module> unless an object already uses that name
# (e.g. the function ``linearize``); they always remain importable as conatus.<module>. They are not
# in __all__ (0.2): the canonical path of a module is conatus.<module>.
for _name, _mod in _mods.items():
    if _name not in globals():
        globals()[_name] = _mod
    elif globals()[_name] is not _mod:
        _o = globals()[_name]
        MODULE_NAME_TAKEN[_name] = f"{getattr(_o, '__module__', '?')}.{getattr(_o, '__name__', _name)}"


def __getattr__(name: str):
    """Refuse a withheld ambiguous name with the explicit aliases; resolve a name that 0.1 exported
    here and 0.2 does not, with a DeprecationWarning naming its module (PEP 562 module hook)."""
    if name in _WITHHELD:
        aliases = _WITHHELD[name]
        others = _WITHHELD_ELSEWHERE.get(name, ())
        raise AmbiguousNameError(
            f"conatus.expert.{name} is ambiguous",
            why="; ".join(f"conatus.{m}.{name} is not the same object" for m in (*aliases, *others)),
            fix=" or ".join([f"conatus.expert.{a} (= conatus.{m}.{name})" for m, a in aliases.items()]
                            + [f"conatus.{m}.{name}" for m in others]))
    if name in _FORMER_MODULES:                    # conatus.expert.telemetry: a 0.1 attribute
        warnings.warn(f"conatus.expert.{name} is not part of the expert layer since "
                      f"{SURFACE_NARROWED_IN}: use conatus.{name}, the same module; "
                      f"conatus.expert.{name} stops working in {SURFACE_REMOVED_IN}",
                      DeprecationWarning, stacklevel=2)
        return importlib.import_module(f"conatus.{name}")
    if name in _UNLISTED:
        mod, attr = _UNLISTED[name]
        warnings.warn(f"conatus.expert.{name} is not part of the expert layer since "
                      f"{SURFACE_NARROWED_IN} (only documented names are): use conatus.{mod}.{attr}, "
                      f"the same object; conatus.expert.{name} stops working in {SURFACE_REMOVED_IN}",
                      DeprecationWarning, stacklevel=2)
        return getattr(importlib.import_module(f"conatus.{mod}"), attr)
    raise AttributeError(f"module 'conatus.expert' has no attribute '{name}'")


def explain() -> str:
    """What the single import contains, and every name that is reachable only through its module."""
    lines = [f"conatus.expert re-exports {len(__all__)} documented names from: {', '.join(_MODULES)}, "
             "verify."]
    for n, aliases in sorted(_WITHHELD.items()):
        lines.append(f"  ambiguous name: '{n}' is not exported; use "
                     + ", ".join([f"{a} (conatus.{m}.{n})" for m, a in aliases.items()]
                                 + [f"conatus.{m}.{n}" for m in _WITHHELD_ELSEWHERE.get(n, ())]) + ".")
    for n, mods in sorted(SHADOWED.items()):
        lines.append(f"  name clash: '{n}' here is the first definition "
                     f"({'conatus.expert' if n in _OWN or n == 'explain' else 'earlier module'}); "
                     f"also in {', '.join(mods)} (use conatus.{mods[0]}.{n}).")
    for m, obj in sorted(MODULE_NAME_TAKEN.items()):
        lines.append(f"  module name taken: '{m}' in conatus.expert is {obj}; the module is conatus.{m}.")
    for m in _MODULE_ONLY:
        lines.append(f"  module only: conatus.expert.{m} (= conatus.{m}; its functions are not flattened).")
    for m, names in sorted(_KEEP_IN_MODULE.items()):
        lines.append(f"  kept in their module: {', '.join(sorted(names))} (use conatus.expert.{m}.<name>).")
    by_mod: dict[str, int] = {}
    for mod, _attr in _UNLISTED.values():
        by_mod[mod] = by_mod.get(mod, 0) + 1
    lines.append(f"  not flattened (undocumented, or not part of the expert layer): {len(_UNLISTED)} names, "
                 "each importable from its module ("
                 + ", ".join(f"conatus.{m}: {k}" for m, k in sorted(by_mod.items())) + ").")
    return "\n".join(lines)
