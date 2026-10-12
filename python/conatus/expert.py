"""The expert layer (the IR) in one import: ``from conatus.expert import ...``.

Two layers, one story: ``import conatus as cn`` is the user layer
(native models ``cn.Box``, ``cn.Model``, ... and the utilities ``cn.problems``, ``cn.runs``,
``cn.doctor``, ...); ``conatus.expert`` is the expert layer, the IR and every toolkit, flattened.
A name present in both means the same thing in both: the same object, or, for ``discretize``,
``linearize``, ``methods`` and ``verify``, the package ``cn.<name>`` whose call is the function
``conatus.expert.<name>`` (``conatus._callable``). ``conatus.api`` is the deprecated former name of
this module (a FutureWarning points here).

Re-exports the public objects (``__all__``) of every user-facing module under their own names:
the IR, base flows, solvers (EVP, NLBVP, parametric pencils), time stepping (IVPSolver), the
dynamics toolkit (adjoint FlowMap, invariant solutions, continuation, Floquet--Hill, transient
growth), units, study presets, the methods text generator, telemetry, run provenance, the error
classes, the GPU query and verification. Nothing is added or changed here.

Name clashes are never resolved silently. Three kinds, all listed by ``explain()``:

* two modules export *different* objects under the same name: the first one in ``_MODULES``
  order is kept here and the other stays reachable through its module
  (``conatus.expert.ecs.VectorField``), listed in ``SHADOWED``; except names in ``_WITHHELD``,
  whose meanings are too different to pick one (``record``: telemetry frame vs run provenance):
  the bare name is refused (``AmbiguousNameError``) and explicit aliases are exported
  (``telemetry_record``, ``provenance_record``);
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
import types

from .errors import AmbiguousNameError
from .verify import Check, VerificationReport, verify

# Order matters only for clashes (earlier wins); see SHADOWED.
# "discretize" and "reference": the examples import ``discretize`` from conatus.discretize;
# "reference" gives FieldData (explicit data layout) and grid. "presets", "methods", "telemetry",
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
_MODULES = ("ir", "flows", "units", "discretize", "reference", "solvers", "linearize", "timestep",
            "adjoint", "ecs", "continuation", "floquet", "growth", "nonmodal", "presets", "methods",
            "telemetry",
            "provenance", "errors", "gpu", "bifurcation", "slices", "manifolds",
            "sensitivity", "datadriven", "chaos", "ensembles")

# Modules reachable as conatus.expert.<module> but not flattened: their verbs (find, get, record,
# compare...) would read as unrelated functions in user scripts. ``runs`` = the run registry;
# ``diagnostics`` (post-processing on grids: its Chebyshev, Grid, Periodic, explain are grid
# descriptions that would shadow or be shadowed by the IR bases).
# ``tolerances`` (REQUIRED, require, the DECLARED defaults; its explain() would shadow).
_MODULE_ONLY = ("runs", "diagnostics", "tolerances")

# Names with two unrelated meanings: not re-exported bare; one explicit alias per meaning.
_WITHHELD = {"record": {"telemetry": "telemetry_record", "provenance": "provenance_record"}}

# Generic helper names left in their module only (conatus.expert.provenance.write, ...): flattened,
# they would read as unrelated verbs (``write``, ``environment``) in user scripts.
_KEEP_IN_MODULE = {"provenance": {"code_state", "numerics", "environment", "problem_hash", "write",
                                  "parameter_values"},
                   "errors": {"did_you_mean", "require_cupy", "catalogue"},
                   # the name belongs to the package conatus.sensitivity, not to this helper
                   "solvers": {"sensitivity"}}

# Defined in this module; a module export under one of these names never replaces it.
_OWN = {"Check": Check, "VerificationReport": VerificationReport, "verify": verify}

__all__ = ["Check", "VerificationReport", "verify", "explain", "SHADOWED", "MODULE_NAME_TAKEN"] \
    + list(_MODULES) + list(_MODULE_ONLY)
SHADOWED: dict[str, list[str]] = {}
MODULE_NAME_TAKEN: dict[str, str] = {}

_mods = {n: importlib.import_module(f"conatus.{n}") for n in _MODULES}
for _name, _mod in _mods.items():
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
                                         "alias in _WITHHELD", fix="add one in api._WITHHELD")
            globals()[_alias] = _o
            __all__.append(_alias)
            continue
        if _n == "explain" or (_n in globals() and globals()[_n] is not _o):
            SHADOWED.setdefault(_n, []).append(_name)
            continue
        if _n not in __all__:
            __all__.append(_n)
        globals()[_n] = _o
for _name in _MODULE_ONLY:
    globals()[_name] = importlib.import_module(f"conatus.{_name}")
# Modules are reachable as conatus.expert.<module> unless an object already uses that name
# (e.g. the function ``linearize``); they always remain importable as conatus.<module>.
for _name, _mod in _mods.items():
    if _name not in globals():
        globals()[_name] = _mod
    elif globals()[_name] is not _mod:
        _o = globals()[_name]
        MODULE_NAME_TAKEN[_name] = f"{getattr(_o, '__module__', '?')}.{getattr(_o, '__name__', _name)}"


def __getattr__(name: str):
    """Refuse a withheld ambiguous name with the explicit aliases (PEP 562 module hook)."""
    if name in _WITHHELD:
        aliases = _WITHHELD[name]
        raise AmbiguousNameError(
            f"conatus.expert.{name} is ambiguous",
            why="; ".join(f"conatus.{m}.{name} is not the same object" for m in aliases),
            fix=" or ".join(f"conatus.expert.{a} (= conatus.{m}.{name})" for m, a in aliases.items()))
    raise AttributeError(f"module 'conatus.expert' has no attribute '{name}'")


def explain() -> str:
    """What the single import contains, and every name that is reachable only through its module."""
    lines = [f"conatus.expert re-exports {len(__all__)} names from: {', '.join(_MODULES)}, verify."]
    for n, aliases in sorted(_WITHHELD.items()):
        lines.append(f"  ambiguous name: '{n}' is not exported; use "
                     + ", ".join(f"{a} (conatus.{m}.{n})" for m, a in aliases.items()) + ".")
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
    return "\n".join(lines)
