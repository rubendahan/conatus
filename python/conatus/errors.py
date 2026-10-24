"""Public exception hierarchy of Conatus.

Every error Conatus raises on purpose derives from :class:`ConatusError`. A message says

* **what** is wrong (first line, unchanged from the historical message where one existed),
* **where** (equation, term, variable, group),
* **why** it is refused,
* **fix**: a concrete suggestion. Conatus never applies it for you (no silent substitution);
  the user edits the problem and runs again,
* **docs**: the page of the user guide and its section (``guide/errors#conatuserror``), with the
  command that locates it (``python -m conatus docs guide/errors``): a name valid in an installed
  copy, never a path of the source repository. ``err.doc`` keeps the source
  path (``docs/site/guide/errors.md#...``) for tools.

Compatibility: the classes that existed before v0.1 (``ProblemError``, ``NonlinearLHSError``,
``DiscretizationError``, ``RankError``, ``SpectralError``, ``SymmetryError``,
``SymmetryBreakingError``, ``NewtonNotConverged``) are defined here and re-exported from their
historical modules, and every class also derives from the built-in exception its raise sites used
before (``ValueError``, ``TypeError``, ``KeyError``, ``RuntimeError``, ``NotImplementedError``,
``ImportError``), so ``except ValueError`` and ``pytest.raises(ProblemError)`` keep working.

Tree (``explain()`` prints it)::

    ConatusError
    ├── EquationError (= ProblemError)            ValueError
    │   ├── NonlinearLHSError
    │   ├── UndeclaredVariableError                + KeyError
    │   ├── ParameterError                         + KeyError
    │   ├── BoundaryConditionError
    │   └── EquationSyntaxError                    (string equations, conatus.ir.parse)
    ├── DiscretizationError                       ValueError
    │   └── RankError                              + BoundaryConditionError
    ├── SpectralError                             ValueError
    ├── ResolutionError                           ValueError
    ├── CoordinateError                           ValueError
    ├── OptionError                               ValueError
    ├── SymmetryError                             ValueError
    │   └── SymmetryBreakingError
    ├── InputTypeError                            TypeError
    │   ├── DtypeError
    │   └── TensorError
    ├── ConvergenceError                          RuntimeError
    │   └── NewtonNotConverged
    ├── NumericalError                            ArithmeticError
    ├── BackendError                              RuntimeError
    │   └── MissingDependencyError                 + ImportError
    ├── UnsupportedError                          NotImplementedError
    └── AmbiguousNameError                        AttributeError
"""
from __future__ import annotations

import difflib
from collections.abc import Iterable
from typing import Any

# Documented names only (M5, 0.2); the others stay importable from this module. Every
# exception class is documented by the error catalogue of the guide (docs/site/guide/errors.md).
__all__ = ["ConatusError", "EquationError", "ProblemError", "NonlinearLHSError",
           "UndeclaredVariableError", "ParameterError", "BoundaryConditionError",
           "EquationSyntaxError", "DiscretizationError", "RankError", "SpectralError",
           "ResolutionError", "CoordinateError", "OptionError", "SymmetryError",
           "SymmetryBreakingError", "InputTypeError", "DtypeError", "TensorError",
           "ConvergenceError", "NewtonNotConverged", "NumericalError", "UnderResolvedWarning",
           "IllPosedWarning", "InitialDataWarning", "BackendError", "MissingDependencyError",
           "MissingFileError", "UnsupportedError", "AmbiguousNameError", "ResolutionWarning",
           "CorrectnessWarning", "BaseStateWarning", "GroupWarning", "DefaultChoiceWarning",
           "ExperimentalWarning", "explain"]

# Relative documentation root. The site (Sphinx, docs/site/) is written separately; every class
# points to one anchor of the error guide so the links can be remapped in one place.
DOC_ROOT = "docs/site/guide/errors.md"


def _doc(anchor: str) -> str:
    return f"{DOC_ROOT}#{anchor}"


def doc_reference(doc: str | None) -> str:
    """What the ``docs:`` line of a message prints for a source path ``docs/site/<page>.md#a``:
    ``<page>#a (python -m conatus docs <page>)``, a guide page name valid in an installed copy.
    Any other text is printed as given."""
    if not doc:
        return ""
    path, _, anchor = str(doc).partition("#")
    if not (path.startswith("docs/site/") and path.endswith(".md")):
        return str(doc)
    page = path[len("docs/site/"):-len(".md")]
    return f"{page}{'#' + anchor if anchor else ''}  (python -m conatus docs {page})"


class ConatusError(Exception):
    """Base class. ``ConatusError(message, where=..., why=..., fix=..., doc=...)``.

    Only ``message`` is positional, so the historical one-argument raises keep their text as the
    first line; the structured fields are appended below it and kept as attributes.
    """

    doc = _doc("conatuserror")

    def __init__(self, message: str = "", *, where: str | None = None, why: str | None = None,
                 fix: str | None = None, doc: str | None = None) -> None:
        super().__init__(message)
        self.message, self.where, self.why, self.fix = str(message), where, why, fix
        if doc is not None:
            self.doc = doc

    def __str__(self) -> str:
        lines = [self.message]
        for label, text in (("where", self.where), ("why", self.why), ("fix", self.fix)):
            if text:
                lines.append(f"  {label + ':':6s} " + str(text).replace("\n", "\n         "))
        lines.append(f"  docs:  {doc_reference(self.doc)}")
        return "\n".join(lines)

    def explain(self) -> str:
        """Same text as ``str()``, with the class name and its place in the hierarchy."""
        chain = " < ".join(c.__name__ for c in type(self).__mro__
                           if c not in (object, BaseException, Exception))
        return f"{type(self).__name__} ({chain})\n{self}"


# -- problem declaration ---------------------------------------------------------------------
class EquationError(ConatusError, ValueError):
    """An equation (or the whole problem) is inconsistent with the problem type or variables."""

    doc = _doc("equationerror")


ProblemError = EquationError          # historical name (conatus.ir.problem), same class


class NonlinearLHSError(EquationError):
    """The LHS contains a term that is not linear in the problem variables."""

    doc = _doc("nonlinearlhserror")


class UndeclaredVariableError(EquationError, KeyError):
    """A field is used as an unknown (or needs data) but is neither a variable nor given data."""

    doc = _doc("undeclaredvariableerror")


class ParameterError(EquationError, KeyError):
    """A parameter has no value, or a parameter name is unknown."""

    doc = _doc("parametererror")


class BoundaryConditionError(EquationError):
    """Boundary conditions are missing, in excess, or placed where none is possible."""

    doc = _doc("boundaryconditionerror")


class EquationSyntaxError(EquationError):
    """A string equation does not follow the declared grammar (``conatus.ir.parse.GRAMMAR``) or
    names something that is not in its declared namespace. ``text`` and ``position`` (0-based
    character offset) locate the problem; the message shows the text with a caret."""

    doc = _doc("equationsyntaxerror")

    def __init__(self, message: str = "", *, text: str | None = None, position: int | None = None,
                 **kw) -> None:
        if text is not None and position is not None:
            # the text and a caret under the offending character, first in "where"
            caret = f"{text}\n{' ' * max(0, position)}^ (column {position + 1})"
            kw["where"] = caret + (f"\n{kw['where']}" if kw.get("where") else "")
        super().__init__(message, **kw)
        self.text, self.position = text, position


# -- discretization --------------------------------------------------------------------------
class DiscretizationError(ConatusError, ValueError):
    """The expression cannot be discretized as requested (explicit reason given)."""

    doc = _doc("discretizationerror")


class RankError(DiscretizationError, BoundaryConditionError):
    """The discrete system is under-/over-determined or singular (refused before factorization)."""

    doc = _doc("rankerror")


class SpectralError(ConatusError, ValueError):
    """The expression cannot be evaluated by the pseudospectral RHS evaluator."""

    doc = _doc("spectralerror")


class ResolutionError(ConatusError, ValueError):
    """The resolution is too small for what is asked (boundary conditions, derivatives...)."""

    doc = _doc("resolutionerror")


class ResolutionWarning(UserWarning):
    """A returned result is under-resolved by a declared measure:
    the mode an eigenvalue result selects has its energy in the last coefficients, a Floquet-Hill
    mode in its outermost harmonics. Raised as a warning, never instead of the result: the same
    sentence is in the result and its ``explain()``. Not an error class (``warnings.warn``)."""


class CoordinateError(ConatusError, ValueError):
    """Inconsistent coordinates, bases or domain."""

    doc = _doc("coordinateerror")


class OptionError(ConatusError, ValueError):
    """An option has an invalid value (scheme, backend, method...); close names are suggested."""

    doc = _doc("optionerror")


class SymmetryError(ConatusError, ValueError):
    """Inconsistent symmetry declaration."""

    doc = _doc("symmetryerror")


class SymmetryBreakingError(SymmetryError):
    """A term does not have the parity required by the declared subspace."""

    doc = _doc("symmetrybreakingerror")


class MissingFileError(ConatusError, FileNotFoundError):
    """A file to read (an output, a checkpoint) does not exist, or is a directory. Derives from
    ``FileNotFoundError``, which these reads raised before (N3)."""

    doc = _doc("missingfileerror")


# -- types ------------------------------------------------------------------------------------
class InputTypeError(ConatusError, TypeError):
    """An argument has the wrong type (a Parameter, a Field, a number was expected...)."""

    doc = _doc("inputtypeerror")


class DtypeError(InputTypeError):
    """Real/complex (dtype) mismatch; a conversion is written by the user."""

    doc = _doc("dtypeerror")


class TensorError(InputTypeError):
    """Tensor signatures do not match (scalar vs vector, different coordinate systems)."""

    doc = _doc("tensorerror")


# -- solving ----------------------------------------------------------------------------------
class ConvergenceError(ConatusError, RuntimeError):
    """An iteration did not reach its declared tolerance within its declared budget."""

    doc = _doc("convergenceerror")


class NewtonNotConverged(ConvergenceError):
    """Newton did not reach the declared tolerance within the declared number of iterations."""

    doc = _doc("newtonnotconverged")


class NumericalError(ConatusError, ArithmeticError):
    """A computed state is not finite (NaN or inf): a time integration blew up.
    Raised by the finite-state check of the time steppers (``check_finite=``)."""

    doc = _doc("numericalerror")


# -- backends and scope -----------------------------------------------------------------------
class BackendError(ConatusError, RuntimeError):
    """The requested backend (GPU, compiled core) is not available here."""

    doc = _doc("backenderror")


class MissingDependencyError(BackendError, ImportError):
    """An optional package or the compiled core is missing."""

    doc = _doc("missingdependencyerror")


class UnsupportedError(ConatusError, NotImplementedError):
    """Declared in the IR but not implemented (yet) for this case."""

    doc = _doc("unsupportederror")


class AmbiguousNameError(ConatusError, AttributeError):
    """A name means two different things (two modules export it); use one of the explicit names.

    Derives from ``AttributeError`` because it is raised by a module attribute lookup
    (``conatus.expert.record``), so ``hasattr`` and introspection keep their protocol."""

    doc = _doc("ambiguousnameerror")


# -- warnings (detections: nothing is changed, the run goes on) -------------------------------
class UnderResolvedWarning(UserWarning):
    """A running simulation is under-resolved: the relative spectral tail of an unknown along a
    direction exceeds the declared ``RUN_TAIL_TOL`` (measured at the finite-state check; warned
    once per run and recorded by ``explain()``)."""


class InitialDataWarning(UserWarning):
    """Callable initial data are not fully resolved: their relative spectral tail beyond the
    resolution (sampled on twice the grid) lies between ``INITIAL_TAIL_TOL`` and the refusal level
    ``INITIAL_TAIL_REFUSE`` (a coarse ``N`` of a resolution study); warned once per
    ``initial(...)`` call and recorded by ``explain()``."""


class IllPosedWarning(UserWarning):
    """An implicit term makes the initial-value problem ill-posed: its leading-order growth rate
    increases without bound with the wavenumber (backward diffusion, a negative viscosity),
    detected when the model is compiled and named with the term."""


# Correctness warnings: a statement that the answer may
# not be the one the call meant. They derive from UserWarning and deliberately NOT from
# DeprecationWarning: Python's default filters hide a DeprecationWarning raised outside __main__
# (a module, a test, a job wrapper), and a class deriving from both would be hidden by the same
# `ignore::DeprecationWarning` filter. A pure API deprecation (a renamed argument, a former
# spelling with the same meaning) stays a DeprecationWarning.
class CorrectnessWarning(UserWarning):
    """The returned answer may not be the one the call meant: it is about another base state,
    another group of modes, or a choice the call did not declare. Shown under Python's default
    filters wherever it is raised (never a ``DeprecationWarning``, which Python hides outside
    ``__main__``); the result is still returned. Some warnings are also kept in ``explain()``: the
    base residual in ``lin.explain()``, the selected group in the result's header, and the growth
    rule inferred from the eigenvalue's name.

    Where the message says "refused from 0.3", the same call raises from 0.3."""


class BaseStateWarning(CorrectnessWarning):
    """A linearization about a base state that is not a steady solution: its steady residual
    exceeds ``BASE_RESIDUAL_TOL`` and ``base_residual='accept'`` was not given, so the modes are
    not the normal modes of a flow that exists (accepted in 0.1 and 0.2, refused from 0.3)."""


class GroupWarning(CorrectnessWarning):
    """An eigenvalue call without a group (no ``k=`` on a periodic model, no ``m=``/``kz=``/
    ``ell=`` on a curvilinear one) solved one group only (k = 0, or the first group): the modes
    of every other wavenumber are absent from the result (required from 0.3)."""


class DefaultChoiceWarning(CorrectnessWarning):
    """A choice that decides the answer was left out and the default or guess of 0.1 and 0.2 was
    used: the growth rule of an eigenvalue guessed from its name (``m.eigenvalue("c")`` ranks by
    Im c), the top condition of ``this case`` (no-slip). Declare it; required from 0.3."""


class ExperimentalWarning(UserWarning):
    """A feature of the experimental list was used (once per feature and process): it is shipped and
    tested, but not promised: its API and defaults may change in a minor release without a
    deprecation cycle, and some of its claims rest on fewer measurements. The list:
    ``conatus._experimental.EXPERIMENTAL`` and the guide page "Experimental features". Silence it
    with ``warnings.filterwarnings("ignore", category=cn.errors.ExperimentalWarning)``."""


# -- helpers ----------------------------------------------------------------------------------
#: Where the GPU install is described: the fix of every "no CuPy" / "no CUDA"
#: message points there instead of one recipe. `pip install ".[gpu]"` is right for a pip wheel and
#: wrong in the conda recipe, which takes CuPy from conda-forge (a second CuPy from pip breaks it);
#: both need the core built with CONATUS_CUDA=ON.
GPU_INSTALL_DOC = "docs/site/install.md#gpu-cuda-install"
GPU_INSTALL_FIX = (f"follow the GPU install ({GPU_INSTALL_DOC}): CuPy for your CUDA version "
                   "(conda: cupy-core from conda-forge; pip wheel: the gpu extra) and the core "
                   "built with CONATUS_CUDA=ON")


def require_cupy(what: str):
    """Import CuPy for ``what`` or raise MissingDependencyError saying how to get it."""
    try:
        import cupy
    except ImportError as err:
        raise MissingDependencyError(
            f"{what} needs CuPy, which is not installed in this environment ({err})",
            where=what,
            why="the GPU backend keeps the state in CuPy arrays on the device",
            fix=GPU_INSTALL_FIX + ", in an environment with a CUDA driver; or run on the CPU "
                "with backend='cpu'",
            doc=_doc("missingdependencyerror")) from None
    return cupy


def _scalar_of(value: Any) -> Any:
    """The Python scalar of a 0-d NumPy array (``np.array(0.5)``, ``x.sum()`` of a CuPy array
    after ``.get()``, an element read with ``a[()]``), else ``value`` unchanged. A 0-d array is
    one number, not a container: the checks below accept it as that number (review of the
    inputs audit: ``run(until=np.array(1.0))`` and ``safety=np.array(0.5)`` were
    accepted before the audit and refused after it)."""
    import numpy as np
    if isinstance(value, np.ndarray) and value.ndim == 0 and value.dtype.kind in "biuf":
        return value[()]
    return value


def check_count(value, name: str, where: str, minimum: int = 1) -> int:
    """An integer >= ``minimum``, the one count check of the package (a float or a bool is
    refused, never truncated: ``Nh=10.5`` used to pass a range check and fail later in
    ``range()``/``arange``)."""
    import numbers
    value = _scalar_of(value)
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise OptionError(f"{where}: {name}={value!r} must be an integer >= {minimum}",
                          why="sizes and iteration budgets must be integers",
                          fix=f"{name}={max(minimum, 8)}, for example")
    return int(value)


def check_flag(value, name: str, where: str) -> bool:
    """A switch is ``True`` or ``False`` (a NumPy bool too), the one flag check of the package
    (audit inputs: ``bool(flag)`` read ``complex="no"``, ``reduced="no"`` or
    ``logy="no"`` as True, the opposite of what was written)."""
    import numbers

    import numpy as np
    value = _scalar_of(value)
    if isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_)) \
            and value in (0, 1):
        return bool(value)        # 0 / 1: unambiguous (accepted before the audit; review)
    if not isinstance(value, (bool, np.bool_)):
        raise OptionError(f"{where}: {name}={value!r} is not True or False",
                          why="a switch is never read from the truth value of another object "
                              "(the string 'no' is true)",
                          fix=f"{name}=True or {name}=False")
    return bool(value)


def check_real(value, name: str, where: str, *, positive: bool = False,
               nonnegative: bool = False) -> float:
    """A finite real number (not a bool, a string or a complex number), optionally > 0 or >= 0
    (audit inputs: NaN, inf, ``"0.1"`` or ``1+0j`` reached arithmetic deep inside, or
    were accepted)."""
    import decimal
    import math
    import numbers
    value = _scalar_of(value)
    # a Decimal is a real number (not registered as numbers.Real): accepted like a Fraction
    ok = (isinstance(value, (numbers.Real, decimal.Decimal)) and not isinstance(value, bool)
          and type(value).__name__ != "bool_" and math.isfinite(float(value)))
    if ok and positive:
        ok = float(value) > 0
    if ok and nonnegative:
        ok = float(value) >= 0
    if not ok:
        need = "> 0" if positive else ">= 0" if nonnegative else ""
        raise OptionError(f"{where}: {name}={value!r} must be a finite real number {need}".rstrip(),
                          why="a numerical setting must be a finite real number",
                          fix=f"a finite float for {name}" + (f" {need}" if need else ""))
    return float(value)


def _warn_user(message: str, category: type[Warning]) -> None:
    """``warnings.warn`` attributed to the first caller outside the ``conatus`` package (the
    user's line), whatever the call depth (the renaming wrappers of ``conatus._deprecation`` add
    frames; a fixed ``stacklevel`` pointed into the library, where the default filter hides a
    DeprecationWarning), so that it is shown once per call site of the user's script. The one
    helper for this (earlier versions: ``timestep.integrator._user_level`` duplicated it)."""
    import os
    import sys
    import warnings
    pkg = os.path.dirname(os.path.abspath(__file__)) + os.sep
    level, f = 2, sys._getframe(1)        # stacklevel 2 = the caller of _warn_user
    while f is not None and os.path.abspath(f.f_code.co_filename).startswith(pkg):
        f = f.f_back
        level += 1
    warnings.warn(message, category, stacklevel=level)


def did_you_mean(name: str, choices: Iterable[str]) -> str:
    """``" Did you mean 'RK443'?"`` when a close match exists, else ``""`` (never auto-applied)."""
    choices = [str(c) for c in choices]
    # the same name in another case first ('ra' -> 'Ra', 't' -> 'T'),
    # then the closest spellings
    same = sorted(c for c in choices if c.lower() == str(name).lower() and c != str(name))
    close = same + [c for c in difflib.get_close_matches(str(name), choices, n=3, cutoff=0.5)
                    if c not in same]
    close = close[:3]
    if not close:
        return ""
    return " Did you mean " + " or ".join(repr(c) for c in close) + "?"


def _tree(cls: type, depth: int = 0) -> list[str]:
    extra = [b.__name__ for b in cls.__bases__[1:]]      # first base = the parent printed above
    if not depth:
        extra = [b.__name__ for b in cls.__bases__]
    line = "    " * depth + cls.__name__ + (f"  (+ {', '.join(extra)})" if extra else "")
    out = [line]
    for sub in sorted(cls.__subclasses__(), key=lambda c: c.__name__):
        if sub.__module__ == __name__:
            if depth and sub.__mro__.index(cls) != 1:
                continue            # multiple inheritance: print under its first base only
            out += _tree(sub, depth + 1)
    return out


def explain() -> str:
    """The hierarchy, with the built-in exception each class also derives from."""
    return "\n".join(["Conatus exceptions (conatus.errors):"] + _tree(ConatusError))


def catalogue() -> str:
    """Markdown catalogue (one anchor per class) for the documentation site's error guide."""
    lines = ["# Error guide", "",
             "Each error names what is wrong, where and why, and suggests a fix.",
             ""]
    seen: list[type] = []

    def walk(c: type) -> None:
        if c in seen:
            return
        seen.append(c)
        for s in sorted(c.__subclasses__(), key=lambda k: k.__name__):
            if s.__module__ == __name__:
                walk(s)
    walk(ConatusError)
    for c in seen:
        bases = ", ".join(b.__name__ for b in c.__bases__)
        lines += [f"## {c.__name__}", "", f"`{c.__name__}({bases})` — {c.__doc__.strip()}", ""]
    return "\n".join(lines)
