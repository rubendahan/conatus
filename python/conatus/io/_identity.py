"""The canonical identity of a computation (0.1.1, lot A "identity"; external review 13/10,
findings 1, 5, 8, 22, 49; first record).

**One** definition of what makes two computations the same, consumed by every place that needs
it: the registry cache key (``conatus.runs.keys``), the checkpoint restart and overwrite guards
(``IVPSolver.load_checkpoint``, ``cn.io.CheckpointSeries``), the provenance record
(``conatus.provenance.numerics``), the registry record and the HDF5 append (``cn.io.AnalysisFile``,
``mode="a"``). Before 0.1.1 each of them read its own subset and none read the data of the given
fields, so two forcings ``f = 1`` and ``f = 2`` shared a cache key and a checkpoint of one
restarted silently with the other (review finding 1).

:func:`computation_identity` (version :data:`IDENTITY_VERSION`) holds:

* ``model``: the name of the problem (the native model's name);
* ``equations``: a hash of the equations as compiled (kind, unknowns, left and right sides,
  symmetry subspace), independent of the parameter values; compared only when the resolution
  and the geometry agree (the tau lifts of the compiled equations carry the basis size and
  bounds, so a resolution change would also show as an equation change);
* ``parameters``: the value of every declared parameter;
* ``resolution``: the size of the basis along each coordinate;
* ``geometry``: the basis family and the bounds (or radius) along each coordinate;
* ``data``: every given field / bound coefficient by name, domain (hash of its bases), rank,
  shape, dtype and **content** (sha256 of the bytes the plan uses);
* ``functions``: every user pointwise function of the equations (``m.function``) by its code and
  the values it reads (closure cells, module globals, and the attributes it names of a user
  module or a user class, :func:`callable_identity`);
* ``numerics``: the numerical choices of a solver (:func:`stepping`: scheme, dt, start-up,
  time parameter, boundary-data convention, backends, dtype, ordering, reduced space, pivoting);
* ``undeclared``: the dependencies that cannot be hashed by content (a closure over an object, a
  data function of a mapped domain whose code reads one): a computation that has any is never
  given a cache key (refused loudly, never a silent hit).

A checkpoint records :func:`checkpoint_identity` (all of it but the numerics and ``undeclared``:
the numerics are compared by the solver itself, and a CPU checkpoint may restart on a GPU) in
``meta["identity"]`` (both formats), and a restart compares it with the solver it is loaded into.
Any difference is refused with the list of what differs, unless ``allow=`` names it: a parameter,
given-field or function name (``allow=("kappa",)``: continue this state at a new kappa, the
legitimate use) or one of the categories ``"parameters"``, ``"data"``, ``"functions"``,
``"model"``, ``"equations"``, ``"geometry"``. The resolution can never be allowed (the state would
be read in the wrong layout). An allowed difference is recorded: ``explain()`` prints it and every
later checkpoint carries it (``meta["restart_changes"]``). A checkpoint written before this
record (no ``identity``) is restarted with a warning, after comparing what it holds (the problem
name); one written by 0.1.0 (identity without ``data``) warns that its given fields were not
compared.
"""
from __future__ import annotations

import builtins
import hashlib
import inspect
import sys
import types
import warnings
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from ..errors import OptionError, did_you_mean

#: Version of the identity record (1: v0.1.0, without data, functions and numerics).
IDENTITY_VERSION = 2
#: The compared parts of a checkpoint's identity, in the order a mismatch lists them.
IDENTITY_KEYS = ("model", "resolution", "geometry", "equations", "parameters", "data", "functions")
#: Categories ``allow=`` accepts besides parameter, given-field and function names ("resolution"
#: is never allowed).
ALLOWABLE = ("parameters", "data", "functions", "model", "equations", "geometry")
_DOC = "docs/site/guide/io.md"
#: Top-level packages whose code is identified by name and version, never walked: their
#: executed code is in the cache key's code fingerprint (numpy, scipy, conatus) or is a library.
_LIBRARIES = frozenset({"numpy", "scipy", "conatus", "cupy", "h5py", "cupyx", "mpi4py", "numba"})


class IdentityWarning(UserWarning):
    """A comparison of identities that could not be complete (a file written before the
    identity record held what is now compared): what could not be compared is named."""


class CheckpointWarning(IdentityWarning):
    """A restart from a checkpoint that cannot be fully checked (written before the identity
    record): what could not be compared is named."""


class Undeclared(Exception):  # noqa: N818 -- an internal signal, never raised to the user
    """A dependency of a computation that has no content hash (the message names it)."""


# ------------------------------------------------------------------------------- hashing helpers
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def array_digest(a: Any) -> dict:
    """{shape, dtype, sha256} of an array's exact bytes (CuPy arrays copied to the host)."""
    if type(a).__module__.split(".")[0] == "cupy":
        a = a.get()
    a = np.ascontiguousarray(np.asarray(a))
    if a.dtype == object:
        raise Undeclared("an array of Python objects")
    return {"shape": [int(s) for s in a.shape], "dtype": a.dtype.str, "sha256": _sha(a.tobytes())}


def _is_library(module: str | None) -> bool:
    """True for code identified by its module name and version: the standard library, the
    packages of :data:`_LIBRARIES`, and anything installed in a site-packages directory."""
    if not module:
        return False
    root = module.split(".")[0]
    if root in _LIBRARIES or root in sys.stdlib_module_names or root == "builtins":
        return True
    mod = sys.modules.get(root)
    path = str(getattr(mod, "__file__", "") or "")
    return "site-packages" in path or "dist-packages" in path


def _version(module: str | None) -> str | None:
    mod = sys.modules.get((module or "").split(".")[0])
    v = getattr(mod, "__version__", None)
    return None if v is None else str(v)


def _code_digest(code: types.CodeType) -> str:
    """sha256 of a code object (bytecode, names, constants, nested code), never of its source
    text: a function defined in a notebook or by exec() is identified too."""
    h = hashlib.sha256()
    h.update(code.co_code)
    h.update(repr((code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars)).encode())
    for c in code.co_consts:
        h.update(_const_text(c).encode())
    return h.hexdigest()


def _const_text(c: Any) -> str:
    """Deterministic text of a code constant (a frozenset's order depends on the hash seed)."""
    if isinstance(c, types.CodeType):
        return "code:" + _code_digest(c)
    if isinstance(c, (frozenset, set)):
        return "frozenset:" + ",".join(sorted(_const_text(x) for x in c))
    if isinstance(c, tuple):
        return "tuple:(" + ",".join(_const_text(x) for x in c) + ")"
    return f"{type(c).__name__}:{c!r}"


def _global_names(code: types.CodeType) -> set[str]:
    """Names a code object (and the code nested in it: comprehensions, inner functions) looks up
    as globals. Attribute names (``np.sin`` -> ``sin``) are among them; a name that is not a
    global of the function is ignored by the caller."""
    out = set(code.co_names)
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            out |= _global_names(c)
    return out


#: Names whose call or attribute reads state the identity cannot follow statically: a computed
#: attribute or global name (``getattr(cfg, "factor")``: the name is a string constant, not a
#: code name), the environment, files, and code built at run time. A function whose code (or
#: the code of a user function it calls) names one of them has an undeclared dependency, so it
#: is never given a cache key (review 0.1.1, F5: ``getattr(cfg, 'factor')`` kept the key after
#: ``cfg.factor`` changed, a stale cache hit). Conservative: an attribute of the same name used
#: for something else (``obj.load``) refuses the cache too.
DYNAMIC_READS = frozenset({
    # computed attribute / global / name lookups
    "getattr", "hasattr", "vars", "globals", "locals", "__dict__", "__getattribute__",
    "attrgetter", "eval", "exec", "compile", "__import__", "import_module", "reload",
    # the environment
    "environ", "getenv", "environb",
    # files and other external input
    "open", "read_text", "read_bytes", "load", "loads", "loadtxt", "genfromtxt", "fromfile",
    "read_csv", "read_json", "read_hdf", "File", "urlopen", "input"})


def _dynamic_reads(code: types.CodeType) -> list[str]:
    """The names of :data:`DYNAMIC_READS` a code object (and the code nested in it) uses."""
    return sorted(_global_names(code) & DYNAMIC_READS)


def value_identity(v: Any, what: str, seen: set[int] | None = None,
                   _active: set[int] | None = None) -> Any:
    """Content identity of a value a computation reads: data by content (numbers, strings,
    arrays and containers of them), modules, library functions and classes by name and version,
    user functions by :func:`callable_identity`. Anything else raises :class:`Undeclared`
    naming ``what``: its state cannot be hashed, so a computation reading it is never cached.
    A container that contains itself is undeclared too (identity review: it used to recurse
    until RecursionError, a crash instead of a refusal); ``_active`` holds the containers being
    walked."""
    from ..runs._util import sha256_of
    from ..runs.keys import exact_hash
    if exact_hash(v):
        return {"data": sha256_of(v)}
    if isinstance(v, types.ModuleType):
        return {"module": v.__name__, "version": _version(v.__name__)}
    import enum
    if isinstance(v, enum.Enum):                       # an immutable member: by class and value
        return {"enum": f"{type(v).__module__}:{type(v).__qualname__}.{v.name}",
                "value": value_identity(v.value, f"{what}.value", seen, _active)}
    import functools
    if isinstance(v, (types.FunctionType, types.MethodType, types.BuiltinFunctionType,
                      np.ufunc, functools.partial, type)) or (
            callable(v) and callable(getattr(v, "__wrapped__", None))):
        return callable_identity(v, what, seen)
    if isinstance(v, (list, tuple)) or (isinstance(v, dict) and all(isinstance(k, str) for k in v)):
        _active = set() if _active is None else _active
        if id(v) in _active:
            raise Undeclared(f"{what} contains itself (a recursive container has no content hash)")
        _active.add(id(v))
        try:
            if isinstance(v, dict):
                return {k: value_identity(x, f"{what}[{k!r}]", seen, _active)
                        for k, x in sorted(v.items())}
            return [value_identity(x, f"{what}[{i}]", seen, _active) for i, x in enumerate(v)]
        finally:
            _active.discard(id(v))
    raise Undeclared(f"{what} is a {type(v).__name__}, which has no content hash")


def callable_identity(fn: Any, what: str | None = None, seen: set[int] | None = None) -> dict:
    """Identity of a callable a computation depends on: a library function or class (numpy,
    scipy, the standard library, anything installed) by module, name and version; a user function
    by its code and by the content of every value it reads: its closure cells and the module
    globals its code names (followed into the user functions it calls). Two closures of one
    factory (``make(1)``, ``make(2)``) are two identities (review finding 49). Raises
    :class:`Undeclared` when one of those values has no content hash (an object, an empty cell):
    the computation then has an undeclared dynamic dependency."""
    seen = set() if seen is None else seen
    import functools
    # a Python wrapper (functools.wraps) is identified by its own code and closure, which hold
    # the function it wraps: unwrapping it dropped the decorator's parameters (``@scale(2)`` and
    # ``@scale(3)`` had one identity, identity review); a wrapper object without code
    # (functools.lru_cache) is identified by the function it wraps
    while (not isinstance(fn, (types.FunctionType, types.MethodType))
           and callable(getattr(fn, "__wrapped__", None))):
        fn = fn.__wrapped__
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or type(fn).__name__
    what = what or name
    if isinstance(fn, functools.partial):
        return {"partial": callable_identity(fn.func, what, seen),
                "args": value_identity(list(fn.args), f"{what} (partial args)", seen),
                "keywords": value_identity(dict(fn.keywords), f"{what} (partial keywords)", seen)}
    module = getattr(fn, "__module__", None)
    if isinstance(fn, np.ufunc):
        return {"library": f"numpy.{fn.__name__}", "version": np.__version__}
    if isinstance(fn, type) or isinstance(fn, types.BuiltinFunctionType) or _is_library(module):
        if isinstance(fn, type) and not _is_library(module):
            # a user class: by the code of its methods (its state is not hashed: declared)
            try:
                return {"class": f"{module}:{name}",
                        "source_sha256": _sha(inspect.getsource(fn).encode())}
            except (OSError, TypeError):
                raise Undeclared(f"{what}: class {name} without source code") from None
        return {"library": f"{module}:{name}", "version": _version(module)}
    if isinstance(fn, types.MethodType):
        return {"method": callable_identity(fn.__func__, what, seen),
                "self": value_identity(fn.__self__, f"{what} (bound object)", seen)}
    code = getattr(fn, "__code__", None)
    if not isinstance(code, types.CodeType):
        raise Undeclared(f"{what} is a callable {type(fn).__name__} object, whose state has no "
                         "content hash")
    dyn = _dynamic_reads(code)
    if dyn:
        raise Undeclared(
            f"{what}: its code uses {', '.join(dyn)}, a read whose target is chosen at run time "
            "(a computed attribute or name, the environment or a file): what it reads has no "
            "content hash")
    out: dict[str, Any] = {"function": f"{module}:{name}", "code_sha256": _code_digest(code)}
    if id(fn) in seen:                                  # recursion: its code is enough
        return out
    seen.add(id(fn))
    cells = {}
    for var, cell in zip(code.co_freevars, fn.__closure__ or (), strict=False):
        try:
            v = cell.cell_contents
        except ValueError:
            raise Undeclared(f"{what}: closure variable {var!r} is not set yet") from None
        cells[var] = value_identity(v, f"{what}: closure variable {var!r}", seen)
        attrs = _user_reads(v, _global_names(code), f"{what}: closure variable {var!r}", seen)
        if attrs:
            cells[var]["attributes"] = attrs
    if cells:
        out["closure"] = cells
    g = getattr(fn, "__globals__", {}) or {}
    names = _global_names(code)
    reads = {}
    for n in sorted(names):
        # a module global that shadows a builtin (``max = 3.0``) is read like any other global
        # (identity review: every builtin NAME was skipped, so changing the shadowing value kept
        # the key); only the builtin object itself is left out
        if n in g and not (n in vars(builtins) and g[n] is vars(builtins)[n]):
            reads[n] = value_identity(g[n], f"{what}: global {n!r}", seen)
            attrs = _user_reads(g[n], names, f"{what}: global {n!r}", seen)
            if attrs:
                reads[n]["attributes"] = attrs
    if reads:
        out["globals"] = reads
    if getattr(fn, "__defaults__", None):
        out["defaults"] = value_identity(list(fn.__defaults__), f"{what}: default values", seen)
    if getattr(fn, "__kwdefaults__", None):
        out["kwdefaults"] = value_identity(dict(fn.__kwdefaults__), f"{what}: default values",
                                           seen)
    return out


def _user_reads(v: Any, names: set[str], what: str, seen: set[int]) -> dict:
    """Attributes a code object may read from ``v`` when ``v`` is mutable user state identified
    otherwise by name or source only: a user module (:func:`_module_reads`) or a user class
    (:func:`_class_reads`). Empty for anything else."""
    if isinstance(v, types.ModuleType) and not _is_library(v.__name__):
        return _module_reads(v, names, what, seen)
    if isinstance(v, type) and not _is_library(getattr(v, "__module__", None)):
        return _class_reads(v, names, what, seen)
    return {}


def _class_reads(cls: type, names: set[str], what: str, seen: set[int]) -> dict:
    """Content identity of the attributes of a USER class that a code object may read
    (``Cfg.A``: ``A`` is among the code's names). A class is identified by its source, which
    holds the values it was defined with, not the ones assigned later: ``Cfg.A = 3`` after
    ``Cfg.A = 1`` kept the key (a stale cache hit, the hole of :func:`_module_reads` for a class
    used as a namespace; identity review). Attributes are looked up along the user classes of
    the MRO; methods by :func:`callable_identity` (their code and the globals they read),
    properties by their accessors."""
    out: dict[str, Any] = {}
    mro = [k for k in cls.__mro__ if not _is_library(getattr(k, "__module__", None))]
    for a in sorted(names):
        if a.startswith("__"):
            continue
        members = getattr(cls, "__members__", None)        # an Enum: its members by name
        if isinstance(members, Mapping) and a in members:
            v = members[a]
        else:
            owner = next((k for k in mro if a in vars(k)), None)
            if owner is None:
                continue
            v = vars(owner)[a]
        where = f"{what}.{a}"
        if isinstance(v, (staticmethod, classmethod)):
            out[a] = callable_identity(v.__func__, where, seen)
        elif isinstance(v, property):
            out[a] = {"property": [None if f is None else callable_identity(f, where, seen)
                                   for f in (v.fget, v.fset, v.fdel)]}
        else:
            out[a] = value_identity(v, where, seen)
            sub = _user_reads(v, names, where, seen) if isinstance(v, type) and v is not cls \
                else {}
            if sub:
                out[a]["attributes"] = sub
    return out


def _module_reads(mod: types.ModuleType, names: set[str], what: str, seen: set[int],
                  depth: int = 0) -> dict:
    """Content identity of the attributes of a USER module that a code object may read
    (``cfg.A``: ``A`` is among the code's names). A user module is mutable state, not a library
    identified by its version: identifying it by name only let ``cfg.A = 3`` keep the key of
    ``cfg.A = 1`` (a stale cache hit, critic review of the identity lot). Every attribute of the
    module whose name the code uses is hashed (conservative: a name used for another purpose is
    hashed too); sub-modules of the user package are followed."""
    out: dict[str, Any] = {}
    for a in sorted(names):
        if a.startswith("__") or a not in vars(mod):
            continue
        v = vars(mod)[a]
        if isinstance(v, types.ModuleType):
            if _is_library(v.__name__) or depth >= 4:
                out[a] = value_identity(v, f"{what}.{a}", seen)
            else:
                out[a] = {"module": v.__name__,
                          "attributes": _module_reads(v, names, f"{what}.{a}", seen, depth + 1)}
        else:
            out[a] = value_identity(v, f"{what}.{a}", seen)
            if isinstance(v, type) and not _is_library(getattr(v, "__module__", None)):
                attrs = _class_reads(v, names, f"{what}.{a}", seen)
                if attrs:
                    out[a]["attributes"] = attrs
    return out


def digest(obj: Any) -> str:
    """sha256 of the canonical encoding of an identity (or of any part of one)."""
    from ..runs._util import sha256_of
    return sha256_of(obj)


# --------------------------------------------------------------------------------- the parts
def equations_hash(problem: Any) -> str:
    """Hash of the equations as compiled: kind, unknown names and tensor signatures (in order:
    they lay out the state), both sides of every equation (in any order) and the symmetry
    subspace. Stable across processes, platforms and Python versions (sha256 of canonical
    structural keys; whitespace and the order of terms or equations do not change it). Parameter
    values are not part of it (``Parameter``'s structural key has the name only); the domains of
    the fields enter through the structural keys (size, bounds, dealiasing factor), which is why
    it is compared only at equal resolution and geometry, and why a change of the dealiasing
    factor shows as an equations change."""
    h = hashlib.sha256()
    h.update(str(getattr(problem, "kind", "")).encode())
    for v in getattr(problem, "variables", ()):
        h.update(f"var:{v.name}:{len(v.tensorsig)}".encode())
    # the equations as a set: declaring them in another order is the same system (the state is
    # laid out by the unknowns, hashed in order above); each side's structural hash is a sha256 of
    # the canonical expression (sums and products sorted), never Python's salted hash()
    for e in sorted(f"{eq.lhs.structural_hash}={eq.rhs.structural_hash}"
                    for eq in getattr(problem, "equations", ())):
        h.update(e.encode())
    sym = getattr(problem, "symmetry", None)
    if sym is not None:
        h.update(repr(sym).encode())
    return h.hexdigest()[:16]


def _bases(plan: Any) -> list:
    dom = getattr(plan, "domain", None)
    return list(getattr(dom, "bases", ()) or ())


def given_data(plan: Any) -> tuple[dict, list[str]]:
    """(``{name: {domain, rank, shape, dtype, sha256}}``, undeclared) of every given field /
    bound coefficient of ``plan``: the data the plan uses (``_ncc_data``: grid values evaluated
    once by the plan, or coefficients on curvilinear plans; ``_ncc_src`` on a mapped plan, where a
    data function is identified by :func:`callable_identity`). An entry that cannot be hashed by
    content is listed in ``undeclared`` (and its entry says why)."""
    from ..reference.evaluate import FieldData
    src = getattr(plan, "_ncc_data", None)
    if src is None:
        src = getattr(plan, "_ncc_src", None) or {}
    out: dict[str, Any] = {}
    undeclared: list[str] = []
    for f, d in sorted(src.items(), key=lambda kv: getattr(kv[0], "name", str(kv[0]))):
        name = getattr(f, "name", str(f))
        dom = getattr(f, "domain", None)
        entry: dict[str, Any] = {
            "domain": _sha(repr(getattr(dom, "key", None)).encode())[:16],
            "rank": len(getattr(f, "tensorsig", ()) or ())}
        try:
            if isinstance(d, FieldData):
                entry.update(array_digest(d.data), layout=d.layout)
            elif callable(d) and not isinstance(d, np.ndarray) and not hasattr(d, "__array__"):
                entry["function"] = callable_identity(d, f"data of the given field {name}")
            else:
                entry.update(array_digest(d))
        except Undeclared as why:
            entry["undeclared"] = str(why)
            undeclared.append(f"given field {name}: {why}")
        out[name] = entry
    return out, undeclared


def user_functions(problem: Any) -> tuple[dict, list[str]]:
    """(``{name: identity}``, undeclared) of the user pointwise functions (``m.function``,
    ``conatus.ir.pointwise``) the equations call: their Python code is a dependency of the
    computation that the structural hash (the name only) does not see."""
    from ..ir import expr_algebra as A
    names = set()
    for eq in getattr(problem, "equations", ()):
        for side in (eq.lhs, eq.rhs):
            for e in side.walk():
                if isinstance(e, A.UnaryFunction) and A.is_user_function(e.func):
                    names.add(e.func)
    out: dict[str, Any] = {}
    undeclared: list[str] = []
    for key in sorted(names, key=lambda k: (A.display_name(k), k)):
        # by the user's name, not the registry key (g__2 depends on what the process declared
        # before: F105, numerics review); the code below tells two functions named g apart
        f, deriv, parity = A._USER_FUNCS[key]
        n = A.display_name(key)
        try:
            out[n] = {"f": callable_identity(f, f"function {n}"), "parity": parity,
                      "derivative": None if deriv is None
                      else callable_identity(deriv, f"derivative of function {n}")}
        except Undeclared as why:
            out[n] = {"undeclared": str(why)}
            undeclared.append(f"function {n}: {why}")
    return out, undeclared


def _attr(obj: Any, name: str) -> Any:
    """``getattr`` that records a failure instead of raising (identity is read, never fails)."""
    try:
        return getattr(obj, name, None)
    except Exception as err:  # noqa: BLE001 -- a property that cannot be read is recorded
        return f"unreadable: {type(err).__name__}: {err}"


def stepping(solver: Any) -> dict:
    """The numerical choices of a time stepper that change its numbers: scheme, dt, start-up,
    time parameter, boundary-data convention (``bc_time`` as resolved), the backends of the
    solve and of F, dtype, ordering, reduced space, complexify / decouple, and the pivoting of a
    device system (review finding 22: the record called complete omitted several)."""
    system = getattr(solver, "system", None)
    host = getattr(system, "host", system)
    return {"scheme": _attr(solver, "scheme"), "dt": _attr(solver, "dt"),
            "startup": _attr(solver, "startup"),
            "startup_substeps": _attr(solver, "startup_substeps"),
            "time_parameter": _attr(solver, "time_parameter"),
            "bc_time": _attr(solver, "bc_time"),
            "backend": _attr(solver, "backend"), "f_backend": _attr(solver, "f_backend"),
            "ordering": _attr(host, "ordering"), "reduced_space": _attr(host, "reduced"),
            "complexify": _attr(host, "complexify"), "decouple": _attr(host, "decouple"),
            "pivot": _attr(system, "pivot"),
            "dtype": str(np.dtype(getattr(system, "dtype", np.float64)))}


def checkpoint_identity(solver: Any) -> dict:
    """The identity record a checkpoint carries: :func:`computation_identity` without the
    numerics (compared by the solver at restart; a CPU state may continue on a GPU) and without
    the list of undeclared dependencies."""
    ident = computation_identity(solver)
    return {k: v for k, v in ident.items() if k not in ("numerics", "undeclared")}


def computation_identity(obj: Any) -> dict:
    """The canonical identity of the computation of ``obj`` (a solver, or a plan): module
    docstring. JSON-safe; :func:`digest` gives its hash."""
    from ._describe import parameters_of
    solver = obj if hasattr(obj, "plan") and hasattr(obj, "scheme") else None
    plan = solver.plan if solver is not None else obj
    problem = getattr(plan, "problem", None)
    res, geo = {}, {}
    for b in _bases(plan):
        c = b.coord.name
        res[c] = int(b.size) if not isinstance(b.size, tuple) else [int(s) for s in b.size]
        g = [type(b).__name__]
        bounds = getattr(b, "bounds", None)
        if bounds is not None:
            g.append([float(x) for x in bounds])
        radius = getattr(b, "radius", None)
        if radius is not None and not callable(radius):
            g.append(float(radius))
        geo[c] = g
    data, und_d = given_data(plan)
    funcs, und_f = user_functions(problem)
    out = {"version": IDENTITY_VERSION,
           "model": str(getattr(problem, "name", "") or ""),
           "equations": equations_hash(problem),
           "parameters": parameters_of(solver if solver is not None else _PlanView(plan)),
           "resolution": res,
           "geometry": geo,
           "data": data,
           "functions": funcs,
           "undeclared": und_d + und_f}
    if solver is not None:
        out["numerics"] = stepping(solver)
    return out


class _PlanView:
    """A plan seen as a solver by :func:`conatus.io._describe.parameters_of` (``.plan``)."""

    def __init__(self, plan: Any) -> None:
        self.plan = plan


def allowable_names(solver: Any) -> list[str]:
    """Names ``allow=`` accepts besides the categories: parameters, given fields, functions."""
    from ._describe import parameters_of
    data, _ = given_data(solver.plan)
    funcs, _ = user_functions(getattr(solver.plan, "problem", None))
    return [*parameters_of(solver), *data, *funcs]


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, list) and len(v) == 2 and all(isinstance(x, float) for x in v):
        return f"{complex(v[0], v[1]):g}"
    return repr(v)


def _data_text(name: str, a: dict | None, b: dict | None, there: str) -> str:
    """One line naming how the data of a given field differ (content, shape, dtype, domain)."""
    if a is None or b is None:
        return (f"given field {name}: {'absent' if a is None else 'present'} {there}, "
                f"{'absent' if b is None else 'present'} now")
    what = [k for k in ("domain", "rank", "shape", "dtype", "layout") if a.get(k) != b.get(k)]
    if what:
        return (f"given field {name}: its data differ ({', '.join(what)}: "
                + "; ".join(f"{a.get(k)!r} {there}, {b.get(k)!r} now" for k in what) + ")")
    if "sha256" in a and "sha256" in b:
        return (f"given field {name}: its data differ (same shape and dtype, other values: "
                f"sha256 {str(a['sha256'])[:12]} {there}, {str(b['sha256'])[:12]} now)")
    return f"given field {name}: its data differ (the data function or the values it reads)"


def _differences(old: dict, new: dict, there: str = "in the checkpoint"
                 ) -> list[tuple[str, str, str]]:
    """(allow key, category, text) of every difference between two identity records (``there``:
    where the old one comes from, for the texts)."""
    out = []
    same_layout = old.get("resolution") == new.get("resolution") and \
        old.get("geometry") == new.get("geometry")
    for key in IDENTITY_KEYS:
        if key not in old or key not in new:
            continue
        a, b = old[key], new[key]
        if key == "parameters":
            for p in sorted(set(a) | set(b)):
                if a.get(p) != b.get(p):
                    out.append((p, "parameters", f"parameter {p}: {_fmt(a.get(p)) if p in a else '(absent)'}"
                                f" {there}, {_fmt(b.get(p)) if p in b else '(absent)'} now"))
        elif key == "data":
            for n in sorted(set(a) | set(b)):
                if a.get(n) != b.get(n):
                    out.append((n, "data", _data_text(n, a.get(n), b.get(n), there)))
        elif key == "functions":
            for n in sorted(set(a) | set(b)):
                if a.get(n) != b.get(n):
                    out.append((n, "functions", f"function {n}: its code or the values it reads "
                                "differ" if n in a and n in b else
                                f"function {n}: {'absent' if n not in a else 'present'} {there}, "
                                f"{'absent' if n not in b else 'present'} now"))
        elif key == "equations":
            if same_layout and a != b:
                out.append(("equations", "equations", f"equations: hash {a} {there}, "
                            f"{b} now (another equation, term or sign, or another dealiasing "
                            "factor)"))
        elif key in ("resolution", "geometry"):
            for c in sorted(set(a) | set(b)):
                if a.get(c) != b.get(c):
                    out.append((key, key, f"{key} along {c}: {a.get(c)!r} {there}, "
                                f"{b.get(c)!r} now"))
        elif a != b:
            out.append((key, key, f"{key}: {a!r} {there}, {b!r} now"))
    return out


def unrecorded(old: dict, new: dict) -> list[str]:
    """The parts of ``new`` an older identity record does not hold (so not compared), when they
    are not empty: e.g. the given-field data of a 0.1.0 checkpoint."""
    return [k for k in IDENTITY_KEYS if k not in old and new.get(k)]


def check_allow(allow: Any, params: Iterable[str], where: str) -> tuple[str, ...]:
    """``allow=`` as a tuple of names: parameter, given-field or function names
    (``params``: :func:`allowable_names`) or the categories of ALLOWABLE."""
    if allow is None or allow == ():
        return ()
    if isinstance(allow, str):
        allow = (allow,)
    try:
        names = tuple(str(a) for a in allow)
    except TypeError:
        raise OptionError(f"{where}(allow={allow!r})",
                          why="allow= lists what may differ between the checkpoint and this run",
                          fix="allow=('kappa',) (a parameter) or allow=('parameters',)",
                          doc=_DOC) from None
    params = list(params)
    for n in names:
        if n == "resolution":
            raise OptionError(f"{where}(allow=('resolution',))",
                              why="the state is a vector of spectral coefficients laid out for one "
                                  "resolution: read at another one it would be another state",
                              fix="restart at the resolution of the checkpoint, then interpolate "
                                  "explicitly (sim.grid(...) / initial(...) at the new N)", doc=_DOC)
        if n not in params and n not in ALLOWABLE:
            raise OptionError(f"{where}(allow=({n!r},)): {n!r} is neither a parameter, a "
                              "given field or a function of this run nor one of " + ", ".join(repr(a) for a in ALLOWABLE)
                              + "." + did_you_mean(n, [*params, *ALLOWABLE]),
                              why="allow= names what may differ between the checkpoint and this "
                                  "run; an unknown name would allow nothing and hide the typo",
                              fix="names of this run: " + (", ".join(params) or "(none)"),
                              doc=_DOC)
    return names


def compare_identity(meta: dict, solver: Any, allow: tuple[str, ...], where: str,
                     path: Any = None) -> list[str]:
    """Compare a checkpoint's identity with ``solver``; refuse what ``allow`` does not name.
    Returns the allowed differences (text, recorded by the caller)."""
    mine = checkpoint_identity(solver)
    old = meta.get("identity")
    src = f" {path}" if path is not None else ""
    if old is None:
        prob = meta.get("problem")
        if prob is not None and prob != mine["model"] and "model" not in allow:
            raise OptionError(
                f"{where}: the checkpoint{src} is a state of '{prob}', this run solves "
                f"'{mine['model']}'",
                why="a restart continues the state of the same problem; another model would "
                    "start from a state that is not its own",
                fix="restart from a checkpoint of this model, or allow=('model',) to continue this "
                    "state deliberately (recorded by explain())", doc=_DOC)
        warnings.warn(
            f"{where}: the checkpoint{src} was written before checkpoints recorded their model, "
            "equations, parameters, resolution and geometry: only the problem name and the "
            "numerics (scheme, dt, size of the state) were compared. Re-save it (sim.save(...)) "
            "after the restart to record them.", CheckpointWarning, stacklevel=4)
        return []
    missing = unrecorded(old, mine)
    if missing:
        warnings.warn(
            f"{where}: the checkpoint{src} was written before checkpoints recorded "
            + " and ".join({"data": "the data of the given fields",
                            "functions": "the user functions"}.get(k, k) for k in missing)
            + ": those were not compared (" + ", ".join(sorted(
                n for k in missing for n in mine[k])) + "). Re-save it (sim.save(...)) after "
            "the restart to record them.", CheckpointWarning, stacklevel=4)
    diffs = _differences(old, mine)
    refused = [t for key, cat, t in diffs if key not in allow and cat not in allow]
    if refused:
        raise OptionError(
            f"{where}: the checkpoint{src} is not a state of this run: " + "; ".join(refused),
            why="a restart continues the same problem; a checkpoint of another model, other "
                "equations, other parameters, other given data, another resolution or another "
                "box would continue silently as if it were this run's own state",
            fix=_fix(diffs, allow), doc=_DOC)
    return [t for _k, _c, t in diffs]


def _fix(diffs: list, allow: tuple[str, ...]) -> str:
    """The fix line of a refused restart: what to rebuild, or what ``allow=`` would name."""
    cats = [c for k, c, _ in diffs if k not in allow and c not in allow]
    if "resolution" in cats:
        return ("build the run at the resolution of the checkpoint (a state cannot be read at "
                "another N; interpolate explicitly with sim.grid(...) and initial(...))")
    keys = list(dict.fromkeys(k for k, c, _ in diffs if k not in allow and c not in allow))
    return ("rebuild the run with the model, equations, parameters, given data and box of the "
            "checkpoint, "
            "or name what may change deliberately: allow=(" + ", ".join(repr(k) for k in keys)
            + ("," if len(keys) == 1 else "") + ") (recorded by explain() and in every later "
            "checkpoint)")
