"""Arguments that do not match a public signature are refused with a typed error (hunt 5E, R1).

Python refuses ``sim.run(nsteps=3)`` with a bare ``TypeError`` that names neither the accepted
keywords nor the new name of a former one. Every public callable of the package is therefore
guarded: a function or class named in the ``__all__`` of the ``conatus`` module that defines it, and
the ``__init__`` and public methods of such a class. When a call does not bind to the callable's
signature, the guard raises :class:`~conatus.errors.ArgumentError` (``CN151``, an ``OptionError``
and a ``TypeError``) with

* the keyword that is not accepted, the new name when it is a former one
  (:data:`conatus._deprecation.VOCABULARY`), else the closest accepted names;
* every accepted keyword name, in ``fix``;
* or the missing required argument, or Python's own text for the other binding errors.

The guard costs nothing on a call that binds: it calls the function and looks at the signature
only when a ``TypeError`` comes back, and converts it only when the arguments do not bind to the
signature (a ``TypeError`` raised inside the body is re-raised unchanged).

The guard is installed by the import hook of :mod:`conatus._import_record` when a ``conatus``
module has finished executing (:func:`guard_module`). Classes are patched in place, so their
identity does not change; a module-level function is replaced by its guarded version in its
module and in every ``conatus`` module that imported it while the defining module was still
executing (circular imports), so that one name keeps one object. A module loaded without a source
file is not guarded.
"""
from __future__ import annotations

import functools
import inspect
import sys
import types
from typing import Any

#: Original function id -> guarded function, to replace the references taken during a circular import.
_GUARDED: dict[int, Any] = {}
#: Modules that finished executing while another conatus module was executing (innermost last).
_STACK: list[list[str]] = []


def keyword_names(func: Any) -> list[str]:
    """The names a call of ``func`` may pass by keyword (positional-or-keyword and keyword-only),
    ``self`` and ``cls`` excluded."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return []
    return [n for n, p in params.items() if n not in ("self", "cls")
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]


def keyword_hint(name: str, accepted: list[str] | tuple[str, ...]) -> str:
    """``" Did you mean 'steps'? (nsteps is its former name, removed in 0.3)"`` for a former name
    whose new name is accepted, else the closest accepted names (``errors.did_you_mean``)."""
    from .errors import did_you_mean
    try:
        from ._deprecation import ARGUMENTS_REMOVED_IN, VOCABULARY
    except ImportError:
        ARGUMENTS_REMOVED_IN, VOCABULARY = "0.3", {}
    new = VOCABULARY.get(name)
    if new is not None and new in accepted:
        return (f" Did you mean {new!r}? ({name} is its former name, removed in "
                f"{ARGUMENTS_REMOVED_IN})")
    return did_you_mean(name, accepted)


def _label(func: Any) -> str:
    q = getattr(func, "__qualname__", getattr(func, "__name__", "call"))
    return q[:-len(".__init__")] if q.endswith(".__init__") else q


def refusal(func: Any, args: tuple, kw: dict, err: TypeError) -> Exception | None:
    """The :class:`ArgumentError` for a call of ``func`` whose arguments do not bind to its
    signature, or ``None`` when they bind (``err`` was raised by the body)."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return None
    try:
        sig.bind(*args, **kw)
        return None
    except TypeError as bind_err:
        reason = str(bind_err)
    try:
        from .errors import ArgumentError
    except ImportError:             # a partial copy of the package (the hub image): Python's error
        return None
    where = _label(func)
    accepted = keyword_names(func)
    has_var_kw = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
    listed = ", ".join(accepted) or "none"
    unknown = [] if has_var_kw else [k for k in kw if k not in accepted]
    if unknown:
        k = unknown[0]
        return ArgumentError(f"{where}(..., {k}=...): unknown keyword." + keyword_hint(k, accepted),
                             why=f"{where} accepts the keywords: {listed}",
                             fix=f"use one of: {listed}")
    missing = [n for n, p in sig.parameters.items() if p.default is p.empty
               and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and n not in kw
               and n not in ("self", "cls")]
    if "missing" in reason and missing:
        return ArgumentError(f"{where}(): missing required argument {missing[0]!r}",
                             why=f"{where} has no default for {missing[0]}",
                             fix=f"{where}(..., {missing[0]}=...); accepted keywords: {listed}")
    return ArgumentError(f"{where}(): {reason}", fix=f"accepted keywords: {listed}")


def guard(func: Any) -> Any:
    """``func`` wrapped so that arguments that do not bind to its signature raise
    :class:`ArgumentError` (the function itself when it is already guarded)."""
    if getattr(func, "__conatus_guarded__", False):
        return func

    # the code of the innermost function: a TypeError whose traceback passes through it was
    # raised by its body, so the call did bind (a decorator in between may accept other keywords
    # than the function it wraps, and its signature is not the one inspect.signature reports)
    inner = getattr(inspect.unwrap(func), "__code__", None)

    @functools.wraps(func)
    def guarded(*args: Any, **kw: Any):
        try:
            return func(*args, **kw)
        except TypeError as err:
            tb = err.__traceback__
            if tb is not None and tb.tb_next is not None and not hasattr(func, "__wrapped__"):
                raise       # raised inside the body of a plain function: the call did bind
            while inner is not None and tb is not None:
                if tb.tb_frame.f_code is inner:
                    raise   # raised inside the body of the wrapped function: the call did bind
                tb = tb.tb_next
            new = refusal(func, args, kw, err)
            if new is None:
                raise
            raise new from None
    guarded.__conatus_guarded__ = True
    return guarded


def _guard_class(cls: type) -> None:
    for name, v in list(vars(cls).items()):
        if name.startswith("_") and name != "__init__":
            continue
        if isinstance(v, staticmethod):
            if inspect.isfunction(v.__func__):
                setattr(cls, name, staticmethod(guard(v.__func__)))
        elif isinstance(v, classmethod):
            if inspect.isfunction(v.__func__):
                setattr(cls, name, classmethod(guard(v.__func__)))
        elif inspect.isfunction(v):
            setattr(cls, name, guard(v))


def _public(module: types.ModuleType) -> list[tuple[str, Any]]:
    """(name, object) of the functions and classes of the package that ``module`` lists in its
    ``__all__``: defined there, or re-exported from another conatus module (a package that
    flattens its submodules, ``conatus.expert``)."""
    names = getattr(module, "__all__", None)
    if not isinstance(names, (list, tuple)):
        return []
    out = []
    for n in names:
        obj = module.__dict__.get(n)
        if str(getattr(obj, "__module__", "")).startswith("conatus") and (
                inspect.isfunction(obj) or inspect.isclass(obj)):
            out.append((n, obj))
    return out


def begin_module() -> None:
    """Called when a conatus module starts executing (the import hook)."""
    _STACK.append([])


def guard_module(module: types.ModuleType) -> None:
    """Guard the public callables that ``module`` defines (the import hook, after execution)."""
    nested = _STACK.pop() if _STACK else []
    if _STACK:
        _STACK[-1].append(module.__name__)
        _STACK[-1].extend(nested)
    replaced: dict[int, Any] = {}
    reexported = False
    for n, obj in _public(module):
        if inspect.isclass(obj):
            if not issubclass(obj, BaseException):
                for c in obj.__mro__:       # the methods it inherits from conatus classes too
                    if str(getattr(c, "__module__", "")).startswith("conatus."):
                        _guard_class(c)
        elif not getattr(obj, "__conatus_guarded__", False):
            g = _GUARDED.get(id(obj))
            if g is None or g.__wrapped__ is not obj:
                g = guard(obj)
                _GUARDED[id(obj)] = g
            replaced[id(obj)] = obj
            module.__dict__[n] = g
            reexported |= obj.__module__ != module.__name__
    if not replaced:
        return
    # the same function under another name of this module, and in the modules that took it before
    # it was guarded: those imported meanwhile (circular imports) or, for a re-exported function,
    # every conatus module already loaded (its defining module included)
    if reexported:
        names = [k for k in list(sys.modules) if k == "conatus" or k.startswith("conatus.")]
    else:
        names = [module.__name__, *nested]
    for name in names:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        d = getattr(mod, "__dict__", {})
        for k, v in list(d.items()):
            if id(v) in replaced and v is replaced[id(v)]:
                d[k] = _GUARDED[id(v)]
