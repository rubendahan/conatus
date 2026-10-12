"""Packages whose call is their main function (one API story).

``conatus.linearize``, ``conatus.discretize``, ``conatus.methods`` and ``conatus.verify`` are
packages *and* the names of the expert-layer functions ``conatus.expert.linearize`` etc. After
``import conatus as cn``, ``cn.discretize`` is the package (Python sets a submodule as an attribute
of its parent). So that the same name never means two different things, calling the package calls
the function of the same name, with the same signature (``inspect.signature`` reports it):

    cn.discretize(problem)          # == conatus.expert.discretize(problem)
    cn.discretize.Plan              # the package's contents stay reachable

Nothing else changes: the package is still a module (``isinstance(.., types.ModuleType)``), its
submodules import as before. The development test suite checks that every name shared by ``cn``
and ``conatus.expert`` is the same object or a package whose call is that object.
"""
from __future__ import annotations

import inspect
import sys
import types
from collections.abc import Callable
from typing import Any


class CallablePackage(types.ModuleType):
    """A package module whose ``__call__`` forwards to its main function (``__call_target__``)."""

    __call_target__: Callable[..., Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.__call_target__(*args, **kwargs)


def make_callable(module_name: str, fn: Callable[..., Any]) -> None:
    """Make the (already imported) package ``module_name`` callable as ``fn``."""
    mod = sys.modules[module_name]
    mod.__class__ = CallablePackage
    mod.__call_target__ = fn
    try:
        mod.__signature__ = inspect.signature(fn)
    except (TypeError, ValueError):          # builtins without a signature: none reported
        pass
