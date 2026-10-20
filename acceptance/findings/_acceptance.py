"""Shared helpers of the acceptance scripts: the result line, the refusal test, tolerances.

Every script of this directory reproduces one finding of the external review of 0.1.0 against
the installed package's public API and asserts the CORRECT behaviour. It prints exactly one line

    RESULT F<nnn> PASS|FAIL|SKIP <detail>

and exits 0 (PASS), 1 (FAIL) or 3 (SKIP: the feature is not shipped in this build). An uncaught
exception is a FAIL too (the runner reports it as a crash). Standard library and NumPy only.
"""
from __future__ import annotations

import os
import sys
import tempfile
import warnings

_NUMBER: list[int] = []


def start(number: int, title: str) -> None:
    """Declare the finding this script checks; print its title; work in a fresh directory."""
    _NUMBER[:] = [number]
    print(f"F{number:03d}: {title}", flush=True)
    os.chdir(tempfile.mkdtemp(prefix=f"cn_accept_f{number:03d}_"))
    warnings.simplefilter("default")          # warnings visible in the log, never fatal


def _result(word: str, detail: str, code: int) -> None:
    print(f"RESULT F{_NUMBER[0]:03d} {word} {detail}".rstrip(), flush=True)
    sys.exit(code)


def passed(detail: str = "") -> None:
    _result("PASS", detail, 0)


def failed(detail: str) -> None:
    _result("FAIL", detail, 1)


def skipped(detail: str) -> None:
    _result("SKIP", detail, 3)


def check(condition: bool, detail: str) -> None:
    """FAIL with ``detail`` unless ``condition`` holds."""
    if not condition:
        failed(detail)


def is_refusal(exc: BaseException) -> bool:
    """A clear, typed refusal: an instance of ``conatus.errors.ConatusError`` (the package's own
    error hierarchy, which carries the why / fix of the refusal). A bare ValueError or TypeError
    (e.g. a NumPy broadcast failure), IndexError, KeyError, LinAlgError... is a crash of the
    computation, not a refusal."""
    try:
        from conatus.errors import ConatusError
    except ImportError:                     # a build without the error module: nothing is typed
        return False
    return isinstance(exc, ConatusError)


def describe(exc: BaseException) -> str:
    """One line: exception type and the first line of its message."""
    msg = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {msg[0][:160] if msg else ''}"


def refused(call, /, *args, **kw):
    """Call ``call``; return (True, exc) on a typed refusal, (False, value) on success. Any other
    exception is a FAIL (a crash is not a refusal)."""
    try:
        return False, call(*args, **kw)
    except Exception as exc:  # noqa: BLE001 -- classified below: refusal or crash
        if is_refusal(exc):
            print(f"  refused: {describe(exc)}")
            return True, exc
        failed(f"crash instead of a result or a typed refusal: {describe(exc)}")
    return False, None                      # unreachable (failed exits)


def need(module: str) -> None:
    """SKIP when ``module`` is not in this build (cut from the public export, or optional)."""
    import importlib
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name and (module == exc.name or module.startswith(exc.name + ".")):
            skipped(f"{module} is not shipped in this build")
        raise
