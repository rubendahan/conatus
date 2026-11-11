"""How the Python process ends, for the final status of a watched run (hunt 5F, SF1).

A watch closes at exit (``atexit``) and sends the final status. Before 0.3.2 that status carried
``failed`` only when the exception was raised inside ``sim.run``: a script that raised between
two runs or after the last one, or called ``sys.exit(3)``, was shown "finished". :func:`install`
records

* an uncaught exception of the main thread (``sys.excepthook``, which Python calls before the
  ``atexit`` handlers; ``KeyboardInterrupt`` included);
* a non-zero exit code given to ``sys.exit`` (Python calls no hook for ``SystemExit``, so the
  function is wrapped; a ``raise SystemExit(3)`` written directly is not seen).

:func:`outcome` returns the exception to report, or ``None`` for a normal end. The previous hooks
are called unchanged.
"""
from __future__ import annotations

import sys
from typing import Any

_STATE: dict[str, Any] = {"installed": False, "error": None}


def _excepthook(kind, value, tb, _previous=None):
    _STATE["error"] = value if value is not None else kind()
    (_previous or sys.__excepthook__)(kind, value, tb)


def install() -> None:
    """Install the two recorders once per process."""
    if _STATE["installed"]:
        return
    _STATE["installed"] = True
    previous_hook = sys.excepthook
    sys.excepthook = lambda k, v, t: _excepthook(k, v, t, previous_hook)
    previous_exit = sys.exit

    def exit(status: Any = None) -> None:          # noqa: A001 -- the name of what it wraps
        if status not in (None, 0):
            _STATE["error"] = SystemExit(status)
        previous_exit(status)
    exit.__doc__ = previous_exit.__doc__
    sys.exit = exit


def outcome() -> BaseException | None:
    """The exception that ends the process (an uncaught one, or ``SystemExit`` with a non-zero
    code), or ``None``."""
    err = _STATE["error"]
    if isinstance(err, SystemExit) and err.code in (None, 0):
        return None
    return err
