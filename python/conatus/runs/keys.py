"""Cache keys of the registry: what makes two computations "identical".

A key is the sha256 of the canonical encoding (``_util.canonical``: exact floats, arrays by their
bytes) of four parts, all shown in the cache report:

* ``problem``: structural hash of the IR problem (``conatus.provenance.problem_hash``), or the
  qualified name + source hash of a decorated function;
* ``params``: the parameter values (plan parameters, or the function's bound arguments);
* ``numerics``: the numerical configuration read from the objects (bases, sizes, dealias, scheme,
  dt, backend, ordering, reduced space...), minus the clock (``t``, ``iteration``), plus the input
  (the exact start state: X and multistep history, t, iteration; the number of steps);
* ``code``: the executed code (sha256 of the imported ``conatus`` tree and of the compiled core
  module) and the versions of Python, NumPy, SciPy and the machine architecture.

The git commit is *not* in the key: two commits with the same executed tree compute the same
numbers (documentation-only commits keep the cache valid); a dirty tree changes the tree hash.
"""
from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ._util import jsonable, sha256_of

_CORE_SHA: dict[tuple, str] = {}


def _core_sha(module_path: str | None) -> str | None:
    """sha256 of the compiled core module file (memoized on path, size and mtime)."""
    if not module_path:
        return None
    p = Path(module_path)
    try:
        st = p.stat()
    except OSError:
        return None
    k = (str(p), st.st_size, st.st_mtime_ns)
    if k not in _CORE_SHA:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        _CORE_SHA[k] = h.hexdigest()
    return _CORE_SHA[k]


def code_fingerprint(code: dict | None = None) -> dict:
    """The part of the code state that decides the numbers (see module docstring)."""
    import scipy
    if code is None:
        from .. import provenance
        code = provenance.code_state()
    core = code.get("core") or {}
    return {"python_tree_sha256": code.get("python_tree_sha256"),
            "core_sha256": _core_sha(core.get("module")),
            "core_stale": core.get("stale_vs_sources"),
            "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "machine": platform.machine()}


def _clockless(numerics: dict) -> dict:
    """numerics without the solver clock (t, iteration): those belong to the start state."""
    out = dict(numerics)
    ts = dict(out.get("time_stepping") or {})
    for k in ("t", "iteration"):
        ts.pop(k, None)
    if ts:
        out["time_stepping"] = ts
    return out


def state_digest(solver: Any) -> str:
    """sha256 of an IVPSolver's exact state: X, multistep history, clock and step origins."""
    def host(a):
        return a.get() if type(a).__module__.split(".")[0] == "cupy" else np.asarray(a)
    h = hashlib.sha256()
    for a in [solver.X, *solver._MXh, *solver._Fh]:
        a = np.ascontiguousarray(host(a))
        h.update(f"{a.shape}{a.dtype}".encode())
        h.update(a.tobytes())
    h.update(repr((float(solver.t).hex(), int(solver.iteration), float(solver._t0).hex(),
                   int(solver._it0), len(solver._MXh))).encode())
    return h.hexdigest()


def ivp_parts(solver: Any, nsteps: int) -> dict:
    """Key parts of ``solver.run(nsteps)`` from the solver's current state (code part added by
    :func:`finish_key`)."""
    from .. import provenance
    from .run import problem_info
    num = jsonable(provenance.numerics(solver))
    return {"kind": "ivp",
            "problem": (problem_info(solver) or {}).get("hash"),
            "params": {"problem": (num.get("problem") or {}).get("parameters"),
                       "discretize": (num.get("discretization") or {}).get("params")},
            "numerics": _clockless(num),
            "input": {"start_state_sha256": state_digest(solver), "nsteps": int(nsteps)}}


def exact_hash(v: Any) -> bool:
    """True if ``v`` is hashed by its content (numbers, strings, arrays, and containers of them).
    An object hashed by its ``repr`` could collide with a different object: never a cache key."""
    if v is None or isinstance(v, (bool, int, float, complex, str, np.number, np.bool_, np.ndarray)):
        return not (isinstance(v, np.ndarray) and v.dtype == object)
    if isinstance(v, dict):
        return all(isinstance(k, str) and exact_hash(x) for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return all(exact_hash(x) for x in v)
    return False


def function_parts(target: str, source_sha: str | None, args: dict) -> dict:
    """Key parts of a call of a decorated function (arguments hashed by content)."""
    return {"kind": "function", "problem": {"target": target, "source_sha256": source_sha},
            "params": {k: sha256_of(v) for k, v in args.items()}, "numerics": None}


def finish_key(parts: dict, fingerprint: dict) -> tuple[str, dict]:
    """(key, full parts) once the code fingerprint is known."""
    full = dict(parts, code=fingerprint)
    return sha256_of(full), full


def describe_parts(parts: dict) -> dict:
    """Short, readable form of the parts (hashes truncated) for reports."""
    def short(v):
        if isinstance(v, str) and len(v) == 64:
            return v[:12]
        if isinstance(v, dict):
            return {k: short(x) for k, x in v.items()}
        return v
    return {k: short(v) for k, v in parts.items() if k != "numerics"} | {
        "numerics": "sha256 " + sha256_of(parts.get("numerics"))[:12]}
