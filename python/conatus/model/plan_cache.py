"""Reuse of assembled operators between two ``simulate()`` of the same model (P1).

Building a simulation is dominated by the assembly of the implicit operators: the sparse ``L`` and
``M`` of every Fourier group (``Plan.matrices``; about 1.3 s for Rayleigh-Benard at 64 x 32, more
than the compile, the factorisation or a few hundred steps). A parameter sweep or a notebook cell
run twice paid it each time. ``Model.simulate`` now keeps the assembled
group matrices of its last :data:`PLAN_CACHE_SIZE` builds and copies them into the next plan
when nothing that enters them changed.

What is reused, and what is not:

* reused: the full-space ``GroupMatrices`` (``L``, ``M``, row and column labels) of each group,
  copied (never shared: a solver may scale its own copy), and the NCC expansion reports;
* always rebuilt: the compiled IR problem (cheap, and it evaluates every data callable again),
  the plan object, the time-stepper factorisation (it depends on ``scheme`` and ``dt``), the
  explicit right-hand side, the state.

The key (:func:`plan_key`) is everything the operators depend on: the geometry object, the
unknowns, the equations, operators and conditions as written, the eigenvalue declaration, every
parameter value (bit-exact: ``float.hex``), the per-call overrides, the resolution, the
dealiasing, the NCC options, and a SHA-256 digest of every given-field / boundary data array as
evaluated by *this* compilation (so a callable whose closure changed is seen). Anything that
cannot be digested (a data function kept unevaluated, a non-Cartesian plan, a mapped domain)
disables the reuse for that build, and ``sim.explain()`` says why. ``CN_PLAN_CACHE=0`` disables it
everywhere. Nothing is ever reused silently: ``sim.explain()`` prints whether the operators were
built or reused, and from which key.

Backends and dtypes: what is kept is the host (SciPy) assembly of the plan, before any backend or
precision choice (``IVPSolver`` builds its own CPU or GPU system from copies), so one entry serves
CPU and GPU runs alike and is never a device array. A declared callable enters the key by identity
and the key holds a reference to it, so a collected function's ``id`` cannot be reused by another.
The store of a model is guarded by a lock (two threads may build simulations of one model).
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import threading
from collections import OrderedDict
from typing import Any

import numpy as np

__all__ = ["PLAN_CACHE_SIZE", "plan_key", "plan_for", "remember", "clear"]

PLAN_CACHE_SIZE = 2      # operator sets kept per model (LRU); each holds a copy of every L and M
_LOCK = threading.RLock()  # guards every model's store (OrderedDict moves are not thread-safe)


class _Uncacheable(Exception):
    """Raised by the key builder: this build is not reused (the message says why)."""


def _num(v: Any) -> Any:
    """Bit-exact, hashable form of a number (``float.hex``), or the value itself."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v).hex()
    if isinstance(v, (complex, np.complexfloating)):
        return (complex(v).real.hex(), complex(v).imag.hex())
    return None


def _digest(v: Any, what: str) -> Any:
    """Hashable digest of a declared value or of evaluated data."""
    n = _num(v)
    if n is not None or v is None or isinstance(v, bool):
        return ("n", n)
    if isinstance(v, str):
        return ("s", v)
    if isinstance(v, (tuple, list)):
        return ("t", tuple(_digest(x, what) for x in v))
    data = getattr(v, "data", None) if not isinstance(v, np.ndarray) else v
    if isinstance(data, np.ndarray) or isinstance(v, np.ndarray):
        a = np.ascontiguousarray(data if data is not None else v)
        if a.dtype == object:
            raise _Uncacheable(f"{what} holds Python objects")
        h = hashlib.sha256(a.view(np.uint8).reshape(-1).tobytes() if a.size else b"").hexdigest()
        return ("a", a.shape, a.dtype.str, getattr(v, "layout", None), h)
    if callable(v):                       # declared callables: identity; their data is digested
        try:                              # from the compiled NCC data (see plan_key). The object
            hash(v)                       # itself, not id(v): the key keeps it alive, so its id
        except TypeError:                 # cannot be recycled by another function
            raise _Uncacheable(f"{what} is an unhashable callable") from None
        return ("f", v)
    raise _Uncacheable(f"{what} of type {type(v).__name__} cannot be compared")


def plan_key(comp: Any, plan: Any, ncc_kw: dict) -> tuple:
    """Key of the operators of ``comp`` (a ``Compiled``) discretized as ``plan``: see the module
    docstring. The data of the given fields is digested as the plan evaluated it on its grids
    (``plan._ncc_data``: callables are called again by every build). Raises ``_Uncacheable`` with
    the reason when the build cannot be compared."""
    m = comp.model
    if getattr(m.geometry, "terrain", None) is not None:
        raise _Uncacheable("mapped domain (terrain)")
    ncc = []
    data = getattr(plan, "_ncc_data", None)
    if data is None:
        raise _Uncacheable("the plan keeps no evaluated given-field data")
    for f, d in sorted(data.items(), key=lambda kv: str(kv[0])):
        if callable(d):
            raise _Uncacheable(f"the data of {f} is a function evaluated lazily")
        ncc.append((str(f), _digest(d, f"data of {f}")))
    conds = tuple((c.boundary, c.text, c.unknown, _digest(c.value, f"value at {c.boundary}"),
                   _digest(c.into, "into")) for c in m.conditions)
    givens = tuple((g.name, getattr(g, "rank", 0), _digest(g.value, f"given {g.name}"))
                   for g in m.givens)
    return (id(m.geometry),
            tuple((u.name, u.rank, u.role) for u in m.unknowns),
            tuple(m.eqs), tuple(x[3] for x in m.macros), tuple(sorted(m.functions)),
            conds, givens, m.eig, repr(m.eig_growth),
            tuple(sorted((k, _num(v) if _num(v) is not None else repr(v))
                         for k, v in m.params.items())),
            tuple(sorted((k, _num(v) if _num(v) is not None else repr(v))
                         for k, v in comp.overrides.items())),
            tuple(sorted((str(k), repr(v)) for k, v in comp.N.items())),
            tuple(sorted((str(k), repr(v)) for k, v in comp.dealias.items())),
            tuple(sorted((k, repr(v)) for k, v in ncc_kw.items())),
            tuple(ncc))


def _store(model: Any) -> OrderedDict:
    with _LOCK:
        st = model.__dict__.get("_plan_cache")
        if st is None:
            st = OrderedDict()
            object.__setattr__(model, "_plan_cache", st)
        return st


def _copy(gm: Any) -> Any:
    return dataclasses.replace(gm, L=gm.L.copy(), M=gm.M.copy())


def plan_for(comp: Any, ncc_kw: dict) -> tuple[Any, str, Any]:
    """``(plan, note, key)``: a fresh plan of ``comp``, seeded with the operators of an earlier
    identical build when there is one. ``note`` is the ``explain()`` line; ``key`` is None when
    the build is not cacheable."""
    from ..discretize.plan import Plan
    plan = comp.discretize(**ncc_kw)
    if os.environ.get("CN_PLAN_CACHE", "1") == "0":
        return plan, "operators assembled (reuse disabled: CN_PLAN_CACHE=0)", None
    if type(plan) is not Plan:
        return plan, (f"operators assembled (not kept for reuse: {type(plan).__name__}; "
                      "Cartesian plans only)"), None
    try:
        key = plan_key(comp, plan, ncc_kw)
    except _Uncacheable as why:
        return plan, f"operators assembled (not kept for reuse: {why})", None
    with _LOCK:
        st = _store(comp.model)
        entry = st.get(key)
        if entry is None or entry["geometry"] is not comp.model.geometry:
            return plan, (f"operators assembled (kept for reuse by the next identical simulate(); "
                          f"cache of {PLAN_CACHE_SIZE} per model)"), key
        st.move_to_end(key)
        entry["hits"] += 1
    for k, gm in entry["groups"].items():  # the entry's matrices are never modified: copy outside
        plan._cache[k] = _copy(gm)
    plan.ncc_reports.update(entry["ncc_reports"])
    return plan, (f"operators reused from an earlier simulate() of this model ({len(entry['groups'])}"
                  f" groups, {entry['mib']:.1f} MiB; same equations, conditions, data, parameters, "
                  "N and dealiasing; the factorisation for this scheme and dt is new)"), key


def remember(comp: Any, plan: Any, key: Any) -> None:
    """Keep a copy of the assembled group matrices of ``plan`` under ``key`` (LRU)."""
    if key is None:
        return
    with _LOCK:
        st = _store(comp.model)
        if key in st:
            st.move_to_end(key)
            return
    groups = {k: _copy(v) for k, v in plan._cache.items()
              if isinstance(k, tuple) and len(k) == 2 and k[0] == "full"}
    if not groups:
        return
    mib = sum(g.L.data.nbytes + g.L.indices.nbytes + g.M.data.nbytes + g.M.indices.nbytes
              for g in groups.values()) / 2 ** 20
    with _LOCK:
        st[key] = {"geometry": comp.model.geometry, "groups": groups,
                   "ncc_reports": dict(plan.ncc_reports), "mib": mib, "hits": 0}
        while len(st) > PLAN_CACHE_SIZE:
            st.popitem(last=False)


def clear(model: Any) -> None:
    """Drop the operators kept by ``model`` (they are also dropped with the model)."""
    with _LOCK:
        _store(model).clear()
