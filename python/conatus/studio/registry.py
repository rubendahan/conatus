"""Studio view of the run registry (``conatus.runs``): list / filter, one record, compare two runs.

Read-only (the Studio never writes a registry record) and stdlib + ``conatus.runs`` only, so the
local server and the hub can both serve it. Every function returns JSON-safe dicts and raises
``ValueError`` (``OptionError``) on a bad query, which the handlers turn into HTTP 400.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..runs import Registry, Run, compare, find, get

__all__ = ["list_runs", "run_detail", "compare_pair", "FILTER_ARGS", "MAX_LIST"]

MAX_LIST = 500                                     # rows sent to the page at most (declared)
FILTER_ARGS = ("status", "state", "tag", "name", "problem", "telemetry_id", "limit")


def summary(r: Run) -> dict:
    """One table row of a run: identity, state, status, lineage, host, parameters, results."""
    rec = r.rec
    return {"id": r.id, "name": r.name, "state": r.state, "status": r.status, "tags": r.tags,
            "created_utc": rec.get("created_utc"), "duration_s": rec.get("duration_s"),
            "problem": (rec.get("problem") or {}).get("name"), "parent": r.parent,
            "relation": rec.get("relation"), "reused_from": rec.get("reused_from"),
            "telemetry_id": rec.get("telemetry_id"), "host": (rec.get("hardware") or {}).get("host"),
            "gpu": (rec.get("hardware") or {}).get("gpu"),
            "slurm_run_id": (rec.get("slurm") or {}).get("run_id"),
            "params": r.params, "results": r.results}


def list_runs(registry: Registry, args: Mapping[str, str]) -> dict:
    """Runs of the registry filtered by the query arguments in FILTER_ARGS (all optional)."""
    bad = [k for k in args if k not in FILTER_ARGS]
    if bad:
        raise ValueError(f"unknown filter(s) {bad}; known: {FILTER_ARGS}")
    limit = min(int(args.get("limit") or MAX_LIST), MAX_LIST)
    kw: dict[str, Any] = {"limit": limit, "root": registry}
    for k in ("status", "state", "name", "problem"):
        if args.get(k):
            kw[k] = args[k]
    if args.get("tag"):
        kw["tags"] = [t for t in args["tag"].split(",") if t]
    if args.get("telemetry_id"):
        kw["telemetry_id"] = args["telemetry_id"]
    runs = find(**kw)
    return {"root": str(registry.root), "limit": limit, "runs": [summary(r) for r in runs]}


def run_detail(registry: Registry, run_id: str) -> dict:
    """One run: summary, ``explain()``, lineage and the record (without the long numerics)."""
    r = get(run_id, root=registry)
    rec = dict(r.rec)
    rec.pop("numerics", None)                        # large; shown by explain() in short form
    return {"summary": summary(r), "explain": r.explain(), "lineage": r.lineage().explain(),
            "record": rec}


def compare_pair(registry: Registry, a: str, b: str) -> dict:
    """``conatus.runs.compare(a, b)`` as JSON, with its ``explain()`` text."""
    c = compare(a, b, root=registry)
    return {"explain": c.explain(), **c.to_dict()}
