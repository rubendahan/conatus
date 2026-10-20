"""Studio S2: several runs on one curve (e.g. energy vs t for a fork and its parent).

Stdlib only; works on :class:`conatus.telemetry.RunState` objects (local server and hub alike).
Nothing is interpolated or re-sampled onto a common grid: each run keeps its own abscissae, so
two runs with different output times are shown as they are. Long series are strided (declared in
the answer, the last point always kept). Non-numeric values ("nan", "error: …") become null and
are counted, never dropped silently.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .launch import params_diff

__all__ = ["compare_runs", "MAX_RUNS", "X_AXES"]

MAX_RUNS = 8                     # the page's categorical palette has 8 slots (never cycled)
X_AXES = ("t", "iteration", "wall")


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v)


def _launch_params(st: Any) -> dict:
    """Declared parameters of a run: ``meta.launch.params`` (Studio launch) else ``meta``."""
    meta = ((st.header or {}).get("meta") or {}) if st is not None else {}
    launch = meta.get("launch") if isinstance(meta, dict) else None
    if isinstance(launch, dict) and isinstance(launch.get("params"), dict):
        return dict(launch["params"])
    return {k: v for k, v in meta.items() if k != "launch"} if isinstance(meta, dict) else {}


def compare_runs(runs: Mapping[str, Any], ids: list[str], key: str, *, x: str = "t",
                 max_points: int = 2000) -> dict:
    """Series of scalar ``key`` against ``x`` for each run of ``ids`` (first = reference for the
    parameter diff). Raises ValueError on an unknown run / axis or when no run has ``key``."""
    if not ids or len(ids) > MAX_RUNS or len(set(ids)) != len(ids):
        raise ValueError(f"give 1 to {MAX_RUNS} distinct runs")
    if x not in X_AXES:
        raise ValueError(f"x must be one of {X_AXES}")
    if not isinstance(key, str) or not key or key in ("v", "kind", "run_id", "relay"):
        raise ValueError(f"bad scalar name {key!r}")
    max_points = int(max(10, min(int(max_points), 20000)))
    missing_runs = [r for r in ids if r not in runs]
    if missing_runs:
        raise ValueError(f"unknown run(s) {missing_runs}")
    series, missing = {}, []
    ref = _launch_params(runs[ids[0]])
    for rid in ids:
        recs = [r for r in runs[rid].scalars if key in r]
        if not recs:
            missing.append(rid)
            continue
        stride = max(1, math.ceil(len(recs) / max_points))
        picked = recs[::stride]
        if picked[-1] is not recs[-1]:
            picked.append(recs[-1])                 # the latest value is always shown
        xs = [_num(r.get(x)) for r in picked]
        ys = [_num(r.get(key)) for r in picked]
        series[rid] = {"x": xs, "y": ys, "n": len(recs), "stride": stride,
                       "non_numeric": sum(1 for r in recs if _num(r.get(key)) is None),
                       "params": _launch_params(runs[rid]),
                       "diff_vs_first": params_diff(ref, _launch_params(runs[rid])) if rid != ids[0] else {}}
    if not series:
        raise ValueError(f"no run among {ids} has a scalar {key!r}")
    return {"key": key, "x": x, "runs": ids, "series": series, "missing": missing,
            "note": "each run keeps its own abscissae (no interpolation); strided to <= "
                    f"{max_points} points per run, last point kept"}
