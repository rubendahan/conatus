"""Snapshot matrices from the run registry (``conatus.runs``) or from arrays (data-driven).

``snapshots_from_run(run, fields=[...])`` reads the fields saved with ``run.save_field`` (layout
``data.h5: fields/<name>/<k>`` with attributes ``t`` and ``step``;
read-only) and stacks them component-major into X (n × m): column j is the concatenation of the
flattened fields at save index j. The times come from the ``t`` attributes; a missing ``t`` is
reported as NaN (never invented), and methods that need a uniform Δt check it through
:func:`uniform_dt`.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import OptionError
from .._repr import compact_repr

__all__ = ["Snapshots", "snapshots_from_run", "snapshots_from_arrays", "uniform_dt"]


@dataclass
class Snapshots:
    __repr__ = compact_repr   # one line, arrays by shape
    X: Any                          # n × m (component-major flattening)
    times: np.ndarray               # m (NaN where unknown)
    fields: list[str]
    shapes: dict                    # field -> shape of one snapshot
    source: str

    def unflatten(self, column) -> dict:
        """One column back to {field: array of its shape}."""
        out, i = {}, 0
        for f in self.fields:
            size = int(np.prod(self.shapes[f]))
            out[f] = column[i:i + size].reshape(self.shapes[f])
            i += size
        return out

    def explain(self) -> str:
        return (f"Snapshots from {self.source}: fields {self.fields} (shapes {self.shapes}), "
                f"X {tuple(self.X.shape)}, times {self.times[0]:.6g} … {self.times[-1]:.6g} "
                f"({int(np.isnan(self.times).sum())} unknown).")


def snapshots_from_arrays(arrays: dict, times: Sequence[float]) -> Snapshots:
    """``{field: array (m, *shape)}`` with the snapshot index first -> Snapshots."""
    names = list(arrays)
    t = np.asarray(times, float)
    known = t[np.isfinite(t)] if t.ndim == 1 else t
    # one time per snapshot, increasing where known (NaN = unknown, declared by uniform_dt);
    # audit inputs: decreasing times were kept and gave dt < 0 downstream
    if t.ndim != 1 or np.isinf(t).any() or np.any(np.diff(known) <= 0):
        raise OptionError(f"snapshots_from_arrays: times must be a 1D increasing sequence "
                          f"(NaN for unknown), got shape {t.shape}",
                          fix="times=[t0, t1, ...] in the order of the snapshots")
    m = len(times)
    for f in names:
        if arrays[f].shape[0] != m:
            raise OptionError(f"field {f}: {arrays[f].shape[0]} snapshots, {m} times")
    X = np.concatenate([np.asarray(arrays[f]).reshape(m, -1).T for f in names], axis=0)
    return Snapshots(X, np.asarray(times, float), names,
                     {f: tuple(arrays[f].shape[1:]) for f in names}, "arrays")


def snapshots_from_run(run, fields: Sequence[str], *, start: int = 0,
                       stop: int | None = None) -> Snapshots:
    """Saved fields ``fields`` of a registry run (a ``Run`` or an id), save indices [start, stop)."""
    import h5py

    from .. import runs as _runs
    if isinstance(run, str):
        run = _runs.get(run)
    path = run.path / "data.h5"
    if not path.exists():
        raise OptionError(f"run {run.id} has no data.h5 (no saved field, or still running)")
    arrays, times = {}, None
    with h5py.File(path, "r", locking=False) as h5:
        for f in fields:
            if f"fields/{f}" not in h5:
                raise OptionError(f"run {run.id} has no field {f!r}",
                                  fix=f"fields: {sorted(h5.get('fields', {}).keys())}")
            g = h5[f"fields/{f}"]
            keys = sorted(g.keys(), key=int)[start:stop]
            arrays[f] = np.stack([g[k][()] for k in keys])
            t = np.array([float(g[k].attrs["t"]) if "t" in g[k].attrs else np.nan for k in keys])
            if times is None:
                times = t
            elif not np.array_equal(times, t, equal_nan=True):
                raise OptionError(f"field {f!r} was saved at other times than {fields[0]!r}")
    s = snapshots_from_arrays(arrays, times)
    s.source = f"run {run.id} ({run.name})"
    return s


def uniform_dt(times: np.ndarray, *, rtol: float) -> float:
    """The common Δt of ``times``; refused if any step differs by more than ``rtol`` (declared)."""
    t = np.asarray(times, float)
    # audit inputs: a 2D array of times was flattened through np.diff, decreasing
    # times returned dt < 0 (frequencies of the wrong sign downstream), rtol < 0 refused nothing
    from ..errors import check_real
    rtol = check_real(rtol, "rtol", "uniform_dt()", nonnegative=True)
    if t.ndim != 1:
        raise OptionError(f"uniform_dt: times of shape {t.shape}, one time per snapshot (1D)")
    if np.isnan(t).any():
        raise OptionError("some snapshot times are unknown (NaN): declare dt explicitly")
    if t.size < 2:                  # np.diff would be empty: no step to measure
        raise OptionError(f"{t.size} snapshot time(s): a time step needs at least two snapshots",
                          fix="give two or more snapshots, or declare dt explicitly")
    d = np.diff(t)
    if not np.all(d > 0):
        raise OptionError("uniform_dt: the snapshot times must increase",
                          fix="snapshots in time order")
    dt = float(d.mean())
    if np.abs(d - dt).max() > rtol * abs(dt):
        raise OptionError(f"non-uniform sampling (steps {d.min():.6g} … {d.max():.6g}); DMD/SPOD "
                          "need a uniform Δt (optimized DMD accepts times)")
    return dt
