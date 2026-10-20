"""Fate of every member of an ensemble: basins of attraction, transition probabilities.

``ensemble_fate`` advances an :class:`Ensemble` and asks a user classifier every ``check_every``
steps which members have *decided* (label != 0: e.g. +1 turbulent, -1 laminar, or the index of an
attractor). A member's label is latched at its first decisive check at or after ``min_steps`` (no
verdict before: initial transients can cross thresholds, see ``ecs.edge``); members still
undecided after ``max_steps`` keep the label 0 and are **reported as undecided**, never assigned.
Everything the verdict depends on (classifier thresholds, check interval, min/max horizon) is an
explicit argument and is written into the result.

``proportion_ci`` gives the Wilson score interval of a proportion (transition probability from k
turbulent members out of m), the interval used in the reported tables.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..errors import OptionError
from .._repr import compact_repr

__all__ = ["ensemble_fate", "EnsembleFate", "proportion_ci"]


def proportion_ci(k: int, m: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    """(p, low, high): proportion k/m with its Wilson score interval (default 95 %)."""
    # counts 0 <= k <= m, m > 0, and z > 0 (audit inputs: k > m gave NaN bounds
    # clipped to [0, 1], k = 1.5 and z < 0 an interval without a word)
    from ..errors import check_count, check_real
    m = check_count(m, "m", "proportion_ci()")
    k = check_count(k, "k", "proportion_ci()", minimum=0)
    z = check_real(z, "z", "proportion_ci()", positive=True)
    if k > m:
        raise OptionError(f"proportion_ci(k={k}, m={m}): k successes out of m trials, k <= m")
    p = k / m
    den = 1 + z * z / m
    c = (p + z * z / (2 * m)) / den
    h = z * np.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


@dataclass
class EnsembleFate:
    __repr__ = compact_repr   # one line, arrays by shape
    labels: np.ndarray            # (B,) int, 0 = undecided
    decision_time: np.ndarray     # (B,) time of the latched verdict (nan if undecided)
    settings: dict
    history: list = field(default_factory=list)   # optional per-check records (callback output)

    def counts(self) -> dict:
        vals, cnt = np.unique(self.labels, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, cnt, strict=True)}

    def explain(self) -> str:
        s = self.settings
        c = self.counts()
        return (f"EnsembleFate: {self.labels.size} member(s); checks every {s['check_every']} steps "
                f"(dt = {s['dt']:g}), no verdict before step {s['min_steps']}, horizon "
                f"{s['max_steps']} steps; classifier: {s['classifier']}\n"
                f"  outcomes {c} (0 = undecided, reported, never assigned)")


def ensemble_fate(ens, classify: Callable, *, check_every: int, min_steps: int, max_steps: int,
                  classifier: str, record: Callable | None = None, log=None) -> EnsembleFate:
    """Run ``ens`` until every member is decided or ``max_steps``; see the module docstring.
    ``classify(ens) -> int array (B,)`` (0 = undecided); ``classifier``: its description (thresholds),
    stored in the result; ``record(ens) -> dict`` optional per-check record."""
    from ..errors import check_count
    # counts are integers, never truncated (finding 66)
    check_every = check_count(check_every, "check_every", "ensemble_fate")
    max_steps = check_count(max_steps, "max_steps", "ensemble_fate")
    min_steps = check_count(min_steps, "min_steps", "ensemble_fate", minimum=0)
    if max_steps < check_every:
        raise OptionError("ensemble_fate: need max_steps >= check_every",
                          fix=f"max_steps={check_every} or more")
    B = ens.B
    labels = np.zeros(B, int)
    tdec = np.full(B, np.nan)
    hist = []
    steps = 0
    while steps < max_steps:
        # the last chunk stops at the horizon (finding 67: check_every=4,
        # max_steps=10 ran to step 12); the final check happens at max_steps exactly
        chunk = min(check_every, max_steps - steps)
        ens.run(chunk)
        steps += chunk
        raw = np.asarray(classify(ens))
        if raw.shape != (B,):
            raise OptionError(f"classify must return {B} labels, got shape {raw.shape}")
        lab = raw.astype(int)
        if not np.array_equal(lab, raw):
            raise OptionError("classify must return integer labels (0 = undecided); got "
                              f"non-integer values {raw[lab != raw][:3]!r}",
                              why="a label is never truncated (finding 66)")
        if record is not None:
            hist.append(record(ens))
        if steps >= min_steps:
            new = (labels == 0) & (lab != 0)
            labels[new] = lab[new]
            tdec[new] = ens.t
        if log is not None:
            log(f"step {steps}: t = {ens.t:g}, decided {int((labels != 0).sum())}/{B}")
        if np.all(labels != 0):
            break
    settings = dict(check_every=check_every, min_steps=min_steps, max_steps=max_steps,
                    dt=ens.dt, classifier=classifier, steps_run=steps)
    return EnsembleFate(labels, tdec, settings, hist)
