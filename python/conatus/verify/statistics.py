"""Statistical error of a time average (spec §83): mean of a correlated series with its
standard error, from the integrated autocorrelation time.

::

    rep = cn.time_average(nusselt_series, dt=0.01, discard=200)
    rep.mean, rep.stderr, rep.tau_int, rep.explain()

Method (declared, no hidden choice): after discarding the first ``discard`` samples (a transient
is the user's call), the normalized autocorrelation rho(k) is computed by FFT; the integrated
autocorrelation time tau = 1/2 + sum_{k=1}^{W} rho(k) uses Sokal's automatic window, the
smallest W with W >= C tau(W), C = ``SOKAL_C`` = 5 (Sokal 1997, "Monte Carlo methods in
statistical mechanics", in Functional Integration, NATO ASI Series 361, sect. 3). The standard
error of the mean is sqrt(2 tau var / n) (n samples, var the sample variance), i.e. n / (2 tau)
effectively independent samples. A batch-means estimate (``BATCHES`` = 20 batches) is reported
beside it as a check; a series shorter than ``MIN_TAU_MULTIPLE`` = 50 tau is flagged (the error
estimate itself is then unreliable), as is a window that never closes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import InputTypeError

__all__ = ["time_average", "TimeAverageReport", "SOKAL_C", "BATCHES", "MIN_TAU_MULTIPLE"]

SOKAL_C = 5.0
BATCHES = 20
MIN_TAU_MULTIPLE = 50.0
_DOC = "docs/site/guide/uncertainty.md"


@dataclass
class TimeAverageReport:
    """Mean of a series, its standard error and how it was obtained."""
    mean: float
    stderr: float
    tau_int: float                 # in samples
    window: int
    n: int
    discarded: int
    dt: float | None
    batch_stderr: float
    flags: list[str] = field(default_factory=list)
    name: str = "q"

    @property
    def interval(self) -> tuple[float, float]:
        """mean +- 2 standard errors (about 95 % for a Gaussian error)."""
        return (self.mean - 2 * self.stderr, self.mean + 2 * self.stderr)

    @property
    def n_eff(self) -> float:
        return self.n / (2 * self.tau_int)

    @property
    def trusted(self) -> bool:
        return not self.flags

    def explain(self) -> str:
        unit = "" if self.dt is None else f" = {self.tau_int * self.dt:g} time units"
        lines = [f"Time average of {self.name}: {self.mean:.10g} +- {self.stderr:.3e} (1 standard error)"
                 + ("" if self.trusted else "  [flagged: see below]"),
                 f"  {self.n} samples after discarding {self.discarded}; integrated autocorrelation "
                 f"time tau = {self.tau_int:.3g} samples{unit} (Sokal window W = {self.window}, "
                 f"C = {SOKAL_C:g}); {self.n_eff:.1f} effectively independent samples",
                 f"  check: {BATCHES} batch means give a standard error {self.batch_stderr:.3e}",
                 "  error = sqrt(2 tau var / n); the transient to discard is the user's choice"]
        lines += [f"  ** {f} **" for f in self.flags]
        return "\n".join(lines)

    __str__ = explain


def _autocorrelation(x: np.ndarray) -> np.ndarray:
    n = x.size
    f = np.fft.rfft(x - x.mean(), 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    return acf / acf[0]


def time_average(series: Any, *, dt: float | None = None, discard: int, name: str = "q") -> TimeAverageReport:
    """Mean and standard error of a stationary, correlated series (one value per sample;
    ``dt``, optional, only converts tau to time units). ``discard``: number of initial samples
    dropped (a transient), required: 0 is a choice too."""
    x = np.asarray(series, dtype=float).ravel()
    if isinstance(discard, bool) or not isinstance(discard, (int, np.integer)) or discard < 0:
        raise InputTypeError(f"time_average(discard={discard!r}): a number of samples >= 0",
                             why="the transient is not detected for you", doc=_DOC)
    if not np.all(np.isfinite(x)):
        raise InputTypeError("time_average: non-finite values in the series", doc=_DOC)
    x = x[int(discard):]
    n = x.size
    if n < 4 * BATCHES:
        raise InputTypeError(f"time_average: {n} samples after discard (need >= {4 * BATCHES})", doc=_DOC)
    flags: list[str] = []
    var = float(x.var(ddof=1))
    if var == 0:
        return TimeAverageReport(float(x.mean()), 0.0, 0.5, 0, n, int(discard), dt, 0.0,
                                 ["constant series: no fluctuation, error 0"], name)
    rho = _autocorrelation(x)
    tau, W = 0.5, None
    for w in range(1, n):
        tau += rho[w]
        if w >= SOKAL_C * tau:
            W = w
            break
    if W is None:
        W = n - 1
        flags.append("Sokal window never closed: the series is too short for its correlation time")
    tau = max(tau, 0.5)                       # anti-correlated series: no credit below white noise
    if n < MIN_TAU_MULTIPLE * tau:
        flags.append(f"series of {n} samples < {MIN_TAU_MULTIPLE:g} tau = {MIN_TAU_MULTIPLE * tau:.0f}: "
                     "the error estimate is unreliable (run longer)")
    se = math.sqrt(2 * tau * var / n)
    m = n // BATCHES
    bm = x[: m * BATCHES].reshape(BATCHES, m).mean(axis=1)
    bse = float(bm.std(ddof=1) / math.sqrt(BATCHES))
    return TimeAverageReport(float(x.mean()), se, float(tau), int(W), n, int(discard), dt, bse, flags, name)
