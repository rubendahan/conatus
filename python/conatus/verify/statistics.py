"""Statistical error of a time average: mean of a correlated series with its
standard error and a 95 % interval, from four estimates of the variance of the mean.

::

    rep = cn.time_average(nusselt_series, dt=0.01, discard=200)
    rep.mean, rep.stderr, rep.interval, rep.tau_int, rep.explain()

Method (declared, no hidden choice): after discarding the first ``discard`` samples (a transient
is the user's call), the normalized autocorrelation rho(k) is computed by FFT. The integrated
autocorrelation time tau = 1/2 + sum_k rho(k) is the larger of two truncations of that sum:
Sokal's automatic window, the smallest W with W >= C tau(W), C = ``SOKAL_C`` = 5 (Sokal 1997,
"Monte Carlo methods in statistical mechanics", in Functional Integration, NATO ASI Series 361,
sect. 3), and the initial positive sequence, summing pairs rho(2j) + rho(2j + 1) while they are
positive (Geyer 1992, "Practical Markov chain Monte Carlo", Statistical Science 7, 473-483).
Sokal's window alone closed at a few lags on a slow component carrying 10-20 % of the variance
under fast noise and underestimated tau 5-fold (release 0.3.0).

The standard error is the largest of four estimates of the standard deviation of the mean, each
biased low in its own way: sqrt(2 tau var / n); ``BATCHES`` = 20 batch means; the batch means
extrapolated in the batch length, sqrt(max(2 v_10 - v_20, 0)) with v_B the batch-means variance
of B batches (their bias falls like 1 / batch length); and the mean of the 10 lowest nonzero
periodogram ordinates over n (the spectral density at zero frequency). The interval is mean +- 2
standard errors.

Flags (``trusted`` is False): a series shorter than ``MIN_TAU_MULTIPLE`` = 50 tau; a Sokal window
that never closes; 20 batch means giving a standard error above ``BATCH_RATIO_MAX`` = 2 times the
autocorrelation one. No other filter is applied: in the coverage experiment every filter computed
from the series itself (the length of the positive autocorrelation against the batch length, the
flatness of the low-frequency periodogram) lowered the coverage among the series it trusted,
because a slow component that stayed on one side of its mean also looks less correlated.

Coverage measured (2000 series per setting, true mean 0): AR(1) mixtures
sqrt(a) s + sqrt(1 - a) e (s AR(1) of parameter phi, e white) and other processes; the interval
contained 0 in 93.5-97.7 % of the trusted series in each of 11 settings, and every series of
those settings was trusted, including the 0.3.0 case phi = 0.98, a = 0.15, n = 4000 (64 % before).
Lowest: phi = 0.99, a = 0.1, n = 5000 (93.5 %), where the slow component is correlated over 100
samples and the run is 50 of those. The standard error is conservative where the estimates agree
(about 20 % above the spread of the means for a plain AR(1)); above nominal (100 % in a prototype
run): an oscillating AR(2) whose tau is below 1/2 (no credit is given below white noise).

What ``trusted`` does not establish: stationarity (a drift or a transient left in the series is
the user's to remove: ``discard``); that the run holds a correlation longer than itself (the
series cannot show it); coverage when the run is only a few tens of correlation lengths of a slow
component (93.5-94 % measured there). The interval is approximate, not a bound.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import InputTypeError

__all__ = ["time_average", "TimeAverageReport", "SOKAL_C", "BATCHES", "MIN_TAU_MULTIPLE",
           "BATCH_RATIO_MAX"]

SOKAL_C = 5.0
BATCHES = 20
MIN_TAU_MULTIPLE = 50.0
# batch-means standard error above this many autocorrelation standard errors: flagged (audit
# "assurances": the batch means were printed as "check" and never compared; white
# noise plus a slow sine gave a 3.1x larger batch error and a "trusted" mean 2.7 errors off).
# For a correctly estimated tau the ratio scatters by about 1/sqrt(2 (BATCHES - 1)) = 16 %.
BATCH_RATIO_MAX = 2.0
# lowest nonzero periodogram ordinates averaged for the spectral density at zero frequency
_SPECTRAL_BINS = 10
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
    acf_stderr: float = float("nan")         # sqrt(2 tau var / n)
    extrapolated_stderr: float = float("nan")  # batch means extrapolated in the batch length
    spectral_stderr: float = float("nan")    # low-frequency periodogram

    @property
    def interval(self) -> tuple[float, float]:
        """mean +- 2 standard errors (about 95 % for a Gaussian error)."""
        return (self.mean - 2 * self.stderr, self.mean + 2 * self.stderr)

    @property
    def n_eff(self) -> float:
        return self.n / (2 * self.tau_int)

    @property
    def trusted(self) -> bool:
        """No flag: run longer than 50 tau, Sokal window closed, batch means within 2x the
        autocorrelation error. Measured coverage of the
        interval among trusted series: 93.5-97.7 % (module docstring). It does not establish
        stationarity, nor anything about correlations longer than the run."""
        return not self.flags

    def explain(self) -> str:
        unit = "" if self.dt is None else f" = {self.tau_int * self.dt:g} time units"
        lines = [f"Time average of {self.name}: {self.mean:.10g} +- {self.stderr:.3e} (1 standard error)"
                 + ("" if self.trusted else "  [flagged: see below]"),
                 f"  {self.n} samples after discarding {self.discarded}; integrated autocorrelation "
                 f"time tau = {self.tau_int:.3g} samples{unit} (Sokal window W = {self.window}, "
                 f"C = {SOKAL_C:g}); {self.n_eff:.1f} effectively independent samples",
                 f"  standard error = the largest of: autocorrelation {self.acf_stderr:.3e}, "
                 f"{BATCHES} batch means {self.batch_stderr:.3e}, batch means extrapolated in the "
                 f"batch length {self.extrapolated_stderr:.3e}, low-frequency periodogram "
                 f"{self.spectral_stderr:.3e}; interval mean +- 2 standard errors",
                 "  not established: stationarity (the transient to discard is the user's choice), "
                 "correlations longer than the run"]
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
    # one real value per sample, as given (audit inputs: ravel() made a (samples,
    # quantities) array one interleaved series, a complex series lost its imaginary part with a
    # ComplexWarning, and the text "1234" was read as one sample 1234.0)
    raw = series if isinstance(series, np.ndarray) else (
        None if isinstance(series, (str, bytes)) or series is None else np.asarray(series))
    if raw is None or raw.dtype.kind not in "biuf" or raw.dtype.kind == "b" or raw.ndim != 1:
        what = ("text" if isinstance(series, (str, bytes)) else "None" if series is None else
                f"an array of dtype {raw.dtype} and shape {raw.shape}")
        raise InputTypeError(f"time_average: the series is {what}",
                             why="a series is one real number per sample (a 1D array); several "
                                 "quantities, complex values or text are refused",
                             fix="time_average(q, discard=...) with q of shape (n,): one call per "
                                 "quantity (q[:, j]), or the real part/modulus taken explicitly",
                             doc=_DOC)
    x = raw.astype(float)
    if dt is not None:
        from ..errors import check_real
        dt = check_real(dt, "dt", "time_average()", positive=True)
    if isinstance(discard, bool) or not isinstance(discard, (int, np.integer)) or discard < 0:
        raise InputTypeError(f"time_average(discard={discard!r}): a number of samples >= 0",
                             why="the length of the transient to discard is an explicit choice", doc=_DOC)
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
    tau_s, W = 0.5, None
    for w in range(1, n):
        tau_s += rho[w]
        if w >= SOKAL_C * tau_s:
            W = w
            break
    if W is None:
        W = n - 1
        flags.append("Sokal window never closed: the series is too short for its correlation time")
    # initial positive sequence (Geyer 1992): pairs of the autocorrelation summed while positive;
    # it does not stop at a fast decay while a slow component remains (hunt 5D, Q6)
    G, j = 0.0, 0
    while 2 * j + 1 < n:
        g = rho[2 * j] + rho[2 * j + 1]
        if g <= 0:
            break
        G += g
        j += 1
    tau = max(tau_s, G - 0.5, 0.5)          # anti-correlated series: no credit below white noise
    if n < MIN_TAU_MULTIPLE * tau:
        flags.append(f"series of {n} samples < {MIN_TAU_MULTIPLE:g} tau = {MIN_TAU_MULTIPLE * tau:.0f}: "
                     "the error estimate is unreliable (run longer)")
    m = n // BATCHES
    se_acf = math.sqrt(2 * tau * var / n)
    bse = float(x[: m * BATCHES].reshape(BATCHES, m).mean(axis=1).std(ddof=1) / math.sqrt(BATCHES))
    m2 = n // (BATCHES // 2)
    bse2 = float(x[: m2 * (BATCHES // 2)].reshape(BATCHES // 2, m2).mean(axis=1).std(ddof=1)
                 / math.sqrt(BATCHES // 2))
    ext = math.sqrt(max(2 * bse2 ** 2 - bse ** 2, 0.0))
    per = np.abs(np.fft.rfft(x - x.mean())[1:_SPECTRAL_BINS + 1]) ** 2 / n
    spec = math.sqrt(float(per.mean()) / n)
    se = max(se_acf, bse, ext, spec)
    if bse > BATCH_RATIO_MAX * se_acf:
        flags.append(f"the batch-means check gives a standard error {bse / se_acf:.2g} times the "
                     f"autocorrelation one (> BATCH_RATIO_MAX = {BATCH_RATIO_MAX:g}): a correlation "
                     "longer than the autocorrelation sum saw (slow oscillation or drift); run "
                     "longer, or discard the transient")
    return TimeAverageReport(float(x.mean()), se, float(tau), int(W), n, int(discard), dt, bse, flags, name,
                             acf_stderr=se_acf, extrapolated_stderr=ext, spectral_stderr=spec)
