# Numerical uncertainty

A number computed by a simulation (a growth rate, a mean, a value at a point) depends on the
resolution and on the time step. `cn.uncertainty` reports it with an error estimate. Supply three
resolutions and three time steps to estimate the spatial and temporal errors.

## A value with its error

```{literalinclude} _check_features.py
:language: python
:start-after: "# [uncertainty]"
:end-before: "# [/uncertainty]"
:dedent: 4
```

```{literalinclude} outputs/uncertainty_report.txt
:language: text
```

The check script compares the result with the exact value of this problem (a series of Bessel
functions): it lies inside the reported interval.

## What is computed

`quantity(N=..., dt=...)` is run five times: the three time steps at the finest resolution, and the
three resolutions at the finest time step (the finest run is shared).

- **Time**: the observed order of convergence $p$ from the three values; the Richardson value is
  shown beside the result, and the error estimate is $1.25\,|q_R - q_3|$, with the safety factor of
  Roache's grid-convergence index (Roache 1994). With `scheme_order`, an observed order more than
  0.3 away from it is flagged: the time steps are not yet in the asymptotic range.
- **Space**: two laws are fitted to each three consecutive resolutions, at the resolutions
  actually given: geometric (spectral), $q(N) = q_\infty + C e^{-aN}$, and algebraic,
  $q(N) = q_\infty + C N^{-p}$ (a corner, a discontinuity, a kink in a coefficient). The geometric
  estimate is $1.25\,|d|/(e^{a h}-1)$ with $d$ and $h$ the last difference and increment of $N$; the
  algebraic one is $1.25\,|d|/((N_m/N_{m-1})^p-1)$. Three resolutions fit either law, so the larger
  estimate is given and the study is flagged. With four or more, a law whose rate ($a$ or $p$)
  falls by more than 10 % from one triple to the next is rejected; the geometric estimate is used
  when that law alone holds, the larger one otherwise, and the study is flagged when neither
  holds. Also flagged: differences that change sign or do not decrease, tuple resolutions refined
  in different proportions, and a refinement that leaves the value unchanged next to one that
  changes it. For an even solution an odd $N$ adds a zero coefficient, so `N=[23, 24, 25]` gives
  a last difference of exactly zero; take resolutions of one parity.
- **Result**: the value at the finest resolution and time step (not the Richardson value), the
  sum of both estimates, and the interval. `rep.trusted` is false when a study is flagged, and also
  without `scheme_order`, because the asymptotic range is then unchecked, and with three
  resolutions, because the law in $N$ is then untested. `rep.explain()` prints the numbers and
  assumptions behind the estimate.

`trusted` does not establish that the behaviour seen continues past the finest $N$ and $dt$ (two
terms of different decay can trade places later), that the spatial and temporal errors add
without a cross term, or anything about errors the studies do not vary (solver tolerances, a
statistical error). The interval is an estimate with a safety factor, not a bound. In a sweep of
4000 random two-term closed forms with four or five resolutions, 98.3 % of the trusted intervals
contained the limit; the misses were decays whose slower term takes over past the finest $N$.

The time steps must have a constant ratio and the resolutions must increase, the finest last;
anything else is refused.

## The mean of a statistically steady run

`cn.time_average(series, dt=..., discard=k)` gives the mean of a correlated time series after its
first `k` samples (the transient is yours to judge: `discard` has no default), with a standard
error and the interval mean ± 2 standard errors. The integrated autocorrelation time is the
larger of Sokal's window sum and Geyer's initial positive sequence (Geyer 1992), which does not stop
at a fast decay while a slow component remains. The standard error is the largest of four
estimates, each biased low in its own way: the autocorrelation one, 20 batch means, the batch
means extrapolated in the batch length, and the low-frequency periodogram. A series shorter than
about 50 correlation times, or whose autocorrelation stays positive over more than one batch
(n/20 samples), is flagged.

What `trusted` means, measured: over 2000 series per setting of 11 test processes (white noise,
AR(1), a slow AR(1) component carrying 5 to 15 % of the variance under white noise, heavy-tailed
noise), every series was trusted and the interval contained the true mean in 93.5 to 97.7 % of
them (95 % nominal). The lowest values come from runs only about 50 correlation lengths of the slow
component long. It does not establish stationarity: a drift or a transient left in the series is
yours to remove.

```python
import numpy as np
import conatus as cn

rng = np.random.default_rng(0)                  # stands for the Nusselt number of a run,
nusselt = np.empty(20000)                       # one value every 0.01: a correlated series
nusselt[0] = 6.0                                # with a transient
for i in range(1, nusselt.size):
    nusselt[i] = 4.9 + 0.98 * (nusselt[i - 1] - 4.9) + 0.02 * rng.standard_normal()
rep = cn.time_average(nusselt, dt=0.01, discard=2000)
print(rep.explain())        # mean, standard error, correlation time, batch-means check
```
