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
- **Space**: spectral convergence is geometric, $q(N) = q_\infty + C e^{-aN}$ at the resolutions
  actually given; the ratio $\rho$ of successive differences fixes $a$, and the estimate is
  $1.25\,|d_2|/(e^{a h_2}-1)$ with $h_2$ the last increment of $N$ (equal increments:
  $1.25\,|d_2|\,\rho/(1-\rho)$). Differences that do not decrease geometrically, and tuple
  resolutions refined in different proportions, are flagged.
- **Result**: the value at the finest resolution and time step (not the Richardson value), the
  sum of both estimates, and the interval. `rep.trusted` is false when a study is flagged, and also
  without `scheme_order`, because the asymptotic range is then unchecked. `rep.explain()` prints
  the numbers and assumptions behind the estimate.

The time steps must have a constant ratio and the resolutions must increase, the finest last;
anything else is refused.

## The mean of a statistically steady run

`cn.time_average(series, dt=..., discard=k)` gives the mean of a correlated time series after its
first `k` samples (the transient is yours to judge: `discard` has no default), with its standard
error from the integrated autocorrelation time. A batch-means estimate is printed beside it as a
check, and a series shorter than about 50 correlation times is flagged.

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
