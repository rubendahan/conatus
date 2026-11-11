"""F872 (hunt 5D on 0.3.0, Q6): time_average's trusted intervals cover the true mean.

Oracle: x = sqrt(a) s + sqrt(1 - a) e, s AR(1) of parameter phi = 0.98, a = 0.15, e white, true
mean 0, n = 4000 (tau_int = 7.85 in closed form). 0.3.0: 207 of 300 trusted, coverage 64 %.
Expected over 500 series: coverage of mean +- 2 se at least 92 % among the trusted ones (95 %
nominal, binomial standard deviation 1 %). Control: white noise, coverage 92-98 %.
"""
import math

import numpy as np
from _acceptance import check, passed, start
from scipy.signal import lfilter

start(872, "time_average interval coverage with a slow component")
import conatus as cn  # noqa: E402

rng = np.random.default_rng(872)


def coverage(make, trials=500):
    tr = cov = 0
    for _ in range(trials):
        r = cn.time_average(make(), discard=0)
        if r.trusted:
            tr += 1
            cov += r.interval[0] <= 0 <= r.interval[1]
    return tr, cov / max(tr, 1)


def mix(n=4000, phi=0.98, a=0.15):
    s = lfilter([1.0], [1.0, -phi], rng.normal(size=n + 30000) * math.sqrt(1 - phi ** 2))[30000:]
    return math.sqrt(a) * s + math.sqrt(1 - a) * rng.normal(size=n)


tr, cov = coverage(mix)
check(tr == 0 or cov >= 0.92, f"mixture: {tr}/500 trusted, coverage among trusted {cov:.3f}")
tw, cw = coverage(lambda: rng.normal(size=4000))
check(tw >= 450 and 0.92 <= cw <= 0.98, f"white noise control: {tw}/500 trusted, coverage {cw:.3f}")
passed(f"mixture {tr}/500 trusted, coverage {cov:.3f}; white {tw}/500, {cw:.3f}")
