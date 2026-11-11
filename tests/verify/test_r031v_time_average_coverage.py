"""Coverage of cn.time_average's interval among trusted series (hunt 5D on 0.3.0, Q6).

Oracle: x = sqrt(a) s + sqrt(1 - a) e with s an AR(1) process of parameter phi (unit variance) and
e white noise; the true mean is 0 and tau_int = 1/2 + a phi / (1 - phi) in closed form. 0.3.0
trusted 207 of 300 series of (phi, a, n) = (0.98, 0.15, 4000) whose mean +- 2 se covered 0 only
64 % of the time (Sokal's window closed at tau = 1.6 for a true 7.8).

The slow test is the coverage experiment of the module docstring: 1000 series per setting, the
interval must contain 0 in 92-98 % of the trusted ones (95 % nominal; a binomial standard
deviation is 0.7 % at 1000 trials), and the 0.3.0 failure setting must be trusted in most runs.
"""
import math

import numpy as np
import pytest
from scipy.signal import lfilter

import conatus as cn


def _series(rng, kind, n):
    if kind[0] == "mix":
        _, phi, a = kind
        s = lfilter([1.0], [1.0, -phi], rng.normal(size=n + 30000) * math.sqrt(1 - phi ** 2))[30000:]
        return math.sqrt(a) * s + math.sqrt(1 - a) * rng.normal(size=n)
    if kind[0] == "white":
        return rng.normal(size=n)
    if kind[0] == "t3":
        return lfilter([1.0], [1.0, -kind[1]], rng.standard_t(3, size=n + 30000))[30000:]
    raise ValueError(kind)


def _coverage(kind, n, trials, seed):
    rng = np.random.default_rng(seed)
    tr = cov = 0
    for _ in range(trials):
        r = cn.time_average(_series(rng, kind, n), discard=0)
        if r.trusted:
            tr += 1
            lo, hi = r.interval
            cov += lo <= 0 <= hi
    return tr, cov / max(tr, 1)


def test_hunt_case_is_covered():
    """The 0.3.0 failure: (phi, a, n) = (0.98, 0.15, 4000), 300 series."""
    tr, cov = _coverage(("mix", 0.98, 0.15), 4000, 300, 1)
    assert tr >= 240 and cov >= 0.9, (tr, cov)


def test_tau_sees_a_slow_component():
    """tau of the hunt case: closed form 7.85; Sokal's window alone gave 1.6."""
    rng = np.random.default_rng(2)
    taus = [cn.time_average(_series(rng, ("mix", 0.98, 0.15), 40000), discard=0).tau_int for _ in range(20)]
    assert 0.8 * 7.85 < np.median(taus) < 1.2 * 7.85


SETTINGS = [(("white",), 4000), (("mix", 0.9, 1.0), 20000), (("mix", 0.9, 1.0), 2000),
            (("mix", 0.5, 1.0), 1000), (("mix", 0.99, 0.1), 20000), (("mix", 0.99, 0.1), 5000),
            (("mix", 0.995, 0.05), 20000), (("mix", 0.999, 0.1), 50000), (("mix", 0.98, 0.15), 4000),
            (("mix", 0.98, 0.15), 40000), (("t3", 0.9), 20000)]


@pytest.mark.slow
@pytest.mark.parametrize("i", range(len(SETTINGS)))
def test_coverage_1000_series_per_setting(i):
    kind, n = SETTINGS[i]
    tr, cov = _coverage(kind, n, 1000, 100 + i)
    print(f"{kind} n={n}: trusted {tr}/1000, coverage among trusted {cov:.3f}")
    assert tr >= 500 and 0.92 <= cov <= 0.98, (tr, cov)
