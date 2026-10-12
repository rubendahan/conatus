"""cn.time_average: mean of a correlated series and its statistical error, against AR(1).

x_{n+1} = phi x_n + sqrt(1 - phi^2) e_n (unit variance) has rho(k) = phi^k and an integrated
autocorrelation time tau = (1 + phi) / (2 (1 - phi)) exactly (9.5 at phi = 0.9): the estimated
tau, the reported standard error against the spread of the means over 200 independent series,
and the coverage of the 2-sigma interval (true mean 0) are checked; then the flags.
"""
import numpy as np
import pytest

import conatus as cn
from conatus.errors import InputTypeError

PHI = 0.9
TAU = (1 + PHI) / (2 * (1 - PHI))


def ar1(n, rng, x0=None):
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = rng.standard_normal() if x0 is None else x0
    a = np.sqrt(1 - PHI ** 2)
    for i in range(1, n):
        x[i] = PHI * x[i - 1] + a * e[i]
    return x


def test_ar1_error_and_coverage():
    rng = np.random.default_rng(7)
    reps = [cn.time_average(ar1(20000, rng), dt=0.1, discard=0) for _ in range(200)]
    means = np.array([r.mean for r in reps])
    se = np.array([r.stderr for r in reps])
    taus = np.array([r.tau_int for r in reps])
    assert abs(np.mean(taus) / TAU - 1) < 0.1
    assert abs(np.mean(se) / np.std(means) - 1) < 0.15
    cover = np.mean([lo <= 0 <= hi for lo, hi in (r.interval for r in reps)])
    assert cover > 0.9
    r = reps[0]
    assert r.trusted and abs(r.batch_stderr / r.stderr - 1) < 0.5
    assert "integrated autocorrelation" in r.explain() and "time units" in r.explain()


def test_flags_and_refusals():
    rng = np.random.default_rng(1)
    short = cn.time_average(ar1(200, rng), discard=0)
    assert not short.trusted and any("unreliable" in f for f in short.flags)
    with pytest.raises(InputTypeError, match="discard"):
        cn.time_average(ar1(1000, rng), discard=-1)
    with pytest.raises(InputTypeError, match="samples after discard"):
        cn.time_average(ar1(100, rng), discard=50)
    c = cn.time_average(np.ones(500), discard=0)
    assert c.stderr == 0 and "constant" in c.flags[0]
