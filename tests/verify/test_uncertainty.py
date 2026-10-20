"""cn.uncertainty: value + error estimate from declared N and dt studies, against an exact answer.

Heat equation u_t = nu u_xx on a 2 pi-periodic line, u(x, 0) = exp(cos x): the exact value at
x = 0, t = T is I_0(1) + 2 sum_k I_k(1) exp(-nu k^2 T) (modified Bessel functions). RK222 is of
order 2: the observed temporal order must be 2 +- 0.3, the interval must contain the exact value
(with Roache's safety factor 1.25: the bare Richardson difference was 0.6 % short of the true
error here) and the error estimate must be within a factor 3 of the true error. Then the flags: a wrong formal order, non-geometric spatial differences, a non-constant
dt ratio.
"""
import numpy as np
import pytest
from scipy.special import iv

import conatus as cn
from conatus.errors import InputTypeError, OptionError

NU, T = 0.5, 0.4


def exact():
    k = np.arange(1, 40)
    return float(iv(0, 1) + 2 * np.sum(iv(k, 1) * np.exp(-NU * k ** 2 * T)))


def u_at_0(N, dt):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="heat", nu=NU)
    m.scalar("u")
    m.equations("dt(u) - nu*dx(dx(u)) = 0")
    sim = m.simulate(N=N, scheme="RK222", dt=dt)
    sim.initial(u=lambda x: np.exp(np.cos(x)))
    sim.run(until=T)
    return float(sim.grid("u")[0])                      # x = 0 is the first grid point


def test_interval_contains_the_exact_value():
    rep = cn.uncertainty(u_at_0, N=[8, 12, 16], dt=[0.04, 0.02, 0.01], scheme_order=2, name="u(0, T)")
    ex = exact()
    true_err = abs(rep.value - ex)
    lo, hi = rep.interval
    print(rep.explain(), "\nexact", ex, "true error", true_err)
    assert lo <= ex <= hi
    assert abs(rep.temporal.order - 2) < 0.3 and rep.trusted
    assert true_err / 3 < rep.error < 3 * true_err
    # paper: the bare Richardson difference (error / F_s) falls short of the true error (by 0.6 %
    # here, measured), the safety factor covers it
    assert rep.error / cn.verify.uncertainty.SAFETY < true_err < rep.error
    assert abs(rep.temporal.extrapolated - ex) < 0.05 * true_err      # Richardson removes most of it
    txt = rep.explain()
    for s in ("temporal study", "spatial study", "observed order", "Richardson value", "assumptions"):
        assert s in txt


def test_flags_never_hidden():
    rep = cn.uncertainty(u_at_0, N=[8, 12, 16], dt=[0.04, 0.02, 0.01], scheme_order=3)
    assert not rep.trusted and any("formal order 3" in f for f in rep.temporal.flags)
    vals = iter([1.0, 1.1, 0.9, 1.0, 1.2])           # q(N3,dt3), then dt study, then N study
    rep2 = cn.uncertainty(lambda N, dt: next(vals), N=[1, 2, 3], dt=[4, 2, 1])
    assert any("non-monotone" in f for f in rep2.temporal.flags)
    assert any("do not decrease" in f for f in rep2.spatial.flags)
    assert "flagged" in rep2.explain()


def test_refusals():
    with pytest.raises(OptionError, match="ratio"):
        cn.uncertainty(u_at_0, N=[8, 12, 16], dt=[0.04, 0.02, 0.005])
    with pytest.raises(InputTypeError, match="three"):
        cn.uncertainty(u_at_0, N=[8, 16], dt=[0.04, 0.02, 0.01])
    with pytest.raises(InputTypeError, match="not a number"):
        cn.uncertainty(lambda N, dt: "x", N=[1, 2, 3], dt=[4, 2, 1])


def test_resolutions_in_the_wrong_order_refused():
    """The value is the one at the LAST N; N=[64, 48, 32] used
    to report the coarsest run as the value and its spatial study backwards, silently."""
    calls = []
    q = lambda N, dt: calls.append(N) or 1.0                    # noqa: E731
    for bad in ([16, 12, 8], [8, 8, 16], [(8, 4), (12, 2), (16, 8)], [8, (12, 6), 16]):
        with pytest.raises(InputTypeError, match="increasing resolutions"):
            cn.uncertainty(q, N=bad, dt=[0.04, 0.02, 0.01])
    assert calls == []                                          # refused before any run
    rep = cn.uncertainty(lambda N, dt: 1.0, N=[(8, 4), (12, 6), (16, 8)], dt=[0.04, 0.02, 0.01])
    assert rep.value == 1.0


def test_unequal_resolution_steps_are_modelled():
    """Finding 12: N = [1, 4, 5] was modelled as equally spaced, the error
    was underestimated 13.6x and the report still said trusted."""
    import math
    r = cn.uncertainty(lambda N, dt: 1 + math.exp(-N) + dt ** 2, N=[1, 4, 5],
                       dt=[4e-4, 2e-4, 1e-4], scheme_order=2)
    assert r.interval[0] <= 1 <= r.interval[1]
    assert abs(r.value - 1) <= r.error <= 2 * abs(r.value - 1)
    # equal steps: the same estimate as the closed form |d2| rho / (1 - rho)
    r2 = cn.uncertainty(lambda N, dt: 1 + math.exp(-N), N=[2, 4, 6], dt=[4, 2, 1],
                        scheme_order=2)
    d1, d2 = math.exp(-4) - math.exp(-2), math.exp(-6) - math.exp(-4)
    rho = abs(d2 / d1)
    assert math.isclose(r2.spatial.estimate, 1.25 * abs(d2) * rho / (1 - rho), rel_tol=1e-9)


def test_untrusted_without_checked_hypotheses():
    import math
    r = cn.uncertainty(lambda N, dt: 1 + math.exp(-N) + dt ** 2, N=[2, 4, 6], dt=[4e-1, 2e-1, 1e-1])
    assert not r.trusted and any("scheme_order" in f for f in r.temporal.flags)
    r = cn.uncertainty(lambda N, dt: 1 + math.exp(-N[0]) + dt ** 2, N=[(2, 2), (4, 6), (6, 8)],
                       dt=[4e-1, 2e-1, 1e-1], scheme_order=2)
    assert not r.trusted and any("different proportions" in f for f in r.spatial.flags)
