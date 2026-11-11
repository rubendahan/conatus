"""cn.uncertainty tests its own convergence law (hunt 5D on 0.3.0, Q5).

Oracle: closed forms q(N, dt) = 1 + s(N) + 0.3 dt^2 + 2 dt^3 with limit 1, scheme order 2.
0.3.0 reported trusted with errors 1.3-6x too small for an algebraic s = 0.7 N^-p (a corner, a
discontinuity) and 1400x too small for a parity case s = 0.7 e^{-0.6 floor(N/2)} (an even solution:
an odd N adds a zero coefficient, so q(25) = q(24) and the last difference was 0).
"""
import math

import pytest

import conatus as cn

DT = [4e-3, 2e-3, 1e-3]


def _rep(s, N):
    r = cn.uncertainty(lambda N, dt: 1 + s(N) + 0.3 * dt ** 2 + 2 * dt ** 3, N=N, dt=DT, scheme_order=2)
    return r, abs(r.value - 1)


@pytest.mark.parametrize("p", [1, 2, 3, 4])
@pytest.mark.parametrize("N", [[32, 48, 64], [16, 32, 64], [40, 50, 60]])
def test_three_resolutions_are_flagged_and_cover_algebraic_convergence(p, N):
    r, true = _rep(lambda N: 0.7 * N ** -p, N)
    assert not r.trusted and any("three resolutions" in f for f in r.spatial.flags)
    assert true <= r.error                       # the algebraic tail is in the estimate


@pytest.mark.parametrize("p", [1, 2, 3, 4])
@pytest.mark.parametrize("N", [[16, 24, 32, 40], [32, 48, 64, 80], [100, 110, 120, 130]])
def test_algebraic_law_found_with_four_resolutions(p, N):
    r, true = _rep(lambda N: 0.7 * N ** -p, N)
    assert r.trusted and true <= r.error <= 2 * true
    assert "algebraic" in r.spatial.law


@pytest.mark.parametrize("N", [[8, 12, 16, 20], [8, 16, 32, 64], [4, 8, 12, 16, 20]])
def test_geometric_control(N):
    r, true = _rep(lambda N: 0.7 * math.exp(-0.3 * N), N)
    assert r.trusted and r.spatial.law == "geometric" and true <= r.error <= 2 * true


@pytest.mark.parametrize("N", [[23, 24, 25], [31, 32, 33], [16, 24, 25], [22, 23, 24, 25], [20, 22, 24, 25]])
def test_parity_step_is_flagged(N):
    r, _ = _rep(lambda N: 0.7 * math.exp(-0.6 * (N // 2)), N)
    assert not r.trusted and any("unchanged" in f for f in r.spatial.flags)
    assert r.error > 1e-5                        # never the 0 of the last difference


@pytest.mark.parametrize("N", [[23, 25, 27, 29], [24, 26, 28, 30]])
def test_one_parity_is_trusted_and_covers(N):
    r, true = _rep(lambda N: 0.7 * math.exp(-0.6 * (N // 2)), N)
    assert r.trusted and true <= r.error


def test_converged_to_round_off_stays_trusted():
    """Control: q(N) = 1 + e^{-N} reaches round-off between 32 and 48 (not a parity stall)."""
    r, true = _rep(lambda N: math.exp(-N), [4, 8, 16, 32, 48])
    assert r.trusted and r.spatial.law == "resolved to round-off" and true <= r.error


def test_neither_law_is_flagged():
    """A geometric decay that slows down abruptly (two exponentials, the slow one taking over)."""
    r, _ = _rep(lambda N: math.exp(-2 * N) + 1e-6 * math.exp(-0.01 * N), [4, 6, 8, 10, 12])
    assert not r.trusted and any("neither" in f or "slowly" in f for f in r.spatial.flags)


def test_explain_says_what_is_not_established():
    r, _ = _rep(lambda N: 0.7 * N ** -2.0, [16, 24, 32, 40])
    txt = r.explain()
    assert "convergence law: algebraic" in txt and "not established" in txt


def test_sign_change_with_shrinking_differences_uses_the_last_two():
    """0.3.1 review (hunt 5G, T4): spatial errors +1e-3, -2e-6, +1e-9, -1e-12 change sign but
    shrink; the study stays flagged, and the spatial error is bounded by the last two differences
    (2e-6), not the coarsest one (1e-3). Oracle: the limit 1 is inside the interval."""
    import conatus as cn
    err = {8: 1e-3, 10: -2e-6, 12: 1e-9, 14: -1e-12}
    r = cn.uncertainty(lambda N, dt: 1 + err[N] + dt ** 2, N=[8, 10, 12, 14],
                       dt=[0.004, 0.002, 0.001], scheme_order=2)
    assert not r.trusted and any("change sign" in f for f in r.spatial.flags)
    assert 1e-12 <= r.spatial.estimate <= 3e-6
    assert r.interval[0] <= 1 <= r.interval[1]
