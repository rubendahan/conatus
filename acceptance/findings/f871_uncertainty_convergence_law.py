"""F871 (hunt 5D on 0.3.0, Q5): cn.uncertainty's trusted tests the convergence law in N.

Oracle: q(N, dt) = 1 + s(N) + 0.3 dt^2 + 2 dt^3, limit 1. 0.3.0 said trusted with the error 5x too
small for s = 0.7/N, N = [16, 32, 64], and 1400x too small for the parity case s = 0.7 e^{-0.6
floor(N/2)}, N = [23, 24, 25]. Expected: neither trusted with an interval that misses 1; four
resolutions of an algebraic or geometric decay are trusted and cover 1.
"""
import math

from _acceptance import check, passed, start

start(871, "uncertainty: convergence law tested, parity steps flagged")
import conatus as cn  # noqa: E402

DT = [4e-3, 2e-3, 1e-3]


def rep(s, N):
    r = cn.uncertainty(lambda N, dt: 1 + s(N) + 0.3 * dt ** 2 + 2 * dt ** 3, N=N, dt=DT, scheme_order=2)
    return r, abs(r.value - 1) <= r.error


for name, s, N in (("algebraic", lambda N: 0.7 / N, [16, 32, 64]),
                   ("parity", lambda N: 0.7 * math.exp(-0.6 * (N // 2)), [23, 24, 25]),
                   ("parity, 4 N", lambda N: 0.7 * math.exp(-0.6 * (N // 2)), [22, 23, 24, 25])):
    r, cover = rep(s, N)
    check(cover or not r.trusted, f"{name} N={N}: trusted with error {r.error:.3g}, true {abs(r.value - 1):.3g}")
for name, s, N in (("algebraic", lambda N: 0.7 / N, [16, 24, 32, 40]),
                   ("geometric", lambda N: 0.7 * math.exp(-0.3 * N), [8, 12, 16, 20])):
    r, cover = rep(s, N)
    check(r.trusted and cover, f"control {name} N={N}: trusted={r.trusted} {r.spatial.flags}, cover={cover}")
passed("untested or parity studies not trusted; four-resolution controls trusted and covering")
