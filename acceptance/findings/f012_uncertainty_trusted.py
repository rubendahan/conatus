"""F012 (important): uncertainty() must not mark an underestimated error as trusted.

q(N, dt) = 1 + exp(-N) + dt^2 with N = [1, 4, 5] (not a geometric refinement): the true value 1
lies 13.6 reported errors away and 0.1.0 said trusted=True. Expected: trusted is False, or the
reported interval contains the exact value 1. A typed refusal of the non-geometric study is
accepted.
"""
import math

from _acceptance import check, passed, refused, start

import conatus as cn

start(12, "uncertainty() trusts only checked hypotheses")
was_refused, r = refused(cn.uncertainty, lambda N, dt: 1 + math.exp(-N) + dt ** 2, N=[1, 4, 5],
                         dt=[4e-4, 2e-4, 1e-4], scheme_order=2)
if was_refused:
    passed("non-geometric refinement refused")
inside = r.interval[0] <= 1 <= r.interval[1]
check(not r.trusted or inside, f"trusted={r.trusted}, value {r.value}, error {r.error:.3g}: exact "
      f"value 1 is {(r.value - 1) / r.error:.3g} errors away")
passed(f"trusted={r.trusted}, interval contains 1: {inside}")
