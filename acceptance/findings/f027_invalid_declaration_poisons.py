"""F027 / F061: an invalid declaration must not poison the model.

equations("dt(u) = typo") is refused; the corrected equations("dt(u) = -u") must then be accepted
and the model must run. In 0.1.0 the bad text stayed in the model and every later call failed (a
notebook user could not recover by fixing the typo).
"""
import numpy as np
from _acceptance import check, is_refusal, passed, refused, start, describe, failed

import conatus as cn

start(27, "validate, then mutate: a refused declaration leaves the model unchanged")
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
m.scalar("u")
try:
    m.equations("dt(u) = typo")
    failed("an undeclared name in an equation was accepted")
except Exception as exc:  # noqa: BLE001 -- the refusal is expected
    check(is_refusal(exc), f"the typo crashed instead of being refused: {describe(exc)}")
was_refused, _ = refused(m.equations, "dt(u) = -u")
check(not was_refused, "the corrected equation is refused after the typo")
s = m.simulate(8, scheme="RK443", dt=0.01).initial(u=1)
s.run(steps=10)
u = float(np.mean(s.grid("u")))
check(abs(u - np.exp(-0.1)) < 1e-8, f"u(0.1) = {u}, expected exp(-0.1)")
passed("typo refused, corrected model runs")
