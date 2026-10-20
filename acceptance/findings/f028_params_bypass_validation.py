"""F028: m.params |= {...} must go through the parameter validation.

0.1.0 accepted NaN and undeclared names through the in-place merge of m.params, while
cn.Model(..., nu=nan) is refused. Expected: a typed refusal of both, and the parameters unchanged.
"""
import numpy as np
from _acceptance import check, passed, start, is_refusal, describe, failed

import conatus as cn

start(28, "in-place parameter updates are validated")


def model():
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), nu=0.1)
    m.scalar("u")
    m.equations("dt(u) - nu*dx(dx(u)) = 0")
    return m


for bad in ({"nu": float("nan")}, {"nnu": 0.2}):
    m = model()
    try:
        m.params |= bad
    except Exception as exc:  # noqa: BLE001 -- the refusal is expected
        check(is_refusal(exc), f"{bad}: crash {describe(exc)}")
        check(dict(m.params) == {"nu": 0.1}, f"{bad} refused but params changed to {dict(m.params)}")
        continue
    failed(f"m.params |= {bad} accepted: params = {dict(m.params)}")
passed("NaN and undeclared names refused")
