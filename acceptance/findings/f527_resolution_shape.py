"""F527 (external review of 0.1.1, F17): a malformed coefficient array is a typed refusal.

A 2-entry array for an N = 16 variable raised a bare IndexError from inside the tail
measurement. Expected: a Conatus refusal naming the variable and the expected shape (16,).
"""
import numpy as np
from _acceptance import check, passed, refused, start

start(527, "resolution report refuses a wrong-shape array with the expected shape")
import conatus as cn  # noqa: E402

m = cn.Model(cn.Box(x=cn.periodic(6.0)))
m.scalar("u")
m.equations("dt(u)-lap(u)=0")
s = m.simulate(16, scheme="RK443", dt=0.01)
was, err = refused(s.plan.resolution_report, {"u": np.zeros(2)}, layout="coeff")
check(was, "a 2-entry array for N = 16 was accepted")
check("(16,)" in str(err) and "'u'" in str(err), f"the refusal lacks the shape: {err}")
passed("typed refusal with the expected shape (16,)")
