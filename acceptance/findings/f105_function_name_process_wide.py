"""F105 (new, critic 2026-10-19, review of the runtime lot): a user function name is not
reserved for the whole Python process.

m.function("g", ...) registers "g" in a process-wide registry: a second model (or the same
notebook cell run again, with a new lambda) declaring its own "g" is refused with "already
registered with another function". The runtime lot renamed a test's "g" to "gsin" to dodge
exactly this (the guide's examples register "g" = tanh in the same process). Expected: two
models, each with its own g, both run and each uses its own function.
"""
import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(105, "user function names are scoped to their model")


def model(fn):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
    m.scalar("u")
    m.function("g", fn)
    m.equations("dt(u) = g(u)")
    return m.simulate(8, scheme="RK443", dt=0.01).initial(u=1)


a = model(lambda u: 0 * u + 1.0)        # du/dt = 1
was_refused, b = refused(model, lambda u: 0 * u + 2.0)      # du/dt = 2: a re-run with a new g
check(not was_refused, "a second model declaring its own g is refused (process-wide name)")
a.run(steps=10)
b.run(steps=10)
ua, ub = float(np.mean(a.grid("u"))), float(np.mean(b.grid("u")))
check(abs(ua - 1.1) < 1e-12 and abs(ub - 1.2) < 1e-12, f"u = {ua}, {ub}; expected 1.1, 1.2")
passed("each model uses its own g")
