"""F009 (important): a registry end-state of a CNAB2 run must resume it.

Run a: CNAB2, 11 steps, recorded; run b restarts from a's state_end.npz and takes one step; it
must agree with a taking that same step (the multistep history is part of the state). 0.1.0: the
file had no LX history and b.run crashed with IndexError. A typed refusal of the restart
(incomplete state) is accepted.
"""
import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(9, "registry checkpoint holds the full multistep state")


def make():
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="burgers", nu=0.05)
    m.scalar("u")
    m.equations("dt(u)-nu*dx(dx(u)) = -u*dx(u)")
    return m.simulate(32, scheme="CNAB2", dt=0.002).initial(u=lambda x: np.sin(x) + 0.3 * np.cos(2 * x))


a = make()
r = a.solver.run(11, record={"root": "runs"})
path = r.path / "state_end.npz"
b = make()


def resume():
    b.restart(path)
    b.run(steps=1)


was_refused, _ = refused(resume)
if was_refused:
    passed("restart from an incomplete state refused")
a.run(steps=1)
diff = float(np.max(np.abs(np.asarray(a.grid("u")) - np.asarray(b.grid("u")))))
check(diff < 1e-12, f"resumed run differs from the continuous one by {diff:.3g}")
check(abs(float(b.t) - float(a.t)) < 1e-14, f"clock: resumed t = {b.t}, continuous t = {a.t}")
passed(f"resumed = continuous to {diff:.2g}")
