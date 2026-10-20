"""F307 (new, review of the audit "state" 2026-10-20, F6): sim.flowmap(T)(X0) moves the
simulation.

Calling the time-T map of a simulation ran the simulation's own solver from X0: its state, time,
iteration and multistep history were left at X_N, so a later sim.run() continued from the
map's end point (time 0.1 went back to 0.05). Expected: the map leaves the simulation where it
was, and a later run is bit-identical to one where the map was never called.
"""
import numpy as np
from _acceptance import check, passed, start

import conatus as cn

start(307, "sim.flowmap(T)(X0) leaves the simulation where it was")


def make():
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
    m.scalar("u")
    m.equations("dt(u) - lap(u) = -u*u")
    return m.simulate(16, scheme="SBDF2", dt=2.0 ** -8).initial(u=lambda x: 0.3 * np.sin(x))


a, b = make(), make()
a.run(steps=6)
b.run(steps=6)
t, it = a.t, a.iteration
Y = a.flowmap(8 * a.dt)(a.solver.X)
check(a.t == t and a.iteration == it, f"the clock moved: t {t} -> {a.t}, iteration {it} -> {a.iteration}")
a.run(steps=4)
b.run(steps=4)
check(np.array_equal(a.solver.X, b.solver.X), "the run after the map differs from a run without it")
check(np.isfinite(Y).all(), "the map's value is not finite")
passed("state, clock and history unchanged; later run bit-identical")
