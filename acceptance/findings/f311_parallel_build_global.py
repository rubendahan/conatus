"""F311 (new, review of the audit "state" 2026-10-20, F7): the parallel group assembly passes
its work through a module global.

With CONATUS_BUILD_WORKERS > 1, GlobalSystem set timestep.system._BUILD_SELF to its groups,
forked the workers (which read it) and cleared it: two threads building systems at once could
fork with each other's groups. Expected: the parent never publishes its build in the global
(each pool gets its build through its initializer), and the system is the serial one.
"""
import threading

import numpy as np
from _acceptance import check, passed, start

import conatus as cn
from conatus.timestep import system as S

start(311, "parallel builds do not share a module global")


def plan(n):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), z=cn.walls(0, 1)))
    m.scalar("T")
    m.equations("dt(T) - lap(T) = 0")
    m.at("walls", T=0)
    return m.simulate((n, 8), scheme="RK222", dt=1e-3).plan


plans = [plan(16), plan(24)]
serial = [S.GlobalSystem(p, build_workers=1) for p in plans]
S._BUILD_SELF = sentinel = ("someone else's build",)
out = [None, None]


def build(i):
    out[i] = S.GlobalSystem(plans[i], build_workers=2)


threads = [threading.Thread(target=build, args=(i,)) for i in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()
check(S._BUILD_SELF is sentinel, "the parent's module global was set by a build")
for a, b in zip(serial, out, strict=True):
    check(b is not None and (a.M != b.M).nnz == 0 and (a.L != b.L).nnz == 0,
          "a concurrent parallel build differs from the serial one")
passed("two concurrent parallel builds equal the serial ones; global untouched")
