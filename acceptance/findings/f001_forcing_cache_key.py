"""F001 (blocking): two different forcings must not share a cache entry.

Two models differ only in the numeric content of a given field f (amplitude 1 vs 2). Running the
second with cache=True after the first must not be a cache hit, and its state must be its own:
u(t=0.05) = 2 * 0.05, mean 0.1 (got: a hit and mean 0.05 in 0.1.0). A typed refusal of the
cache (undeclared dynamic dependency) is accepted. Second part of the finding: restarting the
amplitude-2 model from a checkpoint of the amplitude-1 model must be refused (accepted in 0.1.0).
"""
from pathlib import Path

import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(1, "a changed given field must change the cache identity")


def forced(amplitude):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="forced")
    m.scalar("u")
    m.given(f=lambda x: amplitude * np.ones_like(x))
    m.equations("dt(u) = f")
    return m.simulate(8, scheme="RK443", dt=0.01).initial(u=0)


a, b = forced(1.0), forced(2.0)
a.solver.run(5, record={"root": "runs"})
was_refused, r = refused(b.solver.run, 5, record={"root": "runs"}, cache=True)
# a typed cache refusal is accepted, but the restart check below runs in every case (review
# 2026-10-21, F14: an early pass skipped it); u(0.05) = 2 * 0.05 at every grid point
mean = float("nan")
cache_refused = was_refused
if not cache_refused:
    hit = bool(r.cache_report["hit"]) if getattr(r, "cache_report", None) else False
    mean = float(np.mean(b.grid("u")))
    check(not hit, f"cache hit on a different forcing (mean u = {mean:.6g}, expected 0.1)")
    err = float(np.max(np.abs(np.asarray(b.grid("u")) - 0.1)))
    check(err < 1e-10, f"max |u - 0.1| = {err:.3g}")

c, d = forced(1.0), forced(2.0)
c.checkpoints(Path("ckpt").resolve(), every=1, keep=2)
c.run(steps=1)
files = sorted(Path("ckpt").glob("checkpoint_*.h5"))
check(bool(files), "no checkpoint written")
was_refused, _ = refused(d.restart, files[-1])
check(was_refused, "restart of the amplitude-2 model from an amplitude-1 checkpoint accepted")
passed(("cache refused" if cache_refused else f"no hit, u = {mean:.6g}") + "; foreign-forcing restart refused")
