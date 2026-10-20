"""F005 (blocking): appending to an HDF5 analysis file must not mix incompatible runs.

Run a (nu = 0.05) writes task v = u; run b (nu = 0.4) appends task v = dx(u) to the same file.
0.1.0 accepted the append and read everything back as "grid values of u" at nu = 0.05.
Expected: a typed refusal of the append, or explicit segments whose last one carries run b's own
parameters (nu = 0.4) and task source (dx(u)).
"""
from pathlib import Path

import numpy as np
from _acceptance import check, failed, passed, refused, start

import conatus as cn

start(5, "HDF5 append checks the model identity, parameters and task expressions")


def make(nu):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="burgers", nu=nu)
    m.scalar("u")
    m.equations("dt(u)-nu*dx(dx(u)) = -u*dx(u)")
    return m.simulate(32, scheme="RK443", dt=0.002).initial(u=lambda x: np.sin(x) + 0.3 * np.cos(2 * x))


p = Path("analysis.h5").resolve()
a = make(0.05)
a.output(p, tasks={"v": "u"}, every=1)
a.run(steps=2)
b = make(0.4)
b.run(steps=2)


def append():
    b.output(p, tasks={"v": "dx(u)"}, every=1, mode="a")
    b.run(steps=1)


was_refused, _ = refused(append)
if was_refused:
    passed("incompatible append refused")
d = cn.io.read(p)
segments = getattr(d, "segments", None)
if not segments:
    failed(f"append accepted, read back as one run: parameters {d.parameters}, "
           f"source of v {d.sources.get('v')!r} (the last record is dx(u) at nu = 0.4)")
last = segments[-1]
params = getattr(last, "parameters", {}) or {}
source = (getattr(last, "sources", {}) or {}).get("v", "")
check(params.get("nu") == 0.4, f"last segment parameters {params}, expected nu = 0.4")
check("dx" in str(source), f"last segment source of v {source!r}, expected dx(u)")
passed(f"{len(segments)} segments, last: nu = {params.get('nu')}, v = {source}")
