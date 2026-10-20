"""F407 (invariance review 2026-10-20): checkpoints of a mapped-box simulation are refused with a
typed error (they were AttributeErrors: the stepper has no checkpoint format)."""
from pathlib import Path

import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(407, "mapped-box save/restart refused, typed")
geo = cn.MappedBox(x=cn.periodic(2 * np.pi), z=cn.between(bottom=0.0, top=1.0))
m = cn.Model(geo)
m.scalar("u")
m.equations("dt(u) - dz(dz(u)) = 0")
m.at("bottom", "u=0")
m.at("top", "u=0")
s = m.simulate((8, 8), scheme="RK222", dt=1e-3)
out = []
for call in ("save", "restart", "checkpoints"):
    was_refused, _ = refused(getattr(s, call), Path("x.h5").resolve())
    out.append(was_refused)
check(all(out), f"typed refusals: {out}")
passed("save, restart, checkpoints refused")
