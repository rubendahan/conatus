"""F952 (hunt 5C, M4): typed refusals for EmbeddedError flags and numbers and for cn.units;
checkpoints record the Conatus version and a load says when it differs.

Oracle: the written values. adapt="no" is not a switch, tol=True is not a tolerance, tol= without
adapt would be ignored, an infinite or boolean scale, a non-positive prefactor, a missing scale, a
NaN energy and a negative box are refused with a ConatusError. A checkpoint carries
conatus.__version__; a load of one edited to another version warns and names both.
"""
import json
import warnings

import numpy as np
from _acceptance import check, passed, refused, start

start(952, "EmbeddedError and cn.units typed refusals; checkpoint version")
import conatus as cn  # noqa: E402
from conatus import units as U  # noqa: E402
from conatus.expert import EmbeddedError  # noqa: E402

m = cn.Model(cn.Box(z=cn.walls(0, 1)))
m.scalar("u")
m.equations("dt(u) - dz(dz(u)) = 0")
m.at("walls", u=0)
s = m.simulate(N=(16,), scheme="RK443", dt=1e-3)
s.initial(u=lambda z: np.sin(np.pi * z))
for kw in ({"adapt": "no"}, {"adapt": True, "tol": True, "safety": 0.9},
           {"adapt": True, "tol": "1e-6", "safety": 0.9}, {"tol": 1e-6}, {"safety": 0.9},
           {"adapt": True, "tol": 1e-6, "safety": True}):
    ok, _ = refused(EmbeddedError, s.solver, **kw)
    check(ok, f"EmbeddedError({kw}) accepted")
for f in (lambda: U.Scale("L", "len", float("inf")), lambda: U.Scale("L", "len", True),
          lambda: U.Scale("L", "len", "2"), lambda: U.Group("Re", "UL/nu", -1.0),
          lambda: U.Group("Re", "UL/nu", 0.0), lambda: U.Energy(float("nan"), "total"),
          lambda: U.UnitSystem("x", "f", {}).scale("length"),
          lambda: U.Energy(1.0, "modal").to_total(Lx=-1.0, Lz=1.0),
          lambda: cn.periodic(np.array(-2.0))):
    ok, _ = refused(f)
    check(ok, "a bad unit input was accepted")
check(cn.periodic(np.array(2.0)) is not None, "control: periodic of a 0-d array")

s.save("ck.npz")
with np.load("ck.npz") as z:
    meta = json.loads(str(z["meta"]))
    arrays = {k: z[k] for k in z.files if k != "meta"}
check(meta.get("conatus_version") == cn.__version__, f"version recorded: {meta.get('conatus_version')}")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    s.restart("ck.npz")
check(not [w for w in caught if "written by" in str(w.message)], "control: same version warned")
meta["conatus_version"] = "0.0.1"
np.savez("old.npz", meta=json.dumps(meta), **arrays)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    s.restart("old.npz")
check(any("0.0.1" in str(w.message) and cn.__version__ in str(w.message) for w in caught),
      "a checkpoint of another version restarted without a word")
passed("flags, numbers and units refused with types; checkpoint version recorded and compared")
