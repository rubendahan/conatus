"""F947 (review H13): sim.initial refuses masked entries and non-numeric arrays with a typed error.

Oracle: a masked entry has no value, so reading its stored number as data is wrong; object,
structured and datetime arrays are not numbers. Control: a plain float array is accepted and
sim.grid returns it exactly (NumPy equality); a masked array with no masked entry is accepted.
"""
import numpy as np
from _acceptance import check, passed, refused, start

start(947, "sim.initial: masks and non-numeric arrays refused")
import conatus as cn  # noqa: E402

m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
m.scalar("u")
m.equations("dt(u)=0")
sim = m.simulate(8, scheme="RK443", dt=.1)
x = np.linspace(0, 2 * np.pi, 8, endpoint=False)
data = np.cos(x)
cases = {"masked": np.ma.array(data, mask=[False] * 7 + [True]),
         "object": data.astype(object),
         "structured": np.array([(v,) for v in data], dtype=[("v", float)]),
         "datetime": np.zeros(8, dtype="M8[s]")}
for name, a in cases.items():
    ok, _ = refused(sim.initial, u=a)
    check(ok, f"{name}: accepted")
sim.initial(u=data)
check(np.max(np.abs(sim.grid("u") - data)) < 1e-14, "control: plain array")
sim.initial(u=np.ma.array(data, mask=False))
check(np.max(np.abs(sim.grid("u") - data)) < 1e-14, "control: masked array without masked entry")
passed("four bad inputs refused with a typed error; plain data exact")
