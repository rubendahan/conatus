"""F842 (hunt 5E, R3): the refusals SECURITY.md lists carry a code.

Oracle: files built with h5py and NumPy, outside Conatus: an HDF5 checkpoint whose first dataset
is replaced by an external link, a soft link, a virtual dataset or external raw storage, and a
.npz with an object array. Each must be refused by restart and cn.io.read with UnsafeFileError
(CN171, also a ValueError). A string base state must be an InputTypeError. Controls: the
unmodified files restart.
"""
import shutil

import numpy as np
from _acceptance import check, passed, skipped, start

start(842, "file refusals are typed")
try:
    import h5py
except ImportError:
    skipped("h5py is not installed")
import conatus as cn  # noqa: E402
from conatus.errors import ConatusError, InputTypeError, UnsafeFileError  # noqa: E402


def heat():
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="heat", nu=0.1)
    m.scalar("u")
    m.equations("dt(u) - nu*dx(dx(u)) = 0")
    return m.simulate(N=16, scheme="RK443", dt=0.01)


def first_dataset(f):
    found = []
    f.visititems(lambda n, o: found.append(n) if isinstance(o, h5py.Dataset) else None)
    return found[0]


s = heat()
s.initial(u=lambda x: np.sin(x))
s.run(steps=3)
s.save("good.h5")
s.save("good.npz")
for kind in ("extlink", "softlink", "virtual", "external"):
    fn = f"bad_{kind}.h5"
    shutil.copy("good.h5", fn)
    with h5py.File(fn, "a") as f:
        tgt = first_dataset(f)
        data = f[tgt][()]
        del f[tgt]
        if kind == "extlink":
            with h5py.File("other.h5", "w") as g:
                g["d"] = data
            f[tgt] = h5py.ExternalLink("other.h5", "d")
        elif kind == "softlink":
            f["copy"] = data
            f[tgt] = h5py.SoftLink("/copy")
        elif kind == "virtual":
            with h5py.File("vsrc.h5", "w") as g:
                g["d"] = data
            lay = h5py.VirtualLayout(shape=np.shape(data), dtype=np.asarray(data).dtype)
            lay[...] = h5py.VirtualSource("vsrc.h5", "d", shape=np.shape(data))
            f.create_virtual_dataset(tgt, lay)
        else:
            np.asarray(data).tofile("raw.bin")
            f.create_dataset(tgt, shape=np.shape(data), dtype=np.asarray(data).dtype,
                             external=[("raw.bin", 0, h5py.h5f.UNLIMITED)])
    for label, read in (("restart", lambda p: heat().restart(p)), ("io.read", cn.io.read)):
        try:
            read(fn)
            check(False, f"{label} accepted {kind}")
        except UnsafeFileError as err:
            check(err.error_code == "CN171" and isinstance(err, ValueError), f"{label} {kind}")
        except Exception as err:  # noqa: BLE001
            check(False, f"{label} {kind}: {type(err).__name__}: {err}")
d = dict(np.load("good.npz"))
k = next(n for n in d if n != "meta")
d[k] = np.array([{"a": 1}], dtype=object)
np.savez("bad.npz", **d)
try:
    heat().restart("bad.npz")
    check(False, "pickled npz accepted")
except UnsafeFileError:
    pass
for good in ("good.h5", "good.npz"):
    r = heat()
    r.restart(good)
    check(r.iteration == 3, f"control {good}")
m = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=2500, Pr=1)
try:
    m.linearize(about={"T": "-z"})
    check(False, "string base accepted")
except InputTypeError:
    pass
except Exception as err:  # noqa: BLE001
    check(isinstance(err, ConatusError), f"string base: {type(err).__name__}: {err}")
passed("8 HDF5 refusals, 1 npz, 1 string base, typed")
