"""F805 (external review of 0.2.3, finding 6): HDF5 loaders refuse datasets stored in other files.

Oracle: a harmless local sentinel of known floats. The loader must refuse before reading it: it
never returns those values.
"""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from _acceptance import check, need, passed, start

start(805, "HDF5 loaders refuse external and virtual datasets")
need("h5py")
import h5py  # noqa: E402

import conatus as cn  # noqa: E402
from conatus.io import read  # noqa: E402
from conatus.io.checkpoint import checkpoint_meta, read_checkpoint  # noqa: E402


def build():
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
    m.scalar("u")
    m.equations("dt(u)=-u")
    sim = m.simulate(8, scheme="RK443", dt=0.01)
    sim.initial(u=1)
    return sim


def expect_refused(call, what):
    try:
        out = call()
    except ValueError as err:
        check("other files" in str(err), f"{what}: {err}")
        return
    check(False, f"{what} accepted: {out!r}")


with TemporaryDirectory(dir=Path.cwd()) as directory:
    d = Path(directory)
    sim = build()
    data = np.arange(sim.solver.X.size, dtype=float) + 1234.0
    raw = d / "sentinel.bin"
    raw.write_bytes(data.tobytes())
    source = d / "source.h5"
    with h5py.File(source, "w") as f:
        f["X"] = data
    for kind in ("external", "virtual"):
        cp = d / f"checkpoint_{kind}.h5"
        sim.save(cp)
        with h5py.File(cp, "r+") as f:
            del f["state/X"]
            if kind == "external":
                f["state"].create_dataset("X", shape=data.shape, dtype="f8",
                                          external=[(str(raw.resolve()), 0, data.nbytes)])
            else:
                lay = h5py.VirtualLayout(shape=data.shape, dtype="f8")
                lay[:] = h5py.VirtualSource(str(source.resolve()), "X", shape=data.shape)
                f["state"].create_virtual_dataset("X", lay)
        check(checkpoint_meta(cp) is None, f"{kind}: discovery accepted it")
        expect_refused(lambda p=cp: read_checkpoint(p), f"{kind}: read_checkpoint")
        expect_refused(lambda p=cp: build().restart(p), f"{kind}: restart")
    # an ordinary checkpoint and a foreign file read by cn.io.read
    ok = d / "plain.h5"
    sim.save(ok)
    check(checkpoint_meta(ok) is not None, "plain checkpoint refused")
    check(np.array_equal(read_checkpoint(ok)[1]["X"], sim.solver.X), "plain checkpoint values")
    other = d / "foreign.h5"
    with h5py.File(other, "w") as f:
        f.attrs["format"] = "x"
        f.create_dataset("tasks/u", shape=data.shape, dtype="f8",
                         external=[(str(raw.resolve()), 0, data.nbytes)])
    expect_refused(lambda: read(other), "cn.io.read")
    # a hard link that closes a loop of groups is finished, not followed for ever; write_xdmf refuses
    loop = d / "loop.h5"
    sim.save(loop)
    with h5py.File(loop, "r+") as f:
        f["state/back"] = f["/"]
    check(checkpoint_meta(loop) is not None, "a loop of hard links stops the walk")
    from conatus.io.xdmf import write_xdmf  # noqa: E402
    expect_refused(lambda: write_xdmf(other), "write_xdmf")
passed("external and virtual storage refused before any payload read; a plain checkpoint loads")
