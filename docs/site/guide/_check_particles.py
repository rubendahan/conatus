"""Executable examples of the guide page particles.md (tracers and dye inside a native simulation).

Every code block of that page is a region of this file (``# [name]`` ... ``# [/name]``), included
verbatim with ``literalinclude`` (``:dedent: 4``): what the reader sees is what ran. The asserts
after each example are the checks (not shown on the page). Each example runs in a fresh temporary
directory, since it writes its HDF5 file to the current directory. Not a Sphinx source (leading
underscore).

Run (CPU, about a minute; as a job, not on the login node):
    python docs/site/guide/_check_particles.py [name ...]
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import time
import traceback
import warnings


def tracers() -> None:
    """particles.md, *Tracers*."""
    # [tracers]
    import numpy as np

    import conatus as cn

    box = cn.Box(x=cn.periodic(4.0), y=cn.walls(-1, 1))
    m = cn.Model(box, name="plane Couette", nu=0.1)
    m.velocity("u"), m.pressure("p")
    m.equations("dt(u) + grad(p) - nu*lap(u) = -u@grad(u)", "div(u) = 0")
    m.at("bottom", u=(-1, 0))
    m.at("top", u=(1, 0))
    sim = m.simulate(N=(32, 32), scheme="RK443", dt=0.01)
    sim.initial(u=lambda x, y: (y, 0 * x))
    tracers = sim.particles(np.array([[0.0, -1.0], [2.0, 0.3]]), path="tracers.h5", every=50,
                            velocity="u", at_walls="refuse")
    sim.run(until=3.0)
    print(tracers.explain())
    X = tracers.positions(unwrapped=True)       # (P, d), coordinate order
    # [/tracers]
    # exact steady Couette: x(t) = x0 + y0 t, y constant; the wall tracer moves with the wall
    assert np.abs(X - np.array([[-3.0, -1.0], [2.9, 0.3]])).max() < 1e-10
    assert cn.io.read("tracers.h5").tasks["position"].shape[1:] == (2, 2)


def dye() -> None:
    """particles.md, *Dye*."""
    # [dye]
    import numpy as np

    import conatus as cn

    box = cn.Box(x=cn.periodic(4.0), y=cn.walls(-1, 1))
    m = cn.Model(box, name="plane Couette with dye", nu=0.1)
    m.velocity("u"), m.pressure("p")
    m.equations("dt(u) + grad(p) - nu*lap(u) = -u@grad(u)", "div(u) = 0")
    m.at("bottom", u=(-1, 0))
    m.at("top", u=(1, 0))
    m.dye("c", velocity="u", kappa=0.01)
    sim = m.simulate(N=(32, 32), scheme="RK443", dt=0.01)
    sim.initial(u=lambda x, y: (y, 0 * x), c=lambda x, y: np.exp(-((x - 2) ** 2 + y ** 2) / 0.3))
    out = sim.dye("c", "dye.h5", every=50)
    sim.run(until=3.0)
    mass = cn.io.read("dye.h5").tasks["mass"]
    print(out.explain())
    print("relative mass drift:", abs(mass[-1] - mass[0]) / abs(mass[0]))
    # [/dye]
    # no-flux walls: the mass is kept to the tau-correction level (3e-6 at N = 16, spectral in N)
    assert abs(mass[-1] - mass[0]) <= 1e-6 * abs(mass[0])


CHECKS = {"tracers": tracers, "dye": dye}


def main() -> int:
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(CHECKS)
    warnings.simplefilter("ignore")              # resolution warnings are not the point here
    bad = 0
    for name in names:
        buf = io.StringIO()
        t0 = time.perf_counter()
        here = os.getcwd()
        try:
            os.chdir(tempfile.mkdtemp(prefix=f"cn_guide_{name}_"))
            with contextlib.redirect_stdout(buf):
                CHECKS[name]()
            status = "PASS"
        except Exception:  # noqa: BLE001 -- every failure is reported, the run goes on
            status, bad = "FAIL", bad + 1
            traceback.print_exc()
        finally:
            os.chdir(here)
        print(f"{status} {name} ({time.perf_counter() - t0:.1f} s)")
        print("    " + buf.getvalue().replace("\n", "\n    ").rstrip())
    print(f"{len(names) - bad} pass, {bad} fail")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
