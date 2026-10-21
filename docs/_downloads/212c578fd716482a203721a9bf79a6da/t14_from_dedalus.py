"""Tutorial 14 -- importing a script: a convection cell as a native model, then translated from a Dedalus 3 script.

Expected values (checked at the end of the script):
  the Dedalus script rb_cell_dedalus.py (written for this tutorial, next to this file) is
  translated by cn.from_dedalus with no unsupported line: SBDF2, fixed dt = 5e-4, 40 steps;
  the native model (equations as on paper, walls by name, no tau written) and the translation
  (the script's own tau terms and lifts) are two tau formulations of the same equations, so they
  differ by a truncation error that decays spectrally with nz: at nz = 32 the two agree to
  rounding (below 1e-12 relative), at least 1000 times closer than at nz = 12.
  Dedalus is not needed and not imported here; t14b_dedalus_check.py runs the same script with
  Dedalus and compares it with the translation (rounding).
Run: python t14_from_dedalus.py   (about 5 s of CPU)
"""
# [imports]
import os

import numpy as np

import conatus as cn
# [/imports]

# [native]
# Onset of convection in diffusive units: th is the temperature minus the conduction profile 1 - z,
# so it vanishes on both walls; u.ez carries the conduction profile into the heat equation.
RA, PR, PERIOD, NX, DT = 3000.0, 7.0, 2.0, 24, 5e-4


def start(x, z):
    """One roll of the temperature deviation."""
    return 1e-3 * np.cos(2 * np.pi * x / PERIOD) * np.sin(np.pi * z)


def native(nz):
    """The cell as a Conatus model: unknowns by role, equations as written on paper, conditions by
    wall name. No tau field and no lift: the native layer places one tau per condition."""
    m = cn.Model(cn.Box(x=cn.periodic(PERIOD), z=cn.walls(0, 1)), name="convection onset",
                 Ra=RA, Pr=PR)
    m.velocity("u"), m.pressure("p"), m.scalar("th")
    m.equations("""
        momentum:  dt(u) - Pr*lap(u) + grad(p) - Ra*Pr*th*ez = -u·grad(u)
        mass:      div(u) = 0
        heat:      dt(th) - lap(th) - u·ez = -u·grad(th)
    """)
    m.at("bottom", u=0, th=0)
    m.at("top", u=0, th=0)
    sim = m.simulate(N=(NX, nz), scheme="SBDF2", dt=DT)
    sim.initial(th=start)
    return sim.run(steps=40)


cell = native(16)
print(f"native model: t = {cell.t:.3f}, max|u| = {np.abs(cell.grid('u')).max():.3e}")
# [/native]

# [translate]
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rb_cell_dedalus.py")
tr = cn.from_dedalus(SCRIPT)          # read with `ast`, never executed; Dedalus is not imported
print(tr.explain())
counts = tr.counts()
# [/translate]

# [run]
solver = tr.ivp_solver(startup="lower_order")      # the script's SBDF2, dt and initial th
solver.run(tr.steps())                              # 40 steps, no Python time loop
print(f"translation: t = {solver.t:.3f} after {tr.steps()} steps, "
      f"max|u| = {np.abs(solver.grid('u')).max():.3e}")
# [/run]

# [compare]
def translated(nz):
    """The same script at another wall-normal resolution (a declared, reported override)."""
    t = cn.from_dedalus(SCRIPT, overrides={"N_z": nz})
    s = t.ivp_solver(startup="lower_order")
    s.run(t.steps())
    return s


def rel(a, b, name):
    return np.abs(a.grid(name) - b.grid(name)).max() / np.abs(b.grid(name)).max()


gap = {}
for nz in (12, 16, 32):
    a, d = (cell, solver) if nz == 16 else (native(nz), translated(nz))
    gap[nz] = (rel(a, d, "th"), rel(a, d, "u"))
    print(f"nz = {nz:2d}: native vs translated, relative difference th {gap[nz][0]:.1e}, "
          f"u {gap[nz][1]:.1e}")
# [/compare]

# [check]
P = tr.numbers()
assert counts["unsupported"] == 0 and tr.scheme == "SBDF2" and tr.steps() == 40
assert P["Ra"] == RA and P["Pr"] == PR and P["tstep"] == DT       # same physics, same step
assert max(gap[32]) < 1e-12 and max(gap[32]) < 1e-3 * max(gap[12])   # spectral convergence
print("PASS")
# [/check]
