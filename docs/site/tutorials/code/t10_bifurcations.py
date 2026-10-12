"""Tutorial 10 -- bifurcation analysis: branch switching, cycles, codimension 2, normal forms.

Lorenz (1963): dx/dt = sigma (y - x), dy/dt = r x - y - x z, dz/dt = x y - b z, sigma = 10, b = 8/3.
Expected values (checked at the end of the script):
  r = 1: symmetry-breaking pitchfork of the origin, C+- with x^2 = b (r - 1)            (exact)
  Hopf on C+- at r_H = sigma (sigma + b + 3) / (sigma - b - 1) = 470/19, omega^2 = b (sigma + r_H),
  subcritical (Marsden & McCracken 1976)
  Bogdanov-Takens normal form x' = y, y' = b1 + b2 x + x^2 - x y (Kuznetsov 2004, sec. 8.4):
  fold curve b1 = b2^2 / 4, Hopf curve b1 = 0, both ending at the BT point (0, 0)
  Swift-Hohenberg u_t = r u - (1 + d_xx)^2 u - u^3: rolls A^2 = 4 r / 3, xi0^2 = 4, g0 = 3
  (Cross & Hohenberg 1993)
The ODEs are written as Python right-hand sides (ode_problem); the PDE (Swift-Hohenberg) is a
native model (cn.Model) compiled to the dynamical-system interface (ir_system).
Run: python t10_bifurcations.py   (about 20 s of CPU)
"""
# [imports]
import numpy as np

import conatus as cn
from conatus.expert import (EVEN, IVP, CollocationMesh, ContinuationSettings, Reflection,
                            SymmetrySubspace, amplitude_equation, branch_switch, continue_codim2,
                            continue_equilibria, cos_mode_observable, ginzburg_landau, ir_system,
                            normal_form, ode_problem)
# [/imports]

# [system]
lorenz = ode_problem(["x", "y", "z"], {"sigma": 10.0, "r": 0.5, "b": 8 / 3},
                     lambda X, P: [P["sigma"] * (X["y"] - X["x"]),
                                   P["r"] * X["x"] - X["y"] - X["x"] * X["z"],
                                   X["x"] * X["y"] - P["b"] * X["z"]], name="Lorenz")
L = ir_system(lorenz, ["r"])          # G(x; r), exact J, dG/dr, and the multilinear forms B, C
print(L.explain())
# [/system]

# [equilibria]
cfg = ContinuationSettings(ds=0.1, ds_min=1e-9, ds_max=2.0, adapt=1.3, fast_iterations=4,
                           tolerance=1e-11, max_newton=12, max_points=60, dp_fd=None)
origin = continue_equilibria(L, np.zeros(3), "r", cfg, marginal_tol=1e-9, bounds=(0, 3))
(bp,) = origin.events_of("branch point")
S = np.diag([-1.0, -1.0, 1.0])        # (x, y, z) -> (-x, -y, z): a symmetry of Lorenz, declared
cplus = branch_switch(origin, bp, cfg, step=0.05, symmetry=S, bounds=(0, 30))
print(cplus.explain())
(hopf,) = cplus.events_of("hopf")
# [/equilibria]

# [normal_forms]
pf = normal_form(L, bp.point.z[:-1], kind="zero", par={"r": bp.point.z[-1]}, parameter="r", symmetry=S)
hf = normal_form(L, hopf.point.z[:-1], kind="hopf", par={"r": hopf.point.z[-1]}, parameter="r")
print(pf.explain())
print(hf.explain())
# [/normal_forms]

# [cycles]
cfo = ContinuationSettings(ds=0.2, ds_min=1e-9, ds_max=1.0, adapt=1.3, fast_iterations=4,
                           tolerance=1e-9, max_newton=12, max_points=12, dp_fd=None)
cycles = branch_switch(cplus, hopf, cfo, amplitude=0.2, mesh=CollocationMesh.uniform(40, 4),
                       tests=("lpc",))
print(cycles.explain())
# [/cycles]

# [codim2]
bt = ir_system(ode_problem(["x", "y"], {"b1": -0.5, "b2": -1.0},
                           lambda X, P: [X["y"], P["b1"] + P["b2"] * X["x"] + X["x"] * X["x"]
                                         - X["x"] * X["y"]], name="Bogdanov-Takens"), ["b1", "b2"])
cfg2 = ContinuationSettings(ds=0.05, ds_min=1e-9, ds_max=0.3, adapt=1.3, fast_iterations=3,
                            tolerance=1e-11, max_newton=12, max_points=40, dp_fd=None)
eq = continue_equilibria(bt, np.array([(1 - np.sqrt(3)) / 2, 0.0]), "b1", cfg2, marginal_tol=1e-9,
                         bounds=(-2, 2))
fold_pt, hopf_pt = eq.events_of("fold")[0], eq.events_of("hopf")[0]
fold_curve = continue_codim2(bt, fold_pt.point.z[:-1], "fold", ("b1", "b2"), cfg2,
                             par={"b1": fold_pt.point.z[-1]}, bounds={"b2": (-1.5, 1.5)})
hopf_curve = continue_codim2(bt, hopf_pt.point.z[:-1], "hopf", ("b1", "b2"), cfg2,
                             par={"b1": hopf_pt.point.z[-1]}, bounds={"b2": (-1.5, 1.5)})
print(fold_curve.explain())
print(hopf_curve.explain())
# [/codim2]

# [landau]
def swift_hohenberg(k, N=16):
    """Swift-Hohenberg on one wavelength 2 pi / k, as a native model; r is the free parameter."""
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi / k)), name="Swift-Hohenberg", r=0.0)
    m.scalar("u")                                          # the unknown
    m.define("S(f) = f + dx(dx(f))")                       # (1 + d_xx), a named operator
    m.equations("dt(u) - r*u + S(S(u)) = -u*u*u")
    c = m.compile(N, dealias=2)                            # u^3: dealias 2, declared
    # The even subspace u(-x) = u(x) (pins the phase of the rolls) is declared on the compiled
    # problem: the native layer has no symmetry= option yet.
    even = IVP(c.problem.variables, name=c.problem.name,
               symmetry=SymmetrySubspace(Reflection(c.coordsys["x"], about=0), u=EVEN))
    for eq in c.problem.equations:
        even.add_equation(eq)
    s = ir_system(even, ["r"])
    return s, np.zeros(s.n)


sh, u0 = swift_hohenberg(1.0)
cos_coeff = cos_mode_observable(sh, "u")          # A = coefficient of cos x in u (linear functional)
landau = amplitude_equation(sh, u0, "r", kind="stationary", observable=cos_coeff)
gl = ginzburg_landau(swift_hohenberg, 1.0, "r", kind="stationary", observable=cos_coeff, h=1e-2)
print(landau.explain())
print(gl.explain())
# [/landau]

# [check]
sig, b = 10.0, 8 / 3
rH = sig * (sig + b + 3) / (sig - b - 1)                        # 470/19, Lorenz (1963)
assert abs(bp.point.z[-1] - 1) < 1e-12 and cplus.switch["classification"] == "pitchfork (symmetry-breaking)"
xs = np.array([p.z[0] for p in cplus.points])
assert np.abs(xs ** 2 - b * (cplus.column("r") - 1)).max() < 1e-9          # C+: x^2 = b (r - 1)
assert abs(hopf.point.z[-1] - rH) < 1e-9 and abs(hf.omega ** 2 - b * (sig + rH)) < 1e-9
assert hf.l1 > 0                                                          # subcritical
assert cycles.column("r").max() < rH                                      # cycles exist for r < r_H
assert np.abs(fold_curve.column("b1") - fold_curve.column("b2") ** 2 / 4).max() < 1e-10
assert np.abs(hopf_curve.column("b1")).max() < 1e-12
assert np.abs(hopf_curve.events_of("bt")[0].point.z[-2:]).max() < 1e-10
assert abs(landau.g3 + 0.75) < 1e-12 and abs(gl.xi0_sq - 4) < 1e-9 and abs(gl.g0 - 3) < 1e-12
print(f"CHECK r_H = {hopf.point.z[-1]:.12f} (470/19 = {rH:.12f}), l1 = {hf.l1:.6e} > 0; "
      f"SH: A^2/r = {landau.amplitude(1.0) ** 2:.12f} (4/3), xi0^2 = {gl.xi0_sq.real:.10f}, g0 = {gl.g0.real:.12f}")
# [/check]
