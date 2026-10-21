"""F507 (external review of 0.1.1, F8): a Hopf curve continued into real-saddle points.

x' = y, y' = b1 + b2 x + x^2 - x y: the Hopf curve is b1 = 0 with b2 < 0 and ends at the
Bogdanov-Takens point; for b2 > 0 the Jacobian [[0, 1], [b2, 0]] has real eigenvalues. 0.1.1
continued the "hopf (codim 2)" curve to b2 = 1.53. Correct behaviour: every point of the curve
has a pair +-i omega in the Jacobian's eigenvalues (NumPy), so b2 < 0.
"""
import numpy as np
from _acceptance import check, passed, start

from conatus.expert import (ContinuationSettings, continue_codim2, continue_equilibria, ir_system,
                            ode_problem)

start(507, "every point of a Hopf curve has a purely imaginary pair")
bt = ir_system(ode_problem(["x", "y"], {"b1": -0.5, "b2": -1.0},
                           lambda X, P: [X["y"], P["b1"] + P["b2"] * X["x"] + X["x"] ** 2
                                         - X["x"] * X["y"]]), ["b1", "b2"])
cfg = ContinuationSettings(ds=0.05, ds_min=1e-9, ds_max=0.3, adapt=1.3, fast_iterations=3,
                           tolerance=1e-11, max_newton=12, max_points=40, dp_fd=None)
eq = continue_equilibria(bt, np.array([(1 - np.sqrt(3)) / 2, 0.0]), "b1", cfg, marginal_tol=1e-9,
                         bounds=(-2, 2))
h = eq.events_of("hopf")[0]
curve = continue_codim2(bt, h.point.z[:-1], "hopf", ("b1", "b2"), cfg, par={"b1": h.point.z[-1]},
                        bounds={"b2": (-1.5, 1.5)})
for p in curve.points:
    x, y, b2 = p.z[0], p.z[1], p.z[-1]
    lam = np.linalg.eigvals([[0.0, 1.0], [b2 + 2 * x - y, -x]])
    check(np.abs(lam.real).max() < 1e-9 and np.abs(lam.imag).min() > 0,
          f"point b1={p.z[-2]:.3g}, b2={b2:.6g} called Hopf; Jacobian eigenvalues {lam}")
passed(f"{len(curve.points)} points, b2 in [{curve.column('b2').min():.3g}, "
       f"{curve.column('b2').max():.3g}]; stop: {curve.stop_reason[:60]}")
