"""Tutorial 4 -- a nonlinear boundary-value problem and its continuation: the Bratu problem.

u'' + lam exp(u) = 0 on [-1, 1], u(-1) = u(1) = 0.
Expected values (checked at the end of the script):
  closed form u = 2 ln(cosh a / cosh(a x)), lam = 2 a^2 / cosh(a)^2 (Bratu 1914; Gelfand 1963)
  fold at a tanh a = 1:  lam_c = 0.87845767978129...  (= 3.51383071912516 / 4, the classical
  [0, 1] value, e.g. Jacobsen & Schmitt, J. Differ. Equ. 184, 2002)
Written with the native layer (cn.Model): the equation as on paper, the condition by boundary
name; Newton and continuation act on the compiled problem.
Run: python t04_bratu.py   (under a minute of CPU)
"""
# [imports]
import numpy as np
from scipy.optimize import brentq

import conatus as cn
from conatus.expert import ContinuationSettings, NewtonSolver, NLBVPContinuation, grid
# [/imports]

# [problem]
m = cn.Model(cn.Box(x=cn.walls(-1, 1)), name="Bratu", lam=0.3)
u = m.scalar("u")
m.equations("bratu: dx(dx(u)) + lam*exp(u) = 0")   # no dt(): a nonlinear boundary-value problem
m.at("walls", u=0)
print(m.explain())                               # kind, tau placement, dealiasing: all declared
pb = m.compile(64, dealias=1)                    # dealias declared: exp(u) is not alias-free anyway;
                                                 # 64 modes resolve exp(u) to 1e-12 up the upper branch
# [/problem]

# [newton]
newton = NewtonSolver(pb.problem, tol=1e-13, damping=1.0, max_iterations=20, params={"lam": 0.3})
sol = newton.solve({pb.fields["u"]: lambda x: 0 * x})   # initial guess u = 0
print(sol.explain())                             # residual per iteration, observed orders
# [/newton]

# [continuation]
settings = ContinuationSettings(ds=0.2, ds_min=1e-8, ds_max=0.6, adapt=1.5, fast_iterations=3,
                                tolerance=1e-12, max_newton=15, max_points=6, dp_fd=1e-6)
cont = NLBVPContinuation(newton, "lam", settings)
branch = cont.run(sol.coeffs, 0.3, +1, p_bounds=(0.3, 1.0))
print(branch.explain())
(fold,) = branch.folds()                          # located by bisection on the tangent
print(f"fold: lam_c = {fold.p:.14f}")
# [/continuation]

# [check]
a_star = brentq(lambda a: a * np.tanh(a) - 1.0, 0.5, 2.0, xtol=1e-15)
LAM_C = 2 * a_star ** 2 / np.cosh(a_star) ** 2   # 0.87845767978129...
a03 = brentq(lambda a: 2 * a * a / np.cosh(a) ** 2 - 0.3, 1e-12, a_star, xtol=1e-15)
exact = lambda X: 2 * np.log(np.cosh(a03) / np.cosh(a03 * X))   # lower branch
X = grid(pb.bases["x"], dealias=False)          # the Chebyshev points of the solution
assert sol.converged and sol.residual < 1e-13
assert max(sol.observed_orders()) > 1.9          # quadratic Newton
assert np.max(abs(sol.grid("u") - exact(X))) < 1e-11
assert abs(fold.p - LAM_C) < 1e-10, (fold.p, LAM_C)
print(f"OK: lam_c within {abs(fold.p - LAM_C):.1e} of the closed form {LAM_C:.14f}")
# [/check]
