"""Tutorial 13 -- where a global mode is born, and what the data say about it.

Expected values (checked at the end of the script):
  Ginzburg-Landau wavemaker S_0(x) = |exp(-chi^2 x^2)| |chi| / sqrt(pi)   Chomaz (2005), closed form
  d lambda_0 / d mu_2 = 1 / (4 chi^2)                                   closed form
  DMD of a transient: the four least stable eigenvalues, to 1e-9      the discrete spectrum
Run: python t13_sensitivity_data.py   (about 20 s of CPU)
"""
# [imports]
import numpy as np
import scipy.linalg as sla

from conatus.expert import (DataInnerProduct, Y, base_flow_sensitivity, dmd, ginzburg_landau_exact,
                            linear_ginzburg_landau, state_space,
                            structural_sensitivity)
from conatus.diagnostics import Chebyshev, Grid
# [/imports]

# [model]
NU, GAMMA, MU_C, MU_2 = 2 + 0.4j, 1 - 1j, 0.34, -0.01        # Bagheri et al. (2009)
BOUNDS = (-60.0, 60.0)                                     # the whole line, truncated (declared)


def gl(mu):
    """dq/dt = -nu q' + gamma q'' + mu(x) q, q = 0 at both ends; mu is a sympy expression in Y."""
    return linear_ginzburg_landau(nu=NU, gamma=GAMMA, mu=mu, bounds=BOUNDS)


MU = MU_C + MU_2 * Y ** 2 / 2
exact = ginzburg_landau_exact(nu=NU, gamma=GAMMA, mu_c=MU_C, mu2=MU_2, n=0)
print(exact.explain())
# [/model]

# [sensitivity]
bfs = base_flow_sensitivity(gl, MU, N=140, target=exact.eigenvalue, adjoint="continuous",
                            infinite_tol=1e-10, adjoint_bcs=[("q", 0, "left"), ("q", 0, "right")])
pair = bfs.pair
print(pair.explain())
st = structural_sensitivity(pair)
x_star, s_star = st.maximum(points=1201)
print(f"wavemaker maximum S = {s_star:.12f} at x = {x_star:.2e}")
dmu2 = bfs.predict(Y ** 2 / 2)                            # delta mu = x^2/2, i.e. d/d mu_2
print(f"d lambda/d mu_2 = {dmu2:.12f} (density)  {bfs.discrete_shift(Y ** 2 / 2):.12f} (pencil)")
# [/sensitivity]

# [data]
# The same model, smaller and more damped (mu_c = -0.2), in energy-orthonormal coordinates.
small = linear_ginzburg_landau(nu=NU, gamma=GAMMA, mu=-0.2 + MU_2 * Y ** 2 / 2, bounds=(-20, 20))
ss = state_space(small.discretize(32), infinite_tol=1e-10)
grid = Grid([Chebyshev(-20.0, 20.0, 64)])                 # values on 64 Gauss nodes
t_nodes = (2 * grid.axes[0].nodes() + 0.0) / 40.0
E = np.polynomial.chebyshev.chebvander(t_nodes, 31) @ ss.to_state      # q -> grid values
ip = DataInnerProduct.from_grid(grid, components=["q"], weights={"q": 1.0})

# (1) DMD of a free transient started in the span of the 4 least stable global modes
dt = 0.5
lam, V = np.linalg.eig(ss.A)
lead4 = np.argsort(-lam.real)[:4]
q0 = V[:, lead4] @ np.array([1.0, 0.5j, 0.3, 0.2 - 0.1j])
Phi = sla.expm(ss.A * dt)
X = E @ np.stack([np.linalg.matrix_power(Phi, j) @ q0 for j in range(40)], axis=1)
r = dmd(X, dt=dt, rank=4, inner_product=ip, amplitudes="optimal")
print(r.explain())

# [/data]

# [check]
S0 = abs(exact.chi) / np.sqrt(np.pi)                      # Chomaz (2005): S_0(0) = |chi|/sqrt(pi)
assert abs(s_star - S0) < 1e-9 * S0 and abs(x_star) < 1e-4
assert abs(dmu2 - 1 / (4 * exact.chi ** 2)) < 1e-9 * abs(dmu2)
for mu in lam[lead4]:
    assert np.min(np.abs(r.omega - mu)) < 1e-9, (mu, r.omega)
print("OK: wavemaker and d lambda/d mu_2 match the closed forms; DMD finds the four least stable "
      "eigenvalues")
# [/check]
