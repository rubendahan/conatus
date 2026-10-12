"""Tutorial 12 -- chaos and ensembles: Lyapunov spectrum, covariant vectors, Poincaré map, basins.

Lorenz (1963): dx/dt = sigma (y - x), dy/dt = r x - y - x z, dz/dt = x y - b z, sigma = 10, b = 8/3.
Expected values (checked at the end of the script):
  r = 28: exponents (0.906, 0, -14.572), D_KY = 2.062 (Sprott, lorenzle.htm); sum of the exponents
  = -(sigma + 1 + b) = -13.667 exactly for the flow (constant divergence); the neutral covariant
  vector is the flow direction f(x)
  r = 28: successive maxima of z lie on a one-dimensional cusp map (Lorenz 1963, Fig. 4), z_max < 50
  r = 20 (pre-turbulent): initial states end on C+ or C- (no chaotic attractor; a few long transients
  stay undecided at the horizon and are reported as such), basins exchanged by (x, y) -> (-x, -y)
An ODE has no geometry, so the Lorenz system is written with ode_problem (expert layer): the same
kind of problem object a native model compiles to, time-stepped by the same IVPSolver.
Run: python t12_chaos_ensembles.py   (about 1 min of CPU)
"""
# [imports]
import os

import numpy as np

import conatus as cn
from conatus.chaos import lyapunov_spectrum, poincare_section
from conatus.ensembles import Ensemble, ensemble_fate
from conatus.expert import IVPSolver, ode_problem
# [/imports]

# [system]
SIGMA, B = 10.0, 8 / 3


def lorenz(r, dt):
    pb = ode_problem(["x", "y", "z"], {"sigma": SIGMA, "r": r, "b": B},
                     lambda X, P: [P["sigma"] * (X["y"] - X["x"]),
                                   P["r"] * X["x"] - X["y"] - X["x"] * X["z"],
                                   X["x"] * X["y"] - P["b"] * X["z"]], name="Lorenz")
    return IVPSolver(cn.discretize(pb), "RK443", dt)   # scheme and dt: explicit choices


s = lorenz(28.0, 0.01)
# [/system]

# [lyapunov]
rng = np.random.default_rng(0)
X0 = np.column_stack([rng.uniform(-10, 10, 32), rng.uniform(-10, 10, 32), rng.uniform(10, 40, 32)])
res = lyapunov_spectrum(s, X0, n_vectors=3, transient_steps=1000, tangent_transient_steps=1000,
                        steps=10000, reortho_every=10, inner="euclidean", n_blocks=5, seed=1,
                        clv_window=50, clv_backward_transient=150, clv_keep_every=10)
print(res.explain())
# [/lyapunov]

# [clv]
x, y, z = (res.clv_states[..., i] for i in range(3))
f = np.stack([SIGMA * (y - x), 28.0 * x - y - x * z, x * y - B * z], axis=-1)   # vector field
g = res.clv[:, :, 1, :]                                                         # neutral CLV
cos_fg = np.abs((f * g).sum(-1)) / (np.linalg.norm(f, axis=-1) * np.linalg.norm(g, axis=-1))
print(f"neutral CLV vs f(x): min |cos| = {cos_fg.min():.8f} over {cos_fg.size} samples")
# [/clv]

# [poincare]
zmax = poincare_section(s, X0[:8], lambda X: X[:, 0] * X[:, 1] - B * X[:, 2],   # dz/dt = 0
                        direction=-1, steps=5000, g_tol=1e-10, max_iterations=30)
zk, zk1 = zmax.return_pairs(lambda p: p[2])
print(zmax.explain())
print(f"Lorenz map: {zk.size} pairs (z_k, z_k+1), z in [{zk.min():.2f}, {zk.max():.2f}]")
# [/poincare]

# [basins]
s20 = lorenz(20.0, 0.01)
c = np.sqrt(B * 19.0)
Cp, Cm = np.array([c, c, 19.0]), np.array([-c, -c, 19.0])
half = np.linspace(20 / 48, 20, 24)
grid = np.concatenate([-half[::-1], half])      # exactly symmetric (linspace(-20, 20) is not)
GX, GY = np.meshgrid(grid, grid)
starts = np.column_stack([GX.ravel(), GY.ravel(), np.full(GX.size, 19.0)])
ens = Ensemble(s20, len(starts), rhs="batched")          # 2304 members in one batch
ens.set_states(starts)


def attractor(e):
    X = e.states()
    near_p, near_m = np.linalg.norm(X - Cp, axis=1) < 1.0, np.linalg.norm(X - Cm, axis=1) < 1.0
    return np.where(near_p, 1, np.where(near_m, -1, 0))


fate = ensemble_fate(ens, attractor, check_every=100, min_steps=100, max_steps=10000,
                     classifier="|X - C+-| < 1")
basin = fate.labels.reshape(GX.shape)
print(fate.explain())
print(ens.explain())
# [/basins]

# [figures]
out = os.environ.get("CN_TUTORIAL_OUT")
if out:
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)            # a CN_TUTORIAL_OUT that does not exist yet
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].plot(zk, zk1, ".", ms=2)
    ax[0].set(xlabel="$z_k$", ylabel="$z_{k+1}$", title="Lorenz map (maxima of z)")
    ax[1].imshow(basin, origin="lower", extent=[-20, 20, -20, 20], cmap="coolwarm")
    ax[1].set(xlabel="x", ylabel="y", title="basins of $C^\\pm$, r = 20, z = 19")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "t12_lorenz_map_basins.png"), dpi=120)
# [/figures]

# [check]
lam, err = res.exponents, res.stderr
D, dD = res.kaplan_yorke()
assert abs(lam.sum() + (SIGMA + 1 + B)) < 1e-2                 # RK443 truncation ~7e-3 at dt = 0.01
assert abs(lam[0] - 0.906) < 4 * err[0] + 0.01 and abs(lam[1]) < 4 * err[1] + 0.01
assert abs(lam[2] + 14.572) < 4 * err[2] + 0.02 and abs(D - 2.062) < 4 * dD + 0.01
assert cos_fg.min() > 1 - 1e-6
assert zk.size > 100 and 25 < zk.min() and zk.max() < 50
# r = 20: no chaotic attractor, but a start near the stable manifold of the origin can stay in
# transient chaos beyond the horizon: reported as undecided (0), never assigned
assert (fate.labels == 0).sum() <= 0.005 * fate.labels.size
assert np.array_equal(basin, -basin[::-1, ::-1])                # (x, y) -> (-x, -y) swaps C+ and C-
print(f"CHECK lambda = ({lam[0]:.4f} +- {err[0]:.4f}, {lam[1]:.4f} +- {err[1]:.4f}, "
      f"{lam[2]:.4f} +- {err[2]:.4f}), sum = {lam.sum():.5f} (-13.66667), D_KY = {D:.4f} +- {dD:.4f}; "
      f"basins: {int((basin == 1).sum())} / {int((basin == -1).sum())} members, "
      f"{int((basin == 0).sum())} undecided")
# [/check]
