"""Wavemaker and base-flow sensitivity of Orszag's Tollmien-Schlichting mode (plane Poiseuille,
Re = 1e4, alpha = 1), then the steady-forcing sensitivity.

Run: python examples/shear/sensitivity_wavemaker.py   (about 10 s of CPU; figure in $CN_OUT or .)
"""
import os

import numpy as np

from conatus.expert import Y, base_flow_sensitivity, parallel_shear_flow, structural_sensitivity
from conatus.sensitivity import steady_forcing_sensitivity

RE = 1e4


def lns(U):
    """Linearized Navier-Stokes around U(y) e_x (declared: kx, kz, Re, no-slip walls)."""
    return parallel_shear_flow(U, kx=1.0, kz=0.0, Re=RE, walls="no-slip")


adjoint_bcs = [(f, 0, s) for f in ("u", "v", "w") for s in ("left", "right")]
bfs = base_flow_sensitivity(lns, 1 - Y ** 2, N=80, target=-1j * (0.2375265 + 0.0037397j),
                            adjoint="continuous", infinite_tol=1e-10, adjoint_bcs=adjoint_bcs)
print(bfs.explain())
st = structural_sensitivity(bfs.pair)
y_star, s_star = st.maximum(points=2001)
print(f"wavemaker maximum S = {s_star:.6g} at y = {y_star:.6f} (critical layers of the TS mode)")
fs = steady_forcing_sensitivity(bfs, operator={2: 1 / RE}, bcs=[(0, "left"), (0, "right")], N=80)
print(fs.explain())
print(f"uniform extra pressure gradient dF = 1: d sigma = {fs.predict(1):.6g}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y = np.linspace(-1, 1, 801)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    ax[0].plot(st.wavemaker(y), y)
    ax[0].set_title("wavemaker S(y)")
    G = bfs.gradient(y)
    # sigma = -i omega (target -1j * c): Re sigma is the growth rate, Im sigma minus the frequency
    ax[1].plot(G.real, y, label="Re (growth)")
    ax[1].plot(G.imag, y, label="Im (frequency)")
    ax[1].set_title(r"$\delta\sigma/\delta U$")
    ax[1].legend()
    H = fs.gradient(y)
    ax[2].plot(H.real, y)
    ax[2].plot(H.imag, y)
    ax[2].set_title(r"$\delta\sigma/\delta F$")
    ax[0].set_ylabel("y")
    fig.tight_layout()
    out = os.path.join(os.environ.get("CN_OUT", "."), "sensitivity_wavemaker.png")
    fig.savefig(out, dpi=120)
    print("figure:", out)
except ImportError:
    print("matplotlib not available: no figure")
