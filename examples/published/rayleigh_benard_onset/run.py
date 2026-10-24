"""Onset of Rayleigh-Benard convection: the marginal curves Ra(k) for three pairs of walls
(Rayleigh 1916; Chandrasekhar 1961).

References:
  Lord Rayleigh (1916), On convection currents in a horizontal layer of fluid, when the higher
    temperature is on the under side, Phil. Mag. 32, 529-546, doi:10.1080/14786441608635602: for
    two stress-free walls the marginal curve is Ra(k) = (k^2 + pi^2)^3 / k^2, with minimum 27 pi^4 / 4 at k = pi / sqrt(2).
  S. Chandrasekhar (1961), Hydrodynamic and Hydromagnetic Stability, Oxford, ch. II: two rigid
    walls, Ra_c = 1707.762 at k_c = 3.117; rigid bottom and stress-free top,
    Ra_c = 1100.65 at k_c = 2.682.

What is computed: the onset problem for the vertical velocity w(z) and the temperature theta(z)
of a mode exp(i k x + s t) in a layer 0 < z < 1 (Boussinesq, Pr = 1; the marginal curve does not
depend on Pr for this stationary onset), at N = 32 Chebyshev modes with the drift test at N = 48.
For each pair of walls: the neutral Rayleigh number Ra(k) (growth rate Re s = 0) at 25 wavenumbers,
and the minimum (Ra_c, k_c). Gaps: the relative gap |Ra - Ra_ref| / Ra_ref along the stress-free
curve (closed form at every k), and the relative gaps of Ra_c and k_c for the rigid cases, against
the values Chandrasekhar prints (4 to 7 significant digits).

Run: python examples/published/rayleigh_benard_onset/run.py   (writes result.json)
"""
import json

import numpy as np

import conatus as cn
from conatus.expert import critical

WALLS = {                                  # name: conditions at the bottom (z = 0) and at the top (z = 1)
    "rigid-rigid": (("w = 0", "dz(w) = 0", "theta = 0"), ("w = 0", "dz(w) = 0", "theta = 0")),
    "rigid-free": (("w = 0", "dz(w) = 0", "theta = 0"), ("w = 0", "dz(dz(w)) = 0", "theta = 0")),
    "free-free": (("w = 0", "dz(dz(w)) = 0", "theta = 0"), ("w = 0", "dz(dz(w)) = 0", "theta = 0")),
}
CHANDRASEKHAR = {"rigid-rigid": ("1707.762", "3.117"), "rigid-free": ("1100.65", "2.682"),
                 "free-free": (f"{27 * np.pi**4 / 4:.6f}", f"{np.pi / np.sqrt(2):.6f}")}
K = np.linspace(1.0, 7.0, 25)


def onset(bottom, top):
    m = cn.Model(cn.Box(z=cn.walls(0, 1)), name="convection onset", Ra=1700.0, k=3.1, Pr=1)
    m.scalar("w"), m.scalar("theta")          # w(z), theta(z) times exp(i k x + s t)
    m.eigenvalue("s", growth="real")
    m.define("L(f) = dz(dz(f)) - k**2*f")     # the Laplacian of one Fourier mode
    m.equations("s*L(w) = Pr*L(L(w)) - Pr*Ra*k**2*theta",
                "s*theta = w + L(theta)")
    m.at("bottom", *bottom)
    m.at("top", *top)
    return m.parametric(N=32, over=("Ra", "k"), drift=48, drift_tol=1e-6)


out = {"case": "rayleigh_benard_onset", "k": K.tolist(), "walls": {}}
for name, (bottom, top) in WALLS.items():
    P = onset(bottom, top)
    crit = critical(P, "Ra", bracket=(300.0, 3000.0), over={"k": (1.5, 4.5)}, xtol=1e-11)
    Ra = np.array([critical(P, "Ra", bracket=(300.0, 4e4), values={"k": k}, xtol=1e-11).value for k in K])
    ra_ref, k_ref = CHANDRASEKHAR[name]
    d_ra = abs(crit.value - float(ra_ref)) / float(ra_ref)
    d_k = abs(crit.over["k"] - float(k_ref)) / float(k_ref)
    print(f"{name}: Ra_c = {crit.value:.6f}, k_c = {crit.over['k']:.6f}   "
          f"(reference {ra_ref}, {k_ref}; relative gaps {d_ra:.1e}, {d_k:.1e})")
    entry = {"Ra_c": crit.value, "k_c": crit.over["k"], "Ra": Ra.tolist(), "ref": [float(ra_ref), float(k_ref)],
             "gap_Ra_c": d_ra, "gap_k_c": d_k}
    if name == "free-free":
        exact = (K**2 + np.pi**2) ** 3 / K**2
        gap = np.abs(Ra - exact) / exact
        entry["exact"] = exact.tolist()
        entry["max_gap_curve"] = gap.max()
        print(f"free-free curve: max relative gap |Ra - (k^2 + pi^2)^3 / k^2| / Ra over {K.size} "
              f"wavenumbers in [{K[0]:g}, {K[-1]:g}]: {gap.max():.1e}")
    out["walls"][name] = entry

print(" k      Ra rigid-rigid   Ra rigid-free    Ra free-free   closed form (free-free)")
for i, k in enumerate(K):
    w = out["walls"]
    print(f"{k:5.2f}  {w['rigid-rigid']['Ra'][i]:14.4f}  {w['rigid-free']['Ra'][i]:14.4f}  "
          f"{w['free-free']['Ra'][i]:14.4f}  {w['free-free']['exact'][i]:14.4f}")

if __name__ == "__main__":
    # what this case establishes: the stress-free curve to rounding error, and Chandrasekhar's critical
    # Rayleigh numbers to their printed digits (k_c within one unit of his last digit)
    w = out["walls"]
    assert w["free-free"]["max_gap_curve"] < 1e-12
    for name in ("rigid-rigid", "rigid-free"):
        ra_ref, k_ref = CHANDRASEKHAR[name]
        assert abs(w[name]["Ra_c"] - float(ra_ref)) <= 0.5 * 10.0 ** -len(ra_ref.split(".")[1])
        assert abs(w[name]["k_c"] - float(k_ref)) <= 10.0 ** -len(k_ref.split(".")[1])
    with open("result.json", "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item())
