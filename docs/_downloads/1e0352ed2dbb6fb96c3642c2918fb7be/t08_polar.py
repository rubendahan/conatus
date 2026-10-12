"""Tutorial 8 -- curvilinear geometry: a drum, a pipe and two rotating cylinders.

The bases of the disk, the annulus and the cylinder are regular at the axis (Zernike polynomials,
spin components), never 1/(r + eps); the user names the geometry and its walls, nothing more.
Expected values (checked at the end of the script):
  Vibrating drum (Laplacian on the unit disk, T = 0 at the wall): eigenvalues -j_{m,n}^2, the
    squared zeros of the Bessel functions J_m (closed form; scipy.special.jn_zeros)
  Hagen--Poiseuille, R = 1000, (m, k) = (1, 1): centre mode -0.0708640053 + 0.8467498288i,
    wall mode -0.0911426036 + 0.4691428789i (conjugated, see the page);
    (m, k) = (1, 0): -0.0146819706          Meseguer & Trefethen (2003), J. Comput. Phys. 186, Table 1
  Taylor--Couette, eta = 0.5, outer cylinder at rest, axisymmetric: Re_c = 68.19 at k_c = 3.16
    Recktenwald, Luecke & Mueller (1993), Phys. Rev. E 48, 4444, Table I
Run: python t08_polar.py   (under a minute of CPU)
"""
# [imports]
import numpy as np
from scipy.optimize import brentq
from scipy.special import jn_zeros

import conatus as cn
from conatus.expert import eigenvalues
# [/imports]

# [disk]
drum = cn.Model(cn.Disk(radius=1.0), name="vibrating drum")
drum.scalar("T")
drum.eigenvalue("s", growth="real")
drum.equations("s*T = lap(T)")
drum.at("wall", T=0)
print(drum.explain())
plan = drum.discretize(N=(8, 32))                   # (phi, r): one group per azimuthal number m
drum_modes = {}
for g in plan.groups[:3]:                           # m = 0, 1, 2
    res = eigenvalues(plan, g)
    drum_modes[g.m] = np.sort(res.values.real)[::-1][:3]
    print(f"m = {g.m}: s = {np.round(drum_modes[g.m], 8)}  backward residual "
          f"{res.residuals.max():.1e}")
# [/disk]

# [pipe]
hp = cn.problems.pipe_evp(N=32, Re=1000.0, m=1, k=1.0)   # ready problem: equations and walls
print(hp.problem.name, "|", hp.notes.strip())   # hp.explain(): equations, bases, reference
plan = cn.discretize(hp.problem, ncc=hp.ncc)
res = eigenvalues(plan, plan.group(1, 1.0))         # one group per (m, k_z)
top = res.values[np.argsort(-res.values.real)[:3]]
print("rightmost eigenvalues (m, k) = (1, 1):", np.round(top, 10))
print(f"max backward residual {res.residuals.max():.1e}")
s10 = eigenvalues(plan, plan.group(1, 0.0)).values  # k = 0 is also a group of this plan
s10 = s10[np.argmax(s10.real)]
# [/pipe]

# [couette]
def couette_growth(Re, k, N=24):
    """Largest growth rate of axisymmetric (m = 0) perturbations with axial wavenumber k."""
    tc = cn.problems.taylor_couette_evp(N=N, Re=Re, eta=0.5, mu=0.0, m=0, k=k)
    plan = cn.discretize(tc.problem, ncc=tc.ncc)
    return eigenvalues(plan, plan.group(0, k)).values.real.max()


Re_c = {k: brentq(lambda Re, k=k: couette_growth(Re, k), 60.0, 80.0, xtol=1e-6) for k in (3.10, 3.16, 3.22)}
kc = min(Re_c, key=Re_c.get)
print("neutral Re(k):", {k: round(v, 4) for k, v in Re_c.items()})
# [/couette]

# [check]
for m in (0, 1, 2):                                 # -j_{m,n}^2, closed form
    assert np.allclose(drum_modes[m], -jn_zeros(m, 3) ** 2, rtol=1e-12, atol=0), m
MT_CENTRE, MT_WALL, MT_K0 = -0.0708640053 + 0.8467498288j, -0.0911426036 + 0.4691428789j, -0.0146819706
for ref in (MT_CENTRE, MT_WALL):                    # Table 1 lists the conjugate (see the page)
    assert np.min(np.abs(res.values - np.conj(ref))) < 1e-10
assert abs(s10 - MT_K0) < 1e-10 and res.residuals.max() < 1e-13
assert kc == 3.16 and abs(Re_c[kc] - 68.19) < 0.01   # published to 4 digits
print(f"OK: drum = Bessel zeros; pipe eigenvalues to 1e-10 of Meseguer & Trefethen; "
      f"Re_c(3.16) = {Re_c[kc]:.4f}")
# [/check]
