"""Tutorial 3 -- symmetries and symmetry subspaces: Rayleigh--Benard onset in half the unknowns.

Written with the native layer (cn.Model): the convection equations as on paper, conditions by
boundary name. The symmetry subspace itself is declared on the compiled problem (expert layer):
the native layer has no symmetry= option yet, so the parities of the boundary unknowns it created
are written out here.
Expected values (checked at the end of the script):
  rigid-rigid onset Ra_c = 1707.762 at k_c = 3.1163 (Reid & Harris 1958; Chandrasekhar 1961):
  the leading growth rate there is 0 to the published digits, in the full space and in the
  subspace even about mid-depth; both eigenvalues agree to rounding with half the unknowns.
Run: python t03_symmetries.py   (a few seconds of CPU)
"""
# [imports]
import time

import numpy as np

import conatus as cn
from conatus.expert import (EVEN, ODD, Reflection, SymmetryBreakingError, SymmetrySubspace,
                            discretize, eigenvalues, linearize)
# [/imports]

# [problem]
RA_C, K_C = 1707.762, 3.1163                  # Reid & Harris (1958)
box = cn.Box(x=cn.periodic(2 * np.pi / K_C), z=cn.walls(-0.5, 0.5))   # one wavelength at k_c
m = cn.Model(box, name="Boussinesq RB", Ra=RA_C, Pr=1)
u, p, T = m.velocity("u"), m.pressure("p"), m.scalar("T")
m.equations("""
    momentum:   dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)
    continuity: div(u) = 0
    heat:       dt(T) - lap(T) = -u·grad(T)
""")
m.at("bottom", u=0, T=+0.5)                   # rigid plates, fixed temperatures
m.at("top", u=0, T=-0.5)
onset_evp = m.linearize(about={u: 0, T: lambda z: -z})   # conduction state T0 = -z
# [/problem]

# [full]
plan = onset_evp.discretize(32, k=K_C)        # N = 32 across the layer, the Fourier size holds k_c
group = onset_evp.group(plan, K_C)
t0 = time.perf_counter()
res = eigenvalues(plan, group)
wall_full = time.perf_counter() - t0
i = int(res.values.real.argmax())
sf, nf = res.values[i], plan.matrices(group).L.shape[0]
print(f" full space: leading sigma = {sf.real:+.4e}   n = {nf:3d}   QZ {1e3 * wall_full:5.1f} ms   "
      f"residual {res.residuals[i]:.1e}")
# [/full]

# [subspace]
evp, ncc, comp = onset_evp.compile({"x": 4, "z": 32})   # the compiled problem (expert objects)
print("unknowns of the perturbation problem:", ", ".join(v.name for v in evp.variables))
Rz = Reflection(comp.coordsys["z"], about=0)  # z -> -z about mid-depth
# Perturbation unknowns carry the prefix d. Only the "scalar" parity is declared for a vector:
# du=ODD means u_x odd and u_z even (the component along z flips sign under the reflection).
# The boundary unknowns tau_* were created by the native layer (listed by m.explain()); their
# parities follow from the walls they enforce and are declared too.
onset = SymmetrySubspace(Rz, du=ODD, dT=EVEN, dp=ODD, T0=ODD,         # w, T even: the onset roll
                         dtau_u1=EVEN, dtau_u2=ODD, dtau_T1=ODD, dtau_T2=EVEN, dtau_p=ODD)
other = SymmetrySubspace(Rz, du=EVEN, dT=ODD, dp=EVEN, T0=ODD,       # the complementary class
                         dtau_u1=ODD, dtau_u2=EVEN, dtau_T1=EVEN, dtau_T2=ODD, dtau_p=EVEN)
print(onset.explain())
# [/subspace]

# [solve]
out = {}
for label, sym in (("onset class", onset), ("other class", other)):
    sub = linearize(comp.problem, evp.base, evp.eigenvalue, symmetry=sym)
    splan = discretize(sub, ncc=ncc)
    g = next(g for g in splan.groups if g[0].k > 0)
    t0 = time.perf_counter()
    r = eigenvalues(splan, g)
    wall = time.perf_counter() - t0
    j = int(r.values.real.argmax())
    out[label] = (r.values[j], splan.matrices(g).L.shape[0])
    print(f"{label:>11}: leading sigma = {r.values[j].real:+.4e}   n = {out[label][1]:3d}   "
          f"QZ {1e3 * wall:5.1f} ms   residual {r.residuals[j]:.1e}")
    if sym is onset:
        print(sub.symmetry_report())          # mirrored conditions dropped, what the subspace hides
# [/solve]

# [guard]
wrong = SymmetrySubspace(Rz, du=EVEN, dT=EVEN, dp=EVEN, T0=ODD,       # T with the parity of w:
                         dtau_u1=ODD, dtau_u2=EVEN, dtau_T1=EVEN, dtau_T2=ODD, dtau_p=EVEN)
try:                                                                   # not equivariant
    linearize(comp.problem, evp.base, evp.eigenvalue, symmetry=wrong)
except SymmetryBreakingError as e:
    print(f"refused: {e}")
# [/guard]

# [check]
(so, no), (sa, _) = out["onset class"], out["other class"]
assert abs(sf) < 1e-4, sf                     # neutral at (Ra_c, k_c) to the published digits
assert abs(so - sf) < 1e-10                   # the onset class carries the leading mode ...
assert sa.real < -1                           # ... the other class decays at onset
assert no <= 0.55 * nf                        # with about half the unknowns
print(f"OK: neutral mode at Reid & Harris' (Ra_c, k_c) in the onset class, n = {no} vs {nf}")
# [/check]
