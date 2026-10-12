# Ready problems

`cn.problems` gives the canonical flows and model equations ready to solve, each validated
against a published value.

```python
import conatus as cn
import conatus.expert as ex

r = cn.problems.channel_evp(N=64, Re=1e4, alpha=1.0)   # every argument is required
print(r.explain())                                      # equations, units, BCs, reference
res = ex.eigenvalues(ex.discretize(r.problem, ncc=r.ncc), method="shift-invert", target=0.24, n_modes=1)
print(res.values, res.residuals)                        # Orszag (1971): 0.23752649 + 0.00373967i
```

Each function returns a `Ready`: an ordinary IR problem (`r.problem`, an `IVP` or an `EVP`,
written with the public API: open the function, it is meant to be read, copied and modified),
plus the handles needed next: `r.fields` (unknowns by name, taus included), `r.parameters`,
`r.bases` (by coordinate name), `r.ncc` (non-constant-coefficient data for
`discretize(problem, ncc=r.ncc)`), `r.notes`, `r.reference` and `r.extras`.

Nothing has a numerical default: resolution, dealiasing and domain size are keyword arguments, as
are the physical parameters. `<name>` is the IVP, `<name>_evp` its linear-stability EVP.

| Function | Problem | Validation reference |
|---|---|---|
| `channel`, `channel_evp` | plane Poiseuille, 3D IVP / Orr–Sommerfeld | Orszag 1971: c = 0.23752649 + 0.00373967i at Re = 10⁴, α = 1 |
| `plane_couette`, `plane_couette_evp` | plane Couette, 3D IVP / Orr–Sommerfeld | Romanov 1973: Im c < 0 for all Re, symmetric spectrum |
| `rayleigh_benard`, `rayleigh_benard_evp` | 2D Boussinesq convection, rigid plates | Reid & Harris 1958: Ra_c = 1707.762, k_c = 3.117 |
| `taylor_couette`, `taylor_couette_evp` | flow between rotating cylinders | Recktenwald et al. 1993: Re_c = 68.19, k_c = 3.16 (η = 0.5) |
| `pipe`, `pipe_evp` | Hagen–Poiseuille pipe flow | Meseguer & Trefethen 2003, Table 1 (Re = 1000) |
| `kuramoto_sivashinsky` | KS on (0, L) | σ = k² − k⁴ exactly; E1, E2, TW1 at L = 22 |
| `swift_hohenberg` | SH on a periodic box | s = r − (1 − k²)² exactly |
| `kolmogorov` | 2D Kolmogorov flow, streamfunction | Chandler & Kerswell 2013: Re_lin = 9.9669 (n = 4) |
| `rotating_shell_convection`, `rotating_shell_convection_evp` | rotating convection in a spherical shell | Christensen et al. 2001 case 0; onset Ra_c = 55.904299, m_c = 4 |

## Ready problems and native models

Two functions share each name, one per layer: `cn.problems.rayleigh_benard(Nx=, Nz=, ...)` is the
expert ready problem (a `Ready`, resolution in the call), `cn.problems.native.rayleigh_benard(Lx=,
Ra=, Pr=)` the native model (a `cn.Model`, resolution at `simulate`); start with the native one.
A ready problem is an expert-layer object (its taus and bases are written out, so that it can be
read and modified line by line; {doc}`expert`). Every one of them also exists as a native model,
in `cn.problems.native`, written as on paper, with the same equations, units and reference as its
expert counterpart:

| Native function | Returns | Same as the expert |
|---|---|---|
| `rayleigh_benard`, `channel`, `plane_couette`, `pipe`, `taylor_couette`, `kolmogorov`, `rotating_shell_convection`, `kuramoto_sivashinsky`, `swift_hohenberg`, `orr_sommerfeld` | a `cn.Model` | L and M bit for bit |
| `rayleigh_benard_evp`, `pipe_evp`, `taylor_couette_evp`, `rotating_shell_convection_evp` | the linearization of the model about its base state | eigenvalues to round-off |

Each one, with arguments of its validation case, and its unknowns:

```python
import numpy as np
import conatus as cn

nat = cn.problems.native
models = {
    "rayleigh_benard": nat.rayleigh_benard(Lx=2.0, Ra=2500, Pr=1),
    "kuramoto_sivashinsky": nat.kuramoto_sivashinsky(L=22.0),
    "swift_hohenberg": nat.swift_hohenberg(Lx=4 * np.pi, Ly=4 * np.pi, r=0.1),
    "channel": nat.channel(Lx=2 * np.pi, Lz=np.pi, Re=180.0, pressure_gradient=2 / 180),
    "plane_couette": nat.plane_couette(Lx=2 * np.pi, Lz=np.pi, Re=400.0),
    "orr_sommerfeld": nat.orr_sommerfeld(Re=1e4, alpha=1.0, flow="poiseuille"),
    "pipe": nat.pipe(length=4.0, Re=1000.0, pressure_gradient=4e-3),
    "taylor_couette": nat.taylor_couette(eta=0.5, mu=0.0, length=2 * np.pi / 3.16, Re=60.0),
    "kolmogorov": nat.kolmogorov(n=4, Re=20.0),
    "rotating_shell_convection": nat.rotating_shell_convection(eta=0.35, Ra=100, E=1e-3, Pr=1),
}
linearizations = {
    "rayleigh_benard_evp": nat.rayleigh_benard_evp(k=3.1, Ra=1708.0, Pr=1.0),
    "pipe_evp": nat.pipe_evp(k=1.0, Re=1000.0),
    "taylor_couette_evp": nat.taylor_couette_evp(eta=0.5, mu=0.0, k=3.16, Re=68.0),
    "rotating_shell_convection_evp": nat.rotating_shell_convection_evp(eta=0.35, Ra=100, E=1e-3, Pr=1),
}
for name, m in models.items():
    print(f"{name:28s} unknowns {[u.name for u in m.unknowns]}")
for name, lin in linearizations.items():
    print(f"{name:28s} unknowns {[u.name for u in lin.model.unknowns]}, about {sorted(lin.about)}")
print(nat.couette_profile(0.5, 0.0))     # (r_i, r_o, A, B) of V = A r + B/r
```

```python
import numpy as np
import conatus as cn

m = cn.problems.native.taylor_couette(eta=0.5, mu=0.0, length=2 * np.pi / 3.1625, Re=68.19)
print(m.explain())                       # the model as written, and what Conatus decided
```

The evolution problems give the same discrete matrices as their expert versions, bit for bit; the
explicit right-hand sides agree to round-off (the parser orders a sum of products its own way).
The stability problems are written once, as the evolution model, and linearized: their tau
formulation is the native one, so their eigenvalues (not their matrices) are compared with the
expert eigenproblems. The streamfunction gauge of Kolmogorov flow is written as an equation,
`integ(psi) = 0`; Conatus adds its constant multiplier to the vorticity equation and says so in
`explain()`.

A model whose equations contain the time `t` (a base flow written in the right-hand side) cannot be
linearized: Conatus refuses rather than freezing `t = 0`. Write the full velocity and pass the
time-periodic base flow to `linearize(about=..., period=...)` ({doc}`dynamics/nonmodal`).

To write your own variant from scratch, a native model is shorter ({doc}`equations`).
