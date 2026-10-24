# Geometries

A geometry says where the problem lives: its coordinates, which directions are periodic, where the
walls are and what they are called.

```python
import numpy as np
import conatus as cn

box = cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5))   # periodic in x, walls at z = ±0.5
m = cn.Model(box, name="convection", Ra=2500, Pr=1)
```

A geometry holds no resolution and no basis. The spectral basis of each coordinate is fixed by the
geometry (one choice per geometry, listed below and printed by `explain()`); the number of modes
is given only when you solve, with `N=` ({doc}`timesteppers`).

## Catalogue

| Geometry | Coordinates (in order) | Named boundaries |
|---|---|---|
| `cn.Box(x=cn.periodic(Lx), z=cn.walls(a, b))` | the keyword order, 1 to 3 axes | `bottom`, `top`, `walls`, `z=a`, `z=b` |
| `cn.Channel(Lx, Lz=None, h=1)` | `x`, `y` (walls at $y=\pm h$), `z` in 3D | `bottom`, `top`, `walls`, `y=-1`, `y=1` |
| `cn.Disk(R)` | `phi`, `r` | `wall`, `r=R` |
| `cn.Annulus(ri, ro)` | `phi`, `r` | `inner`, `outer`, `walls` |
| `cn.Pipe(R, L)` (also `cn.Cylinder`) | `phi`, `r`, `z` (periodic, length $L$) | `wall` |
| `cn.TaylorCouette(ri, ro, length)` | `phi`, `r`, `z` (periodic) | `inner`, `outer`, `walls` |
| `cn.Sphere(R)` | `phi`, `theta` | none (a closed surface) |
| `cn.Ball(R)` | `phi`, `theta`, `r` | `wall` |
| `cn.Shell(ri, ro)` | `phi`, `theta`, `r` | `inner`, `outer`, `walls` |

`walls` names both ends of the bounded coordinate at once: `m.at("walls", u=0)` is a no-slip
condition on both plates. `bottom` and `top` are the two ends of the bounded Cartesian axis,
whatever its name. Spherical shells have their own page, {doc}`shells`.

```python
import conatus as cn

print(cn.Annulus(1.0, 2.0).boundaries())
# {'inner': [('r', 'left')], 'outer': [('r', 'right')], 'walls': [...], 'r=1': [...], 'r=2': [...]}
print(cn.Annulus(1.0, 2.0).explain())
print(cn.TaylorCouette(1.0, 2.0, 2 * 3.14159).boundaries().keys())   # inner, outer, walls, ...
```

```text
Geometry: Annulus(inner=1, outer=2) (annulus)
  bases (fixed per geometry, sizes given at discretization):
    phi: periodic on [0, 2 pi) -> RealFourier(N_phi)
    r: [1, 2] -> AnnulusBasis(N_r) (Chebyshev T)
  named boundaries: inner = r left; outer = r right; walls = r left, r right; r=1 = r left; r=2 = r right
```

## What each geometry uses

| Geometry | Expansion | Regularity |
|---|---|---|
| Box, Channel | real Fourier series (cos/sin) on periodic axes, Chebyshev polynomials on the wall axis | — |
| Disk, Pipe | Fourier in $\varphi$ × Zernike-type polynomials $r^{\lvert m\rvert} P(2r^2-1)$ in $r$ | the basis enforces regularity at the axis |
| Annulus, Taylor–Couette | Fourier in $\varphi$ (and in $z$) × Chebyshev in $r$ | — |
| Sphere | Fourier in $\varphi$ × spin-weighted spherical harmonics | regular at the poles |
| Ball | spherical harmonics × one-sided Jacobi polynomials in $r$ | regular at the centre |
| Shell | spherical harmonics × Chebyshev in $r$ | — |

`cn.periodic(L, complex=True)` uses complex exponentials instead of the real cos/sin series (for
complex unknowns). For the sphere, the size given for `theta` is the number of spherical-harmonic
degrees $L_{\max}+1$.

## Resolution

The resolution is a tuple in the coordinate order, a dict by coordinate name, or an integer when
the geometry has one coordinate:

```python
import conatus as cn

m = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=2500, Pr=1)   # x periodic, z between walls
m.simulate(N=(64, 32), scheme="RK443", dt=1e-3)          # x: 64, z: 32
m.simulate(N={"x": 64, "z": 32}, scheme="RK443", dt=1e-3)
```

A missing size, or an integer for a geometry with several coordinates, is refused. The
default dealiasing factor follows the degree p of the products of unknowns and given fields
(a given coefficient counts as one factor, so `U*u*u` has p = 3): max(3/2, (p+1)/2)
(3/2 for quadratic terms, 2 for cubic ones), and max(2, (p+1)/2) along Hermite/Laguerre axes,
where no factor is alias-free; give `dealias=` (a number, or a dict by coordinate) to choose it.
The value and its reason are printed by `explain()` and written in the *Methods* paragraph.

## Limits

- A `Box` has at most one bounded axis; two walled directions (a duct, a cavity) are refused with
  `UnsupportedError`.
- Free-slip, no-slip and Navier partial-slip walls are named conditions of the native layer
  (`m.at("top", u=cn.free_slip)`, `cn.no_slip`, `cn.navier_slip(l)`, {doc}`boundary_conditions`);
  other conditions on single components of a vector unknown are written in the
  {doc}`expert layer <expert>`.
- Every geometry runs on the CPU and on a GPU (`backend="cuda"`). Which geometries have fused
  nonlinear kernels, and the status of the GPU path, are listed on the install page
  ({doc}`../install`, "Release status and extras") and in {doc}`gpu`.
