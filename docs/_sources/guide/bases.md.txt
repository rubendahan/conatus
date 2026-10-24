# Bases and domains

In the expert layer a domain is a coordinate system with one spectral basis per coordinate, each with its size and dealiasing factor.

A `Domain` is a coordinate system and one basis per coordinate. Fourier directions are separable:
the discrete problem splits into independent **groups** (one per wavenumber, or per azimuthal
number $m$ in polar geometry), each solved with banded Olver–Townsend operators.

## Dealiasing factors

Expert-layer bases require an explicit `dealias` factor; there is no default. Native models
select a default and report it in `explain()`.

```python
from conatus.expert import CartesianCoordinates, Chebyshev

z = CartesianCoordinates("z")["z"]
try:
    Chebyshev(z, 32, (-1, 1))
except ValueError as err:
    print(err)
# Chebyshev: dealias must be given explicitly
#   why:   the dealiasing factor sets the grid of the nonlinear products, so the user chooses it
#   fix:   dealias=1 (no padding, linear problems) or dealias=3/2 (quadratic nonlinearities)
try:
    Chebyshev(z, 32, (-1, 1), dealias=True)
except TypeError as err:
    print(err)
# Chebyshev: dealias=True/False is ambiguous; give the factor (1, 3/2, 2, ...)
```

Linear problems (EVP, LBVP) use `dealias=1`; quadratic nonlinearities need `3/2`. The grid size is
`floor(N * dealias)`. With `grid="smooth7"` the grid is raised to the
smallest size whose FFT length has no prime factor larger than 7, so dealiasing stays exact and
the FFTs avoid large-prime lengths:

```python
import numpy as np
from conatus.expert import CartesianCoordinates, RealFourier

x = CartesianCoordinates("x")["x"]
print(RealFourier(x, 170, (0, 2*np.pi), dealias=3/2).grid_size)                  # 255
print(RealFourier(x, 170, (0, 2*np.pi), dealias=3/2, grid="smooth7").grid_size)  # 256
```

The choice is printed by `basis.explain()` and recorded in the provenance and *Methods* text.

## Cartesian bases

| Basis | Use | Notes |
|---|---|---|
| `RealFourier(x, N, (a, b), dealias=...)` | real periodic data | cos/sin pairs |
| `ComplexFourier(x, N, (a, b), dealias=...)` | complex periodic data, normal modes | $e^{ikx}$; unknowns declared with `dtype=complex` |
| `Chebyshev(z, N, (a, b), dealias=...)` | bounded interval, walls | $T_n$; tau conditions ({doc}`boundary_conditions`) |
| `Legendre`, `Jacobi(z, N, (a, b), a=, b=, ...)`, `Ultraspherical(..., lam=)`, `ChebyshevU`, `ChebyshevV` | other Jacobi families | sparse operators (Olver–Townsend); taus lifted with `lam=` (below) |
| `MappedChebyshev(y, N, map, dealias=...)` | (semi-)infinite intervals | declared map `TanhMap`, `TanMap`, `RationalMap` |

```python
from conatus.expert import CartesianCoordinates, Chebyshev, MappedChebyshev, TanhMap

z = CartesianCoordinates("z")["z"]
y = CartesianCoordinates("y")["y"]
print(Chebyshev(z, 32, (-1, 1), dealias=3/2).explain())
# Chebyshev(z, size=32, bounds=(-1.0, 1.0), dealias=3/2)
#   family         : Chebyshev T (Jacobi -1/2,-1/2)
#   coefficients   : T_n(t), n = 0..31, t = x
#   dealias        : 3/2 (explicit; grid size 48, floor(N*d))

print(MappedChebyshev(y, 96, TanhMap(L=4.0), dealias=1).explain())
# MappedChebyshev(y, size=96, map=TanhMap(L=4), dealias=1)
#   family         : Chebyshev T in a mapped coordinate
#   coefficients   : T_n(xi), n = 0..95, TanhMap(L=4): y = 0 + 4 artanh(xi), dxi/dy = (1 - xi^2)/4, ...
```

**Legendre, Jacobi and ultraspherical axes.** A wall-bounded axis can be expanded in any Jacobi
family $P_n^{(a,b)}$ ($a, b > -1$; Legendre is $a = b = 0$, $C_n^{(\lambda)}$ the ultraspherical
family, and `ChebyshevU`, `ChebyshevV` are $C^{(1)}$, $C^{(2)}$). An equation of
differential order $q$ is written in the family $P^{(a+q,\,b+q)}$, so its $q$ tau terms are lifted
there: `lift(tau, basis, -1, lam=q)`. Every operator is sparse and exact on the retained modes, the
grid is the Gauss–Jacobi grid of the family, and a non-constant coefficient reports its spectral
tail as on a Chebyshev axis. The checks: Poisson with an exact solution to round-off in each family,
the eigenvalues of a string to $10^{-10}$, and a viscous Burgers run equal to the Chebyshev $T$ run
to $5\times10^{-13}$. In the native layer `cn.walls` is Chebyshev $T$ by default, or a sine/cosine series with
`cn.walls(a, b, basis="sincos")` ({doc}`geometries`); these Jacobi families are
for the expert layer and for scripts imported with `cn.from_dedalus` ({doc}`dedalus`). Their GPU path
(`backend="cuda"`) has run on an A100 only ({doc}`gpu`).

Mapped bases put the resolution where a free shear layer or a critical layer is, without
truncating the line to $[-L, L]$: the Taylor–Goldstein example of the gallery reaches
$J_c = 1/4$ to $2.5\times10^{-12}$ this way. `TanhMap(L, contour=...)` also offers a shifted
complex path. The map and its metric are declared and appear in `explain()`.

The helper `channel(N, base="poiseuille" | "couette" | None)` builds the wall-normal Chebyshev
domain of a plane channel with its base flow as an NCC; everything it chooses is visible in
`ch.explain()` (`dealias=1` by default, since linear problems have no products).

## Curvilinear geometries

The same IR, the same `discretize → plan → eigenvalues / solve / IVPSolver` path; the bases are
regular at the axis or centre. There is no $1/r \to 1/(r+\varepsilon)$ and no condition at
$r = 0$.

| Geometry | Coordinates | Bases | Notes |
|---|---|---|---|
| Disk | `PolarCoordinates("phi", "r")` | `RealFourier`/`ComplexFourier` × `DiskBasis(r, N, radius, dealias=...)` | Zernike-type $r^{\lvert m\rvert} w(2r^2-1)$ (Vasil et al. 2016) |
| Annulus, Taylor–Couette | `PolarCoordinates` | × `AnnulusBasis(r, N, (ri, ro), dealias=...)` | Chebyshev in $r$, equations multiplied by $r^p$ |
| Pipe, periodic cylinder | `CylindricalCoordinates("phi", "r", "z")` | Fourier × `DiskBasis` × Fourier | Hagen–Poiseuille = Meseguer & Trefethen (2003) |
| Sphere $S^2$ | `S2Coordinates("phi", "theta")` | `RealFourier` × `SphereBasis(theta, N, radius, dealias=...)` | spin-weighted harmonics (Vasil et al. 2019) |
| Shell, ball | `SphericalCoordinates("phi", "theta", "r")` | × `SphereBasis` × `ShellBasis(r, N, (ri, ro), ...)` or `BallBasis(r, N, radius, ...)` | regular at the centre (ball) |

Taylor–Couette and pipe flow are written with these bases in the expert layer (see
{doc}`../tutorials/08_polar`); the native geometries `cn.Annulus` and `cn.Pipe` choose them for
you ({doc}`geometries`).

## Symmetry subspaces

A `SymmetrySubspace(Reflection(z), T=EVEN, ...)` passed to a problem (`symmetry=`) restricts every
field to a declared parity, and each system shrinks to the size of the subspace. Parities are
checked on every equation; an inconsistent declaration is refused. See
{doc}`../tutorials/03_symmetries` and {doc}`dynamics/index`.

## Sine/cosine walls (native layer)

`cn.walls(a, b, basis="sincos")` discretizes a wall axis with sine and cosine series,
$\sin(n\pi(z-a)/L)$ and $\cos(n\pi(z-a)/L)$, $n < N_z$, $L = b - a$. It suits stress-free
walls, fixed-temperature or insulating walls and hinged plates. Each unknown is a pure sine or
cosine series, so its wall conditions hold identically and **no tau term** is needed. The compiler
derives each unknown's parity from its wall conditions and checks that they are consistent:

| Condition (at both walls) | Series |
|---|---|
| `T=0` | sine |
| `"dz(T) = 0"` | cosine |
| `"dz(dz(f)) = 0"` with `f = 0` (hinged) | sine |
| `u=cn.free_slip` (velocity) | tangential components cosine, normal component sine |
| no condition (pressure) | the unique parity that makes every equation equivariant |

The same model text runs with `basis="chebyshev"` (taus) and `basis="sincos"`. Free-slip
Rayleigh–Bénard onset, at $k_c = \pi/\sqrt{2}$, on both sides of $Ra_c = 27\pi^4/4$:

```{literalinclude} _check_features.py
:language: python
:start-after: "# [sincos]"
:end-before: "# [/sincos]"
:dedent: 4
```

```{literalinclude} outputs/sincos_free_slip.txt
:language: text
:lines: 1-4
```

The closed form $(s/Pr + K^2)(s + K^2)K^2 = Ra\,k^2$, $K^2 = k^2 + \pi^2$, gives
$-0.07420801918986$ and $+0.07383789745827$: the sine/cosine result is exact to round-off (each
group holds one vertical wavenumber, `n=1` selects it; without `n=` all of them are gathered),
the Chebyshev one is spectrally accurate. `m.explain()` declares the choices:

```{literalinclude} _check_features.py
:language: python
:start-after: "# [sincos-explain]"
:end-before: "# [/sincos-explain]"
:dedent: 4
```

```{literalinclude} outputs/sincos_free_slip.txt
:language: text
:lines: 5-8
```

How it is computed: the axis is the reflection-symmetric subspace (about $z = a$) of a
`RealFourier(2 N_z)` series on the doubled period $[a, 2b - a)$, declared on the IR problem as
`SymmetrySubspace(Reflection(z, about=a), ...)`; plans, eigenvalues and `IVPSolver` run in that
subspace (`reduced=True`, set explicitly; `reduced=False` is refused). The dealiasing factor
applies to the doubled grid. `sim.initial(...)` evaluates your function on $[a, b]$ only and
extends it by the parity (data of a sine field that do not vanish at the walls are refused); `sim.grid(name)` and `sim.coordinate("z")` return the $N_z + 1$ points of $[a, b]$;
`sim.coeffs(name)` and `sim.solver` (expert) keep the doubled-period layout.

What is refused, with what to write instead: a non-homogeneous condition (`T=1` at a wall: write
the perturbation about the conduction profile, or use Chebyshev walls), a Robin or coupled
condition, conditions asking for both a sine and a cosine, no-slip velocity (normal and tangential
components would need opposite parities: use `cn.free_slip` or Chebyshev walls), a coefficient or
base state without parity (`m.given(U=lambda z: 1 - z)`: give it as a cosine/sine series or use
Chebyshev walls), a source term of the wrong parity, and `drift=`. This basis needs no tau
variables, so there are no tau modes to filter; check the spectral convergence of the computed
modes instead.

## Whole line, half line, compound intervals

Three more axes for `cn.Box`, each discretized by a fixed basis printed by `explain()`
(IR: `Hermite`, `Laguerre`, `CompoundChebyshev`):

| Axis | Basis | Conditions | Resolution |
|---|---|---|---|
| `cn.line(center=0, scale=s)` | Hermite functions $\psi_n((x-x_0)/s)$, Gaussian decay built in | none (no boundary) | `N_x` |
| `cn.half_line(start=0, scale=s)` | Laguerre functions $L_n(y)e^{-y/2}$, $y=(x-a)/s$ | one wall: `"wall"`, `"bottom"`, `"x=a"` | `N_x` |
| `cn.walls(a, c, ..., b)` | Chebyshev $T_n$ on each piece $[a, c], \dots$ | walls at $a$ and $b$; continuity at each $c$ is **automatic** | one size per piece, `N_x=(N_1, N_2, ...)` |

`scale` is required. It is the width of the basis functions, a numerical choice like $N$; choose
it of the order of the width of the solution. On the half line an
equation of order $q$ is written in the derivative basis $L^{(q)}_n e^{-y/2}$ (bidiagonal
derivative and conversions, the Laguerre analogue of the ultraspherical method) and takes $q/2$
conditions at its wall. On a compound interval, for each unknown of order $q$ the value and the
derivatives up to $q-1$ are continuous at every interface; each piece receives $q$ tau terms in
its last modes, and `explain()` lists every one of them.

```{literalinclude} _check_features.py
:language: python
:start-after: "# [unbounded]"
:end-before: "# [/unbounded]"
:dedent: 4
```

```{literalinclude} outputs/unbounded_compound.txt
:language: text
:lines: 1-4
```

The half-line error is the time error of CNAB2 at `dt = 1e-3` (second order); the spatial error
is below $10^{-9}$ at $N = 40$. What `m.explain(N=(64, 64, 64))` says about the compound layer:

```{literalinclude} _check_features.py
:language: python
:start-after: "# [compound-explain]"
:end-before: "# [/compound-explain]"
:dedent: 4
```

```{literalinclude} outputs/unbounded_compound.txt
:language: text
:lines: 5-
```

When to use which: a compound interval pays off for a **localized** feature (interior layer,
boundary layer, front); for a smooth solution one interval with the same total $N$ is more
accurate (each piece spends $q$ modes on its taus: $e^x$ on $[-1, 1]$ is $10^{-8}$ with
$(8, 8)$ against $10^{-15}$ with 16). A mapped Chebyshev basis (above) remains the choice for a
bounded but non-decaying solution on an infinite interval; Hermite and Laguerre for decaying ones.

What is refused, with what to write instead: a constant along an unbounded axis (a constant
source, tau or gauge: write the problem for the decaying deviation from the far field), a
non-polynomial coefficient on an unbounded axis (`tanh(x)`: make the term explicit or use a mapped
basis), a condition on `cn.line` (no boundary), one integer for a compound axis
(`N_x=(24, 24)`: one size per piece), incompressible flow on these axes (the gauge/divergence tau
recipe is derived for `walls(a, b)` only: use the IR), symmetry subspaces along them, and the
fused GPU kernel (`f_backend="fused"`: the evaluator runs on the GPU instead, transforms by dense
matrices on the device).
