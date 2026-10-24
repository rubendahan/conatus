# IR and operators

The expert layer writes problems with Python objects: fields, parameters and operators that build an exact symbolic expression.

A Conatus script is written in a small symbolic language, the **IR** (intermediate representation):
fields, parameters and operators build an immutable, hashed expression tree. Nothing is discretized
until you call `discretize`. Everything below comes from a single import:

```python
from conatus.expert import *        # IR, bases, problems, solvers, time stepping, dynamics toolkit
```

`conatus.expert` re-exports the documented public objects of the user-facing modules.
`conatus.expert.explain()` lists the name clashes and how each is handled (see {doc}`explain`). Helpers and result types that the guide does not show stay in their module
(`conatus.<module>.<name>`).

## Fields and parameters

```python
import numpy as np
from conatus.expert import (CartesianCoordinates, Chebyshev, Domain, Field, Parameter,
                            RealFourier, TensorField, VectorField)

cs = CartesianCoordinates("x", "z")
x, z = cs.coords
D = Domain(cs, [RealFourier(x, 64, (0, 2 * np.pi), dealias=3/2),
                Chebyshev(z, 32, (-1, 1), dealias=3/2)])
u = VectorField("u", D)                 # one component per coordinate of cs
S = TensorField("S", D, 2)              # rank 2: one component per pair of coordinates
T = Field("T", D)                       # scalar; dtype=complex for complex unknowns
Ra = Parameter("Ra", 1e5)               # named scalar; its value is not part of the hash
c = Parameter("c", dtype=complex)       # an eigenvalue is a Parameter too
```

A `Field` carries no data: it names an unknown (or a non-constant coefficient, an **NCC**, whose
values you pass to `discretize(..., ncc={U: lambda y: 1 - y**2})`). Data live in the solvers
(`solver.grid("T")`, `solver.coeffs("T")`) or in an explicit `FieldData`.

An NCC must be resolved on the grid: if the relative Chebyshev tail of its sampled values exceeds
`ncc_tail_tol` (default `conatus.tolerances.NCC_TAIL_TOL` $= 10^{-8}$, the tail
{doc}`resolution` calls *resolved*), `discretize` refuses it with the resolution predicted to
resolve it. To accept it as it is, declare it: `ncc_cutoff=1e-6` drops its coefficients below
$10^{-6}\max|a_k|$, `ncc_tail_tol=1e-6` keeps them all; `plan.explain()` reports either. The
native calls take the same arguments (`m.simulate(N, ..., ncc_cutoff=1e-6)`, `m.eigenvalues`,
`m.solve`, `lin.eigenvalues`). Coefficient data (`FieldData(..., "coeff")`) is used as given and
is not refused. A tail below `conatus.tolerances.NCC_ROUNDING` ($10^3$) times the machine epsilon times the
scale of the data is the rounding of the coefficient's own computation, not under-resolution, and
is accepted.

**Residual digits and coefficient digits.** A Newton or eigenvalue residual of $10^{-12}$ next to
an accepted NCC tail of $10^{-9}$ is not a contradiction: the residual measures how well the
*discrete* problem is solved, with each coefficient as represented by its truncated Chebyshev
series; the measured coefficient tail estimates how far that representation is from the
coefficient you gave. The answer
of the continuous problem is known to about the larger of the two (times the conditioning of the
problem), not to the residual alone. `NLBVPSolution.explain()` and
`LBVPSolution.explain()` print an *accuracy floor* line with the largest sampled NCC tail
(`conatus.tolerances.ncc_floor_text`). The refusal level $10^{-8}$ is where the coefficients of the
suite fall either clearly resolved ($10^{-10}$ to $4\times10^{-9}$) or clearly not ($10^{-7}$ and
above); a computation that needs more digits declares a smaller `ncc_tail_tol` and a resolution
that reaches it.

## Operators

| Operator | Meaning | Notes |
|---|---|---|
| `diff(f, x)` | $\partial_x f$ | zero if `f` does not depend on `x` |
| `grad(f)`, `div(u)`, `lap(f)`, `curl(u)` | vector calculus in the domain's coordinate system | Cartesian, polar, cylindrical and spherical systems |
| `dt(f)` | time derivative | LHS of an `IVP`; becomes $\sigma f$ under `linearize` |
| `integ(f)`, `integ(f, x)` | integral over all or some coordinates | `integ` of a constant is refused (ambiguous length) |
| `interp(f, z, "left")`, `interp(f, z, 1.0)` | trace at a point | boundary conditions |
| `lift(tau, basis, -1)` | tau term: `tau` times the polynomial $N-1$ of `basis` | see {doc}`boundary_conditions` |
| `dot`, `cross`, `outer`, `comp(u, i)`, `unit_vector(z)`, `conj` | algebra | `conj` of an unknown on the LHS gives a declared doubled system; `cross(W, u)` on the LHS of a Cartesian problem with a constant vector `W` (unit vectors times numbers or parameters: the Coriolis term `2*Om*cross(ez, u)`) is implicit, a non-constant `W` is refused |
| `exp`, `log`, `sin`, `cos`, `tanh`, `sqrt`, `absolute`, `sign` | pointwise functions | evaluated on the dealiased grid; non-polynomial functions are *not* alias-free and `explain()` says so |

In Cartesian coordinates named `x`, `y`, `z` the cross product uses the physical frame
$e_x \times e_y = e_z$ whatever the order in which the axes are declared (other names: the order of
declaration is taken as right-handed). An implicit Coriolis term couples the velocity components in
each group of the banded solve and stays stable at any $\Omega\,\Delta t$, but the L-stable implicit
stages damp the inertial waves of frequency $2\Omega\cdot k/|k|$ when $2\Omega\,\Delta t \gg 1$
(RK443: 1.7 % of the amplitude per step at $\omega\,\Delta t = 1$, 13 % at 2): stability is not
accuracy, resolve the rotation period to follow the waves. `explain()` gives $|W|\,\Delta t$ for
every implicit rotation term, and above 1 the solver warns once (`UnderResolvedWarning`).

```{literalinclude} _check_features.py
:language: python
:start-after: "# [coriolis]"
:end-before: "# [/coriolis]"
:dedent: 4
```

```{literalinclude} outputs/implicit_coriolis.txt
:language: text
```

Each node knows its differential order in every coordinate, its parity under a reflection, its
Fréchet derivative and its formal transpose. `expr.explain()` prints them:

```python
import numpy as np
from conatus.expert import (CartesianCoordinates, Chebyshev, Domain, Field, RealFourier, diff,
                            exp)

cs = CartesianCoordinates("x", "z")
x, z = cs.coords
D = Domain(cs, [RealFourier(x, 64, (0, 2 * np.pi), dealias=3/2),
                Chebyshev(z, 32, (-1, 1), dealias=3/2)])
T = Field("T", D)
print((exp(T) * diff(T, x)).explain())
# Expression : exp(T)*dx(T)
#   node     : Mul
#   tensor   : scalar
#   domain   : Domain[RealFourier(x, 64) x Chebyshev(z, 32)]
#   dtype    : float64
#   ...
#   diff. order: x:1, z:0; time order: 0
```

## Problems

Four problem classes share one interface: a list of unknowns, then `add_equation((lhs, rhs))`.

| Class | Form | Solved by |
|---|---|---|
| `LBVP` | $L X = F$ | `plan.solve(...)` |
| `EVP` | $(L + \sigma M) X = 0$, eigenvalue a `Parameter` appearing linearly | `eigenvalues(plan)`, `ParametricEVP`, `critical` ({doc}`../tutorials/01_orr_sommerfeld`) |
| `IVP` | $M\,\partial_t X + L X = F(X, t)$ | `IVPSolver` ({doc}`timesteppers`, {doc}`../tutorials/02_rayleigh_benard`) |
| `NLBVP` | $G(X) = 0$ | `NewtonSolver`, continuation ({doc}`../tutorials/04_bratu`) |

The left-hand side must be linear in the unknowns. A nonlinear left-hand side raises
`NonlinearLHSError`; write the intended explicit term on the right-hand side. The RHS of an IVP is
treated explicitly (IMEX).

```python
from conatus.expert import (IVP, CartesianCoordinates, Chebyshev, Domain, Field, Parameter, diff,
                            dt, interp, lap)

cs1 = CartesianCoordinates("x"); x1 = cs1["x"]
D1 = Domain(cs1, [Chebyshev(x1, 24, (-1, 1), dealias=3/2)])     # a 1D Chebyshev domain
u1 = Field("u", D1)
a = Parameter("a", 1.0)
pb = IVP([u1], name="heat 1D")
pb.add_equation((dt(u1) - lap(u1), -a * u1))
pb.add_bc(interp(u1, x1, "left"), interp(u1, x1, "right"))
print(pb.explain())                                    # equations, taus, count analysis
```

## Linearization and transposes

`linearize(ivp, around={u: U, ...}, eigenvalue=sigma)` (also `ivp.linearize(...)`) differentiates
every equation, boundary condition and gauge symbolically and replaces `dt` by `sigma`: the
result is an `EVP` whose eigenvalue is the temporal growth rate. Base states are fields (NCCs);
perturbation unknowns are named `d<name>`. The symmetry of the nonlinear problem is *not*
inherited (a perturbation may break it): pass `symmetry=` explicitly
({doc}`../tutorials/03_symmetries`). At the expression level:

```python
import numpy as np
from conatus.expert import (CartesianCoordinates, Chebyshev, Domain, Field, RealFourier, diff,
                            exp)

cs = CartesianCoordinates("x", "z")
x, z = cs.coords
D = Domain(cs, [RealFourier(x, 64, (0, 2 * np.pi), dealias=3/2),
                Chebyshev(z, 32, (-1, 1), dealias=3/2)])
T = Field("T", D)
print((T * diff(T, x)).linearize({T: T}, {T: Field("dT", D)}))
# dx(T)*dT + dx(dT)*T
```

`expr.transpose(u, w)` returns the formal transpose of the linear map `u -> expr` with the
boundary term of the integration by parts:

```python
from conatus.expert import (IVP, CartesianCoordinates, Chebyshev, Domain, Field, Parameter, diff,
                            dt, interp, lap)

cs1 = CartesianCoordinates("x"); x1 = cs1["x"]
D1 = Domain(cs1, [Chebyshev(x1, 24, (-1, 1), dealias=3/2)])     # a 1D Chebyshev domain
u1 = Field("u", D1)
w = Field("w", D1)
tr = diff(diff(u1, x1), x1).transpose(u1, w)
print(tr.adjoint)    # -dx(-dx(w))
print(tr.boundary)   # (dx(u)*w)(x=1) + (-dx(w)*u)(x=1) - (dx(u)*w)(x=-1) - (-dx(w)*u)(x=-1)
```

The pairing is bilinear, without conjugation. The physical adjoint, below, also needs the inner
product: it is $W^{-1}$ times the conjugate transpose times $W$.

## Left eigenvectors and the physical adjoint

The adjoint of an operator depends on the inner product. `InnerProduct(plan, weights={...})` is the
weighted $L^2$ product $\sum_v w_v \int \bar v_1 v_2\,dV$ on the field unknowns of a group, with
exact Gram matrices (closed-form $\int T_m T_n$ for Chebyshev, exact Fourier norms); tau and
constant unknowns get weight 0. Its `adjoint(A)` is

$$ A^\dagger = W^{-1} A^{H} W , $$

refused when $W$ is singular, which is the case as soon as the group has tau columns:

```text
ValueError: Gram matrix is singular (tau or zero-weight unknowns): the adjoint for this inner
product is not defined on the full column space. Use left eigenvectors (solve_dense(..., left=True))
for eigenvalue sensitivities, or a tau-free (Galerkin) formulation.
```

Eigenvalue sensitivities do not need $W$: they use left eigenvectors,
$\mathrm d\sigma/\mathrm dp = -y^H(\partial_p L + \sigma\,\partial_p M)x / (y^H M x)$, computed by
`eigenvalues(plan, left=True)`, `ParametricEVP.sensitivity` and checked by a Taylor test in
`verify` ({doc}`../tutorials/01_orr_sommerfeld`).

For time-dependent problems, `FlowMap` gives the tangent (`jvp`) and discrete adjoint (`vjp`,
`vjp_physical`) of the discrete time-stepping map itself, obtained by transposing the IR and the
IMEX step, and checked by dot-product and Taylor tests ({doc}`dynamics/index`); on the
spherical shell and for optimal perturbations see {doc}`dynamics/optimal_perturbations`.
