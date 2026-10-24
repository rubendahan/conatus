# Simulations and time steppers

A model with `dt(...)` is integrated by `m.simulate(...)`, with the resolution, the scheme and the
time step written in the call.

```python
import numpy as np
import conatus as cn

m = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=2500, Pr=1)
sim = m.simulate(N=(64, 32), scheme="RK443", dt=5e-4)       # N, scheme and dt are required
sim.initial(T=lambda x, z: -z + 1e-3*np.cos(np.pi*x)*np.cos(np.pi*z))
sim.run(until=0.1)                                           # or sim.run(steps=200)
sim.grid("T"), sim.coeffs("T"), sim.t, sim.iteration
print(sim.explain())
```

`initial(...)` takes callables of the coordinate names, numbers, explicit data, or arrays of grid
values with exactly the shape of `sim.grid(name)`, for instance the state of another simulation at
the same resolution. Arrays are not broadcast or interpolated: another shape is refused. Unknowns
not given start at zero, and `explain()` says so. An array is read in the layout of `sim.grid`
(components first, then the coordinates in the order of the geometry: `np.meshgrid(x, z,
indexing="ij")`). On a grid with two axes of the same size, the shape cannot tell this layout from
NumPy's default `indexing="xy"` (the transpose). A raw array is then refused unless it was
returned by `sim.grid(...)` or you declare `initial(T=array, layout="ij")`. A callable
`T=lambda x, z: ...` has no such ambiguity and is the preferred form. A parameter changed after
`simulate()` (`m.kappa = 2`) reaches the next `simulate()`, not an existing simulation. Conatus
warns and names the simulations that keep the old value, and `sim.explain()` lists the values in
use. `sim.evaluate("dz(T)")` gives grid values of an
expression written as in the equations; an expression constant in space (`"integ(u@u)/2"`) is
returned as a number. The stop time of `run(until=t)` must be a whole number of time steps from
the current time. Otherwise the call is refused and the message gives the number of steps to use.

Callable initial data are checked before the run starts. The callable is sampled a second time on
a grid twice as fine along each direction in turn, and the part of the data that the resolution
cannot hold is measured (the largest Fourier coefficient at `|k| >= N/2`, or Chebyshev coefficient
of degree `>= N`, over the largest one). Two levels (`cn.tolerances.explain()`):

- above `INITIAL_TAIL_REFUSE = 1e-2` the data are clearly wrong and refused with the direction
  named: `cos(15*x)` at `N = 16` (aliased: on the grid it is exactly `cos(x)`), `sin(2*np.pi*x)` on
  `cn.periodic(2*np.pi)` (not periodic on the box; also refused at any `N` when the values at the
  two ends of a periodic direction differ by more than `1e-2` of the largest value);
- between `INITIAL_TAIL_TOL = 1e-6` and `1e-2` the data are only not fully resolved (smooth data at
  a coarse `N`, e.g. `exp(cos(x))` at `N = 8` in a resolution study): one
  `cn.errors.InitialDataWarning`, repeated by `explain()`; the run starts from the projection.

Values that are not finite (`np.log(x - 1)`) are refused as well. `initial(..., check=False)`
silences both levels and `explain()` says so. A callable that returns different values on each call
(random noise) is checked for finiteness only; arrays and numbers are not checked beyond
finiteness, nor curved geometries and sine/cosine axes (said by `explain()`).

## Schemes

All schemes are implicit–explicit: the linear terms are solved implicitly, the nonlinear terms
are explicit ({doc}`equations`).

| Scheme | Family (`solver.family`) | Implicit part on $L$ / explicit part on $F$ | Order | Reference |
|---|---|---|---|---|
| `SBDF1`–`SBDF4` | multistep (`"sbdf"`) | BDF$q$ / $q$-step extrapolation | $q$ (with the default `startup="lower_order"`: $\min(q, 2)$) | Wang & Ruuth 2008 |
| `CNAB1` | multistep (`"cn"`) | Crank–Nicolson / forward Euler | 1 | Wang & Ruuth 2008, eq. 2.5 with $\gamma = 1/2$ |
| `CNAB2` | multistep (`"cn"`) | Crank–Nicolson / Adams–Bashforth 2 | 2 | eq. 2.9 |
| `MCNAB2` | multistep (`"cn"`) | modified CN (9/16, 6/16, 1/16) / Adams–Bashforth 2 | 2 | eq. 2.10 |
| `CNLF2` | multistep (`"cn"`) | CN over $2\,dt$ / leapfrog | 2 | eq. 2.11 |
| `RK111`, `RK222`, `RK443` | Runge–Kutta (`"rk"`) | L-stable SDIRK / explicit RK | 1, 2, 3 | Ascher, Ruuth & Spiteri 1997 |
| `RKSMR` | Runge–Kutta (`"rk"`) | CN per stage (order 2) / Wray RK3 (order 3) | 2 | Spalart, Moser & Rogers 1991 |
| `RKGFY` | Runge–Kutta (`"rk"`) | Crank–Nicolson / Heun | 2 | — |

Their names are listed by `conatus.expert.SCHEMES`. Each scheme is implemented from its paper.
The tests check the order conditions exactly and measure the observed orders (heat equation,
$u_t=-u^2$ with $L=0$, Burgers/Cole–Hopf). `explain()` prints the line of the table for the chosen
scheme. The CN family keeps a history of $MX$, $LX$ and $F$ (saved in checkpoints); `set_dt` is
refused for every multistep scheme. The ensembles run SBDF and
Runge–Kutta schemes only and refuse the CN family with an explicit `OptionError`.

The observed orders of the Crank–Nicolson family and of the two low-storage Runge–Kutta schemes,
on a heat equation with a nonlinear source (implicit $u_{xx}$, explicit $-u^2$), against a
reference computed with RK443 at a ten times smaller step:

```{literalinclude} _check_features.py
:language: python
:start-after: "# [schemes]"
:end-before: "# [/schemes]"
:dedent: 4
```

```{literalinclude} outputs/schemes.txt
:language: text
```

### Time-dependent boundary data

A boundary condition whose value depends on time ($u(\pm1, t) = g(t)$) is a row of the system
without a time derivative. The moment at which it is imposed changes the order of the schemes, so
it is a declared choice, `bc_time`. `explain()` prints it with the equations it concerns.

| `bc_time` | What a stage imposes |
|---|---|
| `"auto"` (default) | `"rk"` when $g'(t)$ is known exactly, otherwise `"stage"` (the reason is printed) |
| `"rk"` | $g_i = g(t_n) + dt\sum_j H_{ij}\, g'(t_n + c_j dt)$: the implicit tableau applied to $g'$ |
| `"stage"` | $g(t_n + c_i dt)$ at every stage |
| `"step"` | $g$ carried by the explicit part, with the explicit weights |

Moving boundaries do not raise the order of a scheme. A first-order scheme stays first order in
every mode. The table gives the observed orders on one test problem: $u_t = u_{yy}/2$ on
$[0, 5]$ with both walls moving, exact solution $u = e^{-y}\cos(t - y)$, Chebyshev $N = 32$, error
at $t = 1$, for $dt = 0.02 \to 0.01 \to 0.005$ (`tests/ivp/test_bc_time.py`).

| Scheme | `"rk"` | `"stage"` | `"step"` |
|---|---|---|---|
| `RK111` | 1.0 | 1.0 | 1 |
| `RK222` | 2.0 | 2.0 | 1.05–1.2 |
| `RK443` | 2.8 | 2.0 | 1.05–1.2 |
| `RKSMR` | 2.0 | 1.9 | 1.05–1.2 |
| `RKGFY` | 2.0 | 2.0 | not measured |
| `SBDF1`–`SBDF4` | as `"stage"` | 1.0, 2.0, 3.0, 4.0 | not measured |
| `CNAB1` | as `"stage"` | 2.0 (no explicit term here) | 1.0 |
| `CNAB2`, `MCNAB2` | as `"stage"` | 2.0 | `CNAB2`: 1.3 then 0.7 |
| `CNLF2` | as `"stage"` | 2.0 | 2.8 then 1.4 |

Multistep schemes (SBDF, CN family) solve once per step and impose $g(t_{n+1})$ in the first three
modes. Their RK443 start-up substeps follow the mode. $g'(t)$ is the symbolic derivative of the
boundary right-hand side (sums, products, powers, `exp`, `sin`, `cos`, `tanh`, `sqrt`, `log`, any
linear spatial operator). A value without a derivative (`sign(t)`) makes `"auto"` fall back to
`"stage"` with the reason in `explain()`, and an explicit `bc_time="rk"` is then refused. With
`"rk"`, the wall value of the state at $t_n$ is $g(t_n)$ up to the local error of the quadrature
of $g'$. This error does not accumulate, because each step starts from $g(t_n)$. With `"stage"`
the wall value is exact. Constant boundary data give the same numbers in every mode, bit for bit.
Checkpoints record the mode, and a restart in another mode is refused. The tangent and adjoint
maps of `FlowMap` are unaffected, because the boundary rows do not depend on the state. The polar
and spherical solvers follow the same rule.

In the native layer `t` is the time: a boundary value is written as text or as a function of `t`.

```python
import numpy as np
import conatus as cn

m = cn.Model(cn.Box(y=cn.walls(0, 5)), name="moving walls")
m.scalar("u")
m.equations("dt(u) - dy(dy(u))/2 = 0")
m.at("bottom", u="cos(t)")                                # text
m.at("top", u=lambda t: np.exp(-5) * np.cos(t - 5))       # numpy functions of t
sim = m.simulate(N=32, scheme="RK443", dt=1e-2, dealias=1)
print(sim.explain())     # "boundary data: ... (bc_time='rk', default 'auto'; RK443 about order 2.8 ..."
sim_d = m.simulate(N=32, scheme="RK443", dt=1e-2, dealias=1, bc_time="step")   # data in the explicit part
```

In the expert layer the same problem reads

```python
import numpy as np
import conatus.expert as ex

cs = ex.CartesianCoordinates("y")
y = cs["y"]
by = ex.Chebyshev(y, 32, (0, 5), dealias=1)
u, t = ex.Field("u", ex.Domain(cs, [by])), ex.Parameter("t", 0.0)
pb = ex.IVP([u])
pb.add_equation((ex.dt(u) - ex.diff(ex.diff(u, y), y) / 2, 0))
pb.add_bc((ex.interp(u, y, 0), ex.cos(t)), (ex.interp(u, y, 5), np.exp(-5) * ex.cos(t - 5)))
plan = ex.discretize(pb)
s = ex.IVPSolver(plan, "RK443", 1e-2, time_parameter="t")                    # bc_time="auto" -> "rk"
s_s = ex.IVPSolver(plan, "RK443", 1e-2, time_parameter="t", bc_time="stage")  # g at the stage times
```

A time-dependent forcing in the volume equations is evaluated at the stage times by $F$; on a stiff
operator it can reduce the observed order of RK443 too (measured about 1.8 on
$u_t = u_{yy} + f(y, t)$ with $N = 8$, independently of `bc_time`): the classical order reduction of
Runge–Kutta methods on stiff forced problems.


```python
import numpy as np
import conatus as cn

m = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=2500, Pr=1)
sim = m.simulate(N=(64, 32), scheme="SBDF3", dt=1e-3, startup="rk443")
```

Either way the choice is printed by `explain()` and written in the *Methods* paragraph.

## Choosing the time step

`dt` stays at the value you give unless you ask for adaptive stepping. Two indicators help you
choose it, and they answer different questions:

- **CFL** (`sim.cfl("u", safety=0.5)`) is the advective stability limit of the current state,
  $\Delta t_{\rm cfl} = \text{safety} / \max \sum_i |u_i|/\Delta x_i$. It limits the step for
  stability. It does not estimate the temporal error.
- **Temporal error**: `EmbeddedError(sim.solver)` compares one step with a lower-order companion
  on the same data (SBDF$q$ against SBDF$q{-}1$, RK443 against RK222, RKSMR or RKGFY against
  RK111; refused for the CN family, which has no companion) and reports the
  difference. It adapts `dt` only if you ask, with `adapt=True` and an explicit `tol=` and
  `safety=` (Runge–Kutta schemes): a step whose estimate exceeds `tol` is rejected, the state
  restored and the step retried with a smaller `dt`; an accepted step proposes the next `dt`.
  If `tol` cannot be met at `dt_min` (or after 20 rejections in a row) a `ConvergenceError` is
  raised, so no step above `tol` is kept.

```python
import numpy as np
import conatus as cn

m = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=2500, Pr=1)
from conatus.expert import EmbeddedError

sim = m.simulate(N=(64, 32), scheme="RK443", dt=5e-4)
sim.initial(T=lambda x, z: -z + 1e-3*np.cos(np.pi*x)*np.cos(np.pi*z))
e = EmbeddedError(sim.solver)
print(e.step())   # relative difference of one step; dt unchanged
```

## A run that blows up stops

A diverging run (time step too large, an instability the resolution cannot follow) is stopped.
The state is tested for NaN or inf every 100 steps, at the end of each `run` call and before each
checkpoint or output write. A `cn.errors.NumericalError` then names the iteration, the time, the
unknown and the last iteration checked finite, and suggests fixes (smaller `dt`, `sim.cfl`,
higher `N`). One reduction over the state per check (one host synchronisation on
the GPU, none inside a replayed CUDA graph): measured at about 0.1 % of the step cost on one CPU core (Kuramoto–Sivashinsky N = 64 and Rayleigh–Bénard up to 256×64). `check_finite=10` checks more
often, `check_finite=False` turns the check off; `sim.explain()` says which.

```python
import numpy as np
import conatus as cn

m = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=2500, Pr=1)
sim = m.simulate(N=(64, 32), scheme="RK443", dt=1e-3, check_finite=10)
```

## Warnings for under-resolution and ill-posed implicit terms

At each finite-state check the relative spectral tail of every unknown along every direction (the
largest coefficient among the last 10 % of the mode numbers over the largest one, as in
{doc}`resolution`) is measured on the coefficients in memory. Above `RUN_TAIL_TOL = 1e-3` (a ratio of
amplitudes: a tail coefficient then carries about $10^{-6}$ of the energy of the largest one; `1e-4` warned on the resolved flagship run, KS `L = 22`,
`N = 32`, tutorial 5) a `cn.errors.UnderResolvedWarning` names the
unknown, the direction, the tail and the time, once per run, and `sim.explain()` repeats it. The
run itself is not modified. Kuramoto–Sivashinsky on `L = 22` with `N = 8` (where `max|u|` reaches 64 instead of
staying below 3) warns at `t = 1` with a tail of 0.76; at `N = 64` it does not. Cost: about 6 ms for
Rayleigh–Bénard 256×64 on one CPU core, 1.5 % of one step, so about 0.015 % at the default cadence
(on the GPU the small envelopes return in one transfer, at the synchronisation the finite check
already makes). Unknowns that stay at round-off size (below `RUN_TAIL_FLOOR = 1e-12` of the largest
coefficient of the state) are not judged, nor are directions whose tail window holds fewer than
`RUN_TAIL_MIN_MODES = 3` coefficients (a Chebyshev axis with `N <= 20`: one or two top coefficients
are not a tail); curved geometries are not checked. `explain()` says which directions were left out.

When the model is compiled, the principal symbol of each evolution equation is read off its implicit
terms: a term that is a number times derivatives of the evolved unknown (`dx`, `lap`, ...)
contributes `c (ik)^p` along each coordinate. If the leading power gives a growth rate that
increases without bound with the wavenumber, a `cn.errors.IllPosedWarning` names the term and the
sign that would make the problem well-posed, and `explain()` keeps it:

```text
ill-posed: in [eq0] the implicit term kappa*lap(T) makes dt(T) grow like +1 k^2 along x (...);
did you mean -kappa*lap(T) on the left side, that is dt(T) = kappa*lap(T) + ...?
```

Lower-order terms of the destabilising sign are physics and are not reported (the `+lap(u)` of
Kuramoto–Sivashinsky is dominated by `+lap(lap(u))`); a term whose symbol is not read (non-constant
coefficient, `grad`/`div`/`curl`, another unknown) of the leading order skips the check for that
equation.

## Restarting exactly

A checkpoint stores the state and the multistep history, so a restarted run continues bit for bit,
including a restart during the start-up phase:

```python
import numpy as np
import conatus as cn

m = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=2500, Pr=1)
sim = m.simulate(N=(64, 32), scheme="RK443", dt=5e-4)
sim.initial(T=lambda x, z: -z + 1e-3*np.cos(np.pi*x)*np.cos(np.pi*z))
sim.run(steps=20)
sim.save("ck_rb.h5")
sim2 = m.simulate(N=(64, 32), scheme="RK443", dt=5e-4)
sim2.restart("ck_rb.h5")                       # same numbers as an uninterrupted run
sim.run(steps=10); sim2.run(steps=10)
print(np.array_equal(sim.grid("T"), sim2.grid("T")))     # True
```

A checkpoint loaded into a simulation with different numerics (another `dt`, scheme or start-up)
is refused with the difference, and so is a checkpoint of another model, other equations, other
parameter values, other given-field data, another box or another resolution ({doc}`io`); `restart(path, allow=("kappa",))`
continues the state at a new value of a named parameter, declared by `explain()`. Periodic
checkpoints, snapshots and runs split over several jobs: {doc}`io`.

## Watching during a run

`sim.run(steps=n, callback=f, every=10)` calls `f(sim)` every 10 steps (the expert
`sim.solver.run(n, callback=f, every=10)` calls `f(sim.solver)`); the period is part of the run and
has no default. The number of steps and the period must be integers.
`sim.run(..., record=True)` records the run in the registry and returns it ({doc}`runs`).
A callback ends a run early by raising `cn.StopRun` (the Studio's *stop* raises the same); with
`on_stop="return"` the run returns and `sim.stopped` says why, with the default `"raise"` the
exception leaves `run`:

```python
import numpy as np
import conatus as cn

sim = cn.problems.native.kuramoto_sivashinsky(L=22.0).simulate(N=32, scheme="RK443", dt=0.01)
sim.initial(u=lambda x: 0.1 * np.cos(2 * np.pi * x / 22))

def guard(sim):
    if np.max(np.abs(sim.grid("u"))) > 1.0 or sim.t >= 0.5 - 1e-12:
        raise cn.StopRun(f"stopped at t = {sim.t:.2f}")

sim.run(steps=100, callback=guard, every=10, on_stop="return")
print(sim.stopped, sim.iteration)          # stopped at t = 0.50 50
```

To keep a run so that you can find, compare and reproduce it, see {doc}`runs`.

- **Watching a run.** For anything you want to see while the run goes on (fields, curves,
  controls), use the {doc}`studio`. Watching does not change the computed numbers.

## On a GPU

Add `backend="cuda"`: the same model, scheme and time step run on the device ({doc}`gpu`).

```python
# requires: gpu
import numpy as np
import conatus as cn

m = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=2500, Pr=1)
sim = m.simulate(N=(256, 128), scheme="RK443", dt=1e-4, backend="cuda")
```

## Underneath

`sim.solver` is the expert time stepper (`IVPSolver`) and `sim.plan` the discretised problem.
Options of the solver can be passed through `simulate(...)`: `startup=`, `f_backend="fused"`
({doc}`gpu`), `complexify=True` (cos/sin pairs solved as complex systems of half size). The
{doc}`expert layer <expert>` describes them.

In a spherical shell, `border=` chooses how the linear solve of each $(m, \ell)$ group is laid
out (`"auto"` by default, the same matrix solved exactly either way): see {doc}`shells`.
