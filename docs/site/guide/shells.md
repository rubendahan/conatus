# Spherical shells

The space between two concentric spheres, where planets and stars convect: simulations on a CPU or
a GPU, rotating convection, discrete adjoints and optimal perturbations, with the same model text as in
a box.

## A shell in a few lines

`cn.Shell(ri, ro)` has the coordinates `phi`, `theta`, `r` and the named walls `inner`, `outer`
and `walls` (both). The resolution is given when you simulate: the number of Fourier modes in
$\varphi$, the number of spherical-harmonic degrees $L_{\max}+1$ in $\theta$, and the number of
Chebyshev modes in $r$.

```{literalinclude} _check_features.py
:language: python
:start-after: "# [shell-native]"
:end-before: "# [/shell-native]"
:dedent: 4
```

```{literalinclude} outputs/shell_native.txt
:language: text
```

`m.at("walls", u=cn.free_slip)` gives stress-free spheres instead ({doc}`boundary_conditions`);
a temperature or any other scalar takes its own values on `inner` and `outer`.

## What Conatus checks before the first step

A common way of writing the wall conditions of a shell in a spectral code creates, at fine radial
resolution and small viscosity, a growing mode that belongs to the discretisation and not to the
flow. The run then blows up after a while, whatever the time step. The native layer writes the
conditions in a form designed to avoid this mode, and checks for it as described below.

Before the first step, Conatus looks for any growing artefact of this kind in the implicit
operator and states the result (the `spurious tau modes` line above, also in `sim.explain()`). A
problem written by hand in the {doc}`expert layer <expert>` that has one is refused with an error
that names the group, the growth per step, and the change that removes it ({doc}`errors`); a mode
that the chosen scheme and time step happen to damp is accepted and declared, since another time
step could make it grow.

## Rotating convection

`cn.problems.rotating_shell_convection` and its linear-onset counterpart
`rotating_shell_convection_evp` give the rotating Boussinesq problem of the Christensen et al.
(2001) dynamo benchmark (non-magnetic), every parameter explicit ({doc}`problems`):

```python
import conatus as cn

r = cn.problems.rotating_shell_convection(Nphi=32, Ntheta=24, Nr=24, eta=0.35,
                                          Ra=100, E=1e-3, Pr=1, dealias=1.5)
print(r.explain())          # equations, units, conditions and reference
```

## On a GPU

`backend="cuda"` runs the same model on the GPU; `f_backend="fused"` plans the nonlinear terms
once (shared terms computed once, products as generated kernels), which is the fast path in a
shell ({doc}`gpu`):

```python
# requires: gpu
import conatus as cn

m = cn.problems.native.rotating_shell_convection(eta=0.35, Ra=100, E=1e-3, Pr=1)
sim = m.simulate((64, 48, 48), scheme="RK443", dt=1e-3, backend="cuda", f_backend="fused")
print(sim.explain())        # what is planned, and which sums change order (rounding only)
```

Each $(m, \ell)$ group of the linear step is one small banded solve, all groups batched together.
A few dense columns that the wall conditions bring into each group are set aside and solved
separately, so the rest stays banded (`border="auto"`, the default; the same matrix, solved
exactly); `border="rows"` turns this off. `explain()` says which groups use it and why the others
do not.

With a Runge–Kutta scheme, `sim.solver.run_graph(n)` captures one whole step as a CUDA graph and
replays it `n` times, also when the right-hand side depends on time: before each
replay the time factors of the step are written to the GPU, so the replayed steps give the same
numbers as `sim.run`, and the two can be mixed. Time-dependent wall data are not captured (use
`sim.run`).

## Gradients, adjoints and optimal perturbations

`sim.flowmap(T)` is the tangent and discrete adjoint of the time stepper over a time `T`, and
`sim.nlop(T=..., E0=...)` the search for the perturbation of given energy that grows most, in the
shell as in a channel, on the CPU and on the GPU: see
{doc}`dynamics/optimal_perturbations` for the shell example and {doc}`dynamics/adjoint` for the
inner product used (the state of a shell is complex, so the pairing is declared by `explain()`).

## Floquet analysis of a periodic shell flow

For a linear problem about an axisymmetric base flow of period $P$, the azimuthal wavenumbers
decouple, and `conatus.floquet.shell_floquet` computes the Floquet multipliers of one wavenumber
$m \geq 1$ from the one-period map of the time stepper itself ($P/\Delta t$ Runge–Kutta steps, on the
CPU or the GPU), by Arnoldi iteration: `shell_floquet(sim.solver, P, steps, m=1, n_modes=4,
tol=1e-10)` on a simulation of the linear problem whose time step divides the period
(`dt = P / steps`); `explain()` prints the exponents, the residual of each multiplier and the number
of map applications.

They are the multipliers of the discrete problem, with the same spatial discretisation and time
stepper as the simulation: they converge to the continuous ones as the number of steps per period
grows, which you check by doubling it. Each multiplier comes with its residual against the true
map, and the coupling of $m$ to other wavenumbers (zero for an axisymmetric base flow) is reported.
Refused with the reason: $m = 0$, a nonlinear right-hand side, a step that does not divide the
period, and multistep schemes (their one-period map depends on the history).

## Not yet

- Several GPUs: one GPU per shell simulation.
- Floquet analysis of the axisymmetric modes ($m = 0$) of a periodic shell flow.
