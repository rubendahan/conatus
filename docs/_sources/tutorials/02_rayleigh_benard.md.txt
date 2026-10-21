# A first initial-value problem: Rayleigh–Bénard 2D, CPU then GPU

**Goal.** Write the Boussinesq equations once, time-step them on the CPU, check that a small
perturbation grows at the rate predicted by the eigenvalue problem of the *same* equations, then
run the same plan on a GPU.
**Expected.** Growth rate $\sigma = 5.5425059$ of the $k = \pi$ roll at $Ra = 2500$, $Pr = 1$
(EVP of the same equations; the streamfunction formulation gives the same value); onset is at
$Ra_c = 1707.762$ (Reid & Harris 1958). CPU and GPU agree to rounding ($\le 10^{-11}$ relative).
**Time:** about 10 s of CPU (2a); about a minute on one GPU, mostly start-up (2b).
**Scripts:** [`t02a_rayleigh_benard_cpu.py`](code/t02a_rayleigh_benard_cpu.py),
[`t02b_rayleigh_benard_gpu.py`](code/t02b_rayleigh_benard_gpu.py)

## The equations

In thermal-diffusion units, $z \in [-1/2, 1/2]$, rigid isothermal walls:

$$
\partial_t \mathbf{u} - Pr\,\nabla^2 \mathbf{u} + \nabla p - Pr\,Ra\,T\,\mathbf{e}_z = -\mathbf{u}\cdot\nabla\mathbf{u},\qquad
\partial_t T - \nabla^2 T = -\mathbf{u}\cdot\nabla T,\qquad \nabla\cdot\mathbf{u} = 0,
$$

with $T = \pm 1/2$ and $\mathbf{u} = 0$ on the walls. They are typed as written: terms may sit on
either side; the linear ones become implicit, the nonlinear ones explicit, and `explain()` lists
the split (override with `implicit(...)` / `explicit(...)`).

## Write the problem

```{literalinclude} code/t02a_rayleigh_benard_cpu.py
:language: python
:start-after: "# [imports]"
:end-before: "# [/imports]"
```

```{literalinclude} code/t02a_rayleigh_benard_cpu.py
:language: python
:start-after: "# [problem]"
:end-before: "# [/problem]"
```

- The geometry names its boundaries (`bottom`, `top`); conditions are given there by unknown.
  `m.explain()` shows the bases and tau terms selected by the native compiler: two taus per second-order equation (modes $N_z-1$, $N_z-2$ of the equation where the unknown has
  its highest $z$-order), the pressure gauge `tau_p` with `integ(p) = 0` and the divergence tau
  `dot(ez, lift(tau_u2, z, -1))`, recognized from the structure ($p$ only under `grad`, `div(u) = 0`
  without time derivative), and the declared default dealiasing 3/2.
- The discrete matrices and right-hand sides are **bit-identical** to the hand-written IR version
  ([`expert/t02a_rayleigh_benard_cpu.py`](code/expert/t02a_rayleigh_benard_cpu.py)), up to the
  order of the unknowns.

## The eigenvalue problem of the same equations

```{literalinclude} code/t02a_rayleigh_benard_cpu.py
:language: python
:start-after: "# [evp]"
:end-before: "# [/evp]"
```

`m.linearize(about=...)` takes the Fréchet derivative of the same model around the conduction state
$T_0 = -z$ and turns $\partial_t$ into the eigenvalue $s$. The perturbation equations come from
this model and the supplied base state.
`eigenvalues(N=24, k=np.pi)` gives 24 Chebyshev modes in $z$ and the smallest Fourier size holding
$k = \pi$ (declared: the matrices of one wavenumber do not depend on the others).

## Time-step, with the error and the CFL kept apart

```{literalinclude} code/t02a_rayleigh_benard_cpu.py
:language: python
:start-after: "# [ivp]"
:end-before: "# [/ivp]"
```

- `m.simulate(N=(16, 24), scheme="RK443", dt=5e-4)`: resolution, scheme and step are explicit. The
  schemes (SBDF1–4, CNAB, RK111/222/443, ...) are listed in {doc}`../guide/timesteppers`.
  `run(until=0.8)` needs a stop time that is a whole number of steps from the current time.
- `sim.cfl("u", 0.5)` is a *stability indicator* built from the velocity. The temporal error
  estimate (`EmbeddedError(sim.solver)`) and everything else of the expert
  `IVPSolver` stay available through `sim.solver`.
- `sim.explain()` shows the automatic decisions, then the systems (one per Fourier wavenumber),
  their bands and how they are solved.

## Measure the growth rate, and write the Methods paragraph

```{literalinclude} code/t02a_rayleigh_benard_cpu.py
:language: python
:start-after: "# [growth]"
:end-before: "# [/growth]"
```

`sim.methods()` writes the "Numerical methods" paragraph of a paper from the plan itself (bases,
sizes, tau method, dealiasing, scheme, step), with the citations it relies on, and states what the
native layer decided.

```{figure} media/t02_growth.png
:alt: The roll amplitude on a log scale against time: the dots bend during a short start, then follow a straight red line over the shaded interval.
:width: 100%

The initial-value problem against the eigenvalue problem. Before t = 0.3 faster modes are still decaying; over the shaded interval the amplitude grows at σ = 5.5425080, the eigenvalue's 5.5425059: a relative difference of 4×10⁻⁷, within the 10⁻⁵ the script checks.
```

## A run over several jobs

A cluster job has a time limit, so a long run is a chain of jobs, and each job runs the same
script. `sim.resume_or_start(directory, T=...)` continues from the latest checkpoint of
`directory` when there is one and starts from the initial state otherwise; `checkpoints` writes
bit-for-bit restart files; `output` writes the fields and numbers you ask for. Without `mode=`, the
output file is created on a fresh start and continued after a restart. On a fresh start an existing
file is refused; give `mode="w"` to replace it. After a restart, rows written after the checkpoint
are dropped and written again by the restarted run.

```{literalinclude} code/t02a_rayleigh_benard_cpu.py
:language: python
:start-after: "# [jobs]"
:end-before: "# [/jobs]"
```

`explain()` says which branch was taken and which mode each file got. The chain gives the state of
the uninterrupted run above bit for bit (checked below), because a checkpoint holds the state, the
scheme's history and the clock. More in {doc}`../guide/io` (what an analysis file holds, `mode=`,
the formats).

**Check.** EVP: $\sigma = 5.5425059231$ (the streamfunction formulation gives $5.542505923$); IVP from a
$10^{-6}$ perturbation: $\sigma = 5.5425080$ over $t \in [0.3, 0.8]$, i.e. $4\times10^{-7}$ relative.

```{figure} media/t02_restart.png
:alt: Two horizontal bars over time: the upper one in two parts, job 1 then job 2, joined at a checkpoint marker at t = 0.3; the lower one a single uninterrupted run to the same end time.
:width: 100%

The jobs block of the page: job 2 resumes from the checkpoint that job 1 wrote at iteration 600 (t = 0.3) and runs to t = 0.8 (iteration 1600). Its final state is identical to that of the uninterrupted run: np.array_equal is True over the 1531 coefficients, largest difference 0. The output file holds 17 rows, t = 0 included.
```

```{literalinclude} code/t02a_rayleigh_benard_cpu.py
:language: python
:start-after: "# [check]"
:end-before: "# [/check]"
```

### Output (as executed)

```{literalinclude} outputs/t02a_rayleigh_benard_cpu.txt
:language: text
```

## The same run on a GPU

The model does not change. Only the backend is chosen, explicitly: `backend="cuda"` in
`m.simulate(...)` moves the
state, the batched banded solves and the transforms to the device; `f_backend="fused"` evaluates the
nonlinear terms with fused kernels (the NumPy `"evaluator"` remains the oracle). `sim.solver.run_graph(n)`
captures one step as a CUDA graph and replays it, which removes the launch latency.

The GPU script starts from the same model, at $Ra = 10^5$. Its last three blocks need a CUDA device.

```{literalinclude} code/t02b_rayleigh_benard_gpu.py
:language: python
:start-after: "# [imports]"
:end-before: "# [/imports]"
```

```{literalinclude} code/t02b_rayleigh_benard_gpu.py
:language: python
:start-after: "# [problem]"
:end-before: "# [/problem]"
```

```{literalinclude} code/t02b_rayleigh_benard_gpu.py
:language: python
:start-after: "# [backends]"
:end-before: "# [/backends]"
```

```{literalinclude} code/t02b_rayleigh_benard_gpu.py
:language: python
:start-after: "# [graph]"
:end-before: "# [/graph]"
```

```{literalinclude} code/t02b_rayleigh_benard_gpu.py
:language: python
:start-after: "# [check]"
:end-before: "# [/check]"
```

CPU and GPU are not bitwise equal (different summation orders), so the check is agreement to
rounding, the cross-backend model of the test suite, and it is made **early** (20 steps). At
$Ra = 10^5$ the conduction state is violently unstable: the linearized problem has growth rates up
to $\sigma \approx 225$ (wavenumber $2\pi$), and the flow saturates nonlinearly within the 200
steps. Any rounding difference is amplified with it: after 200 steps two *CPU* evaluations of the
same nonlinear terms (NumPy evaluator vs fused kernels) differ by $4\times10^{-6}$, as do CPU and
GPU. Comparing trajectories of an unstable flow needs a
time horizon, stated with the tolerance.

**Check.** After 20 steps CPU and GPU (fused kernels) agree to $1.0\times10^{-14}$ relative; the CUDA-graph
replay equals plain GPU steps exactly.

### Output (as executed, one L40S)

```{literalinclude} outputs/t02b_rayleigh_benard_gpu.txt
:language: text
```

## Going further

- Half the unknowns with a reflection symmetry: {doc}`03_symmetries`.
- Checkpoints and exact restart (multistep history included): {doc}`../guide/reproducibility`.
