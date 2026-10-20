# AGENTS.md — Conatus for AI agents (read before writing any Conatus code)

Conatus: spectral PDEs and dynamical systems on CPU and GPU (v0.1, API frozen; additions only).
You write equations as on paper; Conatus discretises (Fourier/Chebyshev/curvilinear bases, taus
placed for you), integrates, solves eigenproblems and finds structure (orbits, bifurcations, Floquet, optimal perturbations).
It never guesses and never changes a numerical choice silently: it refuses, and says what to write.

## 1. Install, verify, orient (machine-readable)

```bash
pip install ".[tutorials]"             # or installed (Python 3.11-3.13; h5py + matplotlib); writes only under ~/.cache/conatus
python -m conatus doctor --json        # what this machine can run; exit 1 if something required is missing
python -m conatus agent                # JSON: versions, CPU/GPU, user-layer names, watched runs
python -m conatus agent explain cn.Model.simulate     # signature, required args, required tolerances (+ unknowns of a ready model)
python -m conatus agent check run.py   # dry run (no solve): refusals + fix, missing tolerances, memory
```

## 2. Mental model

- `import conatus as cn`: **native layer**. Geometry → `cn.Model` (unknowns, parameters, equations
  as text, conditions by boundary name) → numerics given at the call (`N`, `scheme`, `dt`,
  `backend`) → result object with `.explain()`.
- `import conatus.expert as X`: **expert layer** (IR: bases, fields, `X.IVP`/`X.EVP`/…,
  `X.discretize` → `X.Plan`, `X.IVPSolver`). Every model compiles to it: `m.compile(N).problem`,
  `sim.solver`, `sim.plan`. A name present in both layers has one meaning.
- One word per notion: `steps` (number of time steps), `k` (a wavenumber keyword; a problem
  parameter keeps its literature name, e.g. `alpha` of Orr–Sommerfeld), `eta` (radius
  ratio), `n_modes`, `max_iterations`, `tol` (exceptions until v0.2: the fields `ContinuationSettings(tolerance=, max_newton=)`, also read as `cfg.tol`, `cfg.max_iterations`);
  former names warn (`conatus._deprecation.RENAMES`).
- Names below: `m` = `cn.Model`, `sim` = `cn.model.Simulation`, `lin` = `cn.model.Linearization`, `X` = `conatus.expert`.

## 3. Minimal working example (executed by the test suite)

```python
import numpy as np
import conatus as cn

m = cn.Model(cn.Box(x=cn.periodic(22.0)), name="Kuramoto-Sivashinsky")
m.scalar("u")
m.equations("dt(u) + lap(u) + lap(lap(u)) = -u*dx(u)")
sim = m.simulate(N=32, scheme="RK443", dt=0.01)        # numerics explicit, never defaulted
sim.initial(u=lambda x: np.cos(2 * np.pi * x / 22.0))
sim.run(steps=20)                                     # run(until=T) must land on a whole step
print(sim.explain())                                  # short summary; full=True: every decision

os_ = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="Orr-Sommerfeld", Re=1e4, alpha=1.0)
os_.scalar("v")
os_.given(U=lambda y: 1 - y**2)
os_.eigenvalue("c", growth="imag")
os_.define("L2(f) = dy(dy(f)) - alpha**2*f")
os_.equations("L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + c*L2(v) = 0")
os_.at("walls", "v = 0", "dy(v) = 0")
res = os_.eigenvalues(N=64, drift=96, drift_tol=1e-6)  # drift test removes spurious modes
i = int(res.order()[0])                               # least stable first (growth = Im c)
assert res.residuals[i] < 1e-12                       # validate by the residual, not the value
```

Critical point of a ready linearization, and its mode (Rayleigh–Bénard, no-slip walls):

```python
import numpy as np
import conatus as cn
import conatus.expert as X

lin = cn.problems.native.rayleigh_benard_evp(k=3.1, Ra=1708.0, Pr=1.0)
P = lin.parametric(24, over=("Ra", "k"), drift=32, drift_tol=1e-6)   # k: the wavenumber
crit = X.critical(P, "Ra", bracket=(1500.0, 2000.0), over={"k": (2.5, 3.8)}, xtol=1e-9)
assert abs(crit.value - 1707.762) < 1e-3 and abs(crit.over["k"] - 3.1163) < 1e-3
kc = crit.over["k"]
res = cn.problems.native.rayleigh_benard_evp(k=kc, Ra=crit.value, Pr=1.0).eigenvalues(
    24, k=kc, drift=32, drift_tol=1e-6)
md = res.mode(int(res.order()[0]))
T = md.at("T", z=np.linspace(-0.5, 0.5, 101))[0]     # exact, walls included; row x = 0
```

## 4. The surface, by task (one minimal call each)

| Task | Call |
|---|---|
| Geometries | `cn.Box(x=cn.periodic(L), z=cn.walls(a, b))`, `cn.Channel(Lx, Lz)`, `cn.Disk(R)`, `cn.Annulus(ri, ro)`, `cn.Pipe(R, L)` (= `cn.Cylinder`), `cn.TaylorCouette(ri, ro, L)`, `cn.Sphere(R)`, `cn.Ball(R)`, `cn.Shell(ri, ro)`, `cn.MappedBox(x=cn.periodic(L), z=cn.between(b, t))` (wavy walls, `b`/`t` numbers or `lambda x: ...`) |
| Axes | `cn.periodic(L, complex=False)`, `cn.walls(a, b)`, `cn.walls(a, c, b)` (compound: its entry of `N` is a tuple `(N1, N2)`), `cn.walls(a, b, basis="sincos")`, `cn.half_line(0, scale=s)`, `cn.line(0, scale=s)` (`scale` required) |
| Unknowns, data | `m.velocity("u")`, `m.pressure("p")` (also the pressure of a normal-mode model on one wall axis: `1j*k*p`, `dz(p)`), `m.scalar("T")`, `m.vector("B")`, `m.given(U=lambda y: ...)`, `m.eigenvalue("c", growth="imag")` (declare how a mode grows: `"real"` for exp(s t), `"imag"` for a phase speed; left out: v0.1 guesses from the name with a `DefaultChoiceWarning`), `m.define("L2(f) = ...")`, `m.function("g", f, derivative=df)` |
| Equations | `m.equations("dt(u) + grad(p) - lap(u)/Re = -u·grad(u)", "div(u) = 0")`; ops `dt dx..dz grad div lap curl integ interp`, `·`/`@` dot; override split with `explicit(...)`/`implicit(...)`; `cross(ez, u)` (Cartesian frame e_x x e_y = e_z; with a constant vector on the left-hand side it is implicit, e.g. Coriolis `2*Om*cross(ez, u)`: stable at any dt but its waves are damped when |W| dt > 1 — `explain()` gives |W| dt, `UnderResolvedWarning` above 1); parse only: `cn.parse_equation`, `cn.parse_expr` |
| Boundary conditions | `m.at("bottom", u=0, T=1)`, `m.at("walls", "v = 0", "dy(v) = 0")`, `m.at("top", u=cn.free_slip)`, `cn.no_slip`, `cn.navier_slip(l)`, time-dependent `m.at("top", u="cos(t)")`; force a tau target: `into=` |
| Inspect before running | `m.explain(N)`, `m.compile(N)` (`.problem`, `.fields`, `.decisions()`), `m.discretize(N)` → `X.Plan` (`.explain()`, `.rank_report()`, `.check()`); same with `lin.compile(N)`, `lin.discretize(N, k=k)`, `lin.explain(N)`, `lin.group(plan, k)` |
| Expert problems | `X.IVP`, `X.EVP`, `X.LBVP`, `X.NLBVP` (`add_equation("...")`, `add_bc`), `X.discretize(problem, ncc=)` = `cn.discretize`, `X.IVPSolver(plan, "RK443", dt)` (`sim.solver`), `X.CFL`, `X.EmbeddedError`, `X.TauWatch`; `cn.linearize` (IR linearisation), `cn.model`, `cn.expert`, `cn.translate` |
| Parameters | `m.<name> = value` (e.g. Ra) or `m.params["Ra"] = 1e5` (declared names only, validated; the next compile uses it); for one call: `m.simulate(N, ..., Ra=3000)` (also `eigenvalues`, `solve`, `discretize`; printed by `sim.explain()`) |
| Time integration | `sim = m.simulate(N, scheme="RK443", dt=1e-3, backend="cpu")` (schemes: `X.SCHEMES`; NaN/inf stops the run with `cn.errors.NumericalError`: `check_finite=` cadence, default every 100 steps, `False` = off), `sim.initial(T=f)` (`f` a callable, a number, or an array shaped as `sim.grid("T")`; `t=t0` starts the clock at `t0`: forcing, boundary data, outputs and checkpoints on that clock, `resume_or_start(..., t=t0)` refuses a chain started at another time), `sim.run(until=T)` / `sim.run(steps=n, callback=f, every=k)`, `sim.t`, `sim.iteration`, `sim.grid("T")`, `sim.coordinate("z")`, `sim.coeffs("T")`, `sim.evaluate("dz(T)")` (`"integ(T)"` → a number), `sim.properties(every=k, ...)`, `sim.cfl("u", safety=0.5)`, `sim.dt` |
| Boundary-value | `m.solve(N)` (LBVP); nonlinear: `X.NewtonSolver(m.compile(N).problem, tol=1e-10)` `.solve(guess)` (`damping=1.0`, `max_iterations=50` defaulted, printed) |
| Eigenvalues | `m.eigenvalues(N, drift=N2, drift_tol=1e-6)` → `.values`, `.residuals`, `.explain()`, `res.order()` (least stable first by the growth convention), eigenfunction `md = res.mode(i)` → `md["v"]` (complex grid values, taus left out, declared normalisation), `md.coordinate("y")`, `md.at("v", y=pts)` (exact, walls included); modes whose eigenvector has its energy in the last coefficients (spurious tau or unresolved) are flagged by a spectral-tail test on by default (`res.flagged`, listed by `explain()`, ranked last by `order()`; `spurious="none"` turns it off); without `drift=` the selected mode `order()[0]` is also held to `cn.tolerances.UNRESOLVED_TAIL_TOL` (1e-10): above it `res.unresolved_note`, `explain()` and a `ResolutionWarning` say "selected mode under-resolved" — raise N or pass `drift=`; `drift=` is the stronger filter that removes them (`drift` > N, `drift_tol` in (0, 1e-2], refused otherwise; a test that removes every mode raises `ResolutionError`); a model with a periodic axis needs `m.eigenvalues(N, k=wavenumber)` (without it only k = 0 is solved, `cn.errors.GroupWarning`; `k` is always a wavenumber; `n_modes=` counts shift-invert modes; curvilinear: `m=`, `kz=`, `ell=`); `lin = m.linearize(about={u: 0, T: lambda z: -z})` (its steady residual is in `lin.explain()`; a base that is not a steady solution warns, `cn.errors.BaseStateWarning`, unless `base_residual="accept"`), `lin.eigenvalues(N, k=kx)`; `left=True` for adjoint modes; expert `X.eigenvalues(plan, drift_plan=plan2, drift_tol=1e-6)` |
| Eigenvalues in a region | `res = m.eigenvalues(N, region=cn.box(re=(a, b), im=(c, d)))` → every eigenvalue in the box, `res.count` (certified by the argument principle; an eigenvalue on the boundary is refused), `res.count_method`, `res.multiplicity`; any operator: `cn.contour.count(T, box)` (`T` = `(L, M)`, a `HillBlocks` or `z -> T(z)` with `poles=`), `cn.contour.beyn(T, box, poles=0, derivative=dT)` (nonlinear `T(z) v = 0`, roots certified by residual `<= 1e-13`) |
| Parametric, thresholds | `P = m.parametric(N, over=("Re", "alpha"), drift=N2, drift_tol=1e-6)` (drift: spurious tau modes rejected); ready linearizations: `lin.parametric(N, over=("Ra", "k"), drift=N2, drift_tol=1e-6)` (`k` = the wavenumber, `lin.wavenumber_names()`), `X.critical(P, "Re", bracket=(a, b), xtol=1e-9)` (the growth must change sign between `a` and `b`: checked at both ends, else refused with the sub-brackets where it does; a neutral curve can have two branches, e.g. Orr–Sommerfeld at alpha = 1 between Re = 5000 and 40000: bracket each; growth from the eigenvalue's declaration `m.eigenvalue("c", growth="imag")`, `lin`: Re s; only modes passing the spectral-tail test are selected; without a drift pencil the tail of the critical mode is printed and an under-resolved one warns (`crit.unresolved_note`); missing parameters take the model's values), `P.least_stable(...)`, `P.track(...)`, `P.sensitivity(mode, "Re")` |
| Transient growth | `lin.optimal_gain(N, k=k, t0=0, T=T, steps=n, scheme="RK443")` → `.G`; `X.max_growth(ss, (t1, t2), xtol=)` (`n_scan=64` declared default), `X.optimal_growth`, `conatus.growth.growth_at` |
| Floquet / Hill (periodic base) | `m.linearize(about={u: base(y, t)}, period=1.0)`, `lin.floquet(N, k=k, method="hill", Nh=16, targets=[0.0], tol=1e-12)` (`fl.harmonic_tails`; a leading mode with more than `cn.tolerances.HILL_TAIL_TOL` of its weight in the outermost harmonics warns "truncated in Nh": raise Nh) or `method="monodromy", steps=, scheme=`; `lin.harmonics`, `lin.periodic_operator(N, k=k)`; `X.HillBlocks`, `X.shell_floquet` |
| Invariant solutions (ECS) | `nk = X.NewtonKrylovHookstep(tol=1e-12)` (algorithm controls defaulted, printed; trust radii absolute by default, `radii="relative"` for states far from norm 1); **1D periodic problems only** (one real Fourier axis, e.g. KS): `X.find_equilibrium(sim, x0, nk)` (odd subspace), `X.find_travelling_wave(sim, x0, c0, nk)`, `X.find_relative_periodic_orbit(sim, x0, T0, ell0, nk, steps=)`; the seed `x0`: `sim.grid("u")` (grid values, converted and declared in the notes), the simulation itself (its current state) or the state vector `sim.solver.X`; a search converges only from a seed near the solution (most seeds of a chaotic run fall back to an equilibrium or fail: tutorial 5); walls / 2D / 3D: `conatus.ecs.steady.AffineSteadyProblem`, `conatus.ecs.pipe.PipeTravellingWave`; stability `X.equilibrium_eigenvalues(sim, tw, n_modes, arnoldi_tol=1e-12)`, `X.floquet_multipliers(sim, rpo, n_modes, arnoldi_tol=1e-12)` (a `Simulation` or its solver, as the finders); edge: `X.EdgeTracker`, `X.ThresholdBisection` |
| Continuation, bifurcation | `X.ContinuationSettings(ds=, ds_min=, ds_max=, tolerance=, max_points=)` (`adapt`, `fast_iterations`, `max_newton`, `dp_fd="auto"` defaulted), `X.NLBVPContinuation(newton, "lam", settings).run(...)`, `X.continue_equilibria(X.ir_system(X.ode_problem(...), ["r"]), x0, "r", cfg, marginal_tol=)`, `X.continue_orbits`, `X.branch_switch`, `X.normal_form`, `X.continue_codim2` |
| Manifolds, slices, chaos | `X.ecs_system(sim.solver, odd=, degree=, dense_max=)`, `X.unstable_manifold`, `X.FirstFourierModeSlice`, `X.lyapunov_spectrum`, `X.Ensemble`, `X.ensemble_fate`, `X.poincare_section` |
| Lyapunov, covariant vectors | `res = sim.lyapunov(k, transient=, align=, T=, reortho_every=10, inner="euclidean", n_blocks=, seed=, starts=[{...}], zero_mean="u", covariant=True, clv_window=, clv_backward=, clv_checkpoints=s)` (durations: whole steps/intervals, never rounded; exact tangent of the step) → `res.exponents`, `res.stderr`, `res.kaplan_yorke()`, `res.clv`, `res.clv_grid("u", i)`, `res.angles(j)`, `res.pair_angle(i, j)`, `res.domination(tau)` (`.clusters`, `.n_physical`, `.violation(j)`), `res.clv_covariance_residual` |
| Particles, dye | `pt = sim.particles(X0, path=, every=, velocity="u", at_walls="refuse")` (exact spectral velocity at the points, RK4 per step: order 4 steady, 3 unsteady; `"clip"` puts wall-crossers back, counted) → `pt.positions(unwrapped=True)`, `pt.velocities()`; `m.dye("c", velocity="u", kappa=1e-3)` (conservative form, no-flux walls; `kappa=0`: pure advection, no filter), `sim.dye("c", path, every=)` (mass, variance, min, max, tail) |
| Adjoint, optimal perturbations | `fm = sim.flowmap(T, t0=, checkpoints=)` → `fm.jvp`, `fm.vjp` (dot-product test); `nl = sim.nlop(T=, E0=, t0=)`, `nl.linear_gain(n_modes=1, tol=1e-12)`; long reverse sweeps over several jobs: `X.FlowMap(..., checkpoint_store="disk").vjp(X0, W, journal=, budget_s=)` raises `X.VJPSuspended`, resume with `fm.vjp(X0, None, journal=)`; `X.compare_adjoints`, `X.structural_sensitivity` |
| Adjoint, optimal perturbations | `fm = sim.flowmap(T, t0=, checkpoints=)` → `fm.jvp`, `fm.vjp` (dot-product test; on a shell with `f_backend="fused"` the vjp follows the planned path, `fm.explain()` says which equations); `nl = sim.nlop(T=, E0=, t0=)`, `nl.linear_gain(n_modes=1, tol=1e-12)`; long reverse sweeps over several jobs: `X.FlowMap(..., checkpoint_store="disk").vjp(X0, W, journal=, budget_s=)` raises `X.VJPSuspended`, resume with `fm.vjp(X0, None, journal=)`; `X.compare_adjoints`, `X.structural_sensitivity` |
| Symmetry | native parities: `cn.walls(a, b, basis="sincos")` (parity read from the BCs); expert `X.SymmetrySubspace(X.Reflection(x), u=X.EVEN, T=X.ODD)`; check what a subspace hides: `X.symmetry_breaking_check` |
| GPU, distributed | install: `docs/site/install.md` (GPU section, tested CUDA 12.9 / CuPy 14.2); `m.simulate(..., backend="cuda", f_backend="fused")`, `X.gpu_available()`, `X.gpu_report()` |
| Memory, cost | `conatus.runtime.memory.plan_memory(plan, "RK443")` (device memory before any GPU), `conatus.runtime.memory.memory_report(sim.solver)` (CUDA solvers; totals include the declared CUDA process floor `CUDA_PROCESS_FLOOR_BYTES`); `python -m conatus agent check run.py --memory` |
| Resolution | `sim.plan.resolution_report(sim.solver, velocity="u")` → `.ok`, `.recommend()`; `cn.resolution.estimate_modes`; NCC options `ncc_cutoff=`, `ncc_tail_tol=` on any native call taking `N` |
| I/O, restart | `sim.output(path, tasks=["u"], every=50, mode="w")` (the state at the start is a row too: `initial=True` by default), `sim.checkpoints(dir, every=100)` (`keep=2` most recent by default), `sim.save(path)`, `sim.restart(cn.io.latest(dir))` (bit-for-bit; `latest` refuses a missing directory), `cn.io.read(path)` (`.to_xarray()`; every array has `dims` + HDF5 dimension scales; Cartesian 2D/3D outputs get a ParaView sidecar `<file>.xmf`, `cn.io.write_xdmf(path)`); one script for every job of a chain: `sim.resume_or_start(dir, T=f)` then `checkpoints`/`output` without `mode=` |
| Uncertainty | `cn.uncertainty(f, N=[8, 12, 16], dt=[0.04, 0.02, 0.01], scheme_order=2)` → `.value`, `.error`, `.trusted`; `cn.time_average(series, dt=, discard=)` |
| Data-driven | `X.dmd(X, dt=, rank=, inner_product=)` (amplitudes default `"first"`), `X.pod`, `X.spod`, `X.edmd`, `X.DataInnerProduct` |
| Ready problems | `cn.problems.native.rayleigh_benard(Lx=, Ra=, Pr=)` → `cn.Model` (start here; `cn.problems.<name>` of the same name is the expert `Ready`, resolution in the call) (also `channel`, `plane_couette`, `pipe`, `taylor_couette`, `kolmogorov`, `kuramoto_sivashinsky`, `swift_hohenberg`, `rotating_shell_convection`, `orr_sommerfeld`, `*_evp` → `lin`); expert `cn.problems` (`.problem`, `.ncc`, `.explain()`) |
| Provenance, runs | `sim.methods()` (Methods paragraph + BibTeX), `cn.methods`, `cn.verify`, `cn.runs.track(name)`, `cn.runs.find(...)`, `cn.runs.compare(a, b)`, `cn.runs.reproduce(run)` |
| Dedalus scripts | `cn.from_dedalus("script.py")` → `.explain()`, `.problem`, `.to_script()` |
| Studio (watch live) | `sim.watch("run_id", film={"T": cn.view("T")}, stats={...}, settable={"dt": (1e-4, 1e-2)}, every_s=0.5)` (periods in wall seconds: `every_s`, `scalars_every_s`, `controls_every_s`), `sim.unwatch()`; `cn.series(f)`, `cn.profile("T", "z")`, `cn.spectrum("u", "x")`; a Studio stop raises `cn.StopRun` in the run; every Studio/hub/relay limit and timing: `conatus.studio.constants.explain()` |
| Notebook, units | `cn.notebook`, `X.UnitSystem`, `X.convert`, `X.presets` |
| Errors, tolerances | `conatus.errors.explain()` (tree), `cn.errors`, `cn.tolerances.explain()` (every declared threshold), `cn.doctor()` |

## 5. Rules that prevent agent mistakes

1. **Nothing is silent.** A refusal is a `cn.errors.ConatusError` with `.message`, `.where`, `.why`,
   `.fix`, `.doc`. Read `fix`, apply it yourself only if it is what you mean; never wrap Conatus
   calls in `except Exception: pass`.
2. **Accuracy tolerances have no default** (`REQUIRED`): `drift_tol`, `xtol` of
   `X.critical`, `tol` of `method="hill"` / `X.shell_floquet` / `nl.linear_gain` / `NewtonKrylovHookstep`,
   `tolerance` of `ContinuationSettings`, `tol` of `X.NewtonSolver`, `arnoldi_tol` of
   `X.equilibrium_eigenvalues` (omitted: v0.1 default 1e-12 with a DeprecationWarning), `safety` of `sim.cfl` and of adaptive `X.EmbeddedError`.
   Write a value and say why; `agent explain <name>` lists them. Algorithm controls (trust radii,
   step growth, Krylov and iteration budgets, FD step) have declared defaults (`cn.tolerances.explain()`).
   Renamed arguments (`nmodes`/`k` → `n_modes`, `X0` → `x0`, `xatol` → `xtol`, `maxit` →
   `max_iterations`, `nsteps` → `steps`, `radius_ratio` → `eta`, ...) still work with a `DeprecationWarning`; write the new names.
   A statement that the answer may not be the one you meant is a visible `cn.errors.CorrectnessWarning`
   (`BaseStateWarning`, `GroupWarning`, `DefaultChoiceWarning`), never a `DeprecationWarning`
   (which Python hides outside `__main__`): treat it as an error and declare the choice it names.
   Never loosen a tolerance to make a refusal go away: find what is under-resolved.
3. **Under-resolved coefficients are refused** (non-constant coefficient tail > `cn.tolerances.NCC_TAIL_TOL`):
   raise `N`, or declare `ncc_cutoff=` / `ncc_tail_tol=` explicitly. Never truncate in your code.
   Callable initial data are refused when aliased, not periodic on a periodic axis or not finite,
   and warn (`cn.errors.InitialDataWarning`) when only not fully resolved (a coarse `N`);
   `initial(..., check=False)` only if you mean it; a run warns once
   (`cn.errors.UnderResolvedWarning`) when its spectral tail exceeds `RUN_TAIL_TOL`, and an
   ill-posed implicit term (backward diffusion) warns at compile (`cn.errors.IllPosedWarning`):
   treat both as errors in your script.
4. **Validate by residuals**: `res.residuals`, `sol.residual`, `E1.residual`, `rank_report()`,
   `resolution_report()`, convergence in `N` and `dt` (`cn.uncertainty`). A bare number is not a result.
5. **Read `explain()`** of every object you report (`m`, `sim`, `res`, `lin`, plan): a short summary
   by default (problem, scheme, checks, warnings); `explain(full=True)` states bases,
   tau placement, implicit/explicit split, dealiasing (default (p+1)/2 for the highest polynomial
   degree p, at least 3/2: declared), filters, residuals.
6. `run(until=T)` must land on a whole number of steps (refused otherwise); `dt` is never adapted unless you
   build an adaptive controller; CFL is a stability indicator, not error control.
7. Restarts are bit-for-bit only with identical numerics (scheme, `dt`, `N`, options) and the same
   problem (model, equations, parameters, box): a mismatch is refused with the diff; a deliberate
   change is `sim.restart(path, allow=("kappa",))`. Use `sim.resume_or_start(dir, ...)` (or
   `cn.io.latest(dir)` + `sim.restart(...)`, which find `.h5` and `.npz`) for multi-job runs;
   both `.h5` and `.npz` restart bit for bit on CPU and GPU. Root searches raise when unconverged (`allow_unconverged=True` returns the
   flagged iterate); a Newton result with `trivial=True` is u = 0, not a new branch.
8. Cost first: `python -m conatus agent check run.py --memory` and `plan_memory` before a GPU job;
   `X.gpu_available()` before `backend="cuda"`. Small `N` on the CPU first, then scale. A second identical
   `m.simulate(...)` in the same process reuses the assembled operators (`sim.explain()` says "reused";
   any change of parameters, `N`, dealiasing, equations, conditions or data rebuilds them).
9. A solution found in a symmetry subspace is not declared stable until checked in the full space.
10. Transposed matrix ≠ physical adjoint; use `left=True`, `X.compare_adjoints`, `fm.vjp` as documented.
11. **Experimental (shipped, tested, not promised; may change without deprecation)**: the Studio,
    the local hub, shells larger than 64 × 48 × 48, 3-D channel and pipe
    orbits (`conatus.ecs.shooting.ShootingRPO`, `conatus.ecs.pipe.PipeTravellingWave`). First use warns `cn.errors.ExperimentalWarning`;
    say so in what you report. `X` exports documented names only: anything else is
    `conatus.<module>.<name>` (`X.<name>` warns `DeprecationWarning`, gone in 0.3).

## 6. Driving a run live (Studio channel, JSON, no server needed on a cluster)

```bash
python -m conatus studio runs                   # every run, live first (JSON, "api": 1)
python -m conatus studio status RUN             # state, t, iteration, dt, scalars, alerts, commands, settable
python -m conatus studio watch RUN --for 600    # one JSON line per change; always bounded
python -m conatus studio send RUN pause|resume|checkpoint
python -m conatus studio send RUN set dt=2e-3   # only if declared settable, within bounds
python -m conatus studio send RUN stop          # exit 5 + confirm_token; resend with --confirm TOKEN
python -m conatus studio log RUN -n 50 ; python -m conatus studio frame RUN --png out.png
```

Exit codes: 0 done (command written; confirm via events `command_received` → `param_set`/`set_refused`),
1 refused, 2 usage, 3 not authorised, 4 no such run, 5 confirmation needed, 6 hub unreachable.
Alerts: `non_finite`, `silent`, `set_refused`, `failed`, `log_errors`, … Act on one alert at a time,
check it was applied, then send the next command. Python: `from conatus.studio.client import connect`
(`cl.status`, `cl.send`, `cl.wait`). Hub tokens are per agent with a scope (`read` | `command`) and
run patterns (`python -m conatus.studio.hub --state DIR addagent NAME --scope command --runs 'host:kh_*' --token-out FILE`).
On a cluster: run agents and servers in compute-node jobs, never on login nodes; no background
process left behind; never poll the scheduler in a loop.

## 7. `python -m conatus agent` — JSON contract (api 1; fields only added within a version)

- `info [--no-probe] [--no-runs]` → `api, command, conatus, python, platform, host, cpus, packages,
  gpu{probed, available, backend, cupy, core, visible_devices, how}, surface{cn, modules, expert}, runs{available, runs[]},
  commands, docs, rules`.
- `explain NAME` (`cn.<name>`, `expert.<name>`, `conatus.<module>.<name>`) → `qualified, kind, signature, summary, value, required,
  required_tolerances, conditional[{argument, when, suggest, kind}], deprecated{old: new},
  methods | names, explain`. Exit 4 if the name does not exist; `explain FILE.py` is refused
  (exit 2: it never runs a script, use `check`).
- `check SCRIPT.py [--memory] [--timeout S] [-- script args]` runs the script for real up to its
  first expensive call (a CPU `simulate` is built, a CUDA one stops; then `run`/`eigenvalues`/`solve`/
  Newton/continuation/… stop it, list `conatus.agent.EXPENSIVE`); Conatus calls that write files
  (`watch`, `output`, `checkpoints`, `save`) are skipped. Not a sandbox: the script's own code (file
  writes, subprocesses) runs with your privileges; never catch `BaseException` around Conatus calls. →
  `ok, calls[{call, line, action}], stopped_at, missing[{call, argument, kind, suggest}],
  refusals[{type, message, where, why, fix, doc, line}], errors, warnings, explain{model, simulation,
  plan, ...}, plan{type, groups}, memory{peak_bytes (allocators), peak_with_process_floor_bytes (+ the
  declared CUDA context/kernels floor, 512 MiB measured), report} | null, notes, elapsed_s`.
  Exit 0 ok, 1 refusal/missing/error, 2 usage. Python: `conatus.agent.check(path)`, `conatus.agent.info()`, `conatus.agent.describe(name)`.

## 8. Where the full documentation is

User guide `docs/site/guide/` (geometries, equations, boundary_conditions, timesteppers, gpu, io,
studio_agents, explain, errors, dynamics/*), tutorials `docs/site/tutorials/`, API reference
`docs/site/api/`, frozen surface `python -m conatus._surface`. Web: `/docs/guide/agents.html`.
