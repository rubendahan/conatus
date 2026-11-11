# Changelog

All notable changes to Conatus.

## [0.4.0] - 2026-11-11

Performance on one CPU node: a benchmark suite with oracle checks, threaded banded solves, a faster
right-hand side, cost estimates before a run, and measured MPI scaling. The threaded solves and the
right-hand side changes leave the result of a time step unchanged, bit for bit, at any number of
threads (checked against 0.3.0 on the IVP cases of the suite). This release also carries the
fixes planned for 0.3.2 (typed refusals of unknown keywords and unsafe files, run status at exit,
record trust, Studio connection state).

### Behaviour changes

- `OMP_NUM_THREADS` (or `CONATUS_SOLVE_THREADS`, which takes precedence) now also sets the threads
  of the CPU banded solves. Results do not change; time per step does.
- An unknown or missing keyword in a public call raises `cn.errors.ArgumentError` (`CN151`), still a
  `TypeError`. Code that matched the text of Python's `TypeError` sees a different message.
- `conatus.expert.<name>` for a name it does not export raises `NameNotExportedError` (`CN231`), an
  `AttributeError`.
- The file readers refuse HDF5 links, external storage, virtual datasets and pickled `.npz` arrays
  (`UnsafeFileError`, `CN171`, a `ValueError`).
- A watched script that fails outside `sim.run`, or exits with `sys.exit(n)` and n != 0, is shown
  `failed` instead of `finished`.
- `run.reproduce(trust_record=True)` refuses a record that changed on disk since it was loaded.
- A function given to `pointwise()` receives a copy of its grid values on the CPU.
- `import conatus` takes about 38 ms instead of 15 ms on the benchmark node (the keyword guard is
  installed at import).

### Known limitations

- A watched script ended by `SIGTERM`, or by `raise SystemExit(n)` written directly, is still shown
  as it was last reported (`running` or `finished`).
- Callable initial data given to the expert interface (`plan`, `IVPSolver.set_state`) are not checked
  for periodicity; `sim.initial` is.
- The solve thread rule and the cost-estimate coefficients were measured on one AMD EPYC 9654 node.

### Added

- A benchmark suite of nine fixed problems (IVPs in 1D, 2D and 3D with Fourier and Chebyshev
  bases, an LBVP, dense and shift-invert eigenvalue problems, a continuation, Floquet exponents).
  Each case checks its answer against an oracle in the same run and records wall time, time per
  step, peak memory, threads, hardware and versions. See the Benchmarks guide page.
- `conatus.cost.estimate`: predicted time per step, run time and memory of a CPU run before it
  starts, from the discretised system, with the error measured on held-out resolutions.

### Changed

- The batched banded solves of a CPU time step can run on several threads
  (`CONATUS_SOLVE_THREADS`, else `OMP_NUM_THREADS`, else one thread). The result is the same bit
  for bit for any number of threads. The number of threads a solve uses follows a measured rule:
  min(allowed, systems, √(unknowns / 900)), one thread below about 3600 unknowns.
  `explain()` of a solver bucket states the threads a solve uses. The rule was measured on one
  node type (AMD EPYC 9654).
- A function given to `pointwise()` now receives a copy of its grid values on the CPU (the grid
  of a field is now shared by the terms of one right-hand side). Writing into the argument has
  no effect outside the function, as in 0.3.0.
- A keyword that a public call does not accept raises `cn.errors.ArgumentError` (code `CN151`),
  instead of Python's bare `TypeError`. It is an `OptionError` and a `TypeError`, so existing
  `except` clauses still catch it. The message names the new name of a former keyword
  (`sim.run(nsteps=)` says `steps`) and lists every accepted keyword. A missing required argument
  raises the same error.
- `m.eigenvalues(nmodes=)` suggests `n_modes`, and its list of options includes the method's own
  keywords (`n_modes`, `drift`, `k`, `dealias`).
- `conatus.expert.<name>` for a name it does not export raises `NameNotExportedError` (`CN231`, an
  `AttributeError`) naming the module that defines it, for example `conatus.ecs.shooting.ShootingRPO`.
- The loaders refuse HDF5 links, external storage, virtual datasets and pickled `.npz` arrays with
  `UnsafeFileError` (`CN171`, a `ValueError`), in `restart`, `cn.io.read` and the other readers.
  `m.linearize(about={"T": "-z"})` raises `InputTypeError` instead of an internal `TypeError`.

### Fixed

- `plan.resolution_report(...)` on a polar, spherical or mapped plan raised `AttributeError`. It
  raises `UnsupportedError`, as the function form did. A mapped plan has `plan.rank_report()`
  (refused, `plan.check()` reports its rank) and `plan.methods()`.
- A watched script that raised between two `sim.run` calls or after the last one, or ended with
  `sys.exit` and a non-zero code, was shown `finished` in the Studio, the agent API and the MCP
  server. It is shown `failed`, with the exception type in the event log.
- `run.reproduce(trust_record=True)` checks the registry root too, and runs only the record it has
  just checked on disk: a record that changed since the run was loaded is refused.
- When the link to the hub hangs without closing, the Studio page says within about 20 s that the
  connection is lost and that the values shown are the last known ones. It used to keep showing
  `running` and the last latency until the reconnection.
- The event log no longer lists `kind`, `run_id` and `relay` as warnings on every run.
- Every Studio screen shows bidirectional and control characters of server text as U+FFFD, the Board
  included.
- The install page says the wheels need glibc 2.17 or newer, which their tags and symbols show; it
  said 2.28, the version they are built and tested on.

### Performance

- The CPU right-hand side evaluates all equations with one cache of sub-expressions and grids, and
  avoids per-call overhead in the reference transforms and the Chebyshev derivative. The results
  are the same bit for bit (tested on the bytes of F and on the state of the benchmark runs).
- Against 0.3.0, three interleaved rounds on one node: 1.02 to
  1.11 times faster per step on one thread and 1.04 to 1.72 times on 8 threads for the IVP cases of
  the suite (largest: 3D Rayleigh-Benard 16 x 16 x 24 and 2D at 256 x 64), with identical states.

## [0.3.1] - 2026-11-11

A correction release. The wrong results listed as known limitations of 0.3.0 are fixed or refused,
with those of four later bug hunts on eigenvalue problems, time stepping, run records, orbits and
statistics. Several calls that returned a number with an assurance now refuse, flag or warn; they
are listed first.

### Changed: new refusals, flags and warnings

- `cn.uncertainty` needs four or more resolutions for `trusted`: three fit any convergence law, so
  the study is flagged and the larger of the geometric and algebraic estimates is the error.
- A restart, an output append and a checkpoint chain refuse a user function or given field whose
  value has no content hash ("cannot be compared"), unless `allow=` names it.
- `continue_equilibria(direction=...)` refuses 0, NaN, a boolean and an array (`InputTypeError`); a
  vector goes in `tangent=`.
- `cn.from_dedalus`: `ivp_solver()`, `discretize()`, `initial_state()` and `ncc()` refuse a
  translation with an unsupported line unless `allow_incomplete=True` (then a `CorrectnessWarning`).
  File handlers, their tasks and `GlobalFlowProperty` are output only: skipped with a note, they do
  not make a translation incomplete. `CFL` stays unsupported.
- `branch_switch` answers `"undetermined"` when its two classifications disagree.
- A double crossing that bisection cannot separate is an event `"unresolved crossing"` with
  `converged=False`.
- `critical()` warns when the growth changes sign more than once in the bracket. Its scan of the
  bracket runs once per call (8 more dense solves); with `over=` the inner calls do not repeat it.
- `ginzburg_landau(kind="oscillatory")` returns `v_g = NaN`, with the reason in the notes, when the
  observable cannot tell the direction of the wave.
- The sensitivity of a multiple eigenvalue raises `SpectralError` with the slopes of its branches.
- `amplitude_equation` refuses a degenerate pitchfork and an observable blind to the critical mode.
- `ShootingRPO` returns status `"retracted"` for a travelling wave, an equilibrium or the trivial
  state; the orbit finders return status `"repeated"` (or `repetitions` in the point data) for k
  laps of a shorter orbit.
- Shift-invert with fewer converged modes than `n_modes`, a Hill solve with a larger resolved
  exponent outside the targets, and an unconfirmed crossing pair in `poincare_section` give a
  `CorrectnessWarning`.
- Loading a checkpoint written by another version, or by a version before 0.3.1, gives a
  `CheckpointWarning` that names both versions.
- Typed refusals where values were accepted or bare NumPy errors were raised: `save_field` of masked,
  object, datetime, timedelta or unicode arrays; `sim.initial` with masked entries or object,
  structured or datetime arrays; a `DynamicalSystem` state of the wrong length or complex, an
  unknown name in `par=`; `EmbeddedError` with a non-boolean `adapt`, or `tol` and `safety` without
  `adapt=True`; `cn.units` scales that are infinite, boolean, text or not positive; reversed or
  non-finite continuation `bounds`; a run record whose `state` or `status` is not a known value.
- The cache no longer stores a run with a callback, a masked, matrix or recarray return, or a key
  holding a structured array with object fields: such calls are computed every time.
- `run(until=...)` refuses a stop time more than 2**53 steps away (`until=1e300` ran for ever).
- `cn.time_average` reports the largest of four standard errors: intervals on a plain AR(1)
  series are about 20 % wider than the true error, and a correlation longer than a batch is flagged.


### Fixed: run records, cache, restart, inputs, display

- A run with a callback is never a cache source: the callback can change the state, and the
  cache key does not represent it. A later run without the callback computed 4 for 1 from it.
- A cached return keeps what it can: a masked array, matrix or recarray return is not stored, so
  the call is not cached (a masked mean came back as 50.5 for 1). A 0-d array stays 0-d, a dict
  keeps its key order and a Fortran-ordered array stays Fortran-ordered. Not kept, and stated: the
  read-only flag and other strides.
- Structured arrays with an object field are not cache keys (they were hashed by pointer).
- The scan for hidden state counts external processes (`subprocess`, `os.system`, `os.popen`).
  The guide says the scan is a list of names.
- A restart, an output append and a checkpoint chain compare a user function or a given field
  that reads a value without a content hash as "cannot be compared", never as equal. A restart
  refuses it unless `allow=` names it; the allow is recorded by `explain()`.
- `run.save_field` refuses masked arrays, Python objects, datetimes, timedeltas and unicode
  strings with `InputTypeError` at the call. They were accepted, then lost in the writer thread.
  `load_field` of a field with no stored snapshot raises `OptionError` with the writer's error.
- `sim.initial` refuses a masked array with masked entries, and object, structured or datetime
  arrays, with `InputTypeError`.
- Typed errors: `continue_equilibria(direction=<array>)` (`InputTypeError`), a mode switch in
  `critical` (`ConvergenceError`, still a `RuntimeError`), reversed or non-finite continuation
  `bounds` (`OptionError`), and a `DynamicalSystem` state of the wrong length or complex, an
  unknown parameter in `par=`, a non-integer `n` or `max_iterations` (they were accepted or raised
  bare NumPy errors). `floquet_multipliers` at the zero state names the absent marginal directions
  in `marginal_absent` instead of NaN overlaps.
- The HTML table of `cn.runs.find()` escapes every field, and a record whose `state` or `status`
  is not a known value is refused on load.
- `EmbeddedError` measures the unknowns and leaves out the tau variables, whose round-off grows
  as dt shrinks. At c dt = 1e-6 for u_t = c u_zz it reported 9e-2 for a true error of 1e-12, and an
  adaptive run with tol = 1e-7 failed at c >= 1e4 with a message that blamed tol. The estimate
  is now the same for t -> c t. A run of rejections says whether the estimate fell with dt.
- `EmbeddedError` refuses a non-boolean `adapt` (`"no"` adapted), a boolean or text `tol` or
  `safety`, and `tol` or `safety` without `adapt=True` (they were ignored). `cn.units` refuses an
  infinite, boolean or text scale, a prefactor <= 0, a NaN or negative energy and a box size
  <= 0 with `OptionError`; a missing scale is an `OptionError` instead of a bare `KeyError`.
  `cn.periodic` accepts a 0-d array.
- An analysis file records the time step of each row (`scales/dt`, `cn.io.read(...).dt`), and
  its `dt` attribute is the step of the last row. A dt change during a run left the first dt there,
  and the chain could not resume with either dt; it now appends after a restart.
- Checkpoints record `conatus_version`; a load of one written by another version, or by one
  before 0.3.1, warns (`CheckpointWarning`) and names both.

### Changed: bifurcation events and continuation

- Branch points, Hopf points, period doublings, Neimark-Sacker points, branch points of cycles and
  the codimension-2 points other than folds are located on the sign change of their test function,
  until the bracket is narrower than `crossing_tol` times the step. The event reports the bracket
  and the values, at its two ends, of the factor that changed sign; `explain()` prints both.
  Folds and LPC still stop at `|t_p| <= fold_tol`. `event_max_iterations` defaults to 60 (was 30).
- The corrector of `continue_equilibria` divides each |G_i| by max_j |[G_x G_p]_ij| x max(1, max|z|): the
  tolerance is relative to the size of the equations, so a change of time unit changes nothing.
  `find_equilibrium`, `find_travelling_wave` and the Krylov continuation multiply their residual
  scale by min(1, ‖J u‖) over two unit probes u: the test only gets stricter, for slow problems.
- Factors at rounding level (an exactly neutral decoupled mode) are left out of the sign of the
  test functions. A bordered Jacobian that is exactly singular at the start (for example y' = 0)
  is solved in the least-squares sense, which keeps that direction fixed; the curve notes say so.
- `branch_switch` returns the classification `"undetermined"` when the ABE coefficients and the
  normal form disagree between transcritical and pitchfork.
- Each point of `continue_orbits` keeps its mesh in `data["mesh"]`; `PeriodicOrbitProblem.on_mesh(mesh)`
  reads a point on its own mesh.
- `CriticalPoint` has `crossings` (every sign change found by the scan of the bracket) and
  `direction`.

### Fixed: bifurcation events and continuation

- Two crossings in one continuation step were lost (branch points or Hopf points 0.01 apart, two
  period doublings, two Neimark-Sacker points). The number of eigenvalues across the boundary is now
  compared with the located events, and the step is bisected until they agree. A double crossing
  at one point is reported as an event `"unresolved crossing"` with `converged=False`.
- An event could be placed at the wrong parameter, with `converged=True`, when a decoupled mode was
  slow or exactly neutral (p = 0.099 for a branch point at 0 with a mode of rate -1e-12), or when
  every rate was below about 1e-11. An exactly neutral mode raised `LinAlgError`.
- Hopf points at rates of 1e-12 were placed at p = 0.038 for 0; at rates of 1e12 they did not
  converge.
- The corrector tolerance of `continue_equilibria` acted as an absolute one: x' = e(p - x²) with
  e = 1e-10 gave a branch off by 0.33 and a fold at p = -0.125. `find_equilibrium` returned the seed
  of a slow problem after 0 iterations.
- `NewtonKrylovHookstep` computed a zero residual norm at a residual of 1e-200 (underflow) and
  returned the seed. Its norms and its trust-region step are now scaled.
- `branch_switch` called x' = p x - 1e-9 x² a pitchfork.
- `ginzburg_landau` could follow another mode in its finite differences of the growth rate (D =
  0.405 for 4). It now follows the critical eigenvector, and refuses when the mode cannot be told.
- `critical()` returned one of three neutral points without a word. It now lists every sign change
  found by its scan, gives the direction of the returned one, and warns when there are several.
- After `continue_orbits(remesh_every=...)`, earlier points were read on the last mesh (field error
  0.4). Branch switching at a period doubling uses the point's own mesh.
- `poincare_section` lost both crossings when two fell within one step (2 of 4 crossings of
  y = 0.99 on a limit cycle at dt = 0.2, return time read as twice the period). A parabola through
  g at the last three steps now flags such steps; g is evaluated at its predicted extremum with one
  step of the scheme and both crossings are located. A predicted pair that is not confirmed is
  stated in the notes and with a `CorrectnessWarning`.
- A Neimark-Sacker point of cycles with two slow decoupled modes was rejected as a neutral saddle.
  Cycle events and Hopf points are now classified by the factor that changed sign across the
  bracket, not by the factor closest to zero.
- `continue_orbits` raised a bare `ValueError` at the fold of a planar cycle with the default
  tests: a test whose set of resolved multipliers changes between two points no longer counts as
  a sign change.
- `ginzburg_landau(kind="oscillatory")` gave the group velocity of a rightward wave with the wrong
  sign (v_g = -|V| for advection speed V). It now takes the eigenvalue of the e^{+ik_c x} mode,
  told from its conjugate by `cos_mode_observable`; with an observable that cannot tell the
  direction, `v_g` is NaN and the notes say why.
- `amplitude_equation` built a Landau equation with g3 = 0 for a degenerate pitchfork
  (u' = p u - u^5), and divided by zero for an observable that does not see the critical mode. Both
  are now refused with a typed error; `LandauEquation.amplitude` and `GinzburgLandau.tau0` refuse a
  zero coefficient.

### Fixed: stability, Floquet and orbits

- Floquet by the Hill method: `fl.leading` is the leading exponent. Shift-invert finds the exponents
  nearest the targets; a Hill matrix of at most `dense_check` rows (default 1500) is now also solved
  densely, and any resolved exponent with a larger real part is added with a `CorrectnessWarning`
  (with `targets=[0.0]` and a true exponent of 19 it returned 4). Above that size
  `details["exponents returned"]` and `explain()` say that larger exponents are not excluded.
  `n_modes` counts distinct modes nearest each target: copies `mu + i n omega` count once.
- Floquet by the Hill method: two independent modes with close exponents (+-1e-9) are both
  returned, and the result no longer varies between calls (fixed Arnoldi start vector). Copies of
  one exponent are recognised by their eigenvectors, not by the distance between exponents.
- `equilibrium_eigenvalues`, `floquet_multipliers`, `symmetry_breaking_check` and
  `ecs_system(odd=False)` measure whether the equations conserve the mean. A mean that is not
  conserved (a reaction term) is kept as an unknown; its rate was missing (top -0.5 for +0.5,
  "no symmetry-breaking instability" at rate 1). The decision is in `sol.stability`,
  `result["space"]`, `result["mean"]` and `explain()`.
- `ShootingRPO` (experimental) no longer returns a travelling wave, an equilibrium or the trivial
  state as a relative periodic orbit: the result has `kind` and `status` (`"retracted"`), as the
  1-D finder's.
- Orbit finders (`find_relative_periodic_orbit`, `ShootingRPO`, collocation orbits) say when a
  solution is k laps of a shorter orbit: status `"repeated"` with the primitive period, or
  `repetitions` in the point data. A seed at 3T of a known orbit used to be reported with period 3T
  and no note.
- Collocation orbits: `|mu_triv - 1|` is described as the error of the trivial multiplier only.
  A Liouville defect (the product of all multipliers against `exp(T int tr J)`) is stored on every
  point and returned by `floquet_from_collocation`.

### Fixed: eigenvalue problems, contour counts and sensitivity

- Eigenvalues of the tested problems no longer depend on their units (diffusion in SI units and
  scale factors from 1e-16 to 1e16, shift-invert near an eigenvalue). Dense QZ and shift-invert run on a
  balanced pencil: the eigenvalue in the unit of the slowest physical row, tau columns scaled to
  their rows, rows equilibrated. `s*T = kappa*dz(dz(T))` in SI units lost 3 digits
  (kappa = 1.4e-7, H = 4 km), was off by a factor 6.6 for kappa = 1e-6, H = 2900 km, and
  `s*u = c*u''` with c = 1e16 returned no finite mode. The finite/infinite split no longer
  depends on units. Pencils near their own units (scale factors within 1e-2 .. 1e2) are solved as
  before. The published Orr-Sommerfeld, pipe-flow and Rayleigh-Benard cases change at rounding
  level only: residual columns and the 11th to 12th digit of a few eigenvalues. Tutorial 13
  prints the position of the wavemaker maximum as |x| < 1e-6 instead of 4e-11: the maximum is
  flat and its position, set by rounding, differs between machines. Tutorial 10 prints the branch point at r = 1 (0.3.0:
  0.999999999999) and the Hopf point of the Lorenz system at 24.736842105263 = 470/19 (0.3.0:
  24.736842105252).
- Shift-invert with the target on an eigenvalue (target 0 beside a neutral mode) no longer raises
  a bare `RuntimeError`, and no longer drops the other requested modes as infinite: the shift
  moves by 1e-5 (relative) and the log says so. Fewer modes than `n_modes` raise a
  `CorrectnessWarning`.
- `contour.count` and the Floquet winding (`HillBlocks.winding`) miscounted a double eigenvalue
  close to the boundary (2 for 3 at a quarter of the positions along an edge). Each half step now
  bounds the whole complex increment of log det T.
- `sensitivity` / `eigenvalue_sensitivity(EVPResult, ...)` at a multiple eigenvalue returned an
  arbitrary number; it now raises `SpectralError` with the slopes of the branches.
  `solvers.evp.cluster_sensitivity` returns those slopes.
- `LBVPSolution.coordinate` works on disk, annulus and other curvilinear plans.

### Changed: Dedalus translation

- `cn.from_dedalus`: `tr.ivp_solver()`, `tr.discretize()`, `tr.initial_state()` and `tr.ncc()`
  refuse a translation with an unsupported statement. The `UnsupportedError` names the statements.
  `allow_incomplete=True` proceeds and gives a `CorrectnessWarning` that names them.
- `cn.from_dedalus` counts the equations against the variables. When an equation line was not
  translated, the difference is an unsupported line and the solver refuses it before the build.
  When every equation translated, the count is a note: Dedalus checks squareness per group on the
  modes each variable has there (a constant tau field in no equation is accepted, as in the
  official pipe-flow example), and so does the discretization.
- `cn.from_dedalus`: after the solve or the time loop, a statement that writes no field data and
  does not solve again (plots, `set_state`, loops over the eigenvalues) is skipped with its reason.
  Selecting a Dedalus subproblem is skipped too. A loop around the solve (a parameter sweep, as in
  the official Mathieu example) is unsupported: the translation computes one solve.
- `cn.from_dedalus`: an `if` is read as an MPI guard only when it compares the rank of
  `MPI.COMM_WORLD` or `dist.comm` with 0, and only when its branches change no field data or
  problem. Any other condition is unsupported.
- `cn.from_dedalus`: a grid at another scale (`local_grid(..., scale=2)`, `change_scales(2)` before
  the data) is approximated: the data are sampled on the grid of the basis size.

### Fixed: Dedalus translation

- `cn.from_dedalus` replays field data in program order. `v['g'] = u['g']; u['g'] += 1` leaves `v`
  at its value at the assignment. A field that reads itself (`u['g'] - np.max(u['g'])`) no
  longer recurses. `b = a; b += 1` changes the array `a`, as in Python.
- `cn.from_dedalus` reports as unsupported a change of field data through a view or a method:
  `d = u['g']; d += 1`, `u['g'].fill(3)`, `out=u['g']`, and a helper or loop that writes data. The
  data of that field is then not translated. These used to be dropped with `complete=True`.
- `cn.from_dedalus`: `if rank_threshold > 1:` was taken as an MPI guard because of its text, and its
  body was run.
- `cn.from_dedalus` raised `KeyError`, `TypeError`, `ValueError`, `RecursionError` or a private
  exception on valid scripts (`3 // 2` in data, `u['g'][0] = 16`, a dropped equation, a helper in
  data). Each is now an unsupported line, or an `UnsupportedError` that names the line. An f-string
  equation is translated with the script's numbers.
- `cn.from_dedalus`: `tr.steps()` is the first stop limit reached, as in Dedalus. With
  `stop_sim_time = 1`, `dt = 1/8` and `stop_iteration = 16`, the translation ran 16 steps to t = 2
  with `complete=True`. A time loop is read only as `while solver.proceed:`. Another condition
  (`solver.proceed and solver.iteration < 8`) is unsupported, because `solver.run(n)` would ignore
  it. `stop_wall_time` is approximated: the run is assumed to end on another limit first.
- `cn.from_dedalus`: field data written after the time loop or the solve is unsupported. It used to
  be replayed into the initial state.

### Changed: verification and statistics

- `cn.uncertainty` accepts three or more resolutions and tests the convergence law in N. Each
  three consecutive values give a geometric rate (q_inf + C e^-aN) and an algebraic order
  (q_inf + C N^-p). Three resolutions fit either law: the larger estimate is given and the study
  is flagged, so `trusted` now needs four. With four or more, a law whose rate drops by more than
  `LAW_TOL` = 10 % between triples is rejected; the report names the law it kept
  (`rep.spatial.law`) and prints both estimates. A refinement that leaves the value unchanged next
  to one that changes it (a parity step: odd and even N give the same result for an even
  solution) is flagged. On 0.3.0 such studies were trusted with errors 1.3 to 1400 times too small.
- `cn.time_average`: tau is the larger of Sokal's window sum and Geyer's initial positive sequence;
  the standard error is the largest of the autocorrelation, 20-batch-means, batch-means
  extrapolated and low-frequency periodogram estimates (new fields `acf_stderr`,
  `extrapolated_stderr`, `spectral_stderr`). A positive autocorrelation longer than one batch is
  flagged. A slow component carrying 15 % of the variance gave trusted intervals that covered the
  mean 64 % of the time on 0.3.0; the measured coverage is now 93.5-97.7 % in 11 settings.
- `verify()`'s "spectral tail" measures every piece of a CompoundChebyshev unknown and every Fourier
  line of the group. It measured the last piece of the last line only, so a mode whose first piece
  was unresolved (eigenvalue 2-5 % off) passed. Its text says the tail is necessary, not sufficient.

### Fixed: found in the review of this release

- `pencil_spectrum` reported the moduli of its ambiguous roots in the unit of the normalised
  pencil, off by max|J| / max|M|.
- The primitive-period check of collocation orbits divided by the largest state value: one lap of
  a limit cycle far from the origin of the state read as eight laps.
- The tolerances table listed a constant of a module that the published package does not contain.
- With a stiff decoupled mode (rate 1e8), events were placed where their factor entered the
  rounding band of the eigenvalues, not at its zero: a branch point at p = 0 at p = -6.7e-7 and a
  Hopf point at -3.3e-7, with a bracket of 1e-11 of the step; at rate 1e10, 100 times further. The
  zero of the factor is now located by following that factor; `explain()` says so.
- The corrector of `continue_equilibria` used one scale for all equations, so a stiff decoupled
  equation loosened the others by its rate: a fold was placed at p = -6.3e-6 for 0 beside a mode
  of rate 1e8. The residual is now relative equation by equation.

### Known limitations

- Two crossings of one test factor within one continuation step (a factor that changes sign and
  changes back) are not seen; the crossing count compares the ends of the step.
- `critical()` scans 8 interior points of the bracket: two neutral points closer than the scan
  spacing are not listed.
- The correctors of `continue_codim2` and `continue_orbits` and the stop test of
  `ParametricEVP.track` are absolute.
- `poincare_section` detects a pair of crossings within one step from a parabola through the last
  three steps: a pair in the first step is not seen, and a pair is a prediction until confirmed.
- `EmbeddedError` measures the whole state, tau variables included, on polar, sphere and
  distributed solvers, as in 0.3.0.
- The primitive-period check of the ECS orbit finders divides by the norm of the state: a large
  mean state can hide a difference.
- `continue_orbits` continued into the end of a cycle family at a Hopf point can turn back there
  and report a fold of cycles at that point.
- Collocation multipliers are checked together (the Liouville defect), not one by one.
- Every checkpoint written before 0.3.1 warns on load.
- An eigenvalue problem whose decoupled parts differ in time scale by 1e12 (two diffusion
  equations at rates 1 and 1e12 in one pencil) loses modes of the fast part to the infinite
  ones without a warning; at 1e9 every mode is found.
- Orr-Sommerfeld written in primitive variables with a dimensional velocity scale U0 <= 1e-3
  returns a spurious unstable mode (c = 0.60 + 4.44i at N = 64, residual 7e-16) as the least
  stable one; 0.3.0 did the same. `drift=` (a second resolution) removes it: use it when the
  equations carry dimensional coefficients.
- `cn.uncertainty`: a clean study whose spatial errors change sign while they shrink (the heat
  equation of the guide at N = 8 to 20) is flagged and not trusted; its error is then the larger
  of the last two differences, which can be well above the true error.
- Contour counts are sampled tests, not proofs; a cluster of four eigenvalues at 1e-7 from an edge
  is refused.

## [0.3.0] - 2026-11-01

A release for agents and the Studio. The new `python -m conatus mcp` serves documentation, runs and records to MCP clients, read-only by default; the Studio pass adds a run log, Compare tables and a sign-in page. The deprecations announced for 0.3 now raise. No numerical result changes.

### Added

- Studio: run names and parameters on gallery cards, with the reason when a run failed and any
  non-finite value. The gallery uses more columns on a wide screen.
- Studio: a **Log** sheet: status changes, commands, checkpoints, warnings and errors with their times.
- Studio: Controls shows each command as sending, pending, done or failed, with the reason. New run
  events `checkpoint_failed` and `fork_failed`; `checkpoint_written` and `fork_written` carry `seq`.
- Studio: axis labels on plots, with units from `meta={"units": {...}}`; a legend in Compare.
- Studio Compare: a table of the parameters that differ, and each run's last value, minimum, maximum
  and record count.
- `packaging/acceptance/studio_truth.sh`: browser checks of the installed Studio against each run's
  stored data, including reload, stop and a lost connection on runs that are running.

### Changed

- The sign-in page of `hub serve --local` asks for the token only and says why a sign-in failed. It
  follows the system's light or dark theme. An empty form no longer counts as a failed attempt.
- Compare lists runs in name order, independent of the clicks, and refreshes while a selected run is
  live. It selects up to six runs, as the page always did.
- The agent API and the Studio report a paused run that has gone silent for 30 s as `stale`. A paused
  run now sends a status every 10 s. A run of an earlier version paused for more than 30 s reads `stale`.

### Fixed

- An exact steady base could read as not steady, from rounding of the base at a wall: plane Poiseuille at about half of the resolutions from 16 to 128. It was a false `BaseStateWarning` in 0.2 and would have been a refusal in 0.3. A base that really is not steady is still refused.
- A failure while writing the initial row of an analysis file ended a watched run as `finished`. It now ends as `failed`, with the reason.
- A hub reading a run slower than the run writes called the run `stale`. It keeps saying `running` while records arrive, and the page and the agent API say how far behind the view is. The local hub reads up to 8 MB/s, and one fast run no longer holds back the others.
- One run file with a line nested too deeply stopped the whole list of runs of the agent channel. That line is now counted as malformed.
- Whole numbers with more digits than the display precision were rounded and looked exact: iteration
  12345 showed as 12350. Counts now show every digit.
- A run killed while paused stayed `paused` for ever.
- The last statistic and frame of a run could be older than its final time. `sim.unwatch()` now
  records the final state, except after a failure.
- After the connection to the Studio dropped, the state and values on screen still looked current. A
  banner now gives their time, states read `last known: ...`, and commands are disabled with the reason.
- A statistic with only non-finite values said "No data yet", and one with a single point too.
- A non-finite simulated time showed as `t = –`; it is now shown as it is (`t = nan`).
- Compare did not update a curve while its run was live.
- A malformed escape in a `/run/` address left a blank page.
- Bidirectional control characters in a run name could reverse the text around it. They now show as `�`.
- The access log of the hub included query strings, so a token pasted into a URL could reach a log.
- A failed checkpoint or fork was visible only in the log; a request without `checkpoint_dir` gave no
  answer in Controls.
- Compare was not reachable from the header at phone width.
- `python -m conatus mcp` (extra `conatus[mcp]`, experimental): an MCP server on stdin and stdout over the
  agent channel and the run registry. Read-only by default. `--allow <command>` adds one write tool and states
  what it permits. No tool runs a script, submits a job or calls `reproduce()`. Run and record text is
  marked as data. Rate limited. Guide page `mcp`.
- `llms.txt` and `docs-bundle.json` (every documentation page as JSON, includes expanded) at the root of
  the site, generated from the pages at site build. `/AGENTS.md` is served there too.
- Recipes (guide page `recipes`): common tasks mapped to the shortest correct code. A test runs each one
  and checks its value against a closed form or a printed value.
- Stable error codes: every typed error has `error_code` (`CN150`), and `ConatusError.as_dict()` returns the
  refusal and its fix as data. `agent check` reports the code. Guide page `error_codes`.

### Changed: new refusals (the deprecations announced for 0.3)
Each former use below now raises, and the message says what to write. None of them changes a result that was
computed: a script that ran without these warnings runs the same.
- Renamed arguments: every former name listed under the releases 0.1 to 0.2 (`maxit=`, `nsteps=`, `k=` for a
  count, `X0=`, `xatol=`, `tolerance=` of `NewtonSolver`, ...) raises `OptionError` naming the new one.
- `conatus.expert.<name>`, for a name the expert layer no longer lists, raises `AttributeError` naming its module.
- `ContinuationSettings` and `KrylovContinuationSettings` take keywords only. `TaylorCouette.radius_ratio` is gone: read `.eta`.
- `m.eigenvalue(name)` needs `growth=`. It was a `DefaultChoiceWarning`.
- `eigenvalues()` on a model with several groups needs the group (`k=`, or `m=` with `kz=` or `ell=`). It was a `GroupWarning`.
- A linearization about a base state above `BASE_RESIDUAL_TOL` raises `OptionError` unless declared with
  `base_residual="accept"`. It was a `BaseStateWarning`.
- `equilibrium_eigenvalues` and `floquet_multipliers` need `arnoldi_tol=`.
- `eigenvalues(k=<integer>)` as a count, and `n_modes=` with a dense method, raise `OptionError`.
- The classes `BaseStateWarning` and `GroupWarning` stay importable and are no longer emitted.

### Changed: new refusals and warnings
- The spool transport refuses more than 60 commands a minute per spool (code 429). Answers of a refused
  command also hold `error_code`.

### Known limitations: open wrong results in the numerics
Each item below was reproduced on this release with a small script and a known answer. None is new in 0.3.0 except where stated. Until they are fixed, check results in these situations by another route.

Bifurcation events and continuation
- **Two crossings in one step are lost.** A double eigenvalue, or two crossings closer than a continuation step, gives no event for branch points, Hopf, period doubling, branch points of cycles and Neimark-Sacker. The unstable count jumps (from 0 to 2 or 4) with no event and no warning.
- **Regression in 0.2.4: an event can be placed at the wrong parameter when another mode is slow or neutral.** Take a branch point at p = 0 with a decoupled mode of rate -1e-12: the event is reported converged at p = 0.099 (it was right in 0.2.3). `crossing_tol` is applied to the smallest factor rather than to the factor that changed sign, unlike what `explain()` prints. The same happens when all rates are below about 1e-11 (p = 0.199 where the crossing is at 0). With two or more spectators the event can vanish, and an exactly neutral spectator can raise `LinAlgError`.
- **Slow Hopf points.** With every rate scaled by 1e-12, a Hopf point at p = 0 is reported converged at p = 0.038. Scaled by 1e12, the event says it did not converge.
- **"Relative" `tolerance` of the corrector and of `find_equilibrium` acts as an absolute one.** On x' = e(p - x²) with `tolerance=1e-10`, the branch is off by 9e-3 at e = 1e-8, and points still say converged. On a slow equilibrium problem (rate 1e-10) the seed is returned after 0 iterations as the solution.
- **`NewtonKrylovHookstep(scale=...)` can report a zero residual at the wrong point.** With a residual scale of 1e-200 the seed is returned with residual 0, at distance 1 from the root.
- **`branch_switch` calls a transcritical point a pitchfork** when the quadratic coefficient is small (1e-9); `normal_form` gives the right kind.
- **`continue_orbits(remesh_every=...)`** can continue from a remeshed point that no longer solves the problem (field error 0.4, residual 5), and the point says converged.
- **`ginzburg_landau`** with a finite-difference step `h` larger than the distance to a second mode's crossing gives a wrong `D` (0.405 for 4 at a gap of 4e-5, `h=0.1`). A smaller `h` gives the right value.
- **`critical()` with a growth rate that changes sign three times in the bracket** returns one of the neutral points without saying it is not the first (3 where the first is 1).

Stability and Floquet
- **Hill `leading` is only the leading exponent near the targets.** With a target at 0 and a true exponent of 19, it returns 4; with a target of -9 and a true exponent of 0.5, it returns -2.5, which reads as stable. There is no warning. `n_modes` can return fewer modes than asked. Compare with the monodromy.
- **Hill with slow rates.** With exponents of ±1e-9 and a tolerance of 1e-12, it returns one exponent where there are two, and the unstable one is missing in about half of the calls. The result varies between calls.
- **The mean is dropped from three ECS stability paths without checking that it is conserved.** `equilibrium_eigenvalues`, `floquet_multipliers` and `ecs_system(odd=False)` on an equation with a reaction term report a top value of -0.5 (multiplier 0.61) where the truth is +0.5 (1.65). `explain()` calls the mean conserved.
- **`symmetry_breaking_check`** can say there is no symmetry-breaking instability when there is one: on a problem with a true rate of 1 it reports 0 (and -2 where the rate is -1).

Dedalus translation (`cn.from_dedalus`)
- **`complete=True` with a wrong initial state.** Aliases and in-place changes are lost or misapplied: `d=u['g']; d+=1` gives 1 for 2, `u['g'].fill(3)` is dropped, and `v['g']=u['g']; u['g']+=1` changes `v` too.
- **A branch on a name such as `rank_threshold` is taken as if true.** `if rank_threshold > 1:` runs its body with `rank_threshold = 0`, with `complete=True`.
- **`ivp_solver()` on an incomplete translation gives no warning.** Only `complete` and `explain()` say so: initial data set in a loop or by `exec` come out as 0.
- **Crashes on valid scripts.** `u['g']-np.mean(u['g'])` raises `RecursionError` with `complete=True`. Others raise `KeyError`, `ValueError`, or an error about a non-square system where an equation was dropped.

Run records, cache and restart
- **The cache ignores what a callback did.** `run(callback=..., record=...)` changes the state, and `run(cache=True)` without the callback returns the changed state (4 for 1).
- **The cache loses masks.** A cached `MaskedArray` comes back plain with a different mean (50.5 for 1).
- **Structured arrays with object fields are hashed by pointer,** so a changed value can return the old result.
- **`subprocess` is not in the scan for hidden state,** so a cached call that reads one returns the old value.
- **`restart` accepts a changed function.** If `m.function` closes over a changed value, the restart does not report the change.
- **`save_field`** on datetime, string or object arrays records the run as completed and fails on load with `IndexError`. It removes masks.
- **A cache hit differs from the computed return in type details:** a 0-d array becomes a scalar, read-only becomes writable, Fortran order becomes C, and dict order is sorted. Values are the same.

Inputs and errors
- **`sim.initial` reads a masked entry as data,** without a warning. Object, structured and datetime arrays raise bare NumPy errors.
- **Bare errors:** `continue_equilibria(direction=<array>)`, a mode switch in `critical`, and NaN overlaps in `floquet_multipliers` at the zero state.

Notebook display
- **The HTML table of `cn.runs.find()` does not escape `state` and `status` of a stored record.** Display only registries you trust.

## [0.2.4] - 2026-10-31

A correction release, from an external review of 0.2.3. Several results can differ from 0.2.3; they are listed here.

### Security
- HDF5 loaders refuse datasets stored in other files (external storage, virtual datasets), as they
  already refused links, before any payload read. This covers checkpoints, `cn.io.read`, analysis
  files and the HDF5 data of run records. A virtual dataset you made yourself is refused too.

### Changed: results that can differ
- `generalized_eigenvalues` and everything built on it (`continue_equilibria`, codimension-2
  continuation, normal forms, manifolds, amplitude equations) keep every finite root when M is
  nonsingular. Stiff systems gain large roots and can gain unstable counts. A root at rounding level of
  a singular M is still removed; `continue_equilibria` counts it in `data['ambiguous']` and warns.
- Period-doubling, Neimark-Sacker, branch-point and zero-Hopf events are located and judged by the
  distance of their own critical multiplier or eigenvalue. Locations change in systems with several
  noncritical modes, and an event can now report not converged where a product used to pass. Pairs
  with sum 0 or product 1 that are not conjugate are no longer reported as Hopf or Neimark-Sacker.
- `normal_form(kind="zero")`: the kind no longer depends on parameter units. "transcritical" needs
  its own conditions. The new kind "unresolved" (with `degenerate` True) replaces the former answer
  when `parameter` is missing or the parameter does not unfold the point. A fold with a2 = 0 and a
  pitchfork with mu = 0 are flagged degenerate. A Hopf normal form at a real pair is refused.
- `record()` stores `growth_max` by the declared growth convention, plus `growth_convention`.
- Function cache keys and records distinguish list and tuple, NumPy and Python scalars, int, float and
  bool, bytes and dict order. Calls cached by earlier versions are computed again. Cached results are
  not reused across this change.

### Changed: new refusals and warnings
- `find_travelling_wave`, `find_relative_periodic_orbit`, `find_relative_equilibrium_dae` and
  `ShootingRPO` refuse, with `SymmetryError`, equations that do not commute with the translation they
  assume. The DAE finder and `AffineSteadyProblem` refuse time-dependent right-hand sides.
- `ir_system` refuses any parameter-dependent mass above rounding, however small, and refuses evaluation
  beyond 1e15 times the reference value of a left-hand-side parameter.
- `reproduce`: `code_identical` also covers helpers, closures and globals, and is None when they cannot
  be compared. `require_same_code=True` raises in both cases. A decorated-call record from before 0.2.4
  that holds a list is refused.
- `m.solve` and `m.simulate` warn (`UnderResolvedWarning`) when a callable given field disagrees with the
  interpolant of its grid values off the grid. Coordinates written inside an equation's text are not probed.
- `m.solve` and `m.simulate` call a callable given up to five more times per direction, off the grid. The
  computed values do not change. The global random states are restored. A callable that holds a random
  generator, is not repeatable or takes over one second per evaluation is not checked, and a solve's
  `explain()` names it. A callable that counts its own calls sees the extra ones.
- A closure or global holding a tuple or NumPy scalar changes the computation identity, so a checkpoint
  or HDF5 append from an earlier version that holds one reports an identity difference.

## [0.2.3] - 2026-10-31

A correction release. It adds a resolution statement to boundary-value solves, fixes the security
items listed below, and fixes the Studio gallery. Some calls now warn or refuse, listed under Changed.

### Security

- `to_script()`: a Dedalus file name with quotes and a newline could put code into the generated program.
  It is now escaped, and only the template's imports are generated.
- `?format=png` on a frame is limited to 2^24 values, checked before anything is allocated, and runs off
  the hub's event loop. A small planted frame could exhaust memory and stall the hub.
- `HubClient` refuses a redirect to another origin. It used to send the bearer token to the new address.
- Control files, relay reads and atomic record writes do not follow planted links.
- Recorded command lines also hide `--pass`, `--pwd`, `--key`, `--bearer`, `--signature`, `TOKEN=value`,
  a value that starts with a dash, and URL passwords and secret query values.
- Checkpoint discovery reads `.npz` shapes from headers, so a small file declaring gigabytes no longer
  allocates them. HDF5 checkpoints holding a link to other data are skipped.
- The native core refuses an `ldab` above the range of `int`, is safe when an RHS callback calls
  `set_rhs(None)` on itself, and is built with a stack protector and full RELRO. FORTIFY is on, and it rewrites no call in this code.
- Added `SECURITY.md`. A run registry carries code: treat one you receive as you would a script.
- Hub: an HTTP connection idle for 300 s is closed, and a throttled address writes one audit line per 15 minutes.
  `[::1]evil.com` is no longer read as loopback.

### Changed

Behaviour changes:

- `m.solve()` and other boundary-value solves warn with `UnderResolvedWarning` when an unknown's relative
  spectral tail exceeds `SOLVE_TAIL_TOL = 1e-4` (8 or more mode numbers), and `explain()` prints the
  tail. They used to say nothing. Nothing else changes: the values and `N` are as before. A mapped domain
  or a compound, Hermite or Laguerre direction is not judged, and `explain()` says so. The right-hand side of a
  Cartesian linear solve is read as well, with `SOLVE_DATA_TAIL_TOL = 1e-2`. This fixes the limitation
  listed under 0.2.2. A solution with real content in the last 10 % of its modes is warned even if it is
  exact there, such as a polynomial of degree `N - 1`.
- A coefficient that is exactly zero, or terms that cancel, now lower the order that boundary conditions
  are counted against. `nu*dz(dz(u)) - u = 1` with `nu = 0` and two wall conditions raises
  `BoundaryConditionError`. It used to return a wrong answer with residual 0. This also applies to
  evolution and eigenvalue models: Euler written as `nu*lap(u)` with `nu = 0` and no-slip or free-slip
  conditions on Chebyshev walls is refused. A zero number or a given callable that is zero on the grid
  counts as a zero coefficient, also under a derivative.
- A SymPy string given as a base flow or coefficient (`parallel_shear_flow`, `conatus.sensitivity`) raises
  `InputTypeError`, because SymPy evaluates a string as Python. Pass an expression, a number or a callable.
- `conatus[studio]` needs `tornado>=6.6`.
- `hub serve --agent-scope` defaults to `read`, and an agent or user is limited to 60 commands a minute on
  the API (429 above).
- `?format=png` scales a frame by its longest side, not its width, to reach 256 px.

### Fixed

- One NaN in a run's watch metadata emptied the Studio gallery for every run. Non-finite values are now
  stored and sent as the strings `nan`, `inf` and `-inf`, and displayed as such. The hub also reads them
  from older files.
- One malformed status record made `/api/v1/runs` and `/board` answer 500 for all runs. A run whose
  records cannot be read is now listed as `unreadable`.
- The hub's limit of 200 runs is stated in the gallery, and a deep link to a dropped run says so.
  `hub serve --max-runs` raises it, and the hub logs the first time it drops a run.
- `0**-1` in an equation raises a typed error instead of `ZeroDivisionError`.

### Known limitations

- An order drop that shows only across a first-order system, or a point `nu = 0` of a `parametric` pencil, is
  not refused. The resolution statement reads the solution and the data, and passes a small unresolved
  layer beside a much larger smooth part. Compare two values of `N`.
- A malformed checkpoint can still cost time before it is refused. Read files from people you trust.
- Recorded command lines keep short options such as `-p` and code records keep the diff of tracked
  files, so a secret passed or committed that way stays in the record.
- The hub's content security policy allows any WebSocket address, and it accepts HTTP/1.0 requests without
  a `Host` header. The hosted mode, which is not part of the public release, trusts proxy headers on its port.

## [0.2.2] - 2026-10-30

A correction release from an external review of 0.2.1. Upgrading is recommended: several paths returned
wrong values without an error, listed first under Fixed. Some calls that used to return a value now
refuse, listed under Changed.

### Security

- Behaviour change: reproducing a decorated call now needs `fn=` or `trust_record=True`, and only trust
  records you made. `run.reproduce()` used to import and call the function a run record names. A
  registry you unpack or copy from someone else belongs to your account, so the ownership check did
  not protect you: such a record could run code of its author's choosing when you called
  `reproduce()` without `fn=`. Loading and searching a registry never ran code. The ownership and mode
  checks still apply under `trust_record=True`. `reproduce()` in 0.2.1 and earlier is affected:
  upgrade, and until then pass `fn=` for any record you did not make.

### Fixed

Wrong results returned without an error:

- `FlowMap.vjp_physical` returns the Hermitian adjoint on complex states. It returned a bilinear
  transpose, which could reverse the sign of a gradient. Real states are unchanged.
- Integrals along a nonperiodic axis were wrong in the batched and fused right-hand sides and in their
  tangent, for Chebyshev, Jacobi and ultraspherical bases.
- A multistep restart reused its history after an operator had changed. A checkpoint now records the
  operators: an allowed change discards the history with a warning and the declared start-up runs from
  the state. Reordered equations with history are refused unless `allow=("equations",)` is passed.
- `conj` of a native complex unknown is kept; it was dropped.
- `normal_form` no longer changes its decision when an equation is multiplied by a constant.
- The SPOD interval `confidence(alpha)` was reported for cases it does not cover. See Changed.
- Standalone particles check the final position against the walls, and the first adaptive step
  respects the bounds on `dt`.
- The vector potential and the dye use the wavenumbers of the first derivative at Nyquist; a mode with
  only Nyquist wavenumbers, which no grid curl produces, is refused.
- Dye variance is centred before squaring, so it no longer loses a small variance at a large offset.
- The first output of an integer value no longer fixes an integer dataset; a complex value after a real
  one no longer fails. Integer tasks stay integer.
- Optimized DMD returned NaN modes when the clock origin was far from the data.
- POD and EDMD of integer snapshots wrapped in their Gram products.
- `rbf` used an algebraic square for complex states and dropped the imaginary part of complex centers.
- Hopf events no longer depend on stable spectator modes, in `PseudoArclength` and
  `continue_equilibria`.
- `cos_mode_observable` keeps the imaginary part of a complex vector; `discrete_shift` is a first-order
  derivative for a nonlinear factory; Poincare cutoffs apply to the crossing time, and refined
  crossings stay in their segment.
- A parametric derivative no longer loses an order-one entry when its small coefficient is pruned.
- Newton keeps a declared complex unknown complex on every basis.
- `validity_radius` scans radii before it bisects, because the defect need not be monotone. The answer
  is a sampled one and the docstring says so.
- Subspace invariance is measured as the residual in the range of `M Q`.
- A scalar neutral mode no longer returns a NaN residual.

Refused or flagged with a clear message:

- Singular pencils (`solve_dense`, descriptors), a degenerate cubic coefficient, non-finite eigenvalue
  callbacks and Poincare sections, Lyapunov vectors beyond a projected dimension, an antilinear
  right-hand side with `f_transpose="dense"`, products of different base-flow harmonics in
  `periodic_operator` and `hill_from_plan`, and Newton on overflowed data.
- `DataInnerProduct.gram` refuses non-finite and non-Hermitian matrices, also at extreme scales.
- `uniform_dt` refuses infinite times and an overflowing step.
- Restart and append guards compare `time_parameter`. `latest` skips an incomplete checkpoint with a
  warning, and a load validates names and shapes before it changes the solver.
- `xarray` export keeps user names; a colliding coordinate or coefficient array gets a free name, listed
  in the Dataset attributes. XDMF comments and dataset references are valid XML.
- Strict replay compares the source before the call and refuses a change; otherwise it warns and
  records `code_identical=False`. Differences of booleans and large integers are exact.
- A cached call that reads a file, the environment or an unseeded generator through an alias no longer
  gets a content identity.

### Changed

- `SPOD.confidence(alpha)` holds for one entry and non-overlapping blocks and refuses other cases.
  `confidence(alpha, approximate=True)` returns the same formula with a stated, measured coverage.
- `spod(onesided=True)` refuses a Gram matrix with an imaginary part.
- Optimized DMD with `times` not starting at 0: the amplitudes hold at the first time
  (`DMD.time_origin`), and `reconstruct` takes times on the clock of the data.
- `edmd(dt=...)` needs a finite `dt > 0` or `None`, as `dmd` does. `spod` and `welch_blocks` check `dt`
  and the window.
- `cn.uncertainty` flags spatial differences of opposite sign, so such a study is no longer `trusted`.
- `count(control=(box, n))` counts no pole in the control box; `(box, n, poles)` declares them.

### Documentation

- The Studio guide starts the server first and then a run long enough to be seen live. It no longer
  promises a registry view in the local hub.
- The published pages now match the source pages for the nonmodal mathematics, Floquet and install
  pages. `docs/site/build.sh --setup` works on the public tree.
- The pipe-flow API states the tabulated frequencies and the sign of the eigenvalue of `exp(s t)`.
- Corrections to the Kelvin-Helmholtz caption, the growth-rate rule (alpha Im c, for alpha > 0), the
  energy slope, the weighted adjoint and tutorial 8, which now minimises over k.
- Plots stay readable in the dark theme; duplicate page ids are removed.

### Known limitations

- A small residual does not show that a linear boundary-value solution is resolved. With an
  under-resolved boundary layer (`nu = 1e-10`, `N = 32`, `nu*dz(dz(u)) - u = 1`, `u = 0` at both walls)
  `solve()` returns values down to -1.995, where the solution lies in [-1, 0], with residual 7e-18 and no
  warning. It also accepts `nu = 0` with two wall conditions. Check by raising `N`.
- Contour counts are sampled tests, not proofs. `validity_radius` is sampled as well.
- `SPOD.confidence(approximate=True)` is a pointwise approximation for many blocks and separated eigenvalues.
- The dependency tracking of cached calls is name-based for a reader reached through an object method
  (`Path.read_text` on a variable).
- `solve_dense` removes a root at the scale of the noise of `M` and reports it in `infinite_ambiguous`,
  because such a pencil cannot be told apart from a noise row in double precision.

## [0.2.1] - 2026-10-24

A correction release for the local Studio. Upgrading is recommended for anyone who uses it: in
0.2.0 the Studio app installed from the wheel showed a blank page. The solver library is unchanged
apart from how a watched run reports its end.

### Fixed

- Studio: the web app of the installed package showed a blank page. The release build changed the
  app's JavaScript after it was built, and the result did not parse. The app is now shipped as
  built, and every release is checked by signing in to the Studio of the installed wheel in a
  headless browser.
- Studio: the app's files have a new name in each release. Browsers keep these files for a year,
  so a 0.2.0 page would otherwise have kept the broken file after the upgrade.
- Studio: a watched run that raises (a blow-up stopped by the finite-state check, an error in a
  callback, an interrupt) is reported as failed, with the exception type and message, in the app
  and to `python -m conatus studio`. It was reported as finished.
- Studio: a run stopped from the Studio is shown as stopped in the gallery, the run header and the
  Board, as the agent interface already reported it. The app showed it as finished. The app now
  shows the six states of the agent interface (running, paused, stale, finished, failed, stopped).
- Studio: "Confirm stop" and "Confirm fork" go back to "Stop" and "Fork" after 5 seconds. A later
  click no longer sends the command.
- `python -m conatus studio status <run>` with no `--hub`, `--spool`, `$CN_STUDIO_URL` or
  configuration file now says that no source is configured, instead of an empty list of known runs.

### Documentation

- Studio guide: a fixed state directory is given before the subcommand,
  `python -m conatus.studio.hub --state ./hub_state serve --local ...`; the guide showed it after
  `serve`, which the command refuses.

## [0.2.0] - 2026-10-24

This release adds a set of published studies reproduced by public scripts and corrects the findings
of an external review of 0.1.2. Each fix has a test against an independent reference value.
Upgrading is recommended: some results of 0.1.2 were wrong (see Correctness). Results that can
differ from 0.1.2: invariant solutions with a conserved mean, flow maps after a change of time step,
the finite/infinite split of dense QZ under a rescaled equation, spectral POD confidence intervals
at zero and Nyquist frequency, certified Floquet roots, cached calls of functions that draw random
numbers, angles between complex covariant Lyapunov vectors, and contour counts of a function with
poles near the boundary.

### Added

- Published studies reproduced, in `examples/published/`: the Orr–Sommerfeld spectrum of plane
  Poiseuille flow (Orszag 1971, 32 eigenvalues and the critical point), linearised pipe flow from
  Re = 10² to 10⁷ (Meseguer & Trefethen 2003, Table 1, 17 eigenvalues), the marginal curves of
  Rayleigh–Bénard onset for three pairs of walls (Rayleigh 1916, Chandrasekhar 1961) and the
  equilibria and travelling waves of Kuramoto–Sivashinsky at L = 22 (Cvitanović, Davidchack &
  Siminos 2010). Each script prints, for every published value, the computed value, the gap and
  whether it is within the rounding of the printed digits; `examples/published/run.sh <case>`
  records the output with the script hash, the version and the machine. The site has a page that
  draws the published values over the computed ones, and Validation rows link to the script that
  prints them.
- `EVPResult.infinite_ambiguous`: the moduli of the eigenvalue pairs removed as infinite on
  numerical rank alone.
- `SPOD.dof()` (degrees of freedom per frequency) and `SPOD.real_data`.
- `FlowMap.dt`: the time step a flow map was built with.

### Security

- Studio hub sessions carry a second secret next to the cookie: a session key that the app keeps in
  the storage of its own origin and sends with every data request, command and WebSocket. A cookie
  replayed by another service on the same host name, without the key, is refused (401). Bearer
  tokens of agents are unchanged.

### Correctness

- Invariant solutions (`find_equilibrium`, `find_travelling_wave`, `find_relative_periodic_orbit`)
  keep the mean of the initial guess; it was reset to 0 (1 + cos x came back as cos x). A result
  whose full equations, frozen mean included, miss the tolerance is refused
  (`NewtonNotConverged`); `u_t = 1` returned an "equilibrium". The orbit Newton uses the
  derivative of the discrete map for its period column, so it converges at its full rate.
- `FlowMap` keeps the time step it was built with; after `solver.set_dt`, calling the map ran at
  the new step while `trajectory` and `jvp` kept the old one. Repeated orbit residuals no longer
  depend on the period evaluated before.
- Dense QZ: the finite/infinite split no longer depends on the scale of each equation. A pair
  removed only because `M` is numerically singular is listed with its modulus in
  `EVPResult.infinite_ambiguous` and in the log.
- `SPOD.confidence`: N_b degrees of freedom at f = 0 and Nyquist of real data (2 N_b elsewhere);
  the nominal 95 % intervals covered 83 % there.
- `CertifiedRoot.certify` requires a finite root that matches the finest resolution of its study;
  changing the root or its evidence afterwards withdraws the certification.
- Run cache: a function that reads a global random generator, the clock or an unseeded
  `default_rng()` is never served from the cache (it was, after reseeding). A generator built from
  a seed argument, or a module-level value named like one (`seed = 7`), keeps the cache key.
- `clv_subspace_angles` and `LyapunovResult.pair_angle` use the Hermitian product for complex
  vectors.
- Eigenvalue results report a failure to identify the tau variables in `spurious_note` /
  `filter_note` instead of treating it as "no tau".
- `cn.contour.count` for a function `T(z)` that may have poles: a zero just inside an edge and a
  pole just outside it (1e-3 apart) were counted 0 instead of 1. The walk now refines where
  `log det T` bends, and when a zero or a pole lies closer to the boundary than 1e-3 of the
  perimeter the count warns (`CorrectnessWarning`, also in `CountResult.near_edge`) and suggests
  moving or shrinking the box.
- The Studio example's kinetic energy is ½∫|u|² (it was half of that).

### Documentation

- The methods pages state the derivative each method uses and the hypotheses of each result: the
  tangent of a Runge–Kutta stage sums over the earlier stages, the phase residual of
  pseudo-arclength continuation carries the weight W of its bordered row, DMD modes are not
  orthogonal in general, the growth rate of a mode with phase speed c is α Im c, and the
  conventions of the energy-gain slope, the Floquet fundamental matrix, the real and complex
  adjoints and the covariance of Lyapunov vectors. Each identity is checked by a test.
- References shown by the site, the docs, the docstrings and the examples are checked against
  Crossref or arXiv; one reference that does not exist was removed, twenty were corrected, and a
  wrong equation number fixed.
- Guides, tutorials and the site describe what the code does, without comparison with other codes.
  A gallery picture reports only what its own run measured; the install page and the docs build
  their tables of releases and optional extras from one source. Docstrings of the public API carry
  no internal identifiers.

### Changed

- The package follows one import rule between its layers (base, IR, reference, discretisation,
  execution, methods, model, user). Six large modules were split; their former import paths still
  work.
- The warnings that 0.1 announced as errors "in v0.2" (`BaseStateWarning`, `GroupWarning`,
  `DefaultChoiceWarning`, `arnoldi_tol`, the `k=`/`n_modes` names and the other renamed arguments)
  stay warnings in 0.2.0 and become errors in 0.3. The messages and the guide say so.
- The status table of the documentation lists the local Studio server
  (`python -m conatus.studio.hub serve --local`) as shipped; serving the hub beyond 127.0.0.1 and a
  relay to a remote hub are not supported in this release.

### Known limitations

- A contour count for a function `T(z)` with poles is a test on samples, not a proof: a zero and
  a pole on opposite sides of an edge whose bend is hidden by a stronger smooth variation of
  `det T` can still cancel.
- The GPU paths were not run again on GPU hardware for this release; the GPU statements of the
  documentation refer to the last release that ran them (0.1.x).
- Unchanged from 0.1.2: the dense QZ limitation on pencils with a singular `M` and finite
  eigenvalues above about 1e12 times the scale of the pencil.

## [0.1.2] - 2026-10-21

A correction release for the findings of an external review of 0.1.1 and of a review of its
documentation. Each fix has a test against an independent reference value. Upgrading is
recommended: some results of 0.1.1 were wrong (see Correctness). Results that can differ from
0.1.1: large finite eigenvalues of dense QZ, parametric pencils with small parameter terms,
contour results with close roots, `critical(over=)` and the Hopf curves of `continue_codim2`.

### Security

- Local Studio: the session cookie is `cn_session_<port>` with `--local` and `__Host-cn_session`
  over HTTPS, and each session is bound to the `Host` it was opened at. The cookie was valid when
  replayed from any port of 127.0.0.1. On plain HTTP, another service on the same host name and
  port can still read it; the Studio guide says so.
- Telemetry spools, Studio audit logs and token files are created 0600 in 0700 directories,
  whatever the umask; they took the umask's mode. Run records and outputs keep the umask's mode.
- `hub serve --local --state hub_state` uses the directory given; the explicit spelling of the
  default was replaced by a temporary directory.

### Correctness

- Dense QZ (`solve_dense`, `m.eigenvalues(method="dense")`): with a nonsingular `M` every
  eigenvalue is finite and returned; `L = -diag(1, 1e13)`, `M = I` lost 1e13 as infinite. The
  rank of `M` is measured on `M` as given, so a row of rounding noise where `M` has a zero row
  does not make `M` look nonsingular (its infinite eigenvalue was returned as a finite value near
  1e17). The log states the rank and names the pairs it cannot classify.
- Parametric pencils (`m.parametric`): a coefficient entry is discarded only at rounding level
  for that entry; a whole matrix 1e-14 times the pencil's largest entry was deleted as
  "identically zero" (eigenvalue 1e-14 r read 0 at r = 1e14). An evaluation where the discarded
  entries could matter is refused.
- `continue_codim2(kind="hopf")` stops at the Bogdanov–Takens point (κ = ω² = 0) and keeps only
  points whose Jacobian has the pair ±iω; it continued into real-saddle points. Tutorial 10, its
  figure and its checks are corrected.
- `cn.contour.beyn`: close distinct roots (0.2 and 0.2000001) are returned separately, each
  counted on its own square; they were one root "of multiplicity 2" at their mean.
  `multiplicity` is the number of eigenvalues in a square of half-side `enclosure` (new field)
  around each value.
- `cn.contour.beyn(counted=...)` refuses a count of another operator (it evaluates `det T` again
  at points the count recorded); it accepted any count of the same box.
- `cn.contour.count` is described as a test on samples of the boundary. For a function `T(z)`
  with poles, a zero and a pole on opposite sides of the boundary, closer to it than a step, are
  not seen; `count_method`, `explain()` and the new `CountResult.scope` and
  `poles_by_construction` say so. The contour results no longer use the word "certified".
- `BeynResult.residual_definition` states the scale of the residuals; the guide's formula for a
  function `T(z)` was not the one computed.
- `critical(over=)` computes the neutral value at `over_points` points of the interval (default
  9) and refines every local minimum of that scan; one bounded Brent search could return a higher
  local minimum as the critical value. The log lists the minima found, and a point where the
  pencil is refused is logged as unchecked.
- The run cache refuses a function whose code reads a value chosen at run time (`getattr`,
  `vars`, `os.environ`, `open`, `np.load`, ...); such a read kept the cache key, and a changed
  configuration returned the old value.

### Checks that reported success without measuring it

- `verify()` measures the right and left residuals of the vectors the mode holds when it is
  called; it printed the residual stored by `track()`, so a mode whose vector had been changed
  still passed. A zero vector, or a vector of the wrong size, is not a pass. A point where the
  parametric pencil is refused is reported as a failed check instead of an exception.
- `resolution_report` is `ok` only when every variable of the plan was measured; an empty or
  partial state is reported as not measured (`complete`, `unmeasured`). An array of the wrong
  shape is refused with the expected shape instead of an `IndexError`.
- `python -m conatus agent check` checks the argument values of `Simulation.run` and
  `IVPSolver.run` at the stop (`run(steps=-1)` was reported ok); the new `validated` field says
  whether the values of the stopped call were checked.
- The acceptance scripts shipped with the package check the mathematical requirement of each
  case, and their runner counts a script that crashes or prints no result as a failure.
- `uncertainty(...).explain()` prints tuple resolutions such as `N=[(8, 8), (12, 12), (16, 16)]`
  (it raised `TypeError`).

### Documentation and website

- The home page and the README start with one complete calculation, the critical Reynolds number
  of plane Poiseuille flow, shown with its recorded output. The site and the documentation share
  one navigation and one getting-started route.
- The code shown on the guide and tutorial pages runs when copied in order, and the outputs shown
  are what it prints; a test runs every page.
- Tutorial figures redrawn: tables where a plot carried little information, captions computed
  from the plotted data, values below the axis floor drawn as upper limits.
- Wording: claims are stated with their scope and threshold, without slogans; the time-stepper
  orders are given per scheme; the pages no longer compare Conatus with other codes.
- The Validation page lists published values and closed forms; the values are labelled as
  reported by the author, since their run records are not public.
- Pages no longer cite examples or test files that the package does not ship; the deprecation
  versions and the release status are read from the package.
- `SHA256SUMS` lists bare file names, so `sha256sum -c SHA256SUMS` works in the directory that
  holds the downloaded wheels.

### Known limitations

- Unchanged from 0.1.1 (see below).
- A contour count for a user function `T(z)` with poles is a test on samples, not a proof.
- Local Studio over plain HTTP: another service on the same host name and port can replay the
  session cookie.
- Dense QZ: a pencil whose `M` is singular and which also has finite eigenvalues larger than
  about 1e12 times the scale of the pencil can have pairs that cannot be classified; they are
  kept as finite and named in the log.

## [0.1.1] - 2026-10-20

A correction release: every finding of an external review of 0.1.0, and of our own audits of
security, input validation, shared state, invariance and assurances (checks that
reported success without measuring it), is fixed, each with a test. Upgrading is
recommended: some results of 0.1.0 were wrong (see Correctness). Changes that can alter a result
you computed with 0.1.0: the default dealiasing grid of some models, complex fields, the modes
kept by the eigenvalue filter, and the random draws of Lyapunov spectra (all below).

### Added

- **Conatus Studio, local**: `python -m conatus studio` serves live runs, telemetry and controls on
  your machine (`pip install "conatus[studio]"`; on a cluster, a batch job and an ssh tunnel). It
  is experimental (see below); a hosted hub is not part of this release.
- **An interface for AI agents**: `python -m conatus agent` (`info`, `explain <name>`,
  `check <script.py>`), with `AGENTS.md` at the root of the repository.
- **Eigenvalues in a region of the complex plane**:
  `m.eigenvalues(N, ..., region=cn.box(re=(a, b), im=(c, d)))` returns every eigenvalue of the
  discretised problem in the box, with `res.count` certified by the argument principle (computed
  from the sparse LU, never the determinant itself) and the eigenvalues from Beyn's
  contour-integral method, each validated by its residual. `cn.contour.count` and
  `cn.contour.beyn` do the same for a pencil, a Floquet-Hill operator or a function `z -> T(z)`
  (nonlinear eigenproblems).
- **Experimental features are marked**: `cn.errors.ExperimentalWarning`, once per process, on the
  first use of a feature that is shipped and tested but not promised (the Studio,
  a spherical shell larger than 64 × 48 × 48, `ShootingRPO`, `PipeTravellingWave`); the guide
  page "Experimental features" lists them.
- One-line `repr` of simulations, solvers and `NewtonKrylovHookstep`; the under-resolved warning
  of an eigenvalue states the accuracy to expect (about sqrt(tail) relative, an order of
  magnitude, not a bound).

### Deprecated

- `conatus.expert` exports only the names the guide documents. Any other name written as
  `conatus.expert.<name>` still works, with a `DeprecationWarning` naming the module to import it
  from, and stops working in 0.3. The Studio protocol moved to `conatus.telemetry`, the BibTeX
  strings to `conatus.methods.BIBTEX`.
- The `__all__` of the public modules lists documented names only; the other names stay
  importable at the same path.

### Security

- `run.reproduce()` of a decorated call no longer imports and calls the `module:function` named by
  a run record that another account can write (the record, its run directory or `runs/` not owned
  by you, or writable by group or others): a typed `OptionError` names the two ways out (`fn=`,
  `chmod go-w`). New run directories are created without group/other write whatever the umask, so
  your own records are never refused; a deliberately group-writable shared registry needs `fn=`.
- `from_dedalus` refuses private and dunder names of `dedalus.public` (`d3.__dict__`, `d3.tr`,
  `d3.__reduce_ex__`) and attributes of the distributor outside a whitelist: they handed the
  script the translator's own interpreter.
- Run records redact the value of credential-like command-line options (`--token X`,
  `--api-key=X`) in `invocation.argv`.
- File names in the run registry (`Registry.run_dir`, `Run.file_path`, `save_file`) cannot leave
  the run directory; a forged inline array in a record (shape not matching its values, too long,
  negative dimension, non-numeric dtype) is refused; temporary `.npz` names are random
  (`mkstemp`); the adjoint journal refuses file names that are not its own generation files.

### Correctness (wrong results now fixed)

Simulation and models:
- A field declared complex is solved in a complex system (its imaginary part was dropped).
- The default dealiasing factor counts spatially varying given fields as factors of a product
  (`U*u*u` is cubic: factor 2, not 3/2), and a function of given fields alone (`exp(G)`, `1/G`) as
  one factor (`exp(G)*u**3`: 5/2); `explain()` states the condition (fields truncated at N modes)
  instead of a universal "exact (alias-free)". **This changes the default grid** of a model whose
  equations multiply given fields with a total degree of 3 or more (printed by `explain()`); pass
  `dealias=3/2` to keep the 0.1.0 grid.
- `plan.solve(data)` with new data for a coefficient of the operator (`q` in `q*u = f`) rebuilds
  the operator instead of changing only the right-hand side, and the residual is that of the
  updated problem (mapped, Cartesian, polar and spherical plans). Data are compared by value (an
  array changed in place is new data); the last rebuilt plan is reused when the same coefficient
  is passed again.
- Integrals on a sine/cosine axis (`walls(a, b, basis="sincos")`) are over the physical interval
  [a, b], exact for both parities, in `evaluate()`, the dye diagnostics and equations (they were
  over the doubled period).
- `EmbeddedError(adapt=True)` rejects a step whose error estimate exceeds `tol` and retries it with
  a smaller `dt` (the step used to be kept, only the next `dt` changed); `ConvergenceError` when
  `tol` cannot be met at `dt_min`.
- SBDF3 and SBDF4 with the default `startup="lower_order"` converge at order 2, and now say so:
  `solver.effective_order` and `explain()` give the effective order,
  `scheme_description(name, effective=)` gives the formal and the effective order;
  `startup="rk443"` keeps the nominal order.
- A failing model declaration (`equations`, `at`, `given`, `dye`, `define`, `function`,
  `eigenvalue`) leaves the model unchanged, so the corrected call works.
- User functions (`m.function`) work on the sphere; tensor boundary conditions keep their rows
  whatever their scale.
- A user function name belongs to its model: another model, or a re-run notebook cell, can declare
  its own `g`.
- No false "under-resolved" warning for a field exactly represented on very few modes.
- `sim.particles`: splitting a run into several `run()` calls no longer lowers the order.
- Ensembles stop exactly at `max_steps`; counts, indices and labels are never truncated;
  `Ensemble.states()` and velocity snapshots are copies.
- Checks that reported success without having measured it now say what they found: NaN or
  infinite data make `resolution_report` and `estimate_modes` say "non-finite" (an all-NaN field
  was "all resolved") and give a NaN or infinite CFL; `verify()` measures the tail of every
  component of a vector unknown and fails a tail it could not measure; `critical(over=)` says
  when the minimum sits at an end of `over`; `time_average` flags an unreliable batch-means error;
  NaN residuals are kept, never shown as small.

Analysis:
- Floquet–Hill operators: a singular intermediate block no longer gives NaN (a pivoted banded LU
  takes over), and `solve` checks its backward error.
- Generalized eigenvalue filtering is invariant under a scaling of the equations.
- `uncertainty()`: the spatial error uses the actual resolution increments; `trusted` requires
  `scheme_order` and a consistent refinement.
- `max_growth()` compares the ends of the interval; a `CertifiedRoot` is never certified when
  unconverged.
- POD reports the fraction of the total energy; the low-level optimal response is the propagated
  state.
- Continuation of boundary-value problems converges at the fold of a uniform state (the residual
  scale includes the parameter terms).
- `from_dedalus`: `**options` in calls are expanded, a problem's own `namespace={...}` resolves its
  equations, NumPy keywords in initial data are kept, and `complete` is true only without
  unsupported or approximated lines. `DedalusTranslation.to_script()` writes a runnable program (it
  replays the translation and runs it).

Results that changed silently with unchecked inputs (now refused with a typed error naming the fix):
- `ncc_cutoff=nan` reduced every non-constant coefficient to its mean; `P.least_stable(re=...)` on
  a pencil in `Re` used the declared Re; `time_parameter=` naming no parameter froze the forcing;
  the CFL step of a 1D scalar velocity was computed from its first grid value; a complex `M` in
  `generalized_eigenvalues` was cast to float; switches given as text were truthy
  (`complex="no"` was True); 2D or complex series in `cn.time_average` were flattened or cast;
  decreasing snapshot times gave a negative `dt`; `Decimal` parameter values compiled as complex;
  a shift-invert `n_modes` above the pencil size was clipped.
- `plan.params` writes are checked, and a change after the group matrices were assembled is
  refused (the cached groups kept the old value).
- A model that declares `t` itself and uses it in a time-dependent model gets a
  `DefaultChoiceWarning` (its `sin(t)` is constant).

State shared between calls:
- A solver keeps at most `FACTOR_CACHE_SIZE` (16) factorizations, least recently used dropped
  first: an adaptive step (`EmbeddedError(adapt=True)`) or a search over the period kept one more
  set per step until memory ran out (CPU Cartesian, polar, spherical and batched solvers).
- Package sources (or the compiled core) changed after `import conatus` are declared by
  `provenance.code_state()` (`changed_since_import`, not reproducible, "restart the interpreter"),
  and such a run is never a cache source or a cache hit: the hashes named the files on disk, not
  the code that ran. The change is judged by the content of the modules the process imported
  (sha256 recorded at import), not by modification times: an immediate edit or a deleted module is
  seen, a `touch` or a checkout back to the same text is not a change.
- `diagnostics.Grid.weights()` and `cell_volumes()`, `sim.coordinate("theta")` on the sphere,
  `sim.coeffs()` on a mapped domain, `EVPMode.grid/coeffs/coordinate` and
  `PipeTravellingWave.phi` return the caller's own arrays (an in-place edit changed later
  integrals, coordinates or the simulation's state); the sphere's cached grid and the cached Jacobi
  operators are read-only.
- `sim.flowmap(T)(X0)` leaves the simulation where it was (state, time, iteration, multistep
  history); it used to move it to the map's end point.
- A user pointwise function redeclared with another `value_at_zero` no longer overwrites the value
  of the first declaration; user pointwise functions nothing uses any more are released (a cell
  run again with a new lambda kept every earlier closure alive).
- The parallel group assembly (`CONATUS_BUILD_WORKERS`) no longer goes through a module global (two
  concurrent builds could exchange their groups).

Equivalent computations give the same answer:
- `solver.effective_order` and `explain()` include the boundary-data convention:
  RK222/RK443/RKSMR/CNAB2/CNLF2 with time-dependent wall data under `bc_time="step"` are first
  order, RK443 under `"stage"` second order (they said "global order 3"); the mapped-box stepper
  states its order (SBDF3/SBDF4 and RK443 with moving walls: 2).
- The spurious-mode drift test is relative to the eigenvalue, floored at a fraction of the
  spectrum's median, instead of max(|sigma|, 1): the modes kept no longer depend on the unit of
  the eigenvalue.
- Dense eigenvalue solves equilibrate the rows of the pencil: an equation multiplied by a constant
  no longer loses digits (Orr-Sommerfeld times 1e-8 lost every mode).
- The rank check of a system equilibrates rows and columns: a well-posed equation written times
  1e8 is no longer refused as singular.
- Newton (boundary-value problems) and its continuation measure the relative residual of each
  row in its own unit: an equation multiplied by a constant converges as the original.
- Lyapunov spectra draw each trajectory's tangent vectors from the seed and its start: a
  trajectory's exponents no longer depend on its place in the batch (random draws differ from
  0.1.0).

Low-level solvers (`conatus.runtime`):
- `BorderedBandedBatch` takes its dtype from every matrix, not the first.
- `ImexIntegrator` (SBDF) no longer mixes the history of a previous run into a new one: `run`
  continues from the returned (x, t) or refuses; new `reset()`. A callback that refers to its
  integrator is collected by the garbage collector (it leaked).

Identity of a computation (cache, checkpoints, provenance, run registry, HDF5):
- The cache key, checkpoint restart, checkpoint directory guard, provenance record and HDF5 append
  share one identity of the computation, which includes the data of every given field by content
  and user functions by their code: two forcings no longer share a cache key or restart into each
  other; `allow=("f",)` names a deliberate change of a given field.
- The provenance record includes `time_parameter`, `bc_time`, `f_backend`, complexify/decouple and
  pivoting. `reproduce()` compares bits (dtype, sign of zero) and the whole end state (history and
  clock), and keeps positional-only arguments. A record written by 0.1.0 lacks these keys:
  `reproduce()` names them in an `IdentityWarning` and still compares the end state bit for bit.
- Cache of decorated functions: closure values and the globals read are part of the key, with the
  attributes read from a user module or a user class; the user's own decorators under `@track` are
  called (they were skipped) and keyed. A cache hit returns the recorded types, and its series are
  readable from the new run.
- Simulation cache: the `series` functions and `series_every` are part of the key (a hit with other
  series returned the source run's).
- Run registry: series are streamed to disk during the run; `close(timeout=)` is one deadline; a
  malformed `run.json` no longer breaks `find()`.

### Refusals (silent before, loud now)

- Complex coefficients or initial data on fields declared real: `DtypeError` (they were cast).
- About 40 public entry points refuse NaN, infinity, text, flags, wrong shapes and unknown
  names or keywords with a `conatus.errors` class (each a subclass of the built-in type raised
  before); 0-d arrays, `Decimal`/`Fraction` values and 0/1 switches are accepted.
- `sim.save("x.hdf")` (or `.nc`, ...) is refused instead of writing a NumPy file `x.hdf.npz`; a
  Revolve budget of `checkpoints=0` is refused (it was the schedule of 1 and stored a state); the
  polar and spherical solvers check `time_parameter=` and `reduced=` like the Cartesian one;
  `eigenvalues(method=...)` and the shift-invert target are checked before anything is assembled;
  `optimal_gain` checks its arguments before building the operator.
- `m.params |= {...}` with NaN or undeclared names; an unknown axis in `dealias={...}`;
  `dealias=nan` or `inf`; non-finite scalar initial data; repeated macro arguments; keyword-only
  data functions; complex or non-finite initial data, and `dt=inf`, on a mapped domain.
- `EmbeddedError`: `tol`, `dt_min` and `dt_max` are validated first.
- The integral of a sine series inside an equation.
- `sim.dye()` checks what it can compute before creating its file (refused on curvilinear
  geometries); `sim.cfl()` and `CFL` state their Cartesian contract and refuse curvilinear
  coordinates.
- `output(..., mode="a")` refuses to append another computation (other parameters, data,
  tasks...).
- Checkpoints: a refused write is refused again on retry; foreign files in a checkpoint chain
  directory are refused before anything is deleted; `cn.io.latest` ignores files that are not
  Conatus checkpoints.
- Cache of decorated functions: a value without a content hash, or a container that contains
  itself, disables the cache for that call, with a warning.
- Run registry: a bad gzip `level` is refused at the call.
- `from_dedalus`: a time loop that changes the simulation is reported unsupported;
  `to_script()` refuses untranslated lines.
- `normal_form` refuses a point that is not an equilibrium; DMD refuses zero or non-finite data
  early.
- Low-level solvers: complex data is refused, never cut to its real part, by real factors and by a
  real `ImexIntegrator` (matrices, `x0`, `F(x, t)`); a right-hand side of the wrong shape is refused
  instead of reshaped; a matrix with NaN or Inf is refused (it reported a condition number of 0).
- C++ core: every LAPACK return code is checked, and array sizes are checked for overflow.

### Documentation and website

- A ten-minute getting-started page (one equation, one plot, one physical number, four guided
  changes); the website shows the same page as the documentation.
- Claims corrected: run recording and the resolution check are on request; a table says which GPU
  paths have run on GPU hardware; the licences of the bundled GCC runtimes are stated exactly;
  digits are given as absolute and relative differences; the spin convention of tutorial 8; stale
  limits removed.
- Tutorial 11 no longer touches the working directory; the `tutorials` extra includes xarray.
- Tutorial outputs and figures re-executed with this release.
- Every gallery script asserts its published value; the user API page lists every public name.
- Install commands pinned to the release and copyable; films can be paused; the home-page
  animation is labelled as a browser illustration; search links open real sections and say when
  results are capped.
- The acceptance scripts of the review ship in `acceptance/findings/`: run them against an
  installed wheel with `python acceptance/findings/run_all.py --installed`.

### Known limitations

- Unchanged from 0.1.0: the CPU path is a NumPy reference implementation (slower than established
  CPU codes on large grids); the GPU path and the CUDA 12 wheel are in beta (the wheel needs a
  driver supporting CUDA 12.9); one CPU process or one GPU per simulation; set `OMP_NUM_THREADS`
  as `python -m conatus doctor` advises; the 0.1 API may still change, through a
  `DeprecationWarning`.
- SBDF3 and SBDF4 reach their nominal order only with `startup="rk443"`.
- Time-dependent wall data: RK222, RK443, RKSMR, CNAB2 and CNLF2 are first order under
  `bc_time="step"`; RK443 is second order under `bc_time="stage"` (`solver.effective_order`
  says which).
- `sim.cfl()`, `CFL` and `sim.dye()` are Cartesian only.
- Records written by 0.1.0 are reproduced with a warning: the keys 0.1.0 did not record are not
  compared.

## [0.1.0] - 2026-10-12

First public release. Linux x86-64, CPython 3.11–3.13. BSD-3-Clause.

### What ships

- **Models written as on paper**: `cn.Model` on boxes, channels, disks, annuli, pipes, spheres,
  balls and spherical shells; unknowns, parameters and equations as text; boundary conditions by
  boundary name; every decision (bases, boundary-condition placement, gauge, implicit/explicit
  split, dealiasing) printed by `explain()`.
- **Simulation**: IMEX time steppers (SBDF1–4, Crank–Nicolson family, RK222, RK443) on the CPU,
  and on one NVIDIA GPU (beta, `backend="cuda"`); checks for non-finite states, spectral tails
  and aliased initial data while stepping; self-describing HDF5 outputs and checkpoints.
- **Linear stability**: eigenvalues with residuals and resolution drift, adjoint modes,
  critical parameters (`critical`, `least_stable`), non-modal growth of steady flows.
- **Floquet–Hill** analysis of time-periodic linear systems, with a residual for every exponent.
- **Invariant solutions and continuation**: Newton–Krylov with hookstep, equilibria, travelling
  waves, periodic orbits of one-dimensional models, pseudo-arclength continuation, bifurcation
  detection and branch switching.
- **Dynamics and data**: Lyapunov spectra and covariant Lyapunov vectors, tracers and dye, DMD and
  SPOD.
- **Run registry** (`cn.runs`): record, query and compare runs.
- Documentation: tutorials, user guide, API reference and the mathematics of each method.

### Install

`pip install conatus` (CPU wheel from PyPI, compiled core included). The CUDA 12 wheel (beta,
version `0.1.0+cu12`) is attached to the GitHub release only; install that file with the `[gpu]`
extra (CuPy). From source: `pip install .` with LAPACK/BLAS available.

### Known limitations

- The CPU path is a NumPy reference implementation: slower than established CPU codes on large
  grids. Speed comes from the GPU path, which is in beta (tested on NVIDIA A100 and L40S; the CUDA
  wheel needs a driver supporting CUDA 12.9).
- One CPU process or one GPU per simulation: no multi-GPU or MPI execution.
- NumPy, SciPy and the core each load OpenBLAS: set `OMP_NUM_THREADS` (about half the CPUs) to
  avoid oversubscription; `python -m conatus doctor` prints the value to set.
- The 0.1 API may still change; changes go through a `DeprecationWarning`.

Features under development are listed on the website's
[Coming next](https://rubendahan.github.io/conatus/coming-next/) page.
