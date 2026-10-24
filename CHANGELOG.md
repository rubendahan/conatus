# Changelog

All notable changes to Conatus.

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
