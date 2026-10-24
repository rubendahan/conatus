# Input and output

Conatus writes two kinds of HDF5 files. **Analysis files** hold what you asked to see (fields,
derived quantities, numbers) at a fixed cadence. **Checkpoints** hold everything needed to continue
a run exactly. Both are readable with `h5py` alone. Both carry the provenance record of the run
({doc}`reproducibility`).

The recipes come first. The rules behind them are in the reference sections after.

## Recipe: a run split over several jobs

On a cluster a long run is a chain of jobs, and every job runs the same script.
`sim.resume_or_start(directory, u=...)` continues from the latest checkpoint of `directory`, or
starts from the given initial state when there is none. An output without `mode=` follows the same
branch: its file is created on a fresh start and continued after a restart.

```{literalinclude} _check_features.py
:language: python
:start-after: "# [io]"
:end-before: "# [/io]"
:dedent: 4
```

```{literalinclude} outputs/io_hdf5.txt
:language: text
```

The second job starts from the checkpoint of step 300. `sim.explain()` says which branch each job
took and which mode each file got. Delete `burgers_chain/` to start the chain again from t = 0.
The check script compares these two jobs with one uninterrupted run of 500 steps: the final states
are equal bit for bit.

## Recipe: continue a state with another parameter

A restart refuses a checkpoint written with another parameter value. To continue the state at a
new value on purpose, name the change with `allow=`:

```{literalinclude} _check_features.py
:language: python
:start-after: "# [io-allow]"
:end-before: "# [/io-allow]"
:dedent: 4
```

```{literalinclude} outputs/io_allow.txt
:language: text
```

The allowed change is printed by `later.explain()` and recorded in every later checkpoint.

## Recipe: a run that starts at a time other than zero

`sim.initial(u=..., t=t0)` gives the state at the time `t0`. Forcing, boundary data, outputs and
checkpoints then use that clock:

```{literalinclude} _check_features.py
:language: python
:start-after: "# [initial-time]"
:end-before: "# [/initial-time]"
:dedent: 4
```

```{literalinclude} outputs/initial_time.txt
:language: text
```

## Recipe: a long adjoint computation

The reverse sweep of an adjoint over a long time can also be split over several jobs, with its own
checkpoints kept on the GPU, in host memory or on disk: see {doc}`dynamics/adjoint`.

## Reference

### Analysis files

`sim.output(path, tasks=..., every=k)` writes one row every `k` steps.

- By default (`initial=True`) the first row is the state at the start of the run, so a time series
  starts from the initial condition. A restarted job does not write that row again. A start
  between two multiples of `k` writes none.
- With `initial=False` the first row is at step `k`. The output's `explain()` says which applies.

A task is one of:

- the name of an unknown or of a component (`"u"`, `"u.z"`);
- an expression written as in the equations (`"dx(u)"`, `"0.5*u@u"`);
- a function of the simulation, in a dictionary (`{"KE": lambda s: ...}`), for anything else.

| In the file | Content |
|---|---|
| `/scales/sim_time`, `/scales/iteration`, `/scales/write_number` | one entry per write |
| `/scales/<coordinate>` | the grid points of each coordinate |
| `/scales/component` | the coordinate names: the component axis of vector tasks |
| `/tasks/<name>` | one row per write: grid values, or the number returned by your function |
| attributes | problem, scheme, `dt`, `every`, the source of each task, provenance, `conatus_version`, `parameters`, `coordinates`, `geometry` |

`cn.io.read(path)` returns the times, iterations and tasks as arrays, and `explain()` describes
the file.

### Reading the files with other tools

The datasets carry coordinate labels, component names and dimension scales as standard HDF5
attributes, so other tools see named arrays on coordinates:

- every task has a `dims` attribute (for example `["t", "component", "x", "z"]`) and each of its
  dimensions is labelled and attached to its coordinate dataset as an HDF5 **dimension scale**
  (`t` to `/scales/sim_time`, `x` to `/scales/x`, `component` to `/scales/component`); vector
  tasks also carry `component_names`;
- `cn.io.read(path).to_xarray()` gives an `xarray.Dataset` (dimensions `t`, `component` and the
  coordinates; `iteration` along `t`; parameters and version as attributes). xarray is optional:
  without it the call says how to install it;
- for a **2D or 3D box or channel**, `sim.output` also writes an **XDMF sidecar** `<file>.xmf` at
  the end of every `sim.run` call: open it in ParaView (or VisIt) to see every real task on the
  grid over time, vector tasks as one array per component (`u.x`, `u.z`). The sidecar points into
  the HDF5 file (no copy). ParaView's X axis is the **last** coordinate (HDF5 arrays are stored in
  coordinate order, row-major): for an `(x, z)` box, z is along ParaView's X. For the file of a job
  that was killed, `cn.io.write_xdmf(path)` writes the sidecar for the rows it holds.
- **Not covered by the sidecar**: curvilinear geometries (disk, annulus, pipe, sphere, ball,
  shell: their arrays are self-describing in HDF5 and xarray, but no XDMF mesh of their points is
  written yet), mapped boxes with curved walls, 1D boxes, complex tasks (XDMF has no complex
  type), and tasks that are not on the grid (numbers returned by a function). The output's
  `explain()` says when no sidecar was written and why.

Checkpoints are labelled the same way: `/fields/<name>` (spectral coefficients, dimensions
`<coordinate>_mode`) and, with `grids=True`, `/grids/<name>` on `/scales/<coordinate>`;
`cn.io.read(path).to_xarray()` works on them too. Files written by earlier versions have none of
these labels: they still read and restart, and `to_xarray()` names only the dimensions whose
length identifies a coordinate (the rest are `<name>_dim<k>`). Native models declare no units, so
no unit attribute is written.

### Modes of an existing output file

An existing file is replaced or extended only when you say so:

- `mode="w"` replaces it;
- `mode="a"` appends to it (time must go forward);
- `mode="resume"` continues it after a restart from a checkpoint older than its last rows (the
  previous job was killed between two checkpoints). The rows after the restart are dropped first,
  `explain()` prints their number, and the restarted run writes them again.

On a fresh start of `resume_or_start`, an existing file is refused. Use `mode="w"` to replace it,
for example after a first job killed before its first checkpoint.

One file holds one computation. An append is refused, with the list of what differs, when any of
these differs from the computation the file records: the model, equations, parameters, given-field
data, resolution, box, scheme, `dt`, `every`, or the source of a task. A task function is compared
by its code and the values it reads. A function that reads an object without a content hash is
compared by its code only, and `explain()` says so.

`every` is a whole number of steps. The time step is not adjusted to land on an output time.

The file is closed after each write, so a job killed during a write leaves every completed row
readable.

### Writing and finding checkpoints

- `sim.save(path)` writes a checkpoint now. The suffix chooses the format: `.h5` (HDF5) or `.npz`
  (NumPy). Both restart to the same numbers.
- `sim.checkpoints(directory, every=k, keep=n)` writes `checkpoint_<iteration>.h5` every `k` steps
  and keeps the `n` most recent. The default `keep=2` keeps the one before the latest, in case a
  job is killed while writing.
- `sim.restart(path)` continues from a checkpoint.
- `cn.io.latest(directory)` returns the most recent complete checkpoint of a directory.
  `resume_or_start` uses the same search.

How `cn.io.latest` chooses:

- it reads the iteration from the file, not from the file name;
- it finds both `checkpoint_<iteration>.h5` and `checkpoint_<iteration>.npz` (written with
  `sim.save(directory / f"checkpoint_{sim.iteration:09d}.npz")`);
- it ignores a file still being written, an HDF5 file of another format, and a `.npz` without the
  state;
- it refuses two files of the same iteration in the two formats (keep one format per chain);
- it returns `None` for an empty directory and refuses a directory that does not exist, so a
  misspelt path does not restart a chain from t = 0.

With `grids=True` a checkpoint also stores the grid values of each unknown, for reading with
`h5py`.

### What a restart continues

A checkpoint holds the state, the history of a multistep scheme and the clock. The restarted run
gives the same numbers as the uninterrupted run, bit for bit, for every scheme
({doc}`timesteppers`), on the CPU and on a GPU.

The restarting simulation must use the same numerics: scheme, `dt`, resolution, start-up and
options. Restart rejects incompatible numerical settings and lists the differences.

### What a restart compares

Every checkpoint records the problem it belongs to:

- the model name and a hash of the compiled equations;
- the parameter values;
- the data of every given field, by content (a forcing `f = 1` is not `f = 2`);
- the user functions (`m.function`), by their code;
- the resolution and the bounds of every direction;
- the start time of its chain.

A restart refuses any difference and lists what differs, for example
`parameter kappa: 1 in the checkpoint, 5 now`, `given field f: its data differ ...`,
`geometry along x: ...` or `equations: hash ... (another equation, term or sign)`. A job chain
whose script was edited between two jobs therefore cannot mix two problems in one output.

### Allowing a change

`sim.restart(path, allow=...)` and `resume_or_start(directory, ..., allow=...)` accept:

- a parameter, given-field or function name, such as `allow=("kappa",)`;
- a group: `"parameters"`, `"data"`, `"functions"`, `"model"`, `"equations"` or `"geometry"`.

A change of resolution cannot be allowed. The allowed change is printed by `explain()` and recorded in
every later checkpoint.

### A checkpoint directory used by two runs

Before anything is written or deleted, every checkpoint already in the directory is checked against
the one the run restarted from. A directory that holds checkpoints of another run is refused. A
refused write is refused again on every retry.

### Checkpoints written by earlier versions

A checkpoint written before the problem record restarts with a `CheckpointWarning`. The warning
says that only the problem name and the numerics could be compared. For a checkpoint written by
0.1.0, it says that its given-field data were not compared.

### Start time

- With `sim.initial(u=..., t=t0)`, time-dependent forcing and boundary data are evaluated from
  `t0`, and `sim.t` starts there.
- `run(until=...)` is read on that clock. Output rows and checkpoints carry the times `t0 + k dt`.
- `explain()` states the start time, also after a restart.
- `resume_or_start(directory, u=..., t=t0)` refuses to continue a chain that started at another
  time, since the two scripts would describe different runs.
- A checkpoint written before the start time was recorded is continued only without `t=`. Native
  runs started at t = 0 until then.
- A non-finite or non-real `t` is refused.
- In a model with an unknown named `t`, there is no time `t`: `initial(t=...)` gives that unknown's
  data.

## Not yet

- Expressions as tasks in curved geometries (disk, sphere, shell, ...): write a function of the
  simulation instead.
- One file shared by several MPI ranks (parallel HDF5).
