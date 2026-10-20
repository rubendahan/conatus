# Input and output

Two kinds of HDF5 files: **analysis files**, which hold what you asked to see (fields, derived
quantities, numbers) at a fixed cadence, and **checkpoints**, which hold everything needed to
continue a run exactly. Both are readable with `h5py` alone, and both carry the provenance record
of the run (code version, environment, every numerical choice: {doc}`reproducibility`).

## A run split over two jobs

On a cluster a long run is a chain of jobs, and every job runs the same script.
`sim.resume_or_start(directory, u=...)` continues from the latest checkpoint of `directory` when
there is one, and starts from the given initial state otherwise (first job; the directory is
created). An `output` without `mode=` then follows the same branch: its file is created on a fresh
start (an existing file is refused there: `mode="w"` replaces it explicitly, e.g. after a first
job killed before its first checkpoint) and continued after a restart (`mode="resume"`: rows written after the checkpoint are
dropped, the restarted run writes them again). `sim.explain()` says which branch was taken and
which mode each file got:

```{literalinclude} _check_features.py
:language: python
:start-after: "# [io]"
:end-before: "# [/io]"
:dedent: 4
```

```{literalinclude} outputs/io_hdf5.txt
:language: text
```

The last line is the check made after the example: the two jobs give the same numbers as one
uninterrupted run, bit for bit.

## Analysis files

`sim.output(path, tasks=..., every=k)` writes every `k` steps, and the state at the start of the
next run first (`initial=True`, the default: a time series starts from the initial condition, as
in Dedalus; a restarted job does not write that row again, and a start between two multiples of
`k` writes none; `initial=False` starts at step `k`; the output's `explain()` says which). A task is the name of an unknown or
of a component (`"u"`, `"u.z"`), an expression written as in the equations (`"dx(u)"`,
`"0.5*u@u"`), or a function of the simulation in a dictionary (`{"KE": lambda s: ...}`) for
anything else.

| In the file | Content |
|---|---|
| `/scales/sim_time`, `/scales/iteration`, `/scales/write_number` | one entry per write |
| `/scales/<coordinate>` | the grid points of each coordinate |
| `/scales/component` | the coordinate names: the component axis of vector tasks |
| `/tasks/<name>` | one row per write: grid values, or the number returned by your function |
| attributes | problem, scheme, `dt`, `every`, the source of each task, provenance, `conatus_version`, `parameters`, `coordinates`, `geometry` |

`cn.io.read(path)` returns the times, iterations and tasks as arrays, and `explain()` describes
the file.

## Reading the files with other tools

The files describe themselves with plain HDF5 means, so other tools see named arrays on
coordinates:

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

Conatus does not decide for you:

- `every` is a whole number of steps: the time step is never adjusted to land on an output time;
- an existing file is never replaced or extended without your saying so: `mode="w"` (replace),
  `mode="a"` (append: time must go forward) or `mode="resume"`. One file holds one computation:
  an append is refused, with the list of what differs, when the model, equations, parameters,
  given-field data, resolution, box, scheme, `dt`, `every` or the source of any task differs from
  the computation the file records (a task function is compared by its code and the values it
  reads; one reading an object without a content hash is compared by its code only, said by
  `explain()`);
- `mode="resume"` is for a job that restarts from a checkpoint older than the last rows of the file
  (the previous job was killed between two checkpoints): the rows after the restart are dropped
  first, their number is printed by `explain()`, and the restarted run writes them again;
- the file is closed after each write, so a job killed during a write leaves every completed row
  readable.

## Checkpoints and exact restarts

`sim.save(path)` writes a checkpoint now; `sim.checkpoints(directory, every=k, keep=n)` writes one
every `k` steps and keeps the `n` most recent (`keep=2` by default: the latest and the one before
it, in case a job is killed while writing). `sim.restart(path)` continues from a checkpoint and
`cn.io.latest(directory)` finds the most recent complete one, in either format:
`checkpoint_<iteration>.h5` (written by `sim.checkpoints`) and `checkpoint_<iteration>.npz`
(`sim.save(directory / f"checkpoint_{sim.iteration:09d}.npz")`). The iteration is read from the file, a file still being written is ignored, two files of the
same iteration in the two formats are refused (keep one format per chain), an empty directory gives
`None`, and a directory that does not exist is refused: a misspelt path never restarts a chain from
t = 0. `resume_or_start` uses the same search.

A checkpoint holds the state, the history of a multistep scheme and the clock, so the restarted run
is the uninterrupted run bit for bit, for every scheme ({doc}`timesteppers`), on the CPU and on a
GPU: both formats are tested bit for bit on a device, including periodic `.h5` checkpoints restarted
from `cn.io.latest` with a one-step and a multistep scheme and the CUDA graph replayed after a
restart. The restarting simulation must use the same numerics (scheme, `dt`, resolution, start-up,
options): a difference is refused with the option to change, never adapted. It must also be the same
problem: every checkpoint records the model name, a hash of the compiled equations, the parameter
values, the data of every given field (by content: a forcing `f = 1` is not `f = 2`), the user
functions (`m.function`, by their code), the resolution and the bounds of every direction, and a
restart refuses any difference with the list of what differs (`parameter kappa: 1 in the
checkpoint, 5 now`; `given field f: its data differ ...`; `geometry along x: ...`;
`equations: hash ... (another equation, term or sign)`), so a job chain whose script was edited
between two jobs cannot mix two problems in one output. To continue a state at another value
deliberately, name it: `sim.restart(path, allow=("kappa",))` (a parameter, given-field or function
name, or `allow="parameters"`, `"data"`, `"functions"`, `"model"`, `"equations"`, `"geometry"`; the
resolution can never be allowed). The allowed change is printed by
`explain()` and recorded in every later checkpoint. A checkpoint written before this record restarts
with a `CheckpointWarning` saying that only the problem name and the numerics could be compared (one
written by 0.1.0: that its given-field data were not compared). A checkpoint directory that holds
files of another run is refused as well: every checkpoint already in the directory is checked
against the one the run restarted from before anything is written or deleted, and a refused
write is refused again on every retry. `cn.io.latest` considers only complete Conatus checkpoints
(an HDF5 file of another format, or a `.npz` without the state, is ignored). With `grids=True` a checkpoint also stores the
grid values of each unknown, for reading with `h5py`. The suffix chooses the format: `.h5` or `.npz`
(NumPy), which restart to the same numbers.

## A run that starts at a time other than zero

`sim.initial(u=..., t=t0)` gives the state at the time `t0` (default 0). Time-dependent forcing and
boundary data are then evaluated from `t0`, `sim.t` starts there, `run(until=...)` is read on that
clock, every output row and checkpoint carries the times `t0 + k dt`, and `explain()` states the
start time. Every checkpoint records the start time of its chain: `resume_or_start(directory,
u=..., t=t0)` refuses to continue a chain that started at another time (the two scripts would not
describe the same run), and a checkpoint written before this record is continued only without `t=`
(native runs started at t = 0 until then). A non-finite or non-real `t` is refused. In a model
with an unknown named `t` there is no time `t`, and `initial(t=...)` gives that unknown's data, as
before. Restarted from a checkpoint, `explain()` also says when the chain started.

```{literalinclude} _check_features.py
:language: python
:start-after: "# [initial-time]"
:end-before: "# [/initial-time]"
:dedent: 4
```

```{literalinclude} outputs/initial_time.txt
:language: text
```

## Long adjoint computations

The reverse sweep of an adjoint over a long time can also be split over several jobs, with its own
checkpoints kept on the GPU, in host memory or on disk: see {doc}`dynamics/adjoint`.

## Not yet

- Expressions as tasks in curved geometries (disk, sphere, shell, ...): write a function of the
  simulation instead.
- One file shared by several MPI ranks (parallel HDF5).
