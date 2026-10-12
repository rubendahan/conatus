# Checking an installation

`python -m conatus doctor` says what your machine can run (compiled core, GPU) and, for
anything missing, what to do.

```bash
python -m conatus doctor          # text; exit code 0 if nothing required is missing, else 1
python -m conatus doctor --json   # the same, machine-readable
```

```python
import conatus as cn
rep = cn.doctor()          # DoctorReport: rep.ok, rep.explain(), rep.to_dict(); a table in Jupyter
```

Each line gives one of three statuses (ok / missing / warning), the version or value found, what it
is used for and, when something is missing, **what to do**. Each probe is short, runs in a
subprocess with a timeout where needed, and turns any failure into a report line. The whole report
takes about 2 s. Required: Python 3.11–3.13 (a newer, untested Python is a warning), NumPy, SciPy,
h5py (outputs and checkpoints) and the CPU part of the compiled core. The tutorials also need
matplotlib: a missing one is a warning with the exact line that installs it
(`pip install "matplotlib>=3.7"`, or `pip install "conatus[tutorials]"`).
Everything else is optional.

The **first line** says how Conatus is installed. In a *development checkout* (the repository's
`python/`) the advice is the developer's (`source env/env.sh`, `bash core/build.sh`) and `CN_TMP` is
checked; for an *installed package* the advice is pip's (`pip install .`, the CUDA build option)
and the line names the directories it writes to: the telemetry spool and the user cache
(`$XDG_CACHE_HOME/conatus`, by default `~/.cache/conatus`; `CN_TELEMETRY`, `CN_CACHE`, `CN_TMP`
override), never the Python environment itself.

| Check | What is looked at |
|---|---|
| compiled core (CPU / CUDA) | the `_core` module; built with CUDA; CUDA architectures read from the build (for example `sm_80`, `sm_89`), "unknown" for an installed wheel |
| GPU (`backend='cuda'`) | `conatus.gpu.gpu_report()`: CuPy sees a device **and** the core has CUDA with a device; warns if the device's `sm` is not among the built archs |
| CuPy | version |
| Dedalus | optional cross-check oracle (never imported by Conatus) |
| install | development checkout or installed package; telemetry spool and user cache directories |
| h5py | required: `sim.output`, `sim.checkpoints`, `cn.io` |
| matplotlib (tutorials) | the tutorials' figures; the install line if missing |
| ipywidgets | notebook widgets (optional) |
| `$CN_RUNS`, `CN_TMP` | free space; on CephFS the **project quota** is read (`ceph.quota.max_bytes`), because the file-system free space is the whole cluster's; warns under `RUNS_FREE_WARN_GB` = 10 GiB |

On a node without a GPU the report is OK, with the GPU line saying that CuPy sees no CUDA device
and to run on a GPU node; on a GPU node it names the device and its architecture.
