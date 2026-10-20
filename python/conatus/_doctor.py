"""``conatus doctor``: what this machine can run, and what to do about what is missing (v0.1).

``doctor()`` returns a :class:`DoctorReport`; ``python -m conatus doctor [--json]`` prints it and
exits 0 when nothing *required* is missing, 1 otherwise. Every check is a line with a status
(``ok`` / ``missing`` / ``warning`` / ``optional``), the detected version or value, and, when
something is missing, what to do. ``optional`` is an optional component that is absent (GPU, CuPy,
MPI, NCCL, Dedalus, ipywidgets, an undeclared run registry): nothing is wrong; ``warning`` is kept
for something that may be (a clean CPU install showed six warnings for nothing
wrong). A check never raises: a failing probe becomes a report line. Checks are
cheap (lazy imports, subprocesses with a timeout, no MPI initialisation in this process), but
importing CuPy alone can take ~2 s on a cold network file system.

Required: Python >= MIN_PYTHON (and below MAX_PYTHON, the first untested minor), NumPy, SciPy,
h5py (outputs and checkpoints: a declared dependency since), the compiled core (CPU
part). The tutorials also need matplotlib (``pip install ".[tutorials]"``), reported with the
exact install line. Everything else (GPU, CuPy, MPI, NCCL, Dedalus, ipywidgets) is optional.

The first line says how Conatus is installed (``conatus._paths.install_mode``): a development
checkout gets developer advice (the CMake build of the install page), an installed
package gets ``pip`` advice and the user directories it writes to.
"""
from __future__ import annotations

import html
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO

__all__ = ["doctor", "DoctorReport", "Check", "MIN_PYTHON", "RUNS_FREE_WARN_GB",
           "SUBPROCESS_TIMEOUT_S"]

MIN_PYTHON = (3, 11)              # pyproject.toml: requires-python = ">=3.11,<3.14"
MAX_PYTHON = (3, 14)              # first minor NOT tested (exclusive bound of requires-python)
# The tutorials' extra dependency and h5py: the exact lines that install them (pyproject).
TUTORIALS_FIX = 'pip install "matplotlib>=3.7" "xarray>=2023.1"   (or the tutorials extra: pip install ".[tutorials]")'
H5PY_FIX = 'pip install "h5py>=3.8"   (or: conda install -c conda-forge h5py)'
NOTEBOOK_FIX = 'pip install "ipywidgets>=8"   (from the source tree: pip install ".[notebook]")'
RUNS_FREE_WARN_GB = 10.0          # warn when the runs root / CN_TMP has less free space (GiB)
SUBPROCESS_TIMEOUT_S = 5.0        # every external probe (ompi_info, MPI library query) is bounded

THREADPOOLCTL_FIX = 'pip install "threadpoolctl>=3.1"   (optional: thread counts of the BLAS libraries)'
# environment variables that set the thread count of the BLAS / OpenMP libraries
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")

# the GPU install section, not one recipe (conatus.errors.GPU_INSTALL_FIX)
_PIP_GPU = ("follow the GPU install (docs/site/install.md#gpu-cuda-install): CuPy for your CUDA "
            "version (conda: cupy-core; pip wheel: the gpu extra), then run on a GPU node")


@dataclass
class Check:
    """One line of the report."""

    name: str
    status: str                   # "ok" | "missing" | "warning" | "optional" (absent, not needed)
    value: str                    # detected version / value ("-" when nothing was found)
    detail: str = ""              # what was looked at, why it is missing
    fix: str = ""                 # what to do (never done for you)
    required: bool = False

    def line(self) -> str:
        s = f"  [{self.status:8s}] {self.name:22s} {self.value}"
        if self.detail:
            s += f"  ({self.detail})"
        if self.fix and self.status != "ok":
            s += f"\n            -> {self.fix}"
        return s


@dataclass
class DoctorReport:
    """All checks, in order; ``ok`` is False when a required check is missing."""

    checks: list[Check]
    host: str
    elapsed_s: float
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(c.required and c.status == "missing" for c in self.checks)

    def check(self, name: str) -> Check:
        return next(c for c in self.checks if c.name == name)

    def explain(self) -> str:
        miss = [c.name for c in self.checks if c.required and c.status == "missing"]
        head = ("OK: nothing required is missing" if self.ok else
                "NOT OK: required missing: " + ", ".join(miss))
        ver = self.extra.get("conatus", "?")
        lines = [f"Conatus {ver} doctor on {self.host} ({self.elapsed_s:.2f} s): {head}"]
        lines += [c.line() for c in self.checks]
        lines.append("  required: Python, NumPy, SciPy, h5py, compiled core (CPU); the tutorials "
                     "also need matplotlib; the rest is optional ([optional]: absent, not needed)")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.explain()

    def to_dict(self) -> dict:
        return {"ok": self.ok, "host": self.host, "elapsed_s": self.elapsed_s,
                "checks": [asdict(c) for c in self.checks], **self.extra}

    def _repr_html_(self) -> str:
        esc = html.escape
        color = {"ok": "#2e7d32", "missing": "#c62828", "warning": "#ef6c00", "optional": "#616161"}
        rows = "".join(
            f"<tr><td>{esc(c.name)}{' *' if c.required else ''}</td>"
            f"<td style='color:{color.get(c.status, 'inherit')}'>{esc(c.status)}</td>"
            f"<td>{esc(c.value)}</td><td>{esc(c.detail)}</td>"
            f"<td>{esc(c.fix) if c.status != 'ok' else ''}</td></tr>" for c in self.checks)
        return (f"<div><b>Conatus doctor</b> on {esc(self.host)}: "
                f"{'OK' if self.ok else 'NOT OK'} (* required)"
                "<table><tr><th>check</th><th>status</th><th>value</th><th>detail</th>"
                f"<th>what to do</th></tr>{rows}</table></div>")


# ---------------------------------------------------------------- helpers
def _dist_version(name: str) -> str | None:
    """Installed distribution version without importing the package (None if absent)."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except Exception:             # noqa: BLE001 -- metadata unreadable: treated as unknown
        return None


def _run(cmd: list[str]) -> tuple[int, str]:
    """(returncode, stdout+stderr) of a bounded subprocess; (-1, reason) on failure/timeout."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_S)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, f"timed out after {SUBPROCESS_TIMEOUT_S:g} s"
    except Exception as err:      # noqa: BLE001 -- any failure of the command is reported, not raised
        return -1, f"{type(err).__name__}: {err}"


def _ceph_quota(p: Path) -> tuple[Path, float] | None:
    """(directory, free GiB) from the nearest CephFS quota (``ceph.quota.max_bytes`` minus
    ``ceph.dir.rbytes``) above ``p``; None when no ancestor carries one (not CephFS, no quota).
    On CephFS ``statvfs`` reports the whole cluster, not the quota that actually limits writes."""
    for q in (p, *p.parents):
        try:
            mx = int(os.getxattr(q, "ceph.quota.max_bytes"))
        except (OSError, ValueError, AttributeError):
            continue
        if mx > 0:
            used = int(os.getxattr(q, "ceph.dir.rbytes"))
            return q, (mx - used) / 2 ** 30
    return None


def _free_gib(path: Path) -> tuple[Path, float, str]:
    """(checked directory, free GiB, source) for the first existing ancestor of ``path``:
    the CephFS quota when there is one, else ``shutil.disk_usage``."""
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    p = p.resolve()
    quota = _ceph_quota(p)
    if quota is not None:
        return p, min(quota[1], shutil.disk_usage(p).free / 2 ** 30), f"CephFS quota on {quota[0]}"
    return p, shutil.disk_usage(p).free / 2 ** 30, "file system free space"


def _safe(name: str, fn: Callable[[], list[Check]], required: bool = False) -> list[Check]:
    """Run one probe; any exception becomes a report line (a check never crashes doctor)."""
    try:
        return fn()
    except Exception as err:      # noqa: BLE001 -- a probe never crashes doctor: it becomes a line
        return [Check(name, "missing" if required else "warning", "-",
                      f"probe failed: {type(err).__name__}: {err}", required=required)]


# ---------------------------------------------------------------- checks
def _python() -> list[Check]:
    v = sys.version_info
    ok = (v.major, v.minor) >= MIN_PYTHON
    lo, hi = ".".join(map(str, MIN_PYTHON)), ".".join(map(str, MAX_PYTHON))
    if ok and (v.major, v.minor) >= MAX_PYTHON:      # runs, but outside the tested range: said
        return [Check("Python", "warning", f"{v.major}.{v.minor}.{v.micro}",
                      f"{sys.executable}; not tested (tested: >= {lo}, < {hi})",
                      "create the env with a tested Python: micromamba create ... python=3.12",
                      required=True)]
    return [Check("Python", "ok" if ok else "missing", f"{v.major}.{v.minor}.{v.micro}",
                  sys.executable, "" if ok else f"use Python >= {lo}, < {hi} (3.12 recommended)",
                  required=True)]


def _install() -> list[Check]:
    """How Conatus is installed and where it writes by default (conatus._paths)."""
    from . import _paths
    d = _paths.describe()
    if _paths.in_checkout():
        detail = (f"python/ of {_paths.REPO_ROOT}; scratch data under $CN_TMP or scratch/ "
                  f"(telemetry {d['telemetry']})")
    else:
        detail = (f"{d['package']}; telemetry spool {d['telemetry']}; user cache {d['cache']} "
                  "(override: CN_TELEMETRY, CN_CACHE or XDG_CACHE_HOME); run registry: CN_RUNS "
                  "or root=")
    return [Check("install", "ok", d["mode"], detail)]


def _module(label: str, name: str, required: bool, purpose: str, fix: str,
            note: str = "", absent: str = "optional") -> list[Check]:
    """Presence + version of a Python package (module = distribution ``name``), found without
    importing it. A missing optional package has the status ``absent`` (``"optional"``: the
    tutorials' matplotlib too, an optional extra, external review 0.1.0 acceptance F102)."""
    found = importlib.util.find_spec(name) is not None
    ver = _dist_version(name) or ("installed (version unknown)" if found else "-")
    if found:
        return [Check(label, "ok", ver, purpose, required=required)]
    return [Check(label, "missing" if required else absent, "-",
                  purpose + ("" if required else note), fix, required=required)]


def _core(ctx: dict) -> list[Check]:
    """Compiled core: CPU part (required), CUDA part and the CUDA archs it was built for."""
    try:
        from . import runtime
    except Exception as err:      # noqa: BLE001 -- MissingDependencyError, or a broken .so
        fix = getattr(err, "fix", None) or _core_fix()
        return [Check("compiled core (CPU)", "missing", "-", str(err).splitlines()[0], fix,
                      required=True),
                Check("compiled core (CUDA)", "warning", "-", "compiled core not loaded")]
    core = runtime._core
    ctx["core"] = core
    from . import __version__
    built = getattr(core, "__version__", "?")
    if built == __version__:     # one version: shown once, in the header
        out = [Check("compiled core (CPU)", "ok", "same version as the package",
                     f"{runtime.CORE_SOURCE}: {runtime.CORE_FILE}", required=True)]
    else:                        # a core built from another checkout or an older build
        out = [Check("compiled core (CPU)", "warning", f"_core {built} (package {__version__})",
                     f"{runtime.CORE_SOURCE}: {runtime.CORE_FILE}; built for another version",
                     _core_fix(), required=True)]
    cu = getattr(core, "cuda", None)
    if cu is None:
        out.append(Check("compiled core (CUDA)", "optional", "not built",
                         "the core has no CUDA part (CONATUS_CUDA=OFF)",
                         _cuda_fix()))
        return out
    archs = _cuda_archs(runtime.CORE_FILE)
    ctx["archs"] = archs
    out.append(Check("compiled core (CUDA)", "ok", "built with CUDA",
                     "archs " + (", ".join(f"sm_{a}" for a in archs) if archs else
                                 "unknown (no build.ninja next to the module)")))
    return out


def _owner(path: str) -> str:
    """Which package a loaded shared library belongs to (``numpy.libs/`` -> numpy, ...)."""
    parts = Path(path).parts
    for i, part in enumerate(parts):
        if part.endswith(".libs") or part in ("site-packages", "dist-packages"):
            if part.endswith(".libs"):
                return part[:-5]
            if i + 1 < len(parts):
                return parts[i + 1]
    return str(Path(path).parent)


def _blas() -> list[Check]:
    """BLAS/LAPACK libraries loaded by NumPy, SciPy and the compiled core, and their thread counts
    (threadpoolctl, optional; without it the loaded libraries are read from /proc/self/maps and
    the counts are not known). Each package that ships its own OpenBLAS (pip NumPy, SciPy, the
    core of a binary wheel) has its own thread pool: one line lists them all."""
    import numpy  # noqa: F401 -- loads NumPy's BLAS
    import scipy.linalg  # noqa: F401 -- loads SciPy's own BLAS (a pip SciPy bundles one)
    try:
        from . import runtime  # noqa: F401 -- the compiled core's LAPACK/BLAS
    except Exception:          # noqa: BLE001 -- the core line already reports why it is missing
        pass
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    env = ", ".join(f"{k}={os.environ[k]}" for k in THREAD_VARS if k in os.environ)
    env = env or "no thread variable set"
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        libs = sorted({ln.split()[-1] for ln in open("/proc/self/maps")
                       if re.search(r"(openblas|mkl_rt|libblis|flexiblas|libblas|liblapack)", ln)})
        value = ", ".join(f"{Path(p).name} ({_owner(p)})" for p in libs) or "-"
        return [Check("BLAS threads", "optional", value,
                      f"{cpus} CPUs available; {env}; thread counts need threadpoolctl",
                      THREADPOOLCTL_FIX)]
    pools = threadpool_info()
    blas = [p for p in pools if p.get("user_api") == "blas"]
    value = "; ".join(f"{p.get('internal_api')} {p.get('version') or ''} ({_owner(p['filepath'])}): "
                      f"{p.get('num_threads')} threads" for p in blas) or "none loaded"
    omp = [p for p in pools if p.get("user_api") == "openmp"]
    detail = f"{cpus} CPUs available; {env}"
    if omp:
        detail += "; OpenMP: " + ", ".join(f"{Path(p['filepath']).name} {p.get('num_threads')} threads"
                                           for p in omp)
    return [blas_threads_check(blas, cpus, value, detail)]


def blas_threads_check(blas: list[dict], cpus: int | None, value: str, detail: str) -> Check:
    """The BLAS thread pools against the CPUs this process may use. Each pool starts one thread
    per CPU by default; the pip NumPy, the pip SciPy and the core of a binary wheel load three
    OpenBLAS copies, each then taking every CPU. Measured on 8 CPUs, the computations of tutorial
    6: 1185 s with the defaults (3 pools of 8 threads), 26 s with OMP_NUM_THREADS=1, 16 s with
    OMP_NUM_THREADS=4 (run). Warned: several pools of which one has
    as many threads as CPUs (or more), or one pool with more threads than CPUs."""
    cpus = cpus or 1
    full = [p for p in blas if (p.get("num_threads") or 0) >= cpus]
    over = [p for p in blas if (p.get("num_threads") or 0) > cpus]
    if not over and not (len(blas) > 1 and full and cpus > 1):
        return Check("BLAS threads", "ok", value, detail)
    per = max(1, cpus // 2) if len(blas) > 1 else cpus
    why = (f"; {len(blas)} pools, {len(full)} of them with a thread per CPU ({cpus} CPUs): they "
           "compete and dense solves can be many times slower" if len(blas) > 1 else
           f"; a pool with more threads than the {cpus} CPUs")
    return Check("BLAS threads", "warning", value, detail + why,
                 f"export OMP_NUM_THREADS={per} before starting Python (thread counts are never "
                 "changed for you)")



def _core_fix() -> str:
    """How to (re)build the compiled core, for this install mode."""
    from ._paths import in_checkout
    if in_checkout():
        return "CMake build of core/ into $CN_BUILD/core, see Install, development tree"
    return ("reinstall the conatus wheel, or pip install . from the source tree in a conda env "
            "with liblapack and libblas (README, Install)")


def _cuda_fix() -> str:
    """How to get the CUDA part of the core, for this install mode."""
    from ._paths import in_checkout
    if in_checkout():
        return "rebuild with CUDA: -DCONATUS_CUDA=ON in the CMake build of core/ (nvcc in the env)"
    return ("reinstall with CUDA: pip install . -Ccmake.define.CONATUS_CUDA=ON (nvcc, cuFFT, cuBLAS "
            "in the env; documentation, Install page, GPU section)")


def _cuda_archs(core_file: str | None) -> list[int]:
    """sm_XX of the dev build, read from the ``--generate-code`` flags in its build.ninja
    (``$CN_BUILD/core/build.ninja``); empty when not discoverable (installed wheel)."""
    if not core_file:
        return []
    ninja = Path(core_file).resolve().parents[1] / "build.ninja"
    if not ninja.is_file():
        return []
    found: set[int] = set()
    with open(ninja, errors="replace") as fh:
        for line in fh:
            if "generate-code" in line or "-arch=sm_" in line:
                found.update(int(a) for a in re.findall(r"code=\[?(?:compute_\d+,)?sm_(\d+)", line))
                found.update(int(a) for a in re.findall(r"-arch=sm_(\d+)", line))
    return sorted(found)


def _gpu(ctx: dict) -> list[Check]:
    """GPU usable by backend='cuda' (conatus.gpu), and the device's arch vs the built archs."""
    from .gpu import _core_status, _cupy_status
    cupy_ok, cupy_why = _cupy_status()
    ctx["cupy_ok"] = cupy_ok
    core_ok, core_why = _core_status()
    if cupy_ok and core_ok:
        detail, status = f"{core_why}; {cupy_why}", "ok"
        try:
            import cupy
            cc = int(cupy.cuda.Device().compute_capability)
            archs = ctx.get("archs") or []
            detail += f"; device sm_{cc}"
            if archs and cc not in archs:
                status = "warning"
                detail += f" not among the built archs {archs} (PTX JIT or failure)"
        except Exception as err:  # noqa: BLE001 -- the architecture check is advisory; said
            detail += f"; device architecture unreadable ({type(err).__name__})"
        return [Check("GPU (backend='cuda')", status, "available", detail)]
    why = "; ".join(w for ok, w in ((cupy_ok, cupy_why), (core_ok, core_why)) if not ok)
    return [Check("GPU (backend='cuda')", "optional", "not available", why,
                  "run in a job on a GPU node of your cluster with CuPy and a CUDA-built core "
                  "(docs/site/install.md, GPU section); backend='cpu' works here")]


def _cupy() -> list[Check]:
    if importlib.util.find_spec("cupy") is None:
        return [Check("CuPy", "optional", "-", "GPU arrays", _PIP_GPU)]
    ver = _dist_version("cupy") or next((_dist_version(d) for d in
                                        ("cupy-cuda12x", "cupy-cuda13x", "cupy-cuda11x")
                                        if _dist_version(d)), None)
    return [Check("CuPy", "ok", ver or "installed", "GPU arrays")]


def _mpi() -> list[Check]:
    """mpi4py, the MPI library (queried in a subprocess: no MPI_Init here), CUDA-awareness."""
    if importlib.util.find_spec("mpi4py") is None:
        return [Check("MPI (mpi4py)", "optional", "-", "distributed runs",
                      "conda install mpi4py openmpi (in your env)")]
    ver = _dist_version("mpi4py") or "?"
    code = ("import mpi4py; mpi4py.rc.initialize = False; mpi4py.rc.finalize = False\n"
            "from mpi4py import MPI; print(MPI.Get_library_version().splitlines()[0].strip())")
    rc, out = _run([sys.executable, "-c", code])
    lib = next((ln for ln in out.splitlines() if "MPI" in ln and "UCX" not in ln), "").strip()
    checks = [Check("MPI (mpi4py)", "ok" if rc == 0 else "warning", f"mpi4py {ver}",
                    (lib[:90] if rc == 0 else f"library query failed: {out.strip()[:120]}"),
                    "" if rc == 0 else "check that mpi4py matches the MPI library of the env")]
    # CUDA-aware MPI: Open MPI's compile-time flag via ompi_info (mpi4py 4.1 has no
    # Query_cuda_support binding; nothing is guessed: 'unknown' when no source answers)
    ompi = shutil.which("ompi_info")
    value, detail = "unknown", "no ompi_info found (not Open MPI, or not on PATH)"
    if ompi:
        rc, out = _run([ompi, "--parsable", "--all"])
        m = re.search(r"mpi_built_with_cuda_support:value:(\w+)", out)
        if m:
            value = "yes" if m.group(1) == "true" else "no"
            detail = ("ompi_info: mpi_built_with_cuda_support (compile time; the runtime path "
                      "also needs UCX/driver support on the node)")
        else:
            detail = f"ompi_info gave no mpi_built_with_cuda_support ({out.strip()[:80]})"
    checks.append(Check("CUDA-aware MPI", "ok" if value == "yes" else "warning", value, detail,
                        "" if value == "yes" else
                        "use transport='host' (declared staging) or 'nccl' for GPU runs"))
    return checks


def _nccl(ctx: dict) -> list[Check]:
    if importlib.util.find_spec("cupy") is None:
        return [Check("NCCL", "optional", "-", "needs CuPy (cupy.cuda.nccl)", _PIP_GPU)]
    from cupy.cuda import nccl
    if not getattr(nccl, "available", False):
        return [Check("NCCL", "warning", "-", "cupy.cuda.nccl present but libnccl not loadable",
                      "conda install nccl (in your env); or transport='cuda-aware'/'host'")]
    v = int(nccl.get_version())
    return [Check("NCCL", "ok", f"{v // 10000}.{(v // 100) % 100}.{v % 100}",
                  "cupy.cuda.nccl (transport='nccl')")]


def _disk(name: str, root: Path | None, why: str, fix: str) -> list[Check]:
    if root is None:
        return [Check(name, "warning", "-", why, fix)]
    where, free, src = _free_gib(root)
    exists = root.exists()
    writable = os.access(where, os.W_OK)
    status = "ok" if free >= RUNS_FREE_WARN_GB and writable else "warning"
    detail = (f"{root}{'' if exists else f' (not created yet; checked {where})'}"
              f"{'' if writable else ', NOT writable'}; {src}; warn under {RUNS_FREE_WARN_GB:g} GiB")
    return [Check(name, status, f"{free:.1f} GiB free", detail,
                  "" if status == "ok" else fix)]


def _runs() -> list[Check]:
    from .runs.store import default_root
    try:
        root = default_root()
    except Exception as err:      # noqa: BLE001 -- OptionError: no root / root under the home
        # no root declared (an installed package, CN_RUNS and CN_TMP unset): the registry is
        # optional (cn.runs only); a default root under the home is a warning
        undeclared = not (os.environ.get("CN_RUNS") or os.environ.get("CN_TMP")) and \
            "under the home" not in str(err)
        return [Check("runs root ($CN_RUNS)", "optional" if undeclared else "warning", "-",
                      str(err).splitlines()[0],
                      getattr(err, "fix", "") or "set CN_RUNS=/path/on/scratch")]
    src = ("$CN_RUNS" if os.environ.get("CN_RUNS") else "$CN_TMP/runs" if os.environ.get("CN_TMP")
           else "default (scratch/runs of the checkout)")
    out = _disk("runs root ($CN_RUNS)", root, "", "free space on the runs root, or set CN_RUNS "
                "to another scratch path")
    out[0].detail = f"{src}: {out[0].detail}"
    return out


def _cn_tmp() -> list[Check]:
    from ._paths import in_checkout
    v = os.environ.get("CN_TMP")
    if not v:
        if not in_checkout():           # an installed package does not need it (user dirs)
            return [Check("CN_TMP", "ok", "unset", "not needed by an installed package (optional "
                          "scratch root: telemetry and runs go under it when set)")]
        return [Check("CN_TMP", "warning", "unset", "scratch root for builds, outputs, runs",
                      "export CN_TMP=/scratch/path")]
    return _disk("CN_TMP", Path(v), "", "free space under $CN_TMP")


# ---------------------------------------------------------------- entry points
def doctor(file: TextIO | None = None) -> DoctorReport:
    """Run every check and return the report; with ``file`` (e.g. ``sys.stdout``) also print
    ``explain()`` to it. Never raises for a missing component."""
    import platform
    t0 = time.perf_counter()
    ctx: dict[str, Any] = {}
    checks: list[Check] = []
    checks += _safe("install", _install)
    checks += _safe("Python", _python, True)
    checks += _safe("NumPy", lambda: _module("NumPy", "numpy", True, "arrays",
                                             "pip install 'numpy>=1.26'"), True)
    checks += _safe("SciPy", lambda: _module("SciPy", "scipy", True, "sparse LU, FFT oracle",
                                             "pip install 'scipy>=1.11'"), True)
    checks += _safe("compiled core (CPU)", lambda: _core(ctx), True)
    checks += _safe("BLAS threads", _blas)
    checks += _safe("GPU (backend='cuda')", lambda: _gpu(ctx))
    checks += _safe("CuPy", _cupy)
    checks += _safe("MPI (mpi4py)", _mpi)
    checks += _safe("NCCL", lambda: _nccl(ctx))
    checks += _safe("Dedalus", lambda: _module(
        "Dedalus", "dedalus", False, "cross-check oracle for tests, never imported by conatus",
        "conda install -c conda-forge dedalus (only to run the cross-check tests)"))
    checks += _safe("h5py", lambda: _module(
        "h5py", "h5py", True, "outputs and checkpoints: sim.output, sim.checkpoints, cn.io",
        H5PY_FIX), True)
    checks += _safe("matplotlib (tutorials)", lambda: _module(
        "matplotlib (tutorials)", "matplotlib", False, "figures of the tutorials", TUTORIALS_FIX,
        note=": the tutorials stop at their first plot without it"))     # an optional extra: never a warning
    checks += _safe("ipywidgets", lambda: _module("ipywidgets", "ipywidgets", False,
                                                  "notebook widgets (conatus.notebook)",
                                                  NOTEBOOK_FIX))
    checks += _safe("runs root ($CN_RUNS)", _runs)
    checks += _safe("CN_TMP", _cn_tmp)
    from . import __version__
    rep = DoctorReport(checks, platform.node(), time.perf_counter() - t0,
                       {"conatus": __version__})
    if file is not None:
        print(rep.explain(), file=file)
    return rep


def main(argv: list[str] | None = None) -> int:
    """``python -m conatus doctor [--json]``: exit 0 if nothing required is missing, else 1."""
    import argparse
    p = argparse.ArgumentParser(prog="conatus doctor",
                                description="Check what this machine can run (read-only).")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    a = p.parse_args(argv)
    rep = doctor()
    print(json.dumps(rep.to_dict(), indent=2) if a.json else rep.explain())
    return 0 if rep.ok else 1
