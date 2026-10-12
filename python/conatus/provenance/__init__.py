"""Run provenance: one JSON record per run (spec §85).

``record(run, ...)`` collects what is needed to reproduce and audit a run:

- structural hash of the problem (IR) and of each equation, problem kind and name;
- the complete numerical configuration read from the objects themselves (bases, sizes, bounds,
  dealias, groups, symmetry, NCC cutoff, parameters; scheme, dt, startup, backend, ordering,
  reduced space; eigenvalue name; tolerances passed by the caller);
- versions (python, numpy, scipy, conatus, compiled core and CUDA availability), host, GPU;
- the **executed code**, verified at record time: git HEAD, dirty paths under ``python/`` and
  ``core/``, sha256 of the ``conatus`` source tree actually imported, whether that tree is the
  one of the repository, and whether the compiled module is older than the C++ sources
  (the a logged commit that is not the code that ran).

Nothing is guessed: a field that cannot be read is recorded as ``None`` with a reason.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

import conatus
from conatus.errors import InputTypeError

PKG_DIR = Path(conatus.__file__).resolve().parent
REPO_ROOT = PKG_DIR.parents[1]

__all__ = ["record", "code_state", "numerics", "environment", "problem_hash", "parameter_values",
           "write"]


# ---------------------------------------------------------------------------------- code state
def _in_git_checkout() -> bool:
    """True when this package is the ``python/conatus`` of a git checkout of Conatus itself."""
    return (REPO_ROOT / ".git").exists() and PKG_DIR == REPO_ROOT / "python" / "conatus"


def _git(*args: str) -> str | None:
    """Output of a git command in the Conatus checkout; None for an installed package.

    ``git -C`` searches upwards for a repository: from ``<venv>/lib/python3.x`` it would find the
    user's own project when the environment lives inside it (``.venv/``) and record that commit
    as the code that ran. Only a checkout of Conatus itself is asked.
    """
    if not _in_git_checkout():
        return None
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                             text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _tree_sha256(root: Path, suffixes: tuple[str, ...]) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(root)).encode())
            h.update(b"\0")
            h.update(p.read_bytes())
            n += 1
    return h.hexdigest(), n


#: Sources of the compiled core whose content decides whether a build is stale.
CORE_SUFFIXES = (".hpp", ".cpp", ".cu", ".cuh", ".txt")
#: Written next to the compiled module by ``core/build.sh``: the hash of the core sources it was
#: built from.
CORE_STAMP = "_core.sources.json"


def core_sources_sha256(repo: Path | None = None) -> tuple[str, int]:
    """sha256 of the C++/CUDA sources of the core (``core/`` with :data:`CORE_SUFFIXES`, and the
    top-level ``CMakeLists.txt``): paths and contents, never modification times."""
    repo = REPO_ROOT if repo is None else Path(repo)
    h = hashlib.sha256()
    tree, n = _tree_sha256(repo / "core", CORE_SUFFIXES)
    h.update(tree.encode())
    top = repo / "CMakeLists.txt"
    if top.is_file():
        h.update(b"CMakeLists.txt\0" + top.read_bytes())
        n += 1
    return h.hexdigest(), n


def record_core_sources(module_dir: str | Path, repo: str | Path | None = None) -> Path:
    """Write :data:`CORE_STAMP` in ``module_dir`` (the directory of the built ``_core``) with the
    hash of the sources just compiled; called by ``core/build.sh`` after a successful build."""
    digest, n = core_sources_sha256(None if repo is None else Path(repo))
    out = Path(module_dir) / CORE_STAMP
    out.write_text(json.dumps({"core_sources_sha256": digest, "files": n,
                               "repo": str(Path(repo).resolve() if repo else REPO_ROOT),
                               "written": _dt.datetime.now(_dt.UTC).isoformat(
                                   timespec="seconds")}, indent=1) + "\n")
    return out


def _core_state() -> dict:
    """Compiled module location and staleness against core/ sources (never rebuilds)."""
    info: dict[str, Any] = {"module": None, "source": None, "cuda_compiled": None,
                            "cuda_device": None, "stale_vs_sources": None}
    try:
        from conatus import runtime
        core = runtime._core if hasattr(runtime, "_core") else None
        if core is None:
            import importlib
            core = importlib.import_module("conatus._core")
    except Exception as exc:  # noqa: BLE001 -- optional core (ImportError, OSError, ABI); reason recorded
        info["reason"] = f"compiled core not importable: {type(exc).__name__}: {exc}"
        return info
    path = Path(getattr(core, "__file__", "") or "")
    info["module"] = str(path) if path else None
    # Which copy the loader picked: CONATUS_CORE_PATH, installed (pip/wheel/conda) or dev-build.
    info["source"] = getattr(runtime, "CORE_SOURCE", None)
    cuda = getattr(core, "cuda", None)
    info["cuda_compiled"] = cuda is not None
    if cuda is not None:
        try:
            info["cuda_device"] = bool(cuda.available())
        except Exception as exc:  # noqa: BLE001 -- recorded in info['reason']
            info["cuda_device"] = None
            info["reason"] = f"cuda.available() failed: {exc}"
    src = REPO_ROOT / "core"
    if path.is_file() and src.is_dir():
        # Content, not times: worktrees share one build, and
        # a fresh checkout has newer source mtimes than the module although the sources are the
        # ones compiled, so the mtime test cried "rebuild" in every new worktree.
        stamp = path.parent / CORE_STAMP
        try:
            built = json.loads(stamp.read_text()).get("core_sources_sha256")
            # a module rebuilt after the stamp (by a build script without the stamp step, e.g.
            # an older branch) is not described by it: back to the mtime test
            if stamp.stat().st_mtime < path.stat().st_mtime:
                built = None
        except (OSError, ValueError, AttributeError):
            built = None
        if built:
            now, _ = core_sources_sha256()
            info["core_sources_sha256"] = {"built": built, "checkout": now}
            info["stale_vs_sources"] = built != now
            info["stale_test"] = "content hash of core/ sources recorded at build time"
        else:
            newest = max((p.stat().st_mtime for p in src.rglob("*")
                          if p.is_file() and p.suffix in CORE_SUFFIXES), default=0.0)
            info["stale_vs_sources"] = bool(newest > path.stat().st_mtime)
            info["stale_test"] = (f"modification times (no valid {CORE_STAMP} next to the "
                                  "module: a build older than the hash stamp, or a module rebuilt "
                                  "after its stamp; core/build.sh records one)")
    return info


TREE_MARKER = ".conatus_tree.json"


def _exported_tree() -> dict | None:
    """Marker written by ``ci/export_tree.sh`` in a tree exported without ``.git``."""
    if (REPO_ROOT / ".git").exists():
        return None
    try:
        return json.loads((REPO_ROOT / TREE_MARKER).read_text())
    except (OSError, ValueError):
        return None


def code_state() -> dict:
    """What code is executing now, checked at call time.

    In an exported tree (no ``.git``, CI jobs of) the base commit and the overlaid paths are
    read from the export marker: ``git_head`` is the base commit, the overlaid paths under
    ``python/`` or ``core/`` are reported as dirty, and ``code_matches_commit`` stays False.
    Without a marker ``git_head`` is None and a ``warning`` says why (never a guessed commit).
    """
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--", "python", "core")
    dirty = [ln[3:] for ln in (status or "").splitlines() if ln.strip()]
    tree, nfiles = _tree_sha256(PKG_DIR, (".py",))
    in_repo = (REPO_ROOT / ".git").exists() and PKG_DIR == (REPO_ROOT / "python" / "conatus")
    exported = _exported_tree() if head is None else None
    if exported is not None:
        head = exported.get("base_head")
        dirty = [q for q in exported.get("overlay", []) if q.split("/")[0] in ("python", "core")]
    state = {
        "git_head": head,
        "git_dirty_paths": dirty,
        "code_matches_commit": (head is not None and not dirty and in_repo),
        "package_dir": str(PKG_DIR),
        "package_in_repo": in_repo,
        "python_tree_sha256": tree,
        "python_tree_files": nfiles,
        "core": _core_state(),
    }
    if exported is not None:
        state["exported_tree"] = exported
    if state["core"].get("stale_vs_sources"):
        state["code_matches_commit"] = False
        state["warning"] = (
            "compiled core differs from the core/ sources of this checkout (content hash recorded "
            "at build time): rebuild (core/build.sh) before trusting this run"
            if state["core"].get("core_sources_sha256") else
            "compiled core is older than core/ sources (modification times; no source hash "
            "recorded with this build): rebuild (core/build.sh) before trusting this run")
    elif dirty:
        state["warning"] = f"uncommitted changes in {len(dirty)} path(s) under python/ or core/"
    elif head is None and PKG_DIR.parent.name != "python":
        state["installed_version"] = conatus.__version__
        state["warning"] = (f"installed package (conatus {conatus.__version__}, no git checkout): "
                            "the version and python_tree_sha256 identify the code, not a commit")
    elif head is None:
        state["warning"] = (f"no git checkout and no {TREE_MARKER} marker at {REPO_ROOT}: "
                            "the executed commit is unknown")
    elif exported is not None:
        state["warning"] = (f"exported tree (no .git) based on {str(head)[:10]}: code matches that "
                            "commit plus the recorded overlay only if the tree was not edited")
    return state


def environment() -> dict:
    """Versions, host, Slurm job and GPU of the running process (nvidia-smi only with a GPU)."""
    import scipy
    env = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "conatus": conatus.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "run_id": os.environ.get("CN_RUN_ID"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": None,
    }
    if env["cuda_visible_devices"] not in (None, ""):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                                  "--format=csv,noheader"], capture_output=True, text=True,
                                 timeout=20)
            env["gpu"] = out.stdout.strip() or None
        except (OSError, subprocess.TimeoutExpired):
            env["gpu"] = None
    return env


# ---------------------------------------------------------------------------------- numerics
def _basis(b) -> dict:
    d = {"type": type(b).__name__, "coord": b.coord.name, "size": int(b.size),
         "bounds": list(b.bounds), "dealias": float(b.dealias), "grid_size": int(b.grid_size),
         "grid": getattr(b, "grid", "floor"), "repr": repr(b)}
    for attr in ("a", "b", "lam", "center", "scale", "start"):
        if hasattr(b, attr) and not callable(getattr(b, attr)):
            try:
                d[attr] = float(getattr(b, attr))
            except (TypeError, ValueError):
                pass
    return d


def _method(plan, bases) -> str:
    """One line naming the spatial method, read from the bases (every plan with a
    non-periodic axis was recorded as "Chebyshev T + ultraspherical", also Legendre, Hermite or
    Laguerre; a plan without ``layout`` (polar, spherical) raised). The Chebyshev T line is
    unchanged, so that the cache keys and ``reproduce()`` of the runs recorded before still match."""
    from ..ir.bases import Chebyshev, _Fourier
    zb = getattr(getattr(plan, "layout", None), "zbasis", None)
    if zb is None and not (getattr(plan, "is_polar", False) or getattr(plan, "is_sphere", False)):
        others = [b for b in bases if not isinstance(b, _Fourier)]
        if not others:
            return "Fourier (diagonal per mode)"
        zb = others[0]
    if zb is not None and type(zb) is Chebyshev:
        return "Chebyshev T + ultraspherical (Olver–Townsend), generalized tau with lifts"
    curved = [b.family for b in bases if not isinstance(b, _Fourier)]
    return ("; ".join(curved) + ", generalized tau with lifts" if curved
            else "Fourier (diagonal per mode)")


def problem_hash(problem) -> str:
    """Structural hash of a problem: kind, variables, equations (lhs and rhs), symmetry."""
    h = hashlib.sha256()
    h.update(problem.kind.encode())
    for v in problem.variables:
        h.update(f"var:{v.name}:{v.domain!r}".encode())
    for eq in problem.equations:
        h.update(eq.lhs.structural_hash.encode())
        h.update(b"=")
        h.update(eq.rhs.structural_hash.encode())
    if getattr(problem, "symmetry", None) is not None:
        h.update(repr(problem.symmetry).encode())
    ev = getattr(problem, "eigenvalue", None)
    if ev is not None:
        h.update(f"eig:{ev.name}".encode())
    return h.hexdigest()


def parameter_values(problem) -> dict:
    """{name: value} of every Parameter of the equations (None: no value, e.g. an eigenvalue)."""
    out = {}
    for eq in problem.equations:
        for side in (eq.lhs, eq.rhs):
            for p in side.free_parameters():
                v = p.value
                out[p.name] = (None if v is None else complex(v) if np.iscomplexobj(v)
                               else float(v))
    return dict(sorted(out.items()))


def _unwrap(run) -> tuple[Any, Any, Any]:
    """(problem, plan, solver) from a Problem, Plan or solver object."""
    solver = plan = problem = None
    if hasattr(run, "plan") and hasattr(run, "scheme"):
        solver, plan = run, run.plan
    elif hasattr(run, "problem") and hasattr(run, "groups"):
        plan = run          # Cartesian, polar, spherical or mapped plan (not only those with a layout)
    elif hasattr(run, "equations") and hasattr(run, "variables"):
        problem = run
    elif run is not None:
        raise InputTypeError(f"record(): cannot read provenance from {type(run).__name__}",
                             fix="pass a Problem, a Plan or an IVPSolver")
    if plan is not None:
        problem = plan.problem
    return problem, plan, solver


def numerics(run, tolerances: Mapping[str, float] | None = None,
             inner_product: Any = None) -> dict:
    """Complete numerical configuration, read from the objects (never defaulted here)."""
    problem, plan, solver = _unwrap(run)
    out: dict[str, Any] = {}
    if problem is not None:
        out["problem"] = {
            "kind": problem.kind,
            "name": problem.name,
            "structural_hash": problem_hash(problem),
            "variables": [v.name for v in problem.variables],
            "equations": [{"lhs": str(eq.lhs), "rhs": str(eq.rhs),
                           "lhs_hash": eq.lhs.structural_hash}
                          for eq in problem.equations],
            "symmetry": None if problem.symmetry is None else repr(problem.symmetry),
            "eigenvalue": (problem.eigenvalue.name
                           if getattr(problem, "eigenvalue", None) is not None else None),
            # Values of the IR Parameters (Ra, Re, kappa...): they are not part of the structural
            # hash, so without them two runs at different Ra had the same record.
            "parameters": parameter_values(problem),
        }
    if plan is not None:
        if getattr(plan, "_ncc_data", None) and not plan.ncc_reports:
            plan.matrices(plan.groups[0])    # NCC expansions are reported at assembly (cached)
        bases = plan.domain.bases if plan.domain is not None else ()
        out["discretization"] = {
            "bases": [_basis(b) for b in bases],
            "n_groups": len(plan.groups),
            "ncc_cutoff": plan.ncc_cutoff,
            "ncc": {str(k): (v.explain() if hasattr(v, "explain") else str(v))
                    for k, v in plan.ncc_reports.items()},
            "params": {k: (complex(v) if isinstance(v, complex) else v)
                       for k, v in plan.params.items()},
            "method": _method(plan, bases),
        }
    if solver is not None:
        system = getattr(solver, "system", None)
        out["time_stepping"] = {
            "scheme": solver.scheme,
            "dt": solver.dt,
            "startup": getattr(solver, "startup", None),
            "startup_substeps": getattr(solver, "startup_substeps", None),
            "backend": getattr(solver, "backend", None),
            "ordering": getattr(getattr(system, "host", system), "ordering", None),
            "reduced_space": getattr(getattr(system, "host", system), "reduced", None),
            "dtype": str(np.dtype(getattr(system, "dtype", np.float64))),
            "iteration": getattr(solver, "iteration", None),
            "t": getattr(solver, "t", None),
        }
    out["tolerances"] = dict(tolerances) if tolerances else {}
    out["inner_product"] = None if inner_product is None else (
        inner_product.explain() if hasattr(inner_product, "explain") else repr(inner_product))
    return out


# ---------------------------------------------------------------------------------- record
def record(run=None, *, seed: int | None = None, tolerances: Mapping[str, float] | None = None,
           inner_product: Any = None, started: _dt.datetime | None = None,
           extra: Mapping[str, Any] | None = None, out: str | os.PathLike | None = None,
           require_clean: bool = False) -> dict:
    """Build the provenance record of ``run`` (Problem, Plan or IVPSolver, or None for a bare
    environment record). With ``out`` (a directory or a .json path) the record is written.
    ``require_clean=True`` raises if the executed code is not exactly a commit."""
    now = _dt.datetime.now(_dt.UTC)
    code = code_state()
    if require_clean and not code["code_matches_commit"]:
        raise RuntimeError("provenance: executed code does not match a commit "
                           f"({code.get('warning', 'package outside repository')})")
    rec = {
        "schema": "conatus.provenance/1",
        "recorded_utc": now.isoformat(),
        "started_utc": started.astimezone(_dt.UTC).isoformat() if started else None,
        "seed": seed,
        "code": code,
        "environment": environment(),
        "numerics": numerics(run, tolerances, inner_product) if run is not None else {},
        "extra": dict(extra or {}),
    }
    if out is not None:
        write(rec, out)
    return rec


def write(rec: Mapping[str, Any], out: str | os.PathLike) -> Path:
    """Write ``rec`` atomically to ``out`` (a .json path, or a directory: ``provenance.json``)."""
    p = Path(out)
    if p.suffix != ".json":
        p.mkdir(parents=True, exist_ok=True)
        p = p / "provenance.json"
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=2, default=str))
    os.replace(tmp, p)                      # atomic
    return p
