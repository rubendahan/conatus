"""Runner of the public benchmark suite (docs page docs/site/guide/benchmarks.md).

    python examples/benchmarks/suite.py [--cases all|name,name] [--size default|large]
                                         [--threads 1 8] [--out DIR] [--repeat 1]

Each (case, thread count) runs in a fresh Python process, so that the thread settings are read
before NumPy loads its BLAS, and the peak resident memory (``ru_maxrss``) is that case's alone.
The child sets nothing itself: the parent exports, for a thread count T,
OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = T; CN_FFT_WORKERS and
CONATUS_BUILD_WORKERS are left as they are in the caller's environment and recorded.

Output in DIR (default $CN_OUT, else the current directory):
  suite_<case>_t<T>[_r<i>].json   one record per run (kind = "conatus_suite"), with the case's
                                  problem, timings, accuracy, memory, threads, hardware, versions;
  suite_summary.json / .md        all records and a table.
The exit code is 1 if any accuracy check failed or a case raised.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
DECLARED_VARS = THREAD_VARS + ("CN_FFT_WORKERS", "CONATUS_BUILD_WORKERS", "CONATUS_CORE_PATH")


def _rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0      # Linux: KiB


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def hardware() -> dict:
    model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        allowed = len(os.sched_getaffinity(0))
    except AttributeError:
        allowed = os.cpu_count()
    return dict(host=socket.gethostname().split(".")[0], cpu_model=model, cpus_on_node=os.cpu_count(),
                cpus_allowed=allowed, partition=os.environ.get("SLURM_JOB_PARTITION", ""),
                slurm_job=os.environ.get("SLURM_JOB_ID", ""), machine=platform.machine())


def versions() -> dict:
    import numpy as np
    import scipy

    import conatus
    from conatus import runtime
    v = dict(conatus=conatus.__version__, python=platform.python_version(), numpy=np.__version__,
             scipy=scipy.__version__, git_commit=_git("rev-parse", "HEAD"),
             git_dirty=bool(_git("status", "--porcelain", "--", "python", "core", "benchmarks")),
             core_source=getattr(runtime, "CORE_SOURCE", ""), core_file=str(getattr(runtime, "CORE_FILE", "")))
    stamp = ROOT / "TREE_COMMIT"                 # an exported tree (no .git): the commit it was cut from
    if not v["git_commit"] and stamp.is_file():
        v["git_commit"], v["git_dirty"] = stamp.read_text().strip(), False
        v["exported_tree"] = str(ROOT)
    sources = Path(v["core_file"]).parent / "_core.sources.json" if v["core_file"] else None
    if sources is not None and sources.is_file():
        v["core_sources_sha256"] = json.loads(sources.read_text()).get("core_sources_sha256", "")
    try:
        from threadpoolctl import threadpool_info
        v["threadpools"] = [dict(api=p.get("internal_api"), lib=Path(p.get("filepath", "")).name,
                                 version=p.get("version"), threads=p.get("num_threads"))
                            for p in threadpool_info()]
    except ImportError:
        v["threadpools"] = "threadpoolctl not installed"
    return v


def child(case: str, size: str, out: Path) -> int:
    """Run one case in this process and write its record."""
    rss0 = _rss_mib()
    t0 = time.perf_counter()
    import numpy  # noqa: F401  (import cost and BLAS load before the case)

    import conatus  # noqa: F401
    from cases import CASES
    rss_import = _rss_mib()
    rec = dict(kind="conatus_suite", case=case, size=size, run_id=os.environ.get("CN_RUN_ID", ""),
               date=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               threads={k: os.environ.get(k, "") for k in DECLARED_VARS})
    try:
        rec.update(CASES[case](size))
        ok = bool(rec["accuracy"]["passed"])
    except Exception as exc:  # noqa: BLE001 -- the record says what failed; the suite goes on
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()
        ok = False
    rec["wall_s"] = time.perf_counter() - t0
    rec["memory"] = dict(peak_rss_mib=_rss_mib(), rss_after_import_mib=rss_import,
                         rss_at_start_mib=rss0)
    rec["hardware"] = hardware()
    rec["versions"] = versions()
    out.write_text(json.dumps(rec, indent=1, default=float))
    acc = rec.get("accuracy", {})
    print(f"{case:22s} {size:8s} thr={rec['threads']['OMP_NUM_THREADS'] or '-':>3s} "
          f"wall {rec['wall_s']:8.2f} s  per-step {rec.get('timings', {}).get('s_per_step', float('nan'))*1e3:9.3f} ms  "
          f"err {acc.get('error', float('nan')):.2e} (tol {acc.get('tolerance', float('nan')):.0e}) "
          f"{'PASS' if ok else 'FAIL ' + rec.get('error', '')}", flush=True)
    return 0 if ok else 1


def summary(records: list[dict], out: Path) -> None:
    (out / "suite_summary.json").write_text(json.dumps(records, indent=1, default=float))
    rows = ["| case | size | threads | wall s | setup s | run s | ms/step | peak MiB | error | tol | check |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        t, a = r.get("timings", {}), r.get("accuracy", {})
        sps = t.get("s_per_step")
        rows.append(f"| {r['case']} | {r['size']} | {r['threads'].get('OMP_NUM_THREADS', '')} | "
                    f"{r.get('wall_s', 0):.2f} | {t.get('setup_s', float('nan')):.2f} | "
                    f"{t.get('run_s', float('nan')):.3f} | {'' if sps is None else f'{sps * 1e3:.3f}'} | "
                    f"{r['memory']['peak_rss_mib']:.0f} | {a.get('error', float('nan')):.2e} | "
                    f"{a.get('tolerance', float('nan')):.0e} | "
                    f"{'PASS' if a.get('passed') else 'FAIL'} |")
    if records:
        h, v = records[0]["hardware"], records[0]["versions"]
        rows += ["", f"Hardware: {h['cpu_model']} on {h['host']} ({h['cpus_allowed']} CPUs allowed, "
                     f"partition {h['partition'] or '-'}, job {h['slurm_job'] or '-'}).",
                 f"Versions: conatus {v['conatus']} ({v['git_commit'][:10]}{' dirty' if v['git_dirty'] else ''}), "
                 f"Python {v['python']}, NumPy {v['numpy']}, SciPy {v['scipy']}, core "
                 f"{v.get('core_sources_sha256', '?')[:12]} ({v['core_source']})."]
    (out / "suite_summary.md").write_text("\n".join(rows) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", default="all")
    ap.add_argument("--size", default="default")
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", default=os.environ.get("CN_OUT", "."))
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--json", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    sys.path.insert(0, str(HERE))
    if args.child:
        return child(args.child, args.size, Path(args.json))
    from cases import CASES
    names = list(CASES) if args.cases == "all" else args.cases.split(",")
    unknown = [n for n in names if n not in CASES]
    if unknown:
        ap.error(f"unknown case(s) {unknown}; known: {', '.join(CASES)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records, rc = [], 0
    for T in args.threads:
        env = dict(os.environ, **{k: str(T) for k in THREAD_VARS})
        for name in names:
            for r in range(args.repeat):
                tag = f"_r{r}" if args.repeat > 1 else ""
                path = out / f"suite_{name}_{args.size}_t{T}{tag}.json"
                code = subprocess.run([sys.executable, str(Path(__file__)), "--child", name,
                                       "--size", args.size, "--json", str(path)], env=env).returncode
                rc |= int(code != 0)
                if path.is_file():
                    records.append(json.loads(path.read_text()))
    summary(records, out)
    return rc


if __name__ == "__main__":
    # every case checks its oracle in its own process; main() returns 1 when one missed it or raised
    assert main() == 0, "a benchmark case missed its oracle or raised: see its record"
