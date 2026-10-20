#!/usr/bin/env python3
"""Run every acceptance script of this directory and print the PASS / FAIL / SKIP table.

    python run_all.py [--installed] [--timeout S] [--out DIR] [fNNN ...]

Each ``f<nnn>_*.py`` runs in its own interpreter (``sys.executable``), from a fresh scratch
directory, with a wall-clock timeout (default 300 s). It prints ``RESULT F<nnn> PASS|FAIL|SKIP``;
no result line (a crash, a timeout) counts as FAIL. ``--installed`` refuses to run unless the
imported ``conatus`` lives in a site-packages directory (a stranger's install, not a checkout on
the path). ``--out`` writes ``results.tsv`` and the full log of every script there.
Exit status: 0 when nothing FAILs, 1 otherwise, 2 on a usage or environment error.
"""
from __future__ import annotations

import argparse
import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULT = re.compile(r"^RESULT (F\d{3}) (PASS|FAIL|SKIP) ?(.*)$", re.M)


def where() -> str:
    """The conatus that a child interpreter imports (path and version)."""
    probe = ("import conatus, sys; print(conatus.__version__, conatus.__file__, "
             "sys.version.split()[0], sep='|')")
    p = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd="/")
    if p.returncode:
        sys.exit(f"run_all: cannot import conatus:\n{p.stderr[-2000:]}")
    return p.stdout.strip()


def run_one(script: Path, timeout: float, out: Path | None) -> tuple[str, str, float, float]:
    """(status, detail, wall s, child CPU s) of one script."""
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                           timeout=timeout, cwd=HERE)
        text, code = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as exc:
        text = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        text, code = text + f"\nTIMEOUT after {timeout:g} s", None
    wall = time.perf_counter() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    if out is not None:
        (out / f"{script.stem}.log").write_text(text)
    m = RESULT.search(text)
    if code is None:
        return "FAIL", f"timeout after {timeout:g} s", wall, cpu
    if m is None:
        last = [ln for ln in text.strip().splitlines() if ln.strip()][-1:] or ["no output"]
        return "FAIL", f"crash (exit {code}): {last[0][:200]}", wall, cpu
    return m.group(2), m.group(3), wall, cpu


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("only", nargs="*", help="finding ids to run (f001 or 1); default: all")
    ap.add_argument("--installed", action="store_true", help="require an installed conatus")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    version, path, py = where().split("|")
    print(f"conatus {version} from {path} (Python {py})")
    if a.installed and "site-packages" not in path:
        print("run_all: --installed but conatus is not imported from site-packages", file=sys.stderr)
        return 2
    wanted = {f"f{int(re.sub(r'[^0-9]', '', s)):03d}" for s in a.only}
    scripts = sorted(p for p in HERE.glob("f[0-9][0-9][0-9]_*.py")
                     if not wanted or p.name[:4] in wanted)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in scripts:
        status, detail, wall, cpu = run_one(s, a.timeout, a.out)
        rows.append((s.name[:4].upper(), status, wall, cpu, s.stem[5:], detail))
        print(f"{rows[-1][0]}  {status:4s}  {wall:6.1f} s  {cpu:6.1f} s cpu  {s.stem[5:]}: {detail}",
              flush=True)
    counts = {k: sum(r[1] == k for r in rows) for k in ("PASS", "FAIL", "SKIP")}
    print(f"== {len(rows)} findings: {counts['PASS']} PASS, {counts['FAIL']} FAIL, {counts['SKIP']} SKIP")
    if a.out:
        with open(a.out / "results.tsv", "w") as f:
            f.write(f"# conatus {version} {path} python {py} host {os.uname().nodename}\n")
            f.write("finding\tstatus\twall_s\tcpu_s\tname\tdetail\n")
            for r in rows:
                f.write(f"{r[0]}\t{r[1]}\t{r[2]:.1f}\t{r[3]:.1f}\t{r[4]}\t{r[5]}\n")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
