"""Execute the tutorial scripts of the documentation site and record their verified output.

Each tutorial is a plain script in ``docs/site/tutorials/code/tNN_*.py`` that ends with its own
checks against a published or closed-form value (``assert``). This runner executes them one by one
in the *conatus* environment (never the docs environment), in a scratch directory, and writes

  <out>/<name>.txt     header (run_id, commit, host, elapsed, exit code, sha256 of the script that
                       ran) + stdout/stderr; the website shows an output only next to the script
                       whose hash it records (web/scripts/gen-data.mjs, external review 0.1.0, 72)
  <out>/<name>/        files the script wrote (figures), via CN_TUTORIAL_OUT
  <out>/TUTORIALS.json one entry per tutorial: PASS | FAIL | SKIP (+ reason)  (not STATUS.json:

With ``--sync`` the text outputs and figures are copied to ``docs/site/tutorials/outputs/`` (the
pages include them verbatim: what the reader sees is what ran). Local paths in the captured text
(the checkout, the Python environment, the home directory, the user name) are replaced by neutral
placeholders such as ``<conatus>/python/...`` (``_scrub``); nothing else is changed. A tutorial that
writes to stderr is flagged (``"stderr": true``) so its warnings get fixed at their source.
A script whose docstring contains
``requires: gpu`` is skipped (declared, never silently) unless ``--gpu`` is given.

  python docs/site/run_tutorials.py [--only t01,t05] [--gpu] [--out DIR] [--sync]
"""
from __future__ import annotations

import argparse
import datetime
import getpass
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

SITE = Path(__file__).resolve().parent
ROOT = SITE.parents[1]
CODE = SITE / "tutorials" / "code"
SYNC = SITE / "tutorials" / "outputs"


def _commit() -> str:
    try:
        c = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet"]).returncode != 0
        return c + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def _scrub(text: str) -> str:
    """Replace machine-specific paths and the user name by placeholders (published outputs)."""
    subs = [(str(ROOT / "python"), "<conatus>/python"), (str(ROOT), "<conatus>"),
            (os.path.realpath(sys.prefix), "<env>"), (sys.prefix, "<env>"),
            (os.path.expanduser("~"), "~")]
    for real, alias in sorted(subs, key=lambda p: -len(p[0])):   # longest first
        if real and real not in ("/", "~"):
            text = text.replace(real, alias)
    # any other absolute path under a cluster or home mount: keep the last two parts only
    text = re.sub(r"(?<![\w<>~])/(?:home|Users|scratch|tmp)(?:/[^\s:'\"]+)+",
                  lambda m: "<path>/" + "/".join(m.group(0).split("/")[-2:]), text)
    try:
        user = getpass.getuser()
    except (KeyError, OSError):
        user = ""
    if len(user) >= 3:
        text = re.sub(rf"\b{re.escape(user)}\b", "<user>", text)
    return text


def _needs_gpu(script: Path) -> bool:
    head = script.read_text().split('"""')
    return len(head) > 1 and "requires: gpu" in head[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default="", help="comma-separated prefixes (t01,t05b)")
    ap.add_argument("--gpu", action="store_true", help="also run scripts marked 'requires: gpu'")
    ap.add_argument("--out", default=os.environ.get("CN_OUT", str(ROOT / "scratch/output/tutorials_local")))
    ap.add_argument("--sync", action="store_true", help="copy outputs into docs/site/tutorials/outputs")
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds per script")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    only = [p for p in args.only.split(",") if p]
    scripts = sorted(p for p in CODE.glob("t[0-9][0-9]*.py")
                     if not only or any(p.stem.startswith(o) for o in only))
    run_id, commit, host = os.environ.get("CN_RUN_ID", "local"), _commit(), socket.gethostname()
    status_path = out / "TUTORIALS.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    for s in scripts:
        name = s.stem
        if _needs_gpu(s) and not args.gpu:
            status[name] = {"status": "SKIP", "reason": "requires: gpu (run with --gpu on a GPU node)"}
            print(f"SKIP {name}: requires gpu")
            continue
        figdir = out / name
        figdir.mkdir(exist_ok=True)
        sha = hashlib.sha256(s.read_bytes()).hexdigest()[:16]      # the script as it is executed
        env = dict(os.environ, MPLBACKEND="Agg", CN_TUTORIAL_OUT=str(figdir), PYTHONUNBUFFERED="1")
        t0 = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, str(s)], cwd=figdir, env=env, capture_output=True,
                               text=True, timeout=args.timeout)
            err = p.stderr.strip()
            code, text = p.returncode, p.stdout + (("\n[stderr]\n" + p.stderr) if err else "")
        except subprocess.TimeoutExpired as e:
            partial = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            code, text, err = -9, f"TIMEOUT after {args.timeout} s\n{partial}", ""
        dt = time.perf_counter() - t0
        verdict = "PASS" if code == 0 else "FAIL"
        header = (f"# {name}: {verdict} (exit {code}) in {dt:.1f} s | run_id {run_id} | commit {commit}"
                  f" | {host} | {datetime.date.today().isoformat()} | script sha256 {sha}\n")
        (out / f"{name}.txt").write_text(_scrub(header + text))
        status[name] = {"status": verdict, "exit": code, "elapsed_s": round(dt, 1), "run_id": run_id,
                        "commit": commit, "host": host, "stderr": bool(err), "script_sha256": sha}
        print(f"{verdict} {name} ({dt:.1f} s){'  [wrote to stderr]' if err else ''}", flush=True)
        if args.sync and verdict == "PASS":
            SYNC.mkdir(exist_ok=True)
            shutil.copy(out / f"{name}.txt", SYNC / f"{name}.txt")
            for f in figdir.glob("*.png"):
                shutil.copy(f, SYNC / f"{name}_{f.name}")
    status_path.write_text(json.dumps(status, indent=2))
    if args.sync:
        merged = SYNC / "STATUS.json"
        old = json.loads(merged.read_text()) if merged.exists() else {}
        old.update({k: v for k, v in status.items() if v["status"] != "SKIP" or k not in old})
        merged.write_text(json.dumps(old, indent=2, sort_keys=True) + "\n")
    bad = [k for k, v in status.items() if v["status"] == "FAIL" and (not only or any(k.startswith(o) for o in only))]
    print("FAILED: " + ", ".join(bad) if bad else "all selected tutorials passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
