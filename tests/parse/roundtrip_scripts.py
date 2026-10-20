"""Round trip IR -> string -> IR on every equation of the given scripts (tutorials, examples).

    python tests/parse/roundtrip_scripts.py <out.json> [--timeout S] script.py ...

Each script runs in its own process with a hook on ``Problem`` (conatus.ir.problem): every
equation accepted by ``add_equation`` and, at exit, the final equations of every Problem built
(including the tau lifts that ``add_bc`` writes into the bulk equations) are converted with
``conatus.ir.parse.to_string`` and parsed back in the namespace of the problem's variables, the
equation's parameters and its lift bases; the result must be structurally equal. The script
itself runs normally (its checks are its own). A script that fails or times out fails the round
trip (exit 1), even when every equation it declared before stopping round-trips: its later
equations were never checked (review finding 100: such a run counted as passed).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HOOK = r'''
import atexit, json, os, runpy, sys, traceback
sys.argv = [SCRIPT]
from conatus.ir import problem as _P
from conatus.ir.parse import Namespace, to_string, parse_equation
from conatus.ir.expr import Lift
_out = open(OUT, "a")
_seen = set()
_problems = []

def _check(pb, eq, where):
    key = (id(pb), str(eq))
    if key in _seen:
        return
    _seen.add(key)
    rec = {"script": SCRIPT, "problem": pb.name, "kind": pb.kind, "where": where}
    try:
        s = to_string(eq)
        rec["string"] = s
        vals = {v.name: v for v in pb.variables}
        for side in (eq.lhs, eq.rhs):
            for f in list(side.free_fields()) + list(side.free_parameters()):
                vals.setdefault(f.name, f)
        ns = Namespace(vals)
        for side in (eq.lhs, eq.rhs):
            for n in side.walk():
                if isinstance(n, Lift):
                    ns._add_basis(n.basis)
        back = parse_equation(s, ns)
        rec["ok"] = bool(back.lhs == eq.lhs and back.rhs == eq.rhs)
        if not rec["ok"]:
            rec["back"] = str(back)
    except Exception as err:
        rec["ok"] = False
        rec["error"] = f"{type(err).__name__}: {str(err).splitlines()[0]}"
    _out.write(json.dumps(rec) + "\n"); _out.flush()

_orig_add = _P.Problem.add_equation
def _add(self, eq, *a, **k):
    out = _orig_add(self, eq, *a, **k)
    _check(self, out, "add_equation")
    return out
_P.Problem.add_equation = _add
_orig_init = _P.Problem.__init__
def _init(self, *a, **k):
    _orig_init(self, *a, **k)
    _problems.append(self)
_P.Problem.__init__ = _init

@atexit.register
def _final():
    for pb in _problems:
        for eq in pb.equations:
            _check(pb, eq, "final")
    _out.close()

os.chdir(os.path.dirname(os.path.abspath(SCRIPT)))
runpy.run_path(SCRIPT, run_name="__main__")
'''


def main(argv: list[str]) -> int:
    out = argv[0]
    timeout = 900.0
    rest = argv[1:]
    if rest[:1] == ["--timeout"]:
        timeout, rest = float(rest[1]), rest[2:]
    lines = out + ".lines"
    if os.path.exists(lines):
        os.remove(lines)
    runs = []
    for script in rest:
        script = os.path.abspath(script)
        code = f"SCRIPT = {script!r}; OUT = {os.path.abspath(lines)!r}\n" + HOOK
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               timeout=timeout)
            status = "completed" if p.returncode == 0 else f"exit {p.returncode}"
            tail = (p.stderr or "").strip().splitlines()[-3:]
        except subprocess.TimeoutExpired:
            status, tail = f"timeout after {timeout:.0f} s (only the equations declared so far checked)", []
        runs.append({"script": script, "status": status, "seconds": round(time.time() - t0, 1),
                     "stderr_tail": tail})
        print(f"{os.path.basename(script)}: {status}", flush=True)
    recs = [json.loads(line) for line in open(lines)] if os.path.exists(lines) else []
    incomplete = [r for r in runs if r["status"] != "completed"]
    summary = {"equations": len(recs), "ok": sum(r["ok"] for r in recs),
               "failed": [r for r in recs if not r["ok"]], "runs": runs,
               "scripts_incomplete": incomplete, "per_script": {}}
    for r in recs:
        d = summary["per_script"].setdefault(os.path.basename(r["script"]), {"n": 0, "ok": 0})
        d["n"] += 1
        d["ok"] += int(r["ok"])
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"round trip: {summary['ok']}/{summary['equations']} equations identical")
    for r in summary["failed"]:
        print("FAILED", os.path.basename(r["script"]), r.get("string"), r.get("error"),
              r.get("back"))
    for r in incomplete:
        print("SCRIPT NOT COMPLETED", os.path.basename(r["script"]), r["status"], *r["stderr_tail"])
    return 0 if summary["ok"] == summary["equations"] and recs and not incomplete else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
