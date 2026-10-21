"""F525 (external review of 0.1.1, F13): agent check refuses run(steps=-1) like the real run.

The dry run stopped at Simulation.run and returned ok=True for a script whose run raises
OptionError. Expected: the same OptionError among the refusals, ok False.
"""
import tempfile
from pathlib import Path

from _acceptance import check, passed, start

start(525, "agent check validates the argument values of run")
from conatus.agent import check as agent_check  # noqa: E402
from conatus.errors import OptionError  # noqa: E402

p = Path(tempfile.mkdtemp()) / "bad.py"
p.write_text("import conatus as cn\nm=cn.Model(cn.Box(x=cn.periodic(6.)))\n"
             "m.scalar('u'); m.equations('dt(u)-lap(u)=0')\n"
             "s=m.simulate(16,scheme='RK443',dt=.01)\ns.run(steps=-1)\n")
try:
    exec(compile(p.read_text(), str(p), "exec"), {})
    real = None
except OptionError as err:
    real = type(err).__name__
r = agent_check(p)
print(f"  real run: {real}; check: ok={r['ok']}, refusals={[x['type'] for x in r['refusals']]}")
check(real == "OptionError", "the oracle (the script run for real) did not refuse")
check(not r["ok"] and [x["type"] for x in r["refusals"]] == ["OptionError"],
      "check did not report the refusal of run(steps=-1)")
passed("OptionError reported, ok False")
