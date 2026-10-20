"""F200 (security audit 2026-10-20): reproduce() must not run a function named by a run record
that another account can write.

``run.reproduce()`` of a decorated call imports ``module:function`` from ``run.json`` and calls it
with the recorded arguments. A record made writable by others (a shared registry) and rewritten
to name another function ran that function. Expected: a typed refusal naming ``fn=``, and the
named function never runs.
"""
import json
import os
import sys
from pathlib import Path

from _acceptance import check, failed, passed, refused, start

import conatus.runs as R

start(200, "reproduce() refuses a record writable by others")
mods = Path("mods").resolve()
mods.mkdir()
(mods / "f200_good.py").write_text("def area(w, h):\n    return w * h\n")
(mods / "f200_evil.py").write_text("def payload(w, h):\n    open(w, 'w').write('ran')\n    return 0\n")
sys.path.insert(0, str(mods))
import f200_good  # noqa: E402

root = Path("registry").resolve()
f = R.track("area", root=root)(f200_good.area)
check(f(2.0, 3.0) == 6.0, "the tracked call returned a wrong value")
run = R.find(name="area", root=root)[0]
marker = Path("marker.txt").resolve()
rec_path = run.path / "run.json"
rec = json.loads(rec_path.read_text())
rec["inputs"]["target"] = "f200_evil:payload"
rec["inputs"]["args"] = {"w": str(marker), "h": 1}
rec_path.write_text(json.dumps(rec))
os.chmod(rec_path, 0o666)
was_refused, out = refused(R.get(run.id, root=root).reproduce, record=False)
if marker.exists():
    failed("the function named by a group/other-writable record ran")
check(was_refused, f"reproduce() of a group/other-writable record returned {out!r}")
check("fn=" in str(out), "the refusal does not name fn=")
passed("refused before importing the named module")
