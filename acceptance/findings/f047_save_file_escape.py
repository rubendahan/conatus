"""F047: run.save_file(name=...) must not write outside the run directory.

name="../../../outside.txt" escaped the run's files/ directory in 0.1.0. Expected: a typed
refusal (at the call or reported by the run), and no file outside the run directory.
"""
import warnings
from pathlib import Path

from _acceptance import check, passed, refused, start

import conatus as cn

start(47, "save_file names stay inside the run directory")
base = Path("work").resolve()
(base / "a" / "b" / "c").mkdir(parents=True)
root = base / "a" / "b" / "c" / "registry"
Path("payload.txt").write_text("payload\n")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    with cn.runs.track("escape", root=root) as run:
        was_refused, _ = refused(run.save_file, "payload.txt", name="../../../outside.txt")
        run_dir = Path(run.path).resolve()
escaped = [p for p in base.rglob("outside.txt") if run_dir not in p.resolve().parents]
check(not escaped, f"file written outside the run directory: {[str(p) for p in escaped]}")
reported = was_refused or any("outside" in str(w.message) for w in caught)
check(reported, "nothing escaped but the bad name was neither refused nor reported")
passed("refused" if was_refused else "reported by a warning")
