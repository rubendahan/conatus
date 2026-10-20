"""F048: one malformed run.json must not break registry search.

A registry holding one good run and one directory whose run.json is "{}": find() raised KeyError
in 0.1.0. Expected: find() returns the good run (the bad record is skipped and reported).
"""
from pathlib import Path

from _acceptance import check, passed, refused, start

import conatus as cn

start(48, "registry search survives a malformed record")
root = Path("registry").resolve()
with cn.runs.track("good", root=root, params={"Re": 100.0}) as run:
    run.result(x=1.0)
reg = cn.runs.registry(root)
bad = reg.runs_dir / "bad_record"
bad.mkdir()
(bad / "run.json").write_text("{}")
was_refused, found = refused(cn.runs.find, root=root)
check(not was_refused, "find() refused the whole registry because of one bad record")
names = [r.name for r in found]
check(names == ["good"], f"find() returned {names}, expected ['good']")
passed("bad record skipped, good run found")
