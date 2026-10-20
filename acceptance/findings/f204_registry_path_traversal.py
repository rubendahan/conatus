"""F204 (security review 2026-10-20): a run id or an attached file name cannot leave the registry.

cn.runs.get("../x") and run.file_path("../run.json") resolved paths outside the registry and the
run's files/ directory. Expected: typed refusals.
"""
from pathlib import Path

from _acceptance import check, passed, refused, start

import conatus as cn

start(204, "registry path components stay inside the registry")
root = Path("reg").resolve()
with cn.runs.track("paths", root=root) as run:
    pass
outcomes = []
for rid in ("../x", "..", "a/b", "/etc"):
    was_refused, _ = refused(cn.runs.get, rid, root=root)
    outcomes.append(was_refused)
for name in ("../run.json", "..", "/etc/passwd"):
    was_refused, _ = refused(run.file_path, name)
    outcomes.append(was_refused)
check(all(outcomes), f"refusals: {outcomes}")
passed("run ids and file names refused when they are paths")
