"""F206 (security review 2026-10-20): an adjoint journal only references its own files.

A forged state.json naming "../../outside.npy" made the resume read, hard-link and later delete
files outside the journal directory. Expected: a typed refusal when the journal is loaded.
"""
import json
from pathlib import Path

from _acceptance import check, passed, refused, start

start(206, "adjoint journal names stay in the journal")
from conatus.adjoint.checkpoints import Journal  # noqa: E402

d = Path("journal").resolve()
d.mkdir()
fp = {"x": 1}
(d / "state.json").write_text(json.dumps({"fingerprint": fp, "generation": 0, "cursor": 0,
                                          "progress": None, "bars": ["../../outside.npy"],
                                          "work": None, "checkpoints": {}, "extra": {}}))
was_refused, _ = refused(Journal(d, fp).load)
check(was_refused, "a journal naming ../../outside.npy was loaded")
passed("refused")
