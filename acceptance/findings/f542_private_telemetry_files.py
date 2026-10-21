"""F542 (external review of 0.1.1, F12): telemetry and Studio audit files are private whatever
the umask.

Under umask 022 the spool was created 0755 and ``<run>.jsonl`` 0644: other accounts could read
a run's log, numbers and film. Expected: the spool directory 0700, every file in it 0600, and the
Studio audit log 0600, under umask 022.
"""
import os
import stat
from pathlib import Path

import numpy as np

from _acceptance import check, passed, start

start(542, "telemetry spool 0700 and its files 0600 under umask 022")
from conatus.telemetry import Telemetry  # noqa: E402

spool = Path("spool")
old = os.umask(0o022)
try:
    t = Telemetry("review_run", spool=spool, capacity=32, scalar_interval=0.0, frame_interval=0.0,
                  max_px=16, film_max_bytes=1 << 20)
    t.log("permission probe")
    t.frame("u", np.ones((4, 4)), t=0.0, iteration=0)
    t.close()
    audit = None
    try:
        from conatus.studio.auth import AuditLog
        AuditLog(spool / "studio_audit.jsonl", echo=False).write(event="probe")
        audit = spool / "studio_audit.jsonl"
    except ModuleNotFoundError:
        pass                                      # Studio not shipped: telemetry only
finally:
    os.umask(old)

modes = {p.name: oct(stat.S_IMODE(p.stat().st_mode)) for p in spool.iterdir()}
dmode = oct(stat.S_IMODE(spool.stat().st_mode))
check(dmode == "0o700", f"spool directory {dmode}")
check("review_run.jsonl" in modes and "review_run.film.ndjson" in modes, f"files {sorted(modes)}")
check(set(modes.values()) == {"0o600"}, f"file modes {modes}")
passed(f"spool {dmode}; {len(modes)} files 0600" + ("" if audit else " (no Studio audit: not shipped)"))
