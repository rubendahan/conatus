"""F203 (security audit 2026-10-20): the Studio hub's auth file is written 0600, never through a
planted link.

``AuthDB.save`` wrote the fixed name ``auth.tmp`` with ``O_CREAT | O_TRUNC``: a symbolic link at
that name had its target overwritten. Expected: the target untouched, ``auth.json`` mode 0600.
"""
import os
import stat
from pathlib import Path

from _acceptance import check, need, passed, start

start(203, "AuthDB.save ignores a planted link")
need("conatus.studio.auth")
from conatus.studio.auth import AuthDB  # noqa: E402

d = Path("state").resolve()
d.mkdir()
victim = Path("victim.txt").resolve()
victim.write_text("precious\n")
(d / "auth.tmp").symlink_to(victim)
db = AuthDB({}, d / "auth.json")
db.add_relay("cluster")
db.save()
check(victim.read_text() == "precious\n", "the link target was overwritten with the auth file")
check(stat.S_IMODE(os.stat(d / "auth.json").st_mode) == 0o600, "auth.json is not 0600")
passed("auth.json 0600, link target untouched")
