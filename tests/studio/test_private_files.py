"""Review F12 (external review 0.1.1): Studio, agent and run-telemetry files are private.

Under the common umask 022 the telemetry spool was created 0755 and ``<run>.jsonl`` 0644, so
other accounts on a shared machine could read a run's log, scalars and films; the hub and spool
audit logs (user names, addresses, commands) were 0644 too. Expected, whatever the umask:
directories created by Conatus 0700, files 0600, and a 0644 file of an older version brought
back to 0600 when it is written again. The oracle is the literal mode, read with ``os.stat``.
"""
from __future__ import annotations

import os
import stat
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from conatus.telemetry import Telemetry


def mode(p: Path) -> int:
    return stat.S_IMODE(p.stat().st_mode)


@contextmanager
def umask(value: int):
    old = os.umask(value)
    try:
        yield
    finally:
        os.umask(old)


@pytest.mark.parametrize("mask", [0o022, 0o002, 0o000])
def test_telemetry_spool_and_files_are_private_whatever_the_umask(tmp_path, mask):
    spool = tmp_path / "a" / "spool"
    with umask(mask):
        t = Telemetry("review_run", spool=spool, capacity=32, scalar_interval=0.0,
                      frame_interval=0.0, max_px=16, film_max_bytes=1 << 20)
        t.log("permission probe")
        t.scalars(energy=1.0)
        t.frame("u", np.arange(16.0).reshape(4, 4), t=0.0, iteration=0)
        t.close()
    assert mode(spool) == 0o700 and mode(spool.parent) == 0o700
    files = sorted(spool.iterdir())
    names = {p.name for p in files}
    assert {"review_run.jsonl", "review_run.frame.u.json", "review_run.film.ndjson"} <= names
    assert {p.name: mode(p) for p in files} == {p.name: 0o600 for p in files}


def test_an_older_world_readable_log_is_tightened_when_written_again(tmp_path):
    spool = tmp_path / "spool"
    spool.mkdir()
    old = spool / "review_run.jsonl"
    old.write_text("")
    os.chmod(old, 0o644)
    t = Telemetry("review_run", spool=spool, capacity=8, scalar_interval=0.0, frame_interval=0.0,
                  max_px=8)
    t.log("again")
    t.close()
    assert mode(old) == 0o600


def test_audit_logs_are_private(tmp_path):
    pytest.importorskip("tornado")
    from conatus.studio.auth import AuditLog
    with umask(0o022):
        AuditLog(tmp_path / "state" / "audit.jsonl", echo=False).write(event="login", user="u")
    assert mode(tmp_path / "state") == 0o700
    assert mode(tmp_path / "state" / "audit.jsonl") == 0o600


def test_hub_secret_directory_and_auth_file_are_private(tmp_path):
    pytest.importorskip("tornado")
    from conatus.studio import hub
    from conatus.studio.auth import AuthDB
    with umask(0o022):
        out = hub._write_secret(tmp_path / "cfg" / "conatus" / "token", "t" * 43)
        db = AuthDB({}, path=tmp_path / "state2" / "auth.json")
        db.add_user("u", None)
        db.save()
    assert mode(out) == 0o600 and mode(out.parent) == 0o700 and mode(out.parent.parent) == 0o700
    assert mode(tmp_path / "state2") == 0o700 and mode(tmp_path / "state2" / "auth.json") == 0o600


def test_launcher_registry_and_audit_are_private(tmp_path):
    pytest.importorskip("tornado")
    from conatus.studio.launch import Launcher
    lab = Launcher.__new__(Launcher)
    lab.audit_path = tmp_path / "jobs" / "studio_jobs_audit.jsonl"
    with umask(0o022):
        lab._audit(op="catalog", user="u", result="ok")
    assert mode(lab.audit_path.parent) == 0o700 and mode(lab.audit_path) == 0o600


def test_helpers(tmp_path):
    from conatus._private import open_private, private_dir
    with umask(0o022):
        d = private_dir(tmp_path / "x" / "y")
        with open_private(d / "f.bin", "wb") as fh:
            fh.write(b"1")
        with open_private(d / "f.bin", "ab") as fh:
            fh.write(b"2")
    assert (d / "f.bin").read_bytes() == b"12"
    assert mode(d) == mode(d.parent) == 0o700 and mode(d / "f.bin") == 0o600
    existing = tmp_path / "shared"
    existing.mkdir(mode=0o755)
    os.chmod(existing, 0o755)
    private_dir(existing)
    assert mode(existing) == 0o755                     # an existing directory is not changed
    with pytest.raises(ValueError):
        open_private(d / "g", "r")
    (tmp_path / "file").write_text("")
    with pytest.raises(NotADirectoryError):
        private_dir(tmp_path / "file")


def test_open_private_refuses_a_symbolic_link(tmp_path):
    """Integration review of 0.1.2: ``open_private`` followed a symbolic link at the last path
    component, so a link planted in a shared spool redirected the appends (and the ``fchmod``) to
    its target. Oracle: the target's bytes and mode, read back with ``os.stat``."""
    from conatus._private import open_private
    target = tmp_path / "target"
    target.write_bytes(b"keep")
    os.chmod(target, 0o644)
    link = tmp_path / "run.jsonl"
    link.symlink_to(target)
    for m in ("a", "w"):
        with pytest.raises(OSError):
            open_private(link, m)
    assert target.read_bytes() == b"keep" and mode(target) == 0o644
