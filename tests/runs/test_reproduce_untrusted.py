"""reproduce() never runs code chosen by a record someone else can write (security).

``run.reproduce()`` of a decorated call imports ``module:function`` named in ``run.json`` and calls
it with the recorded arguments. On a registry another account can write (a group store, a world-
writable directory), a forged record naming any importable function (``os:popen``, a planted
module) ran it. The record, its run directory and the registry's ``runs/`` directory must now
belong to the caller and be writable by nobody else; otherwise a typed refusal names ``fn=``.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

import conatus.runs as R
from conatus.errors import OptionError

GOOD = "def area(w, h):\n    return w * h\n"
EVIL = ("def payload(w, h):\n    import pathlib\n"
        "    pathlib.Path(w).write_text('code from a forged record ran')\n    return 0\n")


@pytest.fixture()
def recorded(tmp_path, monkeypatch):
    mods = tmp_path / "mods"
    mods.mkdir()
    (mods / "audit_goodmod.py").write_text(GOOD)
    (mods / "audit_evilmod.py").write_text(EVIL)
    monkeypatch.syspath_prepend(str(mods))
    for m in ("audit_goodmod", "audit_evilmod"):
        sys.modules.pop(m, None)
    import audit_goodmod
    f = R.track("area")(audit_goodmod.area)
    assert f(2.0, 3.0) == 6.0
    run = R.find(name="area")[0]
    return run, tmp_path / "marker.txt"


def _forge(run, marker):
    rec_path = run.path / "run.json"
    rec = json.loads(rec_path.read_text())
    rec["inputs"]["target"] = "audit_evilmod:payload"
    rec["inputs"]["args"] = {"w": str(marker), "h": 1}
    rec_path.write_text(json.dumps(rec))
    return rec_path


def test_own_private_record_is_reproduced(recorded):
    run, _ = recorded
    rep = R.get(run.id).reproduce(record=False)
    assert rep.identical


@pytest.mark.parametrize("where", ["record", "run_dir", "runs_dir"])
def test_record_writable_by_others_is_refused_before_any_import(recorded, where):
    run, marker = recorded
    rec_path = _forge(run, marker)
    target = {"record": rec_path, "run_dir": run.path, "runs_dir": run.path.parent}[where]
    os.chmod(target, os.stat(target).st_mode | 0o022)
    try:
        with pytest.raises(OptionError, match="writable by group or others") as e:
            R.get(run.id).reproduce(record=False)
        assert "fn=" in str(e.value)
        assert not marker.exists(), "the function named by a forged record ran"
        assert "audit_evilmod" not in sys.modules, "the module named by the record was imported"
    finally:
        os.chmod(target, os.stat(target).st_mode & ~0o022)


def test_fn_given_by_the_caller_is_not_checked(recorded):
    run, marker = recorded
    rec_path = run.path / "run.json"
    os.chmod(rec_path, 0o666)
    import audit_goodmod
    assert R.get(run.id).reproduce(fn=audit_goodmod.area, record=False).identical
    assert not marker.exists()
