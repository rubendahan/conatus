"""Review F28 (external review 0.1.1): where `hub serve --local` keeps its token.

The guide named ``hub_state/local_user_token``; without ``--state`` the local hub used a new
temporary directory, and an explicit ``--state hub_state`` was replaced by one too (the code
compared the value with the default string). Expected: an explicit ``--state`` is used as given;
``$CN_HUB_STATE`` next; then a private temporary directory for ``--local`` and ``hub_state`` for
a hosted hub; the guide says to read the printed path.
"""
from __future__ import annotations

import stat
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("tornado")
from conatus.studio import hub  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def test_explicit_state_is_kept_even_when_spelt_like_the_default(monkeypatch):
    monkeypatch.delenv("CN_HUB_STATE", raising=False)
    assert hub.state_dir("hub_state", local=True) == "hub_state"
    assert hub.state_dir("/x/y", local=True) == "/x/y"


def test_environment_then_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("CN_HUB_STATE", str(tmp_path / "env_state"))
    assert hub.state_dir(None, local=True) == str(tmp_path / "env_state")
    monkeypatch.delenv("CN_HUB_STATE")
    assert hub.state_dir(None, local=False) == "hub_state"
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    d = Path(hub.state_dir(None, local=True))
    assert d.parent == tmp_path and d.name.startswith("conatus_hub_")
    assert stat.S_IMODE(d.stat().st_mode) == 0o700


def test_main_passes_the_explicit_state_to_serve(monkeypatch, tmp_path):
    seen = {}

    async def fake_serve(a):
        seen["state"] = a.state
    monkeypatch.delenv("CN_HUB_STATE", raising=False)
    monkeypatch.setattr(hub, "_serve", fake_serve)
    monkeypatch.chdir(tmp_path)
    hub.main(["--state", "hub_state", "serve", "--local", "--port", "1"])
    assert seen["state"] == "hub_state"


def test_guide_tells_to_read_the_printed_path():
    text = (REPO / "docs" / "site" / "guide" / "studio.md").read_text()
    assert "--state ./hub_state" in text and "prints the path" in text
    assert "`hub_state/local_user_token`" not in text
    readme_path = REPO / "studio" / "README.md"   # not in the public tree
    readme = readme_path.read_text() if readme_path.exists() else ""
    assert "writes its one-session token to hub_state/local_user_token" not in readme
