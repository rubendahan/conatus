"""Provenance never records a commit of a repository that is not Conatus.

An installed package lives in ``<env>/lib/python3.x/site-packages/conatus``; ``git -C`` from there
searches upwards and, for a virtual environment inside the user's project (``.venv/``), found the
user's repository: its HEAD was recorded as the Conatus commit that ran.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

import conatus
import conatus.provenance as prov
from conatus.runs import analysis

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _user_repo_with_env(tmp_path):
    """A git repository (the user's project) holding a fake environment with conatus installed."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=u", "-c", "user.email=u@x",
                    "commit", "-q", "--allow-empty", "-m", "user project"], check=True)
    pkg = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages" / "conatus"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    return pkg


def test_installed_package_inside_a_user_repository_has_no_git_head(tmp_path, monkeypatch):
    pkg = _user_repo_with_env(tmp_path)
    monkeypatch.setattr(prov, "PKG_DIR", pkg)
    monkeypatch.setattr(prov, "REPO_ROOT", pkg.parents[1])
    assert not prov._in_git_checkout()
    assert prov._git("rev-parse", "HEAD") is None          # the user's HEAD is never reported
    state = prov.code_state()
    assert state["git_head"] is None and state["git_dirty_paths"] == []
    assert state["code_matches_commit"] is False
    assert state["installed_version"] == conatus.__version__
    assert "installed package" in state["warning"]


def test_git_diff_stat_refuses_option_like_commits(tmp_path, monkeypatch):
    pkg = _user_repo_with_env(tmp_path)
    monkeypatch.setattr(prov, "PKG_DIR", pkg)
    monkeypatch.setattr(prov, "REPO_ROOT", pkg.parents[1])
    out = tmp_path / "written"
    assert analysis._git_diff_stat(f"--output={out}", "HEAD") is None
    assert not out.exists()
    assert analysis._git_diff_stat("abc1234", "def5678") is None    # not a Conatus checkout
