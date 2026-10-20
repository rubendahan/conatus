"""`hub serve --local` writes its one-session token to a 0600 file by default.

It used to print the token in the process output (a job log is readable by the group) unless
``--user-token-out`` was given. Now the token goes to ``--user-token-out`` or
``<state>/local_user_token`` (0600, atomic, never through a symbolic link) and only the path is
printed; ``--print-token`` is the explicit opt-in to print it."""
from __future__ import annotations

import argparse
import os
import stat

import pytest

pytest.importorskip("tornado")
from conatus.studio import hub  # noqa: E402

TOK = "tok-" + "x" * 40


def _ns(tmp_path, **kw):
    a = dict(state=str(tmp_path / "state"), user_token_out=None, print_token=False)
    a.update(kw)
    return argparse.Namespace(**a)


def test_default_is_a_0600_file_and_only_its_path_is_printed(tmp_path):
    msg = hub._local_user_token(_ns(tmp_path), 8765, TOK)
    f = tmp_path / "state" / hub.LOCAL_USER_TOKEN_FILE
    assert TOK not in msg and str(f) in msg
    assert f.read_text().strip() == TOK
    assert stat.S_IMODE(f.stat().st_mode) == 0o600


def test_print_token_is_an_explicit_opt_in(tmp_path):
    msg = hub._local_user_token(_ns(tmp_path, print_token=True), 8765, TOK)
    assert TOK in msg and "--print-token" in msg
    assert not (tmp_path / "state" / hub.LOCAL_USER_TOKEN_FILE).exists()
    out = tmp_path / "chosen"
    msg = hub._local_user_token(_ns(tmp_path, print_token=True, user_token_out=str(out)), 1, TOK)
    assert TOK in msg and out.read_text().strip() == TOK


def test_secret_file_never_follows_a_link_nor_reuses_a_loose_file(tmp_path):
    victim = tmp_path / "victim"
    victim.write_text("untouched\n")
    link = tmp_path / "token"
    link.symlink_to(victim)
    hub._write_secret(link, TOK)
    assert victim.read_text() == "untouched\n"            # the link was replaced, not followed
    assert not link.is_symlink() and link.read_text().strip() == TOK
    loose = tmp_path / "loose"
    loose.write_text("old\n")
    os.chmod(loose, 0o644)
    hub._write_secret(loose, TOK)
    assert stat.S_IMODE(loose.stat().st_mode) == 0o600
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_cli_declares_print_token():
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        hub.main(["serve", "--help"])
    assert "--print-token" in buf.getvalue() and hub.LOCAL_USER_TOKEN_FILE in buf.getvalue()
