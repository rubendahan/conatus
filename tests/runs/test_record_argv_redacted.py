"""Run records never hold a credential given on the command line (security).

``run.json`` stored ``sys.argv`` verbatim; it lives on a shared store, is copied by the registry
tools and shown by the Studio. The value of a credential-like option is now ``<redacted>``."""
from __future__ import annotations

import sys

import conatus.runs as R


def _plus_one(n):
    return n + 1


def test_credential_options_are_redacted(monkeypatch):
    secret = "Sx" * 22
    monkeypatch.setattr(sys, "argv", ["job.py", "--token", secret, f"--api-key={secret}",
                                      "--hub-password", secret, "--passes", "3", "--Ra", "1e5",
                                      "--keep-token-file", "-v"])
    f = R.track("argv")(_plus_one)
    assert f(1) == 2
    argv = R.find(name="argv")[0].rec["invocation"]["argv"]
    assert secret not in " ".join(argv)
    assert argv == ["job.py", "--token", "<redacted>", "--api-key=<redacted>", "--hub-password",
                    "<redacted>", "--passes", "3", "--Ra", "1e5", "--keep-token-file", "-v"]
