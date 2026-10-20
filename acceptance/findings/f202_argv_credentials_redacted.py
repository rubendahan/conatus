"""F202 (security audit 2026-10-20): a run record must not store a credential given on the
command line.

``run.json`` kept ``sys.argv`` verbatim (``--token X``, ``--api-key=X``) on a shared store.
Expected: the value of every credential-like option is ``<redacted>``; other options are kept.
"""
import sys
from pathlib import Path

from _acceptance import check, passed, start

import conatus.runs as R

start(202, "credential options of the command line are redacted in run records")
secret = "Qw" * 22
sys.argv = ["job.py", "--token", secret, f"--api-key={secret}", "--passes", "3"]


def plus_one(n):
    return n + 1


root = Path("registry").resolve()
check(R.track("argv", root=root)(plus_one)(1) == 2, "the tracked call returned a wrong value")
argv = R.find(name="argv", root=root)[0].rec["invocation"]["argv"]
check(secret not in " ".join(argv), f"the credential is in the run record: {argv}")
check(argv[-2:] == ["--passes", "3"], f"an ordinary option was altered: {argv}")
passed("argv redacted")
