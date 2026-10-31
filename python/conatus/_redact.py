"""Credential filter for recorded command lines. Standard library only, because a job wrapper
loads this file directly. Used for ``run.json`` (``runs.run._argv``) and for the job registries."""
from __future__ import annotations

import re
from collections.abc import Iterable

#: option names whose value is a credential, any case; not ``--passes`` nor ``--keyframes``
SECRET_NAME = re.compile(r"TOKEN|SECRET|PASSW|PASSPHRASE|CREDENTIAL|(API|PRIVATE|ACCESS)[-_]?KEY|"
                         r"AUTH|COOKIE|SESSION|BEARER|SIGNATURE|(^|[-_])(PASS|PWD|KEY)$", re.I)
_URL_PASSWORD = re.compile(r"(\b[a-zA-Z][a-zA-Z0-9+.-]*://[^/\s:@]*:)[^/\s@]+(@)")
_URL_QUERY = re.compile(r"([?&](?:[A-Za-z0-9_-]*(?:token|secret|signature|password|key))=)[^&\s]+", re.I)
_PLAIN_NAME = re.compile(r"-*[A-Za-z0-9_.-]+$")       # NAME=value, not a URL with a query
#: the option itself is a credential, so its value is hidden even when it starts with a dash
_EXACT_NAME = re.compile(r"(^|[-_])(TOKEN|SECRET|PASSWORD|PASSWD|PASS|PWD|KEY|BEARER|SIGNATURE|AUTH)$", re.I)
HIDDEN = "<redacted>"


def _text(arg: str) -> str:
    return _URL_QUERY.sub(rf"\1{HIDDEN}", _URL_PASSWORD.sub(rf"\1{HIDDEN}\2", arg))


def redact_argv(argv: Iterable[str]) -> list[str]:
    """``argv`` with credential values replaced: ``--token X``, ``--api-key=X``, ``TOKEN=X``, and a
    URL's password or secret query value. The argument after ``--token`` goes even when it starts
    with a dash; ``--keep-token-file -v`` and ``--no-auth -v`` keep their ``-v``. A short option
    (``-p``) is kept: its meaning depends on the program, and hiding it would hide ordinary options
    from the record."""
    out: list[str] = []
    hide: str | bool = False
    for a in map(str, argv):
        if hide == "any" or (hide == "value" and not a.startswith("-")):
            out.append(HIDDEN)
            hide = False
            continue
        hide = False
        name, eq, _value = a.partition("=")
        if eq and _PLAIN_NAME.match(name) and SECRET_NAME.search(name.lstrip("-")):
            out.append(f"{name}={HIDDEN}")
        elif a.startswith("--no-") and "=" not in a:        # a switch: it has no value
            out.append(a)
        elif a.startswith("--") and SECRET_NAME.search(a.lstrip("-")):
            out.append(a)
            hide = "any" if _EXACT_NAME.search(a.lstrip("-")) else "value"
        else:
            out.append(_text(a))
    return out
