"""Every hub command the guide and the module docstring show is accepted by the hub's own parser
(0.2.1: the Studio guide said to pass ``--state ./hub_state`` after ``serve``, which argparse
refuses: ``--state`` is an option of the hub, before the subcommand). The oracle is argparse."""
from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

pytest.importorskip("tornado")
from conatus.studio import hub  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CMD = re.compile(r"python -m conatus\.studio\.hub((?:[ \t]+[^\s`#|;)\"']+)+)")
SHELL_VAR = re.compile(r'^(\w+)="python -m conatus\.studio\.hub([^"]*)"$', re.M)   # H="python -m ... --state D"
PLACEHOLDER = {"<dir>": "spool_dir", "DIR": "state_dir", "FILE": "token_file", "NAME": "agent1"}


def _commands() -> list[tuple[str, str]]:
    """(where, arguments after ``python -m conatus.studio.hub``) of every command shown, a shell
    variable holding the start of the command included (``$H addagent ...``)."""
    out = []
    for f in sorted((ROOT / "docs/site/guide").glob("*.md")) + [ROOT / "python/conatus/studio/hub.py"]:
        text = f.read_text()
        for m in CMD.finditer(text):
            if text[m.start() - 1] != '"':
                out.append((f"{f.name}: {m.group(0).strip()}", m.group(1)))
        for v in SHELL_VAR.finditer(text):
            for line in re.findall(rf"^\${v.group(1)} ([^#\n]+)", text, re.M):
                out.append((f"{f.name}: ${v.group(1)} {line.strip()}", f"{v.group(2)} {line}"))
    return out


COMMANDS = _commands()


def test_the_guide_shows_hub_commands():
    assert any("--state" in c for c, _ in COMMANDS)
    assert any("serve --local" in c for c, _ in COMMANDS)
    assert any("$H addagent" in c for c, _ in COMMANDS)


@pytest.mark.parametrize("where, args", COMMANDS, ids=[c for c, _ in COMMANDS])
def test_every_documented_hub_command_parses(where, args):
    argv = [PLACEHOLDER.get(a, a) for a in shlex.split(args)]
    try:
        hub.build_parser().parse_args(argv)
    except SystemExit as err:
        pytest.fail(f"{where}: refused by the hub's parser (exit {err.code})")


def test_state_after_serve_is_refused_and_before_is_accepted():
    with pytest.raises(SystemExit):
        hub.build_parser().parse_args(["serve", "--state", "./hub_state", "--local"])
    a = hub.build_parser().parse_args(["--state", "./hub_state", "serve", "--local"])
    assert a.state == "./hub_state" and a.local and a.cmd == "serve"
