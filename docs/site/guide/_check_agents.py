"""Executable example of the guide page "For AI agents" (docs/site/guide/agents.md).

The script that an agent checks is ``outputs/os_check.py`` (included verbatim on the page, so
its line numbers are those of the report); ``--sync`` writes what
``python -m conatus agent check`` returns for it to ``outputs/agent_check.json`` (the long
``explain`` texts cut to their first line, the timing removed, the path relative), included on
the page as is. The development test suite compares that file with a fresh run. Not a Sphinx source (leading underscore).

    python docs/site/guide/_check_agents.py [--sync]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs" / "agent_check.json"

SCRIPT = OUT.with_name("os_check.py")


def stable_report() -> dict:
    """The check report of ``os_check.py`` without its volatile parts (time, path, long texts)."""
    from conatus.agent import check
    rep = check(SCRIPT)
    rep.pop("elapsed_s")
    rep["script"] = "os_check.py"
    rep["explain"] = {k: v.splitlines()[0] + " ..." for k, v in rep["explain"].items()}
    return rep


def main(argv: list[str]) -> int:
    text = json.dumps(stable_report(), sort_keys=True, indent=1) + "\n"
    if "--sync" in argv:
        OUT.write_text(text)
        print(f"wrote {OUT}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
