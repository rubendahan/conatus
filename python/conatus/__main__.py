"""``python -m conatus <command>``: command-line entry points (``doctor``, ``studio``).

    python -m conatus --version       # conatus, compiled core and Python versions (a bug report)
    python -m conatus doctor          # what this machine can run; exit 1 if something required is missing
    python -m conatus doctor --json   # same report as JSON
    python -m conatus studio status kh_demo     # follow / pilot a run (JSON; conatus.studio.cli)
    python -m conatus agent check run.py        # JSON interface for AI agents (conatus.agent)
    python -m conatus docs guide/errors         # where to read a guide page named by an error
"""
from __future__ import annotations

import argparse
import sys


def version_line() -> str:
    """``conatus 0.1.0 (compiled core 0.1.0, Python 3.12.3)``: the first line of a bug report.
    The core is named "not loaded" when it cannot be imported, and its own
    version when it differs from the package's (a build of another checkout)."""
    import platform

    from . import __version__
    try:
        from . import runtime
        built = getattr(runtime._core, "__version__", "?")
        core = "compiled core " + ("same version" if built == __version__ else built)
    except Exception as err:  # noqa: BLE001 -- --version never fails: the reason is printed
        core = f"compiled core not loaded: {type(err).__name__}"
    return f"conatus {__version__} ({core}, Python {platform.python_version()})"


class _Version(argparse.Action):
    """``--version``, computed only when asked (it imports the compiled core)."""

    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS,
                 help="print the conatus, compiled core and Python versions and exit"):
        super().__init__(option_strings, dest, nargs=0, default=default, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        print(version_line())
        parser.exit()


def main(argv: list[str] | None = None) -> int:
    """Dispatch the subcommand; returns the process exit code."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["studio"]:                     # its own parser (many subcommands)
        from .studio.cli import main as studio_main
        return studio_main(argv[1:])
    if argv[:1] == ["agent"]:                      # JSON interface for AI agents (its own parser)
        from .agent import main as agent_main
        return agent_main(argv[1:])
    p = argparse.ArgumentParser(prog="python -m conatus", description="Conatus command line.")
    p.add_argument("--version", action=_Version)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor", help="check Python, the compiled core, GPU, MPI, storage "
                                      "(read-only); exit 0 if nothing required is missing")
    d.add_argument("--json", action="store_true", help="print the report as JSON")
    sub.add_parser("studio", help="follow and pilot Studio runs from a terminal or an agent "
                                  "(python -m conatus studio --help)")
    sub.add_parser("agent", help="JSON interface for AI agents: info, explain <name>, "
                                 "check <script.py> (python -m conatus agent --help)")
    g = sub.add_parser("docs", help="where to read a page of the user guide (the name printed "
                                    "on the 'docs:' line of an error, e.g. guide/errors)")
    g.add_argument("page", nargs="?", default="index", help="guide page, e.g. guide/errors")
    a = p.parse_args(argv)
    if a.command == "doctor":
        from ._doctor import main as doctor_main
        return doctor_main(["--json"] if a.json else [])
    if a.command == "docs":
        return _docs(a.page)
    return 2                                       # unreachable: argparse enforces the choice


def _docs(page: str) -> int:
    """Print where the guide page ``page`` can be read; exit 1 if it is not found."""
    from pathlib import Path
    page = page.split("#", 1)[0].removesuffix(".md").strip("/")
    root = Path(__file__).resolve().parents[2] / "docs" / "site"     # a source tree
    f = root / f"{page}.md"
    if f.is_file():
        print(f)
        print(f"(source of the guide page {page!r}; the HTML site: bash docs/site/build.sh)")
        return 0
    if root.is_dir():
        pages = sorted(str(p.relative_to(root))[:-3] for p in root.rglob("*.md")
                       if "_generated" not in p.parts)
        from .errors import did_you_mean
        print(f"no guide page {page!r} in {root}." + did_you_mean(page, pages))
        return 1
    print(f"this installation has no copy of the user guide: the page {page!r} is "
          f"docs/site/{page}.md in the Conatus source tree (the website is built from it with "
          "docs/site/build.sh); no public site yet")
    return 1


if __name__ == "__main__":
    sys.exit(main())
