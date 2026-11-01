#!/usr/bin/env python3
"""Generate ``llms.txt`` and ``docs-bundle.json`` from the pages of this documentation.

    python docs/site/gen_llms.py --out DIR       # writes DIR/llms.txt and DIR/docs-bundle.json
    python docs/site/gen_llms.py --llms FILE     # writes only the index

The site build runs it on the tree it publishes, so both files list the pages of that tree: titles and
summaries come from the pages, and the bundle holds their text with includes expanded. Standard
library only. The output has no timestamp, so the same pages give the same bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SITE = Path(__file__).resolve().parent
SKIP_DIRS = {"_generated", "_static", "_templates", "_ext", "vendor", "outputs", "media", "_vendor"}
SECTIONS = (("guide", "User guide"), ("tutorials", "Tutorials"), ("math", "Mathematics"),
            ("api", "API reference"), ("gallery", "Gallery"))
ESSENTIAL = ("guide/agents", "guide/recipes", "guide/mcp", "guide/error_codes", "guide/errors",
             "guide/studio_agents", "guide/explain", "guide/experimental")
HEAD = """# Conatus

> Spectral PDEs and dynamical systems on CPU and GPU, in Python. Equations are written as on paper
> (`cn.Model`, conditions by boundary name); Conatus discretises them, integrates, solves
> eigenproblems and finds invariant solutions, bifurcations, Floquet spectra and optimal
> perturbations. It never changes a numerical choice silently: it refuses and says what to write.

Agents: read [AGENTS.md](AGENTS.md) first (the whole usable surface, one call per line, and the
rules that prevent mistakes). Dry-run a script up to its first expensive call: `python -m conatus agent
check script.py` (JSON). Machine facts: `python -m conatus agent` (JSON). MCP clients: `python -m conatus
mcp`, read-only by default. Every page in one JSON file: `docs-bundle.json`.
"""


def pages(site: Path = SITE) -> list[Path]:
    """Every page of the documentation, in a fixed order."""
    out = [p for p in site.rglob("*.md")
           if not (set(p.relative_to(site).parts[:-1]) & SKIP_DIRS) and not p.name.startswith("_")]
    return sorted(out, key=lambda p: p.relative_to(site).as_posix())


def _plain(text: str) -> str:
    text = re.sub(r"\{[a-z:]+\}`(?:[^`<]*<)?([^`>]*)>?`", r"\1", text)      # {doc}`x`, {doc}`t <x>`
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.replace("**", "")
    return re.sub(r"\s+", " ", text).strip()


def describe(text: str) -> tuple[str, str]:
    """(title, summary) of a page: its first heading and the first sentences of its first paragraph."""
    title, summary, block, fence = "", "", [], False
    for line in text.splitlines() + [""]:
        if line.startswith(("```", ":::", "~~~")):
            fence = not fence
            block = []
            continue
        if fence:
            continue
        if not title:
            if line.startswith("# "):
                title = _plain(line[2:])
            continue
        if line.strip() and not line.startswith(("#", "|", "- ", "* ", "<", "(", "1. ", ">")):
            block.append(line)
        elif block:
            summary = _plain(" ".join(block))
            break
    if len(summary) > 200:
        cut = summary.rfind(". ", 0, 200)
        summary = summary[:cut + 1] if cut > 40 else summary[:200].rsplit(" ", 1)[0] + " ..."
    return title, summary


def experimental_names(site: Path) -> list[str]:
    """The features listed by the page on experimental features, as that page names them."""
    page = site / "guide" / "experimental.md"
    if not page.is_file():
        return []
    return [_plain(n) for n in re.findall(r"^- \*\*([^*]+)\*\*", page.read_text(encoding="utf-8"), re.M)]


def llms(site: Path = SITE) -> str:
    """The text of ``llms.txt``."""
    described = {}
    for p in pages(site):
        rel = p.relative_to(site).with_suffix("").as_posix()
        described[rel] = describe(p.read_text(encoding="utf-8"))

    def row(rel: str) -> str:
        title, summary = described[rel]
        return f"- [{title or rel}](docs/{rel}.html)" + (f": {summary}" if summary else "")
    out = [HEAD, "## Essentials", "", "- [AGENTS.md](AGENTS.md): compact reference for AI agents"]
    out += [row(r) for r in ESSENTIAL if r in described]
    names = experimental_names(site)
    if names:
        out += ["", "Experimental features (shipped, tested, not promised; first use warns "
                    "`ExperimentalWarning`): " + "; ".join(names) + ".", ""]
    else:
        out += [""]
    top = [r for r in described if "/" not in r]
    out += ["## Start", ""] + [row(r) for r in top] + [""]
    for folder, heading in SECTIONS:
        rows = [row(r) for r in described if r.startswith(folder + "/") and r not in ESSENTIAL]
        if rows:
            out += [f"## {heading}", ""] + rows + [""]
    return "\n".join(out).rstrip() + "\n"


_INCLUDE = re.compile(r"^```\{(literalinclude|include)\} ([^\n]+)\n((?::[^\n]*\n)*)```$", re.M)


def _select(body: str, opts: dict[str, str]) -> str:
    """The part of an included file that the options of the directive name."""
    after, before = opts.get("start-after", "").strip('"'), opts.get("end-before", "").strip('"')
    if after and after in body:
        body = body.split(after, 1)[1].split("\n", 1)[-1]
    if before and before in body:
        body = body.split(before, 1)[0].rsplit("\n", 1)[0]
    if "lines" in opts:
        rows = body.splitlines()
        keep: list[str] = []
        for part in opts["lines"].split(","):
            lo, _, hi = part.strip().partition("-")
            start = int(lo) - 1 if lo else 0
            keep += rows[start:(int(hi) if hi else len(rows)) if "-" in part else start + 1]
        body = "\n".join(keep)
    if "dedent" in opts and opts["dedent"].isdigit():
        n = int(opts["dedent"])
        body = "\n".join(r[n:] if r[:n].strip() == "" else r for r in body.splitlines())
    return body


def expand(text: str, page: Path) -> str:
    """The page with its include directives replaced by the files they name. An include that names
    something missing keeps its directive."""
    def one(m: re.Match) -> str:
        target = (page.parent / m.group(2).strip()).resolve()
        if not target.is_file():
            return m.group(0)
        body = target.read_text(encoding="utf-8", errors="replace")
        if m.group(1) == "include":
            return expand(body, target)
        opts = dict(re.findall(r":([a-z-]+):[ \t]*([^\n]*)", m.group(3)))
        body = _select(body, opts)
        return f"```{opts.get('language', '')}\n{body.rstrip()}\n```"
    return _INCLUDE.sub(one, text)


def bundle(site: Path = SITE) -> dict:
    """Every page as JSON: path, address, title, summary, sha256 and text with includes expanded."""
    rows = []
    for p in pages(site):
        rel = p.relative_to(site).with_suffix("").as_posix()
        text = expand(p.read_text(encoding="utf-8"), p)
        title, summary = describe(text)
        rows.append({"path": rel, "url": f"docs/{rel}.html", "title": title, "summary": summary,
                     "sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text})
    init = site.parents[1] / "python" / "conatus" / "__init__.py"
    version = None
    if init.is_file():
        m = re.search(r'^__version__ = "([^"]+)"', init.read_text(encoding="utf-8"), re.M)
        version = m.group(1) if m else None
    return {"schema": 1, "name": "conatus", "version": version, "pages": rows}


def main(argv: list[str] | None = None) -> int:
    """The command line (module docstring)."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", help="directory for llms.txt and docs-bundle.json")
    p.add_argument("--llms", help="write only llms.txt, at this path")
    p.add_argument("--site", default=str(SITE), help="documentation directory (default: this one)")
    a = p.parse_args(argv)
    site = Path(a.site).resolve()
    if a.llms:
        Path(a.llms).write_text(llms(site), encoding="utf-8")
    if a.out:
        out = Path(a.out)
        (out / "llms.txt").write_text(llms(site), encoding="utf-8")
        (out / "docs-bundle.json").write_text(json.dumps(bundle(site), ensure_ascii=False, sort_keys=True,
                                                         separators=(",", ":")) + "\n", encoding="utf-8")
    return 0 if a.out or a.llms else 2


if __name__ == "__main__":
    raise SystemExit(main())
