"""Pages generated at build time for the documentation site (called by ``conf.py``).

- gallery:  one page per script of ``examples/`` (docstring, result row of examples/README.md,
            regression test, full source), plus the summary table;
- API:      autosummary tables of ``conatus.expert``, one section per source module.

Everything is written to ``_generated/`` folders (gitignored) and rewritten only when the text
changes, so incremental builds stay incremental.
"""
from __future__ import annotations

import ast
import importlib
import re
import types
from pathlib import Path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def _readme_rows(readme: Path) -> dict[str, list[str]]:
    """Rows of the examples/README.md table keyed by script path (e.g. 'shear/kelvin_helmholtz.py')."""
    rows, width = {}, None
    for line in readme.read_text().splitlines():
        if not line.startswith("|"):
            width = None
            continue
        # cells are separated by unescaped bars; a bar inside a cell is written \| (|k|)
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip()[1:-1])]
        if width is None:
            width = len(cells)                        # the header row of the table
        elif len(cells) != width:
            raise ValueError(f"{readme}: {len(cells)} cells instead of {width} (an unescaped | ?): "
                             f"{line[:80]}")
        m = re.search(r"`([\w/]+\.py)`", cells[1]) if len(cells) > 1 else None
        if m and len(cells) >= 5:
            rows[m.group(1)] = cells
    return rows


def gallery(site: Path, root: Path) -> None:
    out = site / "gallery" / "_generated"
    rows = _readme_rows(root / "examples" / "README.md")
    table = ["| Case | Obtained | Reference |", "|---|---|---|"]
    for script in sorted((root / "examples").glob("*/*.py")):
        rel = f"{script.parent.name}/{script.name}"
        slug = f"{script.parent.name}_{script.stem}"
        doc = ast.get_docstring(ast.parse(script.read_text())) or script.stem
        # the title is the docstring's first paragraph (it may wrap over several lines), the body the rest
        head, _, body = doc.partition("\n\n")
        title = " ".join(head.split())
        row = rows.get(rel)
        # the script asserts its reference itself (its last lines); the regression tests of the
        # gallery are not part of the published tree, so the page does not name them
        lines = [f"# {title.rstrip('.')}", "",
                 "**Goal.** Run this case and compare the value it obtains with the reference"
                 + (" below." if row else " named in the script."), "",
                 f"*Family:* `{script.parent.name}` · *Script:* `examples/{rel}`", ""]
        if row:
            lines += ["```{list-table}", ":widths: 20 80", "",
                      "* - Obtained", f"  - {row[2]}", "* - Reference", f"  - {row[3]}", "```", ""]
        lines += ["```text", body.strip(), "```", "",
                  "Run it: `python examples/" + rel + "` (in the `conatus` environment).", "",
                  "```{literalinclude} ../../../../examples/" + rel, ":language: python",
                  ":linenos:", "```", ""]
        _write(out / f"{slug}.md", "\n".join(lines))
        table.append(f"| {{doc}}`{title.rstrip('.')} <_generated/{slug}>` | {row[2] if row else '—'} | "
                     f"{row[3] if row else '—'} |")
    _write(out / "table.md", "\n".join(table) + "\n")


def _user_page_objects(page: Path) -> set[int]:
    """ids of the objects the user-layer page (api/user.md) documents with autosummary.

    The expert layer re-exports some of them (the contour result classes are ``cn.contour.*`` and
    ``conatus.expert.*``): documented twice, Sphinx reports a duplicate object description (strict
    build). Their expert pages are written without an index entry; the user page stays canonical."""
    ids, module, in_table = set(), "conatus", False
    for line in page.read_text().splitlines() if page.exists() else []:
        s = line.strip()
        if s.startswith(".. currentmodule::"):
            module = s.split("::", 1)[1].strip()
        elif s.startswith(".. autosummary::"):
            in_table = True
        elif s.startswith("```"):
            in_table = False
        elif in_table and s and not s.startswith(":") and re.fullmatch(r"[\w.]+", s):
            obj = importlib.import_module(module)
            try:
                for part in s.split("."):
                    obj = getattr(obj, part)
            except AttributeError:
                continue                        # a name the page lists but this build lacks
            ids.add(id(obj))
    return ids


def api_tables(site: Path, root: Path) -> None:
    api = importlib.import_module("conatus.expert")
    on_user_page = _user_page_objects(site / "api" / "user.md")
    seen = set()
    lines = ["# Expert layer by module", "",
             "Every public object re-exported by `conatus.expert`, grouped by the module that defines",
             "it and documented from its docstring.", ""]
    for mod in ("verify",) + tuple(api._MODULES):
        m = importlib.import_module(f"conatus.{mod}")
        names = [n for n in getattr(m, "__all__", []) if n in api.__all__ and n not in seen
                 and getattr(api, n, None) is getattr(m, n, None)
                 and not isinstance(getattr(api, n), types.ModuleType)]
        if not names:
            continue
        seen.update(names)
        summary = ((m.__doc__ or "").strip().split("\n\n")[0]).replace("\n", " ")
        own = [n for n in names if id(getattr(api, n)) not in on_user_page]
        shared = [n for n in names if n not in own]
        lines += [f"## `conatus.{mod}`", "", summary, "", "```{eval-rst}",
                  ".. currentmodule:: conatus.expert", ""]
        if own:
            lines += [".. autosummary::", "   :toctree: _stubs", "", *[f"   {n}" for n in own], ""]
        if shared:                              # indexed on the user page (_user_page_objects)
            lines += [".. autosummary::", "   :toctree: _stubs", "   :template: autosummary/noindex.rst",
                      "", *[f"   {n}" for n in shared], ""]
        lines += ["```", ""]
    _write(site / "api" / "_generated" / "reference.md", "\n".join(lines))


def tutorial_status(site: Path, root: Path) -> None:
    """Table of the last recorded execution of each tutorial script (outputs/STATUS.json)."""
    import json
    status_file = site / "tutorials" / "outputs" / "STATUS.json"
    status = json.loads(status_file.read_text()) if status_file.exists() else {}
    lines = ["| Script | Last execution | Time | Run |", "|---|---|---|---|"]
    for script in sorted((site / "tutorials" / "code").glob("t[0-9][0-9]*.py")):
        st = status.get(script.stem)
        if st is None:
            lines.append(f"| `{script.name}` | {{.cn-skip}}`not run yet` | — | — |")
            continue
        cls = {"PASS": "cn-pass", "FAIL": "cn-fail"}.get(st["status"], "cn-skip")
        run = f"`{st['run_id']}`" if st.get("run_id") else "—"
        if st.get("commit"):
            run += f" ({st['commit']})"
        time = f"{st['elapsed_s']} s" if st.get("elapsed_s") is not None else "—"
        lines.append(f"| `{script.name}` | {{.{cls}}}`{st['status']}` | {time} | {run} |")
    _write(site / "tutorials" / "_generated" / "status.md", "\n".join(lines) + "\n")


def error_catalogue(site: Path, root: Path) -> None:
    """One section per exception class of conatus.errors; anchors match the ``doc`` links that
    the error messages print (``docs/site/guide/errors.md#<classname lowercased>``)."""
    import inspect
    out = site / "guide" / "_generated" / "errors_list.md"
    try:
        errors = importlib.import_module("conatus.errors")
    except Exception as e:  # noqa: BLE001 -- any import failure is reported on the page, not raised
        _write(out, f"*The error catalogue could not be generated ({type(e).__name__}: {e}).*\n")
        return
    lines, seen = [], set()
    for name in errors.__all__:
        cls = getattr(errors, name)
        if not (inspect.isclass(cls) and issubclass(cls, BaseException)) or cls in seen:
            continue
        seen.add(cls)
        aliases = [n for n in errors.__all__ if n != name and getattr(errors, n) is cls]
        bases = ", ".join(f"`{b.__name__}`" for b in cls.__bases__)
        doc = inspect.cleandoc(cls.__doc__ or "")
        lines += [f"### {name}", "", f"*Derives from* {bases}"
                  + (f" · *alias:* {', '.join(f'`{a}`' for a in aliases)}" if aliases else ""), "",
                  "```text", doc or "(no description)", "```", ""]
    _write(out, "\n".join(lines))


def studio_constants(site: Path, root: Path) -> None:
    """The table of every Studio/hub/relay/telemetry tunable (conatus.studio.constants)."""
    out = site / "guide" / "_generated" / "studio_constants.md"
    try:
        c = importlib.import_module("conatus.studio.constants")
    except Exception as e:  # noqa: BLE001 -- any import failure is reported on the page, not raised
        _write(out, f"*The table could not be generated ({type(e).__name__}: {e}).*\n")
        return
    lines = ["| Constant | Value | Kind | What it decides | How to change it |", "|---|---|---|---|---|"]
    for t in c.TUNABLES:
        val = f"{t.value!r}" + (f" {t.unit}" if t.unit else "")
        esc = t.meaning.replace("*", "\\*").replace("|", "\\|")      # Markdown table cell
        lines.append(f"| `{t.where.removeprefix('conatus.')}` | {val} | {t.kind} | {esc} | {t.change} |")
    _write(out, "\n".join(lines) + "\n")


def release_status(site: Path, root: Path) -> None:
    """The extras table (pyproject.toml, with the purposes of web/src/content/status.json) and the
    feature status table, included by the install and GPU pages. One source for the site and the
    documentation; an extra without a stated purpose stops the build."""
    import json
    import tomllib
    status = json.loads((root / "web" / "src" / "content" / "status.json").read_text())
    extras = tomllib.loads((root / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    missing = sorted(set(extras) - set(status["extras"]))
    if missing:
        raise ValueError(f"web/src/content/status.json: no purpose for the extras {missing}")
    rows = ["| Extra | Adds | For |", "|---|---|---|"]
    for name, reqs in extras.items():
        adds = ", ".join(r.replace(">=", " ≥ ").replace("conatus[", "").rstrip("]").replace(",", ", ")
                         for r in reqs)
        rows.append(f"| `{name}` | {adds} | {status['extras'][name]} |")
    _write(site / "guide" / "_generated" / "status_extras.md", "\n".join(rows) + "\n")
    rows = ["| Part | Status | Note |", "|---|---|---|"]
    for f in status["features"]:
        rows.append(f"| {f['feature']} | {f['status']} | {f['note']} |")
    _write(site / "guide" / "_generated" / "status_features.md", "\n".join(rows) + "\n")


def all(site: Path, root: Path) -> None:  # noqa: A001 (module-level entry point)
    release_status(site, root)
    error_catalogue(site, root)
    studio_constants(site, root)
    tutorial_status(site, root)
    gallery(site, root)
    api_tables(site, root)
