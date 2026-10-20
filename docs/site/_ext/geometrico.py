"""Sphinx extension of the Conatus site: goal/result blocks and docstring hygiene.

Directives (MyST: ``:::{propositio}`` ... ``:::``), kept under their historical names so that existing
pages build, rendered soberly (no Latin labels, no Q.E.D. device):

- ``propositio [ignored number]`` -> a block titled "Goal": what the page obtains;
- ``demonstratio``                -> a block titled "Result": the value obtained and its reference;
                                     the ``:qed:`` option text, if any, closes the block as a plain
                                     "Reference:" line.

New pages use the names ``goal`` and ``result`` (option ``:reference:``). Both are plain admonitions with a class (``cn-goal`` / ``cn-result``), styled in
``_static/conatus.css``.

Docstrings: the library's docstrings are short plain text where ``|x|`` means an absolute value;
reStructuredText would read it as a substitution. ``escape_bars`` escapes such bars when autodoc
reads a docstring (the source code is untouched).

Images: ``lazy_images`` marks every figure ``loading="lazy"`` (documentation performance).
"""
from __future__ import annotations

import re

from docutils.parsers.rst import directives
from docutils.parsers.rst.directives.admonitions import BaseAdmonition
from docutils import nodes


class _Titled(BaseAdmonition):
    node_class = nodes.admonition
    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    has_content = True
    option_spec = {"class": directives.class_option, "name": directives.unchanged}
    label = ""
    css = ""

    def run(self):
        self.arguments = [self.label]          # a numbering argument, if any, is not shown
        self.options["class"] = [self.css] + list(self.options.get("class", []))
        return super().run()


class Propositio(_Titled):
    label, css = "Goal", "cn-goal"


class Demonstratio(_Titled):
    label, css = "Result", "cn-result"
    option_spec = {**_Titled.option_spec, "qed": directives.unchanged, "reference": directives.unchanged}

    def run(self):
        ref = (self.options.pop("reference", "") or self.options.pop("qed", "")).strip()
        out = super().run()
        if ref:
            out[0] += nodes.paragraph(text=f"Reference: {ref}", classes=["cn-ref"])
        return out


_BAR = re.compile(r"(?<![\\|])\|(?=[^|\s`])([^|\n`]*?[^|\s\\])\|(?![|_])")


def escape_bars(app, what, name, obj, options, lines):
    """|x| -> \\|x\\| outside tables and literals (grid-table rows are left alone)."""
    in_literal = False
    for i, line in enumerate(lines):
        s = line.lstrip()
        if s.startswith((">>>", "..")) or s.endswith("::"):
            in_literal = s.endswith("::")
            continue
        if in_literal and (not s or line.startswith((" ", "\t"))):
            continue
        in_literal = False
        if s.startswith("+-") or (s.startswith("|") and s.endswith("|") and s.count("|") > 2):
            continue
        lines[i] = _BAR.sub(lambda m: "\\|" + m.group(1) + "\\|", line)


def lazy_images(app, doctree, docname):
    """Every figure loads lazily (``loading="lazy"``) unless the page sets ``:loading:`` itself:
    the browser fetches an image only when it nears the viewport (the tutorial pages
    carry several hundred kB of figures below the fold; Lighthouse counted them in the first load)."""
    for img in doctree.findall(nodes.image):
        img.attributes.setdefault("loading", "lazy")


def pin_version(app, docname, source):
    """``@VERSION@`` in a page (code blocks included) becomes the version documented (conf.py
    ``release``, read from the package): the install commands are pinned to the release and stay
    copyable (external review 0.1.0, finding 79: they held a literal ``<platform>``)."""
    source[0] = source[0].replace("@VERSION@", app.config.release)


def setup(app):
    app.connect("source-read", pin_version)
    app.connect("doctree-resolved", lazy_images)
    app.add_directive("propositio", Propositio)
    app.add_directive("demonstratio", Demonstratio)
    app.add_directive("goal", Propositio)          # preferred names for new pages
    app.add_directive("result", Demonstratio)      # (``:reference:`` = ``:qed:``)
    app.connect("autodoc-process-docstring", escape_bars)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
