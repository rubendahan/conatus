"""Sphinx configuration of the Conatus documentation site (MyST Markdown sources).

Built by ``docs/site/build.sh`` in the separate ``conatus-docs`` environment (never the ``conatus``
compute environment, E5); Conatus itself is imported from the repository (``python/`` on the path).
Pages generated at build time (gallery, method notes, API tables) go to ``_generated`` folders that
are not versioned. Tutorials are *not* executed here: their verified outputs come from
``run_tutorials.py`` (a CPU/GPU job) and are included verbatim.
"""
from __future__ import annotations

import datetime
import hashlib
import os
import re
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parent
ROOT = SITE.parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(SITE))
sys.path.insert(0, str(SITE / "_ext"))

import generate  # noqa: E402  (site-local helper: gallery, method notes, API tables)

generate.all(SITE, ROOT)

# Third-party files served by the docs themselves: MathJax, its font and speech maps, and
# the brand fonts, from the npm tarballs pinned in vendor/lock.json (sha512 verified before use;
# vendor/fetch.py). One directory per lock content, so builds of different branches never mix.
sys.path.insert(0, str(SITE / "vendor"))
import fetch as _vendor  # noqa: E402  (site-local helper)

_LOCK_ID = hashlib.sha256((SITE / "vendor" / "lock.json").read_bytes()).hexdigest()[:12]
VENDOR = Path(os.environ.get("CN_DOCS_VENDOR_DIR", ROOT / "scratch" / "docs_vendor" / _LOCK_ID))
_vendor.build(VENDOR, Path(os.environ.get("CN_DOCS_VENDOR_CACHE",
                                          ROOT / "scratch" / "cache" / "docs_vendor")))

# -- Project -----------------------------------------------------------------------------------
project = "Conatus"
author = "Ruben Dahan and contributors"
_YEAR = datetime.date.today().year
copyright = f"2026{'' if _YEAR == 2026 else f'–{_YEAR}'}, {author}"   # not "2026–2026"
# The version of the package documented (single source: python/conatus/__init__.py), read as text so
# that the docs never import a stale installed copy; shown by the sidebar status line.
release = version = re.search(r'^__version__ = "([^"]+)"', (ROOT / "python" / "conatus" / "__init__.py")
                              .read_text(), re.M).group(1)

# -- Extensions --------------------------------------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
    "geometrico",          # _ext/: goal / result blocks, docstring |x| escaping
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "tutorials/code", "tutorials/outputs", "README.md",
                    "gallery/_generated/table.md", "tutorials/_generated",
                    "guide/_generated",
                    "tutorials/media"]   # tables written by tutorial_media_render.py, {include}d by the pages
templates_path = ["_templates"]

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist", "attrs_inline",
                          "substitution"]
myst_heading_anchors = 3
myst_dmath_double_inline = True
suppress_warnings = ["myst.header"]   # pages that start below heading level 1

# API reference: every public object of conatus.expert, one page each (autosummary).
autosummary_generate = True
autosummary_imported_members = True
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": False}
autodoc_typehints = "description"
autodoc_member_order = "bysource"
numpydoc_show_class_members = False
numpydoc_validation_checks = set()     # short docstrings by project convention
numpydoc_xref_param_type = False

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# MathJax: self-hosted (after the critique-pass-3 fix S8 that pinned the CDN copy with an
# SRI hash). The CDN copy then fetched its font chunks and speech-rule maps from jsdelivr at run time
# without integrity; now every file comes from _static/vendor (exact version and sha512 in
# vendor/lock.json), so the docs load nothing from another origin (checked on the built HTML by the
# development tooling).
# Performance (Lighthouse 41-72 on the getting-started tutorial; MathJax typesetting a long page in one
# main-thread task): mathjax_config.js turns the speech-rule enrichment off by default (the menu turns it
# on), keeps the assistive MathML, and typesets one formula per task in idle time. The font loader
# points at the vendored package. Lazy typesetting (ui/lazy) was measured too and not kept: it left
# off-screen formulas empty (find-in-page, printing, full-page captures) and raised the layout shift
# (CLS 0.24 against 0.06).
mathjax_path = "vendor/mathjax/tex-chtml.js"          # relative: _static/vendor/mathjax/
mathjax_config_path = "mathjax_config.js"

# -- HTML --------------------------------------------------------------------------------------
# Brand charter (docs/brand/guidelines.html): the docs are world C, "Ethica":
# ink on rag paper, one rubric red; dark mode is world B, lab black and silver (rubric lightened
# to #D4574A for titles and links; fluorescein is kept for data, never for text). Variables come
# from docs/brand/tokens.css (the single source); the fonts (IM Fell English for titles >= 18 px,
# IBM Plex Sans, JetBrains Mono) are self-hosted: _static/vendor/fonts/fonts.css (they
# were loaded from Google Fonts on every page view); site-specific rules in _static/conatus.css.
# The marks are copies of docs/brand/mark-ink.svg / mark-silver.svg / favicon.svg.
html_theme = "furo"
html_title = "Conatus"
html_static_path = ["_static", str(ROOT / "docs" / "brand" / "tokens.css"), str(VENDOR)]
html_css_files = ["vendor/fonts/fonts.css", "tokens.css", "conatus.css"]
# Tutorial films (tutorials/*.md, made by tutorial_media.py): play only while visible, never under
# prefers-reduced-motion (the poster stays), as the films of the website.
html_js_files = [("tutorial_films.js", {"defer": "defer"})]
# Sidebar: Furo's, plus the way back to the website and the status line.
html_sidebars = {"**": ["sidebar/brand.html", "sidebar/site.html", "sidebar/search.html",
                        "sidebar/scroll-start.html", "sidebar/navigation.html",
                        "sidebar/scroll-end.html"]}
html_favicon = "_static/favicon.svg"
html_show_sourcelink = False
_FONTS = {
    "font-stack": "'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif",
    "font-stack--monospace": "'JetBrains Mono', 'SFMono-Regular', Menlo, Consolas, monospace",
    "font-stack--headings": "'IM Fell English', 'IM Fell DW Pica', Georgia, serif",
}
html_theme_options = {
    "light_logo": "logo.svg",                   # docs/brand/mark-ink.svg
    "dark_logo": "logo-dark.svg",               # docs/brand/mark-silver.svg
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-brand-primary": "#A8281E",       # rubric: numbers, links, active entries
        "color-brand-content": "#A8281E",
        "color-background-primary": "#EFEAE0",  # rag paper
        "color-background-secondary": "#E6E0D3",
        "color-background-border": "#CFC7B7",
        "color-foreground-primary": "#151310",  # ink
        "color-foreground-secondary": "#5E5A52",
        "color-code-background": "#E6E0D3",
        **_FONTS,
    },
    "dark_css_variables": {
        "color-brand-primary": "#D4574A",       # rubric on black (titles, links)
        "color-brand-content": "#D4574A",
        "color-background-primary": "#0B0B0B",  # lab black
        "color-background-secondary": "#161615",
        "color-background-border": "#2A2A28",
        "color-foreground-primary": "#EDEBE6",  # silver
        "color-foreground-secondary": "#A3A29D",
        "color-code-background": "#161615",
        **_FONTS,
    },
}

# -- One navigation with the website (lot F) ---------------------------------------------------
# web/src/content/nav.json is the single map: the site's header and Home page read it, and so do the
# sidebar of these pages (_templates/sidebar/site.html) and the "Start here" list of the docs home page
# (tutorials/_generated/start.md, included by index.md). A route is a page of the website, one level
# above the docs root; the portable builds rewrite links marked data-cn-route to hash routes.
import html as _html  # noqa: E402
import json as _json  # noqa: E402

_NAV = _json.loads((ROOT / "web" / "src" / "content" / "nav.json").read_text())
html_context = {"cn_nav": _NAV["primary"]}


def _start_list() -> str:
    """The getting-started route as an HTML list, for the docs home page (a page at the docs root)."""
    items = []
    for step in _NAV["start"]:
        if "route" in step:
            href = "../" + step["route"].strip("/") + "/"
            attr = f' data-cn-route="{_html.escape(step["route"])}"'
        else:
            href, attr = step["docs"], ""
        items.append(f'<li><a href="{_html.escape(href)}"{attr}><strong>{_html.escape(step["title"])}</strong></a>'
                     f'<br>{_html.escape(step["text"])}</li>')
    return '<ol class="cn-start">\n' + "\n".join(items) + "\n</ol>\n"


(SITE / "tutorials" / "_generated").mkdir(exist_ok=True)
_start = SITE / "tutorials" / "_generated" / "start.md"
if not _start.exists() or _start.read_text() != _start_list():
    _start.write_text(_start_list())


# Plot figures (tutorials/media, tutorials/outputs, guide/media PNGs) are drawn on a transparent
# background with mid-tone ink. In the dark theme they need a paper panel, and the built pages serve
# them from _images/, so the source path cannot be matched in CSS: the class is set here from the
# source URI (conatus.css, "cn-paper").
_PAPER_DIRS = ("tutorials/media/", "tutorials/outputs/", "guide/media/")


def _paper_class(app, doctree) -> None:
    """Runs before or after Sphinx's image collector, so the URI may be page- or root-relative."""
    import posixpath as _posix

    from docutils import nodes
    doc = app.env.docname.rsplit("/", 1)[0] + "/" if "/" in app.env.docname else ""
    for node in doctree.findall(nodes.image):
        uri = node.get("uri", "").replace("\\", "/")
        full = uri if uri.startswith(("tutorials/", "guide/")) else _posix.normpath(doc + uri)
        if full.endswith(".png") and any(full.startswith(d) for d in _PAPER_DIRS):
            node["classes"].append("cn-paper")


def _no_viewcode_for_unindexed(app, doctree) -> None:
    """Objects documented twice (a ``conatus.contour`` class re-exported by ``conatus.expert``) are
    indexed on the user page only; their expert stub is ``:no-index:`` and has no anchor. viewcode
    would still record the stub and link the source page to ``#conatus.expert.<name>``, a fragment
    that does not exist. Hiding the stub from viewcode (its domain test) leaves the canonical link."""
    from sphinx import addnodes
    for node in doctree.findall(addnodes.desc):
        if node.get("domain") == "py" and (node.get("no-index") or node.get("noindex")):
            node["domain"] = "py-unindexed"


# Exported constants of conatus.expert: autodoc would show the docstring of their type (tuple, int,
# dict, sympy.Symbol), which says nothing about Conatus. Their purpose, written here, replaces it.
DATA_DOCS = {
    "SCHEMES": "Names of every time-stepping scheme the IVP solvers accept: the multistep SBDF1–4, "
               "the Crank–Nicolson families CNAB1, CNAB2, MCNAB2, CNLF2, and the IMEX Runge–Kutta "
               "RK111, RK222, RK443, RKSMR, RKGFY. Pass one as ``scheme=``; the orders, start-up "
               "rules and stability notes are in the time-stepper guide.",
    "SBDF_SCHEMES": "The semi-implicit backward-differentiation schemes, SBDF1 to SBDF4 (order 1 to 4). "
                    "SBDF3 and SBDF4 need a declared start-up (``STARTUPS``).",
    "CN_SCHEMES": "The Crank–Nicolson multistep schemes: CNAB1, CNAB2, MCNAB2, CNLF2.",
    "RK_SCHEMES": "The IMEX Runge–Kutta schemes: RK111, RK222, RK443, RKSMR, RKGFY.",
    "STARTUPS": "How a multistep scheme takes its first steps: ``lower_order`` (lower-order steps of "
                "the same family) or ``rk443`` (RK443 steps).",
    "Y": "The wall-normal coordinate y of the modal tables of ``conatus.nonmodal`` (a real SymPy "
         "symbol). Write base flows and coefficients as expressions in it, for example "
         "``1 - Y**2`` for plane Poiseuille flow.",
    "CHANNEL_SYSTEMS": "The three Reynolds-number conventions of channel flow, by name: "
                       "``centreline`` (Re_c = U_c h / ν), ``bulk`` (Re_b = U_b 2h / ν) and "
                       "``friction`` (Re_τ = u_τ h / ν). Each is a ``UnitSystem`` that converts "
                       "between them.",
    "DMD_METHODS": "The DMD variants of ``dmd(method=...)``: ``exact`` and ``optimized``.",
    "AMPLITUDES": "How ``dmd`` computes the mode amplitudes: ``first`` (fit to the first snapshot) or "
                  "``optimal`` (least squares over all snapshots).",
    "POD_METHODS": "How ``pod`` computes the modes: ``snapshots`` (method of snapshots) or ``svd``.",
    "MEAN_POLICIES": "What ``pod`` and ``spod`` do with the temporal mean: ``subtract`` or ``keep``.",
    "ADJOINT_KINDS": "The adjoint used by the sensitivity tools: ``discrete`` (of the discretised "
                     "operator) or ``continuous`` (discretised from the continuous adjoint).",
    "ORDERINGS": "Layouts of the unknowns in the global system of a time stepper: ``interleaved`` or "
                 "``variable-major``.",
    "SCALES": "Colour scales of a Studio view: ``smoke`` (one-signed), ``signed`` (zero in black) "
              "and ``dye``.",
    "VIEW_KINDS": "Kinds of view a watched run can declare for the Studio.",
    "STAT_KINDS": "Kinds of statistic a watched run can declare: ``series``, ``spectrum``, ``profile``.",
    "FRAME_KINDS": "Kinds of frame record in the telemetry stream.",
    "KINDS": "Kinds of record in the telemetry stream (header, scalars, event, log, status, newton).",
    "PROTOCOL_VERSION": "Version of the telemetry protocol written in every record.",
    "FILM_ENCODING": "Encoding of the 8-bit film frames in the telemetry stream.",
    "FILM_POLICY": "Which frames of a long run the telemetry film keeps.",
    "COMMANDS": "The commands of ``python -m conatus agent`` and what each one does.",
}
_TYPE_BOILERPLATE = ("Built-in immutable sequence", "Built-in mutable sequence", "int([x]) -> integer",
                     "Symbol class is used", "dict() -> new empty dictionary", "str(object=",
                     "Convert a string or number to a floating point")


def _no_type_boilerplate(app, what, name, obj, options, lines) -> None:
    """Replace the docstring of a built-in or SymPy type, inherited by a constant or a property,
    by the constant's purpose (DATA_DOCS), or by nothing."""
    text = "\n".join(lines).lstrip()
    if text.startswith(_TYPE_BOILERPLATE):
        lines[:] = [DATA_DOCS[name.rsplit(".", 1)[-1]]] if name.rsplit(".", 1)[-1] in DATA_DOCS else []


def setup(app):
    app.connect("autodoc-process-docstring", _no_type_boilerplate)
    app.connect("doctree-read", _paper_class)
    app.connect("doctree-read", _no_viewcode_for_unindexed, priority=400)   # before viewcode (500)
