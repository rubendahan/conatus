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
release = version = "0.1.0"

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
                    "guide/_generated"]
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
numpydoc_validation_checks = set()     # short docstrings by project convention (CLAUDE.md)
numpydoc_xref_param_type = False

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# MathJax: self-hosted (after the critique-pass-3 fix S8 that pinned the CDN copy with an
# SRI hash). The CDN copy then fetched its font chunks and speech-rule maps from jsdelivr at run time
# without integrity; now every file comes from _static/vendor (exact version and sha512 in
# vendor/lock.json), so the docs load nothing from another origin (tests/infra/
# test_third_party_scripts.py; docs/site/check_origins.py scans the built HTML).
# Performance (Lighthouse 41-72 on the getting-started tutorial; MathJax typesetting a long page in one
# main-thread task): mathjax_config.js turns the speech-rule enrichment off by default (the menu turns it
# on), keeps the assistive MathML, and typesets one formula per task in idle time. The font loader
# points at the vendored package. Lazy typesetting (ui/lazy) was measured too and not kept: it left
# off-screen formulas empty (find-in-page, printing, full-page captures) and raised the layout shift
# (CLS 0.24 against 0.06; docs/DECISIONS.md).
mathjax_path = "vendor/mathjax/tex-chtml.js"          # relative: _static/vendor/mathjax/
mathjax_config_path = "mathjax_config.js"

# -- HTML --------------------------------------------------------------------------------------
# Brand charter (docs/DECISIONS.md, docs/brand/guidelines.html): the docs are world C, "Ethica":
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
