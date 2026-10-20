"""The Studio page, served identically by the local server and the hub.

``studio/web/index.html`` loads the 3D view with ``<script src="view3d.js"></script>``; the page is
sent as **one** document, the script inlined in place of that tag (no extra route, same Content
Security Policy as the inline main script, works through the relay-less hub too). Stdlib only.
"""
from __future__ import annotations

from pathlib import Path

TAG = '<script src="view3d.js"></script>'


def render_page(index: str | Path) -> bytes:
    """``index.html`` with each ``<script src="X.js"></script>`` of :data:`TAG` replaced by the
    content of the sibling file inside ``<script>``. A missing script is reported in the page
    (console warning), never silently dropped."""
    index = Path(index)
    html = index.read_text(encoding="utf-8")
    if TAG in html:
        js = index.with_name("view3d.js")
        try:
            body = js.read_text(encoding="utf-8").replace("</script", "<\\/script")
        except OSError as err:
            body = f"console.warn({str(f'view3d.js unavailable: {err}')!r});"
        html = html.replace(TAG, "<script>\n" + body + "\n</script>")
    return html.encode("utf-8")
