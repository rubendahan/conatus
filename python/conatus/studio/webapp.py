"""What the local server and the hub share to serve the Studio app.

* the static bundle of ``studio/app`` (built into ``scratch/studio_app/dist``, never in the home):
  :func:`app_dir`, :func:`static_file`, :func:`is_app_route`;
* the film API on top of :class:`conatus.telemetry.FilmIndex`: :func:`film_answer`;
* the light WebSocket stream for phones (``/ws?lite=1``): :func:`lite`.

Stdlib only (tornado stays in the handlers), so it is unit-testable without a server.
"""
from __future__ import annotations

import mimetypes
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from conatus.telemetry.film import MAX_FRAMES_PER_REQUEST, FilmIndex

REPO = Path(__file__).resolve().parents[3]
LITE_MAX_SCALARS = 2000              # scalars per run in a light snapshot (uniform stride, last kept)
HEAVY_KINDS = ("frame3d", "points3d", "vec2d")      # data removed from the light stream
# SPA routes answered with index.html (the app routes itself); everything else is a file or 404
_APP_ROUTE = re.compile(r"^/(?:run/[^/]+|compare|runs|board)?/?$")
# Content-Security-Policy of the app (no inline script at all) and of the classic page (inline).
CSP_APP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
           "img-src 'self' data: blob:; font-src 'self'; connect-src 'self' ws: wss:; "
           "worker-src 'self'; manifest-src 'self'; frame-ancestors 'none'; form-action 'self'; "
           "base-uri 'none'")
CSP_CLASSIC = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' "
               "'unsafe-inline'; img-src 'self' data:; connect-src 'self' ws: wss:; "
               "frame-ancestors 'none'; form-action 'self'; base-uri 'none'")
_TYPES = {".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8", ".html": "text/html; charset=utf-8",
          ".svg": "image/svg+xml", ".webmanifest": "application/manifest+json",
          ".woff2": "font/woff2", ".woff": "font/woff", ".png": "image/png", ".json": "application/json",
          ".webp": "image/webp", ".ico": "image/x-icon", ".map": "application/json"}


# The web files shipped inside the package (an installed release has no repository around it):
# ``static/app`` (the built bundle of studio/app) and ``static/classic`` (studio/web), copied there
# by the public export. A development checkout has neither and uses the repository's files.
STATIC = Path(__file__).resolve().parent / "static"


def app_dir(given: str | Path | None = None) -> Path:
    """The built app: ``given``, else ``$CN_STUDIO_APP``, else ``<repo>/scratch/studio_app/dist``
    when it is built, else the bundle shipped in the package (``static/app``)."""
    if given:
        return Path(given)
    env = os.environ.get("CN_STUDIO_APP")
    if env:
        return Path(env)
    dev = REPO / "scratch" / "studio_app" / "dist"
    shipped = STATIC / "app"
    return shipped if not (dev / "index.html").is_file() and (shipped / "index.html").is_file() else dev


def classic_page() -> Path:
    """The classic page: ``<repo>/studio/web/index.html`` in a checkout, else the shipped copy."""
    dev = REPO / "studio" / "web" / "index.html"
    shipped = STATIC / "classic" / "index.html"
    return shipped if not dev.is_file() and shipped.is_file() else dev


def is_app_route(path: str) -> bool:
    """``/``, ``/run/<key>``, ``/compare``: pages of the single-page app (answered by index.html)."""
    return bool(_APP_ROUTE.match(path))


def static_file(dist: Path, path: str) -> tuple[bytes, str, bool] | None:
    """``(body, content type, immutable)`` of a file of the bundle, or None (missing, or a path
    that would leave the bundle: ``..``, absolute, hidden). Hashed assets are immutable."""
    rel = path.lstrip("/")
    if (not rel or "\x00" in rel or "\\" in rel
            or any(part in ("", "..") or part.startswith(".") for part in rel.split("/"))):
        return None                     # NUL made resolve() raise: a 500 with a traceback
    root = dist.resolve()
    try:
        f = (root / rel).resolve()
        if root not in f.parents or not f.is_file():
            return None
    except (OSError, ValueError):
        return None
    ctype = _TYPES.get(f.suffix) or mimetypes.guess_type(f.name)[0] or "application/octet-stream"
    return f.read_bytes(), ctype, rel.startswith("assets/")


LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")


def loopback_host(host: str) -> bool:
    """``Host`` header names this machine's loopback (any port). The local server answers only
    those: a page of another site that rebinds its DNS name to 127.0.0.1 (DNS rebinding) reaches
    the ssh tunnel with ITS name in ``Host``, and is refused (review)."""
    h = host.strip().lower()
    if h.startswith("["):                                   # [::1]:8765
        h = h[1:h.find("]")] if "]" in h else h
    elif h.count(":") == 1:
        h = h.split(":", 1)[0]
    return h in LOOPBACK_HOSTS


def same_origin(headers: Mapping[str, str], host: str) -> bool:
    """A state-changing request comes from this site's own pages (rule, shared by the hub
    and the local server): ``Sec-Fetch-Site`` decides when the browser sends it (``same-origin``
    or ``none``), else an ``Origin`` header, when present, must be this host."""
    site = headers.get("Sec-Fetch-Site")
    if site is not None:
        return site in ("same-origin", "none")
    origin = headers.get("Origin")
    return origin is None or origin.split("://", 1)[-1].rstrip("/") == host


def missing_app_page(dist: Path) -> bytes:
    """Plain page shown when the bundle is not built (says how to build it; nothing hidden)."""
    return (f"<!doctype html><meta charset=utf-8><title>Conatus Studio</title>"
            f"<body style='font:16px/1.5 system-ui;margin:2rem;max-width:40rem'>"
            f"<h1 style='font-weight:400'>Conatus Studio</h1><p>The Studio app is not built "
            f"(no <code>{dist}/index.html</code>). Build it with <code>bash studio/app/build.sh"
            f"</code>, or open the <a href='/classic'>classic page</a>.</p>").encode()


def film_answer(film: FilmIndex | None, run: str, args: Mapping[str, str],
                frames: bool) -> tuple[int, dict[str, Any]]:
    """The film API for one run: summary (no ``name``), index of a name, or frames.

    ``GET /api/film/<run>`` -> summary; ``?name=N`` -> ``{run, name, kind, index, level}``;
    ``/api/film/<run>/frames?name=N&from=&to=&step=`` -> ``{frames, next}`` (at most
    MAX_FRAMES_PER_REQUEST frames; ``to`` exclusive). Returns (HTTP status, JSON body)."""
    if film is None:
        return 404, {"error": f"run {run!r} has no film (declare film_max_bytes, or watch())"}
    name = args.get("name")
    if not frames:
        if name is None:
            return 200, {**film.summary(), "run": run}
        es = film.names.get(name)
        if es is None:
            return 404, {"error": f"no film named {name!r} for run {run!r}",
                         "names": sorted(film.names)}
        return 200, {"run": run, "name": name, "kind": es[-1].kind if es else None,
                     "index": film.index(name), "level": film.level}
    if name is None:
        return 400, {"error": "frames need ?name="}
    try:
        start = int(args.get("from", 0))
        stop = int(args["to"]) if args.get("to") not in (None, "") else None
        step = int(args.get("step", 1))
        if step < 1 or start < 0:
            raise ValueError
    except ValueError:
        return 400, {"error": "from, to, step must be integers (from >= 0, step >= 1)"}
    out, nxt = film.frames(name, start, stop, step, MAX_FRAMES_PER_REQUEST)
    return 200, {"run": run, "name": name, "frames": out, "next": nxt}


def lite(msg: dict) -> dict:
    """The light version of a WebSocket message for phones: 3D/particle/vector frames without
    their data (``"lite": true``), scalars of a snapshot strided to at most LITE_MAX_SCALARS per
    run (last kept). Declared to the page by the ``lite`` flags; the full data stays on demand."""
    kind = msg.get("type")
    if kind == "items":
        return {**msg, "items": [_lite_frame(x) for x in msg.get("items", [])]}
    if kind == "snapshot":
        runs = {}
        for k, st in msg.get("runs", {}).items():
            sc = list(st.get("scalars", []))
            if len(sc) > LITE_MAX_SCALARS:
                step = -(-len(sc) // LITE_MAX_SCALARS)
                sel = sc[::step]
                sc = sel if sel[-1] is sc[-1] else sel + [sc[-1]]
            runs[k] = {**st, "scalars": sc,
                       "frames": {n: _lite_frame(f) for n, f in st.get("frames", {}).items()}}
        return {**msg, "runs": runs, "lite": True}
    return msg


def _lite_frame(x: Any) -> Any:
    if isinstance(x, dict) and x.get("kind") in HEAVY_KINDS and "data" in x:
        return {k: v for k, v in x.items() if k not in ("data", "values")} | {"lite": True}
    return x
