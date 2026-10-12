#!/usr/bin/env python3
"""Fetch, verify and lay out the documentation's third-party files (MathJax and the brand fonts).

The docs load nothing from another origin (MathJax from a CDN with its
font chunks and speech maps unchecked, the brand fonts from Google Fonts on every page view). The
files are the npm tarballs pinned in ``lock.json`` (version + registry ``sha512`` integrity),
downloaded once into a cache, verified, and only the listed paths are copied into
``<out>/vendor/`` (served as ``_static/vendor/`` by Sphinx, conf.py ``html_static_path``):

    vendor/mathjax/tex-chtml.js, input/, output/, ui/, a11y/, adaptors/, sre/   (MathJax 4.1.3)
    vendor/@mathjax/mathjax-newcm-font/chtml.js, chtml/                         (its font)
    vendor/fonts/<face>/*.woff2 + vendor/fonts/fonts.css                        (@font-face rules)
    vendor/<dest>/LICENSE for every package (its own, or ``licence_from`` when its tarball has none)

    python docs/site/vendor/fetch.py <out>      # <out> = the static directory to fill
    CN_DOCS_VENDOR_CACHE=<dir>                  # tarball cache (default scratch/cache/docs_vendor)

A tarball whose digest differs from the lock is refused (exit 1) and never extracted; a member
path outside the package (``..``, absolute, links) is refused too. Offline once the cache holds
the tarballs. The layout is rebuilt only when ``lock.json`` changes (stamp file).
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK = HERE / "lock.json"
ROOT = HERE.parents[2]

# @font-face rules of the brand (docs/brand/tokens.css names the families): weight, style, file.
FACES = {
    "IBM Plex Sans": ("ibm-plex-sans", [("400", "normal"), ("400", "italic"), ("500", "normal"),
                                         ("600", "normal")]),
    "JetBrains Mono": ("jetbrains-mono", [("400", "normal"), ("600", "normal")]),
    "IM Fell English": ("im-fell-english", [("400", "normal"), ("400", "italic")]),
    "IM Fell English SC": ("im-fell-english-sc", [("400", "normal")]),
}


def _digest_ok(data: bytes, integrity: str) -> bool:
    algo, _, want = integrity.partition("-")
    return base64.b64encode(hashlib.new(algo, data).digest()).decode() == want


def _tarball(name: str, spec: dict, cache: Path) -> bytes:
    """The verified tarball bytes of one package (from the cache, else downloaded)."""
    path = cache / (name.replace("/", "__") + f"-{spec['version']}.tgz")
    if path.is_file():
        data = path.read_bytes()
        if _digest_ok(data, spec["integrity"]):
            return data
        path.unlink()                                   # corrupted cache entry: fetch again
    with urllib.request.urlopen(spec["tarball"], timeout=60) as r:
        data = r.read()
    if not _digest_ok(data, spec["integrity"]):
        raise SystemExit(f"vendor: {name} {spec['version']}: integrity mismatch for "
                         f"{spec['tarball']} (refused, nothing extracted)")
    cache.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return data


def _extract(name: str, spec: dict, data: bytes, out: Path) -> int:
    import io
    dest = out / spec["dest"]
    n = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for m in tf.getmembers():
            rel = m.name.split("/", 1)[1] if "/" in m.name else ""
            if not rel or not any(rel == c or rel.startswith(c.rstrip("/") + "/") for c in spec["copy"]):
                continue
            if not m.isfile() or rel.startswith("/") or ".." in Path(rel).parts:
                if m.isdir():
                    continue
                raise SystemExit(f"vendor: {name}: refused member {m.name!r}")
            target = dest / (Path(rel).name if spec["dest"].startswith("fonts/") else rel)  # flat fonts
            target.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as fh:
                target.write_bytes(fh.read())
            n += 1
    flat = spec["dest"].startswith("fonts/")
    missing = [c for c in spec["copy"] if not (dest / (Path(c).name if flat else c)).exists()]
    if missing:
        raise SystemExit(f"vendor: {name}: paths not found in the tarball: {missing}")
    return n


def fonts_css() -> str:
    rules = ["/* Brand fonts of the docs, self-hosted (docs/site/vendor/fetch.py; @fontsource, OFL). */"]
    for family, (pkg, faces) in FACES.items():
        for weight, style in faces:
            rules.append(f"@font-face {{ font-family: '{family}'; font-style: {style}; "
                         f"font-weight: {weight}; font-display: swap; "
                         f"src: url('{pkg}/{pkg}-latin-{weight}-{style}.woff2') format('woff2'); }}")
    return "\n".join(rules) + "\n"


def build(out: Path, cache: Path) -> Path:
    lock_bytes = LOCK.read_bytes()
    stamp = out / "vendor" / ".lock.sha256"
    want = hashlib.sha256(lock_bytes).hexdigest()
    if stamp.is_file() and stamp.read_text().strip() == want:
        return out / "vendor"
    vendor = out / "vendor"
    if vendor.exists():
        shutil.rmtree(vendor)
    vendor.mkdir(parents=True)
    pkgs = json.loads(lock_bytes)["packages"]
    total = 0
    for name, spec in pkgs.items():
        total += _extract(name, spec, _tarball(name, spec, cache), vendor)
    for name, spec in pkgs.items():                     # licence text next to every package
        lic = vendor / spec["dest"] / "LICENSE"
        if "licence_from" in spec:
            src = vendor / spec["licence_from"]
            if not src.is_file():
                raise SystemExit(f"vendor: {name}: licence_from {spec['licence_from']!r} not vendored")
            lic.write_bytes(src.read_bytes())
            (vendor / spec["dest"] / "LICENSE.note.txt").write_text(spec.get("licence_note", "") + "\n")
        if not lic.is_file():
            raise SystemExit(f"vendor: {name}: no LICENSE vendored (add it to 'copy' or 'licence_from')")
    (vendor / "fonts" / "fonts.css").write_text(fonts_css())
    stamp.write_text(want + "\n")
    print(f"vendor: {len(pkgs)} packages verified, {total} files in {vendor}")
    return vendor


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    cache = Path(os.environ.get("CN_DOCS_VENDOR_CACHE", ROOT / "scratch" / "cache" / "docs_vendor"))
    build(Path(argv[0]), cache)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
