"""Collect the gallery images of finished runs into the website (login node: file copies only).

    source env/env.sh && python web/gallery/collect.py

Reads web/gallery/runs.txt (one "slug run_id" per line, in display order), takes
scratch/output/<run_id>/<slug>.webp, <slug>-800.webp and <slug>.meta.json (written by the case
script through brand_cmaps.save), copies the images to web/public/gallery/ and writes
web/public/gallery/manifest.json. The manifest is generated, never edited by hand.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "scratch" / "output"
PUBLIC = ROOT / "web" / "public" / "gallery"
FIELDS = ("slug", "title", "image", "image_small", "width", "height", "alt", "caption", "colormap",
          "run_id", "script", "example_script", "validated", "geometry", "domain")
GEOMETRIES = {"cartesian", "polar", "cylinder", "shell", "sphere", "ball"}
DOMAINS = {"shear", "convection", "patterns", "pipe", "rotating", "turbulence", "dynamics"}


def entry(slug: str, run_id: str) -> dict:
    run = OUTPUT / run_id
    status = json.loads((run / "STATUS.json").read_text())
    if status.get("exit_code") != 0:
        raise SystemExit(f"{run_id}: exit code {status.get('exit_code')} (not collected)")
    meta = json.loads((run / f"{slug}.meta.json").read_text())
    if meta.get("run_id") != run_id:
        raise SystemExit(f"{slug}: meta run_id {meta.get('run_id')} != {run_id}")
    for suffix in ("", "-800"):
        src = run / f"{slug}{suffix}.webp"
        if src.stat().st_size >= 250_000:
            raise SystemExit(f"{src}: {src.stat().st_size} B >= 250 kB")
        shutil.copyfile(src, PUBLIC / src.name)
    e = dict(meta, image=f"/gallery/{slug}.webp", image_small=f"/gallery/{slug}-800.webp")
    missing = [k for k in FIELDS if k not in e]
    if missing:
        raise SystemExit(f"{slug}: missing manifest fields {missing}")
    if e["colormap"] not in ("smoke", "signed") or e["geometry"] not in GEOMETRIES \
            or e["domain"] not in DOMAINS:
        raise SystemExit(f"{slug}: colormap/geometry/domain outside the declared vocabularies")
    v = e["validated"]
    if not all(k in v for k in ("value", "reference", "source", "run_id")):
        raise SystemExit(f"{slug}: validated needs value, reference, source, run_id")
    return {k: e[k] for k in FIELDS}


def main() -> None:
    PUBLIC.mkdir(parents=True, exist_ok=True)
    lines = [ln.split() for ln in (ROOT / "web/gallery/runs.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    manifest = [entry(slug, run_id) for slug, run_id in lines]
    (PUBLIC / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    for e in manifest:
        print(f"{e['slug']:32s} {e['width']}x{e['height']}  {e['run_id']}")
    print(f"{len(manifest)} images -> {PUBLIC / 'manifest.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
