"""Colormaps and image output of the brand chart (docs/brand/guidelines.html, section III "Data").

Two scales only, linear in sRGB between the stops, exactly as the CSS gradients of the chart:
  * "smoke"  (sequential: magnitude, temperature, energy): #0B0B0B -> #3A3A38 (30 %) -> #8A8A86 (62 %)
             -> #EDEBE6;
  * "signed" (vorticity, perturbation; zero is black): #D4574A (-1) -> #6E2019 (-0.4) -> #0B0B0B (0)
             -> #3F5245 (+0.4) -> #9DBBA6 (+1), symmetric in amplitude.
Images have a lab-black background (#0B0B0B), no axes, no text (the site writes the legend), an
optional faint silver-halide grain, and at most one fluorescein (#C6F03C) line.

Every image is written as WebP (1600 px and 800 px wide, quality ~80, < 250 kB checked) next to a
PNG master at 1600 px, in the run directory ($CN_OUT); web/gallery/collect.py copies them to the site.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

LAB = "#0B0B0B"
DYE = "#C6F03C"
SMOKE_STOPS = [(0.0, "#0B0B0B"), (0.30, "#3A3A38"), (0.62, "#8A8A86"), (1.0, "#EDEBE6")]
# signed value v in [-1, 1] -> position (v + 1) / 2
SIGNED_STOPS = [(0.0, "#D4574A"), (0.30, "#6E2019"), (0.50, "#0B0B0B"), (0.70, "#3F5245"),
                (1.0, "#9DBBA6")]
MAX_BYTES = 250_000


def _rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)


def lut(stops, n: int = 1024) -> np.ndarray:
    """(n, 3) uint8 lookup table, piecewise linear in sRGB between the stops."""
    pos = np.array([p for p, _ in stops])
    cols = np.array([_rgb(c) for _, c in stops])
    x = np.linspace(0.0, 1.0, n)
    out = np.stack([np.interp(x, pos, cols[:, k]) for k in range(3)], axis=1)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


LUTS = {"smoke": lut(SMOKE_STOPS), "signed": lut(SIGNED_STOPS)}


def normalize(f: np.ndarray, kind: str, *, clip_pct: float = 99.7,
              vmin: float | None = None, vmax: float | None = None) -> tuple[np.ndarray, dict]:
    """Field -> [0, 1]. 'signed': symmetric, v / A with A = percentile(|f|, clip_pct) (or vmax),
    zero exactly at the middle (black). 'smoke': (f - vmin) / (vmax - vmin), defaults are the
    (100 - clip_pct, clip_pct) percentiles. NaN (outside a disk, say) stay NaN (background).
    Returns the normalized array and the scale actually used (written to the metadata)."""
    good = np.isfinite(f)
    if kind == "signed":
        A = float(vmax) if vmax is not None else float(np.percentile(np.abs(f[good]), clip_pct))
        return np.clip(0.5 + 0.5 * f / A, 0.0, 1.0), {"kind": kind, "amplitude": A, "clip_pct": clip_pct}
    if kind == "smoke":
        lo = float(vmin) if vmin is not None else float(np.percentile(f[good], 100 - clip_pct))
        hi = float(vmax) if vmax is not None else float(np.percentile(f[good], clip_pct))
        return np.clip((f - lo) / (hi - lo), 0.0, 1.0), {"kind": kind, "vmin": lo, "vmax": hi,
                                                        "clip_pct": clip_pct}
    raise ValueError(f"colormap {kind!r}: 'smoke' or 'signed' (brand chart, section III)")


def colorize(u: np.ndarray, kind: str) -> np.ndarray:
    """[0, 1] array (NaN = background) -> (H, W, 3) uint8 RGB."""
    L = LUTS[kind]
    idx = np.clip(np.rint(np.nan_to_num(u, nan=0.0) * (len(L) - 1)), 0, len(L) - 1).astype(int)
    rgb = L[idx]
    rgb[~np.isfinite(u)] = _rgb(LAB).astype(np.uint8)
    return rgb


def grain(rgb: np.ndarray, sigma: float = 1.6, seed: int = 34) -> np.ndarray:
    """Faint silver-halide grain: Gaussian luminance noise of ``sigma`` (8-bit units), seeded."""
    if sigma <= 0:
        return rgb
    n = np.random.default_rng(seed).normal(0.0, sigma, rgb.shape[:2])[..., None]
    return np.clip(rgb.astype(float) + n, 0, 255).astype(np.uint8)


def overlay_line(rgb: np.ndarray, mask: np.ndarray, color: str = DYE, alpha: float = 0.9) -> np.ndarray:
    """Draw a (thin, antialiased) line given as a [0, 1] coverage mask in fluorescein."""
    c = _rgb(color)
    a = np.clip(mask, 0, 1)[..., None] * alpha
    return np.clip(rgb * (1 - a) + c * a, 0, 255).astype(np.uint8)


def contour_mask(f: np.ndarray, level: float, width_px: float = 1.2) -> np.ndarray:
    """Coverage of the iso-line f = level: 1 - |f - level| / (|grad f| * width), clipped."""
    gy, gx = np.gradient(f)
    g = np.hypot(gx, gy) + 1e-30
    return np.clip(1.0 - np.abs(f - level) / (g * width_px), 0.0, 1.0)


def save(rgb: np.ndarray, slug: str, out: str | Path, meta: dict, *, quality: int = 80) -> dict:
    """Write <slug>.png (master), <slug>.webp (1600 px) and <slug>-800.webp in ``out``, check
    the size limit (quality lowered by steps of 5 only if needed, and recorded), and write
    <slug>.meta.json (manifest fields known to the case script + image facts)."""
    from PIL import Image

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(rgb, "RGB")
    if im.width != 1600:
        im = im.resize((1600, round(im.height * 1600 / im.width)), Image.LANCZOS)
    im.save(out / f"{slug}.png", optimize=True)
    facts = {}
    for suffix, W in (("", 1600), ("-800", 800)):
        img = im if W == 1600 else im.resize((800, round(im.height * 800 / 1600)), Image.LANCZOS)
        q = quality
        while True:
            p = out / f"{slug}{suffix}.webp"
            img.save(p, "WEBP", quality=q, method=6)
            if p.stat().st_size < MAX_BYTES or q <= 50:
                break
            q -= 5
        facts[f"webp{suffix or '-1600'}"] = {"bytes": p.stat().st_size, "quality": q,
                                             "width": img.width, "height": img.height}
        if p.stat().st_size >= MAX_BYTES:
            raise RuntimeError(f"{p.name}: {p.stat().st_size} B >= {MAX_BYTES} B even at quality {q}")
    meta = dict(meta, slug=slug, width=im.width, height=im.height, files=facts,
                run_id=os.environ.get("CN_RUN_ID"))
    (out / f"{slug}.meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"[gallery] {slug}: {im.width}x{im.height}, "
          + ", ".join(f"{k} {v['bytes'] / 1e3:.0f} kB (q{v['quality']})" for k, v in facts.items()))
    return meta


def resample(f: np.ndarray, x: np.ndarray, y: np.ndarray, W: int, H: int, *,
             periodic_x: float | None = None, periodic_y: float | None = None,
             x_range=None, y_range=None) -> np.ndarray:
    """Cubic-spline resampling of f[y, x] (possibly non-uniform x, y, e.g. Chebyshev nodes) onto a
    uniform W x H pixel grid (pixel centres, row 0 at the top = largest y). A periodic direction
    of period P is wrapped (one extra period of samples on each side) before fitting."""
    from scipy.interpolate import RectBivariateSpline

    def wrap(v, F, P, axis):
        if P is None:
            return v, F
        k = 4
        v2 = np.concatenate([v[-k:] - P, v, v[:k] + P])
        F2 = np.concatenate([np.take(F, range(len(v) - k, len(v)), axis=axis), F,
                             np.take(F, range(k), axis=axis)], axis=axis)
        return v2, F2
    xs, F = wrap(np.asarray(x), f, periodic_x, 1)
    ys, F = wrap(np.asarray(y), F, periodic_y, 0)
    x0, x1 = x_range if x_range else ((0.0, periodic_x) if periodic_x else (x.min(), x.max()))
    y0, y1 = y_range if y_range else ((0.0, periodic_y) if periodic_y else (y.min(), y.max()))
    px = x0 + (np.arange(W) + 0.5) * (x1 - x0) / W
    py = y0 + (np.arange(H) + 0.5) * (y1 - y0) / H          # ascending for the spline
    return RectBivariateSpline(ys, xs, F, kx=3, ky=3)(py, px)[::-1]   # row 0 = top


def field_image(f_px: np.ndarray, kind: str, *, grain_sigma: float = 1.6, **norm) -> tuple[np.ndarray, dict]:
    """Pixel field -> RGB with the chart colormap and grain; returns (rgb, scale)."""
    u, scale = normalize(f_px, kind, **norm)
    return grain(colorize(u, kind), grain_sigma), scale
