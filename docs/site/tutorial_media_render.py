"""Drawing step of docs/site/tutorial_media.py: figures, films and the contact sheet.

Reads the data captured from the tutorials (``<key>.npz``, ``<key>.json``) and draws, with the
style file ``tutorial_media.mplstyle``:

  docs/site/tutorials/media/<key>_<name>.png            figures (transparent, both themes)
  docs/site/tutorials/media/media.json                  file, tutorial, alt text and caption
  docs/site/_static/tutorial_films/<key>_<name>.webm    films (VP9), .mp4 (H.264) and .webp poster
  <data>/contact_sheet.png                              every figure and poster, tiled

Each film shows states of the tutorial's own run at recorded times; when the last frame does not
come back to the first, the loop closes with a 0.5 s cross-fade (a display transition, stated in
media.json).  Field images interpolate the solver's grid values with its own expansions (Fourier
in a periodic direction, Chebyshev between walls).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SITE = Path(__file__).resolve().parent
MEDIA = SITE / "tutorials" / "media"
FILMS = SITE / "_static" / "tutorial_films"
STYLE = SITE / "tutorial_media.mplstyle"

INK, RED, GREEN, BRASS, BLUE, SMOKE = "#77736B", "#C8463A", "#4F8A68", "#A8864F", "#4C6FB0", "#8A8A86"
# Categorical series use red and blue with distinct markers or line styles (readable without colour
# and for red-green colour blindness; external review of 0.1.1, figures). GREEN: only the t10 BT figure.
CAT2 = BLUE
LAB, SILVER, DYE = "#0B0B0B", "#EDEBE6", "#C6F03C"
# signed fields: red for positive, blue for negative through the laboratory black (a red-green
# scale is unreadable for the commonest colour blindness; lot H, presentation review of 0.1.1)
SIGNED = LinearSegmentedColormap.from_list("signed", [(0.0, "#8FB0E0"), (0.30, "#2E4A78"), (0.50, LAB),
                                                       (0.70, "#6E2019"), (1.0, "#D4574A")])
SMOKE_CM = LinearSegmentedColormap.from_list("smoke", [(0.0, LAB), (0.30, "#3A3A38"), (0.62, SMOKE),
                                                       (1.0, SILVER)])
FPS, FADE_S, MAX_VIDEO_BYTES = 24, 0.5, 1_500_000

ENTRIES: list[dict] = []
# Figures and films not published (the tutorial-media prune, 0.1.1): each repeats what the page
# already shows (a printed number, the script's own figure, a film of the same run) or is decorative,
# and the heaviest files were among them. A drawing function still draws them (the contact sheet of
# the data directory shows them); they are not written to the pages.
PRUNED = {"t00_growth_rates", "t00_rolls", "t01_mode", "t02_mode", "t04_profiles", "t05_orbit_png",
          "t08_couette", "t11_certified", "t12_attractor", "t12_spread", "t14_fields", "t15_spacetime",
          "t16_dye_png"}
# Figures another lot of 0.1.2 owns (the Bogdanov-Takens diagram stays with lot A): the
# renderer neither redraws them nor changes their manifest entry or their page block.
KEEP = {"t10_bogdanov_takens"}
# The loop of a film that does not close on itself, said in its caption (external review, 77).
FADE_NOTE = "The film closes its loop with a 0.5 s cross-fade: a display transition, not part of the run."


# =============================================================================== helpers
def _sci(x: float) -> str:
    """1 significant digit in a caption: 3.8e-07 -> "4×10⁻⁷"."""
    m, e = f"{x:.0e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{m}×10{str(int(e)).translate(sup)}"


def small_png(path: Path) -> None:
    """Rewrite a figure with a 256-colour palette (transparency kept): line plots on a transparent
    ground lose nothing visible and the files are several times smaller."""
    from PIL import Image
    with Image.open(path) as im:
        q = im.convert("RGBA").quantize(colors=256, method=Image.Quantize.FASTOCTREE)
    q.save(path, optimize=True)


def save(fig, key: str, name: str, alt: str, caption: str, width: str = "100%") -> None:
    """Write figure ``<key>_<name>.png`` and its manifest entry; ``width`` is the width of the
    figure on the page (a small figure is drawn small and shown at a fraction of the column)."""
    f = f"{key}_{name}.png"
    if f"{key}_{name}" in PRUNED or f"{key}_{name}_png" in PRUNED or f"{key}_{name}" in KEEP:
        plt.close(fig)
        return
    fig.savefig(MEDIA / f)
    plt.close(fig)
    small_png(MEDIA / f)
    ENTRIES.append({"file": f"tutorials/media/{f}", "tutorial": key, "kind": "figure", "alt": alt,
                    "caption": caption, "width": width})
    print(f"[render] {f}", flush=True)


def save_table(key: str, name: str, caption: str, header: list[str], rows: list[list[str]],
               align: str | None = None) -> None:
    """A few numbers are a table, not a figure (presentation review of 0.1.1, item 13): write
    ``<key>_<name>.md``, a MyST table with its caption, which the page includes
    (``{include} media/<key>_<name>.md``), and its manifest entry."""
    f = f"{key}_{name}.md"
    align = align or "l" + "r" * (len(header) - 1)
    rule = "|" + "|".join(":--" if a == "l" else "--:" for a in align) + "|"
    lines = [f"```{{table}} {caption}", f":name: {key}-{name}".replace("_", "-"), "",
             "| " + " | ".join(header) + " |", rule]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    lines += ["```", ""]
    (MEDIA / f).write_text("<!-- written by docs/site/tutorial_media_render.py from the captured data; "
                           "do not edit -->\n" + "\n".join(lines), encoding="utf-8")
    ENTRIES.append({"file": f"tutorials/media/{f}", "tutorial": key, "kind": "table", "caption": caption})
    print(f"[render] {f}", flush=True)


def cheb_interp(F: np.ndarray, nodes: np.ndarray, a: float, b: float, fine: np.ndarray, axis: int):
    """Chebyshev interpolant of grid values F(nodes) along ``axis``, evaluated at ``fine``."""
    s = (2 * nodes - (a + b)) / (b - a)
    Fm = np.moveaxis(F, axis, 0)
    c = np.polynomial.chebyshev.chebfit(s, Fm.reshape(len(s), -1), len(s) - 1)
    v = np.polynomial.chebyshev.chebval((2 * fine - (a + b)) / (b - a), c)   # (rest, fine)
    v = np.moveaxis(v.reshape(Fm.shape[1:] + (len(fine),)), -1, 0)
    return np.moveaxis(v, 0, axis)


def fourier_interp(F: np.ndarray, n: int, axis: int = 0) -> np.ndarray:
    """Trigonometric interpolant of periodic grid values on n equispaced points."""
    m = F.shape[axis]
    return np.fft.irfft(np.fft.rfft(F, axis=axis), n=n, axis=axis) * n / m


def box_field(F, xn, zn, Lx, za, zb, nx=320, nz=161):
    """Grid values of a (periodic x, walls z) field -> fine grid (x from xn[0] over one period)."""
    zf = np.linspace(za, zb, nz)
    Fz = cheb_interp(np.real(F), np.asarray(zn), za, zb, zf, axis=1)
    Ff = fourier_interp(Fz, nx, axis=0)
    xf = xn[0] + np.arange(nx) * Lx / nx
    return xf, zf, Ff


def complex_axes(ax, xlabel="real part", ylabel="imaginary part"):
    ax.axhline(0, color=INK, lw=0.6, alpha=0.5)
    ax.axvline(0, color=INK, lw=0.6, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def ffmpeg_exe() -> str:
    exe = os.environ.get("CN_FFMPEG") or shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def film(frames: list[np.ndarray], key: str, name: str, poster: int, alt: str, caption: str,
         closes: bool = False) -> None:
    """WebM (VP9) + MP4 (H.264) at FPS, no audio, each < MAX_VIDEO_BYTES (the constant quality
    is lowered by steps until it fits), WebP poster; the encoder settings of web/gallery/films.py."""
    from PIL import Image
    slug = f"{key}_{name}"
    if slug in PRUNED:
        return
    if not closes:
        caption = f"{caption} {FADE_NOTE}"
    loop = "the last frame returns to the first (the motion is periodic)" if closes else \
        f"run frames, then a {FADE_S:g} s cross-fade from the last frame to the first"
    n = int(round(FADE_S * FPS))
    cap = 8 * FPS - (0 if closes else n)               # at most 8 s, cross-fade included
    if len(frames) > cap:                              # evenly spaced subset of the run frames
        idx = np.unique(np.round(np.linspace(0, len(frames) - 1, cap)).astype(int))
        poster = int(np.argmin(np.abs(idx - poster)))
        frames = [frames[i] for i in idx]
    if not closes:
        a, b = frames[-1].astype(float), frames[0].astype(float)
        frames = frames + [np.clip(np.rint((1 - w) * a + w * b), 0, 255).astype(np.uint8)
                           for w in np.arange(1, n + 1) / (n + 1)]
    H, W = frames[0].shape[:2]
    exe = ffmpeg_exe()
    sizes = {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            Image.fromarray(f, "RGB").save(Path(tmp) / f"f{i:05d}.png", compress_level=1)
        src = ["-framerate", str(FPS), "-i", str(Path(tmp) / "f%05d.png")]
        common = ["-an", "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact",
                  "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709",
                  "-colorspace", "bt709"]
        specs = {"webm": (34, lambda q: ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", str(q), "-row-mt", "1",
                                         "-deadline", "good", "-cpu-used", "2", "-g", str(len(frames))]),
                 "mp4": (24, lambda q: ["-c:v", "libx264", "-preset", "slow", "-crf", str(q), "-profile:v",
                                        "high", "-movflags", "+faststart"])}
        for ext, (q, opts) in specs.items():
            p = FILMS / f"{slug}.{ext}"
            while True:
                subprocess.run([exe, "-y", "-loglevel", "error", *src, *opts(q), *common, str(p)], check=True)
                if p.stat().st_size < MAX_VIDEO_BYTES or q >= 51:
                    break
                q += 3
            if p.stat().st_size >= MAX_VIDEO_BYTES:
                raise RuntimeError(f"{p.name}: {p.stat().st_size} B")
            sizes[ext] = p.stat().st_size
    Image.fromarray(frames[poster], "RGB").save(FILMS / f"{slug}.webp", "WEBP", quality=82, method=6)
    ENTRIES.append({"file": f"_static/tutorial_films/{slug}.webm", "mp4": f"_static/tutorial_films/{slug}.mp4",
                    "poster": f"_static/tutorial_films/{slug}.webp", "tutorial": key, "kind": "film",
                    "width": W, "height": H, "fps": FPS, "frames": len(frames),
                    "duration_s": round(len(frames) / FPS, 2), "bytes": sizes, "loop": loop,
                    "alt": alt, "caption": caption})
    print(f"[render] {slug}: {W}x{H}, {len(frames) / FPS:.1f} s, webm {sizes['webm'] / 1e3:.0f} kB, "
          f"mp4 {sizes['mp4'] / 1e3:.0f} kB", flush=True)


def film_figure(w_px: int, h_px: int):
    """A figure of exactly w_px x h_px pixels on the laboratory black, for film frames."""
    fig = plt.figure(figsize=(w_px / 100, h_px / 100), dpi=100, facecolor=LAB)
    return fig


def load(data: Path, key: str):
    z = np.load(data / f"{key}.npz")
    return {k: z[k] for k in z.files}, json.loads((data / f"{key}.json").read_text())


# =============================================================================== tutorials
def r_t00(d, meta, films=True):
    Ra_c, k_c, sig = meta["Ra_c"], meta["k_c"], meta["sigma"]
    # 1. leading growth rates at the two Rayleigh numbers
    fig, ax = plt.subplots(figsize=(7, 2.6))
    for row, Ra in enumerate((1500, 2500)):
        s = np.sort(d[f"spec_{Ra}"].real)[::-1][:12]
        ax.scatter(s[1:], np.full(s.size - 1, row), s=26, color=SMOKE, zorder=3)
        ax.scatter(s[:1], [row], s=48, color=RED if s[0] > 0 else CAT2, zorder=4)
        ax.annotate(f"{s[0]:+.3f}", (s[0], row), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    ax.axvline(0, color=INK, lw=0.8, ls="--")
    ax.set_yticks([0, 1], ["Ra = 1500", "Ra = 2500"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(-130, 15)
    ax.set_xlabel("growth rate of each mode (thermal diffusion units)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "t00", "growth_rates",
         "Two rows of dots, the growth rates of the roll k = pi at Ra = 1500 and Ra = 2500. Every "
         "dot of the first row lies left of zero; in the second row the leading dot sits right of zero.",
         "The twelve largest growth rates of the roll k = π about the fluid at rest. At Ra = 1500 every "
         "mode decays; at Ra = 2500 one mode grows: the fluid at rest is unstable.")
    # 2. neutral curve from the parametric pencil
    K, R = np.meshgrid(d["ks"], d["ras"])
    G = d["growth"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.contourf(K, R, G, levels=[0, np.inf], colors=[RED], alpha=0.14)
    ax.contour(K, R, G, levels=[0], colors=[RED], linewidths=1.8)
    from matplotlib.patches import Patch
    ax.plot(k_c, Ra_c, "o", color=RED, ms=6, label="critical point")
    ax.plot([np.pi, np.pi], [1500, 2500], "x", color=INK, ms=7, mew=1.5, clip_on=False, label="Ra = 1500 and 2500, k = π")
    h, lab = ax.get_legend_handles_labels()
    h.insert(0, Patch(color=RED, alpha=0.3))
    lab.insert(0, "growth rate > 0")
    ax.legend(h, lab, loc="upper center", fontsize=9)
    ax.set_xlabel("wavenumber k")
    ax.set_ylabel("Rayleigh number Ra")
    save(fig, "t00", "neutral_curve",
         "The neutral curve in the plane of wavenumber and Rayleigh number: a U-shaped red line with "
         "its minimum marked at Ra about 1708 and k about 3.12; the region above it, where modes grow, "
         "is shaded; two crosses at k = pi mark Ra = 1500 and 2500.",
         f"Rayleigh-Bénard onset between rigid plates, Pr = 1: growth rate of the least stable mode over a "
         f"grid of (k, Ra), from the parametric pencil of the onset block. The red line is where it vanishes; "
         f"modes grow in the shaded region. Its minimum is the critical point Ra_c = {Ra_c:.3f} at "
         f"k_c = {k_c:.4f}. The crosses are the two runs of the page, Ra = 1500 (stable) and 2500 (unstable) "
         f"at k = π.")
    # 3. growth at the predicted rate, then saturation
    t, a, nu = d["t"], d["amp"], d["nu"]
    t0 = 0.3                                        # the eigenvalue line passes through the run here
    j = np.searchsorted(t, t0)
    lin = a[j] * np.exp(sig * (t - t[j]))
    tl = t[(t <= t[-1]) & (lin < 0.3)]              # drawn while the amplitude stays small
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.0))
    a1.semilogy(t, a, color=INK, lw=1.4, label="simulation")
    a1.semilogy(tl, a[j] * np.exp(sig * (tl - t[j])), "--", color=RED, lw=1.2,
                label=f"eigenvalue rate σ = {sig:.3f}")
    a1.plot([t[j]], [a[j]], "o", color=RED, ms=4)
    a1.set_ylim(5e-7, 1)
    a1.set_xlabel("time t  (units of d²/κ)")
    a1.set_ylabel("roll amplitude")
    a1.legend(loc="lower right", fontsize=9)
    a2.plot(t, nu, color=INK, lw=1.4, label="simulation")
    a2.axhline(1, color=SMOKE, lw=1.0, ls=":", label="conduction, Nu = 1")
    a2.set_xlabel("time t  (units of d²/κ)")
    a2.set_ylabel("Nusselt number Nu")
    a2.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    late = t >= t[-1] - 1.0
    save(fig, "t00", "growth_saturation",
         "Left: the roll amplitude on a log scale rises along a straight dashed line, then levels off. "
         "Right: the Nusselt number stays at 1, rises to a peak and settles to a constant.",
         f"The simulation of the page at Ra = 2500, Pr = 1; time in thermal diffusion units d²/κ. Left: the "
         f"roll amplitude grows at the rate σ = {sig:.3f} of the leading eigenvalue (dashed line, drawn through "
         f"the run at t = {t0:g}) until the nonlinear terms stop it. Right: the heat flux across the layer in "
         f"units of conduction, Nu = {nu[-1]:.3f} at t = {t[-1]:.0f} (it varies by "
         f"{_sci(max(float(np.ptp(np.asarray(nu)[late])), 1e-16))} over the last time unit).")
    if not films:
        return
    # film: the rolls appear and settle (fixed colour scale, as in the page's figure)
    xn, zn, snaps, st = d["xn"], d["zn"], d["snaps"], d["snap_t"]
    W, H = 720, 400
    frames = []
    for T, tt in zip(snaps, st, strict=False):
        xf, zf, Tf = box_field(T, xn, zn, 2.0, -0.5, 0.5, nx=360, nz=181)
        fig = film_figure(W, H)
        ax = fig.add_axes([0.02, 0.05, 0.96, 0.86])
        ax.imshow((Tf + zf[None, :]).T, origin="lower", extent=[xf[0], xf[0] + 2, -0.5, 0.5],
                  cmap=SIGNED, vmin=-0.3, vmax=0.3, interpolation="bilinear", aspect="equal")
        ax.contour(xf, zf, Tf.T, levels=np.linspace(-0.45, 0.45, 10), colors=SILVER, linewidths=0.5)
        ax.set_axis_off()
        fig.text(0.02, 0.94, f"t = {tt:.2f}", color=SILVER, fontsize=11, family="monospace")
        fig.text(0.98, 0.94, "Ra = 2500, Pr = 1", color=SMOKE, fontsize=10, ha="right")
        frames.append(fig_to_rgb(fig))
        plt.close(fig)
    film(frames, "t00", "rolls", poster=len(frames) - 1,
         alt="Temperature of the fluid between the two plates: from an almost uniform layer, a warm "
             "red column and a cool blue one appear, grow and settle into two steady rolls, with "
             "isotherms bending around them.",
         caption=f"The simulation of the page from t = {st[0]:.1f} to {st[-1]:.1f}: departure of the "
                 "temperature from conduction (warm red, cool blue) and isotherms, from states recorded every 25 time steps.")


def r_t01(d, meta, films=True):
    vals, i = d["values"], int(d["i"])
    c = vals[i]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    others = np.delete(vals, i)
    ax.scatter(others.real, others.imag, s=16, color=SMOKE, zorder=3, label="modes kept by the drift test")
    ax.scatter([c.real], [c.imag], s=60, color=RED, zorder=4, label="least stable mode")
    ax.axhline(0, color=INK, lw=0.6, ls="--")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-1.0, 0.08)
    ax.set_xlabel("phase speed  Re c")
    ax.set_ylabel("Im c  (growth if > 0)")
    ax.legend(loc="lower left", fontsize=9)
    save(fig, "t01", "spectrum",
         "A few eigenvalues in the plane of phase speed and growth, all below the zero line except one red "
         "point just above it near Re c = 0.24.",
         f"Plane Poiseuille flow at Re = 10⁴, α = 1: the {vals.size} modes kept by the drift test of the page "
         f"(found again at N = 96). A mode varies as exp(iα(x − ct)), so Re c is its phase speed and α Im c its "
         f"growth rate. Only the Tollmien-Schlichting wave (red), c = {c.real:.5f} + {c.imag:.5f}i, lies above "
         f"the dashed line Im c = 0.", width="90%")
    y, v = d["y"], d["v"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(y, v.real, color=RED, label="Re v")
    ax.plot(y, v.imag, "-.", color=CAT2, label="Im v")
    ax.plot(y, np.abs(v), "--", color=INK, lw=1.0, label="|v|")
    ax.set_xlabel("wall-normal coordinate y")
    ax.set_ylabel("v(y)")
    ax.legend(loc="lower center", ncol=3)
    save(fig, "t01", "mode",
         "Three curves across the channel from y = -1 to 1: the real part, the imaginary part and the "
         "modulus of the eigenfunction, which vanish at both walls; the modulus peaks at 1 at the centre.",
         "The least stable eigenfunction v(y), normalised as mode.explain() declares: max |v| = 1, real "
         "and positive where |v| is largest. It vanishes with its slope at both walls.")
    a = np.abs(d["coeffs"])
    fig, ax = plt.subplots(figsize=(7, 3.0))
    n = np.arange(a.size)
    ev, od = (n % 2 == 0) & (a > 0), (n % 2 == 1) & (a > 0)
    ax.semilogy(n[ev], a[ev], "o", ms=4, color=RED, label="even n")
    ax.semilogy(n[od], a[od], "s", ms=3.2, mfc="none", color=CAT2, label="odd n")
    ax.set_xlabel("Chebyshev index n")
    ax.set_ylabel("|coefficient|")
    ax.legend(loc="upper right")
    # caption values measured on the plotted arrays (external review of 0.1.1, F26)
    last_even = a[n[ev][-1]]
    odd_med = float(np.median(a[od])) if od.any() else 0.0
    w = max(1, int(np.ceil(0.1 * a.size)))
    tail = float(a[-w:].max() / a.max())
    save(fig, "t01", "coefficients",
         f"The magnitudes of the {a.size} Chebyshev coefficients of the mode on a log scale, even indices as "
         f"red dots and odd indices as open blue squares: the even ones fall from 1 to about "
         f"{_sci(last_even)}; the odd ones stay near {_sci(odd_med)}.",
         f"The Chebyshev coefficients of the mode (mode.coeffs(\"v\")), even and odd indices apart. v is even "
         f"in y, so only the even coefficients carry it: they fall to {_sci(last_even)} at the last even index. "
         f"The odd ones stay at the rounding level, about {_sci(odd_med)}. The largest of the last 10 % of "
         f"the coefficients is {_sci(tail)} of the largest one, the spectral-tail measure that the eigenvalue "
         f"result reports.")
    Re, al, G = d["Re"], d["alpha"], d["growth"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    RR, AA = np.meshgrid(Re, al)
    ax.contourf(RR, AA, G, levels=[0, np.inf], colors=[RED], alpha=0.14)
    ax.contour(RR, AA, G, levels=[0], colors=[RED], linewidths=1.8)
    from matplotlib.patches import Patch
    ax.plot(meta["Re_c"], meta["alpha_c"], "o", color=RED, label="critical point")
    ax.plot(1e4, 1.0, "x", color=INK, ms=7, mew=1.5, label="Re = 10⁴, α = 1 (the spectrum above)")
    h, lab = ax.get_legend_handles_labels()
    ax.legend([Patch(color=RED, alpha=0.3)] + h, ["unstable: Im c > 0"] + lab, loc="upper right", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Reynolds number Re")
    ax.set_ylabel("wavenumber α")
    save(fig, "t01", "neutral_curve",
         "The neutral curve of plane Poiseuille flow: a red tongue opening to the right in the plane of "
         "Reynolds number (log scale) and wavenumber, its nose marked at Re about 5772 and alpha about 1.02.",
         f"Im c of the least stable mode on a grid of (Re, α), from the parametric pencil of the page. Inside "
         f"the red line the flow is unstable. Its nose is the critical point the script finds, "
         f"Re_c = {meta['Re_c']:.1f} at α_c = {meta['alpha_c']:.4f}.")


def r_t02(d, meta, films=True):
    t, a, s = d["t"], d["amp"], meta["sigma_evp"]
    t1 = 0.3                                         # start of the interval the script measures on
    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.axvspan(t1, t[-1], color=SMOKE, alpha=0.12, lw=0, label=f"interval of the measured rate, t = {t1:g} to {t[-1]:g}")
    ax.semilogy(t, a, "o", ms=3, color=INK, label="simulation, every 10 steps")
    j = np.searchsorted(t, t1)
    ax.semilogy(t, a[j] * np.exp(s * (t - t[j])), color=RED, lw=1.2, label=f"eigenvalue rate σ = {s:.4f}")
    ax.set_xlabel("time t")
    ax.set_ylabel("roll amplitude")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "t02", "growth",
         "The roll amplitude on a log scale against time: the dots bend during a short start, then follow "
         "a straight red line over the shaded interval.",
         f"The initial-value problem against the eigenvalue problem. Before t = {t1:g} faster modes are still "
         f"decaying; over the shaded interval the amplitude grows at σ = {meta['sigma_ivp']:.7f}, the "
         f"eigenvalue's {s:.7f}: a relative difference of {_sci(abs(meta['sigma_ivp'] - s) / abs(s))}, within "
         f"the 10⁻⁵ the script checks.")
    # the restart check of the page, as a job chain (presentation review of 0.1.1: a continuous curve
    # cannot show a bitwise restart; the equality test and the largest difference can)
    tc, T = meta["checkpoint_t"], meta["chain_t"]
    fig, ax = plt.subplots(figsize=(7, 1.9))
    ax.barh(1, tc, left=0, height=0.42, color=SMOKE, label="job 1")
    ax.barh(1, T - tc, left=tc, height=0.42, color=RED, label="job 2, resumed from the checkpoint")
    ax.barh(0, T, left=0, height=0.42, color=CAT2, label="uninterrupted run")
    ax.plot([tc], [1], "D", ms=7, color=LAB, mec=SILVER, zorder=4)
    ax.annotate(f"checkpoint, iteration {meta['checkpoint_iteration']}", (tc, 1), xytext=(0, 13),
                textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_yticks([0, 1], ["one job", "two jobs"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, T * 1.02)
    ax.set_ylim(-0.5, 1.75)
    ax.set_xlabel("time t")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=3, fontsize=9)
    same = "identical" if meta["chain_identical"] else "different"
    save(fig, "t02", "restart",
         "Two horizontal bars over time: the upper one in two parts, job 1 then job 2, joined at a "
         "checkpoint marker at t = 0.3; the lower one a single uninterrupted run to the same end time.",
         f"The jobs block of the page: job 2 resumes from the checkpoint that job 1 wrote at iteration "
         f"{meta['checkpoint_iteration']} (t = {tc:g}) and runs to t = {T:g} (iteration {meta['chain_iteration']}). "
         f"Its final state is {same} to that of the uninterrupted run: np.array_equal is "
         f"{meta['chain_identical']} over the {meta['chain_unknowns']} coefficients, largest difference "
         f"{meta['chain_max_abs_diff']:g}. The output file holds {meta['chain_rows']} rows, t = 0 included.")
    xn, zn = d["xn"], d["zn"]
    xf, zf, Tf = box_field(d["T"], xn, zn, 2.0, -0.5, 0.5)
    th = Tf + zf[None, :]
    A = np.abs(th).max()
    has_mode = "mode_T" in d
    fig, axs = plt.subplots(1, 2 if has_mode else 1, figsize=(7.6, 2.9),
                            gridspec_kw={"width_ratios": [2.2, 1]} if has_mode else None)
    ax = axs[0] if has_mode else axs
    ax.imshow(th.T, origin="lower", extent=[xf[0], xf[0] + 2, -0.5, 0.5], cmap=SIGNED, vmin=-A, vmax=A,
              aspect="equal", interpolation="bilinear")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    if has_mode:
        b = axs[1]
        ix = int(np.argmax(np.abs(d["T"] + zn[None, :]).max(axis=1)))
        prof = np.real(d["T"][ix] + zn)
        prof = prof / prof[np.argmax(np.abs(prof))]
        mz, mT = d["mode_z"], d["mode_T"]
        if mT.ndim == 2:                            # (x points of the roll, z): its warmest column
            mT = mT[int(np.argmax(np.abs(mT).max(axis=1)))]
        mT = np.real(mT / mT[np.argmax(np.abs(mT))])
        b.plot(mT, mz, color=RED, label="eigenfunction")
        b.plot(prof, zn, "o", ms=3, color=INK, label="simulation")
        b.set_xlabel("θ(z), normalised")
        b.set_ylabel("z")
        b.legend(loc="lower center", fontsize=8.5, bbox_to_anchor=(0.5, 1.0), ncol=1)
    fig.tight_layout()
    save(fig, "t02", "mode",
         "Left: the temperature perturbation at t = 0.8, one warm and one cool column. Right: its vertical "
         "profile, dots on top of the red eigenfunction curve.",
         "The temperature perturbation θ = T + z at t = 0.8, in the colours of the site (warm red, cool "
         "blue). Right: its profile through the warmest column lies on the leading eigenfunction of the EVP.")


def r_t03(d, meta, films=True):
    # 1. the spectra on the real axis, one row per space (the modes here are real: a complex plane
    # would leave most of the picture empty; presentation review of 0.1.1)
    f, o, a = d["full"], d["onset"], d["other"]
    keep = lambda v: v[(v.real > -160) & np.isfinite(v)]  # noqa: E731
    F, O, A = keep(f), keep(o), keep(a)
    both = np.concatenate([O, A])
    gap = np.array([np.abs(both - v).min() / max(1.0, abs(v)) for v in F])
    im_max = float(np.abs(np.concatenate([F, O, A]).imag).max())
    fig, ax = plt.subplots(figsize=(7, 2.5))
    for v in F:                                     # each full-space value joined to its class
        row = 1 if np.abs(O - v).min() <= np.abs(A - v).min() else 0
        ax.plot([v.real, v.real], [2, row], "-", color=SMOKE, lw=0.6, alpha=0.7, zorder=1)
    ax.scatter(F.real, np.full(F.size, 2), s=46, facecolors="none", edgecolors=INK, lw=1.0, zorder=3,
               label=f"full space, n = {meta['n_full']}")
    ax.scatter(O.real, np.full(O.size, 1), s=22, color=RED, zorder=3, label=f"onset class, n = {meta['n_onset']}")
    ax.scatter(A.real, np.full(A.size, 0), s=30, marker="x", color=CAT2, zorder=3,
               label=f"other class, n = {meta['n_other']}")
    ax.axvline(0, color=INK, lw=0.6, ls="--")
    ax.set_yticks([0, 1, 2], ["other class", "onset class", "full space"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel("growth rate σ  (per thermal diffusion time d²/κ)")
    save(fig, "t03", "spectra",
         "Three rows of eigenvalues along the growth-rate axis: open circles for the full space, red dots for "
         "the onset class, blue crosses for the other class; thin lines join each circle to the class row that "
         "holds it, and the rightmost circle, at zero, joins a red dot.",
         f"The {F.size} eigenvalues of the full space with σ > −160, each joined to the symmetry class that "
         f"holds the same value (largest relative gap {_sci(max(float(gap.max()), 1e-16))}). All are real "
         f"(largest |Im σ| {_sci(max(im_max, 1e-16))}). The neutral mode at the onset point, σ ≈ 0, is in the "
         f"onset class.")
    # 2. size and cost: a table (a few numbers), repeated timings with the machine they ran on
    R, mach = meta["repeats"], meta["machine"]
    rows, med = [], {}
    for label, key in (("full space", "full"), ("onset class", "onset"), ("other class", "other")):
        w = np.asarray(meta[f"wall_{key}_ms"])
        med[key] = float(np.median(w))
        rows.append([label, str(meta[f"n_{key}"]), f"{med[key]:.0f}", f"{w.min():.0f} – {w.max():.0f}"])
    gain = med["full"] / (0.5 * (med["onset"] + med["other"]))
    save_table("t03", "cost",
               f"Size of each eigenproblem and wall time of its dense QZ solve, {R} timings each (median and "
               f"range), from one run on {mach['cpu']} with {mach['threads']} threads (NumPy {mach['numpy']}, "
               f"SciPy {mach['scipy']}). Halving the unknowns made the solve {gain:.1f} times faster here; the "
               f"n³ operation count bounds the saving at 8.",
               ["problem", "unknowns n", "QZ time, median (ms)", "range (ms)"], rows)


def r_t04(d, meta, films=True):
    from scipy.optimize import brentq
    lam, prof, X = d["lam"], d["prof"], d["X"]
    i0 = int(np.argmin(np.abs(X)))
    u0 = prof[:, i0] if abs(X[i0]) < 1e-12 else np.array([np.interp(0, X[::-1] if X[0] > X[-1] else X,
                                                                    p[::-1] if X[0] > X[-1] else p) for p in prof])
    amax = np.linspace(1e-3, 4.0, 400)                       # closed form: u(0) = 2 ln cosh a
    lam_cf, u_cf = 2 * amax ** 2 / np.cosh(amax) ** 2, 2 * np.log(np.cosh(amax))
    a_star = brentq(lambda a: a * np.tanh(a) - 1.0, 0.5, 2.0)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(lam_cf, u_cf, color=SMOKE, lw=1.0, label="closed form")
    ax.plot(lam, u0, "o", ms=4, color=INK, label="continuation")
    ax.plot(meta["lam_c"], 2 * np.log(np.cosh(a_star)), "o", ms=7, color=RED, label="fold located by the script")
    ax.set_xlabel("λ")
    ax.set_ylabel("u(0)")
    ax.set_xlim(0, 0.95)
    ax.set_ylim(0, max(u0.max() * 1.1, 2))
    ax.legend(loc="upper left")
    save(fig, "t04", "branch",
         "The solution branch of the Bratu problem: u(0) rises with lambda, turns back at a red fold near "
         "lambda = 0.878 and climbs on the upper branch, the dots lying on a thin closed-form curve.",
         f"The branch followed by pseudo-arclength continuation (the page's settings, run for more points), "
         f"on the closed form u(0) = 2 ln cosh a. The fold, located by bisection on the tangent, is at "
         f"λ_c = {meta['lam_c']:.10f}.")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    cm = LinearSegmentedColormap.from_list("b", [SMOKE, RED])
    for j, p in enumerate(prof):
        ax.plot(X, p, color=cm(j / max(len(prof) - 1, 1)), lw=1.0)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.text(0.02, 0.92, "grey: lower branch, red: upper branch", transform=ax.transAxes, fontsize=9, color=INK)
    save(fig, "t04", "profiles",
         "Nested arch-shaped profiles u(x) vanishing at x = -1 and 1, grey ones low and red ones tall.",
         "The solutions along the branch, from the start at λ = 0.3 (grey) round the fold to the upper "
         "branch (red).")
    # Newton: four or five numbers, so a table (presentation review of 0.1.1, item 13)
    r = np.asarray(d["res"], dtype=float)
    digits = np.abs(np.log10(np.maximum(r, 1e-300)))   # r(0) = 1 gives 0.0, not -0.0
    rows = [[str(k), f"{v:.1e}", f"{dg:.1f}"] for k, (v, dg) in enumerate(zip(r, digits, strict=True))]
    # observed order of convergence, log(r(k+1)/r(k)) / log(r(k)/r(k-1)): 2 for quadratic convergence
    p = [np.log(r[k + 1] / r[k]) / np.log(r[k] / r[k - 1]) for k in range(1, r.size - 1)
         if r[k + 1] > 0 and r[k] != r[k - 1]]
    tol = meta.get("tol")
    order = (f" The observed order log(r(k+1)/r(k)) / log(r(k)/r(k−1)) is "
             f"{', '.join(f'{v:.2f}' for v in p)} (2 for quadratic convergence)." if p else "")
    save_table("t04", "newton",
               f"Newton from u = 0 at λ = 0.3, as printed by sol.explain(): the relative residual after each "
               f"iteration, {r.size - 1} iterations down to {r[-1]:.0e}"
               + (f", below the declared tolerance {tol:.0e}." if tol else ".") + order,
               ["iteration k", "relative residual r(k)", "−log₁₀ r(k)"], rows)


def r_t05(d, meta, films=True):
    L = meta["L"]
    xf = np.arange(256) * L / 256
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for name, col, ls in (("E1", RED, "-"), ("E2", CAT2, "--"), ("TW1", BRASS, ":")):
        uf = fourier_interp(d[name], 256)
        lab = name + (f" (travels at c = {meta['c']:.5f})" if name == "TW1" else "")
        ax.plot(np.append(xf, L), np.append(uf, uf[0]), ls, color=col, label=lab)
    ax.set_xlim(0, L)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.legend(loc="upper right", fontsize=9, ncol=3, bbox_to_anchor=(1, 1.15))
    save(fig, "t05", "solutions",
         "Three periodic profiles over x from 0 to 22: the equilibria E1 and E2 and the travelling wave TW1.",
         "The two equilibria and the travelling wave found by Newton-Krylov-hookstep from the page's seeds.")
    U = d["orbit"]
    T = meta["T"]
    nt = U.shape[0]
    tt = np.linspace(0, 2 * T, nt)
    Uf = fourier_interp(U, 256, axis=1)
    A = np.abs(Uf).max()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.imshow(Uf, origin="lower", aspect="auto", extent=[0, L, 0, 2 * T], cmap=SIGNED, vmin=-A, vmax=A,
              interpolation="bilinear")
    ax.axhline(T, color=SILVER, lw=0.8, ls="--")
    ax.text(L * 0.99, T, " t = T, reflected  ", color=SILVER, ha="right", va="bottom", fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    save(fig, "t05", "orbit",
         "A space-time map of the orbit, x across and t up over two periods, with red and blue streaks; "
         "the pattern of the upper half is the mirror image of the lower half.",
         f"The pre-periodic orbit over two periods, T = {T:.6f}: after one period the state comes back as "
         "its mirror image u(-x), after two it comes back to itself.")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.4))
    th = np.linspace(0, 2 * np.pi, 400)
    m = d["mult"]
    a1.plot(np.cos(th), np.sin(th), color=SMOKE, lw=0.8)
    a1.scatter(m.real, m.imag, s=26, color=INK, zorder=3)
    lead = max((v for v in m if abs(abs(v) - 1) > 1e-6), key=abs)
    a1.scatter([lead.real, lead.real], [lead.imag, -lead.imag], s=50, color=RED, zorder=4)
    a1.set_aspect("equal")
    complex_axes(a1, "Re Λ", "Im Λ")
    a1.set_title("Floquet multipliers Λ of the orbit", fontsize=10)
    e = d["ev1"]
    a2.scatter(e.real, e.imag, s=26, color=INK)
    top = e[np.argsort(-e.real)[:4]]
    a2.scatter(top.real, top.imag, s=50, color=RED)
    complex_axes(a2, "Re λ", "Im λ")
    a2.set_title("eigenvalues of E1", fontsize=10)
    fig.tight_layout()
    save(fig, "t05", "stability",
         "Left: Floquet multipliers of the orbit with the unit circle; a red pair lies just outside it. "
         "Right: the leading eigenvalues of E1, two red complex pairs with positive real part.",
         f"Stability of the solutions. Left: the multipliers Λ of the orbit; the red pair outside the unit "
         f"circle is the unstable direction, with Floquet exponent μ₁ = log|Λ|/T = {meta['mu1']:.4f}. Right: the "
         f"eigenvalues λ of E1, modes growing as exp(λt); the red ones are the four with the largest real part.")
    if not films:
        return
    W, H = 720, 480
    every = int(np.ceil((nt - 1) / (8 * FPS)))        # equal steps, at most 8 s
    idx = list(range(0, nt - 1, every))
    frames = []
    for k in idx:
        fig = film_figure(W, H)
        a = fig.add_axes([0.08, 0.60, 0.88, 0.33], facecolor=LAB)
        uf = Uf[k]
        a.plot(np.append(xf, L), np.append(uf, uf[0]), color=DYE, lw=1.8)
        a.set_xlim(0, L)
        a.set_ylim(-A * 1.1, A * 1.1)
        a.axhline(0, color="#3A3A38", lw=0.6)
        a.set_axis_off()
        b = fig.add_axes([0.08, 0.06, 0.88, 0.50])
        img = np.full_like(Uf, np.nan)
        img[:k + 1] = Uf[:k + 1]
        b.imshow(img, origin="lower", aspect="auto", extent=[0, L, 0, 2 * T], cmap=SIGNED, vmin=-A, vmax=A,
                 interpolation="bilinear")
        b.set_facecolor(LAB)
        b.set_axis_off()
        fig.text(0.08, 0.95, f"t = {tt[k]:5.2f}", color=SILVER, fontsize=11, family="monospace")
        fig.text(0.96, 0.95, f"one period T = {T:.4f}", color=SMOKE, fontsize=10, ha="right")
        frames.append(fig_to_rgb(fig))
        plt.close(fig)
    closes = np.abs(U[-1] - U[0]).max() < 1e-6 * np.abs(U).max()
    film(frames, "t05", "orbit", poster=len(frames) // 2,
         alt="A yellow-green curve u(x) wriggles in the upper part of the frame while, below, its history "
             "fills a red and blue space-time map from the bottom up over two periods of the orbit.",
         caption="The pre-periodic orbit over two periods: the state u(x) (top) and its space-time map "
                 "filling up (bottom). Every frame is a time step of the orbit's flow map.",
         closes=bool(closes))


def r_t08(d, meta, films=True):
    from scipy.special import jn_zeros
    # the drum: eigenvalues on the Bessel ticks; the errors are all at rounding level, so they are
    # summarised in the caption instead of a panel of roundoff noise (presentation review of 0.1.1)
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    worst, count = 0.0, 0
    for mm, col, mk in zip(range(3), (RED, CAT2, BRASS), ("o", "s", "^"), strict=False):
        s = d[f"drum_{mm}"]
        n = np.arange(1, s.size + 1)
        j2 = jn_zeros(mm, s.size) ** 2
        ax.plot(n, -s, mk, ms=5, color=col, label=f"m = {mm}", zorder=3)
        ax.plot(n, j2, "_", ms=12, mew=1.4, color=INK, label="j²ₘ,ₙ (closed form)" if mm == 0 else None)
        worst, count = max(worst, float((np.abs(s + j2) / j2).max())), count + s.size
    ax.set_xlabel("radial index n")
    ax.set_ylabel("−s  (units of 1/R²)")
    ax.legend(loc="upper left", fontsize=9)
    err = (f"the largest relative error is {_sci(worst)}" if worst > 0 else
           "they agree in every bit of double precision")
    save(fig, "t08", "drum",
         "The computed eigenvalues of the drum for m = 0, 1, 2 as coloured markers, each on a black tick "
         "of the squared Bessel zeros, rising with the radial index.",
         f"The vibrating drum at N = 32 radial modes: the eight slowest modes of each azimuthal number m "
         f"against the closed form −j²ₘ,ₙ (ticks). Over these {count} modes {err}, the level of rounding "
         f"error.", width="85%")
    p = d["pipe"]
    p = p[np.isfinite(p)]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    keep = p.real > -1.2
    top = p[np.argsort(-p.real)[:2]]
    centre, wall = sorted(top, key=lambda v: -abs(v.imag))      # the centre mode travels faster
    rest = p[keep & ~np.isin(p, top)]
    ax.scatter(rest.real, rest.imag, s=16, color=SMOKE, label="other modes")
    ax.scatter([centre.real], [centre.imag], s=54, color=RED, zorder=3, label="centre mode")
    ax.scatter([wall.real], [wall.imag], s=46, marker="s", color=CAT2, zorder=3, label="wall mode")
    complex_axes(ax, "growth rate  Re s", "frequency  Im s")
    ax.legend(loc="lower left", fontsize=9)
    fmt = lambda v: f"{v.real:.4f} {'+' if v.imag >= 0 else '−'} {abs(v.imag):.4f}i"  # noqa: E731
    save(fig, "t08", "pipe_spectrum",
         "The spectrum of pipe flow at Re = 1000 for (m, k) = (1, 1): eigenvalues on three branches in the "
         "left half of the complex plane; the two rightmost are marked, a red dot and a blue square.",
         f"Hagen-Poiseuille flow, R = 1000, (m, k) = (1, 1): the spectrum of the pipe block, modes growing as "
         f"exp(st). All modes decay. The two least stable are the centre mode, s = {fmt(centre)}, and the wall "
         f"mode, s = {fmt(wall)}, of Meseguer and Trefethen (2003).")
    kk, Re = d["tc_k"], d["tc_Re"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(kk, Re, "o-", color=INK, ms=4)
    pts = {float(k): v for k, v in meta["tc_points"].items()}
    ax.plot(list(pts), list(pts.values()), "o", color=RED, ms=6, label="the three wavenumbers of the page")
    kc = min(pts, key=pts.get)
    ax.annotate(f"Re$_c$ = {pts[kc]:.4f} at k = {kc}", (kc, pts[kc]), xytext=(0, -20), textcoords="offset points",
                ha="center", fontsize=10, color=INK)
    ax.fill_between(kk, Re, max(Re) * 1.05, color=RED, alpha=0.10, lw=0)
    ax.text(kk[0] + 0.05, max(Re) * 1.03, "above the curve, Taylor vortices grow", ha="left", va="top", color=RED, fontsize=10)
    ax.set_xlabel("axial wavenumber k")
    ax.set_ylabel("neutral Reynolds number")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "t08", "couette",
         "The neutral curve of Taylor-Couette flow: neutral Reynolds number against axial wavenumber, a "
         "shallow U with three red points near its minimum at about 68.19.",
         "Taylor-Couette flow, η = 0.5, outer cylinder at rest: the neutral Reynolds number of axisymmetric "
         "modes, by root finding on couette_growth of the page at each k.")


def r_t10(d, meta, films=True):
    nm = meta["origin_names"]
    ir = nm.get("r", 3)
    fig, ax = plt.subplots(figsize=(7, 3.8))

    def branch(Z, U, sign=1.0, col=INK, label=None):
        r, x = Z[:, ir], sign * Z[:, 0]
        st = U == 0
        for mask, ls in ((st, "-"), (~st, "--")):
            xm = np.where(mask, x, np.nan)
            ax.plot(r, xm, ls, color=col, lw=1.6, label=label if ls == "-" else None)
    branch(d["origin_Z"], d["origin_unstable"], col=INK)
    branch(d["cplus_Z"], d["cplus_unstable"], 1.0, RED, "C±, stable (solid) and unstable (dashed)")
    branch(d["cplus_Z"], d["cplus_unstable"], -1.0, RED)
    cz = d["cycles_Z"]
    ncy = meta["cycles_names"]
    rc = cz[:, ncy["r"]]
    try:
        orb = cz[:, :-2].reshape(cz.shape[0], -1, 3)
        ax.plot(rc, orb[:, :, 0].max(axis=1), "-", color=BRASS, lw=1.6, label="unstable cycles, max x")
        ax.plot(rc, orb[:, :, 0].min(axis=1), "-", color=BRASS, lw=1.6)
    except ValueError:
        orb = None
    ax.plot(meta["r_bp"], 0, "o", color=INK)
    ax.annotate("pitchfork, r = 1", (meta["r_bp"], 0), xytext=(8, -14), textcoords="offset points", fontsize=9, color=INK)
    xh = np.sqrt(8 / 3 * (meta["r_hopf"] - 1))
    ax.plot([meta["r_hopf"]] * 2, [xh, -xh], "o", color=RED)
    ax.annotate(f"Hopf, r = {meta['r_hopf']:.2f}", (meta["r_hopf"], xh), xytext=(-6, 8), textcoords="offset points",
                ha="right", fontsize=9, color=INK)
    ax.set_xlabel("r")
    ax.set_ylabel("x")
    ax.legend(loc="lower left", fontsize=9)
    save(fig, "t10", "lorenz",
         "Bifurcation diagram of the Lorenz system, x against r: a black line along x = 0 that turns dashed "
         "at r = 1, where two red branches split off; they turn dashed at the Hopf points near r = 24.7, "
         "from which brass lines of cycles bend back.",
         f"The equilibria of the Lorenz system and their stability, as continued by the page (solid: stable, "
         f"dashed: unstable): the origin, the pitchfork at r = 1, the branches C± and their subcritical Hopf "
         f"points at r = {meta['r_hopf']:.6f}, and the unstable cycles born there.")
    if orb is not None:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        cm = LinearSegmentedColormap.from_list("c", [BRASS, RED])
        norm = Normalize(rc.min(), rc.max())
        for j in range(orb.shape[0]):
            o = orb[j]
            ax.plot(np.append(o[:, 0], o[0, 0]), np.append(o[:, 2], o[0, 2]), color=cm(norm(rc[j])), lw=1.1)
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cm), ax=ax, fraction=0.04, pad=0.03)
        cb.set_label("r of the orbit")
        cb.outline.set_visible(False)
        c = np.sign(d["cplus_Z"][-1, 0]) * np.sqrt(8 / 3 * (meta["r_hopf"] - 1))   # the continued branch
        ax.plot(c, meta["r_hopf"] - 1, "+", color=INK, ms=9)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_aspect("equal", adjustable="datalim")
        save(fig, "t10", "cycles",
             "Nested closed loops in the (x, z) plane around the point C+, small ones near it and larger ones "
             "further out.",
             f"The periodic orbits of the cycle branch (orthogonal collocation, 40 intervals of degree 4) "
             f"for r from {rc.max():.3f} down to {rc.min():.3f} (colour bar): born at the Hopf point around the "
             "equilibrium of the continued branch (cross), they grow as r decreases.")
    fz, hz = d["fold_curve_Z"], d["hopf_curve_Z"]
    fn, hn = meta["fold_curve_names"], meta["hopf_curve_names"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(fz[:, fn["b2"]], fz[:, fn["b1"]], "o-", ms=3, color=RED, label="fold curve")
    ax.plot(hz[:, hn["b2"]], hz[:, hn["b1"]], "o-", ms=3, color=GREEN, label="Hopf curve")
    if "bt" in meta:      # the Hopf curve ends at the located BT event (its last kept point is before it)
        ax.plot([hz[-1, hn["b2"]], meta["bt"][1]], [hz[-1, hn["b1"]], meta["bt"][0]], "-", color=GREEN)
    if "bt" in meta:
        ax.plot(meta["bt"][1], meta["bt"][0], "o", ms=8, color=INK)
        ax.annotate("Bogdanov-Takens point", (meta["bt"][1], meta["bt"][0]), xytext=(10, 10),
                    textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlabel("β₂")
    ax.set_ylabel("β₁")
    ax.legend(loc="upper left")
    save(fig, "t10", "bogdanov_takens",
         "Two curves in the parameter plane: a red parabola of folds through the origin, and a green "
         "line of Hopf points at beta1 = 0 for beta2 < 0 that ends at the Bogdanov-Takens point at the "
         "origin.",
         "Codimension two: the fold and Hopf curves of the Bogdanov-Takens normal form, continued in "
         "(β₁, β₂). The fold curve follows β₁ = β₂²/4 through the BT point; the Hopf curve β₁ = 0 "
         "exists for β₂ < 0 and ends at the BT point.")


def _card(ax, x, y, w, h, text, edge=INK, lw=0.9, fill="none", size=9, weight="normal"):
    """A box with centred text in axes units (the registry diagram of tutorial 11)."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                transform=ax.transAxes, fc=fill, ec=edge, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=size, color=INK, linespacing=1.35, weight=weight)


def _arrow(ax, xy0, xy1, color=INK):
    ax.annotate("", xy=xy1, xytext=xy0, xycoords="axes fraction", textcoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.9, "shrinkA": 1, "shrinkB": 1})


def r_t11(d, meta, films=True):
    # The page teaches the registry, so the figure shows the registry: the records, what the query
    # returned, the reproduction and the cache reuse (presentation review of 0.1.1; the decay curves
    # of the heat equation it showed before taught the physics, not the task).
    recs, rep, cache = meta["records"], meta["reproduce"], meta["cache"]
    found = set(meta["found"])
    fig = plt.figure(figsize=(7.6, 3.1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    for x, title in ((0.02, "recorded runs"), (0.375, "query"), (0.68, "reproduce and reuse")):
        ax.text(x, 0.95, title, transform=ax.transAxes, fontsize=10, color=INK, weight="bold")
    W, H, ys = 0.29, 0.2, (0.66, 0.39, 0.12)
    pos = {}
    for r, y in zip(recs, ys, strict=False):
        cert = r["id"] in found
        _card(ax, 0.02, y, W, H, f"{r['name']}   κ = {r['kappa']:g}\nerror vs exp(−9κt): {r['rel_error']:.0e}\n"
                                f"status: {r['status']}", edge=RED if cert else SMOKE, lw=1.6 if cert else 0.9)
        pos[r["id"]] = y + H / 2
    qy, qh = 0.36, 0.3
    _card(ax, 0.375, qy, 0.25, qh, "find(kappa > 0.15,\nstatus = certified)\n"
                                  f"returns {len(found)} runs", edge=RED, lw=1.2)
    for rid in meta["found"]:
        _arrow(ax, (0.02 + W, pos[rid]), (0.375, qy + qh / 2), color=RED)
    orig = next(r for r in recs if r["id"] == rep["original"])
    same = "bit for bit identical" if rep["identical"] else "DIFFERENT"
    _card(ax, 0.66, 0.6, 0.33, 0.27, f"reproduce({orig['name']})\nnew run, end state\n{same}, "
                                    f"largest difference {rep['max_abs_diff']:g}", edge=INK)
    ax.annotate("", xy=(0.66, 0.735), xytext=(0.02 + W, pos[rep["original"]] + 0.06), xycoords="axes fraction",
                textcoords="axes fraction", arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 0.9,
                                                        "connectionstyle": "arc3,rad=-0.25"})
    steps = str(cache["computed"]).split(":")[-1].strip()
    _card(ax, 0.66, 0.1, 0.33, 0.27, f"same call with cache=True\ncache hit, {steps}:\n"
                                    "end state read from the record", edge=INK)
    four = next(r for r in recs if r["kappa"] == 0.4)
    _arrow(ax, (0.02 + W, pos[four["id"]]), (0.66, 0.235))
    save(fig, "t11", "registry",
         "A diagram of the registry: three run cards on the left, two of them outlined in red as certified, "
         "with arrows to a query box; on the right, a reproduction box reporting an identical end state and a "
         "cache box reporting that nothing was computed.",
         f"The registry of the page when the script ends. Each run records its parameters and its measured "
         f"error against the closed form exp(−9κt); the two certified ones (red) are what the query returns. "
         f"reproduce() runs {orig['name']} again and compares the end states: "
         f"{'identical' if rep['identical'] else 'different'}, largest difference {rep['max_abs_diff']:g}. A "
         f"repeated call with cache=True takes no step and reads the recorded state ({meta['nsteps']} steps in "
         f"the original).")


def r_t12(d, meta, films=True):
    P = d["path"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(P[:, 0], P[:, 2], color=INK, lw=0.35, alpha=0.9)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    save(fig, "t12", "attractor",
         "The butterfly-shaped Lorenz attractor drawn as one thin grey line looping around two centres.",
         "One trajectory of the Lorenz system at r = 28 (the page's solver, RK443, dt = 0.01), 80 time "
         "units after a transient, projected on (x, z).")
    logs, dt_i = d["logs"], meta["interval"]
    run = np.cumsum(logs, axis=0) / (dt_i * np.arange(1, logs.shape[0] + 1))[:, None, None]
    tt = dt_i * np.arange(1, logs.shape[0] + 1)
    ref = [0.906, 0.0, -14.572]
    nb = run.shape[1]
    fig, axs = plt.subplots(1, 3, figsize=(7.8, 2.7))
    for i, ax in enumerate(axs):
        lo, hi = np.percentile(run[:, :, i], [10, 90], axis=1)
        ax.fill_between(tt, lo, hi, color=SMOKE, alpha=0.35, lw=0, label="10–90 % of the starts")
        ax.plot(tt, run[:, :, i].mean(axis=1), color=RED, lw=1.4, label="mean over the starts")
        ax.axhline(ref[i], color=INK, ls="--", lw=0.9, label="published value")
        ax.set_title(f"λ{i + 1} = {d['exponents'][i]:.3f} ± {d['stderr'][i]:.1g}", fontsize=10)
        ax.set_xscale("log")
        ax.set_xlabel("averaging time")
        ax.set_ylabel(f"finite-time λ{i + 1}", fontsize=9)
        span = [0.6, 0.6, 2.0][i]
        ax.set_ylim(ref[i] - span, ref[i] + span)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "t12", "lyapunov",
         "Three panels, one per Lyapunov exponent: a grey band of running estimates over the starts narrows "
         "with time around a red mean that settles on the dashed published value.",
         f"Finite-time estimates of the three exponents, averaged from the start of the measurement, over the "
         f"{nb} starts of the page: the band holds the middle 80 % of the starts, the red line is their mean and "
         f"the dashed lines are the published values 0.906, 0 and −14.572. The panel titles give the final mean "
         f"and its standard error over the starts. Time is in the units of the Lorenz equations.")
    zk, zk1 = d["zk"], d["zk1"]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(zk, zk1, ".", ms=2.5, color=RED)
    lo, hi = zk.min() - 1, zk.max() + 1
    ax.plot([lo, hi], [lo, hi], color=SMOKE, lw=0.8, ls=":")
    ax.set_xlabel("$z_k$")
    ax.set_ylabel("$z_{k+1}$")
    ax.set_aspect("equal")
    save(fig, "t12", "lorenz_map",
         "Successive maxima of z plotted against each other: the dots lie on a thin tent-shaped curve "
         "above a dotted diagonal.",
         "The Poincaré section dz/dt = 0 of the page: each maximum of z against the next falls on a "
         "one-dimensional cusp map, as in Lorenz's Fig. 4 of 1963.")
    B, g = d["basin"], d["grid"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    from matplotlib.colors import ListedColormap
    ax.imshow(B, origin="lower", extent=[g[0], g[-1], g[0], g[-1]], cmap=ListedColormap([CAT2, LAB, RED]),
              vmin=-1, vmax=1, interpolation="nearest")
    c = np.sqrt(8 / 3 * 19)
    ax.plot([c], [c], "o", color=SILVER, mec=INK, ms=7)
    ax.plot([-c], [-c], "o", color=SILVER, mec=INK, ms=7)
    ax.annotate("C+", (c, c), xytext=(6, 6), textcoords="offset points", color=SILVER, fontsize=10)
    ax.annotate("C−", (-c, -c), xytext=(6, 6), textcoords="offset points", color=SILVER, fontsize=10)
    from matplotlib.patches import Patch
    ax.legend([Patch(color=RED), Patch(color=CAT2), Patch(facecolor=LAB, edgecolor=INK)],
              ["ends on C+", "ends on C−", "undecided at the horizon"], loc="upper center",
              bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9)
    ax.set_xlabel("x₀")
    ax.set_ylabel("y₀")
    save(fig, "t12", "basins",
         "A square map of starting points coloured red or blue by the equilibrium they end on, forming "
         "interleaved spiral bands, with C+ and C- marked and a legend below.",
         "Basins of C+ (red) and C− (blue) at r = 20 for the 2304 starts of the ensemble (z₀ = 19); the "
         "rare starts still undecided at the horizon are black. The map is symmetric under (x, y) → (−x, −y).")
    if not films:
        return
    cloud = d["cloud"]
    W, H = 640, 520
    frames = []
    for k, Xk in enumerate(cloud):
        fig = film_figure(W, H)
        ax = fig.add_axes([0.04, 0.04, 0.92, 0.86], facecolor=LAB)
        ax.plot(P[:, 0], P[:, 2], color="#3A3A38", lw=0.4)
        ax.scatter(Xk[:, 0], Xk[:, 2], s=4, color=DYE, lw=0)
        ax.set_xlim(-22, 22)
        ax.set_ylim(0, 52)
        ax.set_axis_off()
        fig.text(0.04, 0.94, f"t = {k * meta['cloud_every'] * meta['dt']:5.2f}", color=SILVER, fontsize=11, family="monospace")
        fig.text(0.96, 0.94, "600 starts within 0.001", color=SMOKE, fontsize=10, ha="right")
        frames.append(fig_to_rgb(fig))
        plt.close(fig)
    film(frames, "t12", "spread", poster=len(frames) - 1,
         alt="Six hundred yellow-green points start as one dot on the grey Lorenz attractor, travel together "
             "for a while, then stretch into a filament and spread over both wings.",
         caption="Sensitivity to initial conditions: 600 states within 10⁻³ of a point of the attractor, "
                 "advanced together as one ensemble by the page's solver, projected on (x, z).")


def r_t13(d, meta, films=True):
    # direct mode, adjoint mode and their overlap (the wavemaker) on one shared x axis: the product
    # is read off the two curves above it (presentation review of 0.1.1)
    x, S, q, qa = d["x"], d["S"], d["direct"], d["adjoint"]
    qn, qan = np.abs(q) / np.abs(q).max(), np.abs(qa) / np.abs(qa).max()
    span = x[np.maximum(qn, qan) > 1e-3]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 4.0), sharex=True, gridspec_kw={"height_ratios": [1.25, 1]})
    a1.plot(x, qn, color=RED, label="direct mode |q|")
    a1.plot(x, qan, "--", color=CAT2, label="adjoint mode |q†|")
    a1.set_ylabel("modulus, scaled to 1")
    a1.legend(loc="upper right", fontsize=9)
    a2.plot(x, S, color=INK, label="wavemaker S = |q†| |q|")
    a2.plot(meta["x_star"], meta["s_star"], "o", color=RED, label="maximum of S, at x ≈ 0")
    a2.set_xlim(span.min(), span.max())
    a2.set_xlabel("streamwise position x")
    a2.set_ylabel("S(x)")
    a2.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    rel = abs(meta["s_star"] - meta["S0_closed_form"]) / meta["S0_closed_form"]
    save(fig, "t13", "modes",
         "Top: two bell-shaped curves, the direct global mode peaking downstream (right) and the adjoint mode "
         "upstream (left), overlapping little. Bottom: their product, the wavemaker, a single bump centred at "
         "x = 0 with its maximum marked in red.",
         f"The direct and adjoint global modes of the Ginzburg-Landau model of the page (top) and the wavemaker "
         f"S(x) = |q†(x)| |q(x)| (bottom). Advection pushes the direct mode downstream and the adjoint "
         f"upstream; S is largest where they overlap, the region where a local change of the model moves the "
         f"eigenvalue most. Its maximum, at |x| < {_sci(max(abs(meta['x_star']), 1e-12))}, equals Chomaz's closed "
         f"form |χ|/√π to a relative {_sci(max(rel, 1e-17))}.")
    lam, om, lead = d["lam"], d["omega"], d["lead4"]
    errs = [float(np.min(np.abs(om - mu))) for mu in lam[lead]]
    fig, ax = plt.subplots(figsize=(7, 3.3))
    keep = lam.real > -3
    ax.scatter(lam[keep].real, lam[keep].imag, s=14, color=SMOKE, label="eigenvalues of the model")
    ax.scatter(lam[lead].real, lam[lead].imag, s=90, facecolors="none", edgecolors=RED, lw=1.4,
               label="the four modes in the data")
    ax.scatter(om.real, om.imag, s=40, marker="x", color=INK, label="DMD of 40 snapshots")
    for k, mu in enumerate(lam[lead], 1):
        ax.annotate(str(k), (mu.real, mu.imag), xytext=(7, 5), textcoords="offset points", fontsize=9, color=INK)
    complex_axes(ax, "growth rate  Re λ", "frequency  Im λ")
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "t13", "dmd",
         "Eigenvalues of the model in grey; four of them circled in red and numbered 1 to 4, each with a black "
         "cross from the decomposition of the data at its centre.",
         f"Dynamic mode decomposition of a transient made of four global modes (numbered by growth rate): the "
         f"DMD eigenvalues (crosses), computed from the snapshots alone, against the model's (red circles). "
         f"The distances, too small to see here, are "
         f"{', '.join(_sci(max(e, 1e-17)) for e in errs)} for modes 1 to 4.")


def r_t14(d, meta, films=True):
    nz, th, u = d["nz"], d["gap_th"], d["gap_u"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.semilogy(nz, th, "o-", color=RED, label="temperature θ")
    ax.semilogy(nz, u, "s--", color=CAT2, label="velocity u")
    ax.axhline(1e-12, color=INK, ls=":", lw=0.9, label="level checked at nz = 32")
    ax.set_xlabel("wall-normal modes nz")
    ax.set_ylabel("relative difference")
    ax.legend(loc="upper right")
    save(fig, "t14", "convergence",
         "The relative difference between the native model and the translated script on a log scale, "
         "falling steeply with the number of wall-normal modes down to rounding.",
         "Relative difference between the native model and the translated script after 40 SBDF2 steps, "
         "against nz. The two tau formulations of the same equations differ by an amount that falls "
         "spectrally down to rounding.")
    xn, zn = d["xn"], d["zn"]
    xf, zf, A = box_field(d["th_native"], xn, zn, 2.0, 0.0, 1.0, nx=240, nz=121)
    _, _, Bf = box_field(d["th_dedalus"], xn, zn, 2.0, 0.0, 1.0, nx=240, nz=121)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 2.6))
    M = np.abs(A).max()
    a1.imshow(A.T, origin="lower", extent=[xf[0], xf[0] + 2, 0, 1], cmap=SIGNED, vmin=-M, vmax=M, aspect="equal")
    a1.set_title("θ, native model", fontsize=10)
    D = np.abs(A - Bf) / M
    im = a2.imshow(D.T, origin="lower", extent=[xf[0], xf[0] + 2, 0, 1], cmap=SMOKE_CM, aspect="equal")
    a2.set_title("|native − translated| / max|θ|", fontsize=10)
    cb = fig.colorbar(im, ax=a2, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)
    for a in (a1, a2):
        a.set_xlabel("x")
        a.set_ylabel("z")
    fig.tight_layout()
    save(fig, "t14", "fields",
         "Left: the temperature deviation of the cell, one roll pair in red and blue. Right: the map of "
         "the difference between the two formulations, tiny everywhere.",
         "The temperature deviation after 40 steps at nz = 16 (left), and how far the translation of the "
         "script is from it (right), as a fraction of the largest value.")


def r_t15(d, meta, films=True):
    lam, err, cds = d["exponents"], d["stderr"], d["cds"]
    i = np.arange(1, lam.size + 1)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.errorbar(i, lam, yerr=err, fmt="o", ms=4, color=INK, capsize=2, label="this run")
    ax.plot(np.arange(1, cds.size + 1), cds, "_", ms=16, mew=2, color=RED,
            label=f"published, {cds.size} values (Cvitanović et al. 2010)")
    ax.axvline(meta["n_physical"] + 0.5, color=SMOKE, ls=":", lw=1)
    ax.text(meta["n_physical"] + 0.5, 1.0, "entangled | isolated", ha="center", va="bottom", fontsize=9, color=INK,
            transform=ax.get_xaxis_transform())
    ax.axhline(0, color=INK, lw=0.6, alpha=0.5)
    ax.set_xlabel("index i")
    ax.set_ylabel("Lyapunov exponent λᵢ")
    ax.set_xticks(i)
    ax.legend(loc="lower left")
    save(fig, "t15", "exponents",
         f"Twelve Lyapunov exponents with error bars, close to zero for the first eight and dropping near -2 "
         f"after; the first {cds.size} sit on red ticks of the published values.",
         f"The twelve leading exponents of Kuramoto-Sivashinsky at L = 22, with their standard errors. Red "
         f"ticks mark the {cds.size} published values (Cvitanović et al. 2010), so exponents {cds.size + 1} to "
         f"{lam.size} have no reference tick. The first eight are the entangled modes.")
    t4, t8 = d["theta4"].ravel(), d["theta8"].ravel()
    fig, ax = plt.subplots(figsize=(7, 3.0))
    bins = np.geomspace(max(min(t4.min(), t8.min()), 1e-4), np.pi / 2, 50)
    ax.hist(t4, bins=bins, color=RED, alpha=0.75, label="split 4 | 5 (inside the entangled modes)")
    ax.hist(t8, bins=bins, histtype="step", lw=1.6, color=CAT2, label="split 8 | 9 (entangled | isolated)")
    ax.set_xscale("log")
    ax.set_xlabel("smallest angle between the two subspaces (rad)")
    ax.set_ylabel("samples")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "t15", "angles",
         "Two histograms of angles on a log axis: the filled red one, inside the entangled modes, reaches "
         "very small angles; the blue outline, at the split after eight modes, stays well away from zero.",
         "Angles between the span of the first j covariant vectors and the rest, at each sample. Inside the "
         "entangled modes (j = 4) near-tangencies are frequent; at j = 8 the smallest angle stays away from "
         "zero over this run.")
    nu = d["nu"]
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    k = nu.shape[0]
    M = np.where(np.triu(np.ones_like(nu), 1) > 0, nu, np.nan)
    im = ax.imshow(M, cmap=LinearSegmentedColormap.from_list("v", [SILVER, RED]), vmin=0, vmax=max(np.nanmax(M), 1e-3),
                   extent=[0.5, k + 0.5, k + 0.5, 0.5])
    ax.axvline(meta["n_physical"] + 0.5, color=INK, lw=0.8, ls=":")
    ax.axhline(meta["n_physical"] + 0.5, color=INK, lw=0.8, ls=":")
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    ax.set_xticks(range(1, k + 1))
    ax.set_yticks(range(1, k + 1))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("P(Λᵢ < Λⱼ), τ = 5", fontsize=9)
    cb.outline.set_visible(False)
    save(fig, "t15", "domination",
         "A triangular matrix of probabilities: red squares among the first eight indices, near white "
         "elsewhere.",
         "Domination of Oseledets splitting: how often the local exponent i falls below j over windows of "
         "5 time units. Violations stay among the first eight modes: they form one entangled block.")
    U = d["spacetime"]
    L = meta["L"]
    Uf = fourier_interp(U, 256, axis=1)
    A = np.abs(Uf).max()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.imshow(Uf.T, origin="lower", aspect="auto", extent=[0, U.shape[0] * meta["dt_row"], 0, L], cmap=SIGNED,
              vmin=-A, vmax=A, interpolation="bilinear")
    ax.set_xlabel("time t (after a transient of 100)")
    ax.set_ylabel("x")
    save(fig, "t15", "spacetime",
         "A long space-time map of the chaotic field u(x, t), time across, with red and blue streaks that "
         "merge and split irregularly.",
         "One of the page's starts on the chaotic attractor: u(x, t) over 120 time units, with the model and "
         "numerics of the page.")


def r_t16(d, meta, films=True):
    paths = d["tg_paths"] % (2 * np.pi)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    s = np.linspace(0, 2 * np.pi, 200)
    XX, YY = np.meshgrid(s, s)
    ax.contour(XX, YY, np.sin(XX) * np.sin(YY), levels=np.linspace(-0.9, 0.9, 10), colors=SMOKE, linewidths=0.6)
    raw = d["tg_paths"]
    for p in range(paths.shape[1]):
        seg = paths[:, p, :]
        jump = np.any(np.abs(np.diff(raw[:, p, :], axis=0) - np.diff(seg, axis=0)) > 1, axis=1)
        xs, ys = seg[:, 0].copy(), seg[:, 1].copy()
        xs[1:][jump], ys[1:][jump] = np.nan, np.nan
        ax.plot(xs, ys, color=RED, lw=1.2)
        ax.plot(seg[-1, 0], seg[-1, 1], "o", ms=3, color=RED)
    ax.plot(d["X0"][:, 0], d["X0"][:, 1], "o", ms=3, mfc="none", mec=INK)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 2 * np.pi)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    save(fig, "t16", "tracers",
         "Square cells of grey streamlines with red tracer paths sliding along them, from open circles to "
         "red dots.",
         "The 32 tracers of the page in the decaying Taylor-Green vortices, t = 0 to 2 (dt = 0.025): each "
         "moves along a streamline of ψ = sin x sin y (grey) and slows down as the flow decays.")
    # three step sizes: two refinement ratios, a small check scoped to this flow (presentation review)
    e, dts = np.asarray(d["errors"], dtype=float), np.asarray(d["dts"], dtype=float)
    ratios = e[:-1] / e[1:]
    orders = np.log(ratios) / np.log(dts[:-1] / dts[1:])
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.loglog(dts, e, "o-", color=RED, label="largest tracer error")
    ax.loglog(dts, e[-1] * (dts / dts[-1]) ** 3, "--", color=INK, lw=0.9, label="slope 3, through the last point")
    ax.set_xticks(dts, [f"{v:g}" for v in dts])
    ax.minorticks_off()
    ax.set_xlabel("time step dt")
    ax.set_ylabel("error")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, "t16", "order",
         "The tracer error against the time step on log axes, three red points close to a dashed line of slope 3.",
         f"The largest tracer error against a reference path integrated to 10⁻¹³, for the three time steps of the "
         f"page. Halving dt divides it by {ratios[0]:.1f} then {ratios[1]:.1f}: observed orders "
         f"{orders[0]:.2f} and {orders[1]:.2f}"
         + (", close to 3 in this unsteady flow." if np.all(np.abs(orders - 3) < 0.3) else "."),
         width="60%")
    c, pos, t, xn, yn = d["c"], d["pos"], d["t"], d["xn"], d["yn"]

    def dye_img(k, nx=320, ny=161):
        xf, yf, F = box_field(c[k], xn, yn, 4.0, -1.0, 1.0, nx=nx, nz=ny)
        return xf, yf, F
    fig, axs = plt.subplots(3, 1, figsize=(7, 5.4))
    M = c[0].max()
    for ax, k in zip(axs, (0, 75, 150), strict=False):
        xf, yf, F = dye_img(k)
        ax.imshow(F.T, origin="lower", extent=[xf[0], xf[0] + 4, -1, 1], cmap=SMOKE_CM, vmin=0, vmax=M,
                  aspect="equal", interpolation="bilinear")
        p = pos[k]
        ax.plot(p[:, 0] % 4, p[:, 1], "o", ms=4, color=DYE, mec=LAB, mew=0.5)
        ax.set_xticks([])
        ax.set_yticks([-1, 1])
        ax.set_ylabel("y")
        ax.set_title(f"t = {t[k]:.1f}", fontsize=10)
    axs[-1].set_xticks([0, 1, 2, 3, 4])
    axs[-1].set_xlabel("x")
    fig.tight_layout()
    save(fig, "t16", "dye",
         "Three snapshots of a dye blob in a channel: round at first, then sheared into a tilted streak; "
         "small yellow-green tracers move along their heights.",
         f"The dye blob of the page in plane Couette flow at t = 0, 1.5 and 3, with the five tracers "
         f"(yellow-green). The dye is sheared and diffuses; its mass drifts by {meta['drift']:.0e} at most.")
    if not films:
        return
    W, H = 720, 400
    frames = []
    for k in range(len(c)):
        xf, yf, F = dye_img(k, nx=360, ny=181)
        fig = film_figure(W, H)
        ax = fig.add_axes([0.02, 0.06, 0.96, 0.84])
        ax.imshow(F.T, origin="lower", extent=[xf[0], xf[0] + 4, -1, 1], cmap=SMOKE_CM, vmin=0, vmax=0.6 * M,
                  aspect="equal", interpolation="bilinear")
        p = pos[k]
        ax.plot(p[:, 0] % 4, p[:, 1], "o", ms=6, color=DYE, mec=LAB, mew=0.6)
        ax.set_xlim(0, 4)
        ax.set_ylim(-1, 1)
        ax.set_axis_off()
        fig.text(0.02, 0.93, f"t = {t[k]:.2f}", color=SILVER, fontsize=11, family="monospace")
        fig.text(0.98, 0.93, "walls at ±1 move at ±1", color=SMOKE, fontsize=10, ha="right")
        frames.append(fig_to_rgb(fig))
        plt.close(fig)
    film(frames, "t16", "dye", poster=len(frames) // 2,
         alt="A round grey dye blob in a channel is sheared into a long tilted streak while five yellow-green "
             "tracers glide left and right at their own heights, those on the walls moving with them.",
         caption="The Couette block of the page from t = 0 to 3: the dye field and the five tracers, every "
                 "two time steps.")


RENDER = {k[2:]: v for k, v in dict(globals()).items() if k.startswith("r_t")}


def contact_sheet(out: Path) -> Path:
    from PIL import Image, ImageDraw
    tiles = []
    for e in (e for e in ENTRIES if e["kind"] != "table"):
        f = SITE / (e["poster"] if e["kind"] == "film" else e["file"])
        im = Image.open(f).convert("RGBA")
        bg = Image.new("RGBA", im.size, "#EFEAE0")
        im = Image.alpha_composite(bg, im).convert("RGB")
        im.thumbnail((420, 300))
        tile = Image.new("RGB", (440, 330), "#EFEAE0")
        tile.paste(im, ((440 - im.width) // 2, 4))
        ImageDraw.Draw(tile).text((8, 312), Path(e["file"]).name + ("  [film]" if e["kind"] == "film" else ""),
                                  fill="#151310")
        tiles.append(tile)
    cols = 5
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 440, rows * 330), "#EFEAE0")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 440, (i // cols) * 330))
    p = out / "contact_sheet.png"
    sheet.save(p)
    return p


def sync_pages(entries: list[dict]) -> list[str]:
    """Copy the alt text, width and caption of each figure of the manifest into its
    ```{figure} block on the tutorial pages, so a page says what the figure drawn from the data
    says (captions are computed from the plotted arrays). Returns the pages changed."""
    import re
    by_file = {Path(e["file"]).name: e for e in entries if e["kind"] == "figure"}
    changed = []
    block = re.compile(r"```\{figure\} media/(?P<f>[\w.]+)\n(?P<opts>(?::[\w-]+:.*\n)*)\n(?P<cap>(?:.+\n)+?)```")
    for page in sorted((SITE / "tutorials").glob("*.md")):
        text = page.read_text(encoding="utf-8")

        def fill(m):
            e = by_file.get(m.group("f"))
            if e is None or Path(e["file"]).stem in KEEP:
                return m.group(0)
            return (f"```{{figure}} media/{m.group('f')}\n:alt: {e['alt']}\n:width: {e.get('width', '100%')}\n\n"
                    f"{e['caption']}\n```")
        new = block.sub(fill, text)
        if new != text:
            page.write_text(new, encoding="utf-8")
            changed.append(page.name)
    return changed


def render(a) -> int:
    plt.style.use(STYLE)
    data = Path(a.data).resolve()
    MEDIA.mkdir(parents=True, exist_ok=True)
    FILMS.mkdir(parents=True, exist_ok=True)
    keys = [k for k in RENDER if (not a.only or k in a.only.split(",")) and (data / f"{k}.npz").exists()]
    manifest = MEDIA / "media.json"
    old = json.loads(manifest.read_text()) if manifest.exists() else []
    for k in keys:
        dd, meta = load(data, k)
        RENDER[k](dd, meta, films=not a.no_films)
    done = set(keys)
    kept = [e for e in old if e["tutorial"] not in done or (a.no_films and e["kind"] == "film")
            or Path(e["file"]).stem in KEEP]
    ENTRIES[:0] = kept
    ENTRIES.sort(key=lambda e: (e["tutorial"], e["kind"] == "film"))
    manifest.write_text(json.dumps(ENTRIES, indent=1, ensure_ascii=False) + "\n")
    pages = sync_pages(ENTRIES)
    print(f"[render] {len(ENTRIES)} entries in {manifest.relative_to(SITE.parents[1])}; pages synced: "
          f"{', '.join(pages) or 'none'}; contact sheet {contact_sheet(data)}", flush=True)
    return 0
