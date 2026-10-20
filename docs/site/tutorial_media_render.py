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

INK, RED, GREEN, BRASS, BLUE, SMOKE = "#77736B", "#C8463A", "#4F8A68", "#A8864F", "#6C7FA8", "#8A8A86"
LAB, SILVER, DYE = "#0B0B0B", "#EDEBE6", "#C6F03C"
# the two scales of the brand chart (docs/brand/guidelines.html, "Data"), as in web/gallery
SIGNED = LinearSegmentedColormap.from_list("signed", [(0.0, "#D4574A"), (0.30, "#6E2019"), (0.50, LAB),
                                                       (0.70, "#3F5245"), (1.0, "#9DBBA6")])
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


def save(fig, key: str, name: str, alt: str, caption: str) -> None:
    f = f"{key}_{name}.png"
    if f"{key}_{name}" in PRUNED or f"{key}_{name}_png" in PRUNED:
        plt.close(fig)
        return
    fig.savefig(MEDIA / f)
    plt.close(fig)
    small_png(MEDIA / f)
    ENTRIES.append({"file": f"tutorials/media/{f}", "tutorial": key, "kind": "figure", "alt": alt,
                    "caption": caption})
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
        ax.scatter(s[:1], [row], s=48, color=RED if s[0] > 0 else GREEN, zorder=4)
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
    ax.plot(k_c, Ra_c, "o", color=RED, ms=6)
    ax.annotate(f"onset  Ra$_c$ = {Ra_c:.3f},  k$_c$ = {k_c:.4f}", (k_c, Ra_c), xytext=(12, -16),
                textcoords="offset points", fontsize=10, color=INK)
    for Ra in (1500, 2500):
        ax.plot(np.pi, Ra, "x", color=INK, ms=7, mew=1.5)
    ax.annotate("the two Rayleigh numbers above, k = π", (np.pi, 2500), xytext=(10, 6),
                textcoords="offset points", fontsize=9, color=INK)
    ax.text(4.25, 2950, "convection grows", color=RED, fontsize=10, ha="center")
    ax.text(2.05, 1560, "the fluid at rest is stable", color=INK, fontsize=10, ha="left")
    ax.set_xlabel("wavenumber k")
    ax.set_ylabel("Rayleigh number Ra")
    save(fig, "t00", "neutral_curve",
         "The neutral curve in the plane of wavenumber and Rayleigh number: a U-shaped red line with "
         "its minimum marked at Ra about 1708 and k about 3.12; the region above it is shaded.",
         f"Growth rate of the least stable mode over a grid of (k, Ra), from the parametric pencil "
         f"of the onset block. The red line is where it vanishes; its minimum is the critical point "
         f"the script finds, Ra_c = {Ra_c:.3f} at k_c = {k_c:.4f}.")
    # 3. growth at the predicted rate, then saturation
    t, a, nu = d["t"], d["amp"], d["nu"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.1))
    a1.semilogy(t, a, color=INK, lw=1.4, label="simulation")
    j = np.searchsorted(t, 0.3)
    tl = np.linspace(0.0, 2.3, 50)
    a1.semilogy(tl, a[j] * np.exp(sig * (tl - t[j])), "--", color=RED, lw=1.2, label=f"eigenvalue, σ = {sig:.4f}")
    a1.set_ylim(5e-7, 1)
    a1.set_xlabel("time t")
    a1.set_ylabel("roll amplitude")
    a1.legend(loc="lower right")
    a2.plot(t, nu, color=INK, lw=1.4)
    a2.axhline(1, color=SMOKE, lw=0.8, ls=":")
    a2.text(5.9, 1.04, "conduction", color=INK, fontsize=9, ha="right")
    a2.text(5.9, nu[-1] + 0.04, f"Nu = {nu[-1]:.4f}", color=INK, fontsize=9, ha="right")
    a2.set_xlabel("time t")
    a2.set_ylabel("Nusselt number")
    fig.tight_layout()
    save(fig, "t00", "growth_saturation",
         "Left: the roll amplitude on a log scale rises along a straight dashed line, then levels off. "
         "Right: the Nusselt number stays at 1, rises to a peak and settles to a constant.",
         "The simulation of the page. Left: the rolls grow at the rate of the eigenvalue (dashed) "
         "until the nonlinear terms stop them. Right: the heat carried across the layer, in units of "
         "conduction, settles to a steady value.")
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
             "red column and a cool green one appear, grow and settle into two steady rolls, with "
             "isotherms bending around them.",
         caption=f"The simulation of the page from t = {st[0]:.1f} to {st[-1]:.1f}: departure of the "
                 "temperature from conduction (warm red, cool green) and isotherms, from states recorded every 25 time steps.")


def r_t01(d, meta, films=True):
    vals, i = d["values"], int(d["i"])
    c = vals[i]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.scatter(vals.real, vals.imag, s=16, color=SMOKE, zorder=3)
    ax.scatter([c.real], [c.imag], s=60, color=RED, zorder=4)
    ax.annotate(f"c = {c.real:.6f} + {c.imag:.6f} i", (c.real, c.imag), xytext=(12, 6),
                textcoords="offset points", fontsize=10, color=INK)
    ax.axhline(0, color=INK, lw=0.6, ls="--")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-1.0, 0.08)
    ax.set_xlabel("phase speed  Re c")
    ax.set_ylabel("growth  Im c")
    save(fig, "t01", "spectrum",
         "A few eigenvalues in the plane of phase speed and growth, all below the zero line except one red "
         "point just above it near Re c = 0.24.",
         "The modes kept by the drift test of the page (found again at N = 96), in the plane of phase speed "
         "and growth. Only one lies above the dashed line: the Tollmien-Schlichting wave, unstable at Re = 10⁴.")
    y, v = d["y"], d["v"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(y, v.real, color=RED, label="Re v")
    ax.plot(y, v.imag, color=GREEN, label="Im v")
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
    ax.semilogy(n[a > 0], a[a > 0], "o", ms=3.5, color=INK)
    ax.set_xlabel("Chebyshev index n")
    ax.set_ylabel("|coefficient|")
    save(fig, "t01", "coefficients",
         "The magnitudes of the 64 Chebyshev coefficients of the mode on a log scale: one sequence falls "
         "steadily from 1 to about 1e-12, the other stays near 1e-12.",
         "The Chebyshev coefficients of the mode (mode.coeffs(\"v\")). The even ones fall to about 10⁻¹² by "
         "the last index, the level of the odd ones, which v, even in y, does not need: 64 modes resolve it.")
    Re, al, G = d["Re"], d["alpha"], d["growth"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    RR, AA = np.meshgrid(Re, al)
    ax.contourf(RR, AA, G, levels=[0, np.inf], colors=[RED], alpha=0.14)
    ax.contour(RR, AA, G, levels=[0], colors=[RED], linewidths=1.8)
    ax.plot(meta["Re_c"], meta["alpha_c"], "o", color=RED)
    ax.annotate(f"Re$_c$ = {meta['Re_c']:.2f},  α$_c$ = {meta['alpha_c']:.5f}", (meta["Re_c"], meta["alpha_c"]),
                xytext=(10, -16), textcoords="offset points", fontsize=10, color=INK)
    ax.plot(1e4, 1.0, "x", color=INK, ms=7, mew=1.5)
    ax.annotate("Re = 10⁴, α = 1", (1e4, 1.0), xytext=(8, 6), textcoords="offset points", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Reynolds number Re")
    ax.set_ylabel("wavenumber α")
    save(fig, "t01", "neutral_curve",
         "The neutral curve of plane Poiseuille flow: a red tongue opening to the right in the plane of "
         "Reynolds number (log scale) and wavenumber, its nose marked at Re about 5772 and alpha about 1.02.",
         "Im c of the least stable mode on a grid of (Re, α), from the parametric pencil of the page. Inside "
         "the red line the flow is unstable; its nose is the critical point the script finds.")


def r_t02(d, meta, films=True):
    t, a, s = d["t"], d["amp"], meta["sigma_evp"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.semilogy(t, a, "o", ms=3, color=INK, label="simulation, every 10 steps")
    j = np.searchsorted(t, 0.3)
    ax.semilogy(t, a[j] * np.exp(s * (t - t[j])), color=RED, lw=1.2, label=f"EVP rate σ = {s:.7f}")
    ax.axvspan(0, 0.3, color=SMOKE, alpha=0.12, lw=0)
    ax.text(0.15, a.max() * 0.7, "faster modes\ndecay", ha="center", va="top", fontsize=9, color=INK)
    ax.set_xlabel("time t")
    ax.set_ylabel("roll amplitude")
    ax.legend(loc="lower right")
    save(fig, "t02", "growth",
         "The roll amplitude on a log scale against time: after a short shaded start the dots follow "
         "a straight red line.",
         f"The initial-value problem against the eigenvalue problem. After t = 0.3 the amplitude grows at "
         f"σ = {meta['sigma_ivp']:.7f}, the EVP's {s:.7f}: a relative difference of "
         f"{_sci(abs(meta['sigma_ivp'] - s) / abs(s))}, within the 10⁻⁵ the script checks.")
    kt, ke = d["ke_t"], d["ke"]
    fig, ax = plt.subplots(figsize=(7, 3.0))
    first = kt <= 0.3 + 1e-9
    ax.semilogy(kt[first], ke[first], "o-", ms=4, color=INK, label="first job, t = 0 to 0.3")
    ax.semilogy(kt[~first], ke[~first], "s-", ms=4, color=RED, label="next job, from the checkpoint")
    ax.axvline(0.3, color=SMOKE, ls=":", lw=1)
    ax.set_xlabel("time t")
    ax.set_ylabel("kinetic energy")
    ax.legend(loc="lower right")
    save(fig, "t02", "chain",
         "Kinetic energy against time on a log scale, black circles up to t = 0.3 and red squares after, "
         "continuing the same straight line.",
         "The kinetic energy written to rb.h5 by the two jobs of the chain: the second starts from the "
         "checkpoint of the first and continues the same file. One run, bit for bit.")
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
         "green). Right: its profile through the warmest column lies on the leading eigenfunction of the EVP.")


def r_t03(d, meta, films=True):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    f, o, a = d["full"], d["onset"], d["other"]
    keep = lambda v: v[(v.real > -160) & np.isfinite(v)]  # noqa: E731
    ax.scatter(keep(f).real, keep(f).imag, s=70, facecolors="none", edgecolors=SMOKE, lw=1.0,
               label=f"full space, n = {meta['n_full']}")
    ax.scatter(keep(o).real, keep(o).imag, s=16, color=RED, label=f"onset class, n = {meta['n_onset']}")
    ax.scatter(keep(a).real, keep(a).imag, s=22, marker="x", color=GREEN, label=f"other class, n = {meta['n_other']}")
    complex_axes(ax, "growth rate  Re σ", "frequency  Im σ")
    ax.set_ylim(-1, 1)
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "t03", "spectra",
         "Eigenvalues of the full problem as grey rings; each ring holds either a red dot (onset class) or "
         "a green cross (other class), and the rightmost, at zero, holds a red dot.",
         "Every eigenvalue of the full space (rings) is found in exactly one of the two symmetry classes, "
         "each solved with half the unknowns. The neutral mode at the onset point belongs to the onset class.")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.8))
    labels = ["full", "onset", "other"]
    n = [meta["n_full"], meta["n_onset"], meta["n_other"]]
    w = [meta["wall_full_ms"], meta["wall_onset_ms"], meta["wall_other_ms"]]
    cols = [SMOKE, RED, GREEN]
    a1.bar(labels, n, color=cols)
    a1.set_ylabel("unknowns")
    for x, v in zip(labels, n, strict=False):
        a1.text(x, v, f"{v}", ha="center", va="bottom", fontsize=9, color=INK)
    a2.bar(labels, w, color=cols)
    a2.set_ylabel("QZ time (ms)")
    for x, v in zip(labels, w, strict=False):
        a2.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=INK)
    fig.tight_layout()
    save(fig, "t03", "cost",
         "Two bar charts: the number of unknowns, 268 for the full space and 134 for each class; and the "
         "time of the dense eigenvalue solve, several times shorter for each class.",
         "Size and cost of each eigenproblem as measured in this run. Halving the unknowns makes the dense "
         "QZ solve several times cheaper (it scales as n³).")


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
    ax.plot(meta["lam_c"], 2 * np.log(np.cosh(a_star)), "o", ms=7, color=RED)
    ax.annotate(f"fold  λ$_c$ = {meta['lam_c']:.10f}", (meta["lam_c"], 2 * np.log(np.cosh(a_star))),
                xytext=(-14, 0), textcoords="offset points", ha="right", va="center", fontsize=10, color=INK)
    ax.set_xlabel("λ")
    ax.set_ylabel("u(0)")
    ax.set_xlim(0, 0.95)
    ax.set_ylim(0, max(u0.max() * 1.1, 2))
    ax.legend(loc="upper left")
    save(fig, "t04", "branch",
         "The solution branch of the Bratu problem: u(0) rises with lambda, turns back at a red fold near "
         "lambda = 0.878 and climbs on the upper branch, the dots lying on a thin closed-form curve.",
         "The branch followed by pseudo-arclength continuation (the page's settings, run for more points), "
         "on the closed form u(0) = 2 ln cosh a. The fold is located by bisection on the tangent.")
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
    r = d["res"]
    fig, ax = plt.subplots(figsize=(7, 2.9))
    ax.semilogy(np.arange(r.size), r, "o-", color=INK, ms=5)
    ax.axhline(1e-13, color=RED, ls="--", lw=1)
    ax.text(r.size - 1, 1.6e-13, "tol = 1e-13", color=RED, ha="right", fontsize=9)
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("relative residual")
    ax.set_xticks(np.arange(r.size))
    save(fig, "t04", "newton",
         "The relative residual of Newton's method on a log scale, dropping faster at each iteration "
         "below the dashed tolerance line.",
         "Newton from u = 0 at λ = 0.3, as printed by sol.explain(): the number of correct digits doubles "
         "at each step (quadratic convergence) down to the declared tolerance.")


def r_t05(d, meta, films=True):
    L = meta["L"]
    xf = np.arange(256) * L / 256
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for name, col in (("E1", RED), ("E2", GREEN), ("TW1", BRASS)):
        uf = fourier_interp(d[name], 256)
        lab = name + (f" (travels at c = {meta['c']:.5f})" if name == "TW1" else "")
        ax.plot(np.append(xf, L), np.append(uf, uf[0]), color=col, label=lab)
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
         "A space-time map of the orbit, x across and t up over two periods, with red and green streaks; "
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
    a1.set_title(f"Floquet multipliers, μ$_1$ = {meta['mu1']:.6f}", fontsize=10)
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
         "Stability of the solutions. Left: the multipliers of the orbit; the red pair outside the unit "
         "circle is the unstable direction, μ₁ = log|Λ|/T. Right: the leading eigenvalues of E1.")
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
             "fills a red and green space-time map from the bottom up over two periods of the orbit.",
         caption="The pre-periodic orbit over two periods: the state u(x) (top) and its space-time map "
                 "filling up (bottom). Every frame is a time step of the orbit's flow map.",
         closes=bool(closes))


def r_t08(d, meta, films=True):
    from scipy.special import jn_zeros
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for mm, col in zip(range(3), (RED, GREEN, BRASS), strict=False):
        s = d[f"drum_{mm}"]
        n = np.arange(1, s.size + 1)
        j2 = jn_zeros(mm, s.size) ** 2
        a1.plot(n, -s, "o", ms=5, color=col, label=f"m = {mm}")
        a1.plot(n, j2, "_", ms=12, mew=1.4, color=INK)
        a2.semilogy(n, np.maximum(np.abs(s + j2) / j2, 1e-17), "o-", ms=4, color=col)
    a1.set_xlabel("radial index n")
    a1.set_ylabel("−s")
    a1.legend(loc="upper left", fontsize=9)
    a1.set_title("eigenvalues, ticks at j²ₘ,ₙ", fontsize=10)
    a2.set_xlabel("radial index n")
    a2.set_ylabel("relative error")
    a2.set_title("distance to the Bessel zeros", fontsize=10)
    fig.tight_layout()
    save(fig, "t08", "drum",
         "Left: the computed eigenvalues of the drum for m = 0, 1, 2 as coloured dots, each on a black tick "
         "of the squared Bessel zeros. Right: their relative errors on a log scale, at rounding for the first "
         "modes and growing for the last ones.",
         "The vibrating drum at N = 32 radial modes: the eight slowest modes of each azimuthal group against "
         "the closed form −j²ₘ,ₙ. The first ones agree to rounding; the error grows with n as the modes "
         "become finer than the grid.")
    p = d["pipe"]
    p = p[np.isfinite(p)]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    keep = p.real > -1.2
    ax.scatter(p[keep].real, p[keep].imag, s=16, color=SMOKE)
    top = p[np.argsort(-p.real)[:2]]
    ax.scatter(top.real, top.imag, s=50, color=RED, zorder=3)
    for v in top:
        ax.annotate(f"{v.real:.6f} {'+' if v.imag >= 0 else '-'} {abs(v.imag):.6f} i", (v.real, v.imag),
                    xytext=(-10, 0), textcoords="offset points", ha="right", va="center", fontsize=9, color=INK)
    complex_axes(ax, "growth rate  Re s", "frequency  Im s")
    save(fig, "t08", "pipe_spectrum",
         "The spectrum of pipe flow at Re = 1000 for (m, k) = (1, 1): eigenvalues on three branches in the "
         "left half of the complex plane, the two rightmost marked red.",
         "Hagen-Poiseuille flow, R = 1000, (m, k) = (1, 1): the spectrum of the pipe block. All modes decay; "
         "the two least stable (red) are the centre and wall modes of Meseguer and Trefethen.")
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
    ax.annotate(f"Hopf, r = {meta['r_hopf']:.6f}", (meta["r_hopf"], xh), xytext=(-6, 8), textcoords="offset points",
                ha="right", fontsize=9, color=INK)
    ax.set_xlabel("r")
    ax.set_ylabel("x")
    ax.legend(loc="lower left", fontsize=9)
    save(fig, "t10", "lorenz",
         "Bifurcation diagram of the Lorenz system, x against r: a black line along x = 0 that turns dashed "
         "at r = 1, where two red branches split off; they turn dashed at the Hopf points near r = 24.7, "
         "from which brass lines of cycles bend back.",
         "The equilibria of the Lorenz system and their stability, as continued by the page: the origin, "
         "the pitchfork at r = 1, the branches C± and their subcritical Hopf points, and the cycles born there.")
    if orb is not None:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        cm = LinearSegmentedColormap.from_list("c", [BRASS, RED])
        for j in range(orb.shape[0]):
            o = orb[j]
            ax.plot(np.append(o[:, 0], o[0, 0]), np.append(o[:, 2], o[0, 2]), color=cm(j / max(orb.shape[0] - 1, 1)), lw=1.1)
        c = np.sign(d["cplus_Z"][-1, 0]) * np.sqrt(8 / 3 * (meta["r_hopf"] - 1))   # the continued branch
        ax.plot(c, meta["r_hopf"] - 1, "+", color=INK, ms=9)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_aspect("equal", adjustable="datalim")
        save(fig, "t10", "cycles",
             "Nested closed loops in the (x, z) plane around the point C+, small ones near it and larger ones "
             "further out.",
             f"The periodic orbits of the cycle branch (orthogonal collocation, 40 intervals of degree 4) "
             f"for r from {rc.max():.3f} down to {rc.min():.3f}: born at the Hopf point around the equilibrium of the continued branch (cross), "
             "they grow as r decreases.")
    fz, hz = d["fold_curve_Z"], d["hopf_curve_Z"]
    fn, hn = meta["fold_curve_names"], meta["hopf_curve_names"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(fz[:, fn["b2"]], fz[:, fn["b1"]], "o-", ms=3, color=RED, label="fold curve")
    ax.plot(hz[:, hn["b2"]], hz[:, hn["b1"]], "o-", ms=3, color=GREEN, label="Hopf curve")
    if "bt" in meta:
        ax.plot(meta["bt"][1], meta["bt"][0], "o", ms=8, color=INK)
        ax.annotate("Bogdanov-Takens point", (meta["bt"][1], meta["bt"][0]), xytext=(10, 10),
                    textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlabel("β₂")
    ax.set_ylabel("β₁")
    ax.legend(loc="upper left")
    save(fig, "t10", "bogdanov_takens",
         "Two curves in the parameter plane: a red parabola of folds and a green line of Hopf points at "
         "beta1 = 0, meeting at the Bogdanov-Takens point at the origin.",
         "Codimension two: the fold and Hopf curves of the Bogdanov-Takens normal form, continued in "
         "(β₁, β₂). The fold curve follows β₁ = β₂²/4, the Hopf curve β₁ = 0; both end at the BT point.")


def r_t11(d, meta, films=True):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    cols = {0.1: GREEN, 0.2: BRASS, 0.4: RED}
    for kappa, col in cols.items():
        t, a = d[f"t_{kappa}"], d[f"a_{kappa}"]
        ax.semilogy(t, np.exp(-9 * kappa * t), "-", color=SMOKE, lw=0.9)
        ax.semilogy(t[::4], a[::4], "o", ms=3.5, color=col, label=f"κ = {kappa}")
    ax.set_xlabel("time t")
    ax.set_ylabel("amplitude of sin 3x")
    ax.legend(loc="lower left")
    save(fig, "t11", "decay",
         "Three straight lines on a log scale, the decay of sin 3x for three diffusivities, the coloured dots "
         "on thin grey closed-form lines.",
         "The three recorded runs, sampled every five steps: each decays as exp(−9κt) (grey), the closed form "
         "the registry's results are checked against.")
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ks = sorted(meta["rel_error"], key=float)
    errs = [meta["rel_error"][k] for k in ks]
    cert = [float(k) in meta["certified"] for k in ks]
    ax.bar([f"heat_k{k}" for k in ks], errs, color=[RED if c else SMOKE for c in cert])
    for j, (e, c) in enumerate(zip(errs, cert, strict=False)):
        ax.text(j, e, ("certified\n" if c else "") + f"{e:.1e}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(min(errs) / 10, max(errs) * 100)
    ax.set_ylabel("relative error of the rate")
    save(fig, "t11", "certified",
         "A bar chart of the relative error of the measured decay rate for the three runs, all far below "
         "1e-9; two bars in red are labelled certified.",
         "The result stored with each run, and the two runs certified with that evidence: the ones "
         "cn.runs.find(kappa__gt=0.15, status=\"certified\") returns.")


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
    fig, axs = plt.subplots(1, 3, figsize=(7.8, 2.8))
    for i, ax in enumerate(axs):
        for b in range(run.shape[1]):
            ax.plot(tt, run[:, b, i], color=SMOKE, lw=0.5, alpha=0.5)
        ax.plot(tt, run[:, :, i].mean(axis=1), color=RED, lw=1.4)
        ax.axhline(ref[i], color=INK, ls="--", lw=0.9)
        ax.set_title(f"λ{i + 1} = {d['exponents'][i]:.3f} ± {d['stderr'][i]:.3f}", fontsize=10)
        ax.set_xscale("log")
        ax.set_xlabel("time")
        span = [0.6, 0.6, 2.0][i]
        ax.set_ylim(ref[i] - span, ref[i] + span)
    fig.tight_layout()
    save(fig, "t12", "lyapunov",
         "Three panels, one per Lyapunov exponent: grey running estimates of 32 trajectories converge with "
         "time onto their red mean and the dashed published value.",
         "Running estimates of the three exponents for the 32 starts of the page (grey), their mean (red) "
         "and the published values 0.906, 0 and −14.572 (dashed).")
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
    ax.imshow(B, origin="lower", extent=[g[0], g[-1], g[0], g[-1]], cmap=ListedColormap([GREEN, LAB, RED]),
              vmin=-1, vmax=1, interpolation="nearest")
    c = np.sqrt(8 / 3 * 19)
    ax.plot([c], [c], "o", color=SILVER, mec=INK, ms=7)
    ax.plot([-c], [-c], "o", color=SILVER, mec=INK, ms=7)
    ax.annotate("C+", (c, c), xytext=(6, 6), textcoords="offset points", color=SILVER, fontsize=10)
    ax.annotate("C−", (-c, -c), xytext=(6, 6), textcoords="offset points", color=SILVER, fontsize=10)
    ax.set_xlabel("x₀")
    ax.set_ylabel("y₀")
    save(fig, "t12", "basins",
         "A square map of starting points coloured red or green by the equilibrium they end on, forming "
         "interleaved spiral bands, with C+ and C- marked.",
         "Basins of C+ (red) and C− (green) at r = 20 for the 2304 starts of the ensemble (z₀ = 19); the "
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
    x, S, q, qa = d["x"], d["S"], d["direct"], d["adjoint"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(x, np.abs(q) / np.abs(q).max(), color=RED, label="direct mode |q|")
    ax.plot(x, np.abs(qa) / np.abs(qa).max(), color=GREEN, label="adjoint mode |q†|")
    ax.set_xlabel("x")
    ax.set_ylabel("modulus, scaled to 1")
    ax.legend(loc="upper left")
    save(fig, "t13", "modes",
         "Two bell-shaped curves: the direct global mode peaks downstream (right) and the adjoint mode "
         "upstream (left), barely overlapping.",
         "The direct and adjoint global modes of the Ginzburg-Landau model of the page. Advection pushes the "
         "direct mode downstream and the adjoint upstream: the operator is far from normal.")
    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.plot(x, S, color=INK)
    ax.plot(meta["x_star"], meta["s_star"], "o", color=RED)
    ax.annotate(f"max S = {meta['s_star']:.10f} at x = {meta['x_star']:.1e}", (meta["x_star"], meta["s_star"]),
                xytext=(10, -4), textcoords="offset points", fontsize=10, color=INK, va="top")
    ax.set_xlim(-30, 30)
    ax.set_xlabel("x")
    ax.set_ylabel("wavemaker S(x)")
    save(fig, "t13", "wavemaker",
         "A single bump centred at x = 0, the structural sensitivity, with its maximum marked in red.",
         "The wavemaker S(x) = |q†(x)| |q(x)|, where the overlap of the two modes is largest: the region "
         "where a local change of the model moves the eigenvalue most. The maximum matches Chomaz's closed form.")
    lam, om, lead = d["lam"], d["omega"], d["lead4"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    keep = lam.real > -3
    ax.scatter(lam[keep].real, lam[keep].imag, s=14, color=SMOKE, label="eigenvalues of the model")
    ax.scatter(lam[lead].real, lam[lead].imag, s=90, facecolors="none", edgecolors=RED, lw=1.4,
               label="the four modes in the data")
    ax.scatter(om.real, om.imag, s=40, marker="x", color=INK, label="DMD of 40 snapshots")
    complex_axes(ax, "growth rate", "frequency")
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "t13", "dmd",
         "Eigenvalues of the model in grey; four of them circled in red, each with a black cross from the "
         "decomposition of the data exactly at its centre.",
         "Dynamic mode decomposition of a transient made of four global modes: the DMD eigenvalues (crosses) "
         "land on the model's (red circles) to 10⁻⁹, from the snapshots alone.")


def r_t14(d, meta, films=True):
    nz, th, u = d["nz"], d["gap_th"], d["gap_u"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.semilogy(nz, th, "o-", color=RED, label="temperature θ")
    ax.semilogy(nz, u, "s-", color=GREEN, label="velocity u")
    ax.axhline(1e-12, color=INK, ls="--", lw=0.9)
    ax.text(nz[0], 1.5e-12, "check: below 1e-12 at nz = 32", fontsize=9, color=INK)
    ax.set_xlabel("wall-normal modes nz")
    ax.set_ylabel("relative difference")
    ax.legend(loc="upper right")
    save(fig, "t14", "convergence",
         "The relative difference between the native model and the translated Dedalus script on a log "
         "scale, falling steeply with the number of wall-normal modes down to rounding.",
         "Native model against the translated Dedalus script after 40 SBDF2 steps: two tau formulations of "
         "the same equations, whose difference falls spectrally with nz down to rounding.")
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
         "Left: the temperature deviation of the cell, one roll pair in red and green. Right: the map of "
         "the difference between the two formulations, tiny everywhere.",
         "The temperature deviation after 40 steps at nz = 16 (left), and how far the translation of the "
         "Dedalus script is from it (right), as a fraction of the largest value.")


def r_t15(d, meta, films=True):
    lam, err, cds = d["exponents"], d["stderr"], d["cds"]
    i = np.arange(1, lam.size + 1)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.errorbar(i, lam, yerr=err, fmt="o", ms=4, color=INK, capsize=2, label="this run")
    ax.plot(np.arange(1, cds.size + 1), cds, "_", ms=16, mew=2, color=RED, label="Cvitanović et al. (2010)")
    ax.axvline(meta["n_physical"] + 0.5, color=SMOKE, ls=":", lw=1)
    ax.text(meta["n_physical"] + 0.3, lam.min() * 0.95, "physical | isolated", ha="right", fontsize=9, color=INK)
    ax.axhline(0, color=INK, lw=0.6, alpha=0.5)
    ax.set_xlabel("index i")
    ax.set_ylabel("Lyapunov exponent λᵢ")
    ax.set_xticks(i)
    ax.legend(loc="lower left")
    save(fig, "t15", "exponents",
         "Twelve Lyapunov exponents with error bars, close to zero for the first eight and dropping near -2 "
         "after, each on a red tick of the published values.",
         "The twelve leading exponents of Kuramoto-Sivashinsky at L = 22, with their standard errors, on "
         "the published values. The first eight are the physical modes.")
    t4, t8 = d["theta4"].ravel(), d["theta8"].ravel()
    fig, ax = plt.subplots(figsize=(7, 3.0))
    bins = np.geomspace(max(min(t4.min(), t8.min()), 1e-4), np.pi / 2, 50)
    ax.hist(t4, bins=bins, color=RED, alpha=0.75, label="split 4 | 5 (inside the physical modes)")
    ax.hist(t8, bins=bins, color=GREEN, alpha=0.75, label="split 8 | 9 (physical | isolated)")
    ax.set_xscale("log")
    ax.set_xlabel("smallest angle between the two subspaces (rad)")
    ax.set_ylabel("samples")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "t15", "angles",
         "Two histograms of angles on a log axis: the red one, inside the physical modes, reaches very small "
         "angles; the green one, at the split after eight modes, stays well away from zero.",
         "Angles between the span of the first j covariant vectors and the rest, at every sample. Inside the "
         "physical modes (j = 4) near-tangencies are frequent; at j = 8 the split never closes.")
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
         "A long space-time map of the chaotic field u(x, t), time across, with red and green streaks that "
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
    e, dts = d["errors"], d["dts"]
    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.loglog(dts, e, "o-", color=RED, label="largest tracer error")
    ax.loglog(dts, e[-1] * (np.array(dts) / dts[-1]) ** 3, "--", color=INK, lw=0.9, label="slope 3")
    ax.set_xlabel("time step dt")
    ax.set_ylabel("error")
    ax.legend(loc="upper left")
    save(fig, "t16", "order",
         "The tracer error against the time step on log axes, three red points along a dashed line of slope 3.",
         "The tracer error against the reference path integrated to 10⁻¹³: third order in time, as stated for "
         "an unsteady flow.")
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
    for e in ENTRIES:
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
    kept = [e for e in old if e["tutorial"] not in done or (a.no_films and e["kind"] == "film")]
    ENTRIES[:0] = kept
    ENTRIES.sort(key=lambda e: (e["tutorial"], e["kind"] == "film"))
    manifest.write_text(json.dumps(ENTRIES, indent=1, ensure_ascii=False) + "\n")
    print(f"[render] {len(ENTRIES)} entries in {manifest.relative_to(SITE.parents[1])}; contact sheet "
          f"{contact_sheet(data)}", flush=True)
    return 0
