"""Website films: short looping videos of real Conatus runs (no mock-up, no decorative effect).

Three cases, each a time series of the solver's own state, drawn with the gallery colormaps
(brand_cmaps.py) and encoded as WebM (VP9), MP4 (H.264, fallback) and a WebP poster:

  * kelvin-helmholtz-film  passive dye between two tanh shear layers rolling up (same problem and
                           numerics as the gallery plate, web/gallery/kelvin_helmholtz.py);
  * rayleigh-benard-film   temperature of 2D convection at Ra = 1e6 after a spin-up from noise
                           (the equations of web/gallery/rayleigh_benard.py, a box of aspect 2);
  * swift-hohenberg-film   rolls growing from noise and coarsening, sampled on a logarithmic
                           clock (the equation of web/gallery/swift_hohenberg.py).

Every frame is a state of the run at a recorded time; the loop closes with a short cross-fade
from the last frame to the first (a display transition, stated in the metadata). Numerics are
arguments with the values of the gallery scripts as defaults; nothing is changed silently.

Run (a job, never on the login node):


which runs `films.py --case kh|rb|sh` side by side and writes, in $CN_OUT, <slug>.webm,
<slug>.mp4, <slug>.webp (poster) and <slug>.film.json. Then copy them into the site (file copies
only, allowed on the login node):

    python web/gallery/films.py --collect <run_id>

which fills web/public/films/ and writes web/public/films/manifest.json (never edit it by hand).
The encoder is ffmpeg with libvpx-vp9 and libx264: $CN_FFMPEG, else `ffmpeg` on the PATH, else
the binary of the imageio-ffmpeg package.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brand_cmaps as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "web" / "public" / "films"
SLUGS = {"kh": "kelvin-helmholtz-film", "rb": "rayleigh-benard-film", "sh": "swift-hohenberg-film"}
# Films published on the site, in display order (the Rayleigh-Benard film at Ra = 1e6 is too soft
# at this resolution to show; its case stays runnable).
PUBLISHED = ("kelvin-helmholtz-film", "swift-hohenberg-film")
MAX_VIDEO_BYTES = 1_500_000          # per file (WebM and MP4), checked
FPS = 24
FADE_S = 0.5                         # loop cross-fade (last frame -> first frame)


# ---- encoding ---------------------------------------------------------------------------------------
def ffmpeg_exe() -> str:
    exe = os.environ.get("CN_FFMPEG") or shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:
        raise SystemExit("no ffmpeg: set CN_FFMPEG, put ffmpeg on the PATH or install imageio-ffmpeg") from e


def loop_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Frames of the run, then a cross-fade from the last one back to the first (closes the loop)."""
    n = int(round(FADE_S * FPS))
    a, b = frames[-1].astype(float), frames[0].astype(float)
    fade = [np.clip(np.rint((1 - w) * a + w * b), 0, 255).astype(np.uint8)
            for w in (np.arange(1, n + 1) / (n + 1))]
    return frames + fade


def encode(frames: list[np.ndarray], slug: str, out: Path, poster: np.ndarray) -> dict:
    """WebM (VP9) + MP4 (H.264) at FPS, no audio, no metadata, each < MAX_VIDEO_BYTES (the constant
    quality is lowered by steps until it fits, and recorded); WebP poster."""
    from PIL import Image

    exe = ffmpeg_exe()
    H, W = frames[0].shape[:2]
    facts = {}
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        for i, f in enumerate(frames):
            Image.fromarray(f, "RGB").save(Path(tmp) / f"f{i:05d}.png", compress_level=1)
        src = ["-framerate", str(FPS), "-i", str(Path(tmp) / "f%05d.png")]
        common = ["-an", "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact",
                  "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709",
                  "-colorspace", "bt709"]
        specs = {
            "webm": (34, lambda q: ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", str(q), "-row-mt", "1",
                                    "-deadline", "good", "-cpu-used", "1", "-g", str(len(frames))]),
            "mp4": (24, lambda q: ["-c:v", "libx264", "-preset", "slow", "-crf", str(q), "-profile:v",
                                   "high", "-movflags", "+faststart"]),
        }
        for ext, (q, opts) in specs.items():
            p = out / f"{slug}.{ext}"
            while True:
                subprocess.run([exe, "-y", "-loglevel", "error", *src, *opts(q), *common, str(p)], check=True)
                if p.stat().st_size < MAX_VIDEO_BYTES or q >= 51:
                    break
                q += 3
            if p.stat().st_size >= MAX_VIDEO_BYTES:
                raise RuntimeError(f"{p.name}: {p.stat().st_size} B >= {MAX_VIDEO_BYTES} B")
            facts[ext] = {"bytes": p.stat().st_size, "crf": q}
    pp = out / f"{slug}.webp"
    Image.fromarray(poster, "RGB").save(pp, "WEBP", quality=82, method=6)
    facts["poster"] = {"bytes": pp.stat().st_size}
    return {"width": W, "height": H, "fps": FPS, "frames": len(frames),
            "duration_s": round(len(frames) / FPS, 2), "files": facts}


def finish(slug: str, out: Path, frames: list[np.ndarray], poster: np.ndarray, meta: dict) -> None:
    facts = encode(loop_frames(frames), slug, out, poster)
    meta = dict(meta, slug=slug, video_webm=f"/films/{slug}.webm", video_mp4=f"/films/{slug}.mp4",
                poster=f"/films/{slug}.webp", loop=f"run frames, then a {FADE_S:g} s cross-fade "
                "from the last frame to the first", run_id=os.environ.get("CN_RUN_ID"), **facts)
    (out / f"{slug}.film.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    f = facts["files"]
    print(f"[films] {slug}: {facts['width']}x{facts['height']}, {facts['duration_s']} s, webm "
          f"{f['webm']['bytes'] / 1e3:.0f} kB (crf {f['webm']['crf']}), mp4 {f['mp4']['bytes'] / 1e3:.0f} kB "
          f"(crf {f['mp4']['crf']}), poster {f['poster']['bytes'] / 1e3:.0f} kB", flush=True)


def check_finite(*arrays, t: float) -> None:
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise FloatingPointError(f"non-finite state at t = {t:g}")


# ---- Kelvin-Helmholtz: dye ---------------------------------------------------------------------------
def kelvin_helmholtz(a, out: Path) -> None:
    import kelvin_helmholtz as KH
    from conatus.expert import IVPSolver

    Lz = 2.0
    nu = a.kh_delta / a.kh_Re
    plan, psi, c = KH.build(a.kh_Nx, a.kh_Nz, a.kh_Lx, a.kh_delta, nu, nu / a.kh_Sc, 1.5)
    s = IVPSolver(plan, "RK443", a.kh_dt, backend="cpu", f_backend="fused")
    psi0, c0, kx = KH.initial(a.kh_Lx, a.kh_delta, a.kh_n_mode, 1e-3, 2e-4, 34)
    s.set_state({psi: psi0, c: c0})
    print(s.explain(), flush=True)
    xg = np.arange(a.kh_Nx) * a.kh_Lx / a.kh_Nx
    zg = -1 + np.arange(a.kh_Nz) * Lz / a.kh_Nz
    every = int(round(a.kh_every / a.kh_dt))
    s.run(int(round(a.kh_t0 / a.kh_dt)))
    t0 = time.perf_counter()
    fields, times = [], []
    for _ in range(a.frames):
        cg = s.grid("c")
        check_finite(cg, t=s.t)
        fields.append(cg.copy())
        times.append(float(s.t))
        s.run(every)
    wall = time.perf_counter() - t0
    W, H = a.width, int(round(a.width * Lz / a.kh_Lx))
    frames = []
    for cg in fields:
        px = C.resample(cg.T, xg, zg, W, H, periodic_x=a.kh_Lx, periodic_y=Lz, y_range=(-1, 1))
        rgb, _ = C.field_image(px, "smoke", grain_sigma=0, vmin=0.0, vmax=1.0)
        frames.append(C.overlay_line(rgb, C.contour_mask(px, 0.5, 1.2), alpha=0.85))
    ip = int(np.argmin(np.abs(np.array(times) - a.kh_poster_t)))
    finish(SLUGS["kh"], out, frames, frames[ip], {
        "title": "Kelvin–Helmholtz billows",
        "alt": "A pale band of dye between two shear layers whose edges roll up into rows of spirals, "
               "outlined by a thin yellow-green line.",
        "caption": f"Passive dye between two shear layers rolling up, Re = {a.kh_Re:g}, t = "
                   f"{times[0]:g} to {times[-1]:.2f}. {a.kh_Nx} × {a.kh_Nz} Fourier modes, RK443, "
                   f"dt = {a.kh_dt:g}, CPU.",
        "script": "web/gallery/films.py", "example_script": "examples/shear/kelvin_helmholtz.py",
        "domain": "shear", "colormap": "smoke",
        "parameters": {"Lx": a.kh_Lx, "Lz": Lz, "delta": a.kh_delta, "Re": a.kh_Re, "Sc": a.kh_Sc,
                       "n_mode": a.kh_n_mode, "alpha_delta": kx * a.kh_delta, "seed": 34},
        "numerics": {"Nx": a.kh_Nx, "Nz": a.kh_Nz, "dealias": 1.5, "scheme": "RK443", "dt": a.kh_dt,
                     "backend": "cpu", "steps": s.iteration, "wall_s_frames": wall},
        "times": [times[0], times[-1], a.kh_every], "poster_t": times[ip]})


# ---- Rayleigh-Benard: temperature -------------------------------------------------------------------
def rayleigh_benard(a, out: Path) -> None:
    import rayleigh_benard as RB
    from conatus.expert import IVPSolver

    plan, T = RB.rayleigh_benard(a.rb_Nx, a.rb_Nz, a.rb_Ra, 1.0, a.rb_Lx, 1.5)
    s = IVPSolver(plan, "RK443", a.rb_dt, backend="cpu", f_backend="fused")
    rng = np.random.default_rng(34)
    modes = [(2 * np.pi * j / a.rb_Lx, rng.uniform(0, 2 * np.pi), rng.normal()) for j in range(1, 33)]

    def T_init(X, Z):             # conduction profile + seeded noise vanishing at the walls
        n = sum(amp * np.cos(k * X + ph) for k, ph, amp in modes) / np.sqrt(len(modes))
        return -Z + 1e-3 * n * np.cos(np.pi * Z)
    s.set_state({T: T_init})
    print(s.explain(), flush=True)
    Zg = plan._as_grid(T, lambda X, Z: Z + 0 * X)[0]
    order = np.argsort(Zg)
    zg = Zg[order]
    xg = np.arange(a.rb_Nx) * a.rb_Lx / a.rb_Nx
    t0 = time.perf_counter()
    spin = int(round(a.rb_t0 / a.rb_dt))
    for _ in range(10):           # spin-up in ten blocks, each checked
        s.run(spin // 10)
        check_finite(s.grid("T"), t=s.t)
    s.run(spin - 10 * (spin // 10))
    every = int(round(a.rb_every / a.rb_dt))
    fields, times = [], []
    for _ in range(a.frames):
        Tg = s.grid("T")
        check_finite(Tg, t=s.t)
        fields.append(Tg[:, order].copy())
        times.append(float(s.t))
        s.run(every)
    wall = time.perf_counter() - t0
    G = np.stack(fields)
    # grey range as in the gallery plate: [-A, A], A = 99th percentile of |T| in the interior
    # |z| < 0.4 (here over all frames, one range for the whole film)
    A = float(np.percentile(np.abs(G[:, :, np.abs(zg) < 0.4]), 99.0))
    W, H = a.width, int(round(a.width / a.rb_Lx))
    frames = []
    for g in fields:
        px = C.resample(g.T, xg, zg, W, H, periodic_x=a.rb_Lx, y_range=(-0.5, 0.5))
        frames.append(C.field_image(px, "smoke", grain_sigma=0, vmin=-A, vmax=A)[0])
    ff = np.sqrt(a.rb_Ra)
    finish(SLUGS["rb"], out, frames, frames[len(frames) // 2], {
        "title": "Rayleigh–Bénard convection",
        "alt": "Convecting fluid between two walls: bright hot plumes rise from the bottom and dark "
               "cold plumes sink from the top, swept sideways by large rolls.",
        "caption": f"Temperature in 2D convection at Ra = 1e{int(round(np.log10(a.rb_Ra)))}, Pr = 1, "
                   f"aspect ratio {a.rb_Lx:g}, rigid walls; {(times[-1] - times[0]) * ff:.0f} "
                   f"free-fall times after a spin-up from noise. {a.rb_Nx} × {a.rb_Nz} Fourier × "
                   f"Chebyshev, RK443, dt = {a.rb_dt:g}, CPU.",
        "script": "web/gallery/films.py",
        "example_script": "docs/site/tutorials/code/t02b_rayleigh_benard_gpu.py",
        "domain": "convection", "colormap": "smoke",
        "parameters": {"Ra": a.rb_Ra, "Pr": 1.0, "Lx": a.rb_Lx, "amp": 1e-3, "seed": 34},
        "numerics": {"Nx": a.rb_Nx, "Nz": a.rb_Nz, "dealias": 1.5, "scheme": "RK443", "dt": a.rb_dt,
                     "backend": "cpu", "steps": s.iteration, "wall_s": wall},
        "times": [times[0], times[-1], a.rb_every], "scale": {"vmin": -A, "vmax": A}})


# ---- Swift-Hohenberg: rolls from noise ---------------------------------------------------------------
def swift_hohenberg(a, out: Path) -> None:
    from conatus.expert import (IVP, CartesianCoordinates, Domain, Field, IVPSolver, Parameter,
                                RealFourier, discretize, dt, lap)

    Lx, Ly = 2 * np.pi * a.sh_nx, 2 * np.pi * a.sh_ny
    Nx, Ny = 8 * a.sh_nx, 8 * a.sh_ny
    cs = CartesianCoordinates("x", "y")
    x, y = cs.coords
    D = Domain(cs, [RealFourier(x, Nx, (0, Lx), dealias=2), RealFourier(y, Ny, (0, Ly), dealias=2)])
    u, r = Field("u", D), Parameter("r", a.sh_r)
    ivp = IVP([u], name="Swift-Hohenberg 2D")
    ivp.add_equation((dt(u) - r * u + (u + lap(u)) + lap(u + lap(u)), -u * u * u))
    s = IVPSolver(discretize(ivp, params={"r": a.sh_r}), "SBDF2", a.sh_dt)
    print(s.explain(), flush=True)
    rng = np.random.default_rng(34)                    # random plane waves, |k| in [0.9, 1.1]
    nxs, nys = np.meshgrid(np.arange(-a.sh_nx * 2, a.sh_nx * 2 + 1), np.arange(0, a.sh_ny * 2 + 1))
    kx, ky = 2 * np.pi * nxs / Lx, 2 * np.pi * nys / Ly
    sel = np.abs(np.hypot(kx, ky) - 1.0) < 0.1
    kx, ky = kx[sel], ky[sel]
    amp, ph = 1e-2 * rng.normal(size=kx.size), 2 * np.pi * rng.random(kx.size)
    s.set_state({u: lambda X, Y: sum(amp[j] * np.cos(kx[j] * X + ky[j] * Y + ph[j])
                                     for j in range(kx.size))})
    # logarithmic clock: frame times geometric from t0 to t1, rounded to whole steps
    steps = np.unique(np.rint(np.geomspace(a.sh_t0, a.sh_t1, a.frames) / a.sh_dt).astype(int))
    fields, times = [], []
    t0 = time.perf_counter()
    for n in steps:
        s.run(int(n - s.iteration))
        g = s.grid("u")
        check_finite(g, t=s.t)
        fields.append(g.copy())
        times.append(float(s.t))
    wall = time.perf_counter() - t0
    lim = float(max(np.abs(g).max() for g in fields))
    W, H = a.width, int(round(a.width * Ly / Lx))
    frames = []
    for g in fields:                                    # exact Fourier synthesis on the pixel grid
        Gh = np.fft.rfft2(g.T)
        P = np.zeros((H, W // 2 + 1), complex)
        hy = Ny // 2
        P[:hy, :Gh.shape[1] - 1] = Gh[:hy, :-1]
        P[-hy + 1:, :Gh.shape[1] - 1] = Gh[-hy + 1:, :-1]
        img = (np.fft.irfft2(P, s=(H, W)) * (H * W) / (Nx * Ny))[::-1]
        frames.append(C.field_image(img, "smoke", grain_sigma=0, vmin=-lim, vmax=lim)[0])
    ip = int(np.argmin(np.abs(np.log(np.array(times) / a.sh_poster_t))))
    finish(SLUGS["sh"], out, frames, frames[ip], {
        "title": "Swift–Hohenberg rolls",
        "alt": "Light and dark stripes appear out of grey noise, curve into a labyrinth and slowly "
               "straighten into patches of parallel rolls.",
        "caption": f"Rolls of du/dt = r u − (1 + ∇²)² u − u³ at r = {a.sh_r:g} growing from noise in a "
                   f"periodic box of {a.sh_nx} × {a.sh_ny} roll wavelengths, t = {times[0]:g} to "
                   f"{times[-1]:g} on a logarithmic clock. {Nx} × {Ny} Fourier modes, SBDF2, dt = "
                   f"{a.sh_dt:g}.",
        "script": "web/gallery/films.py", "example_script": "examples/patterns/swift_hohenberg.py",
        "domain": "patterns", "colormap": "smoke",
        "parameters": {"r": a.sh_r, "Lx": Lx, "Ly": Ly, "seed": 34},
        "numerics": {"Nx": Nx, "Ny": Ny, "dealias": 2, "scheme": "SBDF2", "dt": a.sh_dt,
                     "backend": "cpu", "steps": s.iteration, "wall_s": wall},
        "times": [times[0], times[-1], "geometric"], "poster_t": times[ip],
        "scale": {"vmin": -lim, "vmax": lim}})


# ---- collect into the site ---------------------------------------------------------------------------
FIELDS = ("slug", "title", "alt", "caption", "video_webm", "video_mp4", "poster", "width", "height",
          "duration_s", "run_id", "script", "example_script", "domain")


def collect(run_ids: list[str]) -> None:
    """Copy the films of finished runs to web/public/films/ and write manifest.json (display order:
    the order of PUBLISHED; other films are left out)."""
    PUBLIC.mkdir(parents=True, exist_ok=True)
    found = {}
    for rid in run_ids:
        run = ROOT / "scratch" / "output" / rid
        if json.loads((run / "STATUS.json").read_text()).get("exit_code") != 0:
            raise SystemExit(f"{rid}: non-zero exit code (not collected)")
        for p in sorted(run.glob("*.film.json")):
            m = json.loads(p.read_text())
            if m.get("run_id") != rid:
                raise SystemExit(f"{p}: run_id {m.get('run_id')} != {rid}")
            if m["slug"] not in PUBLISHED:
                continue
            for ext in ("webm", "mp4", "webp"):
                shutil.copyfile(run / f"{m['slug']}.{ext}", PUBLIC / f"{m['slug']}.{ext}")
            found[m["slug"]] = {k: m[k] for k in FIELDS}
    manifest = [found[s] for s in PUBLISHED if s in found]
    (PUBLIC / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    for e in manifest:
        print(f"{e['slug']:24s} {e['width']}x{e['height']} {e['duration_s']} s  {e['run_id']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", choices=sorted(SLUGS))
    ap.add_argument("--collect", nargs="+", metavar="RUN_ID", help="copy finished films into the site")
    ap.add_argument("--frames", type=int, default=160, help="frames of the run (before the loop fade)")
    ap.add_argument("--width", type=int, default=720, help="video width in pixels (height from the box)")
    # Kelvin-Helmholtz (gallery plate values)
    ap.add_argument("--kh-Nx", type=int, default=384)
    ap.add_argument("--kh-Nz", type=int, default=192)
    ap.add_argument("--kh-Lx", type=float, default=4.0)
    ap.add_argument("--kh-delta", type=float, default=0.08)
    ap.add_argument("--kh-Re", type=float, default=400.0)
    ap.add_argument("--kh-Sc", type=float, default=1.0)
    ap.add_argument("--kh-n-mode", type=int, default=4)
    ap.add_argument("--kh-dt", type=float, default=2.5e-3)
    ap.add_argument("--kh-t0", type=float, default=2.0, help="time of the first frame")
    ap.add_argument("--kh-every", type=float, default=0.05, help="time between frames")
    ap.add_argument("--kh-poster-t", type=float, default=6.0)
    # Rayleigh-Benard (gallery plate resolution per unit length, box of aspect 2)
    ap.add_argument("--rb-Ra", type=float, default=1e6)
    ap.add_argument("--rb-Lx", type=float, default=2.0)
    ap.add_argument("--rb-Nx", type=int, default=192)
    ap.add_argument("--rb-Nz", type=int, default=96)
    ap.add_argument("--rb-dt", type=float, default=1e-5)
    ap.add_argument("--rb-t0", type=float, default=0.02, help="spin-up (thermal times) before the first frame")
    ap.add_argument("--rb-every", type=float, default=6e-5)
    # Swift-Hohenberg (gallery plate values, box 24 x 12 wavelengths)
    ap.add_argument("--sh-r", type=float, default=0.1)
    ap.add_argument("--sh-nx", type=int, default=24)
    ap.add_argument("--sh-ny", type=int, default=12)
    ap.add_argument("--sh-dt", type=float, default=0.5)
    ap.add_argument("--sh-t0", type=float, default=10.0)
    ap.add_argument("--sh-t1", type=float, default=3000.0)
    ap.add_argument("--sh-poster-t", type=float, default=300.0, help="poster: the labyrinth stage")
    a = ap.parse_args()
    if a.collect:
        return collect(a.collect)
    if not a.case:
        ap.error("--case or --collect")
    out = Path(os.environ.get("CN_OUT", "."))
    out.mkdir(parents=True, exist_ok=True)
    {"kh": kelvin_helmholtz, "rb": rayleigh_benard, "sh": swift_hohenberg}[a.case](a, out)


if __name__ == "__main__":
    main()
