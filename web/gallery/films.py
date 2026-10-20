"""Website films: short looping videos of real Conatus runs (no mock-up, no decorative effect).

Four cases, each a time series of the solver's own state, drawn with the gallery colormaps
(brand_cmaps.py) and encoded as WebM (VP9), MP4 (H.264, fallback) and a WebP poster:

  * kelvin-helmholtz-film  passive dye between two tanh shear layers rolling up (same problem and
                           numerics as the gallery plate, web/gallery/kelvin_helmholtz.py);
  * rayleigh-benard-film   temperature of 2D convection at Ra = 1e6 after a spin-up from noise
                           (the equations of web/gallery/rayleigh_benard.py, a box of aspect 2);
  * swift-hohenberg-film   rolls growing from noise and coarsening, sampled on a logarithmic
                           clock (the equation of web/gallery/swift_hohenberg.py);
  * rossby-haurwitz-film   vorticity of the wavenumber-4 Rossby-Haurwitz wave on the rotating sphere
                           over one period (the problem of web/gallery/rossby_haurwitz.py).

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
SLUGS = {"kh": "kelvin-helmholtz-film", "rb": "rayleigh-benard-film", "sh": "swift-hohenberg-film",
         "rh": "rossby-haurwitz-film", "rt": "rayleigh-taylor-film", "tu": "turbulence-2d-film",
         "td": "turbulence-dye-film"}
# Films published on the site, in display order. Not shown, cases kept runnable: Rayleigh-Benard at
# Ra = 1e6 (too soft at this resolution), Swift-Hohenberg (too little visible change at site size),
# Rossby-Haurwitz and the 2D turbulence films (not chosen).
PUBLISHED = ("kelvin-helmholtz-film", "rayleigh-taylor-film")
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


def fade_black(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
    """Fade in from black over the first n frames and out to black over the last n: an honest
    restart for a film whose end state differs from its start (no cross-fade between them)."""
    w = np.ones(len(frames))
    ramp = np.arange(1, n + 1) / (n + 1)
    w[:n], w[-n:] = ramp, ramp[::-1]
    lab = C._rgb(C.LAB)
    return [np.clip(np.rint(lab + wi * (f.astype(float) - lab)), 0, 255).astype(np.uint8)
            for f, wi in zip(frames, w, strict=True)]


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


def finish(slug: str, out: Path, frames: list[np.ndarray], poster: np.ndarray, meta: dict,
           loop: str | None = None) -> None:
    """Encode and write <slug>.film.json. ``loop``: how the film closes on itself when it needs no
    cross-fade (a travelling wave filmed over exactly one period)."""
    facts = encode(frames if loop else loop_frames(frames), slug, out, poster)
    meta = dict(meta, slug=slug, video_webm=f"/films/{slug}.webm", video_mp4=f"/films/{slug}.mp4",
                poster=f"/films/{slug}.webp", loop=loop or f"run frames, then a {FADE_S:g} s "
                "cross-fade from the last frame to the first", run_id=os.environ.get("CN_RUN_ID"), **facts)
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
        "script": "web/gallery/films.py", "example_script": "web/gallery/kelvin_helmholtz.py",
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


# ---- Rossby-Haurwitz wave: vorticity on the rotating sphere -------------------------------------------
def rossby_haurwitz(a, out: Path) -> None:
    """The problem of web/gallery/rossby_haurwitz.py (Williamson et al. 1992, case 6: T63, RK443,
    dt = 0.03/Omega), filmed over one period of the wave: the run stops when the m = 4 phase of the
    vorticity at 45 deg N has turned by 2 pi (the pattern has moved one wavelength, 90 deg east), and
    the camera turns 90 deg west over the same frames, so the last frame joins the first."""
    import conatus.expert as CN
    import rossby_haurwitz as RHG
    from conatus.discretize import discretize
    from scipy.ndimage import map_coordinates

    rh = RHG.load_driver()
    s2 = CN.S2Coordinates("phi", "theta")
    bt = CN.SphereBasis(s2["theta"], a.rh_L + 1, radius=1.0, dealias=3 / 2)
    Np = 2 * (a.rh_L + 1)
    D = CN.Domain(s2, [CN.RealFourier(s2["phi"], Np, (0, 2 * np.pi), dealias=3 / 2), bt])
    u, h = CN.VectorField("u", D), CN.Field("h", D)
    Cz = CN.TensorField("Cz", CN.Domain(s2, [bt]), 2)
    pb = CN.IVP([u, h])                                       # same equations as the gallery plate
    pb.add_equation((CN.dt(u) + CN.grad(h) + CN.dot(Cz, u), -CN.dot(u, CN.grad(u))))
    pb.add_equation((CN.dt(h) + rh.H0 * CN.div(u), -CN.div((h - rh.H0) * u)))

    def cz(P, T):
        c = 2 * np.cos(T) + 0 * P
        return np.array([[0 * c, c], [-c, 0 * c]])
    plan = discretize(pb, ncc={Cz: cz})
    s = CN.IVPSolver(plan, "RK443", a.rh_dt)
    s.set_state({u: lambda P, T: np.array(rh.initial(P, T)[:2]), h: lambda P, T: rh.initial(P, T)[2]})
    print(s.explain(), flush=True)
    S = plan.space
    zeta = lambda: plan.evaluator.evaluate(CN.curl(u), {u: s.coeffs("u")}, {})   # noqa: E731
    i4 = list(S.ms).index(rh.RW)
    x45 = np.array([np.cos(np.pi / 4)])

    def phase(z):                                             # arg of the m = 4 part at 45 deg N
        return float(np.angle(S.H.values(rh.RW, 0, x45) @ z[i4])[0])
    # frame spacing from the non-divergent Haurwitz speed (an estimate; the stop uses the measured phase)
    R, W = rh.RW, rh.W
    nu = (R * (3 + R) * W - 2) / ((1 + R) * (2 + R))         # rad per unit time (units of 1/Omega)
    period_est = (2 * np.pi / R) / abs(nu) * 1.08            # the run is ~8 % slower (gallery plate)
    every = max(1, int(round(period_est / a.rh_dt / a.frames)))
    days_per_unit = 1.0 / (rh.OMEGA * 86400)
    t0 = time.perf_counter()
    s.run(int(round(a.rh_t0_days / days_per_unit / a.rh_dt)))   # the small-scale ripples set in first
    zs, ph = [zeta()], [phase(zeta())]
    turned = 0.0
    while True:
        s.run(every)
        z = zeta()
        check_finite(z, t=s.t)
        p_ = phase(z)
        turned += (p_ - ph[-1] + np.pi) % (2 * np.pi) - np.pi
        if abs(turned) >= 2 * np.pi - 1e-12:
            break
        zs.append(z)
        ph.append(p_)
        if len(zs) > 4 * a.frames:
            raise RuntimeError("the m = 4 phase did not turn by 2 pi: not a travelling wave here")
    n = len(zs)
    k = int(round(FADE_S * FPS))                              # frames past the period, for the seam
    zs.append(z)
    for _ in range(k - 1):
        s.run(every)
        zs.append(zeta())
        check_finite(zs[-1], t=s.t)
    wall = time.perf_counter() - t0
    period_days = n * every * a.rh_dt * days_per_unit
    ncol, nlon = 361, 720
    colat = np.linspace(0, np.pi, ncol)
    lon = 2 * np.pi * np.arange(nlon) / nlon
    Wpx, Hpx = a.width, a.width // 2
    rad = a.rh_radius * Hpx
    X = (np.arange(Wpx) + 0.5 - Wpx / 2) / rad
    Y = (Hpx / 2 - np.arange(Hpx) - 0.5) / rad
    XX, YY = np.meshgrid(X, Y)
    rho2 = XX ** 2 + YY ** 2
    inside = rho2 <= 1.0
    ZZ = np.sqrt(np.clip(1 - rho2, 0, None))
    p0 = np.radians(a.rh_view_lat)
    ex, ez, ey = ZZ * np.cos(p0) - YY * np.sin(p0), ZZ * np.sin(p0) + YY * np.cos(p0), XX
    c_lat = np.arccos(np.clip(ez, -1, 1))
    limb = np.clip(1.0 - np.abs(np.sqrt(rho2) - 1.0) * rad / 0.8, 0, 1)
    west = -np.sign(nu)                                       # the camera turns against the wave
    vmax = None
    frames = []
    for i, z in enumerate(zs):
        Z = RHG.synth(S, z, colat, lon)
        vmax = vmax or float(np.abs(Z).max())
        centre = west * (np.pi / 2) * i / n                   # longitude at the centre of the view:
        c_lon = np.mod(np.arctan2(ey, ex) + centre, 2 * np.pi)   # 90 deg over the film, joins frame 0
        wrap = np.concatenate([Z, Z[:, :1]], axis=1)
        f = map_coordinates(wrap, [(c_lat / np.pi * (ncol - 1)).ravel(),
                                   (c_lon / (2 * np.pi) * nlon).ravel()], order=3, mode="nearest")
        f = f.reshape(Hpx, Wpx)
        f[~inside] = np.nan
        rgb, _ = C.field_image(f, "signed", grain_sigma=0, vmax=vmax)
        frames.append(C.overlay_line(rgb, limb, color="#3A3A38", alpha=0.9))
    # seam: frames n..n+k-1 sit where frames 0..k-1 sit (one period later, camera one wavelength on);
    # they are cross-faded into those, so the film is frames k..n-1 then the k blended frames.
    blend = [np.clip(np.rint((1 - w) * frames[n + j].astype(float) + w * frames[j]), 0, 255).astype(np.uint8)
             for j, w in enumerate((np.arange(k) + 1) / (k + 1))]
    frames = frames[k:n] + blend
    finish(SLUGS["rh"], out, frames, frames[0], {
        "title": "Rossby–Haurwitz wave",
        "alt": "A turning globe on black covered by large alternating red and green vorticity lobes, "
               "four around each hemisphere, travelling east.",
        "caption": f"Relative vorticity of the wavenumber-4 Rossby–Haurwitz wave (Williamson et al. "
                   f"1992, case 6) in inviscid shallow water on the rotating sphere, over one period of "
                   f"the wave ({period_days:.1f} days, from day {a.rh_t0_days:g}), seen from {a.rh_view_lat:g}° N by a camera "
                   f"turning against it. T{a.rh_L}, RK443, dt = {a.rh_dt:g}/Ω, CPU.",
        "script": "web/gallery/films.py", "example_script": "benchmarks/spherical/rossby_haurwitz.py",
        "domain": "rotating", "colormap": "signed",
        "parameters": {"Lmax": a.rh_L, "R": rh.RW, "view_lat": a.rh_view_lat},
        "numerics": {"Nphi": Np, "dealias": 1.5, "scheme": "RK443", "dt": a.rh_dt, "backend": "cpu",
                     "steps": s.iteration, "steps_per_frame": every, "wall_s": wall},
        "period_days": period_days, "t0_days": a.rh_t0_days, "scale": {"vmax": vmax}},
        loop=f"one period of the wave (m = 4 phase measured), camera turned by one wavelength; the "
             f"last {k} frames are cross-faded with the same view one period earlier (the slowly "
             f"growing small-scale ripples differ)")


# ---- Rayleigh-Taylor: heavy fluid over light ------------------------------------------------------------
def rayleigh_taylor(a, out: Path) -> None:
    """Heavy (cold, T = -1/2) fluid over light (T = +1/2) across a thin tanh interface with a seeded
    multimode ripple, in the Boussinesq equations and walls of web/gallery/rayleigh_benard.py
    (thermal units, walls held at the far-field values). The film starts when the fastest
    plume reaches 5 % of the free-fall speed and spans --rt-span free-fall times."""
    import rayleigh_benard as RB
    from conatus.expert import IVPSolver

    plan, T = RB.rayleigh_benard(a.rt_Nx, a.rt_Nz, a.rt_Ra, 1.0, a.rt_Lx, 1.5)
    s = IVPSolver(plan, "RK443", a.rt_dt, backend="cpu", f_backend="fused")
    rng = np.random.default_rng(34)
    modes = [(2 * np.pi * j / a.rt_Lx, rng.uniform(0, 2 * np.pi), rng.normal()) for j in range(6, 41)]

    def T_init(X, Z):             # interface z = eta(x), a sum of 35 seeded modes
        eta = a.rt_amp * sum(c * np.cos(k * X + ph) for k, ph, c in modes) / np.sqrt(len(modes))
        return -0.5 * np.tanh((Z - eta) / a.rt_delta)
    s.set_state({T: T_init})
    print(s.explain(), flush=True)
    Zg = plan._as_grid(T, lambda X, Z: Z + 0 * X)[0]
    order = np.argsort(Zg)
    zg = Zg[order]
    xg = np.arange(a.rt_Nx) * a.rt_Lx / a.rt_Nx
    uff = np.sqrt(a.rt_Ra)                                  # free-fall speed (thermal units, Pr = 1)
    t0 = time.perf_counter()
    while True:                                             # linear phase: not filmed
        s.run(50)
        ug = s.grid("u")
        check_finite(ug, t=s.t)
        if np.abs(ug).max() > 0.05 * uff:
            break
        if s.t > 0.05:
            raise RuntimeError("no Rayleigh-Taylor growth by t = 0.05")
    t_start = float(s.t)
    every = max(1, int(round(a.rt_span / uff / a.frames / a.rt_dt)))
    fields, times, umax = [], [], 0.0
    for _ in range(a.frames):
        Tg = s.grid("T")
        check_finite(Tg, t=s.t)
        fields.append(Tg[:, order].copy())
        times.append(float(s.t))
        s.run(every)
        umax = max(umax, float(np.abs(s.grid("u")).max()))
    wall = time.perf_counter() - t0
    W, H = a.width, int(round(a.width / a.rt_Lx))
    frames = []
    for g in fields:
        px = C.resample(g.T, xg, zg, W, H, periodic_x=a.rt_Lx, y_range=(-0.5, 0.5))
        frames.append(C.field_image(px, "smoke", grain_sigma=0, vmin=-0.5, vmax=0.5)[0])
    poster = frames[len(frames) // 2]
    nf = int(round(0.3 * FPS))
    finish(SLUGS["rt"], out, fade_black(frames, nf), poster, {
        "title": "Rayleigh–Taylor instability",
        "alt": "Dark heavy fluid falls into pale light fluid in mushroom-shaped fingers that roll up and "
               "mix, while pale plumes rise.",
        "caption": f"Heavy fluid falling into light fluid, Ra {a.rt_Ra:.0e}".replace("e+0", "e")
                   + ", computed with Conatus.",
        "details": f"Boussinesq, Pr = 1, from a rippled interface; {(times[-1] - times[0]) * uff:.1f} "
                   f"free-fall times. {a.rt_Nx} × {a.rt_Nz} Fourier × Chebyshev, RK443, dt = "
                   f"{a.rt_dt:g}, CPU.",
        "script": "web/gallery/films.py",
        "example_script": "docs/site/tutorials/code/t02b_rayleigh_benard_gpu.py",
        "domain": "convection", "colormap": "smoke",
        "parameters": {"Ra": a.rt_Ra, "Pr": 1.0, "Lx": a.rt_Lx, "delta": a.rt_delta, "amp": a.rt_amp,
                       "seed": 34},
        "numerics": {"Nx": a.rt_Nx, "Nz": a.rt_Nz, "dealias": 1.5, "scheme": "RK443", "dt": a.rt_dt,
                     "backend": "cpu", "steps": s.iteration, "wall_s": wall,
                     "cfl_max": umax * a.rt_dt / (a.rt_Lx / a.rt_Nx)},
        "times": [t_start, times[-1], every * a.rt_dt]},
        loop=f"fade from black over the first {nf} frames and to black over the last {nf} (the "
             "mixed end state is not the start state)")


# ---- 2D turbulence: vorticity, and a dye released in it -----------------------------------------------------
def turbulence(a, out: Path) -> None:
    """Decaying 2D Navier-Stokes turbulence in a doubly periodic 2:1 box (the streamfunction form of
    web/gallery/kelvin_helmholtz.py, no base flow): random vorticity peaked at |k| = --tu-k0 with
    r.m.s. 1, run to --tu-t0 so that coherent vortices form, then filmed to --tu-t1 while they
    merge. At --tu-t0 a passive dye (Schmidt 1) is released in horizontal bands: the second film."""
    import kelvin_helmholtz as KH
    from conatus.expert import IVPSolver

    Lx, Lz, Nx, Nz = 4.0, 2.0, a.tu_Nx, a.tu_Nx // 2
    plan, psi, c = KH.build(Nx, Nz, Lx, 0.0, a.tu_nu, a.tu_nu, 1.5)
    rng = np.random.default_rng(34)
    kx = 2 * np.pi * np.fft.fftfreq(Nx, Lx / Nx)[:, None]
    kz = 2 * np.pi * np.fft.fftfreq(Nz, Lz / Nz)[None, :]
    k2 = kx ** 2 + kz ** 2
    kk = np.sqrt(k2)
    spec = kk * np.exp(-(kk / a.tu_k0) ** 2)                 # vorticity amplitude, peak near k0
    q = np.real(np.fft.ifft2(spec * np.exp(2j * np.pi * rng.random((Nx, Nz)))))
    q = (q - q.mean()) / q.std()                             # r.m.s. vorticity 1
    Q = np.fft.fft2(q)
    psi_g = np.real(np.fft.ifft2(np.where(k2 > 0, -Q / np.where(k2 > 0, k2, 1), 0)))
    u0 = np.real(np.fft.ifft2(1j * kz * np.fft.fft2(psi_g)))
    w0 = np.real(np.fft.ifft2(-1j * kx * np.fft.fft2(psi_g)))
    dx = Lx / Nx
    dt_ = a.tu_cfl * dx / float(np.sqrt(u0 ** 2 + w0 ** 2).max())   # declared: CFL of the initial field
    s = IVPSolver(plan, "RK443", dt_, backend="cpu", f_backend="fused")
    s.set_state({psi: lambda X, Z: psi_g})
    print(s.explain(), flush=True)
    print(f"dt = {dt_:.4g} (CFL {a.tu_cfl:g} of the initial max |u|)", flush=True)
    t0 = time.perf_counter()
    s.run(int(round(a.tu_t0 / dt_)))
    P = s.grid("psi")
    check_finite(P, t=s.t)

    def dye(X, Z):                                           # four pale bands, released now
        return 0.5 * (1 + np.tanh(3 * np.sin(4 * np.pi * Z / Lz))) + 0 * X
    s.set_state({psi: lambda X, Z: P, c: dye}, t=s.t)                     # same flow; RK443 has no history
    every = max(1, int(round((a.tu_t1 - a.tu_t0) / a.frames / dt_)))
    om, cs_, times, umax = [], [], [], 0.0
    for _ in range(a.frames):
        P = s.grid("psi")
        check_finite(P, t=s.t)
        o, uu, ww = KH.spectral_fields(P, Lx, Lz)
        om.append(-o)                                        # spectral_fields returns -lap(psi)
        cs_.append(s.grid("c").copy())
        times.append(float(s.t))
        umax = max(umax, float(np.sqrt(uu ** 2 + ww ** 2).max()))
        s.run(every)
    wall = time.perf_counter() - t0
    xg = np.arange(Nx) * Lx / Nx
    zg = -1 + np.arange(Nz) * Lz / Nz
    W, H = a.width, a.width // 2
    A = float(np.percentile(np.abs(np.stack(om[:10])), 99.5))
    fv, fd = [], []
    for o, cg in zip(om, cs_, strict=True):
        px = C.resample(o.T, xg, zg, W, H, periodic_x=Lx, periodic_y=Lz, y_range=(-1, 1))
        fv.append(C.field_image(px, "signed", grain_sigma=0, vmax=A)[0])
        pc = C.resample(cg.T, xg, zg, W, H, periodic_x=Lx, periodic_y=Lz, y_range=(-1, 1))
        rgb, _ = C.field_image(pc, "smoke", grain_sigma=0, vmin=0.0, vmax=1.0)
        fd.append(C.overlay_line(rgb, C.contour_mask(pc, 0.5, 1.2), alpha=0.85))
    common = {"script": "web/gallery/films.py", "example_script": "examples/shear/kelvin_helmholtz.py",
              "domain": "turbulence",
              "parameters": {"Lx": Lx, "Lz": Lz, "nu": a.tu_nu, "k0": a.tu_k0, "seed": 34,
                             "t0": a.tu_t0, "t1": a.tu_t1},
              "numerics": {"Nx": Nx, "Nz": Nz, "dealias": 1.5, "scheme": "RK443", "dt": dt_,
                           "backend": "cpu", "steps": s.iteration, "wall_s": wall,
                           "cfl_max": umax * dt_ / dx},
              "times": [times[0], times[-1], every * dt_]}
    res = f"{Nx} × {Nz} Fourier modes, RK443, dt = {dt_:.3g}, CPU."
    finish(SLUGS["tu"], out, fv, fv[len(fv) // 2], dict(common, colormap="signed",
        title="Two-dimensional turbulence",
        alt="Red and green vortices of both signs swirl on black, pair up and merge into fewer, "
            "larger ones, trailing thin filaments.",
        caption=f"Vorticity of decaying 2D turbulence in a periodic box, ν = {a.tu_nu:g}, t = "
                f"{times[0]:g} to {times[-1]:.0f} (in units of the initial r.m.s. vorticity). {res}"))
    finish(SLUGS["td"], out, fd, fd[len(fd) // 2], dict(common, colormap="smoke",
        title="Dye in two-dimensional turbulence",
        alt="Pale bands of dye are wound into spirals and thin filaments by swirling vortices, "
            "outlined by a thin yellow-green line.",
        caption=f"A passive dye released in bands at t = {times[0]:g} in the same decaying 2D "
                f"turbulence, stirred to t = {times[-1]:.0f}; the line is the half-concentration. {res}"))


# ---- collect into the site ---------------------------------------------------------------------------
FIELDS = ("slug", "title", "alt", "caption", "video_webm", "video_mp4", "poster", "width", "height",
          "duration_s", "run_id", "script", "example_script", "domain", "loop")   # loop: said in the caption


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
    # Rossby-Haurwitz (gallery plate values)
    ap.add_argument("--rh-L", type=int, default=63)
    ap.add_argument("--rh-dt", type=float, default=0.03)
    ap.add_argument("--rh-view-lat", type=float, default=35.0)
    ap.add_argument("--rh-radius", type=float, default=0.6, help="globe radius / film height")
    ap.add_argument("--rh-t0-days", type=float, default=4.0, help="days run before the first frame")
    # Rayleigh-Taylor
    ap.add_argument("--rt-Ra", type=float, default=1e7)
    ap.add_argument("--rt-Lx", type=float, default=2.0)
    ap.add_argument("--rt-Nx", type=int, default=512)
    ap.add_argument("--rt-Nz", type=int, default=256)
    ap.add_argument("--rt-dt", type=float, default=7.5e-7)
    ap.add_argument("--rt-delta", type=float, default=0.02, help="interface half-thickness")
    ap.add_argument("--rt-amp", type=float, default=0.01, help="r.m.s. interface ripple")
    ap.add_argument("--rt-span", type=float, default=6.0, help="free-fall times filmed")
    # 2D turbulence
    ap.add_argument("--tu-Nx", type=int, default=512, help="modes in x (Nz = Nx/2, box 4 x 2)")
    ap.add_argument("--tu-nu", type=float, default=5e-6)
    ap.add_argument("--tu-k0", type=float, default=30.0)
    ap.add_argument("--tu-cfl", type=float, default=0.25)
    ap.add_argument("--tu-t0", type=float, default=20.0)
    ap.add_argument("--tu-t1", type=float, default=65.0)
    a = ap.parse_args()
    if a.collect:
        return collect(a.collect)
    if a.case == "td":
        ap.error("the dye film is made by --case tu")
    if not a.case:
        ap.error("--case or --collect")
    out = Path(os.environ.get("CN_OUT", "."))
    out.mkdir(parents=True, exist_ok=True)
    {"kh": kelvin_helmholtz, "rb": rayleigh_benard, "sh": swift_hohenberg,
     "rh": rossby_haurwitz, "rt": rayleigh_taylor, "tu": turbulence}[a.case](a, out)


if __name__ == "__main__":
    main()
