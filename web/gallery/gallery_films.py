"""Gallery films: one short looping film per gallery case, drawn from a real Conatus run.

Each case reruns the problem of its gallery script (web/gallery/<case>.py: same equations,
parameters, resolution, scheme and time step, imported from that script where it exposes a builder)
and keeps the solver's state at regular intervals. Every frame is a state of the run at a recorded
time, drawn with the colour map of the still (brand_cmaps.py) and one fixed colour range for the
whole film (written to the metadata). No frame is interpolated in time.

Cases (--case):
  kh     Kelvin-Helmholtz: vorticity and dye, two films from one run;
  rb     Rayleigh-Benard at Ra = 1e6 (aspect 4);
  ks     Kuramoto-Sivashinsky L = 22: the space-time diagram scrolling as time advances;
  sh     Swift-Hohenberg rolls growing from noise to the labyrinth of the still (t = 0 to 600);
  rh     Rossby-Haurwitz wave over one period of the wave, fixed camera;
  os     Orr-Sommerfeld (Tollmien-Schlichting) mode: Re(u_hat(y) e^{i alpha (x - c_r t)});
  tc     Taylor-Couette: Taylor vortices growing above onset (the still's mode is steady).
The cases of the gallery's development-only flows (pipe modes, spherical shell, turbulent channel) are
in gallery_films_private.py, next to this file when it is present; --case runs them the same way.
For the linear modes (os, and pipe in the private cases) the time dependence is the mode's own exp(s t): the computed
eigenvector multiplied by exp(i Im(s) t), with the exponential growth or decay exp(Re(s) t) left
out (said in the caption); one period of the mode closes the loop exactly. The other films end
in a state that is not their first one: they fade from and to black where they loop (a display
transition, said in the caption).

Every film is written in $CN_OUT as <slug>.webm (VP9) and <slug>.mp4 (H.264) at the desktop width,
<slug>-800.webm and <slug>-800.mp4 at 800 px, <slug>-poster.webp and <slug>-poster-800.webp, and
<slug>.film.json (what is shown, the colour range, the simulated times, the run's diagnostics).


then list the run in web/gallery/film_runs.txt and collect (file copies only):

    python web/gallery/collect.py
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import brand_cmaps as C  # noqa: E402

FPS = 24
FADE_FRAMES = 7                       # 0.3 s fade from / to black at the loop point
BUDGET = {"": 4_500_000, "-800": 1_500_000}    # bytes per video file (each of WebM and MP4)
MOBILE_W = 800


# ---- encoding ---------------------------------------------------------------------------------------
def ffmpeg_exe() -> str:
    exe = os.environ.get("CN_FFMPEG") or shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


class Film:
    """Frames written to disk as they are drawn (PNG), then encoded. ``fade``: fade from and to black
    over FADE_FRAMES at both ends (the run's end state differs from its start state)."""

    def __init__(self, slug: str, out: Path, n: int, fade: bool):
        self.slug, self.out, self.n, self.fade = slug, out, n, fade
        self.tmp = Path(tempfile.mkdtemp(dir=out, prefix=f".{slug}-"))
        self.i, self.poster, self.size = 0, None, None

    def add(self, rgb: np.ndarray, poster: bool = False) -> None:
        from PIL import Image
        if poster:
            self.poster = rgb.copy()
        w = 1.0
        if self.fade:
            k = FADE_FRAMES
            if self.i < k:
                w = (self.i + 1) / (k + 1)
            elif self.i >= self.n - k:
                w = (self.n - self.i) / (k + 1)
        f = rgb if w == 1.0 else np.clip(np.rint(C._rgb(C.LAB) + w * (rgb.astype(float) - C._rgb(C.LAB))),
                                         0, 255).astype(np.uint8)
        self.size = f.shape[1], f.shape[0]
        Image.fromarray(f, "RGB").save(self.tmp / f"f{self.i:05d}.png", compress_level=1)
        self.i += 1

    def encode(self) -> dict:
        """WebM (VP9) and MP4 (H.264) at the drawn width and at 800 px, no audio, no metadata; the
        constant quality is lowered by steps until each file is under its budget (recorded)."""
        from PIL import Image
        if self.i != self.n:
            raise RuntimeError(f"{self.slug}: {self.i} frames drawn, {self.n} declared")
        exe = ffmpeg_exe()
        W, H = self.size
        threads = os.environ.get("OMP_NUM_THREADS", "4")
        src = ["-framerate", str(FPS), "-i", str(self.tmp / "f%05d.png")]
        common = ["-an", "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact",
                  "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709",
                  "-colorspace", "bt709", "-threads", threads]
        specs = {
            "webm": (31, lambda q: ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", str(q), "-row-mt", "1",
                                    "-deadline", "good", "-cpu-used", "2", "-g", str(self.n)]),
            "mp4": (21, lambda q: ["-c:v", "libx264", "-preset", "slow", "-crf", str(q), "-profile:v",
                                   "high", "-movflags", "+faststart"]),
        }
        facts = {}
        for suffix, width in (("", W), ("-800", MOBILE_W)):
            hh = int(round(H * width / W / 2)) * 2
            scale = [] if width == W else ["-vf", f"scale={width}:{hh}:flags=lanczos"]
            for ext, (q, opts) in specs.items():
                p = self.out / f"{self.slug}{suffix}.{ext}"
                while True:
                    subprocess.run([exe, "-y", "-loglevel", "error", *src, *scale, *opts(q), *common, str(p)],
                                   check=True)
                    if p.stat().st_size < BUDGET[suffix] or q >= 51:
                        break
                    q += 3
                if p.stat().st_size >= BUDGET[suffix]:
                    raise RuntimeError(f"{p.name}: {p.stat().st_size} B over the budget {BUDGET[suffix]} B")
                facts[f"{ext}{suffix}"] = {"bytes": p.stat().st_size, "crf": q, "width": width, "height": hh}
        im = Image.fromarray(self.poster, "RGB")
        for suffix, width in (("", W), ("-800", MOBILE_W)):
            p = self.out / f"{self.slug}-poster{suffix}.webp"
            (im if width == W else im.resize((width, int(round(H * width / W))), Image.LANCZOS)).save(
                p, "WEBP", quality=80, method=6)
            facts[f"poster{suffix}"] = {"bytes": p.stat().st_size}
        shutil.rmtree(self.tmp)
        return {"width": W, "height": H, "fps": FPS, "frames": self.n,
                "duration_s": round(self.n / FPS, 3), "files": facts}


def finish(film: Film, meta: dict) -> None:
    """Encode and write <slug>.film.json (the run id is the job's)."""
    facts = film.encode()
    loop = (f"fade from black over the first {FADE_FRAMES} frames and to black over the last "
            f"{FADE_FRAMES} (the end state is not the start state)") if film.fade else meta.pop("loop")
    meta = dict(meta, slug=film.slug, loop=loop, run_id=os.environ.get("CN_RUN_ID"), **facts)
    (film.out / f"{film.slug}.film.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    f = facts["files"]
    print(f"[films] {film.slug}: {facts['width']}x{facts['height']}, {facts['duration_s']} s, "
          + ", ".join(f"{k} {v['bytes'] / 1e3:.0f} kB" for k, v in f.items()), flush=True)


def check_finite(*arrays, t: float) -> None:
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise FloatingPointError(f"non-finite state at t = {t:g}")


def g4(x: float) -> str:
    return f"{x:.4g}"


# ---- Kelvin-Helmholtz: vorticity and dye (web/gallery/kelvin_helmholtz.py, gallery run values) -----------
def kelvin_helmholtz(a, out: Path) -> None:
    import kelvin_helmholtz as KH
    from conatus.expert import IVPSolver

    Lx, Lz, Nx, Nz, delta, Re, dt_ = 4.0, 2.0, 384, 192, 0.08, 400.0, 2.5e-3
    nu = delta / Re
    plan, psi, c = KH.build(Nx, Nz, Lx, delta, nu, nu, 1.5)
    s = IVPSolver(plan, "RK443", dt_, backend="cpu", f_backend="fused")
    psi0, c0, kx = KH.initial(Lx, delta, 4, 1e-3, 2e-4, 34)
    s.set_state({psi: psi0, c: c0})
    print(s.explain(), flush=True)
    every = int(round(a.kh_every / dt_))          # steps between frames
    sample = int(round(0.1 / dt_))                # growth-rate sampling, as the gallery run
    first = int(round(a.kh_t0 / dt_))
    n = int(round((a.kh_t1 - a.kh_t0) / a.kh_every)) + 1
    om, cs, times, ts, amps = [], [], [], [], []
    t0 = time.perf_counter()
    k = 0
    while len(times) < n:
        if k and k % sample == 0:
            P = s.grid("psi")
            check_finite(P, t=s.t)
            ts.append(float(s.t))
            amps.append(float(np.abs(np.fft.rfft(P, axis=0)[4]).max() / Nx))
        if k >= first and (k - first) % every == 0:
            P = s.grid("psi")
            check_finite(P, t=s.t)
            om.append(KH.spectral_fields(P, Lx, Lz)[0].astype(np.float32))
            cs.append(s.grid("c").astype(np.float32))
            times.append(float(s.t))
        step = min(sample - k % sample, every - (k - first) % every if k >= first else first - k)
        s.run(step)
        k += step
    wall = time.perf_counter() - t0
    ts, A = np.array(ts), np.array(amps)        # growth rate: the gallery script's fit window
    win = (A > 3 * A[0]) & (A < 300 * A[0])
    stop = np.argmax(A > 300 * A[0]) if np.any(A > 300 * A[0]) else len(A)
    win[stop:] = False
    growth = float(np.polyfit(ts[win], np.log(A[win]), 1)[0])
    xg = np.arange(Nx) * Lx / Nx
    zg = -1 + np.arange(Nz) * Lz / Nz
    W, H = a.width, a.width // 2
    amp = float(np.percentile(np.abs(np.stack(om)), 99.7))
    ip = int(np.argmin(np.abs(np.array(times) - 7.0)))
    span = f"t = {times[0]:g} to {times[-1]:g}"
    speed = a.kh_every * FPS
    diag = {"growth_rate_mode": growth, "growth_rate_times_delta": growth * delta,
            "growth_window": [float(ts[win].min()), float(ts[win].max())], "wall_s": wall}
    common = {"script": "web/gallery/gallery_films.py", "case_script": "web/gallery/kelvin_helmholtz.py",
              "example_script": "examples/shear/kelvin_helmholtz.py", "geometry": "cartesian",
              "domain": "shear", "times": [times[0], times[-1], a.kh_every], "poster_t": times[ip],
              "parameters": {"Lx": Lx, "Lz": Lz, "delta": delta, "Re": Re, "Sc": 1.0, "n_mode": 4,
                             "alpha_delta": kx * delta, "amp": 1e-3, "noise": 2e-4, "seed": 34},
              "numerics": {"Nx": Nx, "Nz": Nz, "dealias": 1.5, "scheme": "RK443", "dt": dt_,
                           "backend": "cpu", "steps": s.iteration, "wall_s": wall},
              "diagnostics": diag}
    res = f"{Nx} × {Nz} RealFourier modes (3/2 dealiasing), streamfunction form, RK443, dt = {dt_:g}, CPU"
    fv = Film("kelvin-helmholtz-vorticity", out, n, fade=True)
    fd = Film("kelvin-helmholtz-dye", out, n, fade=True)
    for i, (o, cg) in enumerate(zip(om, cs, strict=True)):
        px = C.resample(o.T, xg, zg, W, H, periodic_x=Lx, periodic_y=Lz, y_range=(-1, 1))
        fv.add(C.field_image(px, "signed", grain_sigma=0, vmax=amp)[0], poster=i == ip)
        pc = C.resample(cg.T, xg, zg, W, H, periodic_x=Lx, periodic_y=Lz, y_range=(-1, 1))
        rgb = C.field_image(pc, "smoke", grain_sigma=0, vmin=0.0, vmax=1.0)[0]
        fd.add(C.overlay_line(rgb, C.contour_mask(pc, 0.5, 1.2), alpha=0.85), poster=i == ip)
    finish(fv, dict(common, title="Kelvin–Helmholtz billows", colormap="signed",
                    scale={"kind": "signed", "lo": -amp, "hi": amp},
                    alt="Two horizontal shear layers on black roll up into rows of spiral vortices, the upper "
                        "row red (negative vorticity), the lower row green (positive), trailing thin "
                        "filaments that wrap around each core.",
                    field=f"Vorticity ω = −∇²ψ (normal to the plane) of two shear layers rolling up, {span}.",
                    colour=f"Colour: ω, red for negative, black at zero, glass green for positive; one scale "
                           f"for the whole film, ending at the 99.7th percentile of |ω| over all frames "
                           f"({amp:.3g}; larger values saturate).",
                    extent="Periodic box 4 × 2 (x × z, lengths in units of the layer velocity scale), the full box.",
                    caption=f"Vorticity of two tanh shear layers (half-thickness δ = {delta:g}, Re = Uδ/ν = "
                            f"{Re:g}), nonlinear IVP seeded by the box mode nearest the most unstable one "
                            f"(αδ = {kx * delta:.3f}); {span}, one frame every {a.kh_every:g} time units, "
                            f"played at {speed:g} time units per second. {res}."))
    finish(fd, dict(common, title="Kelvin–Helmholtz dye", colormap="smoke",
                    scale={"kind": "smoke", "lo": 0.0, "hi": 1.0},
                    alt="A pale horizontal band of passive dye between two shear layers; its edges curl into "
                        "rows of spirals that entrain dark fluid, outlined by one thin fluorescent "
                        "yellow-green line at half concentration.",
                    field=f"Concentration c of a passive dye that starts as the central band, {span}; the "
                          f"yellow-green line is c = 1/2.",
                    colour="Colour: c, from 0 (black) to 1 (light grey), the same for every frame.",
                    extent="Periodic box 4 × 2 (x × z), the full box.",
                    caption=f"Passive dye marking the central band (Schmidt number 1) in the same "
                            f"Kelvin–Helmholtz run, {span}, one frame every {a.kh_every:g}, played at "
                            f"{speed:g} time units per second; the line is the iso-concentration c = 1/2. {res}."))


# ---- Rayleigh-Benard (web/gallery/rayleigh_benard.py, gallery run values) ---------------------------------
def rayleigh_benard(a, out: Path) -> None:
    import rayleigh_benard as RB
    from numpy.polynomial import chebyshev as Ch

    from conatus.expert import IVPSolver

    Ra, Lx, Nx, Nz, dt_ = 1e6, 4.0, 384, 96, 1e-5
    plan, T = RB.rayleigh_benard(Nx, Nz, Ra, 1.0, Lx, 1.5)
    s = IVPSolver(plan, "RK443", dt_, backend="cpu", f_backend="fused")
    rng = np.random.default_rng(34)
    modes = [(2 * np.pi * j / Lx, rng.uniform(0, 2 * np.pi), rng.normal()) for j in range(1, 33)]

    def T_init(X, Z):             # the gallery script's initial condition
        nz = sum(amp * np.cos(k * X + ph) for k, ph, amp in modes) / np.sqrt(len(modes))
        return -Z + 1e-3 * nz * np.cos(np.pi * Z)
    s.set_state({T: T_init})
    print(s.explain(), flush=True)
    Zg = plan._as_grid(T, lambda X, Z: Z + 0 * X)[0]
    order = np.argsort(Zg)
    zg = Zg[order]
    xg = np.arange(Nx) * Lx / Nx

    def z_mean_and_walls(f):      # as in the gallery script
        c = Ch.chebfit(2 * zg, f[..., order].T, Nz - 1)
        ci = Ch.chebint(c)
        dc = Ch.chebder(c)
        return (Ch.chebval(1.0, ci) - Ch.chebval(-1.0, ci)) / 2, 2 * Ch.chebval(-1.0, dc), 2 * Ch.chebval(1.0, dc)
    every = a.rb_every
    n = a.frames
    end = int(round(a.rb_t1 / dt_))
    start = end - every * (n - 1)
    t0 = time.perf_counter()
    for _ in range(10):
        s.run(start // 10)
        check_finite(s.grid("T"), t=s.t)
    s.run(start - 10 * (start // 10))
    fields, times, nus = [], [], []
    for i in range(n):
        if i:
            s.run(every)
        Tg, ug = s.grid("T"), s.grid("u")
        check_finite(Tg, ug, t=s.t)
        fields.append(Tg[:, order].astype(np.float32))
        times.append(float(s.t))
        wT, _, _ = z_mean_and_walls(ug[1] * Tg)
        _, dTb, dTt = z_mean_and_walls(Tg.mean(axis=0)[None, :])
        nus.append((float(1 + wT.mean()), float(-dTb[0]), float(-dTt[0])))
    wall = time.perf_counter() - t0
    G = np.stack(fields)
    A = float(np.percentile(np.abs(G[:, :, np.abs(zg) < 0.4]), 99.0))
    W, H = a.width, int(round(a.width / Lx))
    ff = np.sqrt(Ra)
    nu_m = np.mean(np.array(nus), axis=0)
    span = f"t = {times[0]:.4f} to {times[-1]:.4f} thermal times ({(times[-1] - times[0]) * ff:.0f} free-fall times)"
    speed = every * dt_ * ff * FPS
    film = Film("rayleigh-benard", out, n, fade=True)
    for i, g in enumerate(fields):
        px = C.resample(g.T, xg, zg, W, H, periodic_x=Lx, y_range=(-0.5, 0.5))
        film.add(C.field_image(px, "smoke", grain_sigma=0, vmin=-A, vmax=A)[0], poster=i == n - 1)
    finish(film, {
        "title": "Rayleigh–Bénard convection", "colormap": "smoke", "geometry": "cartesian",
        "domain": "convection", "scale": {"kind": "smoke", "lo": -A, "hi": A},
        "alt": "A long slab of convecting fluid on black: bright hot plumes rise from the bottom wall and dark "
               "cold plumes sink from the top, swept sideways by large convection rolls.",
        "field": f"Temperature T (conduction profile included) of 2D convection at Ra = 10⁶, {span}.",
        "colour": f"Colour: T, from −A (black) to +A (light grey), A = {A:.3g} the 99th percentile of |T| in the "
                  f"interior |z| < 0.4 over all frames; the boundary layers, which reach ±1/2, saturate.",
        "extent": "Box 4 × 1 (x × z), periodic in x, walls at z = ±1/2; lengths in units of the layer depth.",
        "caption": f"Temperature in 2D Rayleigh–Bénard convection at Ra = 1e6, Pr = 1, aspect ratio 4, rigid "
                   f"walls, from seeded noise; {span}, one frame every {every * dt_:g} thermal times, played at "
                   f"{speed:.2g} free-fall times per second. {Nx} × {Nz} RealFourier × Chebyshev, 3/2 "
                   f"dealiasing, RK443, dt = {dt_:g}, CPU.",
        "script": "web/gallery/gallery_films.py", "case_script": "web/gallery/rayleigh_benard.py",
        "example_script": "docs/site/tutorials/code/t02b_rayleigh_benard_gpu.py",
        "times": [times[0], times[-1], every * dt_],
        "parameters": {"Ra": Ra, "Pr": 1.0, "Lx": Lx, "amp": 1e-3, "seed": 34},
        "numerics": {"Nx": Nx, "Nz": Nz, "dealias": 1.5, "scheme": "RK443", "dt": dt_, "backend": "cpu",
                     "steps": s.iteration, "wall_s": wall},
        "diagnostics": {"Nu_vol": nu_m[0], "Nu_bottom": nu_m[1], "Nu_top": nu_m[2],
                        "avg_window": [times[0], times[-1]]}})


# ---- Kuramoto-Sivashinsky (web/gallery/kuramoto_sivashinsky.py, same problem and numerics) ----------------
def kuramoto_sivashinsky(a, out: Path) -> None:
    from conatus.expert import IVP, CartesianCoordinates, Domain, Field, IVPSolver, RealFourier, diff, discretize, dt, lap

    L, N, dt_, transient, sample = 22.0, 32, 0.01, 300.0, 0.1
    cs = CartesianCoordinates("x")
    x = cs["x"]
    u = Field("u", Domain(cs, [RealFourier(x, N, (0, L), dealias=1.5)]))
    pb = IVP([u], name=f"Kuramoto-Sivashinsky L={L:g}")
    pb.add_equation((dt(u) + lap(u) + lap(lap(u)), -u * diff(u, x)))
    s = IVPSolver(discretize(pb), "RK443", dt_)
    print(s.explain(), flush=True)
    rng = np.random.default_rng(34)
    amp, ph = 0.1 * rng.normal(size=6), 2 * np.pi * rng.random(6)
    s.set_state({u: lambda X: sum(amp[j] * np.cos(2 * np.pi * (j + 1) * X / L + ph[j]) for j in range(6))})
    t0 = time.perf_counter()
    s.run(int(round(transient / dt_)))
    win, scroll = a.ks_window, a.ks_scroll
    per = int(round(sample / dt_))
    nsamp = int(round((win + scroll) / sample)) + 1
    coeffs, times = [], []
    for i in range(nsamp):
        if i:
            s.run(per)
        coeffs.append(s.coeffs("u").copy())
        times.append(float(s.t))
    wall = time.perf_counter() - t0
    Cf, times = np.array(coeffs), np.array(times)
    check_finite(Cf, t=s.t)
    E = 0.5 * (Cf[:, 0] ** 2 + 0.5 * (Cf[:, 2:] ** 2).sum(1))
    W, H = a.width, a.width // 4
    xp = L - (np.arange(H) + 0.5) * L / H
    k = 2 * np.pi * np.arange(N // 2) / L
    field = np.cos(np.outer(xp, k)) @ Cf[:, 0::2].T - np.sin(np.outer(xp, k)) @ Cf[:, 1::2].T   # (H, nt)
    A = float(np.percentile(np.abs(field), 99.7))
    n = a.frames
    ends = times[0] + win + np.arange(n) * scroll / (n - 1)
    film = Film("kuramoto-sivashinsky-l22", out, n, fade=True)
    for i, te in enumerate(ends):
        tp = te - win + (np.arange(W) + 0.5) * win / W
        img = np.stack([np.interp(tp, times, row) for row in field])
        film.add(C.field_image(img, "signed", grain_sigma=0, vmax=A)[0], poster=i == n // 2)
    speed = scroll / (n - 1) * FPS
    finish(film, {
        "title": "Kuramoto–Sivashinsky chaos, L = 22", "colormap": "signed", "geometry": "cartesian",
        "domain": "dynamics", "scale": {"kind": "signed", "lo": -A, "hi": A},
        "alt": "A space-time diagram scrolling to the left: wavy red and green stripes of negative and positive u "
               "merge, split and drift, separated by black zero lines, new time entering on the right.",
        "field": f"The solution u(x, t) of the Kuramoto–Sivashinsky equation, L = 22, in a window of "
                 f"{win:g} time units that moves from t = {times[0]:g}–{times[0] + win:g} to "
                 f"t = {ends[-1] - win:g}–{ends[-1]:g}: x upwards, t to the right.",
        "colour": f"Colour: u, red for negative, black at zero, glass green for positive; the scale ends at "
                  f"the 99.7th percentile of |u| over the whole run shown ({A:.3g}).",
        "extent": f"x from 0 to 22 (periodic) upwards; {win:g} time units across the frame.",
        "caption": f"u(x, t) of u_t + u u_x + u_xx + u_xxxx = 0 on a periodic domain L = 22 after a transient "
                   f"of {transient:g}; the {win:g}-unit window scrolls by {scroll:g} time units, played at "
                   f"{speed:.3g} time units per second. RealFourier N = {N} (dealias 3/2), RK443, dt = {dt_:g}.",
        "script": "web/gallery/gallery_films.py", "case_script": "web/gallery/kuramoto_sivashinsky.py",
        "example_script": "docs/site/tutorials/code/t05_ks_orbits.py",
        "times": [float(times[0]), float(times[-1]), sample],
        "parameters": {"L": L, "N": N, "dt": dt_, "transient": transient, "window": win, "scroll": scroll,
                       "seed": 34},
        "numerics": {"N": N, "dealias": 1.5, "scheme": "RK443", "dt": dt_, "steps": s.iteration, "wall_s": wall},
        "diagnostics": {"energy_mean": float(E.mean()), "energy_min": float(E.min()), "energy_max": float(E.max()),
                        "mean_u_max": float(np.abs(Cf[:, 0]).max())}})


# ---- Swift-Hohenberg (web/gallery/swift_hohenberg.py, same problem and numerics) ----------------------------
def swift_hohenberg(a, out: Path) -> None:
    from conatus.expert import IVP, CartesianCoordinates, Domain, Field, IVPSolver, Parameter, RealFourier, discretize, dt, lap

    r_, nx, ny, ppw, dt_, tend = 0.1, 24, 12, 8, 0.5, 600.0
    Lx, Ly, Nx, Ny = 2 * np.pi * nx, 2 * np.pi * ny, ppw * nx, ppw * ny
    cs = CartesianCoordinates("x", "y")
    x, y = cs.coords
    D = Domain(cs, [RealFourier(x, Nx, (0, Lx), dealias=2), RealFourier(y, Ny, (0, Ly), dealias=2)])
    u, r = Field("u", D), Parameter("r", r_)
    ivp = IVP([u], name="Swift-Hohenberg 2D")
    ivp.add_equation((dt(u) - r * u + (u + lap(u)) + lap(u + lap(u)), -u * u * u))
    s = IVPSolver(discretize(ivp, params={"r": r_}), "SBDF2", dt_)
    print(s.explain(), flush=True)
    rng = np.random.default_rng(34)
    nxs, nys = np.meshgrid(np.arange(-nx * 2, nx * 2 + 1), np.arange(0, ny * 2 + 1))
    kx, ky = 2 * np.pi * nxs / Lx, 2 * np.pi * nys / Ly
    sel = np.abs(np.hypot(kx, ky) - 1.0) < 0.1
    kx, ky = kx[sel], ky[sel]
    amp, ph = 1e-2 * rng.normal(size=kx.size), 2 * np.pi * rng.random(kx.size)
    s.set_state({u: lambda X, Y: sum(amp[j] * np.cos(kx[j] * X + ky[j] * Y + ph[j]) for j in range(kx.size))})
    n = a.frames
    total = int(round(tend / dt_))
    every = total // (n - 1)
    if every * (n - 1) != total:
        raise ValueError(f"{total} steps are not {n - 1} equal intervals")
    t0 = time.perf_counter()
    fields, times = [s.grid("u")], [float(s.t)]
    for i in range(1, n):
        s.run(every - 1 if i == n - 1 else every)
        if i == n - 1:
            before = s.grid("u")
            s.run(1)
        g = s.grid("u")
        check_finite(g, t=s.t)
        fields.append(g)
        times.append(float(s.t))
    wall = time.perf_counter() - t0
    rate = float(np.abs(fields[-1] - before).max() / dt_)
    rms = float(np.sqrt(np.mean(fields[-1] ** 2)))
    W, H = a.width, int(round(a.width * Ly / Lx))

    def synth(g):                 # exact Fourier synthesis on the pixel grid, as the gallery script
        G = np.fft.rfft2(g.T)
        P = np.zeros((H, W // 2 + 1), complex)
        hy = Ny // 2
        P[:hy, :G.shape[1] - 1] = G[:hy, :-1]
        P[-hy + 1:, :G.shape[1] - 1] = G[-hy + 1:, :-1]
        return (np.fft.irfft2(P, s=(H, W)) * (H * W) / (Nx * Ny))[::-1]
    lim = float(max(np.abs(synth(g)).max() for g in fields[-5:]))
    film = Film("swift-hohenberg", out, n, fade=True)
    for i, g in enumerate(fields):
        film.add(C.field_image(synth(g), "smoke", grain_sigma=0, vmin=-lim, vmax=lim)[0], poster=i == n - 1)
    speed = every * dt_ * FPS
    finish(film, {
        "title": "Swift–Hohenberg labyrinth", "colormap": "smoke", "geometry": "cartesian", "domain": "patterns",
        "scale": {"kind": "smoke", "lo": -lim, "hi": lim},
        "alt": "Light and dark stripes appear out of an even grey, curve into a dense labyrinth of fixed "
               "width and slowly straighten into patches of parallel rolls that meet at defects.",
        "field": f"The field u(x, y) of the Swift–Hohenberg equation at r = 0.1, from seeded noise at t = 0 to "
                 f"t = {times[-1]:g}.",
        "colour": f"Colour: u, from −{lim:.3g} (black, troughs) to +{lim:.3g} (light grey, crests), the largest "
                  f"|u| of the final state; the same scale for every frame, so the small initial noise is grey.",
        "extent": f"Periodic box of {nx} × {ny} roll wavelengths (2π each), the full box.",
        "caption": f"Rolls of du/dt = r u − (1 + ∇²)² u − u³ at r = {r_:g} growing from noise in a periodic box of "
                   f"{nx} × {ny} roll wavelengths, t = 0 to {times[-1]:g}, one frame every {every * dt_:g}, played "
                   f"at {speed:g} time units per second. RealFourier × RealFourier {Nx} × {Ny} (dealias 2 for the "
                   f"cubic term), SBDF2, dt = {dt_:g}.",
        "script": "web/gallery/gallery_films.py", "case_script": "web/gallery/swift_hohenberg.py",
        "example_script": "examples/patterns/swift_hohenberg.py",
        "times": [times[0], times[-1], every * dt_],
        "parameters": {"r": r_, "Lx": Lx, "Ly": Ly, "seed": 34},
        "numerics": {"Nx": Nx, "Ny": Ny, "dealias": 2, "scheme": "SBDF2", "dt": dt_, "steps": s.iteration,
                     "wall_s": wall},
        "diagnostics": {"rms_u": rms, "rolls_rms": 0.3652433015 / np.sqrt(2), "max_dudt": rate}})


# ---- Rossby-Haurwitz wave (web/gallery/rossby_haurwitz.py): one period of the wave, fixed camera ----------
def rossby_haurwitz(a, out: Path) -> None:
    """The gallery problem run to day 14 (the still), then filmed until the m = 4 phase of the vorticity
    at 45 deg N has turned by 2 pi: the pattern has moved one wavelength (90 deg) and the last frame
    nearly joins the first. The camera is fixed, as for the still. The small-scale ripples grow slowly
    and are not periodic: the last few frames are cross-faded into the first (said in the caption)."""
    import conatus.expert as CN
    import rossby_haurwitz as RHG
    from scipy.ndimage import map_coordinates

    from conatus.discretize import discretize

    rh = RHG.load_driver()
    Lm, dt_, view = 63, 0.03, 35.0
    s2 = CN.S2Coordinates("phi", "theta")
    bt = CN.SphereBasis(s2["theta"], Lm + 1, radius=1.0, dealias=3 / 2)
    Np = 2 * (Lm + 1)
    D = CN.Domain(s2, [CN.RealFourier(s2["phi"], Np, (0, 2 * np.pi), dealias=3 / 2), bt])
    u, h = CN.VectorField("u", D), CN.Field("h", D)
    Cz = CN.TensorField("Cz", CN.Domain(s2, [bt]), 2)
    pb = CN.IVP([u, h])
    pb.add_equation((CN.dt(u) + CN.grad(h) + CN.dot(Cz, u), -CN.dot(u, CN.grad(u))))
    pb.add_equation((CN.dt(h) + rh.H0 * CN.div(u), -CN.div((h - rh.H0) * u)))

    def cz(P, T):
        c = 2 * np.cos(T) + 0 * P
        return np.array([[0 * c, c], [-c, 0 * c]])
    plan = discretize(pb, ncc={Cz: cz})
    s = CN.IVPSolver(plan, "RK443", dt_)
    s.set_state({u: lambda P, T: np.array(rh.initial(P, T)[:2]), h: lambda P, T: rh.initial(P, T)[2]})
    print(s.explain(), flush=True)
    S = plan.space
    zeta = lambda: plan.evaluator.evaluate(CN.curl(u), {u: s.coeffs("u")}, {})   # noqa: E731
    i4 = list(S.ms).index(rh.RW)
    x45 = np.array([np.cos(np.pi / 4)])

    def phase(z):
        return float(np.angle(S.H.values(rh.RW, 0, x45) @ z[i4])[0])
    days_per_unit = 1.0 / (rh.OMEGA * 86400)
    R, Wr = rh.RW, rh.W
    nu = (R * (3 + R) * Wr - 2) / ((1 + R) * (2 + R))
    period_est = (2 * np.pi / R) / abs(nu) * 1.08
    every = max(1, int(round(period_est / dt_ / a.frames)))
    t0 = time.perf_counter()
    s.run(int(round(14.0 / days_per_unit / dt_)))
    zs, ph, times = [zeta()], [phase(zeta())], [float(s.t)]
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
        times.append(float(s.t))
        if len(zs) > 4 * a.frames:
            raise RuntimeError("the m = 4 phase did not turn by 2 pi")
    n = len(zs)
    k = FADE_FRAMES
    extra = [z]
    for _ in range(k - 1):
        s.run(every)
        extra.append(zeta())
    wall = time.perf_counter() - t0
    period_days = n * every * dt_ * days_per_unit
    ncol, nlon = 721, 1440
    colat = np.linspace(0, np.pi, ncol)
    lon = 2 * np.pi * np.arange(nlon) / nlon
    W, H = a.width, a.width * 3 // 4
    rad = 0.46 * H                                            # the still's framing
    X = (np.arange(W) + 0.5 - W / 2) / rad
    Y = (H / 2 - np.arange(H) - 0.5) / rad
    XX, YY = np.meshgrid(X, Y)
    rho2 = XX ** 2 + YY ** 2
    inside = rho2 <= 1.0
    ZZ = np.sqrt(np.clip(1 - rho2, 0, None))
    p0 = np.radians(view)
    ex, ez, ey = ZZ * np.cos(p0) - YY * np.sin(p0), ZZ * np.sin(p0) + YY * np.cos(p0), XX
    c_lat = np.arccos(np.clip(ez, -1, 1))
    c_lon = np.mod(np.arctan2(ey, ex), 2 * np.pi)
    limb = np.clip(1.0 - np.abs(np.sqrt(rho2) - 1.0) * rad / 0.8, 0, 1)
    synths = [RHG.synth(S, z, colat, lon) for z in zs + extra]
    vmax = float(max(np.abs(Z).max() for Z in synths))

    def draw(Z):
        wrap = np.concatenate([Z, Z[:, :1]], axis=1)
        f = map_coordinates(wrap, [(c_lat / np.pi * (ncol - 1)).ravel(), (c_lon / (2 * np.pi) * nlon).ravel()],
                            order=3, mode="nearest").reshape(H, W)
        f[~inside] = np.nan
        return C.overlay_line(C.field_image(f, "signed", grain_sigma=0, vmax=vmax)[0], limb, color="#3A3A38",
                              alpha=0.9).astype(float)
    film = Film("rossby-haurwitz", out, n, fade=False)
    # frames k..n-1 of the period, then frames n..n+k-1 (one period later) cross-faded into 0..k-1
    for i in range(k, n):
        film.add(draw(synths[i]).astype(np.uint8), poster=i == k)
    for j, w in enumerate((np.arange(k) + 1) / (k + 1)):
        film.add(np.clip(np.rint((1 - w) * draw(synths[n + j]) + w * draw(synths[j])), 0, 255).astype(np.uint8))
    speed = every * dt_ * days_per_unit * FPS
    d0 = times[k] * days_per_unit
    finish(film, {
        "title": "Rossby–Haurwitz wave on the sphere", "colormap": "signed", "geometry": "sphere",
        "domain": "rotating", "scale": {"kind": "signed", "lo": -vmax, "hi": vmax},
        "alt": "A globe seen from mid-northern latitudes on black, covered by large alternating green and red "
               "vorticity lobes, four around each hemisphere, travelling east.",
        "field": f"Relative vorticity ζ of the wavenumber-4 Rossby–Haurwitz wave over one period of the wave "
                 f"({period_days:.2f} days, from day {d0:.1f}), seen from {view:g}° N by a fixed camera.",
        "colour": f"Colour: ζ, red for negative, black at zero, glass green for positive; one scale for the film, "
                  f"the largest |ζ| over all frames ({vmax:.3g}, in units of Ω).",
        "extent": "The whole visible hemisphere of the unit sphere, orthographic projection.",
        "caption": f"Relative vorticity of the wavenumber-4 Rossby–Haurwitz wave (Williamson et al. 1992, test "
                   f"case 6), inviscid shallow water on the rotating sphere, over one period of the wave "
                   f"({period_days:.2f} days) from day {d0:.1f}, played at {speed:.2g} days per second. The "
                   f"pattern returns after one period but the faint small-scale ripples do not: the last "
                   f"{k} frames are cross-faded into the first. Fourier × spin-weighted spherical harmonics, "
                   f"T{Lm} ({Np} × {Lm + 1}), implicit Coriolis, RK443, dt = {dt_:g}/Ω.",
        "script": "web/gallery/gallery_films.py", "case_script": "web/gallery/rossby_haurwitz.py",
        "example_script": "benchmarks/spherical/rossby_haurwitz.py",
        "loop": f"one period of the wave (m = 4 phase measured); the last {k} frames are cross-faded with the "
                f"same view one period earlier (the small-scale ripples differ)",
        "times": [times[k], times[-1] + every * dt_ * k, every * dt_],
        "parameters": {"Lmax": Lm, "R": rh.RW, "view_lat": view},
        "numerics": {"Nphi": Np, "dealias": 1.5, "scheme": "RK443", "dt": dt_, "steps": s.iteration,
                     "steps_per_frame": every, "wall_s": wall},
        "diagnostics": {"period_days": period_days, "max_abs_zeta": vmax, "days_start": d0}})


# ---- Orr-Sommerfeld mode (web/gallery/orr_sommerfeld.py): the travelling wave over one period -------------
def orr_sommerfeld(a, out: Path) -> None:
    import orr_sommerfeld as OS
    from numpy.polynomial import chebyshev as Ch

    from conatus.expert import eigenvalues

    N, Re, alpha = 64, 1e4, 1.0
    res = eigenvalues(OS.orr_sommerfeld(N, Re, alpha), drift_plan=OS.orr_sommerfeld(N * 3 // 2, Re, alpha),
                      drift_tol=1e-6)
    i = int(res.values.imag.argmax())
    cval = complex(res.values[i])
    print(res.explain(key="imag"), flush=True)
    cf = res.vectors[:N, i]
    walls = np.abs(np.concatenate([Ch.chebval([-1, 1], cf), Ch.chebval([-1, 1], Ch.chebder(cf))])).max()
    W = a.width
    lam = 2 * np.pi / alpha
    H = int(round(W * 2.0 / lam / 2)) * 2
    yp = 1.0 - (np.arange(H) + 0.5) * 2.0 / H
    xp = (np.arange(W) + 0.5) * lam / W
    uh = 1j * Ch.chebval(yp, Ch.chebder(cf)) / alpha
    uh = uh / uh[np.argmax(np.abs(uh))]
    period = lam / cval.real
    n = a.frames
    film = Film("orr-sommerfeld-mode", out, n, fade=False)
    for j in range(n):
        t = j * period / n
        F = np.real(uh[:, None] * np.exp(1j * alpha * (xp[None, :] - cval.real * t)))
        film.add(C.field_image(F, "signed", grain_sigma=0, vmax=1.0)[0], poster=j == 0)
    finish(film, {
        "title": "Tollmien–Schlichting wave in a channel", "colormap": "signed", "geometry": "cartesian",
        "domain": "shear", "scale": {"kind": "signed", "lo": -1.0, "hi": 1.0},
        "alt": "A wide band of red and green lobes on black sliding steadily to the right: the streamwise velocity "
               "of the least stable wave of plane channel flow, strongest just inside the top and bottom walls.",
        "field": f"Streamwise velocity perturbation Re(û(y) e^(iα(x − c_r t))) of the least stable Orr–Sommerfeld "
                 f"mode, over one period 2π/(α c_r) = {period:.4g} (the wave moves one wavelength).",
        "colour": "Colour: u normalised to a largest |û| of 1, red for negative, black at zero, glass green for "
                  "positive; the slow growth exp(α c_i t) is left out, so the amplitude is the same in every frame.",
        "extent": "One wavelength 2π/α in x (to the right), the channel −1 ≤ y ≤ 1 (upwards), true aspect ratio.",
        "caption": f"Streamwise velocity of the Orr–Sommerfeld mode of plane Poiseuille flow at Re = {Re:g}, "
                   f"α = {alpha:g}, c = {cval.real:.8f} + {cval.imag:.8f}i, travelling at c_r over one period "
                   f"({period:.4g} time units in {n / FPS:.0f} s; the loop closes exactly). The time dependence is the "
                   f"mode's own exp(−iαct) without its growth factor. Chebyshev N = {N}, dense QZ, tau modes removed "
                   f"by an N = {N * 3 // 2} drift test.",
        "script": "web/gallery/gallery_films.py", "case_script": "web/gallery/orr_sommerfeld.py",
        "example_script": "docs/site/tutorials/code/t01_orr_sommerfeld.py",
        "loop": "one period of the travelling wave: the frame after the last is the first (no transition)",
        "times": [0.0, period * (n - 1) / n, period / n],
        "parameters": {"N": N, "Re": Re, "alpha": alpha},
        "diagnostics": {"c": [cval.real, cval.imag], "residual": float(res.residuals[i]),
                        "wall_defect_rel": float(walls / np.abs(cf).max()), "period": period}})


# ---- Taylor-Couette (web/gallery/taylor_couette.py): Taylor vortices growing above onset -------------------
def taylor_couette(a, out: Path) -> None:
    """The still is the critical mode at Re_c, which is steady (growth rate 0, real eigenvalue): it does
    not move. The film is a nonlinear run of the same flow (eta = 0.5, outer cylinder at rest, the
    axial period of the still, two wavelengths 2 pi / k) at Re = --tc-Re above onset, from a small
    seeded axisymmetric perturbation of u_phi: the Taylor vortices grow and saturate. The native model
    conatus.problems.native.taylor_couette (perturbation of circular Couette flow, u = 0 on both
    cylinders); the linear growth rate of the EVP of the still's script at the same Re is the check."""
    import taylor_couette as TC

    from conatus.problems.native import taylor_couette as tc_model

    k, Re, N = 3.1625, a.tc_Re, 24
    Lz = 2 * (2 * np.pi / k)
    rates = {n: float(TC.couette(Re, n * k / 2, N)[2].values.real.max()) for n in (1, 2, 3)}
    print("EVP growth rates at Re =", Re, {f"k = {n * k / 2:.4g}": v for n, v in rates.items()}, flush=True)
    Nz, dt_ = 32, a.tc_dt
    m = tc_model(eta=0.5, mu=0.0, length=Lz, Re=Re)
    sim = m.simulate(N=(4, N, Nz), scheme="RK443", dt=dt_)
    ri, ro = TC.RI, TC.RO
    rng = np.random.default_rng(34)
    amps, phs = a.tc_amp * rng.normal(size=4), 2 * np.pi * rng.random(4)

    def u0(phi, r, z):            # axisymmetric u_phi only (divergence-free), zero on both cylinders
        up = sum(amps[j] * np.cos(2 * np.pi * (j + 1) * z / Lz + phs[j]) for j in range(4))
        up = up * np.sin(np.pi * (r - ri) / (ro - ri)) + 0 * phi
        return np.stack([up, 0 * up, 0 * up])
    sim.initial(u=u0)
    print(sim.explain(), flush=True)
    r, z = sim.coordinate("r"), sim.coordinate("z")
    rec = int(round(a.tc_record / dt_))
    snaps, times, energy = [], [], []
    t0 = time.perf_counter()
    while True:
        g = sim.grid("u")[:, 0]                      # axisymmetric: the phi = 0 plane, (3, Nr, Nz)
        check_finite(g, t=sim.t)
        snaps.append(g[:2].astype(np.float32))
        times.append(float(sim.t))
        energy.append(float(np.mean(g ** 2)))
        n_ = len(energy)
        if n_ > 200 and sim.t > 20:
            back = n_ - int(round(10 / a.tc_record))
            if abs(np.log(energy[-1] / energy[back])) < 1e-3 or sim.t > a.tc_tmax:
                break
        sim.run(steps=rec)
    wall = time.perf_counter() - t0
    gfull = sim.grid("u")
    nonaxi = float(np.abs(gfull - gfull[:, :1]).max())
    E = np.array(energy)
    amp = np.sqrt(E / E[-1])
    ia = int(np.argmax(amp > 0.01))
    ib = int(np.argmax(amp > 0.999))
    ib = min(len(times) - 1, ib + (ib - ia) // 6)
    n = a.frames
    stride = max(1, (ib - ia) // (n - 1))
    idx = [ia + j * stride for j in range(n)]
    if idx[-1] >= len(times):
        raise RuntimeError("not enough recorded states for the film")
    # exponential phase: after the initial transient (the energy minimum), amplitude from 3 times that
    # minimum to 10 % of saturation
    imin = int(np.argmin(E))
    lin = (np.arange(len(E)) > imin) & (amp > 3 * amp[imin]) & (amp < 0.1)
    growth = float(np.polyfit(np.array(times)[lin], 0.5 * np.log(E[lin]), 1)[0]) if lin.sum() > 5 else None
    ts = np.array(times)
    lam_z = 2 * np.pi / k
    W = a.width
    H = int(round(W * (ro - ri) / (2 * lam_z) / 2)) * 2
    kz = 2 * np.pi * np.fft.fftfreq(Nz, Lz / Nz)

    def psi_of(ur):               # u_r = -(1/r) dpsi/dz, psi_hat = i r u_r_hat / k_z (zero mean)
        U = np.fft.fft(ur, axis=1)
        P = np.where(kz != 0, 1j * r[:, None] * U / np.where(kz != 0, kz, 1), 0)
        return np.real(np.fft.ifft(P, axis=1))
    uphis = [snaps[i][0] for i in idx]
    psis = [psi_of(snaps[i][1]) for i in idx]
    vmax = float(max(np.abs(f).max() for f in uphis))
    pmax = float(max(np.abs(p).max() for p in psis))
    order = np.argsort(r)
    rs = r[order]
    film = Film("taylor-couette-mode", out, n, fade=True)
    for j, (up, ps) in enumerate(zip(uphis, psis, strict=True)):
        F = C.resample(up[order], z, rs, W, H, periodic_x=Lz, y_range=(ri, ro))
        P = C.resample(ps[order], z, rs, W, H, periodic_x=Lz, y_range=(ri, ro))
        rgb = C.field_image(F, "signed", grain_sigma=0, vmax=vmax)[0]
        lines = sum(C.contour_mask(P, lv * pmax, width_px=1.0) for lv in (-0.75, -0.45, -0.15, 0.15, 0.45, 0.75))
        film.add(C.overlay_line(rgb, np.clip(lines, 0, 1), color="#EDEBE6", alpha=0.35), poster=j == n - 1)
    t_a, t_b = ts[idx[0]], ts[idx[-1]]
    speed = stride * a.tc_record * FPS
    rc = 68.19
    finish(film, {
        "title": "Taylor–Couette: the first Taylor vortices", "colormap": "signed", "geometry": "cylinder",
        "domain": "rotating", "scale": {"kind": "signed", "lo": -vmax, "hi": vmax},
        "alt": "A horizontal band between two cylinder walls, dark at first, where alternating red and green "
               "vertical stripes of azimuthal velocity grow and brighten, circled by thin silver streamlines "
               "of the Taylor vortices.",
        "field": f"Azimuthal velocity perturbation u_φ in the meridional (z, r) plane, with streamlines of the "
                 f"meridional flow (u_r, u_z), as Taylor vortices grow at Re = {Re:g} from t = {t_a:.4g} to "
                 f"{t_b:.4g}.",
        "colour": f"Colour: u_φ − V(r), red for negative, black at zero, glass green for positive; one scale for the "
                  f"film, the largest |u_φ − V| over all frames ({vmax:.3g}, in units of the inner wall speed). "
                  f"Streamlines at fixed levels of the largest stream function over the film, so they appear as "
                  f"the vortices grow.",
        "extent": "Gap r_i ≤ r ≤ r_o (outer wall at the top) over the axial period, two wavelengths 2π/k, "
                  f"k = {k}.",
        "caption": f"Taylor vortices growing between rotating cylinders, radius ratio η = 0.5, outer cylinder at "
                   f"rest, at Re = {Re:g} ({Re / rc:.2f} times the onset value 68.19 of the still) from a small "
                   f"seeded axisymmetric perturbation, t = {t_a:.4g} to {t_b:.4g} (time unit: gap / inner wall "
                   f"speed), played at {speed:.3g} time units per second. The still shows the critical mode, "
                   f"which is steady; the film is a nonlinear run above onset. Native model "
                   f"conatus.problems.taylor_couette, Fourier × Chebyshev × Fourier 4 × {N} × {Nz}, RK443, "
                   f"dt = {dt_:g}, CPU.",
        "script": "web/gallery/gallery_films.py", "case_script": "web/gallery/taylor_couette.py",
        "example_script": "docs/site/tutorials/code/t08_polar.py",
        "times": [t_a, t_b, stride * a.tc_record],
        "parameters": {"eta": 0.5, "mu": 0.0, "Re": Re, "k": k, "Lz": Lz, "amp": a.tc_amp, "seed": 34},
        "numerics": {"N": [4, N, Nz], "scheme": "RK443", "dt": dt_, "steps": sim.iteration, "wall_s": wall},
        "diagnostics": {"growth_rate": growth, "evp_growth_rate": rates[2],
                        "evp_rates_box_modes": {str(n_): v for n_, v in rates.items()},
                        "max_abs_nonaxisymmetric": nonaxi, "uphi_max": vmax, "t_end": times[-1]}})


CASES = {"kh": kelvin_helmholtz, "rb": rayleigh_benard, "ks": kuramoto_sivashinsky, "sh": swift_hohenberg,
         "rh": rossby_haurwitz, "os": orr_sommerfeld, "tc": taylor_couette}


def main() -> None:
    cases = dict(CASES)
    try:                          # the development-only cases, when the file is present
        import gallery_films_private
        cases.update(gallery_films_private.CASES)
    except ModuleNotFoundError:
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", choices=sorted(cases), required=True)
    ap.add_argument("--frames", type=int, default=241, help="frames of the film (24 per second)")
    ap.add_argument("--width", type=int, default=1600, help="desktop width in pixels (height from the box)")
    ap.add_argument("--kh-t0", type=float, default=3.0)
    ap.add_argument("--kh-t1", type=float, default=9.0)
    ap.add_argument("--kh-every", type=float, default=0.025)
    ap.add_argument("--rb-t1", type=float, default=0.06, help="time of the last frame (the still's time)")
    ap.add_argument("--rb-every", type=int, default=12, help="steps between frames")
    ap.add_argument("--ks-window", type=float, default=125.0)
    ap.add_argument("--ks-scroll", type=float, default=250.0)
    ap.add_argument("--shell-budget", type=float, default=6000.0, help="CPU seconds for the time stepping")
    ap.add_argument("--channel-budget", type=float, default=7300.0, help="CPU seconds for the time stepping")
    ap.add_argument("--tc-Re", type=float, default=80.0, help="Reynolds number of the film (above onset)")
    ap.add_argument("--tc-dt", type=float, default=5e-3)
    ap.add_argument("--tc-record", type=float, default=0.05, help="time between recorded states")
    ap.add_argument("--tc-tmax", type=float, default=600.0)
    ap.add_argument("--tc-amp", type=float, default=1e-5, help="amplitude of the seeded u_phi perturbation")
    a = ap.parse_args()
    out = Path(os.environ.get("CN_OUT", "."))
    out.mkdir(parents=True, exist_ok=True)
    cases[a.case](a, out)


if __name__ == "__main__":
    main()
