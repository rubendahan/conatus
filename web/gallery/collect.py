"""Collect the gallery films and images of finished runs into the website (login node: file copies only).

    source env/env.sh && python web/gallery/collect.py

Films: web/gallery/film_runs.txt lists "slug run_id" for the cases that have a film (runs of
web/gallery/gallery_films.py). For those, the WebM and MP4 files (desktop and 800 px), the two WebP
posters and what the film shows (field, colour, extent, caption, scale, measured values) come from
scratch/output/<run_id>/<slug>.film.json, and the entry gets a `film` object (files, duration,
simulated times, how the loop closes, the command and the SHA-256 of each video). The poster is the
entry's image. A case without a film keeps its still image, below.

Reads web/gallery/runs.txt (one "slug run_id" per line, in display order), takes
scratch/output/<run_id>/<slug>.webp, <slug>-800.webp and <slug>.meta.json (written by the case
script through brand_cmaps.save), copies the images to web/public/gallery/ and writes
web/public/gallery/manifest.json. The manifest is generated, never edited by hand. The exact field
each image shows, and its colour scale, come from web/gallery/fields.json (read from the scripts;
shown on the card and in the lightbox, the alt text keeps the visual description).

What the lightbox reports about a picture comes from the run that drew it, never from a literal:
`measured` is built from the diagnostics in that run's meta.json (one formatter per case below), and
`scale` is the colour range the run actually used. A separate published benchmark of the same solver
belongs to the validation page, not to the picture.
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
          "run_id", "script", "example_script", "geometry", "domain")
SHOWN = json.loads((ROOT / "web/gallery/fields.json").read_text())   # slug -> {field, colour}
GEOMETRIES = {"cartesian", "polar", "cylinder", "shell", "sphere", "ball"}
DOMAINS = {"shear", "convection", "patterns", "pipe", "rotating", "turbulence", "dynamics"}


def _g(x: float, n: int = 4) -> str:
    """A number to n significant digits, with a true minus sign."""
    return f"{x:.{n}g}".replace("-", "\u2212")


def _c(re: float, im: float, n: int = 9) -> str:
    sign = "+" if im >= 0 else "\u2212"
    return f"{_g(re, n)} {sign} {_g(abs(im), n)}i"


def _ks(d, p):
    return (f"Energy over the window: mean {_g(d['energy_mean'])}, between {_g(d['energy_min'])} and "
            f"{_g(d['energy_max'])}; spatial mean of u at most {_g(d['mean_u_max'], 2)}.")


def _pipe(d, p):
    c, w = d["centre"], d["wall"]
    return (f"Centre mode {_c(*c['eigenvalue'])}, residual {_g(c['residual'], 2)}; wall mode "
            f"{_c(*w['eigenvalue'])}, residual {_g(w['residual'], 2)}. Both change by less than "
            f"{_g(max(c['drift_N_to_1.5N'], w['drift_N_to_1.5N']), 2)} from N to 1.5N.")


MEASURED = {   # slug -> sentence from the run's own diagnostics `d` and parameters `p`
    "kelvin-helmholtz-vorticity": lambda d, p: (
        f"Growth rate of the seeded mode {_g(d['growth_rate_mode'], 3)} (times δ: "
        f"{_g(d['growth_rate_times_delta'], 3)}), fitted for t between {_g(d['growth_window'][0], 2)} and "
        f"{_g(d['growth_window'][1], 2)}."),
    "kelvin-helmholtz-dye": lambda d, p: MEASURED["kelvin-helmholtz-vorticity"](d, p),
    "rayleigh-benard": lambda d, p: (
        f"Nusselt number {_g(d['Nu_vol'], 3)} (volume), {_g(d['Nu_bottom'], 3)} (bottom wall), "
        f"{_g(d['Nu_top'], 3)} (top wall), averaged for t between {_g(d['avg_window'][0], 2)} and "
        f"{_g(d['avg_window'][1], 2)}: a short window of one run, not a converged mean."),
    "kuramoto-sivashinsky-l22": _ks,
    "pipe-mode": _pipe,
    "christensen-shell": lambda d, p: (
        f"Kinetic energy {_g(d['E_kin'], 6)}, drift frequency {_g(d['omega_m4'], 5)}; at the benchmark "
        f"point T = {_g(d['local_point']['T'], 5)}, u_φ = {_g(d['local_point']['u_phi'], 6)}."),
    "channel-retau180-streaks": lambda d, p: (
        f"Friction Reynolds number {_g(d['re_tau_lower'])} (lower wall) and {_g(d['re_tau_upper'])} "
        f"(upper wall) at t = {_g(d['t'])}."),
    "taylor-couette-mode": lambda d, p: (
        f"Re_c = {_g(d['Re_c'], 6)} at the fixed k = {_g(p['k'], 5)} ({_g(d['Re_c_N_1.5'], 6)} at 1.5N), "
        f"eigenvector residual {_g(d['residual'], 2)}. The wavenumber is set, not optimised."),
    "swift-hohenberg": lambda d, p: (
        f"r.m.s. of u {_g(d['rms_u'])} (straight rolls: {_g(d['rolls_rms'])}); largest |∂u/∂t| "
        f"{_g(d['max_dudt'], 2)} at the last step."),
    "rossby-haurwitz": lambda d, p: (
        f"Initial vorticity matches the closed form to {_g(d['synthesis_error_t0'], 2)}; "
        f"largest |ζ| after {_g(d['days'], 3)} days {_g(d['max_abs_zeta'], 4)}."),
    "orr-sommerfeld-mode": lambda d, p: (
        f"c = {_c(*d['c'])}, eigenvector residual {_g(d['residual'], 2)}, relative wall defect "
        f"{_g(d['wall_defect_rel'], 2)}."),
    "channel-hydrogen-bubbles": lambda d, p: (
        f"{d['n_bubbles']} particles at t = {_g(d['t'])}; background plane at y⁺ = {_g(d['yplus_background'], 3)}."),
}


def scale_of(meta: dict) -> dict | None:
    """The colour range the run used: {kind, lo, hi}, or None when each panel is normalised apart."""
    sc = meta.get("scale") or (meta.get("diagnostics") or {}).get("scale")
    if not sc:
        return None
    if "amplitude" in sc:
        return {"kind": sc["kind"], "lo": -float(sc["amplitude"]), "hi": float(sc["amplitude"])}
    return {"kind": sc["kind"], "lo": float(sc["vmin"]), "hi": float(sc["vmax"])}


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
    if slug not in SHOWN or not all(SHOWN[slug].get(k) for k in ("field", "colour", "extent")):
        raise SystemExit(f"{slug}: no field, colour and extent in web/gallery/fields.json")
    if slug not in MEASURED:
        raise SystemExit(f"{slug}: no MEASURED formatter (what the run itself reports)")
    measured = MEASURED[slug](meta.get("diagnostics") or {}, meta.get("parameters") or {})
    return {**{k: e[k] for k in FIELDS}, **{k: SHOWN[slug][k] for k in ("field", "colour", "extent")},
            "measured": measured, "scale": scale_of(meta)}


# ---- films (web/gallery/gallery_films.py) -----------------------------------------------------------
CASE_OF = {"kelvin-helmholtz-vorticity": "kh", "kelvin-helmholtz-dye": "kh", "rayleigh-benard": "rb",
           "kuramoto-sivashinsky-l22": "ks", "swift-hohenberg": "sh", "rossby-haurwitz": "rh",
           "orr-sommerfeld-mode": "os", "pipe-mode": "pipe", "christensen-shell": "shell",
           "channel-retau180-streaks": "channel", "taylor-couette-mode": "tc"}
FILM_BYTES = {"": 5_000_000, "-800": 1_500_000}     # per video file, checked (also by the tests)
FILM_MEASURED = {   # slug -> sentence from the film run's own diagnostics; the same keys as the still
    **{k: MEASURED[k] for k in ("kelvin-helmholtz-vorticity", "kelvin-helmholtz-dye", "rayleigh-benard",
                                "kuramoto-sivashinsky-l22", "pipe-mode", "swift-hohenberg", "orr-sommerfeld-mode")},
    "rossby-haurwitz": lambda d, p: (
        f"Period of the wave {_g(d['period_days'], 4)} days, from the turn of the m = 4 phase of the vorticity at "
        f"45° N; largest |ζ| over the film {_g(d['max_abs_zeta'], 4)}."),
    "christensen-shell": lambda d, p: (
        f"Kinetic energy {_g(d['E_kin'], 6)} at the last frame ({_g(d['E_kin_first'], 6)} at the first); drift "
        f"frequency of the m = 4 pattern {_g(d['omega_m4'], 4)}, fitted over the film."),
    "channel-retau180-streaks": lambda d, p: (
        f"Friction Reynolds number {_g(d['re_tau_first'])} at the first frame and {_g(d['re_tau_lower'])} at the "
        f"last (lower wall)."),
    "taylor-couette-mode": lambda d, p: (
        f"Growth rate of the vortices {_g(d['growth_rate'], 3)}, fitted on the energy after the initial transient, "
        f"up to 10 % of the saturated amplitude; the eigenvalue problem of the still's script gives "
        f"{_g(d['evp_growth_rate'], 3)} at the same Re and k."),
}
FILM_FILES = {"webm": "{s}.webm", "mp4": "{s}.mp4", "webm_small": "{s}-800.webm", "mp4_small": "{s}-800.mp4",
              "poster": "{s}-poster.webp", "poster_small": "{s}-poster-800.webp"}


def sha256(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def film_entry(slug: str, run_id: str) -> dict:
    """Manifest entry of a filmed case: the run's film.json, its files copied to the site."""
    run = OUTPUT / run_id
    if json.loads((run / "STATUS.json").read_text()).get("exit_code") != 0:
        raise SystemExit(f"{run_id}: non-zero exit code (not collected)")
    m = json.loads((run / f"{slug}.film.json").read_text())
    if m.get("run_id") != run_id:
        raise SystemExit(f"{slug}: film run_id {m.get('run_id')} != {run_id}")
    files = {k: v.format(s=slug) for k, v in FILM_FILES.items()}
    for k, name in files.items():
        src = run / name
        if k.startswith(("webm", "mp4")) and src.stat().st_size >= FILM_BYTES["-800" if "small" in k else ""]:
            raise SystemExit(f"{src}: {src.stat().st_size} B over the budget")
        shutil.copyfile(src, PUBLIC / name)
    for k in ("field", "colour", "extent", "caption", "alt", "scale", "loop", "times"):
        if not m.get(k):
            raise SystemExit(f"{slug}: the film metadata has no {k!r}")
    if slug not in FILM_MEASURED:
        raise SystemExit(f"{slug}: no FILM_MEASURED formatter (what the film run itself reports)")
    cfg = json.loads((run / "config.json").read_text())
    t = m["times"]
    film = {**{k: f"/gallery/{v}" for k, v in files.items()},
            "duration_s": m["duration_s"], "fps": m["fps"], "frames": m["frames"],
            "bytes": {k: (run / v).stat().st_size for k, v in files.items()},
            "times": t, "loop": m["loop"], "case_script": m["case_script"],
            "span": f"Simulated time {t[0]:.4g} to {t[1]:.4g}, shown in {m['duration_s']:g} s of playback.",
            "generation": {"command": " ".join(["python web/gallery/gallery_films.py --case", CASE_OF[slug],
                                                *cfg.get("args", [])]),
                           "sha256": {k: sha256(run / v) for k, v in files.items() if not k.startswith("poster")}}}
    return {"slug": slug, "title": m["title"], "image": film["poster"], "image_small": film["poster_small"],
            "width": m["width"], "height": m["height"], "alt": m["alt"], "caption": m["caption"],
            "colormap": m["colormap"], "run_id": run_id, "script": m["script"],
            "example_script": m["example_script"], "geometry": m["geometry"], "domain": m["domain"],
            "field": m["field"], "colour": m["colour"], "extent": m["extent"],
            "measured": FILM_MEASURED[slug](m.get("diagnostics") or {}, m.get("parameters") or {}),
            "scale": {"kind": m["scale"]["kind"], "lo": float(m["scale"]["lo"]), "hi": float(m["scale"]["hi"])},
            "film": film}


def read_pairs(path: Path) -> list[list[str]]:
    return [ln.split() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def main() -> None:
    PUBLIC.mkdir(parents=True, exist_ok=True)
    lines = read_pairs(ROOT / "web/gallery/runs.txt")
    films = dict(read_pairs(ROOT / "web/gallery/film_runs.txt"))
    manifest = [film_entry(slug, films[slug]) if slug in films else entry(slug, run_id) for slug, run_id in lines]
    (PUBLIC / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    for e in manifest:
        print(f"{e['slug']:32s} {e['width']}x{e['height']}  {e['run_id']}  {'film' if 'film' in e else 'still'}")
    print(f"{len(manifest)} entries ({sum('film' in e for e in manifest)} films) -> {PUBLIC / 'manifest.json'}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
