"""Figures and films of the tutorials, made by running the tutorials themselves.

Two steps, so that the drawing can be redone without computing again:

  capture   runs each tutorial script of ``docs/site/tutorials/code`` (``runpy``, in a scratch
            directory, exactly as ``run_tutorials.py`` does), then asks the objects the script
            built for the data a figure needs (an eigenvalue spectrum, a branch, a trajectory).
            When a figure needs more than the script keeps (snapshots for a film, a neutral curve
            on a parameter grid), it is computed with the script's own functions and objects,
            with the script's numerics; nothing is invented or fitted.  One ``<name>.npz`` and
            one ``<name>.json`` per tutorial in the data directory.
  render    draws the figures (PNG, ``docs/site/tutorials/media``) and the films (WebM + MP4 +
            WebP poster, ``docs/site/_static/tutorial_films``) from the captured data, with the
            style file ``tutorial_media.mplstyle``, writes ``media.json`` (file, alt text,
            caption, tutorial) next to the figures and a contact sheet of everything into the
            data directory.

  python docs/site/tutorial_media.py capture --out DIR [--only t01,t05] [-j 4]
  python docs/site/tutorial_media.py render --data DIR [--only t01,t05]

Capture is heavy (about ten minutes of CPU in all): run it as a job,
Pillow and ffmpeg (``$CN_FFMPEG``, else ``ffmpeg`` on the PATH, else the binary of the
imageio-ffmpeg package).
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

SITE = Path(__file__).resolve().parent
ROOT = SITE.parents[1]
CODE = SITE / "tutorials" / "code"
PUBLIC_CODE = ROOT / "packaging" / "public" / "docs" / "site" / "tutorials" / "code"
MEDIA = SITE / "tutorials" / "media"
FILMS = SITE / "_static" / "tutorial_films"
STYLE = SITE / "tutorial_media.mplstyle"

# The tutorials of the public documentation, by script.
SCRIPTS = {
    "t00": "t00_getting_started", "t01": "t01_orr_sommerfeld", "t02": "t02a_rayleigh_benard_cpu",
    "t03": "t03_symmetries", "t04": "t04_bratu", "t05": "t05_ks_orbits", "t08": "t08_polar",
    "t10": "t10_bifurcations", "t11": "t11_runs", "t12": "t12_chaos_ensembles",
    "t13": "t13_sensitivity_data", "t14": "t14_from_dedalus", "t15": "t15_ks_covariant",
    "t16": "t16_particles_dye",
}


def script_path(key: str) -> Path:
    """The tutorial script; the public variant when the public documentation has its own."""
    name = SCRIPTS[key] + ".py"
    return PUBLIC_CODE / name if (PUBLIC_CODE / name).exists() else CODE / name


# =============================================================================== capture helpers
def _arr(x) -> np.ndarray:
    return np.asarray(x)


def _sample(sim, until: float, every: int, fn, keep=None):
    """Run ``sim`` (a model simulation) by ``every`` steps up to ``until``; ``fn(sim)`` at each
    sample (and at the start); returns times and stacked values."""
    t, v = [sim.t], [fn(sim)]
    while sim.t < until - 1e-12:
        sim.solver.run(every)
        t.append(sim.t)
        v.append(fn(sim))
    return np.array(t), v


def cap_t00(g):
    convection, m, P, crit = g["convection"], g["m"], g["P"], g["crit"]
    roll_amplitude, nusselt, grid = g["roll_amplitude"], g["nusselt"], g["grid"]
    d, meta = {}, {}
    for Ra in (1500, 2500):                     # the stability block, both Rayleigh numbers
        lin = convection(Ra).linearize(about={"u": 0, "T": lambda z: -z})
        d[f"spec_{Ra}"] = lin.eigenvalues(N=24, k=np.pi).values
    # growth rate of the least stable mode over (k, Ra): the parametric pencil of the onset block
    ks, ras = np.linspace(2.0, 4.8, 29), np.linspace(1500.0, 3200.0, 35)
    d["ks"], d["ras"] = ks, ras
    d["growth"] = np.array([[P.least_stable(Ra=Ra, k=k).sigma.real for k in ks] for Ra in ras])
    meta.update(Ra_c=float(crit.value), k_c=float(crit.over["k"]), sigma=float(g["growth"][2500]))
    # the simulation block again (same model, resolution, scheme, step and start), sampled
    sim = m.simulate(N=(32, 24), scheme="RK443", dt=5e-4)
    sim.initial(T=lambda x, z: -z + 1e-6 * np.cos(np.pi * x) * np.cos(np.pi * z))
    ts, amp, nu, snaps, snap_t = [], [], [], [], []
    while sim.t < 6.0 - 1e-9:
        ts.append(sim.t)
        amp.append(roll_amplitude(sim))
        nu.append(nusselt(sim)[0])
        if 1.2 - 1e-9 <= sim.t <= 3.6 + 1e-9:
            snaps.append(sim.grid("T"))
            snap_t.append(sim.t)
        sim.solver.run(25)                       # 25 steps = 0.0125
    ts.append(sim.t), amp.append(roll_amplitude(sim)), nu.append(nusselt(sim)[0])
    d.update(t=ts, amp=amp, nu=nu, snaps=snaps, snap_t=snap_t, T_end=sim.grid("T"),
             xn=grid.axes[0].nodes(), zn=grid.axes[1].nodes())
    meta["Nu_end"] = float(nu[-1])
    return d, meta


def cap_t01(g):
    cn = g["cn"]
    res, i, mode, P, crit = g["res"], g["i"], g["mode"], g["P"], g["crit"]
    d = {"values": res.values, "residuals": res.residuals, "i": i, "y": mode.coordinate("y"),
         "v": mode["v"], "coeffs": mode.coeffs("v")}
    Re = np.geomspace(3000, 25000, 30)
    al = np.linspace(0.55, 1.35, 33)

    def growth(r, a):
        try:
            return P.least_stable(Re=r, alpha=a).sigma.imag
        except cn.errors.ConvergenceError:      # no mode passes the drift test there: left blank
            return np.nan
    d["Re"], d["alpha"] = Re, al
    d["growth"] = np.array([[growth(r, a) for r in Re] for a in al])
    return d, {"Re_c": float(crit.value), "alpha_c": float(crit.over["alpha"]),
               "c": [float(res.values[i].real), float(res.values[i].imag)]}


def cap_t02(g):
    from conatus.diagnostics import Chebyshev, Grid, Periodic
    m, modes, roll_amplitude, dd = g["m"], g["modes"], g["roll_amplitude"], g["d"]
    sim = m.simulate(N=(16, 24), scheme="RK443", dt=5e-4)        # the IVP block, sampled
    sim.initial(T=lambda x, z: -z + 1e-6 * np.cos(np.pi * x) * np.cos(np.pi * z))
    t, a = _sample(sim, 0.8, 10, roll_amplitude)
    grid = Grid([Periodic(2.0, 16), Chebyshev(-0.5, 0.5, 24)])
    d = {"t": t, "amp": a, "spec": modes.values, "ke_t": dd.sim_time, "ke": dd.tasks["KE"],
         "T": sim.grid("T"), "xn": grid.axes[0].nodes(), "zn": grid.axes[1].nodes()}
    meta = {"sigma_evp": float(g["sigma_evp"]), "sigma_ivp": float(g["sigma_ivp"])}
    # the restart check of the jobs block: the two-job chain against the uninterrupted run
    import re
    Xc, Xs = np.asarray(g["chain"].solver.X), np.asarray(g["sim"].solver.X)
    hit = re.search(r"checkpoint_\d+\.h5 \(iteration (\d+), t = ([0-9.eE+-]+)\)", g["chain"].explain())
    meta.update(chain_identical=bool(np.array_equal(Xc, Xs)), chain_max_abs_diff=float(np.abs(Xc - Xs).max()),
                chain_unknowns=int(Xc.size), chain_iteration=int(g["chain"].solver.iteration),
                chain_t=float(g["chain"].t), chain_rows=int(len(dd.sim_time)),
                checkpoint_iteration=int(hit.group(1)) if hit else None,
                checkpoint_t=float(hit.group(2)) if hit else None)
    try:                                           # the leading eigenfunction, temperature part
        j = int(np.argmax(modes.values.real))
        md = modes.mode(j)
        name = next(n for n in ("T", "dT", "theta") if n in md.names)
        d["mode_z"], d["mode_T"] = md.coordinate("z"), md[name]
    except Exception as e:  # noqa: BLE001 -- optional panel; the reason is recorded
        meta["mode_error"] = repr(e)
    return d, meta


def _timed(fn, repeats: int):
    """``repeats`` wall times of ``fn()`` in ms, and its last result."""
    walls, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        walls.append(1e3 * (time.perf_counter() - t0))
    return walls, out


def _machine() -> dict:
    """Hardware and library versions of a timing (the cost table of tutorial 3)."""
    import platform

    import scipy
    cpu = platform.processor() or "?"
    try:
        cpu = next(ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"cpu": cpu, "threads": os.environ.get("OMP_NUM_THREADS", "unset"), "numpy": np.__version__,
            "scipy": scipy.__version__, "python": platform.python_version()}


def cap_t03(g):
    from conatus.expert import discretize, eigenvalues, linearize
    comp, evp, R = g["comp"], g["evp"], 7          # R timings of each dense solve (first one included)
    d = {"full": g["res"].values}
    walls, _ = _timed(lambda: eigenvalues(g["plan"], g["group"]), R)
    meta = {"n_full": int(g["nf"]), "wall_full_ms": walls, "repeats": R, "machine": _machine()}
    for label, sym in (("onset", g["onset"]), ("other", g["other"])):   # the solve block again
        sub = linearize(comp.problem, evp.base, evp.eigenvalue, symmetry=sym)
        splan = discretize(sub, ncc=g["ncc"])
        grp = next(x for x in splan.groups if x[0].k > 0)
        walls, r = _timed(lambda splan=splan, grp=grp: eigenvalues(splan, grp), R)
        meta[f"wall_{label}_ms"] = walls
        d[label] = r.values
        meta[f"n_{label}"] = int(splan.matrices(grp).L.shape[0])
    return d, meta


def cap_t04(g):
    import dataclasses

    from conatus.expert import NLBVPContinuation
    sol, newton, settings, X = g["sol"], g["newton"], g["settings"], g["X"]
    d = {"res": [it.rel_residual for it in sol.iterations], "X": X}
    # the continuation block with more points, on both sides of the fold
    longer = dataclasses.replace(settings, max_points=60, ds=0.1, ds_max=0.15)
    br = NLBVPContinuation(newton, "lam", longer).run(sol.coeffs, 0.3, +1, p_bounds=(0.05, 1.0))
    names = [v.name for v in sol.plan.variables]
    sizes = [np.size(sol.coeffs[n]) for n in names]
    k = names.index("u")
    lam, prof = [], []
    for p in br.points:
        parts = np.split(np.asarray(p.x), np.cumsum(sizes)[:-1])
        prof.append(np.real(sol.plan.to_grid("u", parts[k].reshape(np.shape(sol.coeffs["u"])))))
        lam.append(p.p)
    d.update(lam=lam, prof=prof)
    tol = getattr(newton, "tol", None)
    return d, {"lam_c": float(g["fold"].p), "tol": None if tol is None else float(tol)}


def _ks_grid(solver, X):
    return np.real(solver.plan.to_grid("u", solver.system.scatter(np.asarray(X))[0]))


def cap_t05(g):
    ks, flow, ppo = g["ks"], g["flow"], g["ppo"]
    d = {"E1": _ks_grid(ks, g["E1"].X), "E2": _ks_grid(ks, g["E2"].X), "TW1": _ks_grid(ks, g["TW1"].X),
         "ev1": g["ev1"], "mult": np.asarray(g["fl"]["multipliers"])}
    nsteps = g["nsteps"]
    flow.X = np.asarray(ppo.X).copy()               # start on the orbit (CPU solver, one-step RK443)
    U = [_ks_grid(flow, flow.X)]
    for _ in range(2 * nsteps):                    # two periods: the reflection closes the loop
        flow.run(1)
        U.append(_ks_grid(flow, flow.X))
    d["orbit"] = U
    return d, {"T": float(ppo.params["T"]), "L": float(g["L"]), "c": float(g["TW1"].params["c"]),
               "mu1": float(g["mu1"]), "theta1": float(g["theta1"])}


def cap_t08(g):
    from scipy.optimize import brentq

    from conatus.expert import eigenvalues
    d, meta = {}, {}
    plan = g["drum"].discretize(N=(8, 32))           # the disk block
    for grp in plan.groups[:3]:                     # m = 0, 1, 2: the eight slowest modes of each
        res = eigenvalues(plan, grp)
        d[f"drum_{grp.m}"] = np.sort(res.values.real)[::-1][:8]
    d["pipe"] = g["res"].values                      # the pipe block, (m, k) = (1, 1)
    cg = g["couette_growth"]
    kk = np.linspace(2.2, 4.4, 12)
    d["tc_k"] = kk
    d["tc_Re"] = [brentq(lambda Re, k=k: cg(Re, k), 50.0, 160.0, xtol=1e-5) for k in kk]
    meta["tc_points"] = {str(k): float(v) for k, v in g["Re_c"].items()}
    return d, meta


def _curve(c):
    Z = np.array([np.asarray(p.z, dtype=float) for p in c.points])
    unstable = []
    for p in c.points:
        ev = p.data.get("eigenvalues") if isinstance(p.data, dict) else None
        unstable.append(-1 if ev is None else int(np.sum(np.real(ev) > 1e-9)))
    return Z, np.array(unstable)


def cap_t10(g):
    d, meta = {}, {}
    for name in ("origin", "cplus", "cycles", "fold_curve", "hopf_curve"):
        Z, u = _curve(g[name])
        d[f"{name}_Z"], d[f"{name}_unstable"] = Z, u
        meta[f"{name}_names"] = {k: int(v) for k, v in g[name].names.items()}
    meta.update(r_bp=float(g["bp"].point.z[-1]), r_hopf=float(g["hopf"].point.z[-1]))
    bt = g["hopf_curve"].events_of("bt")
    if bt:
        meta["bt"] = [float(v) for v in bt[0].point.z[-2:]]
    return d, meta


def cap_t11(g):
    """The registry of the page as it stands at the end of the script: its records, the query
    result, the reproduction and the cache reuse (the figure draws these, not the heat solution)."""
    runs, found, rep, hit = g["runs"], g["found"], g["rep"], g["hit"]
    records = [{"id": r.id, "name": r.name, "kappa": float(k), "status": r.status,
                "rel_error": float(r.results["rel_error"])} for k, r in runs.items()]
    meta = {"records": records, "found": [r.id for r in found],
            "query": 'cn.runs.find(kappa__gt=0.15, status="certified", tags="tutorial")',
            "reproduce": {"original": rep.original.id, "new": rep.new.id if rep.new is not None else None,
                          "identical": bool(rep.identical), "max_abs_diff": float(rep.max_abs_diff),
                          "code_identical": bool(rep.code_identical)},
            "cache": {"run": g["again"].id, "reused_by": hit.id, "hit": bool(hit.cache_report["hit"]),
                      "reused_from": hit.cache_report["reused_from"],
                      "computed": str(hit.cache_report["computed"])},
            "nsteps": int(g["NSTEPS"])}
    return {"kappa": [r["kappa"] for r in records]}, meta


def cap_t12(g):
    from conatus.ensembles import Ensemble
    s, res = g["s"], g["res"]
    d = {"exponents": res.exponents, "stderr": res.stderr, "logs": res.logs, "zk": g["zk"],
         "zk1": g["zk1"], "basin": g["basin"], "grid": g["grid"]}
    one = Ensemble(s, 1, rhs="batched")             # one trajectory on the attractor
    one.set_states(np.array([[1.0, 1.0, 20.0]]))
    one.run(1000)
    path = []
    for _ in range(4000):
        one.run(2)
        path.append(one.states()[0].copy())
    d["path"] = path
    # a small ball of starts on the attractor, followed: nearby states separate (film)
    rng = np.random.default_rng(3)
    x0 = path[-1]
    ball = x0 + 1e-3 * rng.standard_normal((600, 3))
    ens = Ensemble(s, len(ball), rhs="batched")
    ens.set_states(ball)
    cloud = [ens.states().copy()]
    for _ in range(191):
        ens.run(8)
        cloud.append(ens.states().copy())
    d["cloud"] = cloud
    return d, {"interval": float(res.interval), "dt": 0.01, "cloud_every": 8}


def cap_t13(g):
    st, pair, bfs = g["st"], g["pair"], g["bfs"]
    x = np.linspace(-60, 60, 1201)
    dd = pair.evaluate("direct", x)
    da = pair.evaluate("adjoint", x)
    d = {"x": x, "S": st.wavemaker(x), "direct": next(iter(dd.values())),
         "adjoint": next(iter(da.values())), "omega": g["r"].omega, "lam": g["lam"],
         "lead4": g["lead4"]}
    return d, {"x_star": float(g["x_star"]), "s_star": float(g["s_star"]), "S0_closed_form": float(g["S0"]),
               "eigenvalue": [float(np.real(bfs.pair.sigma)), float(np.imag(bfs.pair.sigma))]}


def cap_t14(g):
    from conatus.diagnostics import Chebyshev, Grid, Periodic
    native, translated, rel = g["native"], g["translated"], g["rel"]
    nzs = [10, 12, 14, 16, 20, 24, 28, 32]
    th, u = [], []
    for nz in nzs:
        a, b = native(nz), translated(nz)
        th.append(rel(a, b, "th")), u.append(rel(a, b, "u"))
    cell, solver = g["cell"], g["solver"]
    grid = Grid([Periodic(2.0, 24), Chebyshev(0.0, 1.0, 16)])
    return ({"nz": nzs, "gap_th": th, "gap_u": u, "th_native": cell.grid("th"),
             "th_dedalus": solver.grid("th"), "xn": grid.axes[0].nodes(), "zn": grid.axes[1].nodes()},
            {})


def cap_t15(g):
    res, dom, m, starts = g["res"], g["dom"], g["m"], g["starts"]
    d = {"exponents": res.exponents, "stderr": res.stderr, "nu": dom.nu, "theta8": g["theta8"],
         "theta4": g["theta4"], "cds": g["cds"]}
    sim = m.simulate(N=32, scheme="RK443", dt=0.05)   # the model block, one of its starts
    sim.initial(**starts[0])
    sim.run(until=100.0)
    rows = []
    for _ in range(1200):
        sim.solver.run(2)
        rows.append(np.real(sim.grid("u")))
    d["spacetime"] = rows
    return d, {"n_physical": int(dom.n_physical), "dt_row": 0.1, "L": 22.0}


def cap_t16(g):
    from conatus.diagnostics import Chebyshev, Grid, Periodic
    taylor_green, X0, T = g["taylor_green"], g["X0"], g["T"]
    d = {"errors": g["errors"], "dts": [0.1, 0.05, 0.025], "X0": X0}
    sim = taylor_green(0.025)                       # the finest run of the convergence block
    tr = sim.particles(X0)
    paths = [tr.positions(unwrapped=True).copy()]
    for j in range(1, 81):
        sim.run(until=T * j / 80)
        paths.append(tr.positions(unwrapped=True).copy())
    d["tg_paths"] = paths
    m, starts = g["m"], g["starts"]                 # the Couette block again, sampled
    sim = m.simulate(N=(32, 32), scheme="RK443", dt=0.01)
    sim.initial(u=lambda x, y: (y, 0 * x), c=lambda x, y: np.exp(-((x - 2) ** 2 + y ** 2) / 0.3))
    tr = sim.particles(starts)
    c, pos, ts = [sim.grid("c")], [tr.positions(unwrapped=True).copy()], [sim.t]
    for j in range(1, 151):
        sim.run(until=0.02 * j)
        c.append(sim.grid("c")), pos.append(tr.positions(unwrapped=True).copy()), ts.append(sim.t)
    grid = Grid([Periodic(4.0, 32), Chebyshev(-1.0, 1.0, 32)])
    d.update(c=c, pos=pos, t=ts, xn=grid.axes[0].nodes(), yn=grid.axes[1].nodes(),
             mass=g["mass"])
    return d, {"drift": float(g["drift"])}


CAPTURE = {k: globals()[f"cap_{k}"] for k in SCRIPTS}


def capture_one(key: str, out: Path) -> int:
    """Run tutorial ``key`` and its capture; write ``<key>.npz`` and ``<key>.json`` into ``out``."""
    work = out / f"work_{key}"
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["CN_TUTORIAL_OUT"] = str(work)
    script = script_path(key)
    t0 = time.perf_counter()
    g = runpy.run_path(str(script), run_name="__main__")
    t1 = time.perf_counter()
    data, meta = CAPTURE[key](g)
    arrays = {}
    for k, v in data.items():
        a = np.asarray(v)
        if a.dtype == object:
            raise TypeError(f"{key}: {k} is not a plain array")
        arrays[k] = a
    np.savez_compressed(out / f"{key}.npz", **arrays)
    meta = dict(meta, script=str(script.relative_to(ROOT)), tutorial_s=round(t1 - t0, 1),
                capture_s=round(time.perf_counter() - t1, 1), run_id=os.environ.get("CN_RUN_ID"))
    (out / f"{key}.json").write_text(json.dumps(meta, indent=1))
    print(f"[capture] {key}: tutorial {t1 - t0:.1f} s, capture {meta['capture_s']} s", flush=True)
    return 0


def capture(args) -> int:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    keys = [k for k in SCRIPTS if not args.only or k in args.only.split(",")]
    # longest first, at most -j at a time, each in its own process (scripts change cwd and env)
    cost = {"t12": 9, "t00": 8, "t16": 7, "t15": 6, "t05": 5, "t04": 4, "t01": 3}
    keys.sort(key=lambda k: -cost.get(k, 0))
    running, failed = [], []
    while keys or running:
        while keys and len(running) < args.jobs:
            k = keys.pop(0)
            log = open(out / f"{k}.log", "w")
            running.append((k, subprocess.Popen([sys.executable, __file__, "_one", k, str(out)],
                                                stdout=log, stderr=subprocess.STDOUT), log))
        time.sleep(1.0)
        for item in list(running):
            k, p, log = item
            if p.poll() is not None:
                log.close()
                running.remove(item)
                print(f"{'PASS' if p.returncode == 0 else 'FAIL'} {k}", flush=True)
                if p.returncode:
                    failed.append(k)
    print("FAILED: " + ", ".join(failed) if failed else "all captures done", flush=True)
    return 1 if failed else 0


# =============================================================================== render
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "_one":
    try:
        sys.exit(capture_one(sys.argv[2], Path(sys.argv[3])))
    except Exception:  # noqa: BLE001 -- the job log keeps the traceback of the failing capture
        traceback.print_exc()
        sys.exit(1)

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--out", required=True)
    c.add_argument("--only", default="")
    c.add_argument("-j", "--jobs", type=int, default=4)
    r = sub.add_parser("render")
    r.add_argument("--data", required=True)
    r.add_argument("--only", default="")
    r.add_argument("--no-films", action="store_true")
    a = ap.parse_args()
    if a.cmd == "capture":
        return capture(a)
    from tutorial_media_render import render      # the drawing code (matplotlib, Pillow, ffmpeg)
    return render(a)


if __name__ == "__main__":
    sys.exit(main())
