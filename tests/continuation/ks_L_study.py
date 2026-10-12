"""KS equilibria E1, E2, E3 continued in L from L = 22 (both directions), with a resolution study.

JSON per branch and a summary into ``--out``. Checks recorded with each branch:

* folds located by the continuation (|t_p| at the located point, residual);
* turning points reached with ‖u‖ → 0: pitchfork off u = 0 (Cvitanović, Davidchack & Siminos 2010,
  §2: "every time the system size crosses L/2π = n, n-cell states are generated through pitchfork
  bifurcations off u = 0"), compared with L = 2π n;
* branch returning through L = 22 after a fold: identified with the start or with its half-period
  translate τ_{L/2}, which also lies in the antisymmetric subspace R u = u.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from conatus.continuation.core import ContinuationSettings
from conatus.continuation.ks import KSDomainContinuation
from conatus.ecs import NewtonKrylovHookstep, find_equilibrium, ks_solver

L0 = 22.0
# name: (starting L, {sin mode: amplitude}) — E1–E3 as tests/ecs; C1 = Greene–Kim 1-cell at a = 13
# (L = π√a, convention checked in)
SEEDS = {"E1": (L0, {1: 1.0}), "E2": (L0, {2: 1.0}), "E3": (L0, {3: 2.0}),
         "C1": (float(np.pi * np.sqrt(13.0)), {1: 1.0, 2: 0.5})}


def settings(max_points: int) -> ContinuationSettings:
    return ContinuationSettings(ds=0.05, ds_min=1e-6, ds_max=0.25, adapt=1.4, fast_iterations=3,
                                tolerance=1e-10, max_newton=12, max_points=max_points, dp_fd=1e-6)


def one(task):
    """One branch; any failure is recorded in the result (other branches continue)."""
    try:
        return _one(task)
    except Exception as e:                                   # reported, never hidden
        name, direction, N, bounds, _ = task
        return {"equilibrium": name, "direction": direction, "N": N, "bounds": bounds,
                "points": [], "events": [], "returns_through_start": [],
                "stop": f"FAILED: {e!r}"[:400], "explain": "", "seconds": 0.0}


def _one(task):
    name, direction, N, bounds, max_points = task
    t0 = time.time()
    Ls, modes = SEEDS[name]
    ks = ks_solver(L=Ls, N=N, dealias=1.5, scheme="RK443", timestep=0.01)
    nk = NewtonKrylovHookstep(tol=1e-12, max_iterations=40, gmres_rtol=1e-3, gmres_max_iterations=60,
                              delta0=1.0, delta_min=1e-10, delta_max=10.0)
    c = np.zeros(N)
    for m, a in modes.items():
        c[2 * m + 1] = -a
    X0 = find_equilibrium(ks, ks.system.gather_state([c]), nk).X
    kc = KSDomainContinuation(N=N, dealias=1.5, dp_fd=1e-6, marginal_tol=1e-8, eig_count=10)
    br = kc.run(X0, Ls, settings(max_points), direction=direction, L_bounds=bounds)
    pts = br.points
    norms = [float(np.linalg.norm(q.x)) for q in pts]
    events = []
    for e in br.events:
        i, j = e.between
        ev = {"kind": e.kind, "between": [i, j], "L": e.p, "detail": e.detail,
              "a_GK": None if e.p is None else float((e.p / np.pi) ** 2),   # Greene–Kim a
              "L_bracket": [pts[i].p, pts[j].p], "norm_u": min(norms[i], norms[j])}
        if e.kind == "fold" and ev["norm_u"] < 1e-2 * max(norms):
            n_cell = round(e.p / (2 * np.pi))
            ev["pitchfork_off_u0"] = {"n": n_cell, "L_exact": 2 * np.pi * n_cell,
                                      "error": abs(e.p - 2 * np.pi * n_cell)}
        events.append(ev)
    # returns through L = 22 after the first point: same state or half-period translate?
    ops = kc.at(Ls).ops
    X0h = ops.translate(X0, Ls / 2)
    returns = []
    for k in range(2, len(pts)):
        if (pts[k - 1].p - Ls) * (pts[k].p - Ls) < 0:
            q = pts[k] if abs(pts[k].p - Ls) < abs(pts[k - 1].p - Ls) else pts[k - 1]
            X = kc.full(q.x, q.p)
            returns.append({"L": q.p, "dist_start": float(np.linalg.norm(X - X0) / np.linalg.norm(X0)),
                            "dist_half_shift": float(np.linalg.norm(X - X0h) / np.linalg.norm(X0))})
    out = {"equilibrium": name, "direction": direction, "N": N, "bounds": bounds,
           "points": [{"L": q.p, "energy": kc.energy(q.x, q.p), "unstable": q.unstable,
                       "residual": q.residual, "status": q.status} for q in pts],
           "events": events, "returns_through_start": returns, "stop": br.stop_reason,
           "explain": kc.explain() + "\n" + br.explain(), "seconds": time.time() - t0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--Ns", default="32,48")
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--max-points", type=int, default=160)
    ap.add_argument("--L-min", type=float, required=True)
    ap.add_argument("--L-max", type=float, required=True)
    args = ap.parse_args()
    out = Path(args.out)
    tasks = [(n, d, N, (args.L_min, args.L_max), args.max_points) for N in map(int, args.Ns.split(","))
             for n in SEEDS for d in (+1, -1)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(one, tasks)
    summary = []
    for r in results:
        tag = f"{r['equilibrium']}_{'up' if r['direction'] > 0 else 'down'}_N{r['N']}"
        (out / f"ks_L_{tag}.json").write_text(json.dumps(r, indent=1))
        summary.append({"branch": tag, "stop": r["stop"], "events": r["events"],
                        "returns": r["returns_through_start"], "seconds": r["seconds"]})
    (out / "ks_L_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
