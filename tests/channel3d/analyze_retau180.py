"""Compare a Re_tau ~ 180 run (run_retau180.py -> retau180.json) with Kim, Moin & Moser (1987,
JFM 177, 133-166; scratch/refs/KimMoinMoser1987_JFM.pdf). Reference values, with their place in
the paper:
  * Table 1 (p. 143): Re_tau ~ 180, U_m/u_tau = 15.63, U_c/u_tau = 18.20, U_c/U_m = 1.16,
    C_f = tau_w / (rho U_m^2 / 2) = 8.18e-3;
  * Figure 4 (p. 142): viscous sublayer U+ = y+ (y+ < 5) and log law U+ = 2.5 ln y+ + 5.5;
  * p. 135: box 4 pi x 2 x 2 pi, 192 x 129 x 160 points (dx+ ~ 12, dz+ ~ 7).
Usage: python analyze_retau180.py <run_dir>  (writes retau180_vs_kmm87.json next to the data)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

KMM = dict(Re_tau=180.0, Ub_over_utau=15.63, Uc_over_utau=18.20, Uc_over_Ub=1.16, Cf=8.18e-3)


def main():
    run = Path(sys.argv[1])
    d = json.loads((run / "retau180.json").read_text())
    yp, Up = np.array(d["yplus"]), np.array(d["Uplus"])
    order = np.argsort(yp)
    yp, Up = yp[order], Up[order]
    u_tau = d["u_tau"]
    Ub = d["flow_rate"]
    rec = dict(run_id=d.get("run_id", run.name), status=d["status"], t=d["t"],
               averaging_window=d["average_window"], n_average=d["n_average"],
               averaging_in_viscous_units=d["average_window"] * u_tau,   # t u_tau / h
               grid=d["grid"], box=d["box"],
               dxplus=d["box"][0] / d["grid"][0] * d["Re_tau"],
               dzplus=d["box"][2] / d["grid"][2] * d["Re_tau"])
    ours = dict(Re_tau=d["Re_tau"], Ub_over_utau=d["Ub_over_utau"], Uc_over_utau=d["Uc_over_utau"],
                Uc_over_Ub=d["Uc_over_utau"] / d["Ub_over_utau"],
                Cf=2 * u_tau**2 / Ub**2)
    rec["compare"] = {k: dict(conatus=ours[k], kmm87=KMM[k], rel_diff=ours[k] / KMM[k] - 1)
                      for k in KMM}
    sub = yp < 5
    rec["sublayer_max_rel_dev"] = float(np.max(np.abs(Up[sub] / yp[sub] - 1))) if sub.any() else None
    logr = (yp > 30) & (yp < 0.7 * d["Re_tau"])
    if logr.any():
        law = 2.5 * np.log(yp[logr]) + 5.5
        rec["log_region"] = dict(yplus=[float(yp[logr].min()), float(yp[logr].max())],
                                 max_abs_dev_Uplus=float(np.max(np.abs(Up[logr] - law))),
                                 mean_dev_Uplus=float(np.mean(Up[logr] - law)))
    pts = {}
    for y in (1, 2, 5, 10, 30, 50, 100):
        pts[str(y)] = float(np.interp(y, yp, Up))
    rec["Uplus_at"] = pts
    rec["Re_tau_series_last"] = [round(s["Re_tau"], 2) for s in d["series"][-10:]]
    (run / "retau180_vs_kmm87.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
