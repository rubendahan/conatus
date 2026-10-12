"""Compare the long Re_tau ~ 180 run (run_retau180_long.py -> retau180_long.json) with Kim, Moin &
Moser (1987, JFM 177, 133-166; scratch/refs/KimMoinMoser1987_JFM.pdf). References, with their
place in the paper (read on the page images):
  * Table 1 (p. 143): Re_tau ~ 180, U_m/u_tau = 15.63, U_c/u_tau = 18.20, U_c/U_m = 1.16,
    C_f = 8.18e-3;
  * Fig. 4 (p. 142): U+ = y+ (y+ < 5), U+ = 2.5 ln y+ + 5.5;
  * p. 144: maximum of u_rms at y+ ~ 12; Fig. 6(b) (p. 143): peak u_rms+ ~ 2.65 (figure reading,
    +-0.05), Fig. 6(a): centreline u_rms+ ~ 0.8, v_rms+ ~ w_rms+ ~ 0.65 (figure reading);
  * p. 146: total shear stress -u'v'+ + dU+/dy+ = 1 - y+/delta+ in the equilibrium state (exact
    consequence of the mean momentum balance: checked as a residual, no figure reading).
Both walls are folded (y -> -y) and averaged. Usage: python analyze_retau180_long.py <run_dir>"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial import chebyshev as C

KMM = dict(Re_tau=180.0, Ub_over_utau=15.63, Uc_over_utau=18.20, Uc_over_Ub=1.16, Cf=8.18e-3)
KMM_FIG = dict(urms_peak=2.65, urms_peak_yplus=12.0, urms_centre=0.8, vrms_centre=0.65,
               wrms_centre=0.65)


def main():
    run = Path(sys.argv[1])
    d = json.loads((run / "retau180_long.json").read_text())
    st = d["stats"]
    Re, Q = d["Re"], d["flow_rate"]
    y = np.array(d["y"])
    Uc = np.array(st["U_cheb"])
    dU = C.chebder(Uc)
    tw_lo, tw_hi = C.chebval(-1.0, dU) / Re, -C.chebval(1.0, dU) / Re
    u_tau = float(np.sqrt(0.5 * (tw_lo + tw_hi)))
    Re_tau = u_tau * Re
    lower = y < 0
    yp = (1 + y[lower]) * Re_tau                      # wall units from the lower wall
    fold = lambda a: 0.5 * (np.asarray(a)[lower] + np.asarray(a)[::-1][lower])  # noqa: E731
    Up = fold(st["U"]) / u_tau                          # U even in y
    urms = np.sqrt(fold(st["uu"])) / u_tau
    vrms = np.sqrt(fold(st["vv"])) / u_tau
    wrms = np.sqrt(fold(st["ww"])) / u_tau
    uv = 0.5 * (np.asarray(st["uv"])[lower] - np.asarray(st["uv"])[::-1][lower]) / u_tau**2
    dUp = 0.5 * (C.chebval(y[lower], dU) - C.chebval(-y[lower], dU)) / (Re * u_tau**2)  # odd
    total = -uv + dUp                                   # -u'v'+ + dU+/dy+
    stress_res = float(np.max(np.abs(total - (1 - yp / Re_tau))))
    ours = dict(Re_tau=Re_tau, Ub_over_utau=Q / u_tau, Uc_over_utau=float(C.chebval(0.0, Uc)) / u_tau,
                Cf=2 * u_tau**2 / Q**2)
    ours["Uc_over_Ub"] = ours["Uc_over_utau"] / ours["Ub_over_utau"]
    ip = int(np.argmax(urms))
    rec = dict(run_id=d.get("run_id", run.name), status=d["status"], t=d["t"],
               n_average=d["n_average"], averaging_time=st["averaging_time"],
               averaging_in_viscous_units=st["averaging_time"] * u_tau,
               grid=d["grid"], box=d["box"], Re_tau_walls=[tw_lo**0.5 * Re, tw_hi**0.5 * Re],
               dxplus=d["box"][0] / d["grid"][0] * Re_tau, dzplus=d["box"][2] / d["grid"][2] * Re_tau)
    rec["compare_table1"] = {k: dict(conatus=ours[k], kmm87=KMM[k], rel_diff=ours[k] / KMM[k] - 1)
                             for k in KMM}
    rec["rms"] = dict(urms_peak=float(urms[ip]), urms_peak_yplus=float(yp[ip]),
                      urms_centre=float(urms[-1]), vrms_centre=float(vrms[-1]),
                      wrms_centre=float(wrms[-1]), vrms_max=float(vrms.max()),
                      wrms_max=float(wrms.max()), uv_min=float(uv.min()),
                      uv_min_yplus=float(yp[int(np.argmin(uv))]),
                      kmm87_figure_readings=KMM_FIG)
    rec["total_stress_residual_max"] = stress_res
    sub = yp < 5
    rec["sublayer_max_rel_dev"] = float(np.max(np.abs(Up[sub] / yp[sub] - 1)))
    logr = (yp > 30) & (yp < 0.7 * Re_tau)
    law = 2.5 * np.log(yp[logr]) + 5.5
    rec["log_region"] = dict(max_abs_dev=float(np.max(np.abs(Up[logr] - law))),
                             mean_dev=float(np.mean(Up[logr] - law)))
    rec["profiles"] = dict(yplus=yp.tolist(), Uplus=Up.tolist(), urms=urms.tolist(),
                           vrms=vrms.tolist(), wrms=wrms.tolist(), uv=uv.tolist())
    (run / "retau180_long_vs_kmm87.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "profiles"}, indent=1))


if __name__ == "__main__":
    main()
