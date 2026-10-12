"""Tutorial 14b -- check the translation of rb_cell_dedalus.py against Dedalus itself.

Expected values (checked at the end of the script):
  the Dedalus script rb_cell_dedalus.py is run twice: by Dedalus (imported here at run time, and
  only here), and by Conatus through cn.from_dedalus. Same bases, tau terms, lifts, scheme, fixed
  dt, start-up and dealiasing, so after 40 steps of SBDF2 the temperature deviation and the
  velocity agree to rounding: below 1e-10 relative.
  Without Dedalus the comparison is skipped and the script says so (exit 0).
Run: python t14b_dedalus_check.py   (under a minute of CPU, most of it importing Dedalus)
"""
# [imports]
import logging
import os
import runpy

import numpy as np

import conatus as cn
# [/imports]

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rb_cell_dedalus.py")

# [conatus]
tr = cn.from_dedalus(SCRIPT)
solver = tr.ivp_solver(startup="lower_order")      # Dedalus's multistep start-up, said explicitly
solver.run(tr.steps())
th_cn, u_cn = solver.grid("th"), solver.grid("u")
print(f"Conatus: t = {solver.t:.3f} after {tr.steps()} steps")
# [/conatus]

# [dedalus]
os.environ.setdefault("OMP_NUM_THREADS", "1")      # Dedalus asks for it before its import
try:
    import dedalus  # noqa: F401  (comparison oracle only; the conatus package never imports it)
    have_dedalus = True
except ImportError:
    have_dedalus = False
    print("Dedalus is not installed here: the comparison with Dedalus is skipped (declared)")

if have_dedalus:
    logging.disable(logging.INFO)                   # Dedalus's per-step log lines
    ns = runpy.run_path(SCRIPT, run_name="__main__")   # the script, run by Dedalus as written
    out = {}
    for name in ("th", "u"):
        ns[name].change_scales(1)                   # back to the grid without 3/2 padding
        out[name] = np.array(ns[name]["g"])
    rel_th = np.abs(out["th"] - th_cn).max() / np.abs(out["th"]).max()
    rel_u = np.abs(out["u"] - u_cn).max() / np.abs(out["u"]).max()
    print(f"Dedalus: t = {ns['run'].sim_time:.3f}; relative difference th {rel_th:.1e}, "
          f"u {rel_u:.1e}")
# [/dedalus]

# [check]
assert tr.counts()["unsupported"] == 0 and tr.steps() == 40
if have_dedalus:
    assert rel_th < 1e-10 and rel_u < 1e-10, (rel_th, rel_u)
print("PASS")
# [/check]
