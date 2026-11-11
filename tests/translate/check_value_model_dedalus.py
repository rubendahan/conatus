"""Cross-check of the field-data model (_value_model.py) against Dedalus itself, in a job.

Not collected by pytest (Dedalus is imported here, never by python/conatus). Run:
    python tests/translate/check_value_model_dedalus.py [n_scripts]
Prints one line per disagreement and a final count; exit status 1 on any disagreement.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "_value_model", Path(__file__).resolve().parents[2] / "packaging" / "acceptance" / "findings"
    / "_value_model.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
bad = 0
for seed in range(n):
    st = M.generate(seed)
    a, b = M.model_run(st), M.dedalus_run(st)
    if not all(np.allclose(a[k], b[k], rtol=1e-13, atol=1e-13) for k in "uv"):
        bad += 1
        print("DISAGREE", seed, st, a, b)
for case in M.stop_cases():
    a, b = M.stop_model(*case), M.stop_dedalus(*case)
    if abs(a - b) > 1e-12:
        bad += 1
        print("DISAGREE stop", case, a, b)
print(f"model vs Dedalus: {n} scripts and {len(M.stop_cases())} stop rules, {bad} disagreements")
sys.exit(1 if bad else 0)
