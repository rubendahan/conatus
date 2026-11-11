"""F964 (external synthesis of 0.2.3, P1): every stop rule of the Dedalus loop is honoured.

Oracle: a Python model of Dedalus's solver.proceed (every limit checked before each step),
which agreed with Dedalus 3.0.5 on all cases (tests/translate/check_value_model_dedalus.py).
dt(u) = 1 from 0 with dt = 1/8: stop_sim_time = 1 with stop_iteration = 16 stops at t = u = 1
(the translation ran 16 steps to 2 with complete=True). A loop test besides solver.proceed is
unsupported. Each combination translates exactly or reports complete=False.
"""
import warnings

from _acceptance import check, passed, start
from _value_model import stop_cases, stop_classify

start(964, "Dedalus stop rules: the first limit, and loop conditions")
import conatus as cn  # noqa: E402
from conatus import errors  # noqa: E402

warnings.simplefilter("ignore")
cat, why = stop_classify(1.0, 16, None, cn, errors)
check(cat == "equal", f"stop_sim_time=1 with stop_iteration=16: {cat} {why}")
cat, why = stop_classify(None, 16, "solver.iteration < 8", cn, errors)
check(cat == "incomplete", f"extra loop test: {cat} {why}")
counts: dict = {}
for case in stop_cases():
    cat, why = stop_classify(*case, cn, errors)
    counts[cat] = counts.get(cat, 0) + 1
    check(cat != "BAD", f"{case}: {why}")
check(counts.get("equal", 0) >= 8, f"too few exact translations: {counts}")
passed(f"{len(stop_cases())} stop-rule combinations: {counts}")
