"""F960 (review of 0.2.3 G4, pass 4 H6): field data keeps program order and value semantics.

Oracle: _value_model.py, Python run on a stand-in for a Dedalus field (u['g'] is the data array
itself; u['g'] = value copies). The model agreed with Dedalus 3.0.5 on every generated script
of a job (tests/translate/check_value_model_dedalus.py). Each script either translates with
complete=True and the model's initial state, or has complete=False and ivp_solver() refuses
naming a line, or from_dedalus refuses with a typed error.
"""
import warnings

from _acceptance import check, passed, start
from _value_model import classify, generate

start(960, "translated field data: program order, self-reads, views, in-place changes")
import conatus as cn  # noqa: E402
from conatus import errors  # noqa: E402

warnings.simplefilter("ignore")
EXACT = {   # must translate with complete=True and the model's values
    "copy then change": ["u['g'] = 1", "v['g'] = u['g']", "u['g'] += 1"],
    "read then overwrite": ["u['g'] = 1", "v['g'] = u['g'] + 1", "u['g'] = 5"],
    "self-read": ["u['g'] = 1", "u['g'] = u['g'] + 2"],
    "ones_like of itself": ["u['g'] = 2", "u['g'] = np.ones_like(u['g']) * 3"],
    "max of itself": ["u['g'] = np.cos(x)", "u['g'] = u['g'] - np.max(u['g'])"],
    "array alias +=": ["a = np.zeros(16)", "b = a", "b += 1", "u['g'] = a"],
    "array changed after use": ["a = np.zeros(16)", "u['g'] = a", "a += 1", "v['g'] = a"],
    "list alias +=": ["q = [1.0]", "qq = q", "qq += [2.0]", "u['g'] = len(q)"],
    "floor division": ["u['g'] = 3 // 2", "v['g'] = 7 % 4"],
}
HONEST = {  # Dedalus semantics out of reach: complete=False and a named refusal
    "view +=": ["u['g'] = 1", "d = u['g']", "d += 1"],
    "view [:] =": ["u['g'] = 1", "d = u['g']", "d[:] = 2"],
    "fill": ["u['g'] = 1", "u['g'].fill(3)"],
    "out=": ["u['g'] = 1", "np.add(u['g'], 1, out=u['g'])"],
    "view read after change": ["u['g'] = 1", "d = u['g']", "u['g'] = 7", "v['g'] = d"],
    "loop": ["for i in range(3):\n    u['g'] += 1"],
    "helper": ["def h(f):\n    f['g'] = 5\nh(u)"],
    "index of scalar data": ["u['g'] = 0", "u['g'][0] = 16"],
    "mean": ["u['g'] = np.mean(np.cos(x)**2)"],
}
for name, st in EXACT.items():
    cat, why = classify(st, cn, errors)
    check(cat == "equal", f"{name}: {cat} {why}")
for name, st in HONEST.items():
    cat, why = classify(st, cn, errors)
    check(cat in ("incomplete", "refused"), f"{name}: {cat} {why}")
counts: dict = {}
for seed in range(400):
    cat, why = classify(generate(seed), cn, errors)
    counts[cat] = counts.get(cat, 0) + 1
    check(cat != "BAD", f"generated script {seed}: {why}")
check(counts.get("equal", 0) >= 60, f"too few exact translations to mean anything: {counts}")
passed(f"{len(EXACT)} exact, {len(HONEST)} refused, 400 generated: {counts}")
