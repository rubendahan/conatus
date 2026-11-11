"""F963 (pass 4 H14): valid scripts get UnsupportedError naming the line, never another error.

Oracle: Python and Dedalus's rule of one equation per variable. Each script below is refused
with a typed error (UnsupportedError or another ConatusError) at from_dedalus or at the solver,
or translated with complete=True and the right value; no KeyError, TypeError, ValueError,
RecursionError or private exception reaches the caller.
"""
import warnings

from _acceptance import check, passed, start
from _value_model import HEAD, classify

start(963, "typed refusals of the translator on valid scripts")
import conatus as cn  # noqa: E402
from conatus import errors  # noqa: E402

warnings.simplefilter("ignore")
DATA = {
    "floor division": ["u['g'] = 3 // 2"],
    "component of scalar data": ["u['g'] = 0", "u['g'][0] = 16"],
    "mean of a grid": ["u['g'] = np.mean(np.cos(x)**2)"],
    "grid method": ["y = x", "u['g'] = y.mean()"],
    "helper value": ["def mk():\n    return 1\nu['g'] = mk()"],
    "lambda in data": ["f = lambda z: z\nu['g'] = f(x)"],
    "conditional expression": ["a = 1", "u['g'] = 2 if a > 0 else 3"],
}
for name, st in DATA.items():
    cat, why = classify(st, cn, errors)
    check(cat != "BAD", f"{name}: {why}")
EQS = {
    "constant on the left": "problem.add_equation('dt(u) + 1 = 2')\nproblem.add_equation('dt(v) = 0')",
    "loop of equations": "for e in ['dt(u) = 0', 'dt(v) = 0']:\n    problem.add_equation(e)",
    "d3 name in the text": "problem.add_equation('dt(u) = d3.Integrate(u, coords)')\n"
                           "problem.add_equation('dt(v) = 0')",
}
HEADEQ = HEAD + "problem = d3.IVP([u, v], namespace=locals())\n"
for name, eqs in EQS.items():
    tr = cn.from_dedalus(HEADEQ + eqs + "\nsolver = problem.build_solver(d3.RK222)\n")
    check(not tr.complete, f"{name}: complete=True")
    for allow in (False, True):
        try:
            tr.ivp_solver(dt=0.1, allow_incomplete=allow)
            check(False, f"{name}: solver built")
        except errors.ConatusError as e:
            check("line" in str(e) or "equation" in str(e), f"{name}: refusal names nothing: {e}")
a = 2.5
tr = cn.from_dedalus(HEADEQ + f"a = {a}\nproblem.add_equation(f'dt(u) = {{a}}*u')\n"
                     "problem.add_equation('dt(v) = 0')\nsolver = problem.build_solver(d3.RK222)\n")
check(tr.complete and "2.5" in str(tr.problem.equations[0]),
      f"f-string equation: complete={tr.complete}, {tr.problem.equations[:1]}")
passed(f"{len(DATA)} data statements and {len(EQS)} equation sets refused with typed errors; "
       "f-string equation translated")
