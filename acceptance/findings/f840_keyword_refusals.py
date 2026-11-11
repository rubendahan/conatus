"""F840 (hunt 5E, R1): every public callable refuses a former or unknown keyword with a typed error.

Oracle: the renames written in AGENTS.md section 5 (copied below, not read from the package) and
the parameters of each function's own signature (``inspect.signature`` of the undecorated
function). A former name must give a ConatusError naming the new one; a made-up keyword must give
ArgumentError (CN151, also a TypeError) whose fix lists every accepted name. Control: the new name
is accepted by Simulation.run.
"""
import importlib
import inspect

import numpy as np
from _acceptance import check, passed, start

start(840, "former and unknown keywords are refused with the accepted names")
import conatus as cn  # noqa: E402
import conatus.expert as X  # noqa: E402
from conatus.errors import ArgumentError, ConatusError, NameNotExportedError  # noqa: E402

RENAMES = {"nmodes": "n_modes", "X0": "x0", "xatol": "xtol", "maxit": "max_iterations",
           "nsteps": "steps", "radius_ratio": "eta"}            # AGENTS.md, section 5, rule 2


def surface():
    spaces = [(cn, cn.__all__), (X, X.__all__)]
    for m in ("model", "problems", "problems.native", "runs", "io", "resolution", "tolerances"):
        mod = importlib.import_module(f"conatus.{m}")
        spaces.append((mod, getattr(mod, "__all__", [])))
    seen = set()
    for mod, names in spaces:
        for n in names:
            obj = getattr(mod, n, None)
            if inspect.isclass(obj) and not issubclass(obj, BaseException):
                for k, v in list(vars(obj).items()):
                    if (k == "__init__" or not k.startswith("_")) and inspect.isfunction(v) \
                            and id(v) not in seen:
                        seen.add(id(v))
                        yield f"{n}.{k}", v
            elif inspect.isfunction(obj) and id(obj) not in seen:
                seen.add(id(obj))
                yield n, obj


n_former = n_unknown = 0
for label, f in surface():
    sig = inspect.signature(inspect.unwrap(f))
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        continue
    names = [n for n, p in sig.parameters.items() if n not in ("self", "cls")
             and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    for old, new in RENAMES.items():
        if new in names and old not in names:
            n_former += 1
            try:
                f(**{old: 1})
                check(False, f"{label}({old}=) accepted")
            except ConatusError as err:
                check(new in str(err), f"{label}({old}=): {str(err).splitlines()[0]}")
            except Exception as err:  # noqa: BLE001
                check(False, f"{label}({old}=): untyped {type(err).__name__}: {err}")
    n_unknown += 1
    try:
        f(zz_not_a_keyword=1)
        check(False, f"{label}(zz_not_a_keyword=) accepted")
    except ArgumentError as err:
        check(isinstance(err, TypeError) and err.error_code == "CN151", f"{label}: class")
        missing = [n for n in names if n not in err.fix]
        check(not missing, f"{label}: the fix omits {missing}")
    except Exception as err:  # noqa: BLE001
        check(False, f"{label}(zz_not_a_keyword=): {type(err).__name__}: {str(err)[:150]}")
check(n_former > 40 and n_unknown > 300, f"too few callables checked: {n_former}, {n_unknown}")

m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="heat", nu=0.1)
m.scalar("u")
m.equations("dt(u) - nu*dx(dx(u)) = 0")
sim = m.simulate(N=16, scheme="RK443", dt=0.01)
sim.initial(u=lambda x: np.sin(x))
try:
    sim.run(nsteps=3)
    check(False, "run(nsteps=) accepted")
except ArgumentError as err:
    check("'steps'" in str(err), str(err))
sim.run(steps=3)
check(sim.iteration == 3, "run(steps=3) control")
try:
    X.ShootingRPO                                  # noqa: B018
    check(False, "X.ShootingRPO resolved")
except NameNotExportedError as err:
    check("conatus.ecs.shooting" in str(err) and isinstance(err, AttributeError), str(err))
passed(f"{n_former} former names and {n_unknown} unknown keywords refused")
