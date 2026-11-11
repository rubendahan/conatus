"""A small model of the value semantics of Dedalus field data, and a generator of scripts.

Used by f960 (acceptance) and tests/translate/test_translate_value_semantics.py. It does not
import Conatus or Dedalus. The model is Python itself, run on a stand-in for a Dedalus field:

- ``u['g']`` returns the field's data array itself (a view: Dedalus ``Field.__getitem__``
  returns ``self.data`` after the layout change);
- ``u['g'] = value`` copies ``value`` into that array with broadcasting (Dedalus
  ``Field.__setitem__`` ends with ``np.copyto(self.data, data)``);
- a new field is zero.

Everything else (aliases of arrays and lists, ``+=`` in place, loops, helpers, conditions) is
plain Python. A script is a list of statements over two scalar fields ``u``, ``v`` on a
16-point periodic grid ``x``, a communicator ``comm`` of rank 0 and a number ``rank_threshold``.
"""
from __future__ import annotations

import contextlib
import io
import random
from types import SimpleNamespace

import numpy as np

N = 16
GRID = 2 * np.pi * np.arange(N) / N

# the Dedalus script around the generated statements (also run by Dedalus itself, in a job)
HEAD = """import numpy as np
import dedalus.public as d3
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank_threshold = 0
coords = d3.CartesianCoordinates('x')
dist = d3.Distributor(coords, dtype=np.float64)
xb = d3.RealFourier(coords['x'], size=16, bounds=(0, 2*np.pi), dealias=1)
x = dist.local_grid(xb)
u = dist.Field(name='u', bases=xb)
v = dist.Field(name='v', bases=xb)
"""
TAIL = """
problem = d3.IVP([u, v], namespace=locals())
problem.add_equation('dt(u) = 0')
problem.add_equation('dt(v) = 0')
solver = problem.build_solver(d3.RK222)
solver.stop_iteration = 1
"""


class ModelField:
    """Stand-in for a Dedalus field in grid layout (see the module docstring)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.data = np.zeros(N)

    def __getitem__(self, layout):
        if layout != "g":
            raise KeyError(layout)
        return self.data

    def __setitem__(self, layout, value) -> None:
        if layout != "g":
            raise KeyError(layout)
        np.copyto(self.data, value)


def model_run(statements: list[str]) -> dict[str, np.ndarray]:
    """Final grid data of u and v after the statements, by the model."""
    u, v = ModelField("u"), ModelField("v")
    comm = SimpleNamespace(rank=0, Get_rank=lambda: 0)          # a serial run: rank 0
    ns = {"__name__": "__main__", "np": np, "u": u, "v": v, "x": GRID.copy(), "rank_threshold": 0, "comm": comm,
          "dist": SimpleNamespace(comm=comm)}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile("\n".join(statements) + "\n", "<model>", "exec"), ns)  # noqa: S102
    return {"u": u.data.copy(), "v": v.data.copy()}


def dedalus_run(statements: list[str]) -> dict[str, np.ndarray]:
    """The same with Dedalus (only where it is installed; never imported by conatus)."""
    ns: dict = {"__name__": "__main__"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(HEAD + "\n".join(statements) + "\n", "<dedalus>", "exec"), ns)  # noqa: S102
    out = {}
    for k in ("u", "v"):
        f = ns[k]
        f.change_scales(1)
        out[k] = np.array(f["g"], dtype=float)
    return out


# --------------------------------------------------------------------------- generator
OPS = ["+", "-", "*", "/"]


def _c(r: random.Random) -> str:
    return str(r.choice([0.5, 1.0, 2.0, 3.0, -1.5]))


def generate(seed: int, length: int | None = None) -> list[str]:
    """A random statement sequence: copies, self-reads, data views, in-place operators,
    methods, arrays and lists with aliases, loops, helpers, conditions, MPI guards."""
    r = random.Random(seed)
    have: set[str] = set()
    out: list[str] = []
    n = length or r.randint(2, 7)
    while len(out) < n:
        F, G = r.sample(["u", "v"], 2) if r.random() < 0.7 else (["u", "u"] if r.random() < .5
                                                                else ["v", "v"])
        c, op = _c(r), r.choice(OPS)
        k = r.randint(1, 3)
        pool = [
            f"{F}['g'] = {c}",
            f"{F}['g'] = np.cos({k}*x) + {c}",
            f"{F}['g'] = {G}['g']",
            f"{F}['g'] = {G}['g'] + {c}",
            f"{F}['g'] = {F}['g'] * {c} + {G}['g']",
            f"{F}['g'] {op}= {c}",
            f"{F}['g'] {op}= {G}['g'] + 4",
            f"{F}['g'] = np.ones_like({G}['g']) * {c}",
            f"{F}['g'] = {G}['g'] - np.max({G}['g'])",
            f"{F}['g'] = np.mean({G}['g'])",
            f"{F}['g'] = {r.randint(1, 9)} // {r.randint(1, 4)}",
            f"{F}['g'] = {r.randint(1, 9)} % {r.randint(1, 4)}",
            f"{F}['g'] = {G}['g'][0]",
            f"d = {F}['g']",
            f"{F}['g'].fill({c})",
            f"np.add({F}['g'], {c}, out={F}['g'])",
            f"a = np.zeros(16) + {c}",
            f"s = {c}",
            f"w = {F}\nw['g'] = {c}",
            f"L = [{F}, {G}]\nL[0]['g'] = {c}",
            f"for i in range(2):\n    {F}['g'] += 1",
            f"def h(f):\n    f['g'] = {c}\nh({F})",
            f"if rank_threshold > 1:\n    {F}['g'] = {c}",
            f"if comm.rank == 0:\n    {F}['g'] = {c}",
            f"if comm.rank == 0:\n    print({F}['g'].max())",
            f"q = [1.0]\nqq = q\nqq += [{c}]\n{F}['g'] = len(q)",
        ]
        if "d" in have:
            pool += ["d += 1", f"d[:] = {c}", f"{F}['g'] = d", f"{F}['g'] = 7\n{G}['g'] = d"]
        if "a" in have:
            pool += ["b = a", f"{F}['g'] = a", f"a {op}= {c}"]
        if "b" in have:
            pool += [f"b {op}= {c}", f"{F}['g'] = b"]
        if "s" in have:
            pool += [f"s {op}= {c}", f"{F}['g'] = s"]
        st = r.choice(pool)
        for name in ("d", "a", "b", "s"):
            if st.startswith(f"{name} = "):
                have.add(name)
        out.append(st)
    return out


# --------------------------------------------------------------------------- comparison
def classify(statements: list[str], cn, errors) -> tuple[str, str]:
    """Translate the script and compare its initial state with the model.

    Returns ("equal", ""), ("incomplete", "") when complete=False and ivp_solver() refuses
    naming a line, ("refused", <type>) for a typed refusal of from_dedalus, or ("BAD", why) for
    anything else: complete=True with another state, an untyped error, a silent solver."""
    exp = model_run(statements)
    try:
        tr = cn.from_dedalus(HEAD + "\n".join(statements) + TAIL)
    except errors.ConatusError as e:
        return "refused", type(e).__name__
    except Exception as e:  # noqa: BLE001 -- any other exception is the finding
        return "BAD", f"from_dedalus raised {type(e).__name__}: {e}"
    if tr.complete:
        try:
            init = tr.initial_state()
            got = {}
            for f in tr.problem.variables:
                g = init.get(f)
                got[f.name] = np.zeros(N) if g is None else np.asarray(g(GRID), dtype=float)
        except Exception as e:  # noqa: BLE001 -- a late failure after complete=True
            return "BAD", f"complete=True, then {type(e).__name__}: {str(e)[:120]}"
        for k in ("u", "v"):
            if got[k].shape != exp[k].shape or not np.allclose(got[k], exp[k], rtol=1e-12,
                                                                atol=1e-12):
                return "BAD", (f"complete=True but {k}[:3] = {np.round(got[k][:3], 6)}, "
                               f"model {np.round(exp[k][:3], 6)}")
        return "equal", ""
    if not tr.lines("unsupported"):
        return "BAD", "complete=False without an unsupported line"
    try:
        tr.ivp_solver(dt=0.1)
    except errors.UnsupportedError as e:
        return ("incomplete", "") if "line " in str(e) else ("BAD", "refusal names no line")
    except Exception as e:  # noqa: BLE001 -- the wrong kind of refusal
        return "BAD", f"ivp_solver on an incomplete translation raised {type(e).__name__}"
    return "BAD", "ivp_solver() accepted an incomplete translation without allow_incomplete"


# --------------------------------------------------------------------------- stop rules
# dt(u) = 1 from u = 0 with a fixed dt = 1/8 (binary-exact): u = t after the run. Dedalus's
# solver.proceed is false as soon as any limit is reached (sim_time >= stop_sim_time or
# iteration >= stop_iteration); a loop condition adds its own test.
STOP_HEAD = """import numpy as np
import dedalus.public as d3
coords = d3.CartesianCoordinates('x')
dist = d3.Distributor(coords, dtype=np.float64)
xb = d3.RealFourier(coords['x'], size=16, bounds=(0, 2*np.pi), dealias=1)
u = dist.Field(name='u', bases=xb)
u['g'] = 0
problem = d3.IVP([u], namespace=locals())
problem.add_equation('dt(u) = 1')
solver = problem.build_solver(d3.RK222)
"""
STOP_DT = 0.125


def stop_cases() -> list[tuple]:
    """(stop_sim_time, stop_iteration, extra loop test) with at least one finite limit."""
    out = []
    for t in (None, 1.0, 0.5, np.inf):
        for i in (None, 4, 16):
            for extra in (None, "solver.iteration < 6", "solver.sim_time < 0.75"):
                finite = (t is not None and np.isfinite(t)) or i is not None
                if finite or extra is not None:
                    out.append((t, i, extra))
    return out


def stop_script(t, i, extra) -> str:
    lines = []
    if t is not None:
        lines.append(f"solver.stop_sim_time = {'np.inf' if np.isinf(t) else t}")
    if i is not None:
        lines.append(f"solver.stop_iteration = {i}")
    test = "solver.proceed" + (f" and {extra}" if extra else "")
    return STOP_HEAD + "\n".join(lines) + f"\nwhile {test}:\n    solver.step({STOP_DT})\n"


def stop_model(t, i, extra) -> float:
    """Final time by the model of Dedalus's loop (every limit checked before each step)."""
    sim, it = 0.0, 0
    while True:
        if t is not None and sim >= t:
            break
        if i is not None and it >= i:
            break
        if extra == "solver.iteration < 6" and not it < 6:
            break
        if extra == "solver.sim_time < 0.75" and not sim < 0.75:
            break
        sim += STOP_DT
        it += 1
    return sim


def stop_dedalus(t, i, extra) -> float:
    ns: dict = {}
    exec(compile(stop_script(t, i, extra), "<dedalus>", "exec"), ns)  # noqa: S102
    return float(ns["solver"].sim_time)


def stop_classify(t, i, extra, cn, errors) -> tuple[str, str]:
    """As classify, for the stop rules: complete=True must run to the model's time, u = t."""
    exp = stop_model(t, i, extra)
    try:
        tr = cn.from_dedalus(stop_script(t, i, extra))
    except errors.ConatusError as e:
        return "refused", type(e).__name__
    except Exception as e:  # noqa: BLE001 -- any other exception is the finding
        return "BAD", f"from_dedalus raised {type(e).__name__}: {e}"
    if not tr.complete:
        if not tr.lines("unsupported") and not tr.lines("approximated"):
            return "BAD", "complete=False without a reason line"
        return "incomplete", ""
    n = tr.steps()
    if n is None:
        return "BAD", "complete=True without a number of steps"
    s = tr.ivp_solver()
    s.run(n)
    u = float(np.mean(s.grid("u")))
    if abs(s.t - exp) > 1e-12 or abs(u - exp) > 1e-10:
        return "BAD", f"complete=True but t = {s.t}, u = {u}; Dedalus model t = u = {exp}"
    return "equal", ""
