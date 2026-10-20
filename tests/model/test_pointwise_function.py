"""User pointwise functions (m.function / conatus.ir.pointwise; the role of Dedalus'
GeneralFunction for pointwise functions of one scalar field).

Checks: a user function equal to a built-in (np.sin, derivative np.cos) gives the same IVP
trajectory bit for bit as sin(u) and the same linearized spectrum; a function without
derivative cannot be linearized (refused with the reason); name collisions are refused; the
node has no string form (declared); explain() lists it.
"""
import numpy as np
import pytest

import conatus as cn
from conatus.errors import InputTypeError


def _model(rhs, fn=None, **kw):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), z=cn.walls(0, 1)), name="reaction-diffusion")
    u = m.scalar("u")
    if fn is not None:
        m.function(*fn, **kw)
    m.equations(f"dt(u) - lap(u) = {rhs}")
    m.at("walls", u=0)
    return m, u


def test_same_as_builtin_bit_for_bit():
    # "g" on purpose: user function names are scoped to their model (F105), so the guide's "g" =
    # tanh registered by another test in the same process does not interfere
    sims = []
    for rhs, fn in (("-sin(u)", None), ("-g(u)", ("g", np.sin))):
        m, u = _model(rhs, fn, **({"derivative": np.cos, "parity": "odd"} if fn else {}))
        sim = m.simulate(N=(8, 16), scheme="RK222", dt=1e-3)
        sim.initial(u=lambda x, z: np.sin(np.pi * z) * (1 + 0.5 * np.cos(x)))
        sim.run(steps=10)
        sims.append((m, u, sim))
    assert np.array_equal(sims[0][2].solver.X, sims[1][2].solver.X)
    ev = []
    for m, u, _ in sims:
        lin = m.linearize(about={u: lambda x, z: 0.3 * np.sin(np.pi * z)})
        v = np.asarray(lin.eigenvalues(16, k=1.0).values)
        ev.append(np.sort_complex(v[np.isfinite(v)]))
    assert np.array_equal(ev[0], ev[1])
    assert "Pointwise function: g(u)" in sims[1][0].explain()


def test_without_derivative_and_refusals():
    m, u = _model("-h(u)", ("h", lambda a: a ** 3 / (1 + a ** 2)))
    sim = m.simulate(N=(8, 16), scheme="RK222", dt=1e-3)
    sim.initial(u=lambda x, z: np.sin(np.pi * z))
    sim.run(steps=2)
    with pytest.raises(NotImplementedError, match="without derivative"):
        m.linearize(about={u: 0}).eigenvalues(16, k=1.0)
    m2 = cn.Model(cn.Box(z=cn.walls(0, 1)))
    with pytest.raises(InputTypeError, match="reserved"):          # a built-in name
        m2.function("tanh", np.tanh)
    m2.function("h", np.tanh)                          # another model: its own h (F105)
    with pytest.raises(InputTypeError, match="already declared"):
        m2.function("h", np.sin)                       # the same model: one meaning per name
    from conatus.ir.parse import to_string
    from conatus.errors import UnsupportedError
    with pytest.raises(UnsupportedError, match="no string form"):
        to_string(m.compile((8, 16)).problem.equations[0])


def test_derivative_name_never_overwrites_a_user_function():
    """pointwise("q", f, derivative=d) registers the node
    "q_prime"; it used to overwrite a user function already registered as "q_prime" (every
    expression using it then computed d silently). Since F105 every function has its own key:
    the first function is kept and the derivative node is another one."""
    from conatus.ir import pointwise
    from conatus.ir.expr_algebra import _FUNCS
    def first(a):
        return a + 1.0
    # the builders are kept: a declaration nothing uses is dropped from the registry (F5)
    g1 = pointwise("qrev_prime", first)
    g2 = pointwise("qrev", np.sin, derivative=np.cos)
    assert g1 and g2
    assert _FUNCS["qrev_prime"][0] is first
    assert any(k.startswith("qrev_prime__") and v[0] is np.cos for k, v in _FUNCS.items())


def test_f105_function_names_are_scoped_to_their_model():
    """F105 (-10-19): two models, each with its own g, both run with their own g
    (the second one was refused: the registry was process-wide)."""
    def model(fn):
        m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
        m.scalar("u")
        m.function("g", fn)
        m.equations("dt(u) = g(u)")
        return m.simulate(8, scheme="RK443", dt=0.01).initial(u=1)
    a, b = model(lambda u: 0 * u + 1.0), model(lambda u: 0 * u + 2.0)
    a.run(steps=10)
    b.run(steps=10)
    assert abs(np.mean(a.grid("u")) - 1.1) < 1e-12 and abs(np.mean(b.grid("u")) - 1.2) < 1e-12
    assert "g(u)" in str(b.compiled.problem.equations[0]) and "__" not in str(b.compiled.problem.equations[0])


_HASH_SCRIPT = """
import numpy as np, conatus as cn
from conatus.io import _identity
if {earlier}:
    m0 = cn.Model(cn.Box(x=cn.periodic(2 * np.pi))); m0.scalar("u")
    m0.function("g", np.sin); m0.equations("dt(u) = g(u)")
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi))); m.scalar("u")
m.function("g", np.tanh); m.equations("dt(u) = g(u) + u*g(u)")
pb = m.compile(8).problem
print(_identity.equations_hash(pb), sorted(_identity.user_functions(pb)[0]))
"""


def test_f105_identity_does_not_depend_on_earlier_declarations():
    """Numerics review of F105: the registry key of a user function (g__2 after another model
    declared its own g) entered the structural hash, so the same model had another equations
    hash, term order and computation identity in a fresh process than in a notebook that had
    declared a g before (cache misses, a checkpoint or reproduce() refused as "numerics
    differ"). The hash and the identity use the user's name."""
    import os
    import subprocess
    import sys
    out = [subprocess.run([sys.executable, "-c", _HASH_SCRIPT.format(earlier=e)], check=True,
                          capture_output=True, text=True, env=dict(os.environ)).stdout.strip()
           for e in (False, True)]
    assert out[0] == out[1] and "['g']" in out[0]
