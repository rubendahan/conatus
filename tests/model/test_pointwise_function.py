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
    with pytest.raises(InputTypeError, match="another function"):
        m2.function("h", np.tanh)                      # 'h' already means the function above
    from conatus.ir.parse import to_string
    from conatus.errors import UnsupportedError
    with pytest.raises(UnsupportedError, match="no string form"):
        to_string(m.compile((8, 16)).problem.equations[0])


def test_derivative_name_never_overwrites_a_user_function():
    """pointwise("q", f, derivative=d) registers the node
    "q_prime"; it used to overwrite a user function already registered as "q_prime" (every
    expression using it then computed d silently). Refused now, and the first function is kept."""
    from conatus.ir import pointwise
    from conatus.ir.expr_algebra import _FUNCS
    def first(a):
        return a + 1.0
    pointwise("qrev_prime", first)
    with pytest.raises(ValueError, match="'qrev_prime' of its derivative"):
        pointwise("qrev", np.sin, derivative=np.cos)
    assert _FUNCS["qrev_prime"][0] is first
