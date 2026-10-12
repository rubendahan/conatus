"""The "NOT alias-free" warning is for functions of grid data only (gallery, BACKLOG wave 1).

cos(omega t) or sqrt(1 - q^2) with t, omega, q parameters is a number: it is evaluated exactly and
has no aliasing error. The warning (explain() of the problem and of the node, methods, MMS report)
used to fire for them too (false positive); it must still fire for cos(u) and for tanh(Y) of a
coefficient field.
"""
import numpy as np

from conatus.expert import (IVP, CartesianCoordinates, Chebyshev, Domain, Field, Parameter, cos, diff,
                            dt, has_nonpolynomial, interp, sqrt, tanh)


def _problem(rhs_of):
    cs = CartesianCoordinates("y")
    y = cs["y"]
    D = Domain(cs, [Chebyshev(y, 8, (-1, 1), dealias=1)])
    u, Y = Field("u", D), Field("Y", D)
    t, q = Parameter("t", 0.0), Parameter("q", 0.5)
    pb = IVP([u])
    pb.add_equation((dt(u) - diff(diff(u, y), y), rhs_of(u, Y, t, q)))
    pb.add_bc(interp(u, y, -1), interp(u, y, 1))
    return pb


def test_function_of_parameters_is_not_flagged():
    for rhs in (lambda u, Y, t, q: cos(t), lambda u, Y, t, q: sqrt(1 - q ** 2) * cos(2 * t)):
        pb = _problem(rhs)
        assert "alias" not in pb.explain(), pb.explain()
        assert has_nonpolynomial(pb.equations[0].rhs) == []
    assert all("not alias-free" not in n for n in cos(Parameter("t", 0.0))._notes())


def test_function_of_grid_data_is_still_flagged():
    for rhs, name in ((lambda u, Y, t, q: cos(u), "cos"), (lambda u, Y, t, q: tanh(Y) * cos(t), "tanh")):
        pb = _problem(rhs)
        assert "NOT alias-free" in pb.explain()
        assert has_nonpolynomial(pb.equations[0].rhs) == [name]
    assert np.isfinite(1.0)
