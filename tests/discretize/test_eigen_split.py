"""EVP: the eigenvalue may multiply a sum, (U - c) L(v), as long as the pencil stays linear in it.

Gallery gap: ``(U - c) * L(v)`` was refused ("eigenvalue must appear linearly")
although c enters linearly; worse, ``U * (L(v) - c v)`` compiled the inner ``c v`` into neither L nor
M (the product's M part was dropped because the outer factor had no c): in a system whose other
equations carry c, that term vanished silently. The compiler
now splits every LHS as x0 + c x1 (products of sums distributed) and refuses any other dependence.
"""
import numpy as np
import pytest

from conatus.expert import EVP, Parameter, channel, conj, diff, discretize, exp, interp
from conatus.discretize.compiler import DiscretizationError, eigen_split


def _pair(eq0, eq1):
    """Two-equation pencil in (v, b) with U = 2 + y, S = 1 + y^2 and both matrices returned."""
    ch = channel(16, base=None, interval=(-1.0, 1.0))
    y, v, b, U, S = ch.y, ch.field("v"), ch.field("b"), ch.field("U"), ch.field("S")
    c = Parameter("c", dtype=complex)
    pb = EVP([v, b], eigenvalue=c)
    pb.add_equation((eq0(y, v, b, U, S, c), 0))
    pb.add_equation((eq1(y, v, b, U, S, c), 0))
    pb.add_bc(interp(v, y, -1.0), interp(v, y, 1.0), into=0)
    gm = discretize(pb, ncc={U: lambda Y: 2 + Y, S: lambda Y: 1 + Y ** 2}).matrices()
    return gm.L.toarray(), gm.M.toarray()


def L2(f, y):
    return diff(diff(f, y), y) - 0.25 * f


def test_factored_eigenvalue_equals_expanded():
    """(U - c) L2(v) - U'' v - i a b and i a (U - c) b + S v: identical matrices to the expanded form."""
    fac = _pair(lambda y, v, b, U, S, c: (U - c) * L2(v, y) - diff(diff(U, y), y) * v - 0.5j * b,
                lambda y, v, b, U, S, c: 0.5j * (U - c) * b + S * v)
    exp_ = _pair(lambda y, v, b, U, S, c: U * L2(v, y) - c * L2(v, y) - diff(diff(U, y), y) * v - 0.5j * b,
                 lambda y, v, b, U, S, c: 0.5j * U * b - 0.5j * c * b + S * v)
    for A, B in zip(fac, exp_):
        assert np.array_equal(A, B)


def test_eigenvalue_inside_an_operand_is_not_dropped():
    """U (L2 v - c v): the c-part must reach M (it used to vanish silently when another equation
    carried c, so M was nonzero and nothing was refused)."""
    inner = _pair(lambda y, v, b, U, S, c: U * (L2(v, y) - c * v) + b,
                  lambda y, v, b, U, S, c: c * b - S * v)
    expd = _pair(lambda y, v, b, U, S, c: U * L2(v, y) - c * U * v + b,
                 lambda y, v, b, U, S, c: c * b - S * v)
    assert np.abs(inner[1][:16, :16]).max() > 0.5            # the U v block of M exists
    for A, B in zip(inner, expd):
        assert np.allclose(A, B, rtol=0, atol=1e-14)


@pytest.mark.parametrize("bad", ["square", "inverse", "exp", "conj"])
def test_nonlinear_eigenvalue_refused(bad):
    ch = channel(8, base=None, interval=(-1.0, 1.0))
    v = ch.field("v")
    c = Parameter("c", dtype=complex)
    x = {"square": (1 - c) * (2 - c) * v, "inverse": v / c, "exp": exp(c) * v,
         "conj": conj(c) * v}[bad]
    with pytest.raises(DiscretizationError, match="linear"):
        eigen_split(x, c)


def test_split_parts():
    ch = channel(8, base=None, interval=(-1.0, 1.0))
    y, v, U = ch.y, ch.field("v"), ch.field("U")
    c = Parameter("c", dtype=complex)
    p = eigen_split(diff((U - c) * v, y), c)
    assert p[0] == diff(U * v, y) and p[1] == diff(-1 * v, y)
