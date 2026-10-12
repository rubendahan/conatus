"""add_bc places standard taus explicitly (declared in explain) and refuses ambiguity."""
import numpy as np
import pytest

from conatus.expert import (EVP, CartesianCoordinates, Chebyshev, Domain, Field, Parameter, ProblemError,
                            channel, diff, interp)
from conatus.discretize import discretize
from conatus.solvers import eigenvalues


def test_string_modes_with_add_bc():
    ch = channel(32, base=None)
    u, lam = ch.field("u"), Parameter("lam", dtype=complex)
    pb = EVP([u], eigenvalue=lam)
    pb.add_equation((diff(diff(u, ch.y), ch.y) + lam * u, 0))
    pb.add_bc(interp(u, ch.y, -1), interp(u, ch.y, 1))
    assert "Tau terms (declared)" in pb.explain() and "N-1..N-2" in pb.explain()
    w = np.sort(eigenvalues(discretize(pb)).values.real)
    exact = (np.arange(1, 6) * np.pi / 2) ** 2
    assert np.allclose(w[:5], exact, rtol=1e-10)


def test_add_bc_refuses_ambiguous_target():
    cs = CartesianCoordinates("y")
    D = Domain(cs, [Chebyshev(cs["y"], 16, (-1, 1), dealias=1)])
    u, v, lam = Field("u", D), Field("v", D), Parameter("lam", dtype=complex)
    pb = EVP([u, v], eigenvalue=lam)
    pb.add_equation((diff(u, cs["y"]) + lam * v, 0))
    pb.add_equation((diff(v, cs["y"]) + lam * u, 0))
    with pytest.raises(ProblemError, match="into="):
        pb.add_bc(interp(u, cs["y"], -1))
    pb.add_bc(interp(u, cs["y"], -1), into=0)
    pb.add_bc(interp(v, cs["y"], 1), into=1)
    assert len(pb.variables) == 4


def test_channel_profile_is_explicit():
    with pytest.raises(ValueError, match="unknown base flow"):
        channel(16, base="blasius")
    assert "1 - y^2" in channel(16, base="poiseuille").explain()
