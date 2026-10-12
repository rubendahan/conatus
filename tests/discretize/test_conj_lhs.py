"""conj() in a linear LHS: discretized as the declared doubled system (X, conj X) (gallery gap: "cannot discretize node Conj ('conj(dB)') in the LHS"). Checked against the same
equations written by hand in real variables (the form the CGLE example used before)."""
import numpy as np
import pytest

from conatus.expert import (EVP, IVP, LBVP, CartesianCoordinates, Chebyshev, ComplexFourier, Domain,
                            Field, Parameter, ProblemError, conj, diff, discretize, dt, eigenvalues,
                            interp, linearize, sqrt)


def _cgle_real(k, c1, c3, q):
    """Linearized CGLE in B = X + i Y (the pre-2026-09-28 gallery form)."""
    cs = CartesianCoordinates("x")
    x = cs["x"]
    D = Domain(cs, [ComplexFourier(x, 4, (0, 2 * np.pi / k), dealias=1)])
    X, Y = Field("X", D), Field("Y", D)
    R2, w = 1 - q * q, c3 * (1 - q * q) + c1 * q * q
    dxx = lambda f: diff(diff(f, x), x)                                       # noqa: E731
    dX = dxx(X) - c1 * dxx(Y) - 2 * q * (diff(Y, x) + c1 * diff(X, x)) - q * q * (X - c1 * Y)
    dY = dxx(Y) + c1 * dxx(X) + 2 * q * (diff(X, x) - c1 * diff(Y, x)) - q * q * (Y + c1 * X)
    m = X * X + Y * Y
    ivp = IVP([X, Y])
    ivp.add_equation((dt(X) - X + w * Y - dX, -(m * X - c3 * m * Y)))
    ivp.add_equation((dt(Y) - Y - w * X - dY, -(m * Y + c3 * m * X)))
    return linearize(ivp, around={X: np.sqrt(R2), Y: 0}, eigenvalue=Parameter("s", dtype=complex))


@pytest.fixture(scope="module")
def example():
    """Import examples/<relpath>.py unchanged (as tests/gallery/conftest.py does)."""
    import importlib.util
    from pathlib import Path

    def load(rel):
        path = Path(__file__).resolve().parents[2] / "examples" / f"{rel}.py"
        spec = importlib.util.spec_from_file_location(f"gallery_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return load


def _spectrum(evp):
    plan = discretize(evp)
    return np.sort_complex(np.concatenate([eigenvalues(plan, g).values for g in plan.groups])), plan


@pytest.mark.parametrize("args", [(0.05, 2.0, -0.6, 0.0), (0.1, 0.5, 0.5, 0.3), (0.3, -1.0, 0.7, 0.2)])
def test_cgle_complex_form_equals_real_form(example, args):
    gl = example("patterns/ginzburg_landau")
    a, plan = _spectrum(gl.cgle(*args))
    b, _ = _spectrum(_cgle_real(*args))
    gap = max(np.abs(b - z).min() for z in a) + max(np.abs(a - z).min() for z in b)
    assert a.size == b.size and gap < 1e-12, (a, b)        # same multiset of eigenvalues
    assert "doubled" in plan.explain() and "d/dt" in plan.problem.explain()


def test_lbvp_with_conj_solution_and_partner():
    """u'' + i conj(u) = f on [-1, 1], u(+-1) = 0: the partner u_conj equals conj(u)."""
    cs = CartesianCoordinates("y")
    y = cs["y"]
    D = Domain(cs, [Chebyshev(y, 24, (-1, 1), dealias=1)])
    u, f = Field("u", D, dtype=complex), Field("f", D, dtype=complex)
    pb = LBVP([u])
    pb.add_equation((diff(diff(u, y), y) + 1j * conj(u), f))
    pb.add_bc(interp(u, y, -1), interp(u, y, 1))
    # manufactured: u = (1 - y^2)(1 + 2i y) -> f = u'' + i conj(u)
    ue = lambda Y: (1 - Y ** 2) * (1 + 2j * Y)                               # noqa: E731
    upp = lambda Y: -2 * (1 + 2j * Y) + 2 * (-2 * Y) * 2j                    # noqa: E731
    sol = discretize(pb).solve({f: lambda Y: upp(Y) + 1j * np.conj(ue(Y))})
    from conatus.expert import grid
    Y = grid(D.bases[0], dealias=False)
    assert np.abs(sol.grid("u") - ue(Y)).max() < 1e-13
    assert np.abs(sol.grid("u_conj") - np.conj(ue(Y))).max() < 1e-13


def test_conj_refusals():
    cs = CartesianCoordinates("x")
    x = cs["x"]
    D = Domain(cs, [ComplexFourier(x, 4, (0, 2 * np.pi), dealias=1)])
    u = Field("u", D, dtype=complex)
    ivp = IVP([u])
    ivp.add_equation((dt(u) + conj(u), 0))
    with pytest.raises(ProblemError, match="IVP"):
        discretize(ivp)
    s = Parameter("s", dtype=complex)
    ev = EVP([u], eigenvalue=s)
    ev.add_equation((s * u + conj(s * u), 0))
    with pytest.raises(ProblemError, match="eigenvalue"):
        discretize(ev)
    assert sqrt(4.0) is not None
