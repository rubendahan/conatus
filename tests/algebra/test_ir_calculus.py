"""Level A (§69): linearization, differential orders, transpose (IBP) and vector identities."""
import numpy as np
import pytest

from conatus.ir import (CartesianCoordinates, Chebyshev, Domain, Field, Parameter, RealFourier,
                        VectorField, curl, diff, div, dot, grad, integ, lap)
from conatus.reference import ReferenceEvaluator, evaluate

TWO_PI = 2 * np.pi


def test_linearize_advection_1d():
    cs = CartesianCoordinates("x")
    x = cs["x"]
    D = Domain(cs, [RealFourier(x, 16, (0, TWO_PI), dealias=3 / 2)])
    u, U, v = Field("u", D), Field("U", D), Field("v", D)
    lin = (u * diff(u, x)).linearize(around={u: U}, direction={u: v})
    assert lin == U * diff(v, x) + v * diff(U, x)


def test_linearize_linear_term_is_itself():
    cs = CartesianCoordinates("x", "z")
    x, z = cs.coords
    D = Domain(cs, [RealFourier(x, 8, (0, TWO_PI), dealias=1), Chebyshev(z, 8, (-1, 1), dealias=1)])
    u, v, U = Field("u", D), Field("v", D), Field("U", D)
    nu = Parameter("nu", 0.1)
    L = nu * lap(u) - 3 * diff(u, x)
    assert L.linearize({u: U}, {u: v}) == L.substitute({u: v})
    # around is irrelevant for a linear term
    assert L.linearize({}, {u: v}) == nu * lap(v) - 3 * diff(v, x)


def test_linearize_vector_advection():
    cs = CartesianCoordinates("x", "z")
    x, z = cs.coords
    D = Domain(cs, [RealFourier(x, 8, (0, TWO_PI), dealias=1), Chebyshev(z, 8, (-1, 1), dealias=1)])
    u, U, v = VectorField("u", D), VectorField("U", D), VectorField("v", D)
    lin = dot(u, grad(u)).linearize({u: U}, {u: v})
    assert lin == dot(v, grad(U)) + dot(U, grad(v))


def test_laplacian_order_2d():
    cs = CartesianCoordinates("x", "z")
    x, z = cs.coords
    D = Domain(cs, [RealFourier(x, 8, (0, TWO_PI), dealias=1), Chebyshev(z, 8, (-1, 1), dealias=1)])
    u = Field("u", D)
    assert lap(u).order_in(x) == 2 and lap(u).order_in(z) == 2
    assert diff(diff(u, z), z).order_in(z) == 2 and diff(u, z).order_in(x) == 0
    assert lap(lap(u)).order_in(z) == 4
    U = Field("U", D)  # coefficient: its derivatives do not count for the unknown u
    assert (diff(U, z, ) * u).order_in(z, fields=[u]) == 0
    assert (diff(U, z) * u).order_in(z) == 1


def test_degrees_and_linearity():
    cs = CartesianCoordinates("x")
    x = cs["x"]
    D = Domain(cs, [Chebyshev(x, 8, (-1, 1), dealias=1)])
    u, U = Field("u", D), Field("U", D)
    assert (U * diff(u, x)).is_linear_in([u])
    assert not (u * diff(u, x)).is_linear_in([u])
    assert (u * diff(u, x)).degrees([u]) == frozenset({2})
    assert not (u + 1).is_linear_in([u])  # affine is not linear
    assert not (u ** -1).is_linear_in([u])


# ---------------------------------------------------------------- reference evaluation


def _fourier_cheb_2d(Nx=16, Nz=24):
    cs = CartesianCoordinates("x", "z")
    x, z = cs.coords
    D = Domain(cs, [RealFourier(x, Nx, (0, TWO_PI), dealias=1), Chebyshev(z, Nz, (-1, 1), dealias=1)])
    ev = ReferenceEvaluator(D)
    return cs, x, z, D, ev, ev.grid(x), ev.grid(z)


def test_cheb_derivative_of_monomials():
    cs = CartesianCoordinates("x")
    x = cs["x"]
    D = Domain(cs, [Chebyshev(x, 16, (-1, 1), dealias=1)])
    u = Field("u", D)
    X = ReferenceEvaluator(D).grid(x)
    for n in range(1, 12):
        err = np.max(abs(evaluate(diff(u, x), {u: X ** n}) - n * X ** (n - 1)))
        assert err < 1e-11 * n ** 2, (n, err)


def test_cheb_derivative_mapped_interval():
    cs = CartesianCoordinates("x")
    x = cs["x"]
    D = Domain(cs, [Chebyshev(x, 20, (0, 3), dealias=1)])
    u = Field("u", D)
    X = ReferenceEvaluator(D).grid(x)
    assert np.max(abs(evaluate(diff(u, x), {u: X ** 4}) - 4 * X ** 3)) < 1e-10


def test_div_curl_and_curl_grad_vanish_2d():
    cs, x, z, D, ev, X, Z = _fourier_cheb_2d()
    psi, phi = Field("psi", D), Field("phi", D)
    data_psi = np.sin(2 * X) * (1 - Z ** 2) ** 2 + np.cos(X) * Z ** 5
    data_phi = np.cos(3 * X) * Z ** 3 + Z
    r1 = evaluate(div(curl(psi)), {psi: data_psi})
    r2 = evaluate(curl(grad(phi)), {phi: data_phi})
    assert np.max(abs(r1)) < 1e-10
    assert np.max(abs(r2)) < 1e-10


def test_lap_equals_div_grad_2d():
    cs, x, z, D, ev, X, Z = _fourier_cheb_2d()
    u = Field("u", D)
    data = np.sin(X) * np.exp(Z) + Z ** 4
    assert np.max(abs(evaluate(lap(u) - div(grad(u)), {u: data}))) < 1e-9


# ---------------------------------------------------------------- transpose / IBP identity


def _check_ibp(op, u, w, bindings, tol):
    tr = op.transpose(u, w)
    lhs = integ(op * w) if op.rank == 0 else integ(dot(op, w))
    rhs = integ(u * tr.adjoint) if u.rank == 0 else integ(dot(u, tr.adjoint))
    L = evaluate(lhs, bindings)
    R = evaluate(rhs, bindings) + np.asarray(evaluate(tr.boundary, bindings)
                                             if not tr.boundary.is_zero() else 0.0)
    assert np.max(abs(L - R)) < tol, (float(np.max(abs(L - R))), tr.explain())
    return tr


def test_transpose_derivative_1d_cheb():
    cs = CartesianCoordinates("x")
    x = cs["x"]
    D = Domain(cs, [Chebyshev(x, 32, (-1, 2), dealias=1)])
    u, w = Field("u", D), Field("w", D)
    X = ReferenceEvaluator(D).grid(x)
    tr = _check_ibp(diff(u, x), u, w, {u: np.exp(X), w: X ** 3 + 1}, 1e-11)
    assert tr.adjoint == -diff(w, x)
    assert not tr.boundary.is_zero()


def test_transpose_is_boundary_free_when_periodic():
    cs = CartesianCoordinates("x")
    x = cs["x"]
    D = Domain(cs, [RealFourier(x, 16, (0, TWO_PI), dealias=1)])
    u, w = Field("u", D), Field("w", D)
    tr = diff(u, x).transpose(u, w)
    assert tr.boundary.is_zero()
    assert "periodicity" in tr.proof_obligation()


def test_transpose_advection_diffusion_2d():
    cs, x, z, D, ev, X, Z = _fourier_cheb_2d(16, 32)
    u, w, U = Field("u", D), Field("w", D), Field("U", D)
    nu = Parameter("nu", 0.3)
    op = nu * lap(u) - U * diff(u, x) + 2 * diff(u, z)
    b = {u: np.sin(X) * np.exp(Z), w: np.cos(2 * X) * (1 + Z ** 3), U: 1 - Z ** 2}
    tr = _check_ibp(op, u, w, b, 1e-9)
    assert tr.adjoint.is_linear_in([w])


def test_transpose_grad_div_vector():
    cs, x, z, D, ev, X, Z = _fourier_cheb_2d(16, 24)
    p, W = Field("p", D), VectorField("W", D)
    Wd = np.stack([np.sin(X) * Z ** 2, np.cos(X) * (1 + Z)])
    _check_ibp(grad(p), p, W, {p: np.cos(X) * Z ** 3, W: Wd}, 1e-9)
    v, q = VectorField("v", D), Field("q", D)
    _check_ibp(div(v), v, q, {v: Wd, q: np.sin(2 * X) + Z}, 1e-9)


def test_transpose_rejects_nonlinear():
    cs = CartesianCoordinates("x")
    x = cs["x"]
    D = Domain(cs, [Chebyshev(x, 8, (-1, 1), dealias=1)])
    u, w = Field("u", D), Field("w", D)
    with pytest.raises(ValueError, match="not linear"):
        (u * diff(u, x)).transpose(u, w)


def test_linearize_taylor_numeric():
    """|F(U+eps v) - F(U) - eps dF(U)v| = O(eps^2) (Taylor test, §25)."""
    cs, x, z, D, ev, X, Z = _fourier_cheb_2d(16, 20)
    u, U, v = Field("u", D), Field("U", D), Field("v", D)
    F = u * diff(u, x) + u ** 3 - lap(u) + power_inv(u)
    dF = F.linearize({u: U}, {u: v})
    Ud = 2 + np.sin(X) * (1 - Z ** 2)
    vd = np.cos(2 * X) * Z ** 3
    base = evaluate(F, {u: Ud})
    lin = evaluate(dF, {U: Ud, v: vd})
    errs = []
    for eps in (1e-2, 5e-3):
        errs.append(np.max(abs(evaluate(F, {u: Ud + eps * vd}) - base - eps * lin)))
    assert 3.5 < errs[0] / errs[1] < 4.5, errs  # second order


def power_inv(u):
    return u ** -1
