"""Spherical toolkits: spin-weighted harmonics, angular derivatives, regularity matrix, ball
operators, and the transforms / pseudospectral evaluator against closed forms."""
import numpy as np
import pytest
from rounding_bounds import product_identity_bound

import conatus.expert as C
from conatus.discretize.compiler import DiscretizationError
from conatus.discretize.sphere_eval import SphereEvaluator, SphereSpace
from conatus.discretize.sphere_ops import BallOps, SphereHarmonics, regularity_Q


def test_harmonics_orthonormal_and_derivatives_diagonal():
    """<Phi_l, Phi_l'> = delta on Gauss-Legendre nodes; G_+- (closed form, one Jacobi identity per
    sign pattern) equals the Gauss projection of (d_theta -+ (m + s cos)/sin)/sqrt(2) applied by
    finite differences, and |G_+| = sqrt((l-s)(l+s+1)/2)."""
    L = 10
    H = SphereHarmonics(L)
    x, w = np.polynomial.legendre.leggauss(36)
    th = np.arccos(x)
    ell = np.arange(L + 1)
    for m in range(-4, 5):
        for s in (-2, -1, 0, 1):
            V = H.values(m, s, x)
            assert np.abs(V.T @ (w[:, None] * V) - np.diag(H.valid(m, s).astype(float))).max() < 1e-12
            for sig in (1, -1):
                h = 1e-6
                dV = (H.values(m, s, np.cos(th + h)) - H.values(m, s, np.cos(th - h))) / (2 * h)
                op = (dV - sig * ((m + s * x) / np.sin(th))[:, None] * V) / np.sqrt(2)
                proj = H.values(m, s + sig, x).T @ (w[:, None] * op)
                d = H.G(sig, m, s)
                assert np.abs(proj - np.diag(d)).max() < 1e-8
                mag = np.sqrt(np.maximum((ell - sig * s) * (ell + sig * s + 1), 0) / 2)
                mag = mag * (H.valid(m, s) & H.valid(m, s + sig))
                assert np.abs(np.abs(d) - mag).max() < 1e-12


@pytest.mark.parametrize("m", [0, 2, -3])
def test_spin_laplacian_negative_definite(m):
    """G_-G_+ + G_+G_- on spin s = -(l(l+1) - s^2): the scalar and connection (Bochner) Laplacians."""
    H = SphereHarmonics(9)
    ell = np.arange(10)
    for s in (-1, 0, 1):
        lap = H.G(-1, m, s + 1) * H.G(1, m, s) + H.G(1, m, s - 1) * H.G(-1, m, s)
        assert np.abs(lap + (ell * (ell + 1) - s * s) * H.valid(m, s)).max() < 1e-12


def test_regularity_Q_unitary():
    H = SphereHarmonics(8)
    for m in (0, 1, 3, -2):
        for ell in range(abs(m), 8):
            Q = regularity_Q(H, m, ell)
            ref = np.diag([1.0 if ell else 0.0, 1.0, 1.0 if ell else 0.0])
            assert np.abs(Q.conj().T @ Q - ref).max() < 1e-13


def test_ball_radial_operators_single_diagonal_and_exact():
    """D_+(k) and D_-(k) on rho^k Q_n^(0, k+1/2): compare with d/dr -+ .../r of the synthesized
    function at interior points (no division at r = 0)."""
    B = BallOps(10, 1.3)
    r = np.array([0.1, 0.45, 0.9, 1.2])
    rng = np.random.default_rng(0)
    for k in (0, 1, 3):
        n = 8
        c = rng.standard_normal(n)
        val = np.array([B.point_row(k, 0, ri, 0, n) for ri in r]) @ c
        der = np.array([B.point_row(k, 0, ri, 1, n) for ri in r]) @ c
        up = B.up(k, 0, n) @ c
        assert B.up(k, 0, n).nnz <= n                       # single diagonal
        assert np.abs(np.array([B.point_row(k + 1, 1, ri, 0, n) for ri in r]) @ up
                      - (der - k * val / r)).max() < 1e-11
        if k >= 1:
            dn = B.down(k, 0, n) @ c
            assert np.abs(np.array([B.point_row(k - 1, 1, ri, 0, n) for ri in r]) @ dn
                          - (der + (k + 1) * val / r)).max() < 1e-11


# ----------------------------------------------------------------------------------------------
# transforms and evaluator
# ----------------------------------------------------------------------------------------------

def _cart_to_sph(vx, vy, vz, P, T):
    er = (np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T))
    et = (np.cos(T) * np.cos(P), np.cos(T) * np.sin(P), -np.sin(T))
    ep = (-np.sin(P), np.cos(P), 0 * P)
    d = lambda e: vx * e[0] + vy * e[1] + vz * e[2]          # noqa: E731
    return np.array([d(ep), d(et), d(er)])


def _space(kind):
    if kind == "s2":
        s2 = C.S2Coordinates("phi", "theta")
        D = C.Domain(s2, [C.RealFourier(s2["phi"], 16, (0, 2 * np.pi), dealias=3 / 2),
                          C.SphereBasis(s2["theta"], 8, radius=2.0, dealias=3 / 2)])
        return D, SphereSpace(D)
    sc = C.SphericalCoordinates("phi", "theta", "r")
    br = (C.ShellBasis(sc["r"], 12, (0.5, 1.5), dealias=3 / 2) if kind == "shell"
          else C.BallBasis(sc["r"], 12, 1.0, dealias=3 / 2))
    D = C.Domain(sc, [C.RealFourier(sc["phi"], 16, (0, 2 * np.pi), dealias=3 / 2),
                      C.SphereBasis(sc["theta"], 8, dealias=3 / 2), br])
    return D, SphereSpace(D)


def test_s2_evaluator_closed_forms():
    D, S = _space("s2")
    ev = SphereEvaluator(S)
    f = C.Field("f", D)
    Rad = 2.0
    P, T = np.meshgrid(*S.grids("data"), indexing="ij")
    X, Y, Z = Rad * np.sin(T) * np.cos(P), Rad * np.sin(T) * np.sin(P), Rad * np.cos(T)
    cf = S.forward(X + Y * Z + Z ** 2, D, 0, "data")
    assert np.abs(S.backward(cf, D, 0, "data") - (X + Y * Z + Z ** 2)).max() < 1e-13
    P, T = np.meshgrid(*S.grids(True), indexing="ij")
    X, Y, Z = Rad * np.sin(T) * np.cos(P), Rad * np.sin(T) * np.sin(P), Rad * np.cos(T)
    G = _cart_to_sph(1 + 0 * X, Z, Y + 2 * Z, P, T)[:2]           # tangential part of grad
    assert np.abs(S.backward(ev.evaluate(C.grad(f), {f: cf}, {}), D, 1, True) - G).max() < 1e-12
    lapF = -(2 / Rad ** 2) * X - (6 / Rad ** 2) * Y * Z - (6 / Rad ** 2) * (Z ** 2 - Rad ** 2 / 3)
    for e in (C.lap(f), C.div(C.grad(f)), -C.curl(C.curl(f))):
        assert np.abs(S.backward(ev.evaluate(e, {f: cf}, {}), D, 0, True) - lapF).max() < 1e-12
    assert np.abs(S.backward(ev.evaluate(C.curl(C.grad(f)), {f: cf}, {}), D, 0, True)).max() < 1e-13
    # connection (Bochner) Laplacian of a gradient: grad(lap f) + grad f / R^2
    d = ev.evaluate(C.lap(C.grad(f)), {f: cf}, {}) - ev.evaluate(C.grad(C.lap(f)), {f: cf}, {})
    assert np.abs(d - ev.evaluate(C.grad(f), {f: cf}, {}) / Rad ** 2).max() < 1e-12


@pytest.mark.parametrize("kind", ["shell", "ball"])
def test_3d_evaluator_closed_forms(kind):
    """grad, div, curl, lap, u.grad f, u.grad u of polynomial fields (Cartesian closed forms)."""
    D, S = _space(kind)
    ev = SphereEvaluator(S)
    f, u = C.Field("f", D), C.VectorField("u", D)
    P, T, R = np.meshgrid(*S.grids("data"), indexing="ij")
    X, Y, Z = R * np.sin(T) * np.cos(P), R * np.sin(T) * np.sin(P), R * np.cos(T)
    co = {f: S.forward(X + Y * Z + Z ** 2 + X * X * Y, D, 0, "data"),
          u: S.forward(_cart_to_sph(Y * Z, Z ** 2 + X, X * Y + 1.0, P, T), D, 1, "data")}
    P, T, R = np.meshgrid(*S.grids(True), indexing="ij")
    X, Y, Z = R * np.sin(T) * np.cos(P), R * np.sin(T) * np.sin(P), R * np.cos(T)
    ux, uy, uz = Y * Z, Z ** 2 + X, X * Y + 1
    checks = [                                                # (expression, reference, rank)
        (C.grad(f), _cart_to_sph(1 + 2 * X * Y, Z + X * X, Y + 2 * Z, P, T), 1),
        (C.div(u), 0 * X, 0),
        (C.curl(u), _cart_to_sph(X - 2 * Z, 0 * X, 1 - Z, P, T), 1),
        (C.lap(f), 2 + 2 * Y, 0),
        (C.dot(u, C.grad(f)), ux * (1 + 2 * X * Y) + uy * (Z + X * X) + uz * (Y + 2 * Z), 0),
        (C.dot(u, C.grad(u)), _cart_to_sph(uy * Z + uz * Y, ux + uz * 2 * Z, ux * Y + uy * X, P, T), 1),
    ]
    # round-off amplification: ~eps N^2 per radial derivative; ball measured at N = 12 after the
    # refined Gauss-Jacobi nodes: grad 3e-13, lap 2.9e-11 (was 7.7e-12, 7.9e-10)
    tol = 1e-12 if kind == "shell" else 3e-12
    for e, ref, rank in checks:
        got = S.backward(ev.evaluate(e, co, {}), D, rank, True)
        t = tol * (100 if isinstance(e, C.Laplacian) else 1)
        assert np.abs(got - ref).max() < t * max(1, np.abs(ref).max()), str(e)


def test_refusals():
    s2 = C.S2Coordinates("phi", "theta")
    with pytest.raises(ValueError, match="SphereBasis"):
        C.Domain(s2, [C.Chebyshev(s2["theta"], 8, (0, 3), dealias=1)])
    with pytest.raises(ValueError, match="colatitude"):
        C.Domain(s2, [C.RealFourier(s2["phi"], 8, (0, 2 * np.pi), dealias=1)])
    D = C.Domain(s2, [C.RealFourier(s2["phi"], 16, (0, 2 * np.pi), dealias=1),
                      C.SphereBasis(s2["theta"], 6, dealias=1)])
    with pytest.raises(DiscretizationError, match="Lmax"):       # m = 7 > Lmax = 5: no harmonic
        SphereSpace(D)
    D, S = _space("s2")
    u = C.VectorField("u", D)
    with pytest.raises(DiscretizationError, match="not regular"):
        SphereEvaluator(S).evaluate(C.comp(u, 0), {u: np.zeros(S.coeff_shape(D, 1), complex)}, {})


@pytest.mark.parametrize("N", [12, 32])
def test_ball_radial_transform_is_orthogonal_to_rounding(N):
    """F B = I on every regularity component. With scipy's roots_jacobi(0, 1/2)
    weights |F B - I| was 6e-14 at N = 32 and a cubic's round trip 9.5e-13 (Dedalus: 1.5e-14);
    the Newton-refined long-double nodes and Christoffel weights gave 2.2e-14 at N = 32 and the
    long-double normalisation (polar_radial.log_jacobi_norms) 2.8e-15; round trip 1.6e-14."""
    from conatus.discretize.sphere_ops import BallOps
    op = BallOps(N, 1.0)
    for M in (N, (3 * N) // 2):
        for k in (0, 1, 3, 7):
            size = op.size(k)
            F, B = op.forward(k, M, size), op.backward(k, M, size)
            # worst-case rounding bound 2 M eps |F| |B| (tests/rounding_bounds.py rule): the
            # former 2e-15 / 5e-15 were met at 1.24e-15 (margin 1.6, run
            #); the scipy weights (6e-14 at N = 32) exceed it
            err = np.abs(F @ B - np.eye(size))
            assert (err <= product_identity_bound(F, B)).all(), (N, M, k, err.max())
