"""Eulerian diagnostics against closed-form fields ( §Tests).

Taylor–Green (2D and 3D), Stuart's Kelvin–Helmholtz cat's eyes, ABC (Beltrami) flow, exact
poloidal–toroidal round trip, Parseval for the spectra, closed-form 3x3 eigenvalues against
LAPACK. Every assertion states the measured error bound; nothing is compared to a raw value."""
from __future__ import annotations

import numpy as np
import pytest

from conatus.diagnostics import Chebyshev, Grid, Periodic
from conatus.diagnostics import euler as E
from conatus.errors import UnsupportedError

TWO_PI = 2 * np.pi


def tg2d(n=32):
    g = Grid([Periodic(TWO_PI, n), Periodic(TWO_PI, n)])
    X, Y = g.mesh()
    u = [np.sin(X) * np.cos(Y), -np.cos(X) * np.sin(Y)]
    return g, X, Y, u


def test_grid_derivatives_and_quadrature():
    """Fourier and Chebyshev derivatives exact on smooth data; Fejér rule exact for polynomials."""
    g = Grid([Periodic(TWO_PI, 32), Chebyshev(-1.0, 2.0, 24)])
    X, Y = g.mesh()
    f = np.sin(3 * X) * np.exp(Y)
    assert np.abs(g.d(f, 0) - 3 * np.cos(3 * X) * np.exp(Y)).max() < 1e-12
    assert np.abs(g.d(f, 1) - f).max() < 1e-11
    assert np.abs(g.d(f, 1, 2) - f).max() < 1e-9
    assert abs(float(g.integrate(np.cos(X)**2 * Y**2)) - np.pi * 3.0) < 1e-12
    assert abs(float(g.cell_volumes().sum()) - g.volume()) < 1e-12


def test_taylor_green_2d():
    """omega = 2 sin x sin y, psi = sin x sin y, p = (cos 2x + cos 2y)/4, Q closed form,
    <eps> = nu <omega^2> = nu."""
    g, X, Y, u = tg2d()
    w = E.vorticity(g, u)
    assert np.abs(w - 2 * np.sin(X) * np.sin(Y)).max() < 1e-13
    assert np.abs(E.streamfunction(g, u) - np.sin(X) * np.sin(Y)).max() < 1e-13
    assert np.abs(E.streamfunction_from_vorticity(g, w) - np.sin(X) * np.sin(Y)).max() < 1e-13
    assert np.abs(E.pressure(g, u) - 0.25 * (np.cos(2 * X) + np.cos(2 * Y))).max() < 1e-13
    Q = np.sin(X)**2 * np.sin(Y)**2 - np.cos(X)**2 * np.cos(Y)**2
    assert np.abs(E.q_criterion(g, u) - Q).max() < 1e-13
    nu = 0.01
    assert abs(float(g.mean(E.dissipation(g, u, nu))) - nu) < 1e-15
    assert np.abs(E.divergence(g, u)).max() < 1e-13


def test_streamfunction_with_mean_flow():
    """Periodic box with a uniform mean (U0, V0): psi = psi' + U0 y - V0 x reproduces u exactly."""
    g, X, Y, u = tg2d()
    u = [u[0] + 0.3, u[1] - 0.2]
    psi = E.streamfunction(g, u)
    assert np.abs(psi - (np.sin(X) * np.sin(Y) + 0.3 * Y + 0.2 * X)).max() < 1e-13


def test_stuart_vortices():
    """Stuart (1967) cat's eyes: psi = ln(C cosh y + A cos x), C^2 - A^2 = 1, lap psi = exp(-2 psi)
    so omega = -exp(-2 psi). Periodic x Chebyshev y in [-4, 4]."""
    C = 1.25
    A = np.sqrt(C**2 - 1)
    errs = []
    for ny in (64, 96, 128):                  # spectral convergence in y (poles off the real axis)
        g = Grid([Periodic(TWO_PI, 48), Chebyshev(-4.0, 4.0, ny)])
        X, Y = g.mesh()
        D = C * np.cosh(Y) + A * np.cos(X)
        psi = np.log(D)
        u = [C * np.sinh(Y) / D, A * np.sin(X) / D]
        err_w = np.abs(E.vorticity(g, u) + np.exp(-2 * psi)).max()
        err_psi = np.ptp(E.streamfunction(g, u) - psi)      # exact up to the gauge constant
        errs.append((err_w, err_psi))
    assert errs[0][0] > errs[1][0] > errs[2][0] and errs[2][0] < 1e-9, errs
    assert errs[2][1] < 1e-11, errs


def test_taylor_green_3d_q_lambda2_pressure():
    """3D TG: vorticity closed form; Q and lambda_2 against the analytic gradient (LAPACK
    eigvalsh oracle for lambda_2); pressure p = (cos 2x + cos 2y)(cos 2z + 2)/16."""
    n = 16
    g = Grid([Periodic(TWO_PI, n)] * 3)
    X, Y, Z = g.mesh()
    sx, cx, sy, cy, sz, cz = np.sin(X), np.cos(X), np.sin(Y), np.cos(Y), np.sin(Z), np.cos(Z)
    u = [sx * cy * cz, -cx * sy * cz, 0 * X]
    w = E.vorticity(g, u)
    wex = [-cx * sy * sz, -sx * cy * sz, 2 * sx * sy * cz]
    assert max(np.abs(a - b).max() for a, b in zip(w, wex)) < 1e-13
    # analytic gradient A_ij = d u_i / d x_j
    Aex = np.array([[cx * cy * cz, -sx * sy * cz, -sx * cy * sz],
                    [sx * sy * cz, -cx * cy * cz, cx * sy * sz],
                    [0 * X, 0 * X, 0 * X]])
    S = 0.5 * (Aex + Aex.transpose(1, 0, 2, 3, 4))
    W = 0.5 * (Aex - Aex.transpose(1, 0, 2, 3, 4))
    Qex = 0.5 * ((W**2).sum(axis=(0, 1)) - (S**2).sum(axis=(0, 1)))
    assert np.abs(E.q_criterion(g, u) - Qex).max() < 1e-13
    M = np.einsum("ik...,kj...->ij...", S, S) + np.einsum("ik...,kj...->ij...", W, W)
    lam = np.linalg.eigvalsh(np.moveaxis(M, (0, 1), (-2, -1)))[..., 1]
    assert np.abs(E.lambda2(g, u) - lam).max() < 1e-12
    # closed form: ~sqrt(eps) near repeated eigenvalues (declared)
    assert np.abs(E.lambda2(g, u, method="closed_form") - lam).max() < 1e-7
    pex = (np.cos(2 * X) + np.cos(2 * Y)) * (np.cos(2 * Z) + 2) / 16
    assert np.abs(E.pressure(g, u) - pex).max() < 1e-13


def test_abc_flow_beltrami():
    """ABC flow is Beltrami (curl u = u): helicity density = |u|^2, vector potential = u."""
    g = Grid([Periodic(TWO_PI, 16)] * 3)
    X, Y, Z = g.mesh()
    a, b, c = 1.0, np.sqrt(2 / 3), np.sqrt(1 / 3)
    u = [a * np.sin(Z) + c * np.cos(Y), b * np.sin(X) + a * np.cos(Z), c * np.sin(Y) + b * np.cos(X)]
    h = E.helicity(g, u)
    assert np.abs(h - sum(x * x for x in u)).max() < 1e-13
    A, mean = E.vector_potential(g, u)
    assert max(np.abs(p - q).max() for p, q in zip(A, u)) < 1e-13
    assert max(abs(m) for m in mean) < 1e-15


def test_poloidal_toroidal_round_trip():
    """Build u from (phi, psi, U, W) with the channel convention (spectral derivatives), then
    recover them exactly."""
    g = Grid([Periodic(TWO_PI, 16), Chebyshev(-1.0, 1.0, 20), Periodic(np.pi, 12)])
    X, Y, Z = g.mesh()
    phi = (1 - Y**2)**2 * (np.cos(X) * np.sin(2 * Z) + 0.3 * np.sin(2 * X))
    psi = (1 - Y**2) * np.cos(X + 2 * Z) * Y
    U = 1 - Y**2
    Wm = 0.1 * Y
    d = g.d
    ux = d(d(phi, 1), 0) + d(psi, 2) + U
    uy = -(d(phi, 0, 2) + d(phi, 2, 2))
    uz = d(d(phi, 1), 2) - d(psi, 0) + Wm
    out = E.poloidal_toroidal(g, [ux, uy, uz], wall_axis=1)
    assert np.abs(out["phi"] - phi).max() < 1e-12
    assert np.abs(out["psi"] - psi).max() < 1e-12
    assert np.abs(out["mean_a"] - U[0, :, 0]).max() < 1e-13
    assert np.abs(out["mean_c"] - Wm[0, :, 0]).max() < 1e-13
    assert np.abs(out["mean_w"]).max() < 1e-13


def test_spectra_parseval():
    """sum E(k) = <|u|^2>/2 on a box; sum E_1d = <f^2> along a periodic axis with a wall axis."""
    rng = np.random.default_rng(1)
    g = Grid([Periodic(TWO_PI, 16), Periodic(2 * TWO_PI, 20), Periodic(TWO_PI, 8)])
    u = [rng.standard_normal(g.shape) for _ in range(3)]
    k, Ek = E.energy_spectrum(g, u)
    assert abs(Ek.sum() - float(g.mean(E.kinetic_energy(g, u)))) < 1e-12
    g2 = Grid([Periodic(TWO_PI, 16), Chebyshev(-1.0, 1.0, 12)])
    X, Y = g2.mesh()
    f = np.cos(3 * X) * (1 + Y) + 0.5 * np.sin(X)
    k1, E1 = E.spectrum_1d(g2, f, 0)
    assert abs(E1.sum() - float(g2.mean(f * f))) < 1e-13
    assert abs(E1[3] - float(g2.mean(0.5 * (1 + Y)**2))) < 1e-13


def test_sym3_eigvals_against_lapack():
    """Closed-form eigenvalues of random symmetric 3x3 matrices (incl. repeated ones)."""
    rng = np.random.default_rng(2)
    B = rng.standard_normal((500, 3, 3))
    B = B + B.transpose(0, 2, 1)
    B[:10] = np.diag([1.0, 1.0, -2.0])          # repeated eigenvalue
    B[10:20] = 0.0                              # zero matrix
    M = [[B[:, i, j] for j in range(3)] for i in range(3)]
    got = np.stack(E.sym3_eigvals(M), axis=-1)
    ref = np.linalg.eigvalsh(B)
    assert np.abs(got - ref).max() < 1e-12 * max(1.0, np.abs(ref).max())


def test_refusals():
    """Pressure / vector potential need a fully periodic box (never a guessed wall condition)."""
    g = Grid([Periodic(TWO_PI, 8), Chebyshev(-1.0, 1.0, 8)])
    X, Y = g.mesh()
    with pytest.raises(UnsupportedError):
        E.pressure(g, [X * 0, Y * 0])
    with pytest.raises(UnsupportedError):
        E.streamfunction(Grid([Chebyshev(-1, 1, 8), Periodic(TWO_PI, 8)]), [X * 0, X * 0])


@pytest.mark.gpu
def test_gpu_matches_cpu():
    """Identical code on CuPy: Q, lambda_2, pressure, PT potentials agree with NumPy."""
    cp = pytest.importorskip("cupy")
    g = Grid([Periodic(TWO_PI, 16), Chebyshev(-1.0, 1.0, 20), Periodic(np.pi, 12)])
    rng = np.random.default_rng(3)
    X, Y, Z = g.mesh()
    u = [np.cos(X + 2 * Z) * (1 - Y**2) + 0.1 * rng.standard_normal(g.shape) for _ in range(3)]
    gg = g.on(cp)
    ug = [cp.asarray(c) for c in u]
    for f in (E.q_criterion, E.lambda2, E.helicity):
        assert np.abs(cp.asnumpy(f(gg, ug)) - f(g, u)).max() < 1e-10
    a = E.poloidal_toroidal(g, u)
    b = E.poloidal_toroidal(gg, ug)
    assert np.abs(cp.asnumpy(b["phi"]) - a["phi"]).max() < 1e-10
    gp = Grid([Periodic(TWO_PI, 16)] * 3)
    up = [rng.standard_normal(gp.shape) for _ in range(3)]
    assert np.abs(cp.asnumpy(E.pressure(gp.on(cp), [cp.asarray(c) for c in up]))
                  - E.pressure(gp, up)).max() < 1e-10
