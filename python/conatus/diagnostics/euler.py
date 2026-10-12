"""Eulerian diagnostics (read-only, NumPy or CuPy): vorticity, stream function, vector
potential, poloidal–toroidal potentials, velocity gradient, Q, lambda_2, helicity, dissipation,
pressure, spectra. Spec and conventions:

Every function takes a declared :class:`~conatus.diagnostics.grid.Grid` and physical values
(velocity = sequence of ``grid.ndim`` arrays, component i along axis i) and returns new arrays;
nothing is modified in place. Derivatives are the grid's spectral derivatives (exact on the
interpolant). Anything not exactly defined on a grid is refused with the reason (e.g. pressure
with a wall: it needs the solver's pressure or its boundary condition, never a guess).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..errors import CoordinateError, OptionError, UnsupportedError
from .grid import Chebyshev, Grid, Periodic, array_module

__all__ = ["velocity_gradient", "divergence", "vorticity", "streamfunction",
           "streamfunction_from_vorticity", "vector_potential", "poloidal_toroidal",
           "strain_rotation", "q_criterion", "lambda2", "helicity", "enstrophy", "dissipation",
           "kinetic_energy", "pressure", "energy_spectrum", "spectrum_1d", "sym3_eigvals"]


def _check_vector(grid: Grid, u: Sequence[Any], what: str) -> None:
    if len(u) != grid.ndim:
        raise CoordinateError(f"{what}: {len(u)} velocity components on a {grid.ndim}D grid",
                              fix="pass one component per grid axis, in axis order")
    for c in u:
        if tuple(c.shape) != grid.shape:
            raise CoordinateError(f"{what}: component of shape {tuple(c.shape)} on a grid of "
                                  f"shape {grid.shape}")


def _all_periodic(grid: Grid, what: str) -> None:
    bad = [i for i, ax in enumerate(grid.axes) if not isinstance(ax, Periodic)]
    if bad:
        raise UnsupportedError(f"{what} needs every axis periodic (axes {bad} are bounded)",
                               why="the inversion is exact only on a fully periodic box; with a "
                                   "wall it needs a boundary condition that the velocity alone "
                                   "does not give",
                               fix="use the solver's own field, or poloidal_toroidal() for a "
                                   "channel")


def _kvec(grid: Grid) -> list:
    """Broadcastable angular wavenumber arrays for fftn (full) of a fully periodic grid."""
    out = []
    for i in range(grid.ndim):
        shp = [1] * grid.ndim
        shp[i] = grid.shape[i]
        out.append(grid.wavenumbers(i).reshape(shp))
    return out


# ---------------------------------------------------------------------- gradients
def velocity_gradient(grid: Grid, u: Sequence[Any]) -> list[list[Any]]:
    """A[i][j] = d u_i / d x_j (spectral derivatives)."""
    _check_vector(grid, u, "velocity_gradient")
    return [[grid.d(u[i], j) for j in range(grid.ndim)] for i in range(grid.ndim)]


def divergence(grid: Grid, u: Sequence[Any]):
    """div u (a check of incompressibility; reported, never corrected)."""
    _check_vector(grid, u, "divergence")
    return sum(grid.d(u[i], i) for i in range(grid.ndim))


def vorticity(grid: Grid, u: Sequence[Any]):
    """2D: the scalar omega = dx v - dy u. 3D: the vector curl u (list of 3 arrays)."""
    _check_vector(grid, u, "vorticity")
    d = grid.d
    if grid.ndim == 2:
        return d(u[1], 0) - d(u[0], 1)
    if grid.ndim == 3:
        return [d(u[2], 1) - d(u[1], 2), d(u[0], 2) - d(u[2], 0), d(u[1], 0) - d(u[0], 1)]
    raise CoordinateError(f"vorticity: 2D or 3D grid, got {grid.ndim}D")


# ---------------------------------------------------------------------- 2D stream function
def streamfunction_from_vorticity(grid: Grid, omega):
    """psi with lap psi = -omega on a fully periodic 2D box: exact spectral inversion
    psi_k = omega_k / |k|^2; the k = 0 mode of psi is set to zero (gauge, declared) and the
    mean of omega must vanish on a torus (its value is not used). Mean flow is not
    representable by a periodic psi: use :func:`streamfunction` for u = dy psi, v = -dx psi
    including it."""
    _all_periodic(grid, "streamfunction_from_vorticity")
    if grid.ndim != 2:
        raise CoordinateError("streamfunction_from_vorticity: 2D grid only")
    xp = grid.xp
    kx, ky = _kvec(grid)
    k2 = kx**2 + ky**2
    W = xp.fft.fftn(omega)
    k2 = xp.where(k2 == 0, 1.0, k2)
    P = W / k2
    P[0, 0] = 0.0                                 # gauge: zero mean (declared)
    return xp.fft.ifftn(P).real


def streamfunction(grid: Grid, u: Sequence[Any]):
    """psi with u = dy psi, v = -dx psi (2D, incompressible), exact on the grid:

    * periodic x periodic: psi = psi' + U0 y - V0 x, psi' from the vorticity (spectral Poisson
      inversion, zero mean), (U0, V0) the mean velocity (kept: the identity is exact, psi is then
      not periodic — declared);
    * periodic x (x axis 0) x Chebyshev (axis 1): mode by mode in x, psi_k = i v_k / k (k != 0),
      psi_0(y) = int_{y=a}^{y} u_0 dy' (Chebyshev antiderivative; gauge psi_0(a) = 0, declared).
      The x-mean of v must vanish (true for an incompressible flow between walls);
      :func:`divergence` and the returned field let the caller check the identity.
    Other grids: refused (UnsupportedError)."""
    _check_vector(grid, u, "streamfunction")
    if grid.ndim != 2:
        raise CoordinateError("streamfunction: 2D grid only (3D: vector_potential or poloidal_toroidal)")
    xp = grid.xp
    a0, a1 = grid.axes
    if isinstance(a0, Periodic) and isinstance(a1, Periodic):
        psi = streamfunction_from_vorticity(grid, vorticity(grid, u))
        U0, V0 = grid.mean(u[0]), grid.mean(u[1])
        X, Y = grid.mesh()
        return psi + U0 * (Y - a1.a) - V0 * (X - a0.a)
    if isinstance(a0, Periodic) and isinstance(a1, Chebyshev):
        k = grid.wavenumbers(0, real=True)[:, None]
        V = xp.fft.rfft(u[1], axis=0)
        U = xp.fft.rfft(u[0], axis=0)
        kk = xp.where(k == 0, 1.0, k)
        P = 1j * V / kk
        # k = 0: antiderivative in y of the x-mean of u, zero at y = a
        P[0] = _cheb_antiderivative(grid, U[0].real, 1) + 0j
        if a0.n % 2 == 0:
            P[-1] = 0.0                            # Nyquist mode: v_N = -dx psi_N is not representable
        return xp.fft.irfft(P, n=a0.n, axis=0)
    raise UnsupportedError(f"streamfunction on axes {type(a0).__name__} x {type(a1).__name__}",
                           fix="periodic x periodic or periodic (axis 0) x Chebyshev (axis 1)")


def _cheb_antiderivative(grid: Grid, f1d, axis: int):
    """int_a^y f dy' on the Chebyshev axis ``axis`` for a 1D array of node values (exact on the
    degree n - 1 interpolant; integration constant: zero at y = a)."""
    xp = grid.xp
    ax = grid.axes[axis]
    n = ax.n
    c = grid.cheb_coeffs(f1d, 0, ax)
    # int T_k = T_{k+1}/(2(k+1)) - T_{k-1}/(2(k-1)); coefficients of the antiderivative (n + 1)
    C = xp.zeros(n + 1, dtype=c.dtype)
    cc = xp.concatenate([c, xp.zeros(2, dtype=c.dtype)])
    k = xp.arange(1, n + 1)
    C[1:] = (cc[k - 1] * xp.where(k == 1, 2.0, 1.0) - cc[k + 1]) / (2 * k)
    # value at t = -1: sum C_k (-1)^k; shift so that it vanishes
    sgn = xp.asarray((-1.0) ** np.arange(n + 1))
    C[0] = -(C[1:] * sgn[1:]).sum()
    C = C * (ax.b - ax.a) / 2.0
    # evaluate the degree-n series at the n nodes: t_j = -cos(theta_j)
    t = xp.asarray(-np.cos(np.pi * (np.arange(n) + 0.5) / n))
    th = xp.arccos(t)
    return (C[None, :] * xp.cos(xp.arange(n + 1)[None, :] * th[:, None])).sum(axis=1)


# ---------------------------------------------------------------------- 3D potentials
def vector_potential(grid: Grid, u: Sequence[Any]):
    """A with curl A = u - <u>, div A = 0 (Coulomb gauge) on a fully periodic 3D box:
    A_k = i k x u_k / |k|^2, k = 0 mode zero. Returns (A, mean_velocity); the mean velocity is
    not the curl of a periodic field (declared, returned separately)."""
    _all_periodic(grid, "vector_potential")
    _check_vector(grid, u, "vector_potential")
    if grid.ndim != 3:
        raise CoordinateError("vector_potential: 3D grid only")
    xp = grid.xp
    k = _kvec(grid)
    k2 = k[0]**2 + k[1]**2 + k[2]**2
    k2 = xp.where(k2 == 0, 1.0, k2)
    U = [xp.fft.fftn(c) for c in u]
    A = [1j * (k[1] * U[2] - k[2] * U[1]) / k2,
         1j * (k[2] * U[0] - k[0] * U[2]) / k2,
         1j * (k[0] * U[1] - k[1] * U[0]) / k2]
    mean = [float(grid.mean(c)) for c in u]
    return [xp.fft.ifftn(a).real for a in A], mean


def poloidal_toroidal(grid: Grid, u: Sequence[Any], *, wall_axis: int = 1) -> dict:
    """Poloidal–toroidal potentials of a 3D velocity with two periodic horizontal axes (a, c),
    a < c, and any wall-normal axis w (Conatus channel convention, tests/channel3d):

        u_a = da dw phi + dc psi,   u_w = -lap_h phi,   u_c = dc dw phi - da psi,

    plus the horizontal means U_a(w), U_c(w) (the (0, 0) horizontal group, where phi = psi = 0).
    Mode by mode (k_h != 0): phi = u_w / k_h^2, psi = -(dc u_a - da u_c)^_k / k_h^2 — exact on
    the grid for an incompressible field. Returns dict(phi, psi, mean_a, mean_c, mean_w) with the
    means as arrays of the wall-normal coordinate (mean_w should be ~0 between walls)."""
    _check_vector(grid, u, "poloidal_toroidal")
    if grid.ndim != 3:
        raise CoordinateError("poloidal_toroidal: 3D grid only")
    if wall_axis not in (0, 1, 2):
        raise OptionError(f"wall_axis={wall_axis}: 0, 1 or 2")
    ha, hc = [i for i in range(3) if i != wall_axis]
    for i in (ha, hc):
        if not isinstance(grid.axes[i], Periodic):
            raise UnsupportedError(f"poloidal_toroidal: horizontal axis {i} must be periodic")
    xp = grid.xp
    ka = grid.wavenumbers(ha)
    kc = grid.wavenumbers(hc)
    shp_a, shp_c = [1, 1, 1], [1, 1, 1]
    shp_a[ha], shp_c[hc] = -1, -1
    ka, kc = ka.reshape(shp_a), kc.reshape(shp_c)
    kh2 = ka**2 + kc**2
    zero = kh2 == 0
    kh2s = xp.where(zero, 1.0, kh2)
    ax = (ha, hc)
    Ua = xp.fft.fftn(u[ha], axes=ax)
    Uc = xp.fft.fftn(u[hc], axes=ax)
    Uw = xp.fft.fftn(u[wall_axis], axes=ax)
    Phi = xp.where(zero, 0.0, Uw / kh2s)
    Psi = xp.where(zero, 0.0, -(1j * kc * Ua - 1j * ka * Uc) / kh2s)
    phi = xp.fft.ifftn(Phi, axes=ax).real
    psi = xp.fft.ifftn(Psi, axes=ax).real
    means = {name: u[i].mean(axis=ax) for name, i in (("mean_a", ha), ("mean_c", hc), ("mean_w", wall_axis))}
    return dict(phi=phi, psi=psi, **means)


# ---------------------------------------------------------------------- invariants
def strain_rotation(grid: Grid, u: Sequence[Any]):
    """(S, W): symmetric and antisymmetric parts of the velocity gradient, S_ij = (A_ij + A_ji)/2,
    W_ij = (A_ij - A_ji)/2, A_ij = d u_i / d x_j."""
    A = velocity_gradient(grid, u)
    n = grid.ndim
    S = [[0.5 * (A[i][j] + A[j][i]) for j in range(n)] for i in range(n)]
    W = [[0.5 * (A[i][j] - A[j][i]) for j in range(n)] for i in range(n)]
    return S, W


def _frob2(M) -> Any:
    return sum(M[i][j] * M[i][j] for i in range(len(M)) for j in range(len(M)))


def q_criterion(grid: Grid, u: Sequence[Any]):
    """Q = (|W|^2 - |S|^2) / 2 (Hunt, Wray & Moin 1988); Q > 0: rotation dominates strain."""
    S, W = strain_rotation(grid, u)
    return 0.5 * (_frob2(W) - _frob2(S))


def sym3_eigvals(M):
    """Eigenvalues (ascending) of a field of symmetric 3x3 matrices M[i][j] (arrays), closed form
    (trigonometric solution of the characteristic cubic, Smith 1961), vectorized on NumPy or
    CuPy. Returns (l1, l2, l3) with l1 <= l2 <= l3."""
    xp = array_module(M[0][0])
    q = (M[0][0] + M[1][1] + M[2][2]) / 3.0
    p1 = M[0][1]**2 + M[0][2]**2 + M[1][2]**2
    b00, b11, b22 = M[0][0] - q, M[1][1] - q, M[2][2] - q
    p2 = b00**2 + b11**2 + b22**2 + 2.0 * p1
    p = xp.sqrt(p2 / 6.0)
    ps = xp.where(p == 0, 1.0, p)
    # det(B / p) / 2, B = M - q I
    det = (b00 * (b11 * b22 - M[1][2]**2) - M[0][1] * (M[0][1] * b22 - M[1][2] * M[0][2])
           + M[0][2] * (M[0][1] * M[1][2] - b11 * M[0][2]))
    r = xp.clip(det / (2.0 * ps**3), -1.0, 1.0)
    phi = xp.arccos(r) / 3.0
    l3 = q + 2.0 * p * xp.cos(phi)
    l1 = q + 2.0 * p * xp.cos(phi + 2.0 * np.pi / 3.0)
    l2 = 3.0 * q - l1 - l3
    return l1, l2, l3


def lambda2(grid: Grid, u: Sequence[Any], *, method: str = "eigvalsh"):
    """lambda_2 criterion (Jeong & Hussain 1995): the middle eigenvalue of S^2 + W^2 (3D);
    lambda_2 < 0 marks a vortex core. ``method`` (declared):

    * ``"eigvalsh"`` (default): batched symmetric eigensolver (LAPACK / cuSOLVER), accurate to
      round-off everywhere (eigenvalues of a symmetric matrix are well conditioned);
    * ``"closed_form"``: trigonometric cubic formula (:func:`sym3_eigvals`), no (N, 3, 3) copy,
      but only ~sqrt(eps) |M| accurate near repeated eigenvalues (measured 7e-9 on Taylor–Green,
      tests/diagnostics/test_euler.py) — enough for an isosurface, declared lossy."""
    if grid.ndim != 3:
        raise CoordinateError("lambda2: 3D grid only (in 2D use q_criterion)")
    S, W = strain_rotation(grid, u)
    M = [[sum(S[i][k] * S[k][j] + W[i][k] * W[k][j] for k in range(3)) for j in range(3)]
         for i in range(3)]
    if method == "closed_form":
        return sym3_eigvals(M)[1]
    if method != "eigvalsh":
        raise OptionError(f"lambda2 method {method!r}: 'eigvalsh' or 'closed_form'")
    xp = grid.xp
    B = xp.stack([xp.stack(row, axis=-1) for row in M], axis=-2)      # (..., 3, 3)
    return xp.linalg.eigvalsh(B)[..., 1]


def helicity(grid: Grid, u: Sequence[Any]):
    """Helicity density h = u . omega (3D). Its integral is ``grid.integrate(h)``."""
    if grid.ndim != 3:
        raise CoordinateError("helicity: 3D grid only")
    w = vorticity(grid, u)
    return u[0] * w[0] + u[1] * w[1] + u[2] * w[2]


def enstrophy(grid: Grid, u: Sequence[Any]):
    """Enstrophy density |omega|^2 / 2."""
    w = vorticity(grid, u)
    return 0.5 * (w * w if grid.ndim == 2 else w[0]**2 + w[1]**2 + w[2]**2)


def dissipation(grid: Grid, u: Sequence[Any], nu: float):
    """Viscous dissipation density eps = 2 nu S_ij S_ij (incompressible; its volume mean equals
    nu <|omega|^2> on a periodic box, a check the tests use). ``nu`` is required (declared)."""
    S, _ = strain_rotation(grid, u)
    return 2.0 * float(nu) * _frob2(S)


def kinetic_energy(grid: Grid, u: Sequence[Any]):
    """Kinetic energy density |u|^2 / 2."""
    _check_vector(grid, u, "kinetic_energy")
    return 0.5 * sum(c * c for c in u)


def pressure(grid: Grid, u: Sequence[Any], *, rho: float = 1.0):
    """Kinematic pressure of an incompressible velocity on a fully periodic box:
    lap p = -rho d_i u_j d_j u_i, solved spectrally (exact), zero mean (gauge, declared).
    Body forces with a potential are not included (their potential adds to p; declared).
    With a wall the Neumann condition comes from the momentum equation: refused here."""
    _all_periodic(grid, "pressure")
    _check_vector(grid, u, "pressure")
    xp = grid.xp
    A = velocity_gradient(grid, u)
    n = grid.ndim
    src = -rho * sum(A[i][j] * A[j][i] for i in range(n) for j in range(n))
    k = _kvec(grid)
    k2 = sum(kk**2 for kk in k)
    k2 = xp.where(k2 == 0, 1.0, k2)
    P = -xp.fft.fftn(src) / k2
    P[(0,) * n] = 0.0
    return xp.fft.ifftn(P).real


# ---------------------------------------------------------------------- spectra
def spectrum_1d(grid: Grid, f, axis: int) -> tuple[Any, Any]:
    """One-sided 1D spectrum of ``f`` along the periodic ``axis``, averaged over the other axes
    with the grid quadrature: returns (k >= 0, E) with sum(E) = <f^2> (Parseval, one-sided:
    modes +-k folded)."""
    ax = grid.axes[axis]
    if not isinstance(ax, Periodic):
        raise CoordinateError(f"spectrum_1d: axis {axis} is not periodic")
    xp = grid.xp
    F = xp.fft.rfft(f, axis=axis) / ax.n
    P = xp.abs(F)**2
    fold = np.full(ax.n // 2 + 1, 2.0)
    fold[0] = 1.0
    if ax.n % 2 == 0:
        fold[-1] = 1.0
    shp = [1] * grid.ndim
    shp[axis] = -1
    P = xp.moveaxis(P * xp.asarray(fold).reshape(shp), axis, 0)
    # average over the other axes with their quadrature weights (last one first)
    for j in reversed([j for j in range(grid.ndim) if j != axis]):
        P = xp.tensordot(P, grid.weights(j) / ax_len(grid.axes[j]), axes=([-1], [0]))
    return grid.wavenumbers(axis, real=True), P.reshape(-1)


def ax_len(ax) -> float:
    """Length of an axis (period or interval)."""
    return float(ax.bounds[1] - ax.bounds[0])


def energy_spectrum(grid: Grid, u: Sequence[Any]) -> tuple[Any, Any]:
    """Shell-summed kinetic energy spectrum on a fully periodic box: E(k_m) = sum over
    |k| in [k_m - dk/2, k_m + dk/2) of |u_k|^2 / 2, dk = min_i 2 pi / L_i (declared binning).
    sum(E) = <|u|^2> / 2 (Parseval)."""
    _all_periodic(grid, "energy_spectrum")
    _check_vector(grid, u, "energy_spectrum")
    xp = grid.xp
    N = float(np.prod(grid.shape))
    e = sum(xp.abs(xp.fft.fftn(c) / N)**2 for c in u) * 0.5
    k = _kvec(grid)
    kmag = xp.sqrt(sum(kk**2 for kk in k))
    dk = min(2 * np.pi / ax.L for ax in grid.axes)
    idx = xp.rint(kmag / dk).astype(np.int64).reshape(-1)
    nb = int(idx.max()) + 1
    E = xp.bincount(idx, weights=e.reshape(-1), minlength=nb)
    return xp.arange(nb) * dk, E
