"""DMD and SPOD against closed forms (numerical audit).

* Exact DMD of snapshots of an exact linear map x_{k+1} = A x_k (A non-normal, full rank, known
  eigenvalues): the continuous rates log(eig A)/dt to rounding, and the DMD modes are eigenvectors
  of A (Tu et al. 2014, Theorem 1: exact DMD of data from a linear map returns eig A when the
  snapshot matrix has full rank).
* SPOD of a diagonal vector AR(1) process x_{k+1} = diag(a) x_k + diag(s) e_k (e white, unit
  variance): the cross-spectral density is diagonal with the closed-form entries
  S_j(f) = s_j^2 dt / |1 - a_j exp(-2 pi i f dt)|^2 (two-sided, per unit frequency), so the SPOD
  eigenvalues are the sorted S_j(f); estimated within the declared chi^2 interval.
"""
from __future__ import annotations

import numpy as np

from conatus.datadriven import DataInnerProduct, dmd, spod


def test_exact_dmd_of_a_linear_map_returns_its_eigenvalues():
    rng = np.random.default_rng(7)
    n, dt = 6, 0.1
    lam = np.array([0.95, 0.8, 0.6 + 0.3j, 0.6 - 0.3j, -0.4, 0.2])  # eigenvalues of A
    D = np.diag([0.95, 0.8, 0.6, 0.6, -0.4, 0.2])
    D[2, 3], D[3, 2] = 0.3, -0.3                                     # the pair 0.6 +- 0.3i
    T = np.eye(n) + np.triu(rng.standard_normal((n, n)), 1) * 2.0   # non-normal similarity
    A = T @ D @ np.linalg.inv(T)
    assert np.allclose(np.sort_complex(np.linalg.eigvals(A)), np.sort_complex(lam))
    X = np.empty((n, 30))
    X[:, 0] = rng.standard_normal(n)
    for k in range(29):
        X[:, k + 1] = A @ X[:, k]
    ip = DataInnerProduct.quadrature(np.ones(n))
    r = dmd(X, dt=dt, rank=n, inner_product=ip, amplitudes="first")
    want = np.log(lam.astype(complex)) / dt
    for w in want:
        assert np.min(np.abs(r.omega - w)) < 1e-9 * abs(w) + 1e-10, (w, r.omega)
    for j in range(n):                                               # A phi = exp(omega dt) phi
        phi = r.modes[:, j]
        assert np.linalg.norm(A @ phi - np.exp(r.omega[j] * dt) * phi) < 1e-9 * np.linalg.norm(phi)


def test_spod_of_a_vector_ar1_process_matches_the_closed_form_spectrum():
    rng = np.random.default_rng(11)
    a = np.array([0.5, -0.3, 0.8])
    s = np.array([1.0, 1.5, 0.4])
    dt, nfft, nov, nb = 0.2, 64, 32, 1200
    nt = nov * (nb + 1)
    X = np.zeros((3, nt + 500))
    e = rng.standard_normal((3, nt + 500))
    for k in range(nt + 499):
        X[:, k + 1] = a * X[:, k] + s * e[:, k]
    X = X[:, 500:]                                                   # burn-in: stationary
    ip = DataInnerProduct.quadrature(np.ones(3))
    sp = spod(X, dt=dt, nfft=nfft, noverlap=nov, window="hann", mean="keep", onesided=False,
              inner_product=ip, n_modes=3)
    f = np.asarray(sp.frequencies)
    S = s[None, :] ** 2 * dt / np.abs(1 - a[None, :] * np.exp(-2j * np.pi * f[:, None] * dt)) ** 2
    exact = -np.sort(-S, axis=1)
    est = sp.eigenvalues[:, :3]
    ratio = est / exact
    # all three eigenvalues (review of the paper claims the three; measured mean
    # ratios 0.997 / 1.003 / 1.016 on this seed, within 1.6 % on seeds 1-3)
    assert np.all(np.abs(ratio.mean(axis=0) - 1) < 0.03), ratio.mean(axis=0)  # over 64 frequencies
    assert abs(ratio[:, 0].mean() - 1) < 0.02, ratio[:, 0].mean()
    lo, hi = sp.confidence(0.001)
    inside = (lo[:, :3] < exact) & (exact < hi[:, :3])
    assert np.all(inside.mean(axis=0) > 0.95), inside.mean(axis=0)
    # well separated eigenvalues (ratio of exact S_j > 2): the ordering is recovered
    sep = exact[:, 0] > 2 * exact[:, 1]
    assert np.all(np.abs(ratio[sep, 0] - 1) < 0.25)
