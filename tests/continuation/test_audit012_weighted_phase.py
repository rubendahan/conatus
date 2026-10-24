"""Audit of 0.1.2, F22 (code side): with norm weights W, the corrector's phase residual and its
bordered row use the same metric: h = t_x^T W (x - x_pred) + t_p (p - p_pred), row = (W t_x, t_p).

Oracles: the row equals the central finite-difference gradient of h (NumPy); the corrected point
lies on the weighted hyperplane h = 0 and on the circle x0² + x1² = p (analytic)."""
import numpy as np

from conatus.continuation.core import ContinuationSettings, PseudoArclength

W = np.array([4.0, 0.25])


def _pa():
    def G(x, p):
        return np.array([x[0] ** 2 + x[1] ** 2 - p, x[0] - 2 * x[1]])

    def J(x, p):
        return np.array([[2 * x[0], 2 * x[1]], [1.0, -2.0]])
    cfg = ContinuationSettings(ds=0.1, ds_min=1e-6, ds_max=0.5, tolerance=1e-12, max_points=3)
    return PseudoArclength(G, J, cfg, dGdp=lambda x, p: np.array([-1.0, 0.0]), weights=W)


def test_bordered_row_is_the_gradient_of_the_weighted_phase_residual():
    pa = _pa()
    x, p = np.array([2.0, 1.0]), 5.0
    t = np.array([0.3, -0.7, 0.5])
    z_pred = np.array([1.9, 1.2, 5.1])

    def h(z):
        return float(np.sum(W * t[:2] * (z[:2] - z_pred[:2])) + t[2] * (z[2] - z_pred[2]))
    A, _ = pa._bordered(x, p, t)
    row = np.asarray(A.toarray() if hasattr(A, "toarray") else A)[-1]
    z, e = np.array([2.0, 1.0, 5.0]), np.eye(3)
    fd = np.array([(h(z + 1e-6 * e[i]) - h(z - 1e-6 * e[i])) / 2e-6 for i in range(3)])
    assert np.allclose(row, fd, rtol=1e-9, atol=1e-12)
    assert not np.allclose(row[:2], t[:2])          # the plain dot would differ (W != I)


def test_corrected_point_is_on_the_weighted_hyperplane():
    pa = _pa()
    x0, p0 = np.array([2.0, 1.0]), 5.0
    t = pa.tangent(x0, p0, np.array([0.0, 0.0, 1.0]))
    z_pred = np.concatenate([x0, [p0]]) + 0.2 * t
    z, rel, _ = pa.correct(z_pred, t)
    h = np.sum(W * t[:2] * (z[:2] - z_pred[:2])) + t[2] * (z[2] - z_pred[2])
    assert abs(h) < 1e-11
    assert abs(z[0] ** 2 + z[1] ** 2 - z[2]) < 1e-11 and abs(z[0] - 2 * z[1]) < 1e-11
