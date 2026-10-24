"""F610 (audit of 0.1.2, F22, code side): with norm weights W the corrector's phase residual is
h = t_x^T W (x - x_pred) + t_p (p - p_pred), and the corrected point satisfies h = 0.

Oracle: h computed with NumPy from the returned point; the branch x0² + x1² = p, x0 = 2 x1."""
from _acceptance import check, passed, start

start(610, "weighted continuation phase residual")
import numpy as np  # noqa: E402

from conatus.continuation.core import ContinuationSettings, PseudoArclength  # noqa: E402

W = np.array([4.0, 0.25])
pa = PseudoArclength(lambda x, p: np.array([x[0] ** 2 + x[1] ** 2 - p, x[0] - 2 * x[1]]),
                     lambda x, p: np.array([[2 * x[0], 2 * x[1]], [1.0, -2.0]]),
                     ContinuationSettings(ds=0.1, ds_min=1e-6, ds_max=0.5, tolerance=1e-12,
                                          max_points=3),
                     dGdp=lambda x, p: np.array([-1.0, 0.0]), weights=W)
x0, p0 = np.array([2.0, 1.0]), 5.0
t = pa.tangent(x0, p0, np.array([0.0, 0.0, 1.0]))
z_pred = np.concatenate([x0, [p0]]) + 0.2 * t
z, rel, _ = pa.correct(z_pred, t)
h = float(np.sum(W * t[:2] * (z[:2] - z_pred[:2])) + t[2] * (z[2] - z_pred[2]))
check(abs(h) < 1e-11, f"weighted phase residual {h:.2e} at the corrected point")
check(abs(z[0] ** 2 + z[1] ** 2 - z[2]) < 1e-11, "corrected point off the branch")
passed(f"h = {h:.1e}")
