"""F774 (audit of 0.2.1, N36, N58): optimized DMD does not depend on the origin of the clock.
Oracle: the analytic exponential that generated the data."""
from _acceptance import check, passed, start

start(774, "optimized DMD under a shift of the time origin")
import numpy as np  # noqa: E402

from conatus.datadriven import DataInnerProduct, dmd  # noqa: E402

tau = np.array([0, 0.07, 0.18, 0.32, 0.5, 0.73, 1.0])
X = np.exp(-tau)[None, :]
worst = 0.0
for shift in (0.0, 100.0, 500.0, 1000.0, 2000.0, 1e6):
    f = dmd(X, dt=float(np.diff(tau).mean()), rank=1, method="optimized", times=tau + shift,
            inner_product=DataInnerProduct.euclidean(1), tol=1e-11, max_nfev=500)
    err = float(np.abs(f.reconstruct(tau + shift) - X).max())
    check(np.isfinite(f.amplitudes).all() and err < 1e-8, f"shift {shift}: reconstruction error {err}")
    check(abs(f.omega[0].real + 1) < 1e-6, f"shift {shift}: omega {f.omega[0]}")
    worst = max(worst, err)
passed(f"largest reconstruction error {worst:.1e}")
