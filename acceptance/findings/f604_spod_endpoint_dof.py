"""F604 (audit of 0.1.2, F04): SPOD 95 % intervals at f = 0 and Nyquist of real data use N_b
degrees of freedom (the block coefficient is real there), 2 N_b elsewhere.

Oracle: scipy.stats.chi2 bounds; Monte Carlo coverage of white real noise (true one-sided
density 1 at DC and Nyquist, 2 inside)."""
from _acceptance import check, passed, start

start(604, "SPOD confidence coverage at DC, interior and Nyquist")
import numpy as np  # noqa: E402

from conatus.datadriven import DataInnerProduct, spod  # noqa: E402

rng = np.random.default_rng(1492)
nb, nfft, trials = 16, 8, 1500
hits = np.zeros(3)
for _ in range(trials):
    r = spod(rng.normal(size=(1, nb * nfft)), dt=1.0, nfft=nfft, noverlap=0, window="boxcar",
             mean="keep", onesided=True, inner_product=DataInnerProduct.euclidean(1), n_modes=1)
    lo, hi = r.confidence(0.05)
    for i, (k, true) in enumerate(((0, 1.0), (1, 2.0), (4, 1.0))):
        hits[i] += lo[k, 0] <= true <= hi[k, 0]
cov = hits / trials
check(bool(np.all(np.abs(cov - 0.95) < 0.025)), f"coverage DC/interior/Nyquist {cov}")
passed(f"coverage DC/interior/Nyquist {cov}")
