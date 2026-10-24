"""Audit of 0.1.2, F04: SPOD confidence intervals at f = 0 and Nyquist of real data.

Oracle: for independent real Gaussian blocks, N_b S_hat / S ~ chi2(N_b) where the block
coefficient is real (f = 0, Nyquist of even nfft) and chi2(2 N_b) elsewhere (two quadratures);
scipy.stats.chi2 gives the bounds and the exact coverage. A Monte Carlo coverage check at the
three kinds of frequency follows.
"""
import numpy as np
from scipy.stats import chi2

from conatus.datadriven import DataInnerProduct, spod


def _run(q, nfft=8):
    return spod(q, dt=1.0, nfft=nfft, noverlap=0, window="boxcar", mean="keep", onesided=True,
                inner_product=DataInnerProduct.euclidean(q.shape[0]), n_modes=1)


def test_dof_and_bounds_follow_the_chi2_law_of_each_frequency():
    nb, nfft = 16, 8
    r = _run(np.random.default_rng(0).normal(size=(1, nb * nfft)), nfft)
    nu = r.dof()
    assert nu[0] == nb and nu[-1] == nb and np.all(nu[1:-1] == 2 * nb)
    lo, hi = r.confidence(0.05)
    for k in range(len(nu)):
        n = nb if k in (0, len(nu) - 1) else 2 * nb
        lam = r.eigenvalues[k, 0]
        assert np.isclose(lo[k, 0], n * lam / chi2.ppf(0.975, n), rtol=1e-12)
        assert np.isclose(hi[k, 0], n * lam / chi2.ppf(0.025, n), rtol=1e-12)


def test_complex_data_keep_two_nb_everywhere():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(1, 64)) + 1j * rng.normal(size=(1, 64))
    r = spod(q, dt=1.0, nfft=8, noverlap=0, window="boxcar", mean="keep", onesided=False,
             inner_product=DataInnerProduct.euclidean(1), n_modes=1)
    assert np.all(r.dof() == 2 * r.nblocks)


def test_monte_carlo_coverage_is_nominal_at_dc_interior_and_nyquist():
    """White real noise of variance 1: one-sided density 1 at DC and Nyquist, 2 inside. With the
    former 2 N_b law the DC/Nyquist coverage was 0.833 (exact, chi2); now 0.95 within sampling."""
    rng = np.random.default_rng(1492)
    nb, nfft, trials = 16, 8, 1500
    hits = np.zeros(3)
    for _ in range(trials):
        r = _run(rng.normal(size=(1, nb * nfft)), nfft)
        lo, hi = r.confidence(0.05)
        for i, (k, true) in enumerate(((0, 1.0), (1, 2.0), (4, 1.0))):
            hits[i] += lo[k, 0] <= true <= hi[k, 0]
    cov = hits / trials
    assert np.all(np.abs(cov - 0.95) < 0.025), cov      # 0.833 before: 20 sigma away
