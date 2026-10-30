"""Audit of 0.2.1, lot C: SPOD confidence and folding, RBF distances, optimized DMD under a clock
shift, integer snapshots, Gram validation, sampling intervals.

Oracles are analytic values, ``scipy.stats.chi2``, direct time-domain energies and independent
replicated datasets. The code's own output is never the reference.
"""
import warnings

import numpy as np
import pytest
from scipy.stats import chi2

from conatus.datadriven import (DataInnerProduct, Dictionary, dmd, edmd, monomials, pod, rbf, spod,
                                uniform_dt, welch_blocks)
from conatus.errors import ConvergenceError, OptionError


def _spod(Q, *, nfft=8, noverlap=0, window="boxcar", onesided=True, ip=None, n_modes=1, mean="keep"):
    return spod(Q, dt=1.0, nfft=nfft, noverlap=noverlap, window=window, mean=mean,
                onesided=onesided, inner_product=ip or DataInnerProduct.euclidean(Q.shape[0]),
                n_modes=n_modes)


def _coverage(draw, true, bins, *, trials=1000, approximate=False, **kw):
    hits = np.zeros(len(bins))
    for _ in range(trials):
        lo, hi = _spod(draw(), **kw).confidence(0.05, approximate=approximate)
        hits += [lo[k, 0] <= true[k] <= hi[k, 0] for k in bins]
    return hits / trials


# ------------------------------------------------------------------------------- N01
SCALAR = {
    # (draw, keyword arguments, {bin: population density}): replicated independent datasets
    "real_boxcar": (lambda g: g.normal(size=(1, 128)), {}, {0: 1.0, 1: 2.0, 4: 1.0}),
    "impulse_window": (lambda g: g.normal(size=(1, 128)),
                       {"window": np.array([1.0, 0, 0, 0, 0, 0, 0, 0])}, {1: 2.0}),
    "real_cast_to_complex": (lambda g: g.normal(size=(1, 128)).astype(complex),
                             {"onesided": False}, {0: 1.0, 4: 1.0}),
    "proper_complex": (lambda g: (g.normal(size=(1, 128)) + 1j * g.normal(size=(1, 128))) / np.sqrt(2),
                       {"onesided": False}, {0: 1.0, 1: 1.0, 4: 1.0}),
    "real_hann": (lambda g: g.normal(size=(1, 128)), {"window": "hann"}, {1: 2.0, 2: 2.0}),
}


@pytest.mark.parametrize("case", sorted(SCALAR))
def test_n01_scalar_interval_covers_on_replicated_data(case):
    draw, kw, true = SCALAR[case]
    g = np.random.default_rng(20261030)
    cov = _coverage(lambda: draw(g), true, list(true), **kw)
    assert np.all(np.abs(cov - 0.95) < 0.025), cov


@pytest.mark.parametrize("n", [2, 8])
def test_n01_multivariate_interval_is_refused_without_the_approximation(n):
    r = _spod(np.random.default_rng(1).normal(size=(n, 128)))
    with pytest.raises(OptionError, match="approximate=True"):
        r.confidence(0.05)


def test_n01_overlapping_blocks_are_refused_without_the_approximation():
    r = _spod(np.random.default_rng(1).normal(size=(1, 128)), noverlap=4)
    with pytest.raises(OptionError, match="approximate=True"):
        r.confidence(0.05)


def test_n01_approximation_is_the_stated_formula_and_is_not_calibrated_for_a_repeated_spectrum():
    g = np.random.default_rng(2)
    r = _spod(g.normal(size=(8, 128)))
    lo, hi = r.confidence(0.05, approximate=True)
    nu = 32.0
    assert hi[1, 0] == pytest.approx(nu * r.eigenvalues[1, 0] / chi2.ppf(0.025, nu), rel=1e-14)
    assert lo[1, 0] == pytest.approx(nu * r.eigenvalues[1, 0] / chi2.ppf(0.975, nu), rel=1e-14)
    # the documented limit: white data in 8 channels, population eigenvalue 2 at an interior bin
    cov = _coverage(lambda: g.normal(size=(8, 128)), {1: 2.0}, [1], trials=300, approximate=True)
    assert cov[0] < 0.05
    assert "approximate" in r.explain()


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, float("nan")])
def test_n01_significance_level_is_checked(alpha):
    r = _spod(np.random.default_rng(1).normal(size=(1, 128)))
    with pytest.raises(OptionError):
        r.confidence(alpha)


# ------------------------------------------------------------------------------- N09
def _direct_energy(Q, W):
    return float(np.einsum("ij,ik,kj->", Q.conj(), W, Q).real / Q.shape[1])


W_C = np.array([[1.0, 0.75j], [-0.75j, 1.0]])
W_R = np.array([[1.0, 0.5], [0.5, 2.0]])


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("nb", [1, 3])                 # n x n and N_b x N_b branches
def test_n09_one_sided_complex_metric_is_refused(sign, nb):
    Q = np.tile(np.array([[1.0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]]), (1, nb))
    W = W_C if sign > 0 else W_C.conj()
    with pytest.raises(OptionError, match="onesided"):
        _spod(Q, ip=DataInnerProduct.gram(W))


@pytest.mark.parametrize("nfft", [8, 9])
@pytest.mark.parametrize("nb", [1, 5])
@pytest.mark.parametrize("cast", [False, True])
def test_n09_one_sided_real_metric_keeps_the_time_domain_energy(nfft, nb, cast):
    Q = np.random.default_rng(3).normal(size=(2, nfft * nb))
    W = W_R.astype(complex) if cast else W_R
    r = _spod(Q, nfft=nfft, ip=DataInnerProduct.gram(W))
    assert r.eigenvalues.sum() / nfft == pytest.approx(_direct_energy(Q, W_R), rel=1e-12)


@pytest.mark.parametrize("nfft", [8, 9])
@pytest.mark.parametrize("nb", [1, 5])
def test_n09_two_sided_complex_metric_keeps_the_time_domain_energy(nfft, nb):
    g = np.random.default_rng(4)
    for Q in (g.normal(size=(2, nfft * nb)),
              g.normal(size=(2, nfft * nb)) + 1j * g.normal(size=(2, nfft * nb))):
        r = _spod(Q, nfft=nfft, onesided=False, ip=DataInnerProduct.gram(W_C))
        assert r.eigenvalues.sum() / nfft == pytest.approx(_direct_energy(Q, W_C), rel=1e-12)


# ------------------------------------------------------------------------------- N12
@pytest.mark.parametrize("x,expected", [(1j, np.exp(-1)), (1 + 2j, np.exp(-5)), (1.0, np.exp(-1))])
def test_n12_gaussian_uses_the_hermitian_squared_distance(x, expected):
    D = rbf(np.array([[0.0]]), epsilon=1.0)
    v = D(np.array([[x]]))[0, -1]
    assert v.imag == 0 and v.real == pytest.approx(expected, rel=1e-14)


def test_n12_complex_center_is_kept_and_distance_is_phase_invariant():
    c = np.array([[1 + 1j, 2j]])
    D = rbf(c, epsilon=0.5)
    x = np.array([[0.3 - 0.2j, 1.0 + 0.5j]])
    want = np.exp(-0.25 * np.sum(np.abs(x - c) ** 2))
    assert D(x)[0, -1] == pytest.approx(want, rel=1e-14)
    rotated = np.exp(0.7j)
    assert rbf(c * rotated, 0.5)(x * rotated)[0, -1] == pytest.approx(want, rel=1e-13)


# ------------------------------------------------------------------------- N36, N58
TAU = np.array([0, 0.07, 0.18, 0.32, 0.5, 0.73, 1.0])


@pytest.mark.parametrize("shift", [0.0, 10.0, 100.0, 500.0, 1000.0, 2000.0, 1e6])
@pytest.mark.parametrize("case", ["n36_nonuniform", "n58_dt0.1", "n58_dt0.05", "n58_jitter"])
def test_n36_n58_clock_origin_does_not_change_the_fit(shift, case):
    if case == "n36_nonuniform":
        rel, rate, dt = TAU, -1.0, float(np.diff(TAU).mean())
    else:
        dt = 0.05 if case.endswith("0.05") else 0.1
        rel, rate = dt * np.arange(20), -0.5
        if case == "n58_jitter":
            rel = rel + 0.01 * dt * np.sin(np.arange(20.0))
    X = np.exp(rate * rel)[None, :]
    times = rel + shift
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        f = dmd(X, dt=dt, rank=1, inner_product=DataInnerProduct.euclidean(1), method="optimized",
                times=times, tol=1e-11, max_nfev=500)
        rec = f.reconstruct(times)
    # analytic: the sample of the first time is 1, the rate is the declared one
    assert f.omega[0].real == pytest.approx(rate, abs=1e-6)
    assert abs(f.amplitudes[0] * f.modes[0, 0]) == pytest.approx(1.0, abs=1e-8)
    assert np.abs(rec - X).max() < 1e-8
    assert f.misfit == pytest.approx(np.linalg.norm(rec - X) / np.linalg.norm(X), abs=1e-15)


def test_n36_large_growth_is_representable():
    # x = exp(t), t up to 700: finite, its square is not
    t = np.linspace(0.0, 700.0, 15)
    X = np.exp(t)[None, :]
    f = dmd(X, dt=t[1], rank=1, inner_product=DataInnerProduct.euclidean(1), method="optimized",
            times=t, tol=1e-10, max_nfev=200)
    assert f.omega[0].real == pytest.approx(1.0, rel=1e-8)
    assert np.abs(f.reconstruct(t) / X - 1).max() < 1e-6
    e = dmd(X, dt=t[1], rank=1, inner_product=DataInnerProduct.euclidean(1))
    assert np.isfinite(e.residuals).all() and e.omega[0].real == pytest.approx(1.0, rel=1e-10)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_n36_nonfinite_times_are_refused(bad):
    X = np.exp(-TAU)[None, :]
    t = TAU.copy()
    t[3] = bad
    with pytest.raises(OptionError, match="times"):
        dmd(X, dt=0.1, rank=1, inner_product=DataInnerProduct.euclidean(1), method="optimized",
            times=t, tol=1e-10, max_nfev=50)


def test_n36_unusable_fit_is_refused(monkeypatch):
    import scipy.optimize as so
    real = so.least_squares

    def stuck(fun, x0, **kw):               # an optimizer that ends on a nonfinite parameter
        sol = real(fun, x0, **kw)
        sol.x = sol.x * np.nan
        return sol
    monkeypatch.setattr(so, "least_squares", stuck)
    with pytest.raises(ConvergenceError):
        dmd(np.exp(-TAU)[None, :], dt=0.1, rank=1, inner_product=DataInnerProduct.euclidean(1),
            method="optimized", times=TAU, tol=1e-10, max_nfev=50)


# ------------------------------------------------------------------------------- N56
BASE = np.array([[1, 2, 3, 4], [4, 1, 3, 2]])


@pytest.mark.parametrize("dtype,s", [(np.int16, 100), (np.int32, 10 ** 4), (np.int64, 10 ** 10),
                                     (np.uint16, 100), (np.uint32, 10 ** 4), (np.uint64, 10 ** 10)])
@pytest.mark.parametrize("method", ["snapshots", "svd"])
def test_n56_pod_of_integers_has_the_analytic_spectrum(dtype, s, method):
    X = (s * BASE).astype(dtype)
    r = pod(X, inner_product=DataInnerProduct.euclidean(2), rank=2, method=method, mean="keep")
    assert r.energies / float(s) ** 2 == pytest.approx([13.25, 1.75], rel=1e-12)
    assert r.orthonormality < 1e-12 and r.reconstruction < 1e-12


def test_n56_small_float_is_promoted_as_well():
    r = pod((1000.0 * BASE).astype(np.float16), inner_product=DataInnerProduct.euclidean(2), rank=2,
            method="snapshots", mean="keep")
    assert r.energies / 1e6 == pytest.approx([13.25, 1.75], rel=1e-12)


def test_n56_uint_mean_subtraction():
    X = np.array([[1, 3], [5, 1]], dtype=np.uint8)            # X' = [[-1, 1], [2, -2]]
    r = pod(X, inner_product=DataInnerProduct.euclidean(2), rank=1, method="snapshots",
            mean="subtract")
    assert r.energies[0] == pytest.approx(5.0, rel=1e-13)     # sigma^2 = 10, m = 2


@pytest.mark.parametrize("dtype,s", [(np.int16, 100), (np.int64, 2 * 10 ** 8), (np.int64, 10 ** 9)])
def test_n56_edmd_of_integers_has_the_analytic_eigenvalue(dtype, s):
    x = np.arange(1, 10)[:, None]
    X, Y = (s * x).astype(dtype), None
    Y = (2 * s * x).astype(np.int64)
    e = edmd(X, Y, dictionary=Dictionary(lambda v: v, ["x0"], [0]), rcond=1e-12, dt=None)
    assert e.eigenvalues[0].real == pytest.approx(2.0, rel=1e-12)


def test_n56_edmd_diagonal_map_and_monomials():
    X = (100 * np.array([[1, 0], [0, 1], [1, 1], [2, 1]])).astype(np.int16)
    Y = X * np.array([2, 1], dtype=np.int16)
    e = edmd(X, Y, dictionary=Dictionary(lambda v: v, ["x0", "x1"], [0, 1]), rcond=1e-12, dt=None)
    assert np.sort(e.eigenvalues.real) == pytest.approx([1.0, 2.0], rel=1e-12)
    # monomials of integers: x^2 of 300 does not fit int16
    D = monomials(1, 2)
    assert D(np.array([[300]], dtype=np.int16))[0, 2] == 90000.0


def test_n56_spod_and_gram_of_integers():
    W = np.array([[2, 1], [1, 2]], dtype=np.int16)
    ip = DataInnerProduct.gram(W)
    assert (ip.F.conj().T @ ip.F) == pytest.approx(W.astype(float))
    Q = (100 * np.random.default_rng(5).integers(-9, 9, size=(1, 64))).astype(np.int16)
    r = _spod(Q)
    assert r.eigenvalues.sum() / 8 == pytest.approx(np.mean(Q.astype(float) ** 2), rel=1e-12)


# ------------------------------------------------------------------------------- N57
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_n57_nonfinite_gram_is_refused(bad):
    with pytest.raises(OptionError, match="finite"):
        DataInnerProduct.gram(np.array([[2.0, bad], [0.0, 2.0]]))
    with pytest.raises(OptionError, match="finite"):
        DataInnerProduct.gram(np.array([[2.0, 0.0], [0.0, bad]], dtype=complex))


@pytest.mark.parametrize("s", [1e-300, 1e-200, 1.0, 1e200, 1e300])
def test_n57_non_hermitian_is_refused_at_any_scale(s):
    with pytest.raises(OptionError, match="Hermitian"):
        DataInnerProduct.gram(s * np.array([[2.0, 1.0], [0.0, 2.0]]))
    with pytest.raises(OptionError, match="Hermitian"):
        DataInnerProduct.gram(s * np.array([[2.0, 1j], [1j, 2.0]]))


@pytest.mark.parametrize("s", [1e-300, 1e-200, 1.0, 1e200, 1e300])
def test_n57_valid_metric_is_kept_at_any_scale(s):
    W = np.array([[2.0, 0.5j], [-0.5j, 3.0]])
    F = DataInnerProduct.gram(s * W).F
    assert np.abs((F.conj().T / np.sqrt(s)) @ (F / np.sqrt(s)) - W).max() < 1e-14


def test_n57_indefinite_and_empty_are_refused():
    with pytest.raises(OptionError, match="positive"):
        DataInnerProduct.gram(np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(OptionError):
        DataInnerProduct.gram(np.zeros((0, 0)))
    with pytest.raises(OptionError, match="positive"):
        DataInnerProduct.gram(np.zeros((2, 2)))


# ------------------------------------------------------------------------------- N61
@pytest.mark.parametrize("dt", [0.0, float("nan"), float("inf"), -float("inf"), -0.1])
def test_n61_edmd_refuses_an_invalid_interval(dt):
    X = np.arange(1.0, 6.0)[:, None]
    with pytest.raises(OptionError, match="dt"):
        edmd(X, 2 * X, dictionary=monomials(1, 1), rcond=1e-12, dt=dt)


@pytest.mark.parametrize("dt", [0.1, 2.0])
def test_n61_edmd_rate_is_analytic(dt):
    X = np.arange(1.0, 6.0)[:, None]
    e = edmd(X, 2 * X, dictionary=monomials(1, 1), rcond=1e-12, dt=dt)
    k = int(np.argmax(np.abs(e.eigenvalues)))
    assert e.omega[k].real == pytest.approx(np.log(2.0) / dt, rel=1e-12)
    assert edmd(X, 2 * X, dictionary=monomials(1, 1), rcond=1e-12, dt=None).omega is None


@pytest.mark.parametrize("times", [[0.0, np.inf], [-np.inf, 0.0], [-1e308, 1e308]])
def test_n61_uniform_dt_refuses_an_unusable_interval(times):
    with pytest.raises(OptionError):
        uniform_dt(np.array(times), rtol=1e-9)


def test_n61_uniform_dt_keeps_a_large_finite_step():
    assert uniform_dt(np.array([-1e308, 0.0, 1e308]), rtol=1e-9) == 1e308
    assert uniform_dt(np.array([1.0, 1.5, 2.0]), rtol=1e-9) == 0.5


@pytest.mark.parametrize("dt", [0.0, float("nan"), float("inf"), -1.0])
def test_n61_spod_refuses_an_invalid_interval(dt):
    Q = np.random.default_rng(1).normal(size=(1, 64))
    with pytest.raises(OptionError, match="dt"):
        spod(Q, dt=dt, nfft=8, noverlap=0, window="boxcar", mean="keep", onesided=True,
             inner_product=DataInnerProduct.euclidean(1), n_modes=1)
    with pytest.raises(OptionError, match="dt"):
        welch_blocks(Q, dt=dt, nfft=8, noverlap=0, window="boxcar", mean="keep")


@pytest.mark.parametrize("w", [np.zeros(8), np.full(8, np.nan), np.array([np.inf] + [1.0] * 7)])
def test_n61_spod_refuses_an_unusable_window(w):
    with pytest.raises(OptionError, match="window"):
        welch_blocks(np.ones((1, 64)), dt=1.0, nfft=8, noverlap=0, window=w, mean="keep")
