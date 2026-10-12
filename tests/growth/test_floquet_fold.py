"""Folding of Floquet exponents into (-pi/P, pi/P] moves a value by whole periods only.

``_fold`` replaced every Im mu with ``np.isclose(im, -w/2)`` -- NumPy's
default rtol = 1e-5, atol = 1e-8 -- by w/2 exactly: an exponent at Im mu = -w/2 + 1e-5 came back
as +w/2, its offset (and its distance to the neutral harmonic) silently lost."""
import numpy as np

from conatus.growth.periodic import FOLD_ROUNDING, _fold


def test_fold_keeps_the_offset_near_the_lower_end():
    P = 2 * np.pi / 3.0
    w = 2 * np.pi / P
    for d in (1e-5, 3e-6, 1e-9):
        mu = np.array([-0.1 + 1j * (-w / 2 + d)])
        out = _fold(mu, P)
        assert abs(out[0].imag - (-w / 2 + d)) <= 4 * np.finfo(float).eps * w, (d, out)
        assert out[0].real == mu[0].real


def test_fold_is_a_shift_by_whole_periods():
    rng = np.random.default_rng(0)
    P = 0.7
    w = 2 * np.pi / P
    mu = rng.standard_normal(200) + 1j * rng.uniform(-5 * w, 5 * w, 200)
    out = _fold(mu, P)
    assert np.all(out.imag > -w / 2) and np.all(out.imag <= w / 2)
    k = (mu.imag - out.imag) / w
    assert np.max(np.abs(k - np.round(k))) <= 1e-12
    # the lower end itself (to the rounding of the mod) goes to the upper end
    lo = _fold(np.array([1j * (-w / 2)]), P)[0]
    assert abs(lo.imag - w / 2) <= FOLD_ROUNDING * w
