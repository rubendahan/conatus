"""conatus.cost: features read from the stacked system, argument checks.
The accuracy of the predictions is measured on a compute node against runs at resolutions the fit
did not use (CALIBRATION["checked_on"]); timings are not tested here."""
import numpy as np
import pytest

import conatus as cn
from conatus.cost import CALIBRATION, estimate, features, predict
from conatus.errors import OptionError, ResolutionError


def _rb():
    return cn.problems.native.rayleigh_benard(Lx=2.0, Ra=2500, Pr=1)


def test_features_match_the_bands_of_the_system():
    sim = _rb().simulate(N=(16, 12), scheme="RK443", dt=1e-3)
    f = features(sim)
    sysm = sim.solver.system
    assert f["unknowns"] == sysm.n
    assert f["solves"] == 4                                   # RK443: four implicit stages
    expect = 0
    for (idx, _, _), (_, n, r, kl, ku) in zip(sysm._buckets, sysm.bands(), strict=False):
        expect += len(idx) * n * (2 * kl + ku + 1 + 2 * r)
    if not sysm._cbuckets and not sysm._zbuckets:
        assert f["solve_work"] == expect
    assert f["grid_points"] == 24 * 18                        # dealias 3/2 on both axes
    assert f["state_bytes"] == sysm.n * 8


def test_multistep_scheme_has_one_solve_per_step():
    sim = _rb().simulate(N=(16, 12), scheme="SBDF2", dt=1e-3)
    assert features(sim)["solves"] == 1


def test_prediction_is_linear_in_the_features():
    cal = dict(time=dict(a=1.0, b=2.0, c=3.0, d=4.0), memory=dict(e=1.0, f=0.5, g=2.0, h=7.0))
    f = dict(solves=2, solve_work=10, fft_work=100, factor_bytes=8, state_bytes=16, grid_bytes=32)
    sec, mem = predict(f, cal)
    assert sec == 2 * (10 + 200 + 3) + 4
    assert mem == 8 + 0.5 * 16 * 2 + 64 + 7


def test_estimate_arguments_and_explain():
    m = _rb()
    with pytest.raises(OptionError, match="not both"):
        estimate(m, N=(16, 12), scheme="RK443", dt=1e-3, steps=10, until=0.01)
    with pytest.raises(ResolutionError, match="whole number of steps"):
        estimate(m, N=(16, 12), scheme="RK443", dt=1e-3, until=0.0105)
    est = estimate(m, N=(16, 12), scheme="RK443", dt=1e-3, until=0.5)
    assert est.steps == 500
    assert est.run_s == pytest.approx(500 * est.s_per_step)
    assert np.isfinite(est.s_per_step) and est.s_per_step >= 0
    text = est.explain()
    assert "time per step" in text and "memory" in text and str(CALIBRATION["machine"]) in text
