"""``m.eigenvalues(N, region=cn.box(...))`` on Orr-Sommerfeld against the full dense spectrum."""
from __future__ import annotations

import numpy as np
import pytest

import conatus as cn
from conatus.contour import RegionEVPResult
from conatus.errors import OptionError


def _os_model():
    m = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="Orr-Sommerfeld", Re=1e4, alpha=1.0)
    m.scalar("v")
    m.given(U=lambda y: 1 - y**2)
    m.eigenvalue("c", growth="imag")
    m.define("L2(f) = dy(dy(f)) - alpha**2*f")
    m.equations("L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + c*L2(v) = 0")
    m.at("walls", "v = 0", "dy(v) = 0")
    return m


@pytest.fixture(scope="module")
def os_model():
    return _os_model()


#: boxes of the c plane (edges at least 9e-4 from every dense eigenvalue at N = 48 and 64):
#: the unstable Tollmien-Schlichting mode alone, the A branch, the P branch near the junction,
#: and two boxes holding most of the resolved spectrum
BOXES = [cn.box(re=(0.2, 0.3), im=(-0.01, 0.01)), cn.box(re=(0.1, 0.5), im=(-0.3, 0.05)),
         cn.box(re=(0.55, 0.8), im=(-0.3, -0.05)), cn.box(re=(-0.1, 1.1), im=(-0.5, 0.1))]


@pytest.mark.parametrize("N", [48, 64])
def test_os_box_counts_match_the_dense_spectrum(os_model, N):
    dense = os_model.eigenvalues(N=N, spurious="none")
    for b in BOXES:
        exact = dense.values[b.contains(dense.values)]
        res = os_model.eigenvalues(N=N, region=b, spurious="none")
        assert isinstance(res, RegionEVPResult)
        assert res.count == exact.size == len(res) == int(res.multiplicity.sum())
        for z in exact:                                  # same eigenvalues (QZ accuracy)
            assert np.min(np.abs(res.values - z)) < 1e-9
        assert (res.residuals < 1e-12).all()
        assert "argument principle" in res.count_method


def test_os_region_result_reads_like_an_evp_result(os_model):
    b = BOXES[0]
    res = os_model.eigenvalues(N=48, region=b)
    i = int(res.order()[0])
    assert abs(res.values[i] - (0.23752649 + 0.00373967j)) < 1e-6
    text = res.explain()
    assert "count 1" in text and "certified" in text and "outside the region not computed" in text
    assert "infinite (M singular" not in text
    assert "count 1" in repr(res)
    full = res.explain(full=True)
    assert "Beyn contour integral" in full and "Hankel" in full
    mode = res.mode(i)                                   # eigenfunction on the grid, as dense
    assert abs(abs(mode["v"]).max() - 1) < 1e-12


def test_os_region_with_drift_keeps_the_count(os_model):
    res = os_model.eigenvalues(N=48, region=BOXES[1], drift=72, drift_tol=1e-6)
    assert isinstance(res, RegionEVPResult) and res.count == 12
    assert 0 < len(res) <= 12 and res.drift is not None
    assert "kept" in res.explain()


def test_region_refusals(os_model):
    b = BOXES[0]
    with pytest.raises(OptionError, match="shift-invert"):
        os_model.eigenvalues(N=48, region=b, method="shift-invert")
    # an explicit method="dense" with a region used to be switched to the contour method without
    # a word (critic review of L10): refused, naming the fix; method="contour" is accepted
    with pytest.raises(OptionError, match="drop method="):
        os_model.eigenvalues(N=48, region=b, method="dense")
    assert os_model.eigenvalues(N=48, region=b, method="contour").count == 1
    with pytest.raises(OptionError, match="target"):
        os_model.eigenvalues(N=48, region=b, target=0.2)
    with pytest.raises(OptionError, match="left"):
        os_model.eigenvalues(N=48, region=b, left=True)
    with pytest.raises(OptionError, match="needs region"):
        os_model.eigenvalues(N=48, method="contour")
    with pytest.raises(OptionError, match="cn.box"):
        os_model.eigenvalues(N=48, region=(0.2, 0.3))


def test_entry_points():
    import conatus.expert as ex
    assert cn.contour.count is ex.contour.count and cn.contour.beyn is ex.contour.beyn
    assert cn.box is cn.contour.box
    assert ex.RegionEVPResult is RegionEVPResult and "ContourBox" not in ex.__all__   # M5: undocumented
    assert "argument principle" in cn.contour.explain() or "winding" in cn.contour.explain()


@pytest.mark.slow                             # about 40 s (the cut box re-solved)
def test_os_large_box_needs_a_cut(os_model):
    # N = 96, 46 eigenvalues with P-branch pairs 1e-5 apart: the rounds on the whole box leave
    # roots uncertified, the box is cut, every root is certified (was refused)
    b = BOXES[3]
    dense = os_model.eigenvalues(N=96, spurious="none")
    exact = dense.values[b.contains(dense.values)]
    res = os_model.eigenvalues(N=96, region=b, spurious="none")
    assert res.count == exact.size == len(res) == 46
    assert any("cut at" in s for s in res.contour.log)
    # the S-branch modes are ill-conditioned (condition ~1e8): QZ and the polished roots agree
    # to about 2e-8 there, to 1e-11 elsewhere
    assert max(np.min(np.abs(res.values - z)) for z in exact) < 1e-7
    assert (res.residuals <= 1e-13).all()
