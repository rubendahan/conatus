"""Findings 92 (POD energy fraction normalised on the kept modes) and 97 (DMD
on zero data failed late with NaN then LinAlgError)."""
import numpy as np
import pytest

from conatus.datadriven import dmd, pod
from conatus.datadriven.norms import DataInnerProduct
from conatus.errors import OptionError


def test_pod_energy_fraction_of_the_total():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 10))
    ip = DataInnerProduct.euclidean(20)
    for method in ("svd", "snapshots"):
        p = pod(X, inner_product=ip, rank=2, method=method, mean="keep")
        s = np.linalg.svd(X, compute_uv=False)
        assert p.energy_fraction == pytest.approx((s[:2] ** 2).sum() / (s ** 2).sum(), rel=1e-10)
        assert p.energy_fraction < 0.9
        assert f"{p.energy_fraction:.6g}" in p.explain()
        full = pod(X, inner_product=ip, rank=10, method="svd", mean="keep")
        assert full.energy_fraction == pytest.approx(1.0, rel=1e-12)


def test_dmd_zero_data_refused_early():
    ip = DataInnerProduct.euclidean(4)
    with pytest.raises(OptionError, match="all zero"):
        dmd(np.zeros((4, 6)), dt=0.1, rank=2, inner_product=ip)
    X = np.zeros((4, 6))
    X[0] = 2.0 ** -np.arange(6)
    with pytest.raises(OptionError, match="exceeds the rank"):
        dmd(X, dt=0.1, rank=2, inner_product=ip)
    X[1, 0] = np.nan
    with pytest.raises(OptionError, match="NaN"):
        dmd(X, dt=0.1, rank=1, inner_product=ip)
