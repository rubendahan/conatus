"""Critical point of a ready linearization: ``lin.parametric`` with the
wavenumber as a continuous pencil parameter, the mode at the critical point evaluated exactly
(``EVPMode.at``), one-line reprs."""
import numpy as np
import pytest

import conatus as cn
import conatus.expert as X
from conatus.errors import OptionError


def test_rayleigh_benard_critical_point_from_the_ready_linearization():
    lin = cn.problems.native.rayleigh_benard_evp(k=3.1, Ra=1708.0, Pr=1.0)
    assert lin.wavenumber_names() == {"k": "x"}
    P = lin.parametric(24, over=("Ra", "k"), drift=32, drift_tol=1e-6)
    crit = X.critical(P, "Ra", bracket=(1500.0, 2000.0), over={"k": (2.5, 3.8)}, xtol=1e-9)
    assert abs(crit.value - 1707.7618) < 1e-3 and abs(crit.over["k"] - 3.11632) < 1e-4
    r = repr(crit)
    assert "\n" not in r and r.startswith("CriticalPoint(Ra_c=1707.76") and len(r) < 200
    assert "\n" not in repr(crit.mode)


def test_fixed_wavenumber_matches_eigenvalues():
    lin = cn.problems.native.rayleigh_benard_evp(k=np.pi, Ra=2500.0, Pr=1.0)
    P = lin.parametric(24, over=("Ra",), k=np.pi, drift=32, drift_tol=1e-6)
    m = P.least_stable()                                  # declared Ra = 2500, logged
    assert abs(m.sigma.real - 5.5425059) < 1e-6 and "declared values of the model" in m.note
    with pytest.raises(OptionError, match="Did you mean 'Ra'"):
        lin.parametric(16, over=("Raa",))


def test_mode_at_exact_points_and_walls():
    lin = cn.problems.native.rayleigh_benard_evp(k=np.pi, Ra=2500.0, Pr=1.0)
    res = lin.eigenvalues(24, k=np.pi, drift=32, drift_tol=1e-6)
    md = res.mode(int(res.order()[0]))
    z = md.coordinate("z")
    assert np.allclose(md.at("T", z=z), md["T"], atol=1e-13)
    walls = md.at("T", z=[-0.5, 0.5])
    assert np.abs(walls).max() < 1e-12                    # T = 0 on both walls
    assert md.at("u", z=[0.0]).shape == (2, md["u"].shape[1], 1)
    with pytest.raises(OptionError, match="outside"):
        md.at("T", z=[0.7])
