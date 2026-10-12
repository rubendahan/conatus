"""The kernel parity test under a declared symmetry uses its declared tolerance, and only it.

``np.allclose(S v, -v, atol=PARITY_TOL * max|v|)`` kept NumPy's default
``rtol = 1e-5``, so normal_form (PARITY_TOL = 1e-8) and branch_switch (PARITY_TOL = 1e-7) accepted
a kernel 1e-6 away from either parity in every entry; the S-invariance test of the located point
(INVARIANT_TOL = 1e-9) likewise. ``parity_under`` applies the absolute bound alone."""
import importlib

import numpy as np

from conatus.bifurcation.normal_form import parity_under

# the modules (conatus.bifurcation re-exports the function normal_form under the module's name)
normal_form = importlib.import_module("conatus.bifurcation.normal_form")
branching = importlib.import_module("conatus.bifurcation.branching")


def _vector(n=12, seed=0):
    v = np.random.default_rng(seed).uniform(0.5, 1.0, n)        # every entry of order max|v|
    return v * np.where(np.arange(n) % 2, 1.0, -1.0)


def test_parity_is_decided_by_the_declared_tolerance_alone():
    v = _vector()
    for tol in (normal_form.PARITY_TOL, branching.PARITY_TOL):
        off = -v * (1 + 1e-6)                                    # 1e-6 relative in every entry
        # the defect of the former test: rtol = 1e-5 hid a 1e-6 deviation from S v = -v
        assert np.allclose(off, -v, atol=tol * np.abs(v).max())
        assert parity_under(off, v, tol) is None
        assert parity_under(-v * (1 + 0.1 * tol), v, tol) == -1  # within the declared bound
        assert parity_under(v * (1 - 0.1 * tol), v, tol) == +1
        assert parity_under(v * (1 + 10 * tol), v, tol) is None


def test_call_sites_use_the_absolute_helper():
    """normal_form and branch_switch decide the parity through parity_under (no np.allclose
    with an implicit rtol left at either site)."""
    import inspect
    for mod in (normal_form, branching):
        src = inspect.getsource(mod)
        assert "parity_under(" in src
        for line in src.splitlines():
            if "np.allclose(" in line and "PARITY_TOL" in line:
                raise AssertionError(f"{mod.__name__}: {line.strip()}")
            if "np.allclose(" in line and "INVARIANT_TOL" in line:
                assert "rtol=0.0" in line, line
