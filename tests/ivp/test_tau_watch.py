"""TauWatch: tau sizes recorded along a run, warning past an explicit growth limit."""
import warnings

import numpy as np
import pytest

from ivp_cases import rayleigh_benard_2d

from conatus.timestep import IVPSolver, TauWatch


def test_tau_watch_records_and_warns():
    plan, f = rayleigh_benard_2d(16, 8, Ra=1e5, Lx=2.0)
    s = IVPSolver(plan, "RK222", 1e-3)
    s.set_state({f["T"]: lambda x, z: 1e-2 * np.cos(np.pi * z) * np.cos(np.pi * x)})
    w = TauWatch(s, growth_limit=1e6)
    assert set(w.names) == {"tT1", "tT2", "tu1", "tu2"}      # tp is a gauge, not a lifted tau
    s.run(3, callback=w, every=1)
    assert all(len(h) == 3 for h in w.history.values())
    assert "growth limit 1e+06" in w.explain()
    # an artificial jump of one tau must be flagged, once, by name
    k = [v.name for v in s.system.variables].index("tT1")
    c = s.system.scatter(s.X)
    c[k] = c[k] * 0 + 1e12
    s.X = s.system.gather_state(c, check_subspace=False)
    with pytest.warns(RuntimeWarning, match="tau 'tT1' grew"):
        w()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        w()                      # already flagged: no second warning
