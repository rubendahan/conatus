"""Findings 66 (silent int() truncations), 67 (ensemble_fate overshoots the
horizon) and 68 (copied accessors expose mutable state)."""
import numpy as np
import pytest

from conatus.ensembles import Ensemble, ensemble_fate
from conatus.errors import OptionError
from conatus.timestep import IVPSolver
from ivp_cases import ks_1d


def _ens(B=3):
    plan, _ = ks_1d(N=16, L=22.0)
    return IVPSolver(plan, "RK443", 0.01), plan


def test_fate_stops_at_the_horizon():
    s, _ = _ens()
    e = Ensemble(s, 3, rhs="batched")
    e.set_states(np.zeros((3, s.system.n)))
    fate = ensemble_fate(e, lambda ens: np.zeros(ens.B, int), check_every=4, min_steps=0,
                         max_steps=10, classifier="never decides")
    assert e.iteration == 10 and fate.settings["steps_run"] == 10


@pytest.mark.parametrize("kw", [dict(check_every=2.5), dict(max_steps=9.7), dict(min_steps=1.5)])
def test_fate_counts_are_integers(kw):
    s, _ = _ens()
    e = Ensemble(s, 2, rhs="batched")
    args = dict(check_every=2, min_steps=0, max_steps=10, classifier="c") | kw
    with pytest.raises(OptionError, match="integer"):
        ensemble_fate(e, lambda ens: np.zeros(ens.B, int), **args)


def test_fate_labels_are_not_truncated():
    s, _ = _ens()
    e = Ensemble(s, 2, rhs="batched")
    e.set_states(np.zeros((2, s.system.n)))
    with pytest.raises(OptionError, match="integer labels"):
        ensemble_fate(e, lambda ens: np.full(ens.B, 1.7), check_every=2, min_steps=0,
                      max_steps=4, classifier="c")


def test_member_of_indices_not_truncated():
    s, _ = _ens()
    with pytest.raises(OptionError, match="integer indices"):
        Ensemble.from_solvers([s, s], rhs="batched", member_of=[0, 1.7])
    with pytest.raises(OptionError, match="integer indices"):
        Ensemble.from_solvers([s, s], rhs="batched", member_of=[0, 2])


def test_states_is_a_copy():
    s, _ = _ens()
    e = Ensemble(s, 2, rhs="batched")
    st = e.states()
    st[...] = 7.0
    assert not np.any(e.X == 7.0)
