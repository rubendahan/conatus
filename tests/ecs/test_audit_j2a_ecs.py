"""Audit–R7: translation mode, frame speed, truncation and thresholds in ecs/.

Written before the fixes; each failed on the code it guards (docs/AUDIT_J2a.md §3).
"""
import inspect
import warnings

import numpy as np
import pytest

def test_stability_dae_does_not_remove_a_mode_when_the_state_is_translation_invariant():
    """R3: on the laminar Kolmogorov state (∂x U = 0) the 'translation mode' was argmax(NaN) = the
    most unstable eigenvalue, removed from the count. Now: nothing removed, and said so."""
    from conatus.ecs.dae import stability_dae
    from conatus.ecs.kolmogorov import kolmogorov_solver, laminar_state, linear_stability_dense
    from conatus.ecs.solutions import TravellingWave
    Re = 12.0                                                   # above Re_lin = 9.9669 (CK13)
    s = kolmogorov_solver(Re=Re, n=4, N=12, dealias=1.5, scheme="RK443", timestep=0.01)
    X = laminar_state(s, Re=Re, n=4)
    ev = linear_stability_dense(s, X, dense_max=2000, finite_max=1e8)
    n_true = int(np.sum(ev.real > 0))
    assert n_true >= 1
    sol = TravellingWave("equilibrium", X, {"c": 0.0, "coord": "x"}, 0.0, None, float("nan"))
    st = stability_dae(s, sol, dense_max=2000, finite_max=1e8)
    assert st["n_unstable"] == n_true, st
    assert st["translation"].startswith("not excluded") and st["marginal"] is None
    assert st["translation_kernel_tol"] > 0 and st["translation_overlap_min"] > 0

def _ks(N=16):
    from conatus.ecs.ks import ks_solver
    return ks_solver(L=22.0, N=N, dealias=1.5, scheme="RK443", timestep=0.01)


def test_equilibrium_eigenvalues_requires_c_for_a_travelling_wave():
    """R5: a travelling wave without params['c'] was analysed in the fixed frame (c = 0)."""
    from conatus.ecs.solutions import TravellingWave, equilibrium_eigenvalues
    s = _ks()
    X = np.zeros(s.system.n)
    X[2] = 0.3
    tw = TravellingWave("travelling_wave", X, {}, 0.0, None, 0.0)
    with pytest.raises(KeyError, match="'c'"):
        equilibrium_eigenvalues(s, tw, 4, dense_max=100, arnoldi_tol=1e-12)
    eq = TravellingWave("equilibrium", X, {}, 0.0, None, 0.0)     # equilibrium: c = 0 by definition
    assert len(equilibrium_eigenvalues(s, eq, 4, dense_max=100, arnoldi_tol=1e-12)) == 4


def test_resample_state_reports_truncated_energy():
    """R6: resampling to a coarser grid dropped the high modes without reporting what was lost."""
    from conatus.ecs.kolmogorov import kolmogorov_solver, resample_state
    s16 = kolmogorov_solver(Re=12.0, n=4, N=16, dealias=1.5, scheme="RK443", timestep=0.01)
    s8 = kolmogorov_solver(Re=12.0, n=4, N=8, dealias=1.5, scheme="RK443", timestep=0.01)
    a = np.zeros((16, 16))
    a[2, 0], a[12, 2] = 1.0, 0.5                                 # kx = 1 kept, kx = 6 dropped
    X = s16.system.gather_state([a, np.zeros((1, 1))])
    rep: dict = {}
    with pytest.warns(UserWarning, match="truncat"):
        Y = resample_state(X, s16, s8, report=rep)
    assert rep["kept"] == 8 and abs(rep["truncated_max"] - 0.5) < 1e-15
    assert abs(rep["energy_lost_rel"] - 0.25 / 1.25) < 1e-15
    b, _ = s8.system.scatter(Y)
    assert b[2, 0] == 1.0
    rep2: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")                           # padding loses nothing: no warning
        resample_state(Y, s8, s16, report=rep2)
    assert rep2["energy_lost_rel"] == 0.0


def test_ecs_thresholds_are_declared():
    """R7: 0.99 (marginal overlaps of Floquet vectors), 1e-10 (reflection symmetry test, 'c ≈ 0')
    were literals. They are now keyword arguments, reported in the outputs."""
    from conatus.ecs import dae, solutions
    assert inspect.signature(solutions.floquet_multipliers).parameters["marginal_overlap_min"].default == 0.99
    assert inspect.signature(solutions.symmetry_breaking_check).parameters["symmetry_tol"].default == 1e-10
    assert inspect.signature(dae.find_relative_equilibrium_dae).parameters["c_zero_tol"].default == 1e-10
    # symmetry_breaking_check: the threshold is applied and reported
    s = _ks()
    from conatus.ecs.fourier_ops import FourierOps
    ops = FourierOps(s)
    X = np.zeros(s.system.n)
    X[ops.odd_mask(X)] = 0.1                                    # reflection-symmetric (sin only)
    sol = solutions.TravellingWave("equilibrium", X, {}, 0.0, None, 0.0)
    out = solutions.symmetry_breaking_check(s, sol, dense_max=100, unstable_tol=1e-8, coupling_tol=1e-8)
    assert out["symmetry_tol"] == 1e-10
    Y = X.copy()
    Y[~ops.odd_mask(X) & (np.arange(X.size) != ops.mean_index)] = 1e-10   # asymmetric at 1e-9 rel
    with pytest.raises(ValueError, match="symmetry_tol"):
        solutions.symmetry_breaking_check(s, solutions.TravellingWave("equilibrium", Y, {}, 0.0, None, 0.0),
                                          dense_max=100, unstable_tol=1e-8, coupling_tol=1e-8)
    out = solutions.symmetry_breaking_check(s, solutions.TravellingWave("equilibrium", Y, {}, 0.0, None, 0.0),
                                            dense_max=100, unstable_tol=1e-8, coupling_tol=1e-6,
                                            symmetry_tol=1e-8)
    assert out["symmetry_tol"] == 1e-8
