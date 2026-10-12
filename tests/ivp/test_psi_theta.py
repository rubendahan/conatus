"""Rayleigh–Bénard 2D in streamfunction–temperature form (same formulation as the hand-written
CUDA comparator): needs dt(lap(psi)), i.e. a non-trivial mass operator."""
import numpy as np
import pytest

from ivp_cases import rayleigh_benard_2d_psi_theta
from conatus.ir import diff, interp
from conatus.timestep import IVPSolver

SIGMA_EVP = 5.542505923105        # EVP, Ra = 2500, k = pi, Pr = 1 (comparator README)
NU_COMPARATOR = 4.994307          # comparator 192x64, Ra = 1e5, Lx = 2, Pr = 1


def test_psi_theta_mass_operator_not_empty():
    plan, _ = rayleigh_benard_2d_psi_theta(8, 16, Ra=2500, linear=True)
    for g in plan.groups:
        assert plan.matrices(g).M.nnz > 0
    assert all(r.ok for r in plan.check())


def test_psi_theta_linear_growth_rate_matches_evp():
    plan, f = rayleigh_benard_2d_psi_theta(8, 32, Ra=2500, Lx=2.0, linear=True)
    s = IVPSolver(plan, "RK443", 1e-3)
    s.set_state({f["theta"]: lambda X, Z: 1e-6 * np.cos(np.pi * X) * np.sin(np.pi * Z)})
    s.run(600)
    a1, t1 = np.linalg.norm(s.coeffs("theta")), s.t
    s.run(400)
    a2, t2 = np.linalg.norm(s.coeffs("theta")), s.t
    sigma = np.log(a2 / a1) / (t2 - t1)
    assert abs(sigma - SIGMA_EVP) < 1e-7, sigma


@pytest.mark.slow
def test_psi_theta_steady_nusselt_matches_comparator():
    plan, f = rayleigh_benard_2d_psi_theta(64, 32, Ra=1e5, Lx=2.0)
    s = IVPSolver(plan, "RK443", 2e-4)
    s.set_state({f["theta"]: lambda X, Z: 1e-2 * np.cos(np.pi * X) * np.sin(np.pi * Z)})
    s.run(4000)
    ev, th, z = s.evaluator, f["theta"], f["z"]
    nus = []
    for side in ("left", "right"):
        c = ev.evaluate(interp(diff(th, z), z, side), {th: s.coeffs("theta")}, dict(plan.params))
        nus.append(1 - float(np.mean(np.real(ev.to_grid(c, 0)))))
    assert abs(nus[0] - nus[1]) < 1e-9
    # 64x32 carries a ~2e-5 resolution error against the comparator's converged value
    assert abs(nus[0] - NU_COMPARATOR) < 5e-5, nus
