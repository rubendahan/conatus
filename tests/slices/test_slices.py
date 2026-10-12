"""First Fourier mode slice: reduction / reconstruction, slice border, invariant solutions in
the slice (KS L = 22), and relative periodic orbits of the two-mode model of Budanur,
Borrero-Echeverry & Cvitanović (arXiv:1411.3303, eqs. 40-41 and Table I, read in the PDF)
found as periodic orbits of the reduced flow. Notes: §3."""
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.optimize as so

from conatus.bifurcation import DynamicalSystem, ir_system
from conatus.ecs import FourierOps, NewtonKrylovHookstep, find_equilibrium, ks_solver
from conatus.ecs.solutions import PreperiodicOrbit
from conatus.errors import SymmetryError
from conatus.manifolds import ODEIntegrator, PoincareSection, Trajectory, poincare_crossings
from conatus.slices import (FirstFourierModeSlice, RotationOps, find_rpo_in_slice,
                            slice_equilibrium_check, slice_orbit_check, two_mode_callables,
                            two_mode_problem)

FIXTURE = Path(__file__).resolve().parents[1] / "ecs/data/ks_l22_ppo10.json"


@pytest.fixture(scope="module")
def ks():
    return ks_solver(L=22.0, N=32, dealias=1.5, scheme="RK443", timestep=0.01)


def newton():
    return NewtonKrylovHookstep(tol=1e-12, max_iterations=40, gmres_rtol=1e-3, gmres_max_iterations=60,
                                delta0=1.0, delta_min=1e-10, delta_max=10.0)


def test_reduce_reconstruct_and_invariance(ks):
    """X̂ satisfies the slice condition, reconstruct inverts reduce, and the reduced state does
    not depend on where the group orbit was entered (reduce(τ_ℓ X) = reduce(X))."""
    slc = FirstFourierModeSlice(FourierOps(ks), border_tol=1e-8)
    rng = np.random.default_rng(0)
    c = np.zeros(32)
    c[2:14] = rng.normal(size=12)
    X = ks.system.gather_state([c])
    Xh, ell = slc.reduce(X)
    assert abs(slc.slice_condition(Xh)) < 1e-15
    assert Xh @ slc.p > 0
    assert np.allclose(slc.reconstruct(Xh, ell), X, atol=1e-13)
    for shift in (0.7, 5.3, -9.1):
        Yh, _ = slc.reduce(slc.ops.translate(X, shift))
        assert np.allclose(Yh, Xh, atol=1e-13)


def test_border_is_detected_and_refused(ks):
    """A state without first Fourier mode is on the slice border: flagged, reduce refuses."""
    slc = FirstFourierModeSlice(FourierOps(ks), border_tol=1e-8)
    c = np.zeros(32)
    c[5] = -1.0                                     # sin(2 k1 x) only
    X = ks.system.gather_state([c])
    assert slc.on_border(X)
    with pytest.raises(SymmetryError):
        slc.reduce(X)
    with pytest.raises(SymmetryError):              # template with two wavenumbers: refused
        c2 = np.zeros(32)
        c2[2], c2[4] = 1.0, 1.0
        FirstFourierModeSlice(FourierOps(ks), border_tol=1e-8, template=ks.system.gather_state([c2]))


def test_ks_equilibria_in_the_slice(ks):
    """E1 is a fixed point in the slice (ℓ̇ = 0); E2 and E3 (periods L/2, L/3) lie on the border."""
    slc = FirstFourierModeSlice(FourierOps(ks), border_tol=1e-6)
    for m, a, border in ((1, 1.0, False), (2, 1.0, True), (3, 2.0, True)):
        c = np.zeros(32)
        c[2 * m + 1] = -a
        sol = find_equilibrium(ks, ks.system.gather_state([c]), newton())
        chk = slice_equilibrium_check(slc, ks, sol)
        assert chk["on_border"] is border
        if not border:
            assert chk["in_slice_speed"] < 1e-12 and chk["ell_dot_error"] < 1e-12


def test_ks_ppo_closes_in_the_slice():
    """PPO T = 10.25 (fixture): periodic of period 2T in the slice, net shift 0 mod L; the phase
    reconstructed by quadrature of ℓ̇ agrees with the projected one (trapezoid error)."""
    fx = json.load(open(FIXTURE))
    s = ks_solver(L=22.0, N=32, dealias=1.5, scheme="RK443", timestep=fx["T"] / fx["nsteps"])
    X = s.system.gather_state([np.array(fx["coeffs"])])
    ppo = PreperiodicOrbit("preperiodic_orbit", X, {"T": fx["T"], "ell": fx["ell"],
                                                     "nsteps": fx["nsteps"]}, fx["residual"], None, 0.0)
    slc = FirstFourierModeSlice(FourierOps(s), border_tol=1e-6)
    chk = slice_orbit_check(slc, s, ppo)
    assert chk.closure < 1e-11 and chk.shift_error < 1e-11
    assert chk.min_border_distance > 0.05 and chk.max_phase_jump < 0.1
    quad = [n for n in chk.notes if "trapezoid" in n][0]
    assert float(quad.split("max difference ")[1]) < 1e-3


def _two_mode():
    f, jac = two_mode_callables()
    return DynamicalSystem.from_callables(f, 4, {}, jac=jac, fd_x=1e-5, fd_p=1e-6, name="two-mode")


def test_two_mode_callables_match_ir():
    ds, dsi = _two_mode(), ir_system(two_mode_problem(), ["mu1"])
    x = np.array([0.3, 0.2, -0.1, 0.4])
    assert np.abs(ds.G(x) - dsi.G(x)).max() < 1e-14
    assert np.abs(ds.J(x) - dsi.J(x)).max() < 1e-14


def _tw(ds, slc):
    def vh(x3):
        return slc.velocity(np.array([x3[0], 0, x3[1], x3[2]]), ds.G)[0][[0, 2, 3]]
    tw = so.fsolve(vh, [0.44, -0.39, 0.07], xtol=1e-14)
    return np.array([tw[0], 0, tw[1], tw[2]])


def test_two_mode_relative_equilibrium_in_slice():
    """arXiv:1411.3303, eq. (40): TW in the slice (0.439966, 0, -0.386267, 0.070204);
    eq. (41): λ = 0.05073 ± i2.4527, -5.5055, 0 (co-moving Jacobian)."""
    ds, ops = _two_mode(), RotationOps([1, 2])
    slc = FirstFourierModeSlice(ops, border_tol=1e-8)
    x = _tw(ds, slc)
    assert np.allclose(x, [0.439966, 0, -0.386267, 0.070204], atol=6e-7)
    c = slc.velocity(x, ds.G)[1]
    T = np.array([ops.ddx(e) for e in np.eye(4)]).T
    w = np.sort_complex(np.linalg.eigvals(ds.J(x) - c * T))
    ref = np.sort_complex(np.array([0.05073 + 2.4527j, 0.05073 - 2.4527j, -5.5055, 0]))
    assert np.allclose(w, ref, atol=6e-5)


@pytest.mark.slow                   # 7.9 s in the fast suite (qual_durations0)
def test_two_mode_rpos_of_table_I():
    """RPOs '1' and '01' of Table I (T = 3.64151221, θ = 0.08096967, Λ = -1.48372354;
    T = 7.34594158, θ = -2.94647181, Λ = -2.00054831) as periodic orbits of the reduced flow
    through a Poincaré section of the slice. Measured agreement (job man_twomode): T to 4e-7
    (the table's 9th digit is beyond the agreement of two tolerances), θ to 1e-8."""
    ds, ops = _two_mode(), RotationOps([1, 2])
    slc = FirstFourierModeSlice(ops, border_tol=1e-8)
    x = _tw(ds, slc)
    I = ODEIntegrator(ds, method="DOP853", rtol=1e-12, atol=1e-13)
    T = np.array([ops.ddx(e) for e in np.eye(4)]).T
    c = slc.velocity(x, ds.G)[1]
    w, V = np.linalg.eig(ds.J(x) - c * T)
    vu = V[:, np.argmax(w.real)]
    tr = I.integrate(x + 1e-3 * np.real(vu), 250.0)
    st = slc.reduce_trajectory(tr.t, tr.X)
    trh = Trajectory(tr.t, st.X_hat, "reduced DOP853 dense output",
                     lambda i, tt: slc.reduce(tr._dense(i, tt))[0])
    a = np.imag(vu / np.linalg.norm(vu))
    a[1] = 0.0
    nrm = np.cross(a[[0, 2, 3]], [0, 0, 1.0])
    sec = PoincareSection.hyperplane([nrm[0], 0, nrm[1], nrm[2]], x, direction=-1)
    cr = poincare_crossings(trh, sec, xtol=1e-13, t_min=100.0)
    assert len(cr.t) > 10 and cr.g_residual.max() < 1e-12
    for p, (Tr, thr, Lr) in ((1, (3.64151221, 0.08096967, -1.48372354)),
                             (2, (7.34594158, -2.94647181, -2.00054831))):
        dist = [np.linalg.norm(cr.X[k + p] - cr.X[k]) for k in range(len(cr.t) - p)]
        found = None
        for k in np.argsort(dist)[:6]:
            try:
                o = find_rpo_in_slice(slc, I, sec, cr.X[k], cr.t[k + p] - cr.t[k], tol=1e-11,
                                      fd_step=1e-6, max_iterations=10, steady_tol=1e-6)
            except Exception:
                continue
            if abs(o.T - Tr) < 1e-4:
                found = o
                break
        assert found is not None, f"orbit of period {Tr} not found"
        assert abs(found.T - Tr) < 1e-6 and abs(found.theta - thr) < 1e-7
        assert abs(found.multipliers[0].real - Lr) < 2e-5 * abs(Lr)
        assert np.min(np.abs(found.multipliers - 1)) < 1e-4       # time direction
        assert "RPO in the slice" in found.explain()


def test_relative_equilibrium_is_not_returned_as_an_orbit():
    """The TW is a fixed point of the reduced flow (a 'periodic orbit' of any period): refused
    by the declared steady_tol (the example converged onto it before the guard)."""
    from conatus.errors import ConvergenceError
    ds, ops = _two_mode(), RotationOps([1, 2])
    slc = FirstFourierModeSlice(ops, border_tol=1e-8)
    x = _tw(ds, slc)
    I = ODEIntegrator(ds, method="DOP853", rtol=1e-12, atol=1e-13)
    sec = PoincareSection.hyperplane([0, 0, 1.0, 0], x, direction=0)
    with pytest.raises(ConvergenceError, match="relative equilibrium"):
        find_rpo_in_slice(slc, I, sec, x + 1e-9, 2.5, tol=1e-11, fd_step=1e-6, max_iterations=5,
                          steady_tol=1e-6)
