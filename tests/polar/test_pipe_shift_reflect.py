"""Shift-reflect subspace of the pipe and the constant-flux pipe of the travelling waves.

``ShiftReflect(phi, z)``: (u_phi, u_r, u_z)(phi, r, z) -> (-u_phi, u_r, u_z)(-phi, r, z + Lz/2), the
symmetry S of the pipe travelling waves (Pringle, Duguet & Kerswell 2009; Willis, Cvitanovic &
Avila 2013). Real-linear on the stored coefficients (m >= 0): enforced in the IVP step by a declared
projection, J read off the transforms. References: the grid values of a projected state satisfy the
mirror relation pointwise; the run in the subspace equals the run without the projection from the
same symmetric state (equivariant operators: the projection only removes rounding); the laminar
Hagen-Poiseuille flow is a fixed point of the constant-flux step with G = 4/Re and E = pi Lz / 6.
"""
import numpy as np
import pytest

import conatus.expert as C
from conatus.ecs.pipe import PipeTravellingWave, pipe_flux_problem
from conatus.errors import SymmetryError

KW = dict(Nphi=8, Nr=8, Nz=8, alpha=1.25, Re=200.0, dealias=1.5)


def _state(solver, seed=3, amp=0.05):
    """Laminar flow + a random low-mode perturbation made admissible by one solve (divergence-free,
    no slip), then projected by the solver's declared projections."""
    rng = np.random.default_rng(seed)
    n = solver.system.n
    X = 1e-3 * amp * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    solver.X = solver.symmetrize_state(X)
    solver._reset_history()
    solver.run(1)                                  # admissible state (div u = 0, u = 0 on the wall)
    return solver.X.copy()


def _laminar(solver, case):
    u = case.u
    solver.set_state({u: lambda P, R, Z: np.array(np.broadcast_arrays(0 * P, 0 * R, 1 - R * R + 0 * Z))})
    return solver.X.copy()


def test_projection_inferred_and_grid_mirror():
    case = pipe_flux_problem(m0=2, shift_reflect=True, **KW)
    s = C.IVPSolver(case.plan, "SBDF2", 0.01)
    ex = s.explain(full=True)
    assert "ShiftReflect" in ex and "J antilinear" in ex
    pr = s._m0proj.s
    assert pr.err < 1e-12
    X = s.symmetrize_state(np.random.default_rng(0).standard_normal(s.system.n) + 0j)
    assert pr.distance(X) < 1e-14
    # pointwise: u(-phi, r, z + L/2) = (-u_phi, u_r, u_z)(phi, r, z) on the grid
    s.X = X
    g = s.grid("u")                                 # (3, Mphi, Mr, Mz), non-dealiased grid
    Mp, Mz = g.shape[1], g.shape[3]
    gm = g[:, (-np.arange(Mp)) % Mp][..., (np.arange(Mz) + Mz // 2) % Mz]
    gm[0] *= -1
    assert np.abs(gm - g).max() < 1e-12 * np.abs(g).max()


def test_trajectory_matches_unprojected_run():
    """20 SBDF2 steps from the same symmetric state with and without the shift-reflect projection
    (both in the 2-fold subspace): same states to rounding; leak at rounding."""
    case_s = pipe_flux_problem(m0=2, shift_reflect=True, **KW)
    case_f = pipe_flux_problem(m0=2, shift_reflect=False, **KW)
    ss = C.IVPSolver(case_s.plan, "SBDF2", 0.01)
    sf = C.IVPSolver(case_f.plan, "SBDF2", 0.01)
    X0 = _state(ss)
    assert ss.system.n == sf.system.n               # same reduced layout (S is projected only)
    for s in (ss, sf):
        s.X = X0.copy()
        s._reset_history()
        s.run(20)
    scale = np.abs(sf.X).max()
    assert np.abs(ss.X - sf.X).max() < 1e-11 * scale
    assert ss._m0proj.s.distance(sf.X) < 1e-11
    assert ss.symmetry_leak() < 1e-12


def test_constant_flux_laminar_fixed_point():
    case = pipe_flux_problem(m0=2, shift_reflect=True, **KW)
    s = C.IVPSolver(case.plan, "SBDF2", 0.01)
    X = _laminar(s, case)
    tw = PipeTravellingWave(case, s, T=0.05, fd_eps=1e-7, method="shooting")
    m0 = tw.measures(X)
    assert abs(m0["E_ratio"] - 1) < 1e-12
    Y = tw.advance(X, 5)
    m = tw.measures(Y)
    assert abs(m["G_ratio"] - 1) < 1e-10 and abs(m["E_ratio"] - 1) < 1e-12
    assert np.abs(Y[tw.iu] - X[tw.iu]).max() < 1e-10  # u unchanged (p, G entries filled by the solve)
    assert tw.comoving_residual(Y, 0.3) < 1e-10        # any c: dz of an axisymmetric k = 0 field is 0


def test_asymmetric_data_refused_and_evp_refused():
    case = pipe_flux_problem(m0=2, shift_reflect=True, **KW)
    s = C.IVPSolver(case.plan, "SBDF2", 0.01)
    u = case.u

    def bad(P, R, Z):           # u_z with an odd-in-phi, z-independent part: breaks S
        return np.array(np.broadcast_arrays(0 * P, 0 * R, (1 - R * R) * (1 + 0.1 * R * R * np.sin(2 * P))))
    with pytest.raises(SymmetryError, match="outside"):
        s.set_state({u: bad})
    cc = C.CylindricalCoordinates("phi", "r", "z")
    phi, r, z = cc["phi"], cc["r"], cc["z"]
    br = C.DiskBasis(r, 8, 1.0, dealias=1)
    D = C.Domain(cc, [C.ComplexFourier(phi, 6, (0, 2 * np.pi), dealias=1), br,
                      C.ComplexFourier(z, 4, (0, 2 * np.pi), dealias=1)])
    v = C.Field("v", D, dtype=complex)
    t = C.Field("t", D.without(r), dtype=complex)
    lam = C.Parameter("lam", dtype=complex)
    S = C.ShiftReflect(phi, z)
    pb = C.EVP([v, t], eigenvalue=lam, symmetry=C.SymmetrySubspace(S, v=C.EVEN, t=C.EVEN))
    pb.add_equation((lam * v + C.lap(v) + C.lift(t, br, -1, lam=2), 0))
    pb.add_equation((C.interp(v, r, "right"), 0))
    from conatus.discretize import discretize
    with pytest.raises(NotImplementedError, match="real-linear"):
        plan = discretize(pb)
        plan.matrices(plan.groups[0])


def test_shift_reflect_declaration_checks():
    cc = C.CylindricalCoordinates("phi", "r", "z")
    S = C.ShiftReflect(cc["phi"], cc["z"])
    assert S.det == -1 and S.flips(cc["phi"]) and not S.flips(cc["z"])
    assert "z + L/2" in repr(S)
    # allowed together with the m-fold rotation of the same azimuth (dihedral group)
    G = C.SymmetrySubspace(C.AzimuthalMFold(cc["phi"], 2), S)
    assert G.order == 4
    with pytest.raises(NotImplementedError):
        C.ShiftReflect(cc["r"], cc["z"])


def test_projection_inferred_at_a_larger_resolution():
    """At (32, 16, 16) four unit factors failed their check when read as one quotient
    J z / conj(z) (a small probe entry amplified the transforms' rounding); fitted over two probes."""
    case = pipe_flux_problem(Nphi=32, Nr=16, Nz=16, alpha=1.25, Re=2400.0, m0=2, shift_reflect=True, dealias=1.5)
    s = C.IVPSolver(case.plan, "SBDF2", 0.02)
    pr = s._m0proj.s
    assert pr.err < 1e-12 and np.all(np.abs(np.abs(pr.sigma) - 1) < 1e-12)


def test_a_step_that_breaks_the_symmetry_is_refused_by_run():
    """Review j5j6: the projection used to remove a non-equivariant part of the step in silence
    (the leak was only checked when ``symmetry_leak()`` was called). A forcing that breaks S
    (added to F here) is now refused at the end of ``run``."""
    case = pipe_flux_problem(m0=2, shift_reflect=True, **KW)
    s = C.IVPSolver(case.plan, "SBDF2", 0.01)
    X0 = _state(s)
    s.symmetry_leak(reset=True)
    rng = np.random.default_rng(7)
    bad = 1e-3 * (rng.standard_normal(s.system.n) + 1j * rng.standard_normal(s.system.n))
    base_F = s.F
    s.F = lambda X, t: base_F(X, t) + bad
    s.X = X0.copy()
    s._reset_history()
    with pytest.raises(SymmetryError, match="left the subspace"):
        s.run(2)



def test_projections_stay_on_at_32_radial_modes():
    """Review j5j6: from N_r = 32 on, the radial transforms round at ~1e-12; the m = 0
    projection used to switch itself off there (a per-entry relative test took an m > 0 entry with a
    small random value for a projected one, and the unit factors were held to a fixed 1e-12), and
    the shift-reflect projection, which needs it, refused the (64, 32, 32) pipe. The bound is now
    max(1e-12, 100 x the round-trip rounding (90th percentile)), declared."""
    import time
    t0 = time.perf_counter()
    for sr in (False, True):
        case = pipe_flux_problem(Nphi=16, Nr=32, Nz=8, alpha=1.25, Re=2400.0, m0=2, shift_reflect=sr,
                                 dealias=1.5)
        s = C.IVPSolver(case.plan, "SBDF2", 0.01)
        pr = s._m0proj.m0 if sr else s._m0proj
        assert getattr(pr, "active", False), s._m0proj.explain()
        assert "round-trip rounding" in pr.explain()
        if sr:
            assert s._m0proj.s.err <= s._m0proj.s.tol
    print(f"built in {time.perf_counter() - t0:.1f} s")


def test_eigenvalues_of_the_laminar_flow_in_the_subspace():
    """Arnoldi on the comoving map from arbitrary ARPACK vectors: restricted to the subspace (the
    step's leak check would refuse vectors outside it). Laminar flow, c = 0: every
    eigenvalue is a decay rate (real part < 0), none flagged neutral by the translation test."""
    case = pipe_flux_problem(m0=2, shift_reflect=True, **KW)
    s = C.IVPSolver(case.plan, "SBDF2", 0.01)
    X = s.symmetrize_state(_laminar(s, case))
    tw = PipeTravellingWave(case, s, T=0.05, fd_eps=1e-7, method="shooting")
    X = tw.advance(X, 1)
    ev = tw.eigenvalues(X, 0.0, k=2, T=0.05, tol=1e-8, fd_eps=1e-7)
    assert np.all(ev["lambda"].real < 0), ev["lambda"]
    assert s.symmetry_leak() < 1e-12


def test_neutral_translation_mode_one_at_most_and_degenerate_cluster():
    """Review 2 of j5j6: the neutral translation mode is identified once (best-aligned eigenspace),
    not by a per-eigenvector threshold (stable non-normal eigenvectors of the wave reach |cos| =
    0.96-0.99), and a cluster degenerate with mu = 1 that splits d_z x is still recognised."""
    from conatus.ecs.pipe import _neutral_mode
    rng = np.random.default_rng(1)
    n = 40
    d = rng.standard_normal(n)
    d /= np.linalg.norm(d)
    w = rng.standard_normal((n, 4))
    w -= np.outer(d, d @ w)                           # directions orthogonal to d
    w /= np.linalg.norm(w, axis=0)
    T = 10.0
    # (1) unstable mode at |cos| 0.95 with d (would be dropped by a 0.9 threshold), neutral mode exact
    lam = np.array([0.1, 1e-6, -0.01])
    vec = np.stack([0.95 * d + np.sqrt(1 - 0.95**2) * w[:, 0], d, 0.98 * d + 0.2 * w[:, 1]], axis=1)
    ni, nu, nm, notes = _neutral_mode(lam, np.exp(lam * T), vec, d, 1e-8)
    assert (ni, nu, nm, notes) == (1, 1, 0, [])
    # (2) degenerate cluster at mu = 1: d split between two vectors (|cos| 0.71 each)
    lam = np.array([0.1, 0.0, 0.0, -0.05])
    vec = np.stack([w[:, 0], (d + w[:, 1]) / np.sqrt(2), (d - w[:, 1]) / np.sqrt(2), w[:, 2]], axis=1)
    ni, nu, nm, notes = _neutral_mode(lam, np.exp(lam * T), vec, d, 1e-8)
    assert ni in (1, 2) and nu == 1 and nm == 1 and "degenerate" in notes[0]
    # (3) no eigenvector near d_z x: nothing excluded, said so
    lam = np.array([0.1, 1e-3])
    ni, nu, nm, notes = _neutral_mode(lam, np.exp(lam * T), w[:, :2], d, 1e-8)
    assert ni is None and nu == 2 and "not identified" in notes[0]
