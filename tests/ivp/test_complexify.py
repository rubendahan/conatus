"""RealFourier (cos, -sin) pairs solved as complex half-size systems.

The real path is the oracle; the complexified solve must agree to rounding, halve the group
size and narrow the band, and refuse any group whose structure is not [[A, -B], [B, A]]."""
import numpy as np
import pytest

from conatus.timestep.system import GlobalSystem
from ivp_cases import rayleigh_benard_2d, rayleigh_benard_2d_psi_theta


@pytest.mark.parametrize("case", [rayleigh_benard_2d_psi_theta, rayleigh_benard_2d])
def test_complexified_solve_matches_real_path(case):
    plan, _ = case(16, 16, 1e5)
    real, cplx = GlobalSystem(plan), GlobalSystem(plan, complexify=True)
    b = np.random.default_rng(1).standard_normal(real.n)
    for a, c in [(1.0, 1e-2), (50.0, 1.0)]:
        xr, xc = real.solve(a, c, b), cplx.solve(a, c, b)
        assert np.abs(xr - xc).max() <= 1e-13 * np.abs(xr).max()
    rb = {lab: (n, kl + ku) for lab, n, _, kl, ku in real.bands()}
    for lab, n, _, kl, ku in cplx.bands():
        if lab.endswith("(complex)"):
            n_real, band_real = rb[lab[: -len(" (complex)")]]
            assert 2 * n == n_real and kl + ku < band_real
    assert "complexify=True" in cplx.explain()


def _pair_view(A, B, A2):
    """Call the structure check on a synthetic 2m x 2m (cos, -sin) group."""
    from types import SimpleNamespace
    import scipy.sparse as sp
    m = A.shape[0]
    L = sp.csr_matrix(np.block([[A, -B], [B, A2]]))
    gm = SimpleNamespace(L=L, M=sp.identity(2 * m, format="csr"), label="synthetic",
                         cols=[(0, 0, "field")] * (2 * m))
    meta = [(0, 0, n) for n in range(m)] * 2
    pos = [0] * m + [1] * m
    sysm = object.__new__(GlobalSystem)
    sysm.ordering = "interleaved"
    return GlobalSystem._complex_view(sysm, gm, meta, meta, ["bulk"] * (2 * m), pos, pos)


def test_complexify_accepts_pair_structure():
    rng = np.random.default_rng(2)
    A, B = rng.standard_normal((2, 5, 5))
    cz = _pair_view(A, B, A)
    assert cz.size == 5 and np.allclose(cz.L.toarray(), A + 1j * B)


def test_complexify_refuses_broken_structure():
    from conatus.discretize.plan import RankError
    rng = np.random.default_rng(3)
    A, B = rng.standard_normal((2, 5, 5))
    A2 = A.copy()
    A2[0, 0] += 1.0
    with pytest.raises(RankError, match="complexify refused"):
        _pair_view(A, B, A2)


def test_complexified_trajectory_matches_real_path():
    """20 RK443 steps of psi-theta RB: physical fields agree to rounding. (The tau variables
    of the 4th-order psi equation amplify rounding by ~1e7 in either path, so they are not
    compared; see)"""
    from conatus.timestep import IVPSolver
    plan, f = rayleigh_benard_2d_psi_theta(32, 24, 1e5)
    out = []
    # the third run: the real path from an initial state perturbed by 1e-15 (relative), whose move
    # is the rounding level of these 20 steps; the complexified path must agree within 10x of it
    # ( rule, as test_m0_projection; the former fixed 1e-11 was met at 5.6e-13, margin 2.8,
    # run)
    for cx, noise in ((False, 0.0), (True, 0.0), (False, 1e-15)):
        s = IVPSolver(plan, "RK443", 1e-4, complexify=cx)
        s.set_state({f["theta"]: lambda x, z: 1e-2 * np.sin(np.pi * z) * np.cos(np.pi * x)
                     + 1e-3 * np.sin(2 * np.pi * z) * np.sin(3 * np.pi * x)})
        s.X = s.X * (1 + noise * np.random.default_rng(0).standard_normal(s.X.size))
        s.run(20)
        out.append([s.coeffs("psi"), s.coeffs("theta")])
    for a, b, p in zip(*out, strict=True):
        d_rnd = np.abs(a - p).max() / np.abs(a).max()
        assert np.abs(a - b).max() / np.abs(a).max() <= 10 * max(d_rnd, 1e-15), d_rnd
        assert d_rnd < 1e-9                   # sensitivity: a 1e-6 error of a path shows above it


def test_complexify_in_symmetry_subspace_is_declared_noop():
    plan, _ = rayleigh_benard_2d(16, 16, 1e5, symmetric=True)
    real, cplx = GlobalSystem(plan), GlobalSystem(plan, complexify=True)
    assert not cplx._cbuckets and "nothing to complexify" in cplx.explain()
    b = np.random.default_rng(4).standard_normal(real.n)
    assert np.array_equal(real.solve(1.0, 1e-2, b), cplx.solve(1.0, 1e-2, b))
