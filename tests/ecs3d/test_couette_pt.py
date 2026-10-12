"""Plane Couette flow in the poloidal–toroidal IR formulation: exact laminar profile,
Orr–Sommerfeld/Squire spectra, symmetry group S (equivariance of the discrete step), GHC 2009
diagnostics at laminar flow. §Couette."""
import numpy as np
import scipy.linalg as sl

from couette_cases import BOX_GHC, CouetteDiagnostics, couette_pt, shift_reflect_group
from conatus.timestep import IVPSolver


def _group_spectrum(plan, kx, kz, cut=60.0):
    labs = [plan.layout.group_label(g) for g in plan.groups]
    m = plan.matrices(plan.groups[labs.index(f"x: k={kx}, z: k={kz}")])
    w = sl.eigvals(-m.L.toarray(), m.M.toarray())
    w = w[np.isfinite(w)]
    return w[np.abs(w) < cut]


def test_laminar_couette_is_exact_and_steady():
    plan, f = couette_pt(6, 13, 6, 400.0, BOX_GHC)
    s = IVPSolver(plan, "RK443", 0.02, f_backend="fused", decouple=True)
    s.set_state({f["U"]: lambda X, Y, Z: Y + 0 * X})
    s.run(20)
    U = np.real(s.coeffs("U")).reshape(-1)
    e = np.zeros_like(U)
    e[1] = 1.0
    assert np.max(np.abs(U - e)) < 1e-13
    assert np.max(np.abs(s.coeffs("phi"))) < 1e-14 and np.max(np.abs(s.coeffs("psi"))) < 1e-14
    d = CouetteDiagnostics(s, BOX_GHC)(s.X)
    assert abs(d["E"] - 1 / 6) < 1e-13 and abs(d["D"] - 1) < 1e-12 and abs(d["I"] - 1) < 1e-11
    assert d["norm"] < 1e-13


def test_orr_sommerfeld_couette_and_exact_squire_modes():
    """Group (1, 0) at Re = 1000 linearized about U = y: least-stable OS mode
    c = ±0.605343 - 0.1192302 i (independent oracle: classical_os.py, Chebyshev
    collocation, N = 80/120/160 agree to 7 digits), least-stable Squire mode
    c = ±0.79751396 - 0.11790537 i. kx = 0: Squire modes psi ~ sin/cos(n pi (y+1)/2) are exact,
    sigma = -(kz^2 + (n pi / 2)^2) / Re."""
    Re = 1000.0
    plan, _ = couette_pt(4, 65, 4, Re, (2 * np.pi, 2 * np.pi), linear_about="couette")
    c = 1j * _group_spectrum(plan, 1, 0)
    os_mode = c[np.argmin(np.abs(np.abs(c.real) - 0.605343) + np.abs(c.imag + 0.1192302))]
    assert abs(abs(os_mode.real) - 0.605343) < 2e-6 and abs(os_mode.imag + 0.1192302) < 2e-7
    lead = c[np.argmax(c.imag)]
    assert abs(abs(lead.real) - 0.79751396) < 1e-7 and abs(lead.imag + 0.11790537) < 1e-7
    # spectrum symmetric under c -> -conj(c) (U = y is odd)
    for ci in c[c.imag > -0.5]:
        assert np.min(np.abs(c + np.conj(ci))) < 1e-8
    w = np.sort(_group_spectrum(plan, 0, 1).real)[::-1]
    exact = np.sort(-(1.0 + (np.arange(1, 8) * np.pi / 2) ** 2) / Re)[::-1]
    for e in exact[:4]:
        # 1e-11 (was 1e-12): the dense QZ of this group gives 2.5e-12 on some GitHub runner CPUs
        # and < 1e-12 on the development cluster (same code and libraries: round-off of the solver)
        assert np.min(np.abs(w - e)) < 1e-11


def test_shift_reflect_group_is_a_symmetry_of_the_discrete_step():
    rng = np.random.default_rng(4)
    plan, f = couette_pt(8, 13, 8, 400.0, BOX_GHC)
    s = IVPSolver(plan, "RK443", 0.02, f_backend="fused", decouple=True)
    G = shift_reflect_group(s)
    # smooth wall-compatible random fields (no symmetry), laminar U = y + a perturbation
    Lx, Lz = BOX_GHC
    c = rng.standard_normal((2, 3, 3, 2))

    def g(X, Z, k):
        return sum(c[k, m, n, 0] * np.cos(2 * np.pi * (m * X / Lx + n * Z / Lz))
                   + c[k, m, n, 1] * np.sin(2 * np.pi * (m * X / Lx + n * Z / Lz))
                   for m in range(3) for n in range(3))
    s.set_state({f["phi"]: lambda X, Y, Z: 0.02 * (1 - Y**2) ** 2 * g(X, Z, 0),
                 f["psi"]: lambda X, Y, Z: 0.05 * (1 - Y**2) * g(X, Z, 1) * (1 + Y),
                 f["U"]: lambda X, Y, Z: Y + 0.1 * (1 - Y**2) * (1 + Y + 0 * X),
                 f["W"]: lambda X, Y, Z: 0.1 * (1 - Y**2) * (1 + 2 * Y + 0 * X)})
    X = s.X.copy()

    def fm(X0):
        s.X, s.t = X0.copy(), 0.0
        s.run(2)
        return s.X

    ref = fm(X)
    fe = G.field_entry
    for gi in range(2):
        a = G.apply(gi, fm(G.apply(gi, X)))
        assert np.linalg.norm((a - ref) * fe) < 1e-12 * np.linalg.norm(ref * fe)
    # the fixed subspace is invariant, the isotypic components partition the field entries
    assert G.defect(fm(G.project(X))) < 1e-12
    assert sum(int(m.sum()) for m in G.isotypic.values()) == int(fe.sum())


def test_symmetry_group_refuses_a_forgotten_field():
    """A variable left out of field_signs was treated like a tau: unlabelled, zeroed by project()
    and left out of the isotypic spectra. Only the tau variables (inside lift) stay unlabelled;
    a forgotten field is refused."""
    import pytest
    from couette_cases import S1, S2
    from conatus.ecs.subspace import DiagonalSymmetry, DiagonalSymmetryGroup
    plan, _ = couette_pt(4, 9, 4, 400.0, BOX_GHC)
    s = IVPSolver(plan, "RK443", 0.02, f_backend="fused", decouple=True)
    G = DiagonalSymmetryGroup(s, [S1, S2])
    assert G.unlabelled and all(v.startswith("t") for v in G.unlabelled)   # tphi*, tpsi*, tU*, tW*
    assert "tphi1" in G.explain()
    drop = [DiagonalSymmetry(g.name, g.axes, {k: v for k, v in g.field_signs.items() if k != "U"})
            for g in (S1, S2)]
    with pytest.raises(ValueError, match=r"\['U'\] missing"):
        DiagonalSymmetryGroup(s, drop)
