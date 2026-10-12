"""``cross(Omega, u)`` on the left-hand side in a Cartesian box: implicit Coriolis (feat2).

Rotating triply periodic box, dt(u) + 2 Om cross(ez, u) + grad(p) = 0, div(u) = 0, from one
divergence-free Fourier mode u0 = a0 sin(k.x), a0 perpendicular to k. The amplitude obeys the
3 x 3 ODE a' = -P (2 Om ez x a) (P: projection perpendicular to k), whose eigenvalues are
0 and +- i 2 Om.k/|k| (the inertial-wave dispersion relation, checked on the matrix itself); the
run is compared with expm(A t) a0 and its rotation angle gives the measured frequency. The
implicit term (in L: the components coupled in each group of the banded solve) is compared with
the explicit treatment at small dt (both third order), and at large Om dt the explicit run
diverges while the implicit one stays bounded. Refusals: a cross product with a vector that is
not constant on the left-hand side. Frame: e_x x e_y = e_z whatever the order of the axes."""
import numpy as np
import pytest
from scipy.linalg import expm

import conatus as cn
from conatus.errors import DiscretizationError

K = np.array([1.0, 0.0, 1.0])          # wavevector (x, y, z)
A0 = np.array([1.0, 1.0, -1.0])        # amplitude, A0 . K = 0


def _model(om, implicit=True, axes=("x", "y", "z")):
    L = 2 * np.pi
    m = cn.Model(cn.Box(**{a: cn.periodic(L) for a in axes}), name="rotating box", Om=om)
    m.vector("u")
    m.scalar("p")
    if implicit:
        m.equations("dt(u) + 2*Om*cross(ez, u) + grad(p) = 0", "div(u) = 0")
    else:
        m.equations("dt(u) + grad(p) = -2*Om*explicit(cross(ez, u))", "div(u) = 0")
    return m


def _amplitude(sim, axes=("x", "y", "z")):
    """a with u = a sin(k.x) + (rest): 2 <u_c sin(k.x)> on the grid, components in (x, y, z)."""
    g = {a: np.asarray(sim.coordinate(a)) for a in axes}
    X = np.meshgrid(*[g[a] for a in axes], indexing="ij")
    pos = dict(zip(axes, X, strict=True))
    s = np.sin(K[0] * pos["x"] + K[1] * pos["y"] + K[2] * pos["z"])
    u = sim.grid("u")
    a = np.array([2 * np.mean(u[i] * s) for i in range(3)])
    return np.array([a[axes.index(c)] for c in "xyz"])


def _u0(x, y, z):
    s = np.sin(K[0] * x + K[1] * y + K[2] * z)
    return {"x": A0[0] * s, "y": A0[1] * s, "z": A0[2] * s}


def _run(om, dt, T, implicit=True, axes=("x", "y", "z")):
    sim = _model(om, implicit, axes).simulate(N=(8, 8, 8), scheme="RK443", dt=dt)
    if axes == ("x", "y", "z"):
        sim.initial(u=lambda x, y, z: np.array(list(_u0(x, y, z).values())))
    else:                                              # components in the order of the axes
        sim.initial(u=lambda x, z, y: np.array([_u0(x, y, z)[a] for a in ("x", "z", "y")]))
    sim.run(steps=int(round(T / dt)))
    return sim


def _operator(om):
    kh = K / np.linalg.norm(K)
    P = np.eye(3) - np.outer(kh, kh)
    Wx = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], float)     # ez x a
    return -P @ (2 * om * Wx)


def test_dispersion_relation_of_the_amplitude_operator():
    om = 1.0
    ev = np.sort(np.linalg.eigvals(_operator(om)).imag)
    w = 2 * om * K[2] / np.linalg.norm(K)
    assert np.allclose(ev, [-w, 0.0, w], atol=1e-14)


def test_inertial_wave_frequency_and_explicit_agreement():
    om, T = 1.0, 1.0
    exact = expm(_operator(om) * T) @ A0
    errs = {}
    for dt in (0.02, 0.01):
        imp, exp_ = _run(om, dt, T), _run(om, dt, T, implicit=False)
        errs[dt] = (np.abs(_amplitude(imp) - exact).max(), np.abs(_amplitude(exp_) - exact).max(),
                    np.abs(_amplitude(imp) - _amplitude(exp_)).max())
    for i in (0, 1):
        assert np.log2(errs[0.02][i] / errs[0.01][i]) > 2.7, errs           # third order
    assert errs[0.01][0] < 1e-6 and errs[0.01][2] < 1e-6, errs
    # measured frequency: rotation angle of a(T) in the plane perpendicular to k
    imp = _run(om, 0.005, T)
    a = _amplitude(imp)
    kh = K / np.linalg.norm(K)
    b0 = np.cross(kh, A0)
    theta = np.arctan2(a @ b0 / (b0 @ b0), a @ A0 / (A0 @ A0))
    w = 2 * om * K[2] / np.linalg.norm(K)
    assert abs(abs(theta) / T - w) < 1e-6 * w, (theta / T, w)
    assert "cross(ez, u)" in imp.explain(full=True)


@pytest.mark.filterwarnings("ignore::RuntimeWarning", "ignore::conatus.errors.UnderResolvedWarning")
def test_large_rotation_number_explicit_diverges_implicit_bounded():
    om, dt, n = 50.0, 0.05, 40                         # 2 Om dt = 5
    with pytest.warns(cn.errors.UnderResolvedWarning, match="damp and slow"):
        imp = _run(om, dt, n * dt)
    assert np.abs(imp.grid("u")).max() <= np.abs(A0).max() * (1 + 1e-12)
    with pytest.raises(cn.errors.NumericalError):       # the finite-state check stops it
        _run(om, dt, 400 * dt, implicit=False)


def test_frame_does_not_depend_on_the_order_of_the_axes():
    a = _amplitude(_run(1.0, 0.02, 0.4))
    b = _amplitude(_run(1.0, 0.02, 0.4, axes=("x", "z", "y")), axes=("x", "z", "y"))
    assert np.abs(a - b).max() < 1e-12


def test_cross_table_and_refusals():
    from conatus.ir.expr_algebra import cartesian_cross_table
    sys = cn.Box(x=cn.periodic(1.0), z=cn.periodic(1.0), y=cn.periodic(1.0)).coordsys()
    t = cartesian_cross_table(sys)                       # (x, z, y): e_x x e_z = -e_y
    assert t[(0, 1)] == (2, -1.0) and t[(1, 2)] == (0, -1.0)
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), y=cn.periodic(2 * np.pi), z=cn.periodic(2 * np.pi)),
                 name="variable rotation")
    m.vector("u")
    m.scalar("p")
    m.given("W", lambda x, y, z: np.array([0 * (x + y + z), 0 * (x + y + z), 1 + 0 * (x + y + z) + 0.1 * np.sin(z)]), rank=1)
    m.equations("dt(u) + cross(W, u) + grad(p) = 0", "div(u) = 0")
    with pytest.raises(DiscretizationError, match="needs a constant vector"):
        m.simulate(N=(8, 8, 8), scheme="RK443", dt=1e-2)


@pytest.mark.parametrize("implicit", [True, False])
def test_flowmap_dot_product_and_fused_rhs_with_cross(implicit):
    """The exact discrete tangent/adjoint (FlowMap) through the implicit term (in L) and through
    the explicit one (polynomial F' and F'^T of the cross product), and the fused F = the
    reference F with a cross product on the right-hand side."""
    from rounding_bounds import adjoint_dot_check
    L = 2 * np.pi
    m = cn.Model(cn.Box(x=cn.periodic(L), y=cn.periodic(L), z=cn.periodic(L)), name="rotating", Om=1.0)
    m.vector("u")
    m.scalar("p")
    nl = "- dot(u, grad(u)) - 0.1*cross(u, dz(u))"           # F' and F'^T of a bilinear cross
    m.equations(f"dt(u) + 2*Om*cross(ez, u) + grad(p) = {nl}" if implicit else
                f"dt(u) + grad(p) = -2*Om*explicit(cross(ez, u)) {nl}", "div(u) = 0")
    sim = m.simulate(N=(8, 8, 8), scheme="RK443", dt=1e-2)
    sim.initial(u=lambda x, y, z: np.array([np.sin(z) + 0.3 * np.cos(x + y), np.cos(z + x) + 0 * y,
                                             0.2 * np.sin(x + y) + 0 * z]))
    fm = sim.flowmap(3e-2)
    rng = np.random.default_rng(0)
    n = fm.n
    X0 = sim.solver.X.copy()
    vec = [rng.standard_normal(n) for _ in range(4)]
    adjoint_dot_check(lambda v: fm.jvp(X0, v), lambda w: fm.vjp(X0, w), *vec, cap=1e-9,
                      dims=(n, n))
    fused = m.simulate(N=(8, 8, 8), scheme="RK443", dt=1e-2, f_backend="fused")
    ref = sim.solver.F(X0, 0.0)
    assert np.abs(fused.solver.F(X0, 0.0) - ref).max() <= 1e-12 * np.abs(ref).max()


def test_rotating_channel_with_walls_implicit_equals_explicit():
    """Chebyshev walls in z (no slip): the implicit term couples the components inside each
    (kx, ky) group of the bordered banded solve; it converges at third order to the reference and
    agrees with the explicit run at small dt."""
    def run(implicit, dt):
        m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), y=cn.periodic(2 * np.pi), z=cn.walls(0, 1)),
                     name="rotating channel", Om=2.0, nu=0.05)
        m.vector("u")
        m.scalar("p")
        m.equations("dt(u) + 2*Om*cross(ex, u) + grad(p) - nu*lap(u) = 0" if implicit else
                    "dt(u) + grad(p) - nu*lap(u) = -2*Om*explicit(cross(ex, u))", "div(u) = 0")
        m.at("walls", u=0)
        sim = m.simulate(N=(8, 8, 16), scheme="RK443", dt=dt)
        sim.initial(u=lambda x, y, z: np.array([np.sin(np.pi * z) * np.cos(y) + 0 * x,
                                                 np.sin(np.pi * z) ** 2 * np.sin(x) + 0 * y,
                                                 0 * (x + y + z)]), check=False)
        sim.run(until=0.2)
        return sim.grid("u")
    ref = run(True, 0.00125)
    e = {dt: (np.abs(run(True, dt) - ref).max(), np.abs(run(False, dt) - ref).max())
         for dt in (0.01, 0.005)}
    # implicit: third order (measured 2.97 against dt = 0.000625); the explicit treatment shows
    # the order reduction of IMEX schemes with constraints and walls (measured 2.5 then 1.6)
    assert np.log2(e[0.01][0] / e[0.005][0]) > 2.7, e
    assert e[0.01][1] < 1e-6 and e[0.005][0] < e[0.005][1], e


def test_implicit_rotation_is_declared_and_warned_when_its_waves_are_not_resolved():
    """Review of feat2: the implicit Coriolis term is stable at any dt but damps its waves when
    |W| dt > 1 (|W| = 2 Om bounds the inertial frequency 2 Om.k/|k|): explain() states |W| dt for
    every implicit rotation term, one UnderResolvedWarning per solver above 1 (also after set_dt),
    nothing for the explicit treatment. The damping itself: |R(i omega dt)| of the implicit RK443
    stages, 1.7 % per step at omega dt = 1 and 13 % at 2 (the numbers of the warning's doc)."""
    import warnings

    from conatus.timestep.integrator import rk_tableau
    H = np.array(rk_tableau("RK443")[1], dtype=float)
    damp = {w: abs(np.linalg.solve(np.eye(5) - 1j * w * H, np.ones(5))[-1]) for w in (1.0, 2.0)}
    assert abs(damp[1.0] - 0.983) < 1e-3 and abs(damp[2.0] - 0.870) < 1e-3, damp
    with warnings.catch_warnings():
        warnings.simplefilter("error", cn.errors.UnderResolvedWarning)
        sim = _model(1.0).simulate(N=(8, 8, 8), scheme="RK443", dt=0.1)    # |W| dt = 0.2
    line = [ln for ln in sim.explain().splitlines() if "implicit rotation" in ln]
    assert len(line) == 1 and "2*Om*cross(ez, u): |W| = 2, |W| dt = 0.2" in line[0], line
    assert "resolved in time" in line[0]
    with pytest.warns(cn.errors.UnderResolvedWarning, match=r"\|W\| dt = 6"):
        sim.solver.set_dt(3.0)
    assert "damp and slow" in sim.solver.explain(full=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", cn.errors.UnderResolvedWarning)
        sim.solver.set_dt(4.0)                                             # warned once
        ex = _model(50.0, implicit=False).simulate(N=(8, 8, 8), scheme="RK443", dt=0.05)
    assert "implicit rotation" not in ex.explain()


def test_cartesian_cross_table_refuses_curvilinear_frames():
    """The table orders a Cartesian frame; a curvilinear one (cylindrical (phi, r, z):
    e_phi x e_r = -e_z, not +e_z as the declaration order would say) is refused."""
    import conatus.expert as C
    from conatus.ir.expr_algebra import cartesian_cross_table
    with pytest.raises(cn.errors.TensorError, match="no Cartesian frame table"):
        cartesian_cross_table(C.CylindricalCoordinates("phi", "r", "z"))
