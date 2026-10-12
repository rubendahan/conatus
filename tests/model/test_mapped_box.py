"""Mapped 2D/3D domains: a channel between two surfaces, terrain-following map
z = b + (t - b)(1 + xi)/2, physical operators by the chain rule, metric implicit (one coupled
sparse system). Validation by manufactured solutions (observed spectral convergence), the flat
limit (= cn.Box to rounding), Stokes-Couette over a wavy wall against the first-order
perturbation solution (remainder O(eps^2) observed), time order of the IMEX schemes, and the
refusals (metric/NCC, traces off the surfaces, slip conditions on a curved wall)."""
import numpy as np
import pytest

import conatus as cn
from conatus.errors import ConatusError, DiscretizationError, UnsupportedError

TWO_PI = 2 * np.pi


def _wavy(eps, top=1.0):
    return cn.MappedBox(x=cn.periodic(TWO_PI), z=cn.between(bottom=lambda x: eps * np.cos(x),
                                                              top=top))


# --------------------------------------------------------------------------- Poisson (MMS)
def _ue(x, z):
    return np.exp(np.sin(x)) * np.sin(z) + z ** 2


def _fe(x, z):     # lap(_ue)
    return (np.cos(x) ** 2 - np.sin(x) - 1) * np.exp(np.sin(x)) * np.sin(z) + 2


def _poisson(eps, N, extra=None):
    h = lambda x: eps * np.cos(x)                                                   # noqa: E731
    m = cn.Model(_wavy(eps))
    m.scalar("u")
    m.given("f", _fe)
    m.equations("lap(u) = f")
    m.at("bottom", u=lambda x: _ue(x, h(x)))
    m.at("top", u=lambda x: _ue(x, 1.0))
    s = m.solve(N)
    X, Z = s.coordinate("x"), s.coordinate("z")
    return float(np.abs(s.grid("u") - _ue(X, Z)).max()), s


def test_poisson_mms_spectral_convergence_on_a_wavy_channel():
    errs = []
    for n in (16, 20, 24, 28):
        e, s = _poisson(0.05, (n, n))
        assert s.max_residual < 1e-13
        errs.append(e)
    # geometric decay down to rounding
    assert errs[0] > errs[1] > errs[2] and errs[-1] < 1e-13, errs
    assert errs[0] / errs[2] > 1e3, errs
    text = s.plan.explain()
    assert "J = 2/(t - b)" in text and "metric terms implicit" in text and "Fourier bandwidth" in text


def test_flat_map_is_the_flat_box_to_rounding():
    """bottom = 0, top = 1: the same problem as cn.Box(z=walls(0, 1)), block diagonal in k."""
    def build(box):
        m = cn.Model(box)
        m.scalar("u")
        m.given("f", lambda x, z: np.sin(x) * z + np.cos(2 * x) * z ** 3)
        m.equations("lap(u) - u = f")
        m.at("bottom", u=lambda x: np.cos(x))
        m.at("top", "dz(u) = 0")
        return m
    N = (16, 16)
    a = build(cn.MappedBox(x=cn.periodic(TWO_PI), z=cn.between(bottom=0.0, top=1.0))).solve(N)
    b = build(cn.Box(x=cn.periodic(TWO_PI), z=cn.walls(0.0, 1.0))).solve(N)
    assert np.abs(a.grid("u") - b.grid("u")).max() < 1e-14
    assert np.abs(a.coordinate("z")[0] - b.coordinate("z")).max() == 0.0
    # no coupling between Fourier modes when the map is flat
    P = a.plan
    L = P.matrices().L.tocoo()
    nz = P.Nz
    off = P.var_off[(P.variables[0], ())]
    sel = (L.col >= off) & (L.col < off + P.nk * nz) & (L.row < P.nk * nz)
    assert np.all(L.row[sel] // nz == (L.col[sel] - off) // nz)


def test_variable_coefficient_mms():
    """lap(u) - c u = f with c(x, z) a given field: 2D (Fourier x Chebyshev) NCC operator."""
    eps = 0.1
    h = lambda x: eps * np.cos(x)                                                   # noqa: E731
    c = lambda x, z: 1 + 0.5 * np.sin(x) * z                                        # noqa: E731
    errs = []
    for n in (20, 28):
        m = cn.Model(_wavy(eps))
        m.scalar("u")
        m.given("c", c)
        m.given("f", lambda x, z: _fe(x, z) - c(x, z) * _ue(x, z))
        m.equations("lap(u) - c*u = f")
        m.at("bottom", u=lambda x: _ue(x, h(x)))
        m.at("top", u=lambda x: _ue(x, 1.0))
        s = m.solve((n, n))
        errs.append(np.abs(s.grid("u") - _ue(s.coordinate("x"), s.coordinate("z"))).max())
    assert errs[-1] < 1e-12 and errs[0] > errs[-1], errs
    assert "NCC -c: " in s.plan.explain() and "Fourier offsets" in s.plan.explain()


def test_normal_derivative_condition_on_the_wavy_wall():
    """Neumann condition along the true normal of z = h(x), written with the normal as given
    fields: n_x dx(u) + n_z dz(u) = g (manufactured)."""
    eps = 0.1
    h, hp = (lambda x: eps * np.cos(x)), (lambda x: -eps * np.sin(x))
    nrm = lambda x: np.sqrt(1 + hp(x) ** 2)                                         # noqa: E731
    ux = lambda x, z: np.cos(x) * np.exp(np.sin(x)) * np.sin(z)                     # noqa: E731
    uz = lambda x, z: np.exp(np.sin(x)) * np.cos(z) + 2 * z                         # noqa: E731
    m = cn.Model(_wavy(eps))
    m.scalar("u")
    m.given("f", _fe)
    m.given("nx", lambda x: hp(x) / nrm(x))
    m.given("nz", lambda x: -1 / nrm(x))                 # outward normal of the bottom surface
    m.given("g", lambda x: (hp(x) * ux(x, h(x)) - uz(x, h(x))) / nrm(x))
    m.equations("lap(u) = f")
    m.at("bottom", "nx*dx(u) + nz*dz(u) = g")
    m.at("top", u=lambda x: _ue(x, 1.0))
    s = m.solve((28, 28))
    assert np.abs(s.grid("u") - _ue(s.coordinate("x"), s.coordinate("z"))).max() < 1e-13


# --------------------------------------------------------------------------- Stokes, wavy wall
def _couette_first_order(k=1.0, H=1.0):
    """psi_1 = Re[phi(z) e^{ikx}], (D^2 - k^2)^2 phi = 0, phi(0) = phi(H) = 0, phi'(0) = -1/H
    (transfer of the no-slip condition from z = eps cos kx to z = 0), phi'(H) = 0."""
    def basis(z):
        ch, sh = np.cosh(k * z), np.sinh(k * z)
        return (np.array([ch, z * ch, sh, z * sh]),
                np.array([k * sh, ch + k * z * sh, k * ch, sh + k * z * ch]))
    A = np.array([basis(0.0)[0], basis(H)[0], basis(0.0)[1], basis(H)[1]])
    co = np.linalg.solve(A, [0.0, 0.0, -1.0 / H, 0.0])
    return (lambda z: np.tensordot(co, basis(z)[0], 1)), (lambda z: np.tensordot(co, basis(z)[1], 1))


def test_stokes_couette_over_a_wavy_wall_matches_the_perturbation_solution():
    """u = z/H + eps (phi'(z) cos x, phi(z) sin x) + O(eps^2): the remainder is observed to
    scale as eps^2 (ratio 4.0 per halving), with the residual at rounding."""
    phi, dphi = _couette_first_order()
    rem = []
    for eps in (0.02, 0.01, 0.005):
        m = cn.Model(_wavy(eps))
        m.velocity("u")
        m.pressure("p")
        m.equations("grad(p) - lap(u) = 0", "div(u) = 0")
        m.at("bottom", u=0)
        m.at("top", u=(1, 0))
        s = m.solve((16, 24))
        assert s.max_residual < 1e-14
        U, X, Z = s.grid("u"), s.coordinate("x"), s.coordinate("z")
        eu = np.abs(U[0] - (Z + eps * dphi(Z) * np.cos(X))).max()
        ew = np.abs(U[1] - eps * phi(Z) * np.sin(X)).max()
        rem.append(max(eu, ew))
    r = [rem[i] / rem[i + 1] for i in range(2)]
    assert all(3.9 < q < 4.15 for q in r), (rem, r)


def test_free_slip_on_a_wavy_wall_is_refused():
    m = cn.Model(_wavy(0.05))
    m.velocity("u")
    m.pressure("p")
    m.equations("grad(p) - lap(u) = 0", "div(u) = 0")
    m.at("bottom", u=cn.free_slip)
    m.at("top", u=0)
    with pytest.raises(UnsupportedError, match="mapped wall"):
        m.solve((16, 16))


# --------------------------------------------------------------------------- EVP
def _diffusion_modes(eps, N):
    m = cn.Model(_wavy(eps))
    m.scalar("T")
    m.eigenvalue("s", growth="real")
    m.equations("s*T - lap(T) = 0")
    m.at("walls", T=0)
    r = m.eigenvalues(N, method="shift-invert", target=-9.5, k=4)
    ok = np.isfinite(r.values)
    order = np.argsort(np.abs(r.values[ok] + 9.5))
    return r.values[ok][order], r.residuals[ok][order]


def test_eigenvalues_flat_exact_and_wavy_converged():
    v, res = _diffusion_modes(0.0, (12, 16))
    assert abs(v[0] + np.pi ** 2) < 1e-11 and res[0] < 1e-12
    v1, r1 = _diffusion_modes(0.1, (20, 16))
    v2, _ = _diffusion_modes(0.1, (28, 22))
    # lowest mode: trapped where the gap is widest, above the trough of the wall at x = pi
    # (harmonic WKB: pi^2/1.1^2 + sqrt(0.1 pi^2/1.1^3) ~ 9.02)
    assert abs(v1[0] - v2[0]) < 1e-9 and r1[0] < 1e-12, (v1[0], v2[0])
    assert -9.1 < v2[0].real < -8.9 and abs(v2[0].imag) < 1e-10


def test_linearization_gives_the_same_spectrum_and_refuses_k():
    m = cn.Model(_wavy(0.1))
    T = m.scalar("T")
    m.equations("dt(T) - lap(T) = -T*T")
    m.at("walls", T=0)
    lin = m.linearize(about={T: 0})
    r = lin.eigenvalues((20, 16), method="shift-invert", target=-9.5)
    i = int(np.argmin(np.abs(r.values + 9.5)))
    v, _ = _diffusion_modes(0.1, (20, 16))
    assert abs(r.values[i] - v[0]) < 1e-10 and r.residuals[i] < 1e-12
    with pytest.raises(ConatusError, match="mapped domain"):
        lin.eigenvalues(16, k=1.0)


# --------------------------------------------------------------------------- IVP
def _diffusion_run(scheme, dt, eps=0.1, N=(20, 20), T=0.4, nonlinear=False):
    h = lambda x: eps * np.cos(x)                                                   # noqa: E731
    ue = lambda x, z, t: np.exp(-t) * np.sin(x) * np.cos(z) + z                     # noqa: E731
    m = cn.Model(_wavy(eps))
    m.scalar("u")
    m.given("g", lambda x, z: np.sin(x) * np.cos(z))
    m.given("gb", lambda x: np.sin(x) * np.cos(h(x)))
    m.given("hb", lambda x: h(x))
    m.given("gt", lambda x: np.sin(x) * np.cos(1.0))
    if nonlinear:
        # dt(u) - lap(u) = -u^2 + f: f = exp(-t) g + u_e^2 (lap of u_e = -2 exp(-t) g)
        m.given("zz", lambda x, z: z)
        m.equations("dt(u) - lap(u) = -u*u + exp(-t)*g + (exp(-t)*g + zz)**2")
    else:
        m.equations("dt(u) - lap(u) = exp(-t)*g")
    m.at("bottom", u="exp(-t)*gb + hb")
    m.at("top", u="exp(-t)*gt + 1")
    sim = m.simulate(N, scheme=scheme, dt=dt)
    sim.initial(u=lambda x, z: ue(x, z, 0.0))
    sim.run(until=T)
    return float(np.abs(sim.grid("u") - ue(sim.coordinate("x"), sim.coordinate("z"), sim.t)).max())


@pytest.mark.parametrize("scheme", ["SBDF2", "RK222"])
def test_diffusion_time_order_two(scheme):
    e1, e2 = _diffusion_run(scheme, 0.02), _diffusion_run(scheme, 0.01)
    assert 3.6 < e1 / e2 < 4.6, (e1, e2)


def test_explicit_nonlinear_term_on_the_grid():
    """-u*u evaluated on the dealiased physical grid; manufactured solution, SBDF2 order 2."""
    e1 = _diffusion_run("SBDF2", 0.02, nonlinear=True)
    e2 = _diffusion_run("SBDF2", 0.01, nonlinear=True)
    assert 3.5 < e1 / e2 < 4.6 and e2 < 5e-5, (e1, e2)


# --------------------------------------------------------------------------- 3D
def test_poisson_3d_doubly_wavy():
    eps = 0.01
    h = lambda x, y: eps * np.cos(x) * np.sin(y)                                    # noqa: E731
    top = lambda x: 1.0 + eps * np.sin(x)                                           # noqa: E731
    ue = lambda x, y, z: np.sin(x + y) * np.cosh(z) + z ** 2                        # noqa: E731
    box = cn.MappedBox(x=cn.periodic(TWO_PI), y=cn.periodic(TWO_PI),
                       z=cn.between(bottom=h, top=top))
    m = cn.Model(box)
    m.scalar("u")
    m.given("f", lambda x, y, z: -np.sin(x + y) * np.cosh(z) + 2)
    m.equations("lap(u) = f")
    m.at("bottom", u=lambda x, y: ue(x, y, h(x, y)))
    m.at("top", u=lambda x, y: ue(x, y, top(x)))
    s = m.solve((12, 12, 12))
    X, Y, Z = (s.coordinate(c) for c in "xyz")
    assert np.abs(s.grid("u") - ue(X, Y, Z)).max() < 1e-11


# --------------------------------------------------------------------------- refusals, explain
def test_under_resolved_metric_is_refused_with_the_resolution():
    with pytest.raises(DiscretizationError, match="not resolved") as ei:
        _poisson(0.2, (8, 8))
    assert "raise N_x" in str(ei.value)


def test_refusals_and_explain():
    with pytest.raises(ConatusError, match="above"):
        cn.MappedBox(x=cn.periodic(TWO_PI), z=cn.between(bottom=lambda x: 1 + np.cos(x), top=1.0))
    with pytest.raises(ConatusError):
        cn.MappedBox(x=cn.periodic(TWO_PI, complex=True), z=cn.between(bottom=0.0, top=1.0))
    m = cn.Model(_wavy(0.05))
    m.scalar("u")
    m.equations("lap(u) = 0")
    m.at("bottom", u=0)
    m.at("top", u=1)
    text = m.explain(full=True)
    assert "terrain-following" in text and "J = 2/(t - b)" in text
    comp = m.compile((16, 16))
    from conatus.discretize.plan import Plan
    with pytest.raises(DiscretizationError, match="terrain-following"):
        Plan(comp.problem)
    from conatus.ir import interp
    from conatus.ir.problem import LBVP
    u = comp.fields["u"]
    pb = LBVP([u])
    pb.add_equation((interp(u, comp.coordsys["z"], 0.0), 0))
    with pytest.raises(UnsupportedError, match="trace"):
        cn.discretize(pb).matrices()


# --------------------------------------------------------------------------- review (non-regression)
def test_aliased_coefficient_is_refused_not_substituted():
    """exp(sin 4x / 2) sampled at N = 16 puts its mode 8 on the Nyquist wavenumber (dropped) and
    mode 12 on -4: the kept-mode tail is zero, so the coupled operator silently multiplied by
    another function before the review. Refused now; resolved at N = 64."""
    def build():
        m = cn.Model(_wavy(0.05))
        m.scalar("u")
        m.given("c", lambda x, z: np.exp(0.5 * np.sin(4 * x)))
        m.equations("lap(u) - c*u = 0")
        m.at("bottom", u=0)
        m.at("top", u=1)
        return m
    with pytest.raises(DiscretizationError, match="Nyquist"):
        build().solve((16, 16))
    assert build().solve((64, 16)).max_residual < 1e-13


def test_coupled_system_size_is_refused_above_the_declared_limit(monkeypatch):
    import conatus.discretize.terrain_plan as tp
    m = cn.Model(_wavy(0.05))
    m.scalar("u")
    m.equations("lap(u) = 1")
    m.at("walls", u=0)
    monkeypatch.setattr(tp, "MAX_COUPLED_NNZ", 1000)
    with pytest.raises(UnsupportedError, match="MAX_COUPLED_NNZ"):
        m.solve((16, 16))
    monkeypatch.setattr(tp, "MAX_COUPLED_UNKNOWNS", 100)
    with pytest.raises(UnsupportedError, match="MAX_COUPLED_UNKNOWNS"):
        m.solve((16, 16))


def test_free_slip_on_the_flat_wall_of_a_mapped_box():
    """Only a curved surface refuses free slip: on a flat top (normal e_z) it is the flat-box
    condition (flat map = cn.Box to rounding; wavy bottom: u_z = 0 and dz(u_x) = 0 on top)."""
    def build(box):
        m = cn.Model(box)
        m.velocity("u")
        m.pressure("p")
        m.equations("grad(p) - lap(u) = ex", "div(u) = 0")
        m.at("bottom", u=0)
        m.at("top", u=cn.free_slip)
        return m.solve((16, 16))
    a = build(cn.MappedBox(x=cn.periodic(TWO_PI), z=cn.between(bottom=0.0, top=1.0)))
    b = build(cn.Box(x=cn.periodic(TWO_PI), z=cn.walls(0.0, 1.0)))
    assert np.abs(a.grid("u") - b.grid("u")).max() < 1e-13
    s = build(_wavy(0.05))
    assert s.max_residual < 1e-13
    assert np.abs(s.grid("u")[0]).max() > 0.1
    with pytest.raises(UnsupportedError, match="curved mapped wall"):
        m = cn.Model(_wavy(0.05))
        m.velocity("u")
        m.pressure("p")
        m.equations("grad(p) - lap(u) = 0", "div(u) = 0")
        m.at("walls", u=cn.free_slip)
        m.solve((16, 16))


def test_bc_time_is_declared_stage_and_other_modes_refused():
    m = cn.Model(_wavy(0.05))
    m.scalar("u")
    m.equations("dt(u) - lap(u) = 0")
    m.at("bottom", u="sin(t)")
    m.at("top", u=0)
    sim = m.simulate((16, 16), scheme="RK443", dt=0.01)
    assert sim.bc_time == "stage" and "rk" in sim.explain() and "auto" in sim.explain()
    assert m.simulate((16, 16), scheme="RK443", dt=0.01, bc_time="stage").bc_time == "stage"
    for mode in ("rk", "step"):
        with pytest.raises(UnsupportedError, match="bc_time"):
            m.simulate((16, 16), scheme="RK443", dt=0.01, bc_time=mode)


def test_under_resolved_metric_names_the_unresolved_direction():
    """3D wall varying along y only: the suggestion names N_y (it used to fit along x) and the
    declared tail it prints is rounded up (accepted if declared)."""
    box = cn.MappedBox(x=cn.periodic(TWO_PI), y=cn.periodic(TWO_PI),
                       z=cn.between(bottom=lambda y: 0.05 * np.cos(y), top=1.0))
    m = cn.Model(box)
    m.scalar("u")
    m.equations("lap(u) = 0")
    m.at("bottom", u=0)
    m.at("top", u=1)
    with pytest.raises(DiscretizationError, match="raise N_y") as ei:
        m.solve((8, 12, 12))
    tol = float(str(ei.value).split("ncc_tail_tol=")[-1].split()[0].rstrip(".,"))
    assert m.solve((8, 12, 12), ncc_tail_tol=tol).max_residual < 1e-13
