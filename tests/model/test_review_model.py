"""Regression tests of the external review, lot B "numerics": the native model layer.

Each test reproduces the reviewer's script (finding number in the
test name) and fails on v0.1.0.
"""
import numpy as np
import pytest

import conatus as cn


# ---------------------------------------------------------------------------------- finding 3
def _rhs_coeffs(m, N, dealias, init):
    s = m.simulate(N, scheme="RK222", dt=.01, dealias=dealias)
    s.initial(u=init)
    return s, s.solver.system.scatter(s.solver.F(s.solver.X, 0.))[0]


def test_f3_given_fields_count_in_the_product_degree():
    """Finding 3: U*u*u with a given U is a cubic product: default factor 2, no alias of
    cos(7x)^3 (= 3/4 cos 7x + 1/4 cos 21x; 21 aliased to 3 at 3/2 padding and N = 16)."""
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
    m.scalar("u")
    m.given(U=lambda x: np.cos(7 * x))
    m.equations("dt(u) = U*u*u")
    s, c = _rhs_coeffs(m, 16, None, lambda x: np.cos(7 * x))
    assert s.compiled.dealias == {"x": 2.0}
    assert abs(c[6]) < 1e-12 and abs(c[14] - 0.75) < 1e-12


@pytest.mark.parametrize("box", ["fourier", "chebyshev"])
def test_f3_default_dealias_matches_exact_convolution(box):
    """Finding 3: with the default factor the pseudospectral coefficients of U*u*u equal the
    exact (over-resolved, factor 4 > (3+1)/2) truncated convolution for data resolved at N."""
    N = 16
    rng = np.random.default_rng(3)
    # every mode of the resolution: 8 real Fourier wavenumbers, 16 Chebyshev polynomials (with 8
    # Chebyshev modes the cubic, degree 21, is exact on 24 points: the case passed at 3/2)
    n = 8 if box == "fourier" else 16
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    if box == "fourier":
        geo = cn.Box(x=cn.periodic(2 * np.pi))
        U = lambda x: sum(a[k] * np.cos(k * x) for k in range(8))       # noqa: E731
        u0 = lambda x: sum(b[k] * np.sin(k * x + 1.) for k in range(8))  # noqa: E731
    else:
        geo = cn.Box(x=cn.walls(-1, 1))
        U = lambda x: np.polynomial.chebyshev.chebval(x, a)              # noqa: E731
        u0 = lambda x: np.polynomial.chebyshev.chebval(x, b)             # noqa: E731
    m = cn.Model(geo)
    m.scalar("u")
    m.given(U=U)
    m.equations("dt(u) = U*u*u")
    _, c_default = _rhs_coeffs(m, N, None, u0)
    _, c_exact = _rhs_coeffs(m, N, 4, u0)
    assert np.max(np.abs(c_default - c_exact)) < 1e-12 * np.max(np.abs(c_exact))


def test_f3_explain_states_the_condition_not_a_universal_guarantee():
    """Finding 3: explain() names the counted given fields and the truncation condition."""
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
    m.scalar("u")
    m.given(U=lambda x: np.cos(7 * x))
    m.equations("dt(u) = U*u*u")
    text = m.compile(16).explain(full=True)
    assert "exact (alias-free)" not in text
    assert "given fields" in text and "p = 3" in text


@pytest.mark.parametrize("eq, factor", [("dt(u) = exp(G)*u*u*u", 5 / 2), ("dt(u) = sqrt(G)*u*u*u", 5 / 2),
                                        ("dt(u) = u*u/G", 2.0), ("dt(u) = G*exp(G)*u*u", 5 / 2),
                                        ("dt(u) = exp(G)*u", 3 / 2), ("dt(u) = exp(u)*u", 3 / 2)])
def test_f3_function_of_a_given_field_is_one_factor(eq, factor):
    """Numerics review: a non-polynomial function of given fields alone (exp(G), sqrt(G), 1/G)
    is a known coefficient, one factor of the product like G itself. The lot counted the term as
    "not polynomial" and dropped it from the degree: exp(G)*u**3 got 3/2, below the factor 2 of
    v0.1.0 for the u**3 alone. A function of an unknown stays non-polynomial (listed, not
    counted)."""
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
    m.scalar("u")
    m.given(G=lambda x: 1 + 0.1 * np.cos(x))
    m.equations(eq)
    c = m.compile(16)
    assert c.dealias == {"x": factor}
    assert bool(c.nonpoly) == ("exp(u)" in eq)


# ---------------------------------------------------------------------------------- finding 7
def _sincos_model(eq="dt(T)-lap(T)=0", bc="dz(T)=0"):
    m = cn.Model(cn.Box(z=cn.walls(0, 1, basis="sincos")))
    m.scalar("T")
    m.equations(eq)
    m.at("walls", bc)
    return m


@pytest.mark.parametrize("bc, initial, expected", [
    ("dz(T)=0", lambda z: 1 + 0 * z, (1.0, 1.0)),
    ("T=0", lambda z: np.sin(np.pi * z), (2 / np.pi, 0.5)),
])
def test_f7_sincos_integrals_are_over_the_physical_interval(bc, initial, expected):
    """Finding 7: integ() on a sine/cosine axis was over the doubled period (2.0 2.0 and
    0.0 1.0); it is the physical integral over [0, 1] for both parities."""
    s = _sincos_model(bc=bc).simulate(16, scheme="RK222", dt=.1).initial(T=initial)
    assert abs(s.evaluate("integ(T)") - expected[0]) < 1e-12
    assert abs(s.evaluate("integ(T*T)") - expected[1]) < 1e-12


def test_f7_sincos_integral_in_an_equation():
    """Finding 7: an integral written in an equation is the physical one (implicit or explicit):
    dt(T) = -integ(T) with T = 1 decays like exp(-t), not exp(-2t)."""
    for eq in ("dt(T)-lap(T) = -integ(T)", "dt(T)-lap(T) + integ(T) = 0"):
        s = _sincos_model(eq).simulate(16, scheme="RK443", dt=.01).initial(T=lambda z: 1 + 0 * z)
        s.run(steps=10)
        assert abs(s.grid("T")[0] - np.exp(-0.1)) < 1e-8, eq


def test_f7_sincos_integral_of_a_sine_series_in_an_equation_is_refused():
    """Finding 7: the physical integral of a sine series couples all odd modes: refused."""
    m = _sincos_model("dt(T)-lap(T) = -integ(T)", bc="T=0")
    with pytest.raises(cn.errors.UnsupportedError, match="sine series"):
        m.simulate(16, scheme="RK222", dt=.01)


def test_f7_dye_volume_is_the_physical_volume(tmp_path):
    """Finding 7: the dye diagnostics used the doubled-period volume (variance off)."""
    pytest.importorskip("h5py")
    from conatus.model.dye import DyeFile
    s = _sincos_model().simulate(16, scheme="RK222", dt=.1).initial(
        T=lambda z: 1 + np.cos(np.pi * z))
    d = DyeFile(s, "T", tmp_path / "dye.h5", every=1, mode="w", initial=True, field=False)
    assert d.volume == 1.0
    # variance = int T^2 - (int T)^2 / V = (1 + 1/2) - 1 = 1/2
    m = s.evaluate("integ(T)")
    assert abs(s.evaluate("integ(T*T)") - m * m / d.volume - 0.5) < 1e-12


# ------------------------------------------------------------------------- findings 27-33
def _box():
    return cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="m", nu=0.1)


def test_f27_invalid_equation_does_not_poison_the_model():
    """Findings 27/61: a failing equations() call left the bad line, so the corrected call
    failed too. Validate, then mutate."""
    m = _box()
    m.scalar("u")
    with pytest.raises(cn.errors.ConatusError):
        m.equations("dt(u) = typo")
    assert m.eqs == []
    m.equations("dt(u) = nu*dx(dx(u))")
    assert len(m.eqs) == 1
    m.simulate(8, scheme="RK222", dt=.01).initial(u=lambda x: np.sin(x)).run(steps=1)


def test_f27_failing_at_given_and_dye_leave_the_model_unchanged():
    """Finding 27: at(), given() and dye() fail atomically (no half-declared state)."""
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), z=cn.walls(0, 1)))
    m.scalar("T")
    with pytest.raises(cn.errors.ConatusError):
        m.at("walls", "T = 0", X=0)                   # second item invalid
    assert m.conditions == []
    with pytest.raises(cn.errors.ConatusError):
        m.given(f=lambda x: x, g="not a field")
    assert m.givens == []
    m.velocity("u")
    with pytest.raises(cn.errors.ConatusError):
        m.dye("c", velocity="u", kappa="undeclared")
    assert [u.name for u in m.unknowns] == ["T", "u"] and m.eqs == []


def test_f28_params_ior_is_validated():
    """Finding 28: m.params |= {...} bypassed the validation (NaN, undeclared names)."""
    m = _box()
    for bad in ({"nu": float("nan")}, {"nuu": 1.0}, {"nu": 0.2, "x": 1.0}):
        with pytest.raises(cn.errors.ConatusError):
            m.params |= bad
        assert m.params["nu"] == 0.1
    m.params |= {"nu": 0.3}
    assert m.nu == 0.3 and isinstance(m.params, dict)


def test_f29_unknown_dealias_axis_refused():
    """Finding 29: a dealias key that names no coordinate was silently ignored."""
    m = _box()
    m.scalar("u")
    m.equations("dt(u) = -u*dx(u)")
    with pytest.raises(cn.errors.ResolutionError, match="not coordinate"):
        m.compile(8, dealias={"x": 1.5, "z": 2})


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_f30_nonfinite_scalar_initial_data_refused(bad):
    """Finding 30: initial(u=np.nan) was accepted for a scalar value."""
    m = _box()
    m.scalar("u")
    m.equations("dt(u) = nu*dx(dx(u))")
    s = m.simulate(8, scheme="RK222", dt=.01)
    with pytest.raises(cn.errors.InputTypeError, match="finite"):
        s.initial(u=bad)


def test_f31_duplicate_macro_parameters_refused():
    """Finding 31: define("pick(a, a) = a") was accepted."""
    m = _box()
    with pytest.raises(cn.errors.EquationError, match="repeated"):
        m.define("pick(a, a) = a")
    assert m.macros == []
    m.define("pick(a, b) = a")


def test_f32_keyword_only_user_functions_refused():
    """Finding 32: keyword-only arguments were validated by name, then called by position."""
    m = _box()
    m.scalar("u")
    with pytest.raises(cn.errors.InputTypeError, match="keyword-only"):
        m.given(U=lambda *, x: np.cos(x))
    assert m.givens == []
    m.equations("dt(u) = nu*dx(dx(u))")
    s = m.simulate(8, scheme="RK222", dt=.01)
    with pytest.raises(cn.errors.InputTypeError, match="keyword-only"):
        s.initial(u=lambda *, x: np.cos(x))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_f33_dealias_nan_inf_refused(bad):
    """Finding 33: dealias=nan/inf passed the 'd < 1' check."""
    from conatus.ir import CartesianCoordinates, Chebyshev
    cs = CartesianCoordinates("x")
    with pytest.raises(cn.errors.OptionError, match="finite"):
        Chebyshev(cs["x"], 8, (-1, 1), dealias=bad)
    m = _box()
    m.scalar("u")
    m.equations("dt(u) = -u*dx(u)")
    with pytest.raises(cn.errors.OptionError):
        m.compile(8, dealias=bad)


# ------------------------------------------------------------------------- findings 64, 65
def test_f64_dye_capability_checked_before_the_file(tmp_path):
    """Finding 64: dye diagnostics on the sphere created the file, then failed in evaluate()."""
    m = cn.Model(cn.Sphere(1.0))
    m.scalar("T")
    m.equations("dt(T) - lap(T) = 0")
    s = m.simulate((16, 8), scheme="RK222", dt=1e-3).initial(T=lambda phi, theta: np.cos(theta))
    path = tmp_path / "dye.h5"
    with pytest.raises(cn.errors.UnsupportedError, match="dye"):
        s.dye("T", path, every=1)
    assert not path.exists() and not s.outputs


def test_f65_cfl_geometric_contract():
    """Finding 65: the CFL helper has an explicit contract (Cartesian axes); curvilinear
    coordinates are refused instead of mixing angles and lengths; safety is validated."""
    m = cn.Model(cn.Disk(1.0), nu=0.1)
    m.velocity("u")
    m.pressure("p")
    m.equations("dt(u) + grad(p) - nu*lap(u) = -u@grad(u)", "div(u) = 0")
    m.at("wall", u=0)
    s = m.simulate((8, 12), scheme="RK222", dt=1e-3)
    with pytest.raises(cn.errors.UnsupportedError, match="curvilinear"):
        s.cfl("u", safety=0.5)
    b = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), y=cn.periodic(2 * np.pi)))
    b.velocity("u")
    b.equations("dt(u) = 0")
    sb = b.simulate((8, 8), scheme="RK222", dt=1e-3).initial(
        u=lambda x, y: (np.ones_like(x * y), 0 * x * y))
    assert abs(sb.cfl("u", safety=0.5) - 0.5 * (2 * np.pi / 8)) < 1e-12
    with pytest.raises(cn.errors.OptionError):
        sb.cfl("u", safety=0.0)
