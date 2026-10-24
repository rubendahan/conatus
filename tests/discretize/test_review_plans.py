"""Regression tests of the external review, lot B "numerics": discretized plans
(mapped, flat and curvilinear). Finding number in the test name; each fails on v0.1.0.
"""
import numpy as np
import pytest

import conatus as cn


def _coefficient_model(geo, q):
    m = cn.Model(geo)
    m.scalar("u")
    m.given("q", q)
    m.equations("q*u = q")
    return m


@pytest.mark.parametrize("geometry", ["mapped", "flat", "annulus"])
def test_f4_changing_a_coefficient_rebuilds_the_operator(geometry):
    """Finding 4: solve({q: new}) changed the right-hand side only (2u = 4, residual 0); the
    operator now uses the new coefficient, the residual is the updated problem's, and the plan
    keeps its own coefficient for later solves."""
    if geometry == "mapped":
        geo = cn.MappedBox(x=cn.periodic(2 * np.pi), z=cn.between(bottom=0., top=1.))
        q0, q1 = (lambda x, z: 2. + 0 * x + 0 * z), (lambda x, z: 4. + 0 * x + 0 * z)
    elif geometry == "flat":
        geo = cn.Box(x=cn.periodic(2 * np.pi), z=cn.walls(0., 1.))
        q0, q1 = (lambda z: 2. + 0 * z), (lambda x, z: 4. + z + 0 * x)
    else:
        geo = cn.Annulus(inner=1., outer=2.)
        q0, q1 = (lambda r: 2. + 0 * r), (lambda phi, r: 4. + r + 0 * phi)
    m = _coefficient_model(geo, q0)
    c = m.compile((8, 8))
    p = c.discretize()
    assert abs(p.solve().grid("u").mean() - 1.0) < 1e-12
    r = p.solve({c.fields["q"]: q1})
    assert np.allclose(r.grid("u"), 1.0, atol=1e-10)
    assert r.max_residual < 1e-12
    assert abs(p.solve().grid("u").mean() - 1.0) < 1e-12          # the plan is unchanged


def test_f4_mapped_varying_coefficient_matches_the_rebuilt_plan():
    """Finding 4: solve(data) equals a plan discretized with that coefficient from the start."""
    geo = cn.MappedBox(x=cn.periodic(2 * np.pi), z=cn.between(bottom=0., top=1.))
    q1 = lambda x, z: 3. + np.cos(x) + z          # noqa: E731
    m = cn.Model(geo)
    m.scalar("u")
    m.given("q", lambda x, z: 2. + 0 * x + 0 * z)
    m.given("f", lambda x, z: 1. + 0 * x + z)
    m.equations("q*u = f")
    c = m.compile((8, 8))
    got = c.discretize().solve({c.fields["q"]: q1}).grid("u")
    m2 = cn.Model(geo)
    m2.scalar("u")
    m2.given("q", q1)
    m2.given("f", lambda x, z: 1. + 0 * x + z)
    m2.equations("q*u = f")
    ref = m2.compile((8, 8)).discretize().solve().grid("u")
    assert np.max(np.abs(got - ref)) < 1e-12


def test_f4_coefficient_changed_in_place_is_new_data():
    """Numerics review: the plan compared new data by identity, so an array changed in place and
    passed again was "already bound": the operator kept q = 3 while the right-hand side read
    q = 5 (u = 5/3, residual 0). The plan keeps a copy of its data and compares values."""
    geo = cn.Box(x=cn.periodic(2 * np.pi), z=cn.walls(0., 1.))
    m = _coefficient_model(geo, lambda z: 2. + 0 * z)
    p0 = m.compile((8, 8)).discretize()
    q = next(f for f in p0._ncc_src if f.name == "q")
    from conatus.reference.evaluate import FieldData
    arr = np.full(p0._grid_shape(q.domain, q.tensorsig), 3.0)
    data = FieldData(q, arr, "grid")
    p = type(p0)(p0.problem, ncc={**p0._ncc_src, q: data}, params=p0.params)
    assert np.allclose(p.solve().grid("u"), 1.0, atol=1e-10)
    arr[...] = 5.0
    r = p.solve({q: data})
    assert np.allclose(r.grid("u"), 1.0, atol=1e-10) and r.max_residual < 1e-12


@pytest.mark.parametrize("geometry", ["flat", "mapped"])
def test_f4_same_new_coefficient_reuses_the_rebuilt_plan(geometry):
    """Numerics review (cost of finding 4): solve(data) with the same new coefficient again
    reuses the plan rebuilt for it (one slot) instead of assembling the operator every call."""
    if geometry == "mapped":
        geo = cn.MappedBox(x=cn.periodic(2 * np.pi), z=cn.between(bottom=0., top=1.))
        q0, q1 = (lambda x, z: 2. + 0 * x + 0 * z), (lambda x, z: 4. + z + 0 * x)
    else:
        geo = cn.Box(x=cn.periodic(2 * np.pi), z=cn.walls(0., 1.))
        q0, q1 = (lambda z: 2. + 0 * z), (lambda x, z: 4. + z + 0 * x)
    c = _coefficient_model(geo, q0).compile((8, 8))
    p = c.discretize()
    q = c.fields["q"]
    p.solve({q: q1})
    first = p._rebound
    r = p.solve({q: q1})
    assert p._rebound is first and np.allclose(r.grid("u"), 1.0, atol=1e-10)
    assert abs(p.solve().grid("u").mean() - 1.0) < 1e-12


def test_f34_user_function_on_the_sphere():
    """Finding 34: a user function (m.function) failed on the sphere with KeyError; it now
    gives the same step as the built-in function it wraps."""
    out = []
    for name in ("g", "tanh"):
        m = cn.Model(cn.Sphere(1.0))
        m.scalar("u")
        if name == "g":
            m.function("g", np.tanh, derivative=lambda u: 1 - np.tanh(u) ** 2)
        m.equations(f"dt(u) - lap(u) = -{name}(u)")
        s = m.simulate((16, 8), scheme="RK222", dt=1e-3).initial(
            u=lambda phi, theta: np.cos(theta))
        s.run(steps=2)
        out.append(s.grid("u"))
    assert np.max(np.abs(out[0] - out[1])) < 1e-14


@pytest.mark.parametrize("geo, N", [(cn.Disk(1.0), (8, 12)), (cn.Annulus(0.5, 1.5), (8, 12)),
                                    (cn.Shell(0.5, 1.5), (8, 6, 8))])
def test_f35_tensor_boundary_rows_survive_rescaling(monkeypatch, geo, N):
    """Finding 35: the rows of a tensor boundary condition dot(C, u) were dropped by an absolute
    threshold 1e-14, so C = 1e-15 (Pt, K) (the same condition, rescaled) lost every row. The
    threshold is relative: the rescaled condition keeps every entry, scaled by 1e-15."""
    import conatus.model.taus as mc          # the tau placement reads curved_wall_tensors there

    def matrices():
        m = cn.Model(geo, name="free-slip", nu=0.1)
        m.velocity("u")
        m.pressure("p")
        m.equations("dt(u) + grad(p) - nu*lap(u) = -u@grad(u)", "div(u) = 0")
        m.at("walls" if "walls" in geo.boundaries() else "wall", u=cn.free_slip)
        plan = m.discretize(N)
        return [plan.matrices(g, False).L for g in plan.groups[:4]]

    ref = matrices()
    original = mc.curved_wall_tensors

    def scaled(kind):
        pt, k, meaning = original(kind)
        return (lambda *g: 1e-15 * pt(*g)), (lambda *g: 1e-15 * k(*g)), meaning
    monkeypatch.setattr(mc, "curved_wall_tensors", scaled)
    got = matrices()
    for A, B in zip(ref, got, strict=True):
        assert A.nnz == B.nnz                           # no row lost
        rows = np.unique((A != B).nonzero()[0])         # the boundary rows, scaled
        assert rows.size
        assert np.allclose(B[rows].toarray(), 1e-15 * A[rows].toarray(), rtol=1e-12, atol=0)


def _mapped_heat():
    geo = cn.MappedBox(x=cn.periodic(2 * np.pi), z=cn.between(bottom=0., top=1.))
    m = cn.Model(geo)
    m.scalar("u")
    m.equations("dt(u) - lap(u) = 0")
    m.at("bottom", u=0)
    m.at("top", u=0)
    return m


def test_f36_mapped_complex_initial_data_refused():
    """Finding 36: a small deliberate imaginary part of initial data was dropped by grid()."""
    s = _mapped_heat().simulate((8, 8), scheme="RK222", dt=1e-3)
    with pytest.raises(cn.errors.DtypeError, match="imag"):
        s.initial(u=lambda x, z: np.sin(np.pi * z) * (1 + 1e-10j) + 0 * x)
    with pytest.raises(cn.errors.InputTypeError, match="non-finite"):
        s.initial(u=lambda x, z: np.nan + 0 * x * z)
    s.initial(u=lambda x, z: np.sin(np.pi * z) + 0 * x)        # real data: accepted


@pytest.mark.parametrize("dt", [np.inf, np.nan, 0.0, True])
def test_f37_mapped_dt_must_be_finite_positive(dt):
    """Finding 37: the mapped solver accepted dt=inf."""
    with pytest.raises((cn.errors.OptionError, cn.errors.InputTypeError)):
        _mapped_heat().simulate((8, 8), scheme="RK222", dt=dt)


def test_f4_mapped_right_hand_side_data_keeps_the_operator():
    """Numerics review: new right-hand-side data on a mapped plan reuse its operator (a view
    with its own data caches; the lot rebuilt the whole plan at every call) and give the
    solution of a plan discretized with that data; the plan keeps its own data."""
    geo = cn.MappedBox(x=cn.periodic(2 * np.pi), z=cn.between(bottom=0., top=1.))

    def model(f):
        m = cn.Model(geo)
        m.scalar("u")
        m.given("f", f)
        m.equations("dz(dz(u)) + dx(dx(u)) - u = f")
        m.at("bottom", u=0)
        m.at("top", u=0)
        return m.compile((8, 12))
    f0, f1 = (lambda x, z: 1. + z + 0 * x), (lambda x, z: np.cos(x) * z)
    c = model(f0)
    p = c.discretize()
    u0 = p.solve().grid("u")
    gm = p.matrices()
    r = p.solve({c.fields["f"]: f1})
    ref = model(f1).discretize().solve().grid("u")
    assert np.max(np.abs(r.grid("u") - ref)) < 1e-12 and r.max_residual < 1e-12
    assert getattr(p, "_rebound", None) is None and p.matrices() is gm
    assert np.max(np.abs(p.solve().grid("u") - u0)) == 0.0
