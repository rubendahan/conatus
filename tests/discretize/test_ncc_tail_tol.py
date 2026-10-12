"""An under-resolved NCC is refused, never used silently.

The relative Chebyshev tail of a non-constant coefficient *as sampled on the grid* (the last 3 of
its N grid coefficients) must not exceed ``ncc_tail_tol`` (declared default
``conatus.tolerances.NCC_TAIL_TOL`` = 1e-8, the ``TAIL_TARGET`` of ``cn.resolution``), or the declared ``ncc_cutoff`` if it is larger;
otherwise ``discretize(...).matrices()`` raises DiscretizationError naming the NCC, its tail, the
resolution predicted to resolve it and the ``ncc_cutoff=`` to write instead. Exact polynomial
coefficients (``multiplication(..., exact=True)``, coefficient data of a field) have no tail to
measure. ``NCCResolutionWarning`` is no longer emitted (kept importable one cycle).
"""
import warnings

import numpy as np
import pytest

import conatus as cn
import conatus.expert as C
from conatus.discretize import discretize
from conatus.discretize import ultraspherical as us
from conatus.errors import DiscretizationError, OptionError
from conatus.reference.evaluate import FieldData
from conatus.tolerances import NCC_TAIL_TOL


def _lbvp(N=16):
    cs = C.CartesianCoordinates("x")
    x = cs["x"]
    b = C.Chebyshev(x, N, (-1, 1), dealias=1)
    D = C.Domain(cs, [b])
    u, t1, t2, U = C.Field("u", D), C.Field("t1", D.without(x)), C.Field("t2", D.without(x)), C.Field("U", D)
    pb = C.LBVP([u, t1, t2])
    pb.add_equation((C.diff(C.diff(u, x), x) - U * u + C.lift(t1, b, -1) + C.lift(t2, b, -2), 1))
    pb.add_equation((C.interp(u, x, "left"), 0))
    pb.add_equation((C.interp(u, x, "right"), 0))
    return pb, U


def _tanh(X):
    return np.tanh(4 * X)


def test_default_refuses_an_under_resolved_ncc_and_says_what_to_write():
    pb, U = _lbvp(40)
    with pytest.raises(DiscretizationError) as ei:
        discretize(pb, ncc={U: _tanh}).matrices()
    msg = str(ei.value)
    assert "NCC U: not resolved on this grid" in msg and "tail 5.7" in msg and "> 1e-08" in msg
    assert "along x to about N = " in msg and "ncc_cutoff=1e-06" in msg
    assert "NCC_TAIL_TOL" in msg            # the threshold is named, with its value


def test_suggested_resolution_resolves_it():
    """The N predicted by the message (decay fitted on the coefficients) is enough."""
    pb, U = _lbvp(40)
    with pytest.raises(DiscretizationError) as ei:
        discretize(pb, ncc={U: _tanh}).matrices()
    n_new = int(str(ei.value).split("about N = ")[1].split(" ")[0])
    assert 40 < n_new <= 80
    pb, U = _lbvp(n_new)
    plan = discretize(pb, ncc={U: _tanh})
    plan.matrices()
    assert "ncc_tail_tol = 1e-08 (declared default)" in plan.explain()


def test_declared_cutoff_accepts_the_truncation_and_reports_it():
    pb, U = _lbvp(40)
    plan = discretize(pb, ncc={U: _tanh}, ncc_cutoff=1e-6)
    plan.matrices()
    ex = plan.explain()
    assert "declared ncc_cutoff = 1e-06 (truncation)" in ex and "declared ncc_cutoff=1e-06 dropped" in ex


def test_a_cutoff_smaller_than_the_tail_is_still_refused():
    """A cutoff declares the accepted accuracy; a sampled tail above it is not resolved by it."""
    pb, U = _lbvp(40)
    with pytest.raises(DiscretizationError, match=r"NCC U.*> 1e-07"):
        discretize(pb, ncc={U: _tanh}, ncc_cutoff=1e-7).matrices()


def test_declared_tolerance_accepts_a_larger_tail():
    pb, U = _lbvp(40)
    plan = discretize(pb, ncc={U: _tanh}, ncc_tail_tol=1e-6)
    plan.matrices()
    assert "ncc_tail_tol = 1e-06 (declared)" in plan.explain()
    pb, U = _lbvp(10)                         # exp(x) at N = 10: tail 2.5e-6
    with pytest.raises(DiscretizationError):
        discretize(pb, ncc={U: lambda X: np.exp(X)}).matrices()
    discretize(pb, ncc={U: lambda X: np.exp(X)}, ncc_tail_tol=5e-6).matrices()


def test_one_meaning_of_resolved():
    """The NCC refusal threshold is the tail cn.resolution calls 'resolved' for a field."""
    from conatus.resolution import TAIL_TARGET
    assert NCC_TAIL_TOL == TAIL_TARGET


def test_rounding_level_tail_is_not_judged():
    """U'' of a linear U is numerically zero: its relative tail is O(1) but at the rounding level
    of its computation (NCC_ROUNDING x eps x data scale x derivative amplification): reported,
    not refused (found by the impact run: tests/discretize/test_eigen_split.py)."""
    pb, U = _lbvp(16)
    import conatus.expert as X
    x = pb.variables[0].domain.bases[0].coord
    pb2 = X.LBVP(list(pb.variables))
    u, t1, t2 = pb.variables
    b = u.domain.bases[0]
    pb2.add_equation((X.diff(X.diff(u, x), x) - X.diff(X.diff(U, x), x) * u + X.lift(t1, b, -1)
                      + X.lift(t2, b, -2), 1))
    pb2.add_equation((X.interp(u, x, "left"), 0))
    pb2.add_equation((X.interp(u, x, "right"), 0))
    plan = discretize(pb2, ncc={U: lambda X_: 2 + X_})
    plan.matrices()
    assert "rounding level of its computation" in plan.explain()


def test_resolved_ncc_is_accepted_without_warning():
    pb, U = _lbvp()
    plan = discretize(pb, ncc={U: lambda X: 1 - X ** 2})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plan.matrices()
    assert f"ncc_tail_tol = {NCC_TAIL_TOL:g}" in plan.explain()


def test_coefficient_data_of_a_field_is_exact():
    """Coefficient data is the polynomial itself (no sampling, nothing truncated): accepted at any
    tail (e.g. the Newton iterate of an NLBVP multiplying the correction)."""
    pb, U = _lbvp(16)
    c = 0.5 ** np.arange(16)                  # tail 3e-5: refused as a grid sample
    with pytest.raises(DiscretizationError):
        discretize(pb, ncc={U: FieldData(U, us_grid(c), "grid")}).matrices()
    plan = discretize(pb, ncc={U: FieldData(U, c, "coeff")})
    plan.matrices()
    assert "exact polynomial coefficients" in plan.explain()


def us_grid(c):
    from conatus.reference import transforms as T
    b = C.Chebyshev(C.CartesianCoordinates("x")["x"], c.size, (-1, 1), dealias=1)
    return T.backward(np.asarray(c, float), b, axis=0, dealias=False)


@pytest.mark.parametrize("bad", [0.0, -1e-10, float("nan")])
def test_tolerance_must_be_positive(bad):
    pb, U = _lbvp()
    with pytest.raises(ValueError, match="ncc_tail_tol"):
        discretize(pb, ncc={U: lambda X: 1 - X ** 2}, ncc_tail_tol=bad)


def test_none_is_refused_with_the_new_meaning():
    """ncc_tail_tol=None used to mean 'warn and use as is': refused, never reinterpreted."""
    pb, U = _lbvp()
    with pytest.raises(OptionError, match=r"ncc_tail_tol=None is no longer accepted"):
        discretize(pb, ncc={U: lambda X: 1 - X ** 2}, ncc_tail_tol=None)


def test_native_layer_passes_the_declaration():
    """cn.Model: the refusal names the native fix; ncc_cutoff= is accepted by the solving calls."""
    m = cn.Model(cn.Box(z=cn.walls(-1, 1)), name="ncc")
    m.scalar("u")
    m.given("U", lambda z: np.tanh(4 * z))
    m.equations("dz(dz(u)) - U*u = 1")
    m.at("z=-1", "u = 0")
    m.at("z=1", "u = 0")
    with pytest.raises(DiscretizationError, match=r"(?s)NCC U.*about N = .*ncc_cutoff=1e-06"):
        m.solve(N={"z": 40})
    sol = m.solve(N={"z": 40}, ncc_cutoff=1e-6)
    assert np.all(np.isfinite(sol.grid("u")))


def _curvi_evp(geom):
    """-lap u - U u = s u with a radial coefficient U(r) on a disk, annulus, shell or ball (Dirichlet)."""
    if geom in ("disk", "annulus"):
        cs = C.PolarCoordinates("phi", "r")
        br = (C.DiskBasis(cs["r"], 16, 1.0, dealias=1) if geom == "disk"
              else C.AnnulusBasis(cs["r"], 16, (0.5, 1.5), dealias=1))
        D = C.Domain(cs, [C.RealFourier(cs["phi"], 4, (0, 2 * np.pi), dealias=1), br])
    else:
        cs = C.SphericalCoordinates("phi", "theta", "r")
        br = (C.BallBasis(cs["r"], 16, 1.0, dealias=1) if geom == "ball"
              else C.ShellBasis(cs["r"], 16, (0.5, 1.5), dealias=1))
        D = C.Domain(cs, [C.RealFourier(cs["phi"], 4, (0, 2 * np.pi), dealias=1),
                          C.SphereBasis(cs["theta"], 3, dealias=1), br])
    r = cs["r"]
    two = geom in ("annulus", "shell")
    u, s = C.Field("u", D, dtype=complex), C.Parameter("s")
    taus = [C.Field(f"t{i}", D.without(r), dtype=complex) for i in range(2 if two else 1)]
    U = C.Field("U", C.Domain(cs, [br]))
    pb = C.EVP([u, *taus], eigenvalue=s)
    lhs = s * u + C.lap(u) + U * u
    for i, t in enumerate(taus):
        lhs = lhs + C.lift(t, br, -1 - i, lam=2)
    pb.add_equation((lhs, 0))
    if two:
        pb.add_equation((C.interp(u, r, "left"), 0))
    pb.add_equation((C.interp(u, r, "right"), 0))
    return pb, U


@pytest.mark.parametrize("geom", ["disk", "annulus", "shell", "ball"])
def test_curvilinear_plans_refuse_an_under_resolved_radial_ncc(geom):
    """ in the polar and spherical plans: an under-resolved radial coefficient is refused by
    default, naming it and the radial resolution; a declared ncc_cutoff accepts it; a resolved
    one is accepted; the setting is printed by explain()."""
    pb, U = _curvi_evp(geom)
    if geom in ("disk", "annulus"):
        bad, good = (lambda P, R: np.tanh(20 * (R - 0.7))), (lambda P, R: 1 - R * R)
    else:
        bad, good = (lambda P, T, R: np.tanh(20 * (R - 0.7))), (lambda P, T, R: 1 - R * R)
    plan = discretize(pb, ncc={U: bad})
    with pytest.raises(DiscretizationError, match=r"(?s)NCC U.*along r"):
        plan.matrices(plan.groups[0])
    plan = discretize(pb, ncc={U: good})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plan.matrices(plan.groups[0])
    assert "ncc_tail_tol = 1e-08 (declared default)" in plan.explain()
    plan = discretize(pb, ncc={U: bad}, ncc_cutoff=0.5)
    plan.matrices(plan.groups[0])
    assert "declared ncc_cutoff = 0.5" in plan.explain()


@pytest.mark.parametrize("poly", [[0.5, 0.5], [0.375, 0.5, 0.125], [1.0]])
def test_exact_polynomial_has_no_tail_to_measure(poly):
    """The false alarm: r = (1 + t)/2 and r^2 on (0, 1) as exact Chebyshev coefficients have fewer
    than 4 coefficients, so 'the last 3' are the polynomial itself (tail = 1). Declared exact: not
    refused, and the report says why."""
    with pytest.raises(DiscretizationError):
        us.multiplication(np.array(poly), 8, 1, name="p")
    _, rep = us.multiplication(np.array(poly), 8, 1, name="p", exact=True)
    assert rep.resolved and "exact" in rep.explain()


def test_polar_metric_polynomials_do_not_warn():
    """The exact metric polynomials r = (1 + t)/2 and r^2 of polar.py (phase-1 disk/annulus
    classes) raised a false NCCResolutionWarning (1183 of them in the fast suite)."""
    from conatus.discretize.polar import _mult
    with warnings.catch_warnings():           # exact: neither refused nor warned (a warning fails)
        warnings.simplefilter("error")
        M = _mult([0.5, 0.5], 8, 1)
    assert M.shape == (8, 8)


@pytest.mark.parametrize("name", ["TanhMap", "RationalMap"])
def test_exact_map_metric_does_not_warn(name):
    """Second false alarm of the same kind (found by the impact run): the exact polynomial metric
    dxi/dy = (1 - xi^2)/L of TanhMap / RationalMap has 3 Chebyshev coefficients, 'tail' = 1."""
    from conatus.ir import maps
    cs = C.CartesianCoordinates("y")
    y = cs["y"]
    b = maps.MappedChebyshev(y, 16, getattr(maps, name)(2.0), dealias=1)
    D = C.Domain(cs, [b])
    u, t1 = C.Field("u", D), C.Field("t1", D.without(y))
    pb = C.LBVP([u, t1])
    pb.add_equation((C.diff(u, y) - u + C.lift(t1, b, -1), 0))
    pb.add_equation((C.interp(u, y, "left"), 1))
    with warnings.catch_warnings():           # neither refused nor warned (a warning fails)
        warnings.simplefilter("error")
        discretize(pb, ncc_tail_tol=1e-12).matrices()
        discretize(pb).matrices()


def _upp_lbvp(N):
    """d2u - U'' u + taus = 1 with the NCC U'' computed from the data U (no U alone)."""
    cs = C.CartesianCoordinates("x")
    x = cs["x"]
    b = C.Chebyshev(x, N, (-1, 1), dealias=1)
    D = C.Domain(cs, [b])
    u, t1, t2, U = C.Field("u", D), C.Field("t1", D.without(x)), C.Field("t2", D.without(x)), C.Field("U", D)
    pb = C.LBVP([u, t1, t2])
    pb.add_equation((C.diff(C.diff(u, x), x) - C.diff(C.diff(U, x), x) * u + C.lift(t1, b, -1)
                     + C.lift(t2, b, -2), 1))
    pb.add_equation((C.interp(u, x, "left"), 0))
    pb.add_equation((C.interp(u, x, "right"), 0))
    return pb, U


@pytest.mark.parametrize("N, a", [(64, 4.0), (256, 20.0)])
def test_rounding_exemption_does_not_mask_an_amplified_interpolation_error(N, a):
    """The rounding floor of a derived NCC grows like N^4 for U''; it used to
    exempt U'' whenever its absolute tail was below it, even when the data U were only resolved
    to 6e-11 (N = 64) or 6e-10 (N = 256): U'' then carries U's interpolation error x N^4 (true
    pointwise error 9e-6 and 2e-3), not rounding, and was labelled 'rounding level, not judged'.
    Now exempted only if the data are exact to rounding; otherwise judged (refused here, with the
    N that resolves it)."""
    pb, U = _upp_lbvp(N)
    with pytest.raises(DiscretizationError, match=r"(?s)NCC dx\(dx\(U\)\): not resolved.*about N = "
                                                  ) as ei:
        discretize(pb, ncc={U: lambda X: np.tanh(a * X)}).matrices()
    n_new = int(str(ei.value).split("about N = ")[1].split(" ")[0])
    pb, U = _upp_lbvp(n_new)
    discretize(pb, ncc={U: lambda X: np.tanh(a * X)}).matrices()


def test_rounding_exemption_still_applies_to_data_exact_to_rounding():
    """U = tanh(4x) at N = 256 is exact to rounding (tail 1e-16); U'' then has a rounding-level
    tail amplified by N^4 (above 1e-8 relative, not reducible by raising N): reported, not
    refused."""
    pb, U = _upp_lbvp(256)
    plan = discretize(pb, ncc={U: lambda X: np.tanh(4 * X)})
    plan.matrices()
    assert "rounding level of its computation" in plan.explain()
