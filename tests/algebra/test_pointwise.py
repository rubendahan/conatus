"""Pointwise functions in the IR: exp, log, sin, cos, tanh, sqrt, abs, sign.

Linearization exact (structural + Taylor), parity rules checked on the grid, refusals explicit
(non-finite values, odd argument without parity, transpose, sign linearization, complex abs),
non-polynomial evaluation declared, pseudospectral RHS support, parametric pencils refused.
"""
import numpy as np
import pytest

from conatus.ir import (CartesianCoordinates, Chebyshev, Constant, Domain, EVEN, Field, IVP, ODD,
                        Parameter, RealFourier, Reflection, SymmetryBreakingError, UnaryFunction,
                        absolute, cos, diff, dt, exp, has_nonpolynomial, lap, log, sign, sin, sqrt,
                        tanh)
from conatus.ir.expr import NONPOLY
from conatus.reference import ReferenceEvaluator, evaluate
from conatus.solvers.parametric import laurent_exponents


def _box(Nx=16, Nz=20):
    cs = CartesianCoordinates("x", "z")
    x, z = cs.coords
    bx = RealFourier(x, Nx, (0, 4), dealias=3 / 2)
    bz = Chebyshev(z, Nz, (-1, 1), dealias=3 / 2)
    D = Domain(cs, [bx, bz])
    ev = ReferenceEvaluator(D)
    return cs, x, z, D, ev.grid(x), ev.grid(z)


FUNCS = [exp, log, sin, cos, tanh, sqrt, absolute]


def test_structural_linearization():
    cs, x, z, D, X, Z = _box()
    u, U, v = Field("u", D), Field("U", D), Field("v", D)
    assert exp(u).linearize({u: U}, {u: v}) == exp(U) * v
    assert sin(u).linearize({u: U}, {u: v}) == cos(U) * v
    assert cos(u).linearize({u: U}, {u: v}) == -sin(U) * v
    assert log(u).linearize({u: U}, {u: v}) == U ** -1 * v
    # chain rule through a derivative
    assert exp(diff(u, x)).linearize({u: U}, {u: v}) == exp(diff(U, x)) * diff(v, x)
    # a function of a fixed field has a zero derivative
    w = Field("w", D)
    assert exp(w).linearize({u: U}, {u: v}).is_zero()


@pytest.mark.parametrize("f", FUNCS, ids=lambda f: f.__name__)
def test_taylor_second_order(f):
    """|f(U + eps v) - f(U) - eps f'(U) v| = O(eps^2) on the grid."""
    cs, x, z, D, X, Z = _box()
    u, U, v = Field("u", D), Field("U", D), Field("v", D)
    F = f(u) * diff(u, x) + f(u)
    dF = F.linearize({u: U}, {u: v})
    Ud = 2 + 0.5 * np.sin(np.pi * X / 2) * (1 - Z ** 2)          # > 0: log, sqrt, abs smooth
    vd = np.cos(np.pi * X) * Z ** 3
    base, lin = evaluate(F, {u: Ud}), evaluate(dF, {U: Ud, v: vd})
    errs = [np.max(abs(evaluate(F, {u: Ud + eps * vd}) - base - eps * lin)) for eps in (1e-2, 5e-3)]
    assert 3.5 < errs[0] / errs[1] < 4.5, errs


def test_degrees_nonpolynomial_and_constant_folding():
    cs, x, z, D, X, Z = _box()
    u = Field("u", D)
    assert exp(u).degrees({u}) == frozenset({NONPOLY})
    assert not exp(u).is_linear_in({u})
    w = Field("w", D)
    assert exp(w).degrees({u}) == frozenset({0})
    c = exp(Constant(0.0))
    assert not isinstance(c, UnaryFunction) and c.value == 1.0
    with pytest.raises(ValueError):
        log(Constant(-1.0))


def test_parity_rules_match_numerics():
    cs, x, z, D, X, Z = _box(16, 12)
    Rx = Reflection(x, about=0)
    u = Field("u", D, symmetry={Rx: ODD})
    w = Field("w", D, symmetry={Rx: EVEN})
    data = {u: np.sin(np.pi * X / 2) * (1 + Z), w: 2 + np.cos(np.pi * X / 2) + Z ** 2}
    idx = (-np.arange(16)) % 16
    cases = {sin(u): ODD, tanh(u): ODD, cos(u): EVEN, absolute(u): EVEN, exp(w): EVEN,
             log(w): EVEN, sqrt(w): EVEN, sin(u) * exp(w): ODD}
    for e, p in cases.items():
        assert e.parity_under(Rx) == p, str(e)
        f = evaluate(e, data)
        assert np.max(abs(f[idx, :] - p * f)) < 1e-10, str(e)


def test_parity_refusals():
    cs, x, z, D, X, Z = _box()
    Rx = Reflection(x, about=0)
    u = Field("u", D, symmetry={Rx: ODD})
    for f in (exp, log, sqrt):
        with pytest.raises(SymmetryBreakingError, match="no parity"):
            f(u).parity_under(Rx)


def test_evaluator_refuses_non_finite():
    cs, x, z, D, X, Z = _box()
    u = Field("u", D)
    with pytest.raises(ValueError, match="non-finite"):
        evaluate(log(u), {u: Z})                  # Z <= 0 on part of the grid
    with pytest.raises(ValueError, match="non-finite"):
        evaluate(sqrt(u), {u: Z - 2})


def test_explicit_refusals():
    cs, x, z, D, X, Z = _box()
    u, U, v = Field("u", D), Field("U", D), Field("v", D)
    with pytest.raises(NotImplementedError, match="not differentiable"):
        sign(u).linearize({u: U}, {u: v})
    with pytest.raises((NotImplementedError, ValueError)):
        exp(u).transpose(u, v)
    uc = Field("uc", D, dtype=np.complex128)
    with pytest.raises(TypeError, match="holomorphic"):
        absolute(uc)
    with pytest.raises(TypeError, match="scalars"):
        from conatus.ir import VectorField
        exp(VectorField("q", D))


def test_nonpolynomial_is_declared():
    cs, x, z, D, X, Z = _box()
    u = Field("u", D)
    assert "not alias-free" in exp(u).explain()
    assert has_nonpolynomial(u * tanh(u) + exp(u)) == ["exp", "tanh"]
    pb = IVP([u], name="reaction-diffusion")
    pb.add_equation((dt(u) - lap(u), exp(u)))
    assert "NOT alias-free" in pb.explain()


def test_parametric_pencil_refuses_function_of_parameter():
    cs, x, z, D, X, Z = _box()
    u = Field("u", D)
    p = Parameter("p", 1.0)
    assert laurent_exponents(exp(p) * u, p) is None
    assert laurent_exponents(exp(u) * p, p) == frozenset({1})


def test_pseudospectral_rhs_with_exp():
    """u_t - u_xx = -exp(u), u(x, 0) = 0: spatially uniform, exact u = -log(1 + t)."""
    from conatus.discretize.plan import discretize
    from conatus.reference.evaluate import FieldData
    from conatus.timestep import IVPSolver
    cs = CartesianCoordinates("x")
    x = cs["x"]
    b = RealFourier(x, 16, (0, 2 * np.pi), dealias=3 / 2)
    D = Domain(cs, [b])
    u = Field("u", D)
    pb = IVP([u], name="exp decay")
    pb.add_equation((dt(u) - lap(u), -exp(u)))
    s = IVPSolver(discretize(pb), "RK443", 1e-3, startup="rk443")
    s.set_state({u: FieldData(u, np.zeros(16), "grid")})
    s.run(500)
    got = s.grid("u")
    assert np.max(abs(got - (-np.log(1.5)))) < 1e-10


def test_fused_gpu_rhs_refuses_pointwise_functions():
    """The fused (GPU) RHS does not support pointwise functions yet: explicit refusal."""
    from conatus.discretize.plan import discretize
    from conatus.timestep import IVPSolver
    from conatus.timestep.spectral import SpectralError
    cs = CartesianCoordinates("x")
    x = cs["x"]
    D = Domain(cs, [RealFourier(x, 16, (0, 2 * np.pi), dealias=3 / 2)])
    u = Field("u", D)
    pb = IVP([u], name="exp decay")
    pb.add_equation((dt(u) - lap(u), -exp(u)))
    with pytest.raises(SpectralError, match="UnaryFunction"):
        IVPSolver(discretize(pb), "RK443", 1e-3, startup="rk443", f_backend="fused")
