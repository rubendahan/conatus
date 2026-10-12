"""Full-space steady problem of plane Couette ( S vs full): the translation
generators ∂x, ∂z on the state (StateTranslation) and the phase-condition system of
AffineSteadyProblem (unknowns x + speeds c, rows ⟨D X_ref, X⟩)."""
import numpy as np
import pytest

from couette_cases import BOX_GHC, couette_pt, shift_reflect_group
from conatus.ecs.steady import AffineSteadyProblem
from conatus.ecs.subspace import StateTranslation
from conatus.timestep import IVPSolver
from conatus.timestep.system import GlobalSystem

LX, LZ = BOX_GHC


def _fields(rng):
    c = rng.standard_normal((2, 3, 3, 2))

    def g(X, Z, k, d=None):
        out = 0 * X
        for m in range(3):
            for n in range(3):
                kx, kz = 2 * np.pi * m / LX, 2 * np.pi * n / LZ
                th = kx * X + kz * Z
                a, b = c[k, m, n]
                if d is None:
                    out = out + a * np.cos(th) + b * np.sin(th)
                else:                              # derivative along x (d = 0) or z (d = 1)
                    kk = kx if d == 0 else kz
                    out = out + kk * (-a * np.sin(th) + b * np.cos(th))
        return out
    return g


def _state(s, f, g, d=None):
    s.set_state({f["phi"]: lambda X, Y, Z: 0.02 * (1 - Y**2) ** 2 * g(X, Z, 0, d),
                 f["psi"]: lambda X, Y, Z: 0.05 * (1 - Y**2) * (1 + Y) * g(X, Z, 1, d),
                 f["U"]: lambda X, Y, Z: (0 if d is not None else 1) * (Y + 0.1 * (1 - Y**2)) + 0 * X,
                 f["W"]: lambda X, Y, Z: 0 * X})
    return s.X.copy()


@pytest.mark.slow                   # ≈ 10 s on the login node
def test_state_translation_is_the_derivative_and_commutes_with_the_step():
    rng = np.random.default_rng(3)
    plan, f = couette_pt(8, 13, 8, 400.0, BOX_GHC)
    s = IVPSolver(plan, "RK443", 0.02, f_backend="fused", decouple=True)
    G = shift_reflect_group(s)
    fe = G.field_entry.astype(float)
    g = _fields(rng)
    X = _state(s, f, g)
    for d, coord in enumerate(("x", "z")):
        D = StateTranslation(s, coord)
        ref = _state(s, f, g, d)                    # coefficients of the exact derivative
        assert np.linalg.norm((D.apply(X) - ref) * fe) < 1e-12 * np.linalg.norm(ref * fe)
        assert abs(D.apply(X) @ X) < 1e-14 * np.linalg.norm(X) ** 2      # D antisymmetric
        assert not np.any(D.apply(X) * (1 - fe))                        # taus untouched

        # the discrete step commutes with the translation: Φ'(X) D X = D Φ(X)
        def fm(X0):
            s.X, s.t = X0.copy(), 0.0
            s.run(2)
            return s.X.copy()
        eps = 1e-4
        lhs = (fm(X + eps * D.apply(X)) - fm(X - eps * D.apply(X))) / (2 * eps)
        rhs = D.apply(fm(X))
        assert np.linalg.norm((lhs - rhs) * fe) < 1e-8 * np.linalg.norm(rhs * fe)
    with pytest.raises(ValueError, match="not a RealFourier"):
        StateTranslation(s, "y")


@pytest.mark.slow                   # ≈ 5 s: three compiled systems
def test_full_space_phase_condition_system():
    N = (6, 13, 6)
    plan, f = couette_pt(*N, 300.0, BOX_GHC)
    s = IVPSolver(plan, "RK443", 0.02, f_backend="fused")
    sys2 = GlobalSystem(couette_pt(*N, 200.0, BOX_GHC)[0])
    sys3 = GlobalSystem(couette_pt(*N, 450.0, BOX_GHC)[0])
    sP = IVPSolver(couette_pt(*N, 300.0, BOX_GHC, linear_about="couette")[0], "RK443", 0.02)
    Ds = [StateTranslation(s, "x"), StateTranslation(s, "z")]
    with pytest.raises(ValueError, match="full space"):
        AffineSteadyProblem(s, 300.0, sys2, 200.0, sys3, 450.0, group=shift_reflect_group(s),
                            affine_tol=1e-12, preconditioner=sP, translations=Ds)
    pb = AffineSteadyProblem(s, 300.0, sys2, 200.0, sys3, 450.0, affine_tol=1e-12,
                             preconditioner=sP, translations=Ds)
    assert pb.n_x == pb.n and pb.m == 2
    X = _state(s, f, _fields(np.random.default_rng(5)))
    pb.set_phase_reference(X)
    x = pb.active(X)
    assert x.size == pb.n + 2 and not np.any(pb.speeds(x))
    r = pb.residual(x, 377.0)
    assert np.max(np.abs(r[-2:])) < 1e-14                  # ⟨D X, X⟩ = 0 at the reference
    rng = np.random.default_rng(6)
    x1 = x.copy()
    x1[-2:] = [0.03, -0.02]                                # non-zero speeds
    v = rng.standard_normal(x.size) * np.exp(-np.arange(x.size) / x.size)
    h = 1e-3
    fd = (pb.residual(x1 + h * v, 377.0) - pb.residual(x1 - h * v, 377.0)) / (2 * h)
    Jv = pb.jvp(x1, v, 377.0)
    assert np.linalg.norm(Jv - fd) < 1e-8 * np.linalg.norm(fd)   # quadratic in (x, c): exact
    dp = 1e-3
    fdp = (pb.residual(x1, 377.0 + dp) - pb.residual(x1, 377.0 - dp)) / (2 * dp)
    assert np.linalg.norm(pb.dRdp(x1, 377.0) - fdp) < 1e-6 * np.linalg.norm(fdp)
    # the reported residual ignores the speeds (a traveling wave is never accepted silently)
    assert pb.full_residual(x1, 377.0) == pb.full_residual(x, 377.0)
    assert "phase conditions" in pb.explain()
