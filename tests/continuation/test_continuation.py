"""Pseudo-arclength continuation: folds, eigenvalue crossings, explicit settings, Bratu."""
import numpy as np
import pytest

from conatus.continuation import ContinuationSettings, PseudoArclength


def _cfg(**kw):
    base = dict(ds=0.1, ds_min=1e-8, ds_max=0.3, adapt=1.5, fast_iterations=3, tolerance=1e-12,
                max_newton=15, max_points=60, dp_fd=None)
    base.update(kw)
    return ContinuationSettings(**base)


def test_settings_are_required_and_checked():
    with pytest.raises(TypeError):
        ContinuationSettings(ds=0.1)                       # every setting is mandatory
    with pytest.raises(ValueError):
        _cfg(ds=1.0, ds_max=0.3)
    with pytest.raises(ValueError, match="dp_fd"):
        PseudoArclength(lambda x, p: x, lambda x, p: np.eye(1), _cfg())   # no G_p source


def test_fold_of_a_parabola_located():
    """G(x, p) = x^2 + p - 1: fold at p = 1 (x = 0), passed from the x < 0 side."""
    G = lambda x, p: np.array([x[0] ** 2 + p - 1.0])      # noqa: E731
    J = lambda x, p: np.array([[2 * x[0]]])               # noqa: E731
    pa = PseudoArclength(G, J, _cfg(), dGdp=lambda x, p: np.array([1.0]))
    br = pa.run(np.array([-1.0]), 0.0, +1, p_bounds=(-0.5, 2.0))
    folds = br.folds()
    assert len(folds) == 1 and abs(folds[0].p - 1.0) < 1e-12, br.explain()
    assert br.points[-1].x[0] > 0                           # came back on the other side
    assert all(q.residual < 1e-12 for q in br.points)


def test_fd_parameter_derivative_matches_analytic():
    G = lambda x, p: np.array([x[0] ** 2 + p - 1.0])      # noqa: E731
    J = lambda x, p: np.array([[2 * x[0]]])               # noqa: E731
    pa = PseudoArclength(G, J, _cfg(dp_fd=1e-6))
    br = pa.run(np.array([-1.0]), 0.0, +1, p_bounds=(-0.5, 2.0))
    assert abs(br.folds()[0].p - 1.0) < 1e-10
    assert "central FD, dp = 1e-06" in br.explain()


def test_hopf_and_real_crossings_are_counted():
    G = lambda x, p: np.array([x[0] - p])                 # noqa: E731  trivial branch x = p
    J = lambda x, p: np.array([[1.0]])                    # noqa: E731
    Gp = lambda x, p: np.array([-1.0])                    # noqa: E731
    hopf = PseudoArclength(G, J, _cfg(), dGdp=Gp, eigen=lambda x, p: np.array([p + 1j, p - 1j]),
                           marginal_tol=1e-14)
    br = hopf.run(np.array([-1.0]), -1.0, +1, p_bounds=(-2, 1))
    assert [e.kind for e in br.events] == ["hopf"], br.explain()
    assert abs(br.events[0].p - 0.0) < 1e-10 and "Im λ = 1" in br.events[0].detail  # located
    real = PseudoArclength(G, J, _cfg(), dGdp=Gp, marginal_tol=1e-14,
                           eigen=lambda x, p: np.array([(p - 0.37) * (1 + (p - 0.37) ** 2), -5.0]))
    br = real.run(np.array([-1.0]), -1.0, +1, p_bounds=(-2, 1))
    assert [e.kind for e in br.events] == ["real crossing"], br.explain()
    assert abs(br.events[0].p - 0.37) < 1e-10                                            # located


def test_unconverged_start_refused():
    G = lambda x, p: np.array([x[0] ** 2 + p - 1.0])      # noqa: E731
    J = lambda x, p: np.array([[2 * x[0]]])               # noqa: E731
    pa = PseudoArclength(G, J, _cfg(), dGdp=lambda x, p: np.array([1.0]))
    with pytest.raises(ValueError, match="not converged"):
        pa.run(np.array([-0.5]), 0.0)


@pytest.mark.slow
def test_bratu_branch_through_the_fold():
    """Bratu on [-1, 1] (exp node): lower branch -> fold lam_c = 0.8784576797812903 -> upper branch."""
    from conatus.continuation import NLBVPContinuation
    from conatus.reference import transforms as T
    from conatus.solvers.nlbvp import NewtonSolver
    from nlbvp.nlbvp_cases import bratu, bratu_exact, bratu_lambda_c
    # N = 64: up the upper branch exp(u) needs it (tail 3e-8 at N = 32, 8e-8 at N = 40 along the
    # branch: refused; tutorial 4 runs this branch at 64 modes, tail < 1e-12)
    pb, o = bratu(64, 0.3)
    s = NewtonSolver(pb, tolerance=1e-13, damping=1.0, max_iterations=20, params={"lam": 0.3})
    sol = s.solve({o["u"]: lambda x: 0 * x})
    cfg = _cfg(ds=0.2, ds_max=0.6, max_points=12, dp_fd=1e-6)
    c = NLBVPContinuation(s, "lam", cfg)
    br = c.run(sol.coeffs, 0.3, +1, p_bounds=(0.3, 1.0))
    lc = bratu_lambda_c()
    (fold,) = br.folds()
    assert abs(fold.p - lc) < 1e-10, br.explain()           # >= 10 digits
    x = T.grid(o["basis"], dealias=False)
    i_fold = fold.between[1]
    for k, q in enumerate(br.points):
        branch = "lower" if k < i_fold else "upper"
        ex, _ = bratu_exact(q.p, x, branch)
        u = T.backward(c.coeffs_of(q.x)["u"], o["basis"], dealias=False)
        assert np.max(abs(u - ex)) < 1e-6, (k, q.p, branch)


def test_audit_r3_marginal_modes_do_not_create_crossings():
    """A marginal eigenvalue that flickers at ±1e-15 must not be counted as a crossing."""
    G = lambda x, p: np.array([x[0] - p])                 # noqa: E731
    J = lambda x, p: np.array([[1.0]])                    # noqa: E731
    Gp = lambda x, p: np.array([-1.0])                    # noqa: E731
    flicker = lambda x, p: np.array([1e-15 * np.sign(np.sin(40 * p)), -3.0])   # noqa: E731
    with pytest.raises(TypeError):
        PseudoArclength(G, J, _cfg(), dGdp=Gp, eigen=flicker)          # marginal_tol required
    pa = PseudoArclength(G, J, _cfg(), dGdp=Gp, eigen=flicker, marginal_tol=1e-12)
    br = pa.run(np.array([-1.0]), -1.0, +1, p_bounds=(-2, 1))
    assert not br.events, br.explain()
    assert "marginal" in br.explain() or "marginal" in pa.explain_eigen()
