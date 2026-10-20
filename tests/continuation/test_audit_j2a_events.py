"""Audit located continuation events carry a converged flag; tolerances declared.

Written before the fix; failed on the code it guards.
"""
import numpy as np

def _cfg(**kw):
    from conatus.continuation import ContinuationSettings
    base = dict(ds=0.1, ds_min=1e-8, ds_max=0.3, adapt=1.5, fast_iterations=3, tolerance=1e-12,
                max_newton=15, max_points=60, dp_fd=None)
    base.update(kw)
    return ContinuationSettings(**base)


def _fold_branch(cfg):
    from conatus.continuation import PseudoArclength
    G = lambda x, p: np.array([x[0] ** 2 + p - 1.0])      # noqa: E731
    J = lambda x, p: np.array([[2 * x[0]]])               # noqa: E731
    pa = PseudoArclength(G, J, cfg, dGdp=lambda x, p: np.array([1.0]))
    return pa.run(np.array([-1.0]), 0.0, +1, p_bounds=(-0.5, 2.0))


def test_continuation_events_carry_converged_flag_and_declared_tolerances():
    """R4: event secants stopped on corrector failure or after 30 iterations and the last iterate
    was reported as 'located'; tolerances 1e-12 / 1e-11 / 1e-8 were hard-coded."""
    br = _fold_branch(_cfg())
    (f,) = br.folds()
    assert f.converged is True and abs(f.p - 1.0) < 1e-12
    txt = br.explain()
    for name in ("fold_tol", "crossing_tol", "hopf_imag_tol", "event_max_iterations"):
        assert name in txt, txt
    br1 = _fold_branch(_cfg(event_max_iterations=1, fold_tol=1e-15))
    (f1,) = br1.folds()
    assert f1.converged is False and "not converged" in f1.explain() and f1.reason


def test_continuation_crossing_not_converged_is_flagged():
    from conatus.continuation import PseudoArclength
    G = lambda x, p: np.array([x[0] - p])                 # noqa: E731
    J = lambda x, p: np.array([[1.0]])                    # noqa: E731
    Gp = lambda x, p: np.array([-1.0])                    # noqa: E731
    eig = lambda x, p: np.array([(p - 0.37) * (1 + (p - 0.37) ** 2), -5.0])  # noqa: E731
    ok = PseudoArclength(G, J, _cfg(), dGdp=Gp, marginal_tol=1e-14, eigen=eig)
    (e,) = ok.run(np.array([-1.0]), -1.0, +1, p_bounds=(-2, 1)).events
    assert e.converged is True and abs(e.p - 0.37) < 1e-10
    bad = PseudoArclength(G, J, _cfg(event_max_iterations=1), dGdp=Gp, marginal_tol=1e-14, eigen=eig)
    (e1,) = bad.run(np.array([-1.0]), -1.0, +1, p_bounds=(-2, 1)).events
    assert e1.converged is False and "budget" in e1.reason
