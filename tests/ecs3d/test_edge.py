"""Edge tracking by iterated bisection on a model with a saddle: x' = x (x - 1/2)(1 - x)
has a laminar attractor x = 0, a 'turbulent' attractor x = 1 and the edge state x = 1/2 (one
unstable direction); y' = -y is stable. The tracker converges to (1/2, 0) from anywhere."""
import numpy as np
import pytest

from conatus.ecs.edge import EdgeTracker


def _advance(X, T, n=20):
    h = T / n
    f = lambda z: np.array([z[0] * (z[0] - 0.5) * (1 - z[0]), -z[1]])  # noqa: E731
    for _ in range(n):
        k1 = f(X)
        k2 = f(X + h / 2 * k1)
        k3 = f(X + h / 2 * k2)
        k4 = f(X + h * k3)
        X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return X


def test_edge_tracker_converges_to_the_saddle():
    et = EdgeTracker(_advance, lambda X: -1 if X[0] < 0.1 else (1 if X[0] > 0.9 else 0),
                     lambda X: {"x": float(X[0]), "y": float(X[1])}, np.linalg.norm,
                     np.zeros(2), t_check=1.0, t_max=400.0, t_min=0.0, bisection_tol=1e-8,
                     separation=1e-2)
    et.run(np.array([0.05, 0.8]), np.array([0.97, -0.6]), rounds=4)
    # the pair leaves the saddle only at the declared separation (relative 1e-2): the edge
    # trajectory passes within the bisection accuracy of it, and the stable direction has decayed
    assert min(abs(r["x"] - 0.5) for r in et.trajectory) < 1e-6
    assert abs(et.trajectory[-1]["y"]) < 1e-3 and abs(et.trajectory[-1]["x"] - 0.5) < 1e-2
    assert et.bracket_check["lo"] == -1 and et.bracket_check["hi"] == 1


def test_edge_tracker_refuses_a_segment_that_is_not_a_bracket():
    et = EdgeTracker(_advance, lambda X: -1 if X[0] < 0.1 else (1 if X[0] > 0.9 else 0),
                     lambda X: {"x": float(X[0])}, np.linalg.norm, np.zeros(2), t_check=1.0,
                     t_max=400.0, t_min=0.0, bisection_tol=1e-8, separation=1e-2)
    with pytest.raises(ValueError, match="not a bracket"):
        et.run(np.array([0.05, 0.8]), np.array([0.3, 0.0]), rounds=1)   # both relaminarize


def test_threshold_bisection_brackets_a_known_threshold_and_reuses_known_fates():
    """ThresholdBisection: bracket ]lo, hi] of the amplitude threshold (here 1.2345) to rtol,
    known fates (finished runs) reused, non-monotone or undecided fates refused."""
    from conatus.ecs import ThresholdBisection, UndecidedTrajectory
    calls = []

    def fate(a):
        calls.append(a)
        return 1 if a > 1.2345 else -1
    tb = ThresholdBisection(fate, rtol=1e-4, known={1.0: -1, 2.0: 1, 1.3: 1})
    lo, hi = tb.run(1.0, 2.0)
    assert lo < 1.2345 < hi and (hi - lo) / hi <= 1e-4
    assert 2.0 not in calls and 1.0 not in calls and all(a < 1.3 for a in calls)   # known reused
    assert "threshold in ]" in tb.explain() and "3 known fates" in tb.explain()
    with pytest.raises(ValueError, match="not monotone"):
        ThresholdBisection(fate, rtol=1e-3, known={1.0: 1, 1.5: -1})
    with pytest.raises(UndecidedTrajectory):
        ThresholdBisection(lambda a: 0, rtol=1e-3).run(1.0, 2.0)
    with pytest.raises(ValueError, match="relaminarizes"):
        ThresholdBisection(fate, rtol=1e-3).run(1.0, 1.1)
    with pytest.raises(ValueError, match="transitions"):
        ThresholdBisection(fate, rtol=1e-3).run(1.3, 2.0)
    with pytest.raises(UndecidedTrajectory):                  # 0.7 is not rounded to a fate
        ThresholdBisection(lambda a: 0.7, rtol=1e-3).run(1.0, 2.0)
    # a refused (non-monotone) known fate leaves nothing behind
    tb2 = ThresholdBisection(fate, rtol=1e-3, known={1.0: -1})
    with pytest.raises(ValueError):
        tb2._record(0.5, 1, source="known")
    assert tb2.fates == {1.0: -1}


def test_threshold_bisection_log_scale_known_fates_outside_and_float_resolution():
    """Log cuts; known fates outside [lo, hi] are ignored for the ends; a tiny rtol ends at float
    resolution instead of looping; ends settled by known fates are never re-run."""
    from conatus.ecs import ThresholdBisection
    calls = []

    def fate(a):
        calls.append(a)
        return 1 if a > 1.2345e-7 else -1
    tb = ThresholdBisection(fate, rtol=1e-6, scale="log", known={1e-8: -1, 1e-6: 1})
    lo, hi = tb.run(1e-7, 2e-7)
    assert lo < 1.2345e-7 < hi and calls[:2] == [1e-7, 2e-7]      # outside knowns: ends run
    tb = ThresholdBisection(fate, rtol=1e-14, known={1.2e-7: -1, 1.25e-7: 1})
    calls.clear()
    lo, hi = tb.run(1e-7, 2e-7)
    assert 1e-7 not in calls and 2e-7 not in calls                 # both ends settled by knowns
    assert lo < 1.2345e-7 < hi and (hi - lo) / hi < 1e-13
    with pytest.raises(ValueError, match="rtol"):
        ThresholdBisection(fate, rtol=1e-17)
