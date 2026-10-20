"""Finding 93: the low-level optimal response was normalised instead of
propagated (u_resp must be Φ v_opt, of energy G)."""
import numpy as np

from conatus.growth.propagator import gain


def test_response_is_the_propagated_optimal_state():
    rng = np.random.default_rng(2)
    Phi = rng.standard_normal((5, 5))
    F = np.diag([1.0, 2.0, 0.5, 1.5, 1.0])
    r = gain(Phi, F, None)
    np.testing.assert_allclose(r.u_resp, Phi @ r.v_opt, atol=1e-13)
    assert abs(np.linalg.norm(F @ r.v_opt) - 1) < 1e-12
    assert abs(np.linalg.norm(F @ r.u_resp) ** 2 - r.G) < 1e-10 * r.G
