"""Audit of 0.1.2, F07: principal angles of complex covariant vectors use the Hermitian product.

Oracle: for unit u = (1, i)/sqrt(2), v = (1, 2i)/sqrt(5), |u^H v| = 3/sqrt(10), so the angle is
arccos(3/sqrt(10)) (NumPy vdot); multiplying a vector by a phase leaves it unchanged."""
import numpy as np

from conatus.chaos.lyapunov import _sine, clv_subspace_angles


def test_complex_pair_angle_is_hermitian_and_phase_invariant():
    V = np.array([[1, 1j], [1, 2j]]) / np.sqrt([[2], [5]])
    exact = np.arccos(abs(np.vdot(V[0], V[1])))
    assert np.isclose(exact, np.arccos(3 / np.sqrt(10)))
    assert np.isclose(clv_subspace_angles(V, 1), exact, rtol=1e-12)
    for th in (0.3, 1.7, np.pi / 2):
        W = V.copy()
        W[1] *= np.exp(1j * th)
        assert np.isclose(clv_subspace_angles(W, 1), exact, rtol=1e-12)
        assert np.isclose(float(_sine(np, W[0], W[1], axis=-1)), np.sin(exact), rtol=1e-12)


def test_real_vectors_unchanged():
    V = np.array([[1.0, 0.0], [1.0, 1.0]])
    assert np.isclose(clv_subspace_angles(V, 1), np.pi / 4, rtol=1e-12)
