"""F011 (important): the Hill solver handles a regular, well-conditioned matrix.

Blocks A0 = 1, A(+-1) = 1, Nh = 1, at s = 1j: det = -2j, cond 4.98. 0.1.0 returned NaN from
slogdet and solve (no block pivoting). Expected: slogdet = (phase, log 2) with |phase| = 1, a
finite solve with a small residual, or a typed refusal (never a NaN).
"""
import numpy as np
from _acceptance import check, need, passed, refused, start

start(11, "Hill block solve pivots (or refuses), never returns NaN")
need("conatus.floquet.hill_general")
from conatus.floquet.hill_general import hill_from_blocks  # noqa: E402

h = hill_from_blocks(np.eye(1), np.zeros((1, 1)), {1: np.ones((1, 1)), -1: np.ones((1, 1))},
                     Nh=1, omega=1.0, label="audit", note="")
A = h.dense(1j)
det = np.linalg.det(A)
was_refused, sl = refused(h.slogdet, 1j)
if was_refused:
    passed("slogdet refused")
phase, logabs = sl
check(np.isfinite(phase) and np.isfinite(logabs), f"slogdet = {sl} (det = {det})")
check(abs(logabs - np.log(abs(det))) < 1e-10, f"log|det| = {logabs}, expected {np.log(abs(det))}")
check(abs(phase * np.exp(logabs) - det) < 1e-10, f"phase*exp(logabs) = {phase * np.exp(logabs)}, det {det}")
b = np.ones((3, 1))
was_refused, x = refused(h.solve, 1j, b)
if was_refused:
    passed("slogdet correct, solve refused")
x = np.asarray(x)
check(np.isfinite(x).all(), "solve returned non-finite values")
res = float(np.max(np.abs(A @ x.reshape(3, -1) - b)))
check(res < 1e-12, f"solve residual {res:.3g}")
passed(f"slogdet ({phase:.3g}, {logabs:.6g}), solve residual {res:.2g}")
