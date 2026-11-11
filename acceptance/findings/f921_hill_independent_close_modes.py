"""F921 (GPT pass 2-3, G1): Hill keeps independent eigendirections with close exponents.

Oracle: the diagonal system x' = r x, y' = -r y (period 2 pi, no coupling) has exponents +r and
-r with eigenvectors e_x, e_y. For r = 1e-6, 1e-9, 1e-12 both must be returned, in every call,
with a deterministic value. Control: a doubly degenerate exponent (x' = y' = -x... two copies of
r) is returned twice; copies mu + i n of one exponent count once.
"""
import numpy as np
from _acceptance import check, passed, start

start(921, "Hill keeps independent modes at close exponents, deterministically")
from conatus.growth.periodic import PeriodicOperator, operator_floquet  # noqa: E402


def op(L0, harmonics=None):
    n = L0.shape[0]
    return PeriodicOperator(M=np.eye(n), L0=L0, harmonics=harmonics or {},
                            cols=[(f"x{i}", ()) for i in range(n)], state_cols=np.arange(n),
                            other_cols=np.array([], int), W=np.eye(n), period=2 * np.pi,
                            fields={f"x{i}": 1.0 for i in range(n)}, label="diagonal",
                            rank_tol=1e-12)


for rate in (1e-6, 1e-9, 1e-12):
    o = op(-np.diag([-rate, rate]))
    runs = [operator_floquet(o, method="hill", Nh=2, targets=[-1.1 * rate, 1.1 * rate],
                             n_modes=1, tol=1e-12) for _ in range(6)]
    for r in runs:
        got = np.sort(r.mu.real)
        check(got.size == 2 and abs(got[0] + rate) <= 1e-3 * rate and abs(got[1] - rate) <= 1e-3 * rate,
              f"rate {rate:g}: {r.mu}, oracle [-{rate:g}, {rate:g}]")
    check(all(np.array_equal(r.mu, runs[0].mu) for r in runs), f"rate {rate:g}: varies between calls")
    check(abs(runs[0].leading.real - rate) <= 1e-3 * rate, f"rate {rate:g}: leading {runs[0].leading}")

# a true double exponent (two uncoupled copies of x' = 0.3 x) and a spectator at -5
o = op(-np.diag([0.3, 0.3, -5.0]))
r = operator_floquet(o, method="hill", Nh=3, targets=[0.0], n_modes=2, tol=1e-12)
got = np.sort(r.mu.real)[::-1]
check(got.size >= 2 and np.allclose(got[:2], [0.3, 0.3], rtol=0, atol=1e-10), f"double exponent: {r.mu}")
check(int(np.sum(np.abs(r.mu - 0.3) < 1e-8)) == 2, f"double exponent counted {r.mu}")
passed("+-r kept for r = 1e-6 .. 1e-12 in every call; double exponent kept twice")
