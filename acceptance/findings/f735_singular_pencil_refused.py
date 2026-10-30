"""F735 (audit of 0.2.1, N08): a pencil singular for every s has no spectrum.

Oracle: det(L + s M) = 0 for all s by construction (also after orthogonal transforms), and the regular control L = -I, M = diag(1, 0), det = 1 - s.
"""
from _acceptance import check, passed, start

start(735, "singular pencils refused, regular descriptors kept")
import numpy as np  # noqa: E402


from conatus.errors import RankError  # noqa: E402
from conatus.expert import solve_dense  # noqa: E402

rng = np.random.default_rng(0)
Q1, _ = np.linalg.qr(rng.standard_normal((4, 4)))
Q2, _ = np.linalg.qr(rng.standard_normal((4, 4)))
L = np.diag([-1., 0., -2., -3.])
M = np.diag([1., 0., 1., 1.])
for a, b, name in ((L, M, "diagonal"), (Q1 @ L @ Q2, Q1 @ M @ Q2, "transformed"),
                   (np.diag([-1., 0.]), np.diag([1., 0.]), "2x2")):
    try:
        r = solve_dense(a, b)
        check(False, f"{name}: returned {r.values}")
    except RankError:
        pass
r = solve_dense(-np.eye(2), np.diag([1., 0.]))
check(r.values.size == 1 and abs(r.values[0] - 1) < 1e-13 and r.n_infinite == 1, "regular control")
A = Q1 @ np.diag([-1., -1., -2., -3.]) @ Q2
B = Q1 @ np.diag([1., 0., 1., 1.]) @ Q2
r = solve_dense(A, B)
check(r.n_infinite == 1 and np.allclose(np.sort(r.values.real), [1, 2, 3], atol=1e-10), "regular transformed")
from conatus.nonmodal.pencil import reduce_pencil  # noqa: E402
try:
    reduce_pencil(np.diag([1., 0.]), np.diag([1., 0.]), np.arange(2), np.eye(2), infinite_tol=1e-10)
    check(False, "descriptor reduction accepted a singular pencil")
except RankError:
    pass
passed("refused dense and transformed; regular kept")
