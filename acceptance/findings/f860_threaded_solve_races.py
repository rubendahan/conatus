"""F860: the threaded batched banded solves are race-free.

Oracle: the serial solve of the same batch (bit for bit), and a dense LAPACK solve of each
system (numpy.linalg.solve) to 1e-12. Several Python threads solve one FRESH batch at once, with
the GIL released inside each call and each call itself split over threads; the first transposed
solve builds the transposed border data of every system, the only state a solve writes.
"""
import threading

import numpy as np
import scipy.sparse as sp
from _acceptance import check, passed, start

start(860, "threaded batched solves: concurrent callers get the serial bits")
from conatus.runtime import BorderedBandedBatch  # noqa: E402

rng = np.random.default_rng(860)
n, B = 120, 40
mats = []
for _ in range(B):
    A = sp.diags([rng.standard_normal(n - abs(o)) + (20.0 if o == 0 else 0.0) for o in range(-4, 5)],
                 list(range(-4, 5)), shape=(n, n), format="lil")
    for i in (0, 1):
        A[i, :] = rng.standard_normal(n)
    mats.append(sp.csr_matrix(A))
b = rng.standard_normal((B, n))
for transpose in (True, False):
    ref = BorderedBandedBatch(mats, [0, 1])._s.solve(b, transpose)
    dense = np.array([np.linalg.solve(m.toarray().T if transpose else m.toarray(), bk)
                      for m, bk in zip(mats, b, strict=True)])
    check(np.max(np.abs(ref - dense)) < 1e-12 * np.max(np.abs(dense)) * n,
          f"serial solve vs dense oracle (transpose={transpose})")
    fresh = BorderedBandedBatch(mats, [0, 1])
    outs = [None] * 8
    gate = threading.Barrier(8)

    def work(i, fresh=fresh, transpose=transpose, outs=outs, gate=gate):
        gate.wait()
        outs[i] = [fresh._s.solve(b, transpose, 1 + i % 5).tobytes() for _ in range(4)]

    ths = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    bad = sum(o != ref.tobytes() for out in outs for o in out)
    check(bad == 0, f"{bad} of 32 concurrent solves differ from the serial bits (transpose={transpose})")
passed("8 Python threads x 4 calls x up to 5 core threads: serial bits, dense oracle to 1e-12")
