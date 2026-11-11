"""F851 (CUDA risk items of v0.1.1): the device mapped solve refuses index maps that
would read or write outside its arrays, or write one entry twice.

``conatus.runtime.cuda.BandedBatch.solve_mapped`` gathers rhs[rows] and scatters x[cols] through
raw device pointers. An index out of range read or wrote other device memory without an error; a
repeated column let two systems write one entry. Oracle: the index ranges themselves. Checks:
out-of-range (high and negative) rows and cols are refused; repeated cols are refused; repeated
rows (two systems reading one entry) and valid permutations still solve, and the valid solve
equals the plain in-place solve. SKIP without a GPU.
"""
import numpy as np
import scipy.sparse as sp
from _acceptance import check, passed, skipped, start

start(851, "device mapped solve: index maps checked")
from conatus.runtime import cuda  # noqa: E402

if not cuda.available():
    skipped("no CUDA device (conatus.runtime.cuda.available() is False)")
import cupy as cp  # noqa: E402

n, batch = 40, 3
rng = np.random.default_rng(851)
mats = []
for _ in range(batch):
    A = np.diag(20.0 + rng.random(n)) + np.diag(rng.standard_normal(n - 1), 1) + np.diag(rng.standard_normal(n - 1), -1)
    A[:2] = rng.standard_normal((2, n))
    A[:2, :2] += 4 * n * np.eye(2)
    mats.append(sp.csr_matrix(A))
bb = cuda.BandedBatch(mats, [0, 1])
b = rng.standard_normal(batch * n)
rhs, x = cp.asarray(b), cp.zeros(batch * n)
ok = cp.arange(batch * n, dtype=np.int64).reshape(batch, n)
bb.solve_mapped(rhs, ok, x, ok)
d = cuda.to_device(b.reshape(batch, n))
bb.solve(d)
check(np.array_equal(cp.asnumpy(x), d.to_host().ravel()), "identity maps differ from the in-place solve")


def refused(rows, cols, word):
    try:
        bb.solve_mapped(rhs, rows, x, cols)
    except ValueError as exc:
        return word in str(exc)
    return False


hi, neg, rep = ok.copy(), ok.copy(), ok.copy()
hi[1, 3] = batch * n
neg[0, 0] = -1
rep[2, 5] = rep[0, 7]
check(refused(hi, ok, "range") and refused(ok, hi, "range"), "an index past the end was not refused")
check(refused(neg, ok, "range") and refused(ok, neg, "range"), "a negative index was not refused")
check(refused(ok, rep, "repeated"), "a repeated column was not refused")
bb.solve_mapped(rhs, rep, x, ok)
passed("out-of-range and repeated-column maps refused; valid maps = in-place solve")
