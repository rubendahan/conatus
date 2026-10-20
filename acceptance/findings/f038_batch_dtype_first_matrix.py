"""F038: BorderedBandedBatch takes its dtype from every matrix, not the first one only.

A batch [I, 1j*I]: 0.1.0 built real factors from the first matrix and dropped the imaginary part
of the second. Expected: complex factors (solve of b = 1 gives x = 1 and -1j), or a refusal.
"""
import numpy as np
import scipy.sparse as sp
from _acceptance import check, need, passed, refused, start

start(38, "batch dtype is the common dtype of all matrices")
need("conatus.runtime")
from conatus.runtime.banded import BorderedBandedBatch  # noqa: E402

n = 4
mats = [sp.identity(n, format="csr"), 1j * sp.identity(n, format="csr")]
was_refused, batch = refused(BorderedBandedBatch, mats, border_rows=[])
if was_refused:
    passed("mixed real/complex batch refused")
x = np.asarray(batch.solve(np.ones((2, n), dtype=complex)))
expected = np.array([np.ones(n), -1j * np.ones(n)])
err = float(np.max(np.abs(x - expected)))
check(err < 1e-12, f"dtype {np.dtype(batch.dtype).name}; x[1] = {x[1]}, expected -1j (error {err:.3g})")
passed(f"complex batch, error {err:.2g}")
