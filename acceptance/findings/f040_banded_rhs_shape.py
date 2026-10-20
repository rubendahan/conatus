"""F040: a right-hand side of the wrong shape must be refused, not reshaped.

BorderedBanded of size 3, b of shape (2, 3) (six numbers, but not (3, nrhs)): 0.1.0 reshaped it
to (3, 2) and solved the wrong system.
"""
import numpy as np
import scipy.sparse as sp
from _acceptance import check, need, passed, refused, start

start(40, "banded solve checks the right-hand side shape")
need("conatus.runtime")
from conatus.runtime.banded import BorderedBanded  # noqa: E402

A = sp.diags([1.0, 2.0, 3.0]).tocsr()
s = BorderedBanded(A, border_rows=[])
b = np.arange(6.0).reshape(2, 3)
was_refused, x = refused(s.solve, b)
check(was_refused, f"b of shape {b.shape} accepted for n = 3; returned {np.asarray(x).tolist()}")
passed("refused")
