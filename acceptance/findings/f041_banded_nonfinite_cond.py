"""F041: a non-finite matrix must not report a capacitance condition number of 0.

A bordered system with a NaN in the band: 0.1.0 factorised it and reported cond_1 = 0 (looks
perfectly conditioned). Expected: a refusal, or a non-finite / infinite condition estimate.
"""
import numpy as np
import scipy.sparse as sp
from _acceptance import check, need, passed, refused, start

start(41, "non-finite matrices are refused or flagged")
need("conatus.runtime")
from conatus.runtime.banded import BorderedBanded  # noqa: E402

n = 6
A = sp.diags([np.ones(n - 1), 2 * np.ones(n), np.ones(n - 1)], [-1, 0, 1]).tolil()
A[0, :] = 1.0                          # a dense boundary row: the border
A[3, 3] = np.nan
was_refused, s = refused(BorderedBanded, A.tocsr(), border_rows=[0])
if was_refused:
    passed("non-finite matrix refused")
cond = s.capacitance_cond
check(not (np.isfinite(cond) and cond < 1e12), f"capacitance cond_1 = {cond} for a matrix with NaN")
passed(f"cond_1 = {cond}")
