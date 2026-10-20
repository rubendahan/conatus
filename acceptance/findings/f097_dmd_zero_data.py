"""F097: DMD of all-zero snapshots must be refused up front, not fail late with NaN.

0.1.0 produced NaN then a LinAlgError deep in the fit. Expected: a typed refusal (no
information in the data).
"""
import warnings

import numpy as np
from _acceptance import check, passed, refused, start

start(97, "DMD refuses data without information")
from conatus.datadriven import DataInnerProduct, dmd  # noqa: E402

X = np.zeros((6, 5))
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    was_refused, d = refused(dmd, X, dt=0.1, rank=2, inner_product=DataInnerProduct.quadrature(np.ones(6)))
runtime = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
check(was_refused, f"DMD of zero data returned {type(d).__name__}")
check(not runtime, f"refused, but only after NaN warnings: {runtime[:2]}")
passed("refused up front")
