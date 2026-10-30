"""F775 (audit of 0.2.1, N56): integer snapshots do not wrap in the Gram products of POD and EDMD.
Oracle: POD energies 13.25 s^2, 1.75 s^2 from the hand eigenvalues 53, 7; EDMD eigenvalue 2."""
from _acceptance import check, passed, start

start(775, "integer data in POD and EDMD")
import numpy as np  # noqa: E402

from conatus.datadriven import DataInnerProduct, Dictionary, edmd, pod  # noqa: E402

B = np.array([[1, 2, 3, 4], [4, 1, 3, 2]])
for dtype, s in ((np.int16, 100), (np.int32, 10 ** 4), (np.int64, 10 ** 10)):
    r = pod((s * B).astype(dtype), inner_product=DataInnerProduct.euclidean(2), rank=2,
            method="snapshots", mean="keep")
    check(np.allclose(r.energies / float(s) ** 2, [13.25, 1.75], rtol=1e-12), f"{dtype}: {r.energies}")
x = np.arange(1, 10)[:, None]
for s in (2 * 10 ** 8, 10 ** 9):
    e = edmd(s * x, 2 * s * x, dictionary=Dictionary(lambda v: v, ["x"], [0]), rcond=1e-12, dt=None)
    check(abs(e.eigenvalues[0] - 2) < 1e-10, f"s = {s}: mu = {e.eigenvalues[0]}")
passed("analytic POD energies and EDMD eigenvalue")
