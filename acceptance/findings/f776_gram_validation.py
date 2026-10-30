"""F776 (audit of 0.2.1, N57): gram(W) refuses non-finite and non-Hermitian matrices at any scale
and keeps valid ones. Oracle: the matrix itself, W = F^H F."""
from _acceptance import check, passed, start

start(776, "validation of a Gram matrix")
import numpy as np  # noqa: E402

from conatus.datadriven import DataInnerProduct  # noqa: E402
from conatus.errors import OptionError  # noqa: E402


def refused(W):
    try:
        DataInnerProduct.gram(W)
    except OptionError:
        return True
    return False


for bad in (np.nan, np.inf, -np.inf):
    check(refused(np.array([[2.0, bad], [0.0, 2.0]])), f"entry {bad} accepted")
for s in (1e-200, 1.0, 1e200):
    check(refused(s * np.array([[2.0, 1.0], [0.0, 2.0]])), f"non-Hermitian at scale {s} accepted")
    W = np.array([[2.0, 0.5], [0.5, 3.0]])
    F = DataInnerProduct.gram(s * W).F
    check(np.abs((F.T / np.sqrt(s)) @ (F / np.sqrt(s)) - W).max() < 1e-14, f"factor at scale {s}")
passed("non-finite and non-Hermitian refused; valid metrics kept at scales 1e-200 to 1e200")
