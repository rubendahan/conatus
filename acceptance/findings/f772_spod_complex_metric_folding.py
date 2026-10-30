"""F772 (audit of 0.2.1, N09): one-sided SPOD with a complex Gram matrix is refused, and a real
metric keeps the time-domain energy. Oracle: the direct time-domain energy."""
from _acceptance import check, passed, start

start(772, "one-sided SPOD folding with complex and real Gram matrices")
import numpy as np  # noqa: E402

from conatus.datadriven import DataInnerProduct, spod  # noqa: E402
from conatus.errors import OptionError  # noqa: E402

Q = np.array([[1.0, 0, 0, 0, 0, 0, 0, 0], [0.0, 1, 0, 0, 0, 0, 0, 0]])
kw = dict(dt=1.0, nfft=8, noverlap=0, window="boxcar", mean="keep", n_modes=1)
Wc = np.array([[1.0, 0.75j], [-0.75j, 1.0]])
try:
    spod(Q, onesided=True, inner_product=DataInnerProduct.gram(Wc), **kw)
    check(False, "the complex metric was accepted")
except OptionError:
    pass
want = np.trace(Q.T @ Wc @ Q).real / 8
two = spod(Q, onesided=False, inner_product=DataInnerProduct.gram(Wc), **kw).eigenvalues.sum() / 8
check(abs(two - want) < 1e-13, f"two-sided energy {two}, direct {want}")
Wr = np.array([[1.0, 0.5], [0.5, 2.0]])
one = spod(Q, onesided=True, inner_product=DataInnerProduct.gram(Wr), **kw).eigenvalues.sum() / 8
want = np.trace(Q.T @ Wr @ Q) / 8
check(abs(one - want) < 1e-13, f"one-sided energy {one}, direct {want}")
passed("complex metric refused; energies agree with the time-domain values")
