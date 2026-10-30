"""F777 (audit of 0.2.1, N61): EDMD, SPOD and uniform_dt refuse an unusable sampling interval.
Oracle: the rate log(2)/dt of the map y = 2 x."""
from _acceptance import check, passed, start

start(777, "sampling intervals of the data-driven methods")
import numpy as np  # noqa: E402

from conatus.datadriven import DataInnerProduct, edmd, monomials, spod, uniform_dt  # noqa: E402
from conatus.errors import OptionError  # noqa: E402

X = np.arange(1.0, 6.0)[:, None]


def refused(call):
    try:
        call()
    except OptionError:
        return True
    return False


for dt in (0.0, np.nan, np.inf, -1.0):
    check(refused(lambda dt=dt: edmd(X, 2 * X, dictionary=monomials(1, 1), rcond=1e-12, dt=dt)), f"edmd dt={dt}")
    check(refused(lambda dt=dt: spod(np.ones((1, 64)), dt=dt, nfft=8, noverlap=0, window="boxcar", mean="keep",
                                    onesided=True, inner_product=DataInnerProduct.euclidean(1), n_modes=1)),
          f"spod dt={dt}")
for t in ([0.0, np.inf], [-1e308, 1e308]):
    check(refused(lambda t=t: uniform_dt(np.array(t), rtol=1e-9)), f"uniform_dt {t}")
e = edmd(X, 2 * X, dictionary=monomials(1, 1), rcond=1e-12, dt=0.1)
got = e.omega[int(np.argmax(np.abs(e.eigenvalues)))].real
check(abs(got - np.log(2) / 0.1) < 1e-10, f"rate {got}")
passed("invalid intervals refused; rate log(2)/dt recovered")
