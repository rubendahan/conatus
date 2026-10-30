"""F771 (audit of 0.2.1, N01): the chi-square interval of SPOD is calibrated for one entry and
refused for several, where the ordered eigenvalues follow no scalar law.

Oracle: repeated independent datasets of known population density."""
from _acceptance import check, passed, start

start(771, "SPOD confidence holds for its scalar domain and is refused outside it")
import numpy as np  # noqa: E402

from conatus.datadriven import DataInnerProduct, spod  # noqa: E402
from conatus.errors import OptionError  # noqa: E402

rng = np.random.default_rng(771)


def run(Q, **kw):
    opts = dict(window="boxcar", onesided=True) | kw
    return spod(Q, dt=1.0, nfft=8, noverlap=0, mean="keep", n_modes=1,
                inner_product=DataInnerProduct.euclidean(Q.shape[0]), **opts)


impulse = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
hits = np.zeros(3)
trials = 1500
for _ in range(trials):
    lo, hi = run(rng.normal(size=(1, 128)), window=impulse).confidence(0.05)
    hits[0] += lo[1, 0] <= 2.0 <= hi[1, 0]
    lo, hi = run(rng.normal(size=(1, 128)).astype(complex), onesided=False).confidence(0.05)
    hits[1] += lo[0, 0] <= 1.0 <= hi[0, 0]
    lo, hi = run(rng.normal(size=(1, 128))).confidence(0.05)
    hits[2] += lo[1, 0] <= 2.0 <= hi[1, 0]
cov = hits / trials
check(bool(np.all(np.abs(cov - 0.95) < 0.025)), f"impulse / cast complex / boxcar coverage {cov}")
try:
    run(rng.normal(size=(8, 128))).confidence(0.05)
    check(False, "eight channels were not refused")
except OptionError as err:
    check("approximate=True" in str(err), str(err))
passed(f"scalar coverage {cov}; n = 8 refused")
