"""F737 (audit of 0.2.1, N17): a non-finite eigenvalue is no evidence of stability.

Oracle: the callback values are chosen by hand; [-1] and [1] are the controls.
"""
from _acceptance import check, passed, start

start(737, "non-finite stability evidence")
import warnings  # noqa: E402

import numpy as np  # noqa: E402


from conatus.continuation import ContinuationSettings, PseudoArclength  # noqa: E402


def run(ev):
    cfg = ContinuationSettings(ds=.1, ds_min=.01, ds_max=.1, tolerance=1e-10, max_points=2)
    c = PseudoArclength(lambda x, p: x - p, lambda x, p: np.eye(1), cfg, dGdp=lambda x, p: -np.ones(1),
                        eigen=lambda x, p: np.array(ev), marginal_tol=1e-8)
    return c.run(np.array([0.]), 0.)


for ev in ([np.nan], [-1., np.nan], [np.inf]):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = run(ev)
    check(all(p.unstable is None for p in r.points), f"{ev}: {[p.unstable for p in r.points]}")
    check(any("non-finite" in str(x.message) for x in w), "no warning")
check(all(p.unstable == 0 for p in run([-1.]).points), "stable control")
check(all(p.unstable == 1 for p in run([1.]).points), "unstable control")
passed("unknown count, warned")
