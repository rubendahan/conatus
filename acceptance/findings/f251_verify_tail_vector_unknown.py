"""F251 (audit "assurances", 2026-10-20): verify()'s spectral-tail check measures every component.

The pencil columns of a vector unknown hold its components one after the other; the check took
the last 4 entries of the whole unknown (the top of its last component only), so an unresolved
first component (tail 0.24) passed as 1e-11. A pencil with nothing to measure passed with 0.
"""
from types import SimpleNamespace

import numpy as np
from _acceptance import check, passed, start

start(251, "verify() spectral tail per component, never a pass on nothing")
from conatus.verify import verify  # noqa: E402

n = 8
cols = [("u", (0,), "var")] * n + [("u", (1,), "var")] * n
x = np.concatenate([0.7 ** np.arange(n), 1e-3 * 0.01 ** np.arange(n)])


def pevp(cols):
    # L + sigma M = 0 for sigma = -1: x is an exact eigenvector, only the tail is under test
    # (verify() measures the residual on the pencil since 0.1.2, review F4)
    return SimpleNamespace(recon_error=0.0, rank_check="regular pencil", params=[],
                           gm0=SimpleNamespace(cols=cols),
                           matrices=lambda **v: (np.eye(x.size), np.eye(x.size)))


mode = SimpleNamespace(residual=0.0, y=None, x=x, values={}, sigma=-1.0)
t1 = next(c for c in verify(pevp(cols), mode).checks if c.name == "spectral tail")
t2 = next(c for c in verify(pevp([]), mode).checks if c.name == "spectral tail")
check(t1.status == "FAIL", f"unresolved first component: {t1.explain()}")
check(t2.status == "FAIL", f"nothing measured: {t2.explain()}")
passed(f"tail {t1.value:.2g} FAIL; unmeasured {t2.value} FAIL")
