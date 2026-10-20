"""F253 (audit "assurances", 2026-10-20): time_average applies its batch-means check.

White noise plus a slow sine (period 3000 samples, 4000 samples): Sokal's window closes at
tau = 1.35, the 20 batch means give a 3.1x larger standard error, and the mean, 2.7 standard
errors from the true 0, was reported trusted.
"""
import numpy as np
from _acceptance import check, passed, start

start(253, "time_average flags a batch-means error far above its own")
import conatus as cn  # noqa: E402

k = np.arange(4000)
x = np.random.default_rng(0).standard_normal(4000) + 0.5 * np.sin(2 * np.pi * k / 3000)
r = cn.time_average(x, discard=0)
check(not r.trusted, f"trusted with batch/Sokal = {r.batch_stderr / r.stderr:.2f}, mean "
      f"{r.mean:.3g} +- {r.stderr:.2g} (true 0)")
passed(f"flagged: {r.flags[-1][:80]}")
