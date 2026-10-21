"""F008 (blocking): a refused overwrite of a foreign checkpoint series stays refused on retry.

Series written by run a (nu = 0.05); a writer of run b (nu = 0.4) on the same directory is
refused once, and must be refused again on the second attempt; the file stays run a's.
0.1.0: OptionError, then "write accepted" and the file holds nu = 0.4.
"""
from pathlib import Path

import numpy as np
from _acceptance import check, failed, is_refusal, passed, refused, start, describe

import conatus as cn

start(8, "foreign-checkpoint guard is transactional (fail, retry, original unchanged)")


def make(nu):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="burgers", nu=nu)
    m.scalar("u")
    m.equations("dt(u)-nu*dx(dx(u)) = -u*dx(u)")
    return m.simulate(32, scheme="RK443", dt=0.002).initial(u=lambda x: np.sin(x) + 0.3 * np.cos(2 * x))


root = Path("ckpt").resolve()
a = make(0.05)
a.checkpoints(root, every=1, keep=2)
a.run(steps=1)
b = make(0.4)
b.run(steps=1)
outcomes = []
was_refused, writer = refused(cn.io.CheckpointSeries, b.solver, root, every=1)
for attempt in range(0 if was_refused else 2):      # the same writer, twice
    try:
        writer.write()
        outcomes.append("accepted")
    except Exception as exc:  # noqa: BLE001 -- a typed refusal is the expected outcome
        if not is_refusal(exc):
            failed(f"attempt {attempt + 1}: crash {describe(exc)}")
        outcomes.append(type(exc).__name__)
check("accepted" not in outcomes, f"attempts: {outcomes}")
files = sorted(root.glob("checkpoint_*.h5"))
check(bool(files), "no checkpoint left in the series (all([]) would pass)")
params = [cn.io.read(f).parameters for f in files]
check(all(pp.get("nu") == 0.05 for pp in params), f"series parameters after the attempts: {params}")
passed(f"attempts {outcomes}, series still nu = 0.05")
