"""F803 (external review of 0.2.3, finding 8): the recorded growth follows the declared convention.

Oracle: c u = (2 - 3j) u fixes c = 2 - 3i. Declared growth Im c gives -3, Re c gives 2, a callable
gives its own value. The record and the result's own ranking must agree.
"""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from _acceptance import check, passed, start

start(803, "run records use the declared growth convention")
import conatus as cn  # noqa: E402
import conatus.runs as R  # noqa: E402

# growth undeclared on a Model: the declared default is Im c (with a DefaultChoiceWarning)
CASES = (("imag", -3.0), ("real", 2.0), (None, -3.0), (lambda s: s.real - s.imag, 5.0))
with TemporaryDirectory(dir=Path.cwd()) as root:
    for growth, want in CASES:
        m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
        m.scalar("u")
        if growth is None:
            m.eigenvalue("c")
        else:
            m.eigenvalue("c", growth=growth)
        m.equations("c*u=(2-3j)*u")
        ev = m.eigenvalues(8, k=1)
        run = R.record(ev, root=root, compression=None)
        got = run.results["growth_max"]
        check(abs(got - want) < 1e-10, f"growth={growth!r}: recorded {got}, exact {want}")
        top = ev._growth_fn()(ev.values)[ev.order()[0]]
        check(abs(got - top) < 1e-12, f"growth={growth!r}: record {got}, ranking {top}")
        check("growth_convention" in run.results, "convention not recorded")
passed("record equals the exact growth under imag, real, undeclared and callable conventions")
