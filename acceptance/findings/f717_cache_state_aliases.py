"""F717 (audit of 0.2.1, F06): no cache identity through aliases of state readers.

Oracle: the identity must raise Undeclared on every read of hidden state."""
from _acceptance import check, passed, start

start(717, "no cache identity through aliases of state readers")
import os  # noqa: E402

import numpy as np  # noqa: E402

from conatus.io._identity import Undeclared, callable_identity  # noqa: E402

for target in (os.getenv, open, np.random.normal, os.environ.get):
    def fn(*a, _t=target):
        return _t(*a)
    try:
        callable_identity(fn)
        check(False, f"{target} got a content identity")
    except Undeclared:
        pass
try:
    callable_identity(lambda: np.random.uniform())
    check(False, "np.random.uniform got a content identity")
except Undeclared:
    pass
passed("refused")
