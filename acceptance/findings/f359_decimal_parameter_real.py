"""F359: a Decimal parameter value is real (audit inputs, 2026-10-20).

cn.Model(..., Ra=Decimal("2")) stored 2+0j (a Decimal has .imag) and compiled a complex problem.
"""
from decimal import Decimal

from _acceptance import check, passed, start

import conatus as cn

start(359, "Decimal parameter stored as a real number")
m = cn.Model(cn.Box(x=cn.periodic(1.0)), Ra=Decimal("2"))
check(isinstance(m.params["Ra"], float) and m.params["Ra"] == 2.0, f"stored {m.params['Ra']!r}")
passed("float 2.0")
