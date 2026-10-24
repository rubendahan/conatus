"""F609 (audit of 0.1.2, F37): an unexpected failure while identifying the tau variables is
reported by the spectral-tail check with its cause, not turned into "no tau"."""
from _acceptance import check, passed, start

start(609, "spurious-mode diagnostic failure is reported")
from types import SimpleNamespace  # noqa: E402

from conatus.solvers import evp_modes, spurious  # noqa: E402


def boom(plan):
    raise RuntimeError("tau identification failed on purpose")


evp_modes._tau_names = boom
plan = SimpleNamespace(var_layouts=lambda g: [], layout=SimpleNamespace(fourier_indices=None),
                       problem=SimpleNamespace())
layout, why = spurious.tail_layout(plan, None, False)
check(layout is None and "failed on purpose" in why, f"cause not reported: {why!r}")
passed(why[:120])
