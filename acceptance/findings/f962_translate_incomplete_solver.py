"""F962 (pass 4 H8): the solver of an incomplete translation is refused unless allowed.

Oracle: Python. A loop adding 1 three times gives u = 3, which the translation cannot give:
ivp_solver(), discretize(), initial_state() and ncc() refuse with UnsupportedError naming the
line; allow_incomplete=True proceeds with a CorrectnessWarning naming it. A complete script
gives no warning.
"""
import warnings

from _acceptance import check, passed, start
from _value_model import HEAD, TAIL

start(962, "incomplete translations refused by the solver unless allowed")
import conatus as cn  # noqa: E402
from conatus.errors import CorrectnessWarning, UnsupportedError  # noqa: E402

tr = cn.from_dedalus(HEAD + "u['g'] = 0\nfor i in range(3):\n    u['g'] += 1\n" + TAIL)
check(not tr.complete, "the loop is reported translated")
for what, call in [("ivp_solver", lambda **k: tr.ivp_solver(dt=0.1, **k)),
                   ("discretize", tr.discretize), ("initial_state", tr.initial_state),
                   ("ncc", tr.ncc)]:
    try:
        call()
        check(False, f"{what}() accepted the incomplete translation")
    except UnsupportedError as e:
        check("line 13" in str(e), f"{what}() refusal does not name line 13: {e}")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        call(allow_incomplete=True)
    check(any(issubclass(x.category, CorrectnessWarning) and "line 13" in str(x.message)
              for x in w), f"{what}(allow_incomplete=True) gives no warning naming line 13")
ok = cn.from_dedalus(HEAD + "u['g'] = 3\n" + TAIL)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    s = ok.ivp_solver(dt=0.1)
check(ok.complete and not [x for x in w if issubclass(x.category, CorrectnessWarning)],
      "a complete translation is warned")
check(abs(float(s.grid("u").mean()) - 3) < 1e-12, "complete translation: u != 3")
passed("4 entry points refuse and name the line; allowed calls warn; complete scripts quiet")
