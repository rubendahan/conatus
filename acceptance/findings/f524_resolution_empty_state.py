"""F524 (external review of 0.1.1, F10): an empty state is not reported as resolved.

``resolution_report({})`` had zero entries, ok=True and the headline "all resolved". Expected:
ok False, the unknown named as not measured, and no "every tail is under the target".
"""
from _acceptance import check, passed, start

start(524, "an empty state is unmeasured, not resolved")
import conatus as cn  # noqa: E402

m = cn.Model(cn.Box(x=cn.periodic(6.0)))
m.scalar("u")
m.equations("dt(u)-lap(u)=0")
s = m.simulate(16, scheme="RK443", dt=0.01)
r = s.plan.resolution_report({}, layout="coeff")
text = r.explain()
print("  " + text.splitlines()[0])
check(not r.ok, "ok=True with nothing measured")
check("all resolved" not in text, "the headline says all resolved")
check("every tail is under the target" not in text, "the recommendation says every tail passed")
check("u" in getattr(r, "unmeasured", []), "u is not named as unmeasured")
passed("ok False, u not measured")
