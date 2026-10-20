"""F362: Parameter values and the problem presets refuse NaN and text (audit inputs).

Parameter("Re", nan) and Parameter("Re", "1") were accepted, and so was every preset built on
them (cn.problems.channel_evp(Re=nan)), failing or giving NaN deep in the assembly; a complex
value with a real dtype was a bare TypeError.
"""
from _acceptance import check, need, passed, refused, start

from conatus.expert import Parameter

start(362, "parameter values checked at declaration")
for v in (float("nan"), float("inf"), "1", True, 1j):
    r, _ = refused(Parameter, "Re", v)
    check(r, f"Parameter('Re', {v!r}) accepted")
need("conatus.problems")
import conatus.problems as P  # noqa: E402

r, _ = refused(P.channel_evp, N=32, Re=float("nan"), alpha=1.0)
check(r, "channel_evp(Re=nan) accepted")
passed("refused")
