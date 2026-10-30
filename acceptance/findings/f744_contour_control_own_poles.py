"""F744 (audit of 0.2.1, N26): the poles of the target box do not belong to the control box.

Oracle: f = (z - 0.5)(z - 3)/(z - 2): the zeros and the pole are known.
"""
from _acceptance import check, passed, start

start(744, "contour control boxes count their own poles")
import conatus as cn  # noqa: E402

from conatus.contour import count  # noqa: E402
from conatus.errors import ConvergenceError  # noqa: E402


def f(z):
    return (z - .5) * (z - 3) / (z - 2)


target = cn.box(re=(1.5, 3.5), im=(-1, 1))           # one zero, one pole
plain = cn.box(re=(0., 1.), im=(-1, 1))              # one zero, no pole
pole = cn.box(re=(1.5, 2.5), im=(-1, 1))             # one pole, no zero
r = count(f, target, poles=1, control=(plain, 1))
check(r.count == 1, f"count {r.count}")
for ctrl in ((plain, 2), (pole, 0)):
    try:
        count(f, target, poles=1, control=ctrl)
        check(False, f"control {ctrl[1]} accepted")
    except ConvergenceError:
        pass
check(count(f, target, poles=1, control=(pole, 0, 1)).count == 1, "pole declared in the control")
passed("own count, own poles")
