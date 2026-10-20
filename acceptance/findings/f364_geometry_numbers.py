"""F364: geometries take real numbers only (audit inputs, 2026-10-20).

walls("0", "1"), Disk("1"), Ball("1") and Annulus("1", "2") were parsed from text, walls(False,
True) and Disk(True) took flags for numbers, TaylorCouette(1, 2, length=0 or nan) was accepted,
periodic("6") / periodic((0, 1, 2)) / periodic(None) were bare unpacking errors.
"""
from _acceptance import check, passed, refused, start

import conatus as cn

start(364, "geometry lengths and positions are numbers")
cases = [(cn.walls, ("0", "1")), (cn.walls, (False, True)), (cn.walls, (0j, 1)),
         (cn.Disk, ("1",)), (cn.Disk, (True,)), (cn.Ball, ("1",)), (cn.Annulus, ("1", "2")),
         (cn.TaylorCouette, (1.0, 2.0, 0.0)), (cn.TaylorCouette, (1.0, 2.0, float("nan"))),
         (cn.periodic, ("6",)), (cn.periodic, ((0, 1, 2),)), (cn.periodic, (None,))]
for f, args in cases:
    r, _ = refused(f, *args)
    check(r, f"{f.__name__}{args!r} accepted")
passed(f"{len(cases)} malformed declarations refused")
