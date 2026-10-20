"""F031: a macro with duplicate parameter names must be refused.

define("pick(a, a) = a") was accepted in 0.1.0 (which argument does "a" mean?).
"""
import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(31, "duplicate macro parameters are refused")
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
m.scalar("u")
was_refused, _ = refused(m.define, "pick(a, a) = a")
check(was_refused, "define('pick(a, a) = a') accepted")
passed("refused")
