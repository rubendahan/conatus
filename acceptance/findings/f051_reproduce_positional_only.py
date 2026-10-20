"""F051: reproduce() must replay a function with positional-only parameters.

A recorded call of f(a, /, b) replayed by cn.runs.reproduce(run, fn=f): 0.1.0 called it with
keywords only (TypeError). Expected: an identical reproduction.
"""
import warnings
from pathlib import Path

from _acceptance import check, passed, refused, start

import conatus as cn

start(51, "reproduce() respects positional-only signatures")
root = Path("registry").resolve()


def f(a, /, b):
    return a + 2 * b


tracked = cn.runs.track("posonly", root=root)(f)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    value = tracked(1.0, 2.0)
    run = cn.runs.find(name="posonly", root=root)[0]
    was_refused, rep = refused(cn.runs.reproduce, run, fn=f)
check(value == 5.0, f"recorded call returned {value}")
check(not was_refused, "reproduce refused a plain positional-only function")
check(bool(rep.identical), f"reproduce: identical={rep.identical}, max diff {rep.max_abs_diff}")
passed("identical")
