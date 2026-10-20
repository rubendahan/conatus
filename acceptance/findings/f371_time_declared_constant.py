"""F371: a time-dependent model that declares t itself is warned that sin(t) is frozen (review
of the inputs audit).

cn.Model(..., t=1.0) with dt(u) = sin(t): t is the parameter (the native time is t only
when the model does not use the name), so the forcing is sin(1) for the whole run, without a
word. A DefaultChoiceWarning names the value; a model without the declaration says nothing.
"""
import warnings

import numpy as np
from _acceptance import check, passed, start

import conatus as cn
from conatus.errors import DefaultChoiceWarning

start(371, "t declared as a constant is said")


def build(**params):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="forced", **params)
    m.scalar("u")
    m.equations("dt(u) = sin(t)")
    return m


with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    build(t=1.0).simulate(8, scheme="RK222", dt=0.01)
said = [x for x in w if issubclass(x.category, DefaultChoiceWarning) and "t = 1.0" in str(x.message)]
check(bool(said), "no DefaultChoiceWarning naming t = 1.0")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    s = build().simulate(8, scheme="RK222", dt=0.01).initial(u=0)
check(not [x for x in w if issubclass(x.category, DefaultChoiceWarning)
           and "not the time" in str(x.message)], "warned without a declared t")
s.run(steps=100)                                   # t = 1: u = 1 - cos(1)
check(abs(s.grid("u").mean() - (1 - np.cos(1.0))) < 1e-4, "the native time does not drive sin(t)")
passed("warned with the value; native t unchanged")
