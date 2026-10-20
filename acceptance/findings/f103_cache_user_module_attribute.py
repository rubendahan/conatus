"""F103 (new, critic 2026-10-19, review of the identity lot): a cached function that reads a
mutable attribute of a USER module must not return a stale value.

The function reads ``cfg.A`` where ``cfg`` is a user module (not a library): the module is
identified by name and version only, so changing ``cfg.A`` from 1 to 3 kept the cache key and the
second call returned the first result. Expected: 6 (a cache miss), or a typed refusal / the cache
refused with a warning (undeclared dependency).
"""
import sys
import types
import warnings
from pathlib import Path

from _acceptance import check, passed, refused, start

import conatus as cn

start(103, "user-module attributes are part of the cache identity")
cfg = types.ModuleType("acceptance_user_cfg")
cfg.A = 1.0
sys.modules[cfg.__name__] = cfg
root = Path("registry").resolve()


@cn.runs.track("modattr", cache=True, root=root)
def scaled(x):
    return cfg.A * x


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    first = scaled(2.0)
    cfg.A = 3.0
    was_refused, second = refused(scaled, 2.0)
if was_refused:
    passed("cache refused")
check(first == 2.0, f"first call returned {first}")
check(second == 6.0, f"after cfg.A = 3 the call returned {second} (stale cache hit), expected 6")
passed("cfg.A change misses the cache")
