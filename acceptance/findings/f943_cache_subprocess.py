"""F943 (review H10): a function that runs an external process is never read from the cache.

Oracle: the file the process reads. ``cat`` prints a file whose content changes between two
calls, through ``subprocess.run``, an alias ``import subprocess as sp``, ``check_output`` imported
by name, and ``os.popen``. Control: a function of its argument only is a cache hit.
"""
import os
import subprocess
import subprocess as sp
import warnings
from pathlib import Path
from subprocess import check_output

from _acceptance import check, passed, start

start(943, "external processes are hidden state for the cache")
import conatus as cn  # noqa: E402

root = "registry"
path = Path("value.txt").resolve()
PATH = str(path)             # a string: hashed by content (a Path object has no content hash)


def call(f, *a):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = f(*a)
    return out, any("cache hit" in str(w.message) for w in caught)


def via_run(n):
    return subprocess.run(["cat", PATH], capture_output=True, text=True).stdout


def via_alias(n):
    return sp.run(["cat", PATH], capture_output=True, text=True).stdout


def via_name(n):
    return check_output(["cat", PATH], text=True)


def via_os(n):
    with os.popen(f"cat {PATH}") as fh:
        return fh.read()


for fn in (via_run, via_alias, via_name, via_os):
    f = cn.runs.track(fn.__name__, cache=True, root=root)(fn)
    path.write_text("1")
    a, _ = call(f, 0)
    path.write_text("2")
    b, hit = call(f, 0)
    check((a, b) == ("1", "2"), f"{fn.__name__}: values {a!r}, {b!r} (hit {hit}), file gave '1' then '2'")


@cn.runs.track("pure", cache=True, root=root)
def pure(n):
    return n * 2.0


call(pure, 3)
v, hit = call(pure, 3)
check(hit and v == 6.0, "control: a pure function is not reused")
passed("four ways to run a process are recomputed; a pure function is reused")
