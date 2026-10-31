"""F806 (external review of 0.2.3, finding 7): code_identical covers what the function reads.

Oracle: direct arithmetic. caller(2) is 3 with helper x + 1 and 4 after rebinding it to x + 2.
Unchanged code reports True, a changed helper False or an error when strict, and a dependency
without a content hash reports None (unknown).
"""
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

from _acceptance import check, passed, refused, start

start(806, "code_identical covers helpers, closures and globals")
import conatus.runs as R  # noqa: E402


def helper(x):
    return x + 1


def replacement(x):
    return x + 2


def caller(x):
    return helper(x)


OFFSET = 1


def reads_global(x):
    return x + OFFSET


class Box:
    pass


holder = Box()
holder.v = 1


def reads_object(x):
    return x + holder.v


warnings.simplefilter("ignore")
with TemporaryDirectory(dir=Path.cwd()) as root:
    check(R.track("h", root=root)(caller)(2) == 3, "recorded value")
    run = R.find(name="h", root=root)[0]
    rep = run.reproduce(fn=caller, record=False, require_same_code=True)
    check(rep.code_identical is True and rep.identical, "unchanged helper not identical")
    helper = replacement  # noqa: F811
    check(caller(2) == 4, "oracle")
    rep = run.reproduce(fn=caller, record=False)
    check(rep.code_identical is False and not rep.identical, f"changed helper: {rep.code_identical}")
    check("user_dependencies" in rep.code_diff["changed"], "change not named")
    check(refused(lambda: run.reproduce(fn=caller, record=False, require_same_code=True))[0],
          "strict replay accepted a changed helper")
    R.track("g", root=root)(reads_global)(1)
    run = R.find(name="g", root=root)[0]
    OFFSET = 5
    check(run.reproduce(fn=reads_global, record=False).code_identical is False, "changed global")
    R.track("o", root=root)(reads_object)(1)
    run = R.find(name="o", root=root)[0]
    rep = run.reproduce(fn=reads_object, record=False)
    check(rep.code_identical is None and rep.identical, f"untracked object: {rep.code_identical}")
    check("UNKNOWN" in rep.explain(), rep.explain())
    check(refused(lambda: run.reproduce(fn=reads_object, record=False, require_same_code=True))[0],
          "strict replay accepted an unknown dependency")
passed("unchanged True, changed False or refused, untrackable None or refused")
