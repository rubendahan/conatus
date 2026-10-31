"""F804 (external review of 0.2.3, finding 4): arguments keep their type in the cache and in replay.

Oracle: NumPy indexing and plain Python: A[[0, 1]] sums to 15 and A[(0, 1)] is 1 for
A = arange(9).reshape(3, 3); the other functions return the type name they receive.
"""
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from _acceptance import check, passed, start

start(804, "cache keys and replay distinguish argument types")
import conatus.runs as R  # noqa: E402
from conatus.errors import UnsupportedError as cn_unsupported  # noqa: E402


def indexed_sum(index):
    return int(np.arange(9).reshape(3, 3)[index].sum())


def type_name(a):
    return type(a).__name__ + "".join(type(x).__name__ for x in (a if isinstance(a, (list, tuple)) else ()))


def key_order(a):
    return ",".join(a)


SEEN = {}


def show(a):
    SEEN["v"] = a
    return 0


warnings.simplefilter("ignore")
PAIRS = (([0, 1], (0, 1)), (1, 1.0), (True, 1), (1.0, np.float64(1.0)), (1, np.int32(1)),
         (np.float32(1.0), np.float64(1.0)), (b"ab", "ab"), ([1, 2], (1, 2)), ([(1,)], [[1]]))
with TemporaryDirectory(dir=Path.cwd()) as root:
    cached = R.track("indices", root=root, cache=True)(indexed_sum)
    check(cached([0, 1]) == 15, "list")
    check(cached((0, 1)) == 1, "tuple read the list's cache entry")
    for i, (a, b) in enumerate(PAIRS):
        f = R.track(f"p{i}", root=root, cache=True)(type_name)
        ra, rb = f(a), f(b)
        check(ra == type_name(a) and rb == type_name(b), f"{a!r} vs {b!r}: {ra}, {rb}")
    f = R.track("order", root=root, cache=True)(key_order)
    check(f({"a": 1, "b": 2}) == "a,b" and f({"b": 2, "a": 1}) == "b,a", "dict order")
    same = R.track("again", root=root, cache=True)(type_name)
    same((1, 2))
    same((1, 2))
    runs = R.find(name="again", root=root)
    check(any((r.cache_report or {}).get("hit") for r in runs), "same arguments must still hit")

    for i, value in enumerate(((0, 1), [0, 1], (1, [2, (3,)]), np.float32(2.5), np.int32(3),
                               b"xy", {"b": 1, "a": 2}, True, 1.5, None, (np.float64(1.0),))):
        name = f"rep{i}"
        R.track(name, root=root)(show)(value)
        run = R.find(name=name, root=root)[0]
        check(run.rec["inputs"]["args_exact"], f"{value!r} is not marked exact")
        SEEN.clear()
        run.reproduce(fn=show, record=False)
        v = SEEN["v"]
        check(type(v) is type(value) and repr(v) == repr(value), f"replay of {value!r} gave {v!r}")
        if isinstance(value, dict):
            check(list(v) == list(value), "dict order lost")
    # what cannot come back with its type is not called exact
    from collections import namedtuple
    from pathlib import PurePosixPath
    P = namedtuple("P", "a")

    def anyarg(a):
        return 0
    for i, value in enumerate((P(1), PurePosixPath("/x"), np.matrix([[1.0]]))):
        R.track(f"inexact{i}", root=root)(anyarg)(value)
        rec = R.find(name=f"inexact{i}", root=root)[0].rec["inputs"]
        check(not rec["args_exact"], f"{value!r} called exact")

    # a record written before 0.2.4: no tuple can be told from a list; refused, not guessed
    import json
    R.track("old", root=root)(indexed_sum)((0, 1))
    run = R.find(name="old", root=root)[0]
    p = run.path / "run.json"
    rec = json.loads(p.read_text())
    rec["inputs"].pop("args_format")
    rec["inputs"]["args"] = {"index": [0, 1]}
    p.write_text(json.dumps(rec))
    run = R.find(name="old", root=root)[0]
    try:
        run.reproduce(fn=indexed_sum, record=False)
        check(False, "old record with a list replayed")
    except cn_unsupported as err:
        check("before 0.2.4" in str(err), str(err))
# a long double has no Python value (item() returns it unchanged): a recursion before the fix
from conatus.runs._util import typed_sha256  # noqa: E402
check(typed_sha256(np.longdouble(1)) != typed_sha256(np.longdouble(2)), "long doubles collide")
passed("list/tuple, int/float, bool/int, NumPy/Python, bytes and dict order keep their identity")
