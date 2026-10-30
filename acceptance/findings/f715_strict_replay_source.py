"""F715 (audit of 0.2.1, N53): strict replay refuses a changed function source.

Oracle: the function was not called, its call list stays empty."""
from _acceptance import check, passed, start

start(715, "strict replay refuses a changed function source")
import importlib  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from conatus.errors import OptionError  # noqa: E402
from conatus.runs.integrations import call_recorded  # noqa: E402

d = Path(tempfile.mkdtemp())
body = "calls = []\ndef compute(x):\n    calls.append(x)\n    return x + {}\n"
(d / "f715_user.py").write_text(body.format(1))
sys.path.insert(0, str(d))
mod = importlib.import_module("f715_user")
_, run = call_recorded(mod.compute, (2,), {}, name="f715", root=d / "runs", _with_run=True)
mod.calls.clear()
(d / "f715_user.py").write_text(body.format(200))
os.utime(d / "f715_user.py", (2e9, 2e9))
importlib.invalidate_caches()
mod = importlib.reload(mod)
try:
    run.reproduce(fn=mod.compute, record=False, require_same_code=True)
    check(False, "strict replay ran a changed function")
except OptionError:
    pass
check(mod.calls == [], "the changed function was called")
passed("refused before the call")
