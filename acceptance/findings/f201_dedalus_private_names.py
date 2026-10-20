"""F201 (security audit 2026-10-20): from_dedalus must not hand a script the translator's
internals.

``d3.<name>`` was resolved with a plain ``getattr`` on the translator's own object: ``d3.__dict__``
and ``d3.tr`` gave the script the interpreter, ``d3.__reduce_ex__(1)`` gave ``copyreg`` callables.
Expected: every private or dunder name of ``dedalus.public`` reported "unsupported".
"""
from _acceptance import check, passed, start

from conatus.translate import from_dedalus

start(201, "from_dedalus refuses private names of dedalus.public")
for line in ("leak = d3.__dict__", "leak = d3.tr", "leak = d3.__reduce_ex__(1)",
             "leak = d3.__class__", "from dedalus.public import __dict__ as leak"):
    tr = from_dedalus("import dedalus.public as d3\n" + line + "\n")
    status = {r.lineno: r.status for r in tr.report}
    check(status.get(2) == "unsupported", f"{line!r} was {status.get(2)!r}, not refused")
passed("5 private-name accesses refused")
