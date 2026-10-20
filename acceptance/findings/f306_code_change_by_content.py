"""F306 (new, review of the audit "state" 2026-10-20, F8): "changed since import" is decided by
the content of the modules actually imported, not by modification times.

The first fix compared each file's modification time with the import time (2 s clock slack):
an edit within 2 s of the import and a deleted module were not seen, while a touch, or a
checkout to another branch and back (same text, new mtime), was declared a change and the run
refused the cache. Expected: a touch is not a change; an immediate edit and a deletion of an
imported module are; an edit of a module the process never imported is not (it would run the
new text, which the tree hash describes).
"""
import os
import shutil
import subprocess
import sys
import tempfile

from _acceptance import check, passed, start

import conatus

start(306, "code changes after import are judged by content, for the imported modules")
tmp = tempfile.mkdtemp(prefix="cn_accept_f306_tree_")
shutil.copytree(os.path.dirname(conatus.__file__), os.path.join(tmp, "conatus"),
                ignore=shutil.ignore_patterns("__pycache__"))
child = r'''
import os, sys
sys.path.insert(0, sys.argv[1])
import conatus
from conatus import provenance
pkg = os.path.join(sys.argv[1], "conatus")
never = [m for m in ("units.py", "notebook.py", "agent.py", "_doctor.py", "resolution.py")
         if "conatus." + m[:-3] not in sys.modules and os.path.exists(os.path.join(pkg, m))][0]
os.utime(os.path.join(pkg, "errors.py"), (2e9, 2e9))            # touch: same text
with open(os.path.join(pkg, never), "a") as fh:                  # never imported
    fh.write("\n# edited\n")
print("TOUCH", provenance.changed_since_import())
with open(os.path.join(pkg, "errors.py"), "a") as fh:            # imported: edited at once
    fh.write("\n# edited\n")
os.remove(os.path.join(pkg, "provenance", "__init__.py"))      # imported: deleted
print("EDIT", provenance.changed_since_import())
'''
env = dict(os.environ)
env.pop("PYTHONPATH", None)
p = subprocess.run([sys.executable, "-c", child, tmp], capture_output=True, text=True, env=env,
                   timeout=600)
shutil.rmtree(tmp, ignore_errors=True)
check(p.returncode == 0, f"child failed: {p.stderr[-800:]}")
lines = dict(line.split(" ", 1) for line in p.stdout.strip().splitlines() if " " in line)
check(lines.get("TOUCH") == "[]", f"a touch or an unimported edit declared: {lines.get('TOUCH')}")
check("'errors.py'" in lines.get("EDIT", ""), f"an immediate edit not declared: {lines.get('EDIT')}")
check("provenance/__init__.py (deleted)" in lines.get("EDIT", ""),
      f"a deleted imported module not declared: {lines.get('EDIT')}")
passed(f"touch and unimported edit ignored; edit and deletion declared: {lines.get('EDIT')}")
