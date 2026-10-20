"""F305 (new, audit "state" 2026-10-20): provenance and the run-cache key describe the code on
disk, not the code that runs.

provenance.code_state() and conatus.runs.code_fingerprint() hash the package files when they
are called. A process that imported conatus, then saw its sources edited (git pull, a checkout,
an editor) recorded the new tree's hash for results computed by the old code, with no warning, and
stored them under the new code's cache key: a fresh process with the new code then got the old
code's result as a cache hit. Expected: a source edited after import is declared, the run is
not reproducible, and the cache is not used.
"""
import os
import shutil
import subprocess
import sys
import tempfile

from _acceptance import check, passed, start

import conatus

start(305, "sources edited after import are declared by the provenance")
tmp = tempfile.mkdtemp(prefix="cn_accept_f305_tree_")
shutil.copytree(os.path.dirname(conatus.__file__), os.path.join(tmp, "conatus"),
                ignore=shutil.ignore_patterns("__pycache__"))
child = r'''
import os, sys
sys.path.insert(0, sys.argv[1])
import conatus
from conatus import provenance
with open(os.path.join(sys.argv[1], "conatus", "errors.py"), "a") as fh:   # imported above
    fh.write("\n# edited after import\n")
st = provenance.code_state()
print("CHANGED", bool(st.get("changed_since_import")), "REPRO", st.get("code_reproducible"))
'''
env = dict(os.environ)
env.pop("PYTHONPATH", None)
p = subprocess.run([sys.executable, "-c", child, tmp], capture_output=True, text=True, env=env,
                   timeout=600)
shutil.rmtree(tmp, ignore_errors=True)
out = p.stdout.strip().splitlines()[-1:] or [p.stderr[-500:]]
check(p.returncode == 0, f"child failed: {p.stderr[-500:]}")
check("CHANGED True" in out[0], f"an edit after import is not declared: {out[0]}")
check("REPRO False" in out[0], f"the run is still called reproducible: {out[0]}")
passed("edit after import declared, run not reproducible")
