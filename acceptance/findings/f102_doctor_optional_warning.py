"""F102 (new, critic 2026-10-19): `conatus doctor` on a plain install has no [warning] line for
an optional extra.

Definition of done of the release: `pip install conatus` then `conatus doctor` "with no [warning]
for optional items". The doctor of 0.1.0 prints "[warning ] matplotlib (tutorials)" on a plain
install (the tutorials extra is optional). Expected: exit 0 and optional extras reported as
information, not warnings.
"""
import subprocess
import sys

from _acceptance import check, passed, start

start(102, "doctor: optional extras are not warnings")
p = subprocess.run([sys.executable, "-m", "conatus", "doctor"], capture_output=True, text=True, timeout=240)
check(p.returncode == 0, f"doctor exit {p.returncode}: {p.stderr.strip()[-200:]}")
warn = [ln.strip() for ln in p.stdout.splitlines() if "[warning" in ln]
optional = [ln for ln in warn if any(k in ln for k in ("(tutorials)", "(notebook)", "(gpu)", "(mpi)",
                                                       "(docs)", "(dev)", "(studio)"))]
check(not optional, f"{len(optional)} optional item(s) as warnings: {optional[0][:160] if optional else ''}")
passed(f"{len(warn)} warning line(s), none for an optional extra")
