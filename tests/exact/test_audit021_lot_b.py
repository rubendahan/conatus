"""Regression tests of lot B of the 0.2.1 audit: each runs one acceptance script of
packaging/acceptance/findings (f731 .. f756), whose oracle is analytic, NumPy or SciPy."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

DIR = Path(__file__).resolve().parents[2] / "packaging" / "acceptance" / "findings"
SCRIPTS = sorted(p for p in DIR.glob("f7[3-5]?_*.py") if 731 <= int(p.name[1:4]) <= 756)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_acceptance(script):
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=DIR,
                       timeout=280)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
