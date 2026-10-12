""" resolution claims, checked against the archived runs (gpu3 harvest).

METRICS/ROADMAP state that the leading stable Floquet multiplier of Nanda & Thakur's RPO1 is
converged in spatial resolution to the declared tolerance between the 24- and 32-grid orbits,
with true residuals far below that difference, and that the 32-grid orbit is converged to
Newton tolerance. This test re-reads the archived JSON, so a claim and its runs cannot drift.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

RUNS = Path(__file__).resolve().parents[2] / "results" / "runs"
F16 = RUNS / "" / "floquet_main.json"
F24 = RUNS / "" / "floquet_main.json"
F32 = RUNS / "" / "floquet_main.json"
N32 = RUNS / "" / "newton_check.json"
TOL_REL = 0.02            # tolerance on the 24 -> 32 change of the leading stable |multiplier|,
                          # set after the measurement (METRICS): a consistency check only
# Published reference: the text's "|Λ| ≈ 0.5" is the largest along the Re
# continuation; the value at Re = 1000 is in Nanda & Thakur's figure 4 (48 x 81 x 48), read off the
# image: leading pair 0.336 + 0.308i (|Λ| 0.455, arg 42.5 deg), reading uncertainty ~0.005
FIG4_LEAD = complex(0.336, 0.308)
FIG4_READ = 0.005

pytestmark = pytest.mark.skipif(not all(p.exists() for p in (F16, F24, F32, N32)),
                                reason="archived runs not in this checkout")


def _leading_stable(path: Path) -> tuple[float, float]:
    """(|Λ|, true residual) of the largest multiplier that is not neutral (|Λ| < 0.99)."""
    sub = json.loads(path.read_text())["components"]["subspace"]
    rows = [(a, r) for a, r in zip(sub["abs"], sub["residual_true"]) if a < 0.99]
    return max(rows)


def _leading_stable_value(path: Path) -> complex:
    """The leading non-neutral multiplier itself (upper member of a pair)."""
    sub = json.loads(path.read_text())["components"]["subspace"]
    vals = [complex(*z) for z in sub["multipliers"]]
    return max((z for z in vals if abs(z) < 0.99), key=lambda z: (abs(z), z.imag))


def test_leading_stable_multiplier_converged_between_24_and_32():
    l24, r24 = _leading_stable(F24)
    l32, r32 = _leading_stable(F32)
    assert abs(l32 - l24) / l32 <= TOL_REL, (l24, l32)
    assert max(r24, r32) <= 1e-3 * abs(l32 - l24)       # Arnoldi error ≪ resolution change
    assert abs(l32 - 0.5) <= 0.06                         # text: "≈ 0.5" (one digit, whole branch)


def test_leading_pair_is_the_published_mode():
    # same mode as the published figure 4 (a complex pair, at its argument), |Λ| within 1.5 % of
    # it, and the 24 -> 32 change goes towards it (the claim of METRICS)
    z24, z32 = _leading_stable_value(F24), _leading_stable_value(F32)
    assert z32.imag > 0.1 and z24.imag > 0.1                              # a complex pair
    assert abs(np.angle(z32) - np.angle(FIG4_LEAD)) < np.radians(2.0)
    assert abs(abs(z32) - abs(FIG4_LEAD)) <= 0.015 * abs(FIG4_LEAD) + FIG4_READ
    assert abs(abs(FIG4_LEAD) - abs(z32)) < abs(abs(FIG4_LEAD) - abs(z24))


def test_sixteen_grid_is_not_in_the_convergence_claim():
    # the 16-grid orbit is under-resolved: its leading stable multiplier (real, 0.50) is not the
    # complex pair that leads at 24 and 32; it is outside the declared tolerance on purpose
    l16, _ = _leading_stable(F16)
    l32, _ = _leading_stable(F32)
    assert abs(l16 - l32) / l32 > TOL_REL


def test_32_grid_orbit_converged_and_neutral_directions_identified():
    nt = json.loads(N32.read_text())
    assert nt["status"] == "converged" and nt["residual"] <= 1e-12
    assert abs(nt["T"] - 41.0329525) <= 1e-6               # T of the converged 32-grid orbit
    fl = json.loads(F32.read_text())
    assert fl["orbit_residual"] <= 2e-10
    span = fl["components"]["subspace"].get("neutral_span", {})
    assert span and min(span.values()) >= 0.99
