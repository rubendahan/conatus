"""F401 (audit invariance): the spurious-mode drift filter does not depend on the eigenvalue's unit.

Orr-Sommerfeld at Re = 1e4, alpha = 1, N = 48, drift against N = 64 (drift_tol = 1e-6): the
wave speed c, or the same equation written for c' = c / 1e4. 0.1.1 before the audit compared
|dc| / max(|c|, 1): in the unit c', every eigenvalue is below 1 and the drift became absolute,
so 9 modes passed instead of 1.
"""
import warnings

import numpy as np
from _acceptance import check, passed, start

import conatus as cn

start(401, "drift filter is unit-free")

# Independent anchor (review 2026-10-21, F14): Orszag's (1971) least stable mode at Re = 1e4,
# alpha = 1, c = 0.23752649 + 0.00373967i, must be among the modes kept.
ORSZAG = 0.23752649 + 0.00373967j


def modes(unit):
    m = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="OS", Re=1e4, alpha=1.0)
    m.scalar("v")
    m.given(U=lambda y: 1 - y**2)
    m.eigenvalue("c", growth="imag")
    m.define("L2(f) = dy(dy(f)) - alpha**2*f")
    m.equations(f"OS: L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + {unit!r}*c*L2(v) = 0")
    m.at("walls", "v = 0", "dy(v) = 0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.sort_complex(m.eigenvalues(N=48, drift=64, drift_tol=1e-6).values * unit)


ref = modes(1.0)
check(ref.size >= 1 and np.min(np.abs(ref - ORSZAG)) < 1e-7,
      f"unit 1: modes {ref} do not contain Orszag's c = {ORSZAG}")
for unit in (1e-4, 1e4):
    got = modes(unit)
    check(got.size == ref.size, f"unit {unit:g}: {got.size} modes kept, {ref.size} in the unit of c")
    check(np.allclose(got, ref, rtol=1e-9, atol=0), f"unit {unit:g}: {got} vs {ref}")
passed(f"{ref.size} mode(s) kept in every unit")
