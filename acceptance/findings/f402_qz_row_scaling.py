"""F402 (audit invariance): an eigenvalue problem with one equation multiplied by a constant has
the same modes.

Orr-Sommerfeld (Re = 1e4, alpha = 1, N = 48, drift against N = 64): the equation times 1e-8, or
the condition dy(v) = 0 written times 1e6. The dense solver (QZ) did not equilibrate the rows of
the pencil: the scaled row lost its digits and the drift test removed every mode
(ResolutionError), while the unscaled problem keeps Orszag's mode.
"""
import warnings

import numpy as np
from _acceptance import check, describe, failed, passed, start

import conatus as cn

start(402, "eigenvalues invariant under the scaling of one equation")


def modes(c_eq, c_bc):
    m = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="OS", Re=1e4, alpha=1.0)
    m.scalar("v")
    m.given(U=lambda y: 1 - y**2)
    m.eigenvalue("c", growth="imag")
    m.define("L2(f) = dy(dy(f)) - alpha**2*f")
    m.equations(f"OS: {c_eq!r}*(L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + c*L2(v)) = 0")
    m.at("walls", "v = 0", f"{c_bc!r}*dy(v) = 0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.sort_complex(m.eigenvalues(N=48, drift=64, drift_tol=1e-6).values)


ref = modes(1.0, 1.0)
for c_eq, c_bc in ((1e-8, 1.0), (1.0, 1e6)):
    try:
        got = modes(c_eq, c_bc)
    except Exception as exc:  # noqa: BLE001 -- any exception is the failure reported
        failed(f"equation x {c_eq:g}, condition x {c_bc:g}: {describe(exc)}")
    check(got.size == ref.size and np.allclose(got, ref, rtol=1e-9, atol=0),
          f"equation x {c_eq:g}, condition x {c_bc:g}: {got} vs {ref}")
passed(f"the same {ref.size} mode(s) at every scaling")
