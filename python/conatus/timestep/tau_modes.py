"""Spurious growing tau modes of an IVP's implicit operator.

A tau discretization can give the implicit pencil -L x = lambda M x an eigenvalue with
Re(lambda) > 0 whose eigenvector lives almost entirely on the tau unknowns: a numerical mode, not
a physical one. The first-order form of the spherical shell with both taus lifted into the same
derivative basis (the form of Dedalus's shell examples) has one, Re(lambda) ~ 1e7 E from N_r ~ 48,
identical in Dedalus. An implicit step amplifies such a mode when |R(lambda dt)| > 1 (RK443:
0 < lambda dt < 7.9; SBDF2: 0 < lambda dt < 4) and damps it otherwise, so a run explodes for
some (N_r, E, dt) and not for others. :func:`check_tau_modes` finds these modes on a few
representative groups (dense generalized eigenproblem of each, once per solver and dt), refuses a
step that amplifies one, and returns a note for ``explain()`` otherwise.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from ..errors import UnsupportedError

__all__ = ["implicit_amplification", "check_tau_modes", "TAU_FRACTION", "MAX_CHECK_SIZE"]

TAU_FRACTION = 0.5
"""A growing eigenmode is a tau mode when more than this fraction of its squared norm sits on the
tau unknowns (the spurious modes measured: > 0.99999; physical modes: < 1e-20)."""

MAX_CHECK_SIZE = 1500
"""Groups larger than this are not checked (dense eigenproblem); declared in the note."""


def implicit_amplification(solver, z: complex) -> float:
    """|amplification| per step of the solver's implicit scheme on x' = lambda x (implicit part
    only, z = lambda dt): stability function of the IMEX RK tableau, spectral radius of the
    SBDF / CN recurrences (start-up steps aside)."""
    fam = solver.family
    if fam == "rk":
        _, H, c = solver._rk
        s = len(c) - 1
        X = [1.0 + 0j]
        for i in range(1, s + 1):
            acc = 1.0 + z * sum(H[i][k] * X[k] for k in range(i))
            den = 1.0 - z * H[i][i]
            if den == 0:
                return float("inf")
            X.append(acc / den)
        return float(abs(X[s]))
    if fam == "sbdf":
        from .integrator import sbdf_coefficients
        a, _ = sbdf_coefficients(solver.order)
        poly = [a[0] - z] + list(a[1:])                   # (a0 - z) rho^q + a1 rho^(q-1) + ...
    else:
        a, b, _ = solver._cn
        poly = [a[j] - z * b[j] for j in range(len(a))]
    if poly[0] == 0:
        return float("inf")
    return float(np.abs(np.roots(poly)).max())


def _tau_mask(host, slot) -> np.ndarray:
    """True for the columns of a group that are tau (non-bulk) unknowns."""
    plan = host.plan
    meta = host._smeta(plan.var_slots(slot.group),
                       [(v.domain, v.rank) for v in host.variables], slot.group)
    gm = plan.matrices(slot.group, host.reduced)
    meta = [meta[j] for j in gm.col_index]
    return np.array([m[2] is None for m in meta], dtype=bool)


def _representatives(host) -> list:
    """Up to two groups: the smallest-degree group of m = 0 with l >= 1 and the last group."""
    slots = list(host.slots)
    if not slots:
        return []
    first = next((s for s in slots if getattr(s.group, "m", None) == 0
                  and min(getattr(s.group, "ells", [0]) or [0]) >= 1), slots[0])
    out = [first]
    if slots[-1] is not first:
        out.append(slots[-1])
    return out


def _find_tau_modes(solver) -> tuple[list, list]:
    """(found, skipped): growing tau modes (group label, lambda, tau fraction) of the
    representative groups, and the groups too large to check. The pencil does not depend on dt
    (L and M are frozen with plan.params): computed once per solver, reused by set_dt."""
    cached = getattr(solver, "_tau_modes_found", None)
    if cached is not None:
        return cached
    host = getattr(solver.system, "host", solver.system)
    found, skipped = [], []
    for slot in _representatives(host):
        if slot.size > MAX_CHECK_SIZE:
            skipped.append(f"{slot.label} (n={slot.size})")
            continue
        mask = _tau_mask(host, slot)
        if not mask.any():
            continue
        A, B = -slot.L.toarray(), slot.M.toarray()
        w, V = sla.eig(A, B)
        ok = np.isfinite(w) & (np.abs(w) < 1e14)
        for i in np.nonzero(ok & (w.real > 0))[0]:
            v = np.abs(V[:, i]) ** 2
            frac = float(v[mask].sum() / v.sum())
            if frac > TAU_FRACTION:
                found.append((slot.label, complex(w[i]), frac))
    solver._tau_modes_found = (found, skipped)
    return found, skipped


def check_tau_modes(solver, where: str) -> str:
    """Refuse (UnsupportedError) when the solver's implicit step amplifies a spurious growing
    tau mode of a representative group; return the explain() note otherwise.

    Every mode found is tested: the amplification set of each scheme on the
    positive real axis is an interval (0, z*), so the decisive mode is the one with the smallest
    lambda dt, not the largest (the spurious lambda grows with l: 1070 at l = 1 ... 1135 at l = 5,
    N_r = 48, E = 1e-4)."""
    found, skipped = _find_tau_modes(solver)
    if not found:
        note = "spurious tau modes: none (representative groups checked)"
        return note + (f"; not checked: {', '.join(skipped)}" if skipped else "")
    amps = [implicit_amplification(solver, lam * solver.dt) for _, lam, _ in found]
    k = int(np.argmax(amps))
    (lab, lam, frac), amp = found[k], amps[k]
    text = (f"[{lab}] the implicit operator has a spurious growing tau mode, lambda = "
            f"{lam.real:.4g} ({100 * frac:.4f}% of its norm on the tau unknowns"
            + (f"; {len(found)} such modes found, this one the most amplified" if len(found) > 1
               else "") + ")")
    if amp > 1.0 + 1e-12:
        raise UnsupportedError(
            f"{text}: the {solver.scheme} step amplifies it by {amp:.3g} per step at dt = "
            f"{solver.dt:g} (lambda dt = {lam.real * solver.dt:.3g}); the run would blow up",
            where=where,
            why="a tau discretization can have a growing eigenmode that lives on the tau "
                "unknowns; in the first-order form of the spherical shell it appears when both "
                "taus are lifted into the same derivative basis (the form of the Dedalus shell "
                "examples, which has the same mode), from N_r ~ 44-48",
            fix="lift the tau of the momentum (or diffusion) equation one basis higher than the "
                "tau inside the gradient: lift(tau_2, basis, -1, lam=2) with "
                "lift(tau_1, basis, -1, lam=1) (no spurious mode, same exact Stokes eigenvalues)")
    return (f"{text}; damped by {solver.scheme} at dt = {solver.dt:g} (|R(lambda dt)| = "
            f"{amp:.3g}): another dt may make it grow -- lift tau_2 one basis higher than tau_1"
            + (f"; not checked: {', '.join(skipped)}" if skipped else ""))
