"""Floquet exponents of a linear periodic IVP in the spherical shell by the monodromy map.

The perturbation u of a time-periodic *axisymmetric* base flow (NCC of m = 0 only: the spherical
plans refuse any phi dependence of a coefficient) obeys a linear IVP whose azimuthal wavenumbers
decouple. For one stored wavenumber m >= 1 the discrete one-period map

    Phi_m : X_m(t0) -> X_m(t0 + P)        (nsteps steps of the solver's Runge--Kutta scheme)

is complex-linear on the unknowns of the groups of m; its eigenvalues lambda are the Floquet
multipliers of the *discrete* problem (same spatial discretization, same time stepper as the
IVP; continuous-time exponents up to the scheme's error, which converges with nsteps), and
mu = log(lambda) / P the exponents (Im mu folded into (-pi/P, pi/P]). They are computed by
implicitly restarted Arnoldi (ARPACK through SciPy, largest |lambda|) on P_a Phi_m P_a, with P_a
the exact admissible projection of the shell energy (SphereAdmissibleProjector: continuity and
boundary conditions; the taus free, so that every step output is admissible), the solver run on the CPU or the GPU (a matvec is
nsteps steps; vectors cross to the host only for ARPACK).

Reported, never hidden: each multiplier's residual ||Phi x - lambda x|| / ||Phi x|| recomputed with
the true map, the leak of Phi out of the groups of m (axisymmetry of the coefficients), the number
of map applications. Refused (clear error): m = 0 (the real-field reality conditions make Phi_0
only real-linear; not implemented), a nonlinear or affine right-hand side, a step that does not
divide the period, multistep schemes, non-shell geometries.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..errors import OptionError, UnsupportedError
from ..tolerances import REQUIRED, require
from .._deprecation import renames

if TYPE_CHECKING:
    from ..growth.periodic import FloquetExponents

__all__ = ["shell_floquet", "group_indices"]

_DOC = "docs/site/guide/shells.md"   # public guide page (was an internal design note)


def group_indices(solver, m: int) -> np.ndarray:
    """Positions in the global state of the unknowns of every group of azimuthal wavenumber m."""
    host = solver.system.host if hasattr(solver.system, "host") else solver.system
    idx = [np.arange(s.offset, s.offset + s.size) for s in host.slots if int(s.group.m) == int(m)]
    if not idx:
        raise OptionError(f"no group of azimuthal wavenumber m = {m} (stored m: "
                          f"{sorted({int(s.group.m) for s in host.slots})})", doc=_DOC)
    return np.concatenate(idx)


def _check(solver, period: float, steps: int, m: int) -> None:
    plan = solver.plan
    if not getattr(plan, "is_sphere", False) or not plan.space.shell:
        raise UnsupportedError("shell_floquet: a spherical-shell IVPSolver is expected")
    if solver._rk is None:
        raise UnsupportedError(f"shell_floquet: scheme {solver.scheme} is multistep (its one-period "
                               "map depends on the history)",
                               fix="an IMEX Runge--Kutta scheme (RK443)")
    if int(m) == 0:
        raise UnsupportedError(
            "shell_floquet: m = 0 is not implemented",
            why="for real fields the m = 0 modes carry reality conditions: the one-period map is "
                "only real-linear there (its eigen-decomposition needs the real form R^2n)",
            fix="m >= 1 (the axisymmetric base couples no m to another)")
    state = list(solver.system.variables)
    for i, e in enumerate(solver._rhs_exprs):
        if e is not None and not e.is_zero() and not e.is_linear_in(state):
            raise UnsupportedError(
                f"shell_floquet: right-hand side of equation {i} is not homogeneous linear in the "
                f"unknowns (degrees {sorted(e.degrees(state))})",
                fix="the linearized perturbation problem (no u.grad u, no forcing)")
    from ..timestep.integrator import is_whole
    if steps < 1 or not is_whole(period / solver.dt) or round(period / solver.dt) != steps:
        raise OptionError(f"shell_floquet: steps * dt = {steps * solver.dt!r} != period {period!r}",
                          fix=f"IVPSolver(..., dt=period / steps) (e.g. dt = {period / max(steps, 1)!r})",
                          doc=_DOC)


@renames("conatus.floquet.shell:shell_floquet")
def shell_floquet(solver, period: float, steps: int, *, m: int, n_modes: int = 6, t0: float = 0.0,
                  fields: tuple[str, ...] = ("u",), tol: float = REQUIRED, ncv: int | None = None,
                  seed: int = 0, max_iterations: int | None = None) -> FloquetExponents:
    """Floquet exponents of azimuthal wavenumber m of a linear period-``period`` shell IVP
    (module doc): the ``n_modes`` multipliers of largest modulus of the discrete one-period map
    (``steps`` steps of ``solver`` from ``t0``; dt = period / steps required)."""
    from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs

    from ..adjoint.sphere_energy import SphereAdmissibleProjector, SphereEnergyInnerProduct
    # local import: conatus.growth.periodic imports conatus.floquet, whose __init__ imports this
    # module; at module level `import conatus.growth` first failed (circular import)
    from ..growth.periodic import FloquetExponents, _fold
    require(tol, "tol", where="shell_floquet()", suggest="tol=1e-10",
            why="tol is the Arnoldi (ARPACK) stopping tolerance on the one-period map: it sets the "
                "accuracy of the returned multipliers; 1e-10 was the value used before v0.1")
    _check(solver, period, steps, m)
    xp = solver.xp
    n = solver.system.n
    idx = group_indices(solver, m)
    idx_d = xp.asarray(idx)
    k = idx.size
    # free_taus=True: the admissible set contains every step output (the taus absorb the
    # constraint residuals), so it is invariant under Phi and P_a Phi P_a has Phi's multipliers
    P = SphereAdmissibleProjector(SphereEnergyInnerProduct(solver, tuple(fields)), free_taus=True)
    dt, A = solver.dt, solver._rk
    count = {"maps": 0}

    def embed(v):
        X = xp.zeros(n, dtype=complex)
        X[idx_d] = xp.asarray(v)
        return X

    def phi(X):
        count["maps"] += 1
        for j in range(steps):
            X = solver._rk_step(X, t0 + j * dt, dt, A)        # stage times of IVPSolver.step
        return X

    def host(X):
        return X.get() if hasattr(X, "get") else np.asarray(X)

    leak = [0.0]

    def mv(v):
        X = phi(P(embed(np.asarray(v, dtype=complex).ravel())))
        Y = P(X)
        inside = Y[idx_d]
        out = Y.copy()
        out[idx_d] = 0                                   # the part outside the groups of m
        tot = float(xp.linalg.norm(Y))
        if tot > 0:
            leak[0] = max(leak[0], float(xp.linalg.norm(out)) / tot)
        return host(inside)

    if n_modes >= k - 1:
        raise OptionError(f"n_modes = {n_modes} >= {k - 1} (unknowns of m = {m} minus one)", doc=_DOC)
    v0 = np.random.default_rng(seed).standard_normal(k) + 1j * np.random.default_rng(seed + 1).standard_normal(k)
    v0 = host(P(embed(v0))[idx_d])
    op = LinearOperator((k, k), matvec=mv, dtype=complex)
    try:
        lam, vec = eigs(op, k=n_modes, which="LM", tol=tol, v0=v0, ncv=ncv, maxiter=max_iterations)
    except ArpackNoConvergence as err:
        raise OptionError(f"shell_floquet: Arnoldi did not converge ({len(err.eigenvalues)} of {n_modes} "
                          "multipliers)", fix="raise max_iterations or ncv, or loosen tol", doc=_DOC) from err
    res = []
    for j in range(len(lam)):
        x = embed(vec[:, j])
        Y = P(phi(P(x)))[idx_d]
        res.append(float(xp.linalg.norm(Y - lam[j] * xp.asarray(vec[:, j])) / max(float(xp.linalg.norm(Y)), 1e-300)))
    res = np.array(res)
    nz = np.abs(lam) > 0
    mu = _fold(np.log(lam[nz].astype(complex)) / period, period)
    order = np.argsort(-mu.real)
    return FloquetExponents(mu[order], res[nz][order], "monodromy (Arnoldi)", period, {
        "m": int(m), "nsteps": int(steps), "dt": dt, "scheme": solver.scheme, "t0": t0,
        "unknowns of m": int(k), "map applications": count["maps"],
        "leak of Phi out of m (axisymmetry)": f"{leak[0]:.1e}",
        "residual": "||Phi x - lambda x|| / ||Phi x||, true map, admissible projection",
        "backend": solver.backend})
