"""Two-dimensional Kolmogorov flow for the ECS workflow.

Chandler & Kerswell (2013), J. Fluid Mech. 722, 554–595, §2 (p. 558): ∂u/∂t + u·∇u + ∇p =
(1/Re)∇²u + sin(n y) x̂ on the torus [0, 2π]², n = 4 in the paper. Streamfunction form here
(u = ψ_y, v = −ψ_x, ω = −∇²ψ):

    ∂t ∇²ψ − (1/Re) ∇⁴ψ + τ = −ψ_y (∇²ψ)_x + ψ_x (∇²ψ)_y + n cos(n y),     ∫ψ = 0,

τ a constant (gauge: the (0, 0) mode of the first equation only fixes τ = 0). Laminar state:
ψ_lam = −(Re / n³) cos(n y), i.e. u = (Re / n²) sin(n y). Every numerical choice is an argument.
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg as sla

from .._deprecation import renames
from ..discretize import discretize
from ..ir import IVP, CartesianCoordinates, Domain, Field, Parameter, RealFourier, diff, dt, integ, lap
from ..timestep import IVPSolver

__all__ = ["kolmogorov_solver", "laminar_state", "linear_stability_dense", "resample_state",
           "dissipation_ratio"]


@renames("conatus.ecs.kolmogorov:kolmogorov_solver")
def kolmogorov_solver(*, Re: float, n: int, N: int, dealias: float, scheme: str,
                      timestep: float | None = None) -> IVPSolver:
    """IVPSolver for Kolmogorov flow on [0, 2π]² with RealFourier(N) × RealFourier(N); ``timestep``
    is the step (``dt_``: its deprecated former name, v0.1)."""
    from .._deprecation import given            # dt_= (v0.1 name) is mapped by @renames
    dt_ = given("kolmogorov_solver", "timestep", timestep)
    cs = CartesianCoordinates("x", "y")
    x, y = cs.coords
    D = Domain(cs, [RealFourier(x, N, (0, 2 * np.pi), dealias=dealias),
                    RealFourier(y, N, (0, 2 * np.pi), dealias=dealias)])
    psi, tau, f = Field("psi", D), Field("tau", Domain(cs, [])), Field("f", D)
    nu = Parameter("nu", 1.0 / Re)
    pb = IVP([psi, tau], name=f"Kolmogorov 2D n={n} Re={Re:g}")
    q = lap(psi)
    pb.add_equation((dt(q) - nu * lap(q) + tau,
                     -diff(psi, y) * diff(q, x) + diff(psi, x) * diff(q, y) + f))
    pb.add_equation((integ(psi), 0))
    plan = discretize(pb, ncc={f: lambda X, Y: n * np.cos(n * Y)})
    return IVPSolver(plan, scheme, dt_)


def laminar_state(solver: IVPSolver, *, Re: float, n: int) -> np.ndarray:
    """ψ_lam = −(Re/n³) cos(n y), τ = 0, as a solver state vector."""
    g = solver.plan.domain
    shp = tuple(b.size for b in g.bases)
    c = np.zeros(shp)
    c[0, 2 * n] = -Re / n ** 3                 # RealFourier pair n along y: cos coefficient
    return solver.system.gather_state([c, np.zeros(1)])


def linear_stability_dense(solver: IVPSolver, X: np.ndarray, *, dense_max: int,
                           finite_max: float) -> np.ndarray:
    """Eigenvalues σ of (J_F(X) − L) v = σ M v, dense QZ (size ≤ ``dense_max``, explicit).
    M is singular (gauge / algebraic rows): infinite eigenvalues (|σ| > ``finite_max``, explicit)
    are removed and their number is reported in the returned array's ``n_infinite`` attribute."""
    from ..adjoint import FlowMap
    sysm = solver.system
    n = sysm.n
    if n > dense_max:
        from ..errors import ResolutionError
        raise ResolutionError(f"linear_stability_dense: n = {n} > dense_max = {dense_max}",
                              why="dense QZ costs O(n^3) time and O(n^2) memory",
                              fix=f"raise dense_max to {n} if that is affordable, or lower N")
    lin = FlowMap(solver, 1)
    J = np.column_stack([lin._JF(X, e) for e in np.eye(n)]) - sysm.L.toarray()
    a, b = sla.eig(J, sysm.M.toarray(), right=False, homogeneous_eigvals=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        sig = a / b
    ok = np.isfinite(sig) & (np.abs(sig) < finite_max)
    out = np.sort_complex(sig[ok])[::-1]
    out = out[np.argsort(-out.real)]
    res = np.asarray(out).view(_Eig)
    res.n_infinite = int((~ok).sum())
    return res


class _Eig(np.ndarray):
    """Finite eigenvalues; ``n_infinite`` counts the infinite ones of the same problem. A view,
    slice or copy keeps the count (it describes the solve, not the entries kept): without
    ``__array_finalize__`` a slice fell back to the class default 0."""
    n_infinite: int = 0

    def __array_finalize__(self, obj) -> None:
        if obj is not None:
            self.n_infinite = getattr(obj, "n_infinite", 0)


def resample_state(X: np.ndarray, src: IVPSolver, dst: IVPSolver, *,
                   report: dict | None = None) -> np.ndarray:
    """Copy the RealFourier coefficients of ψ (and τ) from ``src`` to ``dst`` (spectral
    interpolation: truncation or zero padding of the high modes; explicit, exact on the kept modes).

    What a truncation drops is measured: ``report`` (if given) receives ``kept``
    (coefficients kept per direction), ``truncated_max`` (max |a| dropped) and ``energy_lost_rel``
    (Σ a² dropped / Σ a², plain coefficient norm); a UserWarning is issued when it is non-zero."""
    a, t = src.system.scatter(X)
    na, nb = a.shape[0], dst.plan.domain.bases[0].size
    b = np.zeros((nb, nb))
    m = min(na, nb)
    b[:m, :m] = a[:m, :m]
    dropped = a.copy()
    dropped[:m, :m] = 0.0
    tot = float(np.sum(a ** 2))
    lost = float(np.sum(dropped ** 2)) / tot if tot > 0 else 0.0
    tmax = float(np.abs(dropped).max()) if dropped.size else 0.0
    if report is not None:
        report.update(kept=m, truncated_max=tmax, energy_lost_rel=lost)
    if lost > 0:
        warnings.warn(f"resample_state: truncation {na} -> {nb} drops coefficients up to "
                      f"{tmax:.2e} (relative coefficient energy lost {lost:.2e})", stacklevel=2)
    return dst.system.gather_state([b, np.asarray(t).reshape(1, 1)], check_subspace=False)


def dissipation_ratio(solver: IVPSolver, X: np.ndarray, *, Re: float, n: int) -> float:
    """D / D_lam with D = (1/Re)⟨|∇u|²⟩_V = (1/Re)⟨ω²⟩_V on the torus and D_lam = Re / (2n²)
    (Chandler & Kerswell 2013, eqs. 2.14–2.15). Exact Parseval sum on the RealFourier pairs:
    ⟨cos²⟩ = ⟨sin²⟩ = 1/2 for k ≠ 0, 1 for k = 0; ω̂ = |k|² ψ̂."""
    c = solver.system.scatter(X)[0]
    N = c.shape[0]
    k = np.repeat(np.arange(N // 2), 2)                  # wavenumber of each coefficient slot
    w = np.where(k == 0, 1.0, 0.5)
    om = (k[:, None] ** 2 + k[None, :] ** 2) * c
    mean_om2 = float(np.sum(om ** 2 * w[:, None] * w[None, :]))
    return mean_om2 / Re / (Re / (2 * n ** 2))
