"""A :class:`~conatus.bifurcation.DynamicalSystem` view of a 1D RealFourier IVP for the manifold
tools: the vector field v = M⁻¹(F − L X) of ``conatus.ecs.Flow`` in active coordinates
(conserved mean frozen, optionally the antisymmetric subspace R u = u of ``conatus.ecs``), with

* J(x): dense, one exact JVP (IR linearization) per column — for reduced systems only
  (size ≤ ``dense_max``, declared);
* B(x; u, w) = Jv(x + u) w − Jv(x) w — **exact for a quadratic v** (e.g. Kuramoto–Sivashinsky),
  C = 0. The declared ``degree`` = 2 is verified by a third finite difference along a random
  line (:func:`~conatus.manifolds.polynomial_degree_defect`), refused above ``degree_tol``.

The matching integrator is :func:`ecs_integrator` (exact discrete map of the solver in the same
coordinates).
"""
from __future__ import annotations

import numpy as np

from ..errors import OptionError, UnsupportedError

__all__ = ["ecs_system", "ecs_integrator"]


def _space(solver, odd: bool):
    from ..ecs.fourier_ops import FourierOps
    from ..ecs.solutions import _Space
    ops = FourierOps(solver)
    return ops, _Space(ops, solver.system.n, odd=odd)


def ecs_system(solver, *, odd: bool, degree: int, dense_max: int, degree_tol: float = 1e-8):
    """DynamicalSystem of the solver's vector field in active coordinates (see module doc)."""
    from ..bifurcation.system import DynamicalSystem
    from ..ecs.solutions import Flow
    from .parametrization import polynomial_degree_defect
    if degree != 2:
        raise UnsupportedError("ecs_system: only quadratic vector fields (degree = 2) are supported")
    ops, sp = _space(solver, odd)
    v = Flow(solver)
    n = int(sp.mask.sum())
    if n > dense_max:
        raise OptionError(f"ecs_system: {n} active dofs > dense_max = {dense_max} (dense Jacobian)")

    def G(x, par):
        return sp.active(v(sp.full(x)))

    def Jv(x, w, par):
        return sp.active(v.jvp(sp.full(x), sp.full(w)))

    def J(x, par):
        return np.column_stack([Jv(x, e, par) for e in np.eye(n)])

    def B(x, u, w, par):
        return Jv(np.asarray(x) + u, w, par) - Jv(x, w, par)

    def C(x, u, w, z, par):
        return np.zeros(n)

    def Gp(x, par, nm, order=1):
        raise OptionError("ecs_system: no parameters")

    ds = DynamicalSystem(n, params={}, origin=f"conatus.ecs.Flow of {solver.plan.problem.name} "
                         f"({sp.describe()})", G=G, J=J, Gp=Gp, JF_dir=Jv, B=B, C=C,
                         notes=[f"active coordinates: {sp.describe()}",
                                "B = Jv(x + u) w − Jv(x) w (exact for a quadratic field), C = 0"])
    X0 = np.zeros(solver.system.n)
    rng = np.random.default_rng(1)
    X0[sp.mask] = rng.standard_normal(n) * 0.5
    dd = polynomial_degree_defect(ds, sp.active(X0), 2)
    if dd > degree_tol:
        raise UnsupportedError(f"ecs_system: the vector field is not quadratic (defect {dd:.1e})")
    ds.notes.append(f"quadratic check: third finite difference defect {dd:.1e}")
    ds.space, ds.ops = sp, ops
    return ds


def ecs_integrator(solver, system, *, dt: float):
    """Exact discrete map of ``solver`` at step ``dt`` in the active coordinates of ``system``."""
    from .trajectory import SolverIntegrator
    sp = system.space
    return SolverIntegrator(solver, dt=dt, embed=sp.full, restrict=sp.active)
