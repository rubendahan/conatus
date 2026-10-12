"""Named wall conditions of the native layer: ``cn.free_slip`` and ``cn.no_slip``.

``m.at("walls", u=cn.free_slip)`` on a velocity ``u`` means: no flow through the wall and no
tangential stress, for a plane Cartesian wall of normal n = e_z:

    u·n = 0,   dz(u_t) = 0   (each tangential component).

It is compiled as ONE vector condition on the single unknown u,

    B(u) = (I - n n)·dz(u) + n (n·u) = 0     at the wall,

so the vector-tau machinery of the native layer applies unchanged (one vector tau per wall in the
equation where u has its highest wall-normal order, and the divergence tau of the Cartesian
recipe). On a sine/cosine wall axis (``walls(a, b, basis="sincos")``) the same declaration gives
the velocity the parity that satisfies it exactly: tangential components cosine series, normal
component sine series (no tau). ``cn.no_slip`` is the zero vector (``u=0``), named for symmetry
of the API.

On a curved wall r = R (disk, annulus, pipe, ball, shell; normal e_r) the tangential stress
e_r·E·t of the strain rate E = (grad u + grad u^T)/2 has a curvature term: with u_r = 0 on the
wall, 2 e_r·E·t = r d/dr(u_t / r) on a curved direction (u_phi; u_theta, u_phi on a sphere) and
d/dr(u_z) on the straight axis of a pipe. Multiplied by R (polynomial, never divided by r) the
condition is again ONE vector condition at r = R,

    B(u) = R Pt·dr(u) + K·u = 0,    Pt = I - e_r e_r,    K = e_r e_r - Pc,

with Pc the projector on the curved tangential directions (Pc = Pt on the disk, annulus, ball and
shell; Pc = e_phi e_phi on a pipe): its normal component is u_r, its tangential components
R dr(u_t) - u_t (curved) or R dr(u_z) (pipe axis). Pt and K are rotation-invariant tensors given
on the wall (constant spin components); the curvilinear compilers apply them to the boundary trace
(polar_plan.boundary_tensor, sphere_plan.surface_tensor). Same tau machinery as no-slip (one
vector tau per wall) and spherical.md.

``cn.navier_slip(l)`` is the partial-slip condition of Navier on a plane
wall: no flow through the wall and a tangential slip proportional to the shear,

    u·n = 0,   u_t + l dn(u_t) = 0     (dn: derivative along the OUTWARD normal),

i.e. u_t = l dz(u_t) at a bottom wall and u_t = -l dz(u_t) at a top wall; l = 0 is no-slip and
l -> infinity free-slip. Again ONE vector condition, B(u) = (I - n n)·(u + s l dz(u)) + n (n·u)
with s = -1 at the left (bottom) wall and +1 at the right (top) wall, n = e_z. On a curved wall
r = R the tangential shear 2(E·n)_t carries the curvature term; with the free-slip operator
S(u) = R Pt·dr(u) + K·u above (normal u_r, tangential R 2(E·n)_t), the condition multiplied by R is
B(u) = s l S(u) + R Pt·u + (1 - s l)(I - Pt)·u = 0 (normal u_r, tangential R u_t + s l R 2(E·n)_t).
"""
from __future__ import annotations

from ..ir import diff, dot, unit_vector

__all__ = ["WallCondition", "NavierSlip", "free_slip", "no_slip", "navier_slip", "free_slip_operator",
           "navier_slip_operator", "curved_free_slip_operator", "curved_wall_tensors"]


class WallCondition:
    """A named wall condition on a vector unknown (``cn.free_slip``, ``cn.no_slip``)."""

    def __init__(self, name: str, meaning: str) -> None:
        self.name, self.meaning = name, meaning

    def __repr__(self) -> str:
        return f"cn.{self.name}"

    def __str__(self) -> str:
        return f"{self.name} ({self.meaning})"

    def explain(self) -> str:
        """The condition written out (every key object has one)."""
        return (f"cn.{self.name}: {self.meaning}, one vector condition on a velocity at a wall "
                "(m.at(wall, u=...)); its taus and the equations it becomes: m.explain(N)")


free_slip = WallCondition("free_slip", "u·n = 0, dn(u_t) = 0")
no_slip = WallCondition("no_slip", "u = 0")


class NavierSlip(WallCondition):
    """Navier partial slip of slip length ``length`` (``cn.navier_slip(l)``)."""

    def __init__(self, length: float) -> None:
        import math
        import numbers
        from ..errors import InputTypeError
        if isinstance(length, bool) or not isinstance(length, numbers.Real) or not length >= 0 \
                or not math.isfinite(length):
            raise InputTypeError(f"navier_slip({length!r}): the slip length is a finite number >= 0",
                                 fix="cn.free_slip for an infinite slip length, cn.no_slip (or u=0) "
                                     "for l = 0", doc="docs/site/guide/boundary_conditions.md")
        self.length = float(length)
        super().__init__("navier_slip", f"u·n = 0, u_t + {self.length:g} dn(u_t) = 0")

    def __repr__(self) -> str:
        return f"cn.navier_slip({self.length:g})"


def navier_slip(length: float) -> NavierSlip:
    """Navier partial-slip wall condition u·n = 0, u_t + l dn(u_t) = 0 (dn: outward normal
    derivative), for ``m.at(wall, u=cn.navier_slip(l))`` on plane walls."""
    return NavierSlip(length)


def free_slip_operator(u, coord):
    """B(u) = (I - n n)·dz(u) + n (n·u) with n = e_coord (a plane Cartesian wall): its normal
    component is u·n, its tangential components dz(u_t)."""
    n = unit_vector(coord)
    du = diff(u, coord)
    return du - n * dot(n, du) + n * dot(n, u)


def navier_slip_operator(u, coord, length: float, sign: float):
    """B(u) = (I - n n)·(u + s l dz(u)) + n (n·u), n = e_coord, s = sign of the outward normal
    along coord (-1 at the left wall, +1 at the right one)."""
    n = unit_vector(coord)
    v = u + (sign * length) * diff(u, coord)
    return v - n * dot(n, v) + n * dot(n, u)


def curved_wall_tensors(kind: str):
    """(Pt, K, meaning) of a curved wall of normal e_r: data callables of the grids returning the
    physical components (in the coordinate order of the geometry) of Pt = I - e_r e_r and
    K = e_r e_r - Pc (see the module docstring) and the printable condition (``{u}``: the
    unknown); ``kind`` is 'polar' (phi, r), 'cylinder'
    (phi, r, z) or 'spherical' (phi, theta, r)."""
    import numpy as np
    if kind == "polar":
        pt, k = np.diag([1.0, 0.0]), np.diag([-1.0, 1.0])
        meaning = "{u}·n = 0, r d/dr({u}_phi / r) = 0"
    elif kind == "cylinder":
        pt, k = np.diag([1.0, 0.0, 1.0]), np.diag([-1.0, 1.0, 0.0])
        meaning = "{u}·n = 0, r d/dr({u}_phi / r) = 0, d/dr({u}_z) = 0"
    elif kind == "spherical":
        pt, k = np.diag([1.0, 1.0, 0.0]), np.diag([-1.0, -1.0, 1.0])
        meaning = "{u}·n = 0, r d/dr({u}_t / r) = 0"
    else:
        raise ValueError(f"internal: unknown curved wall kind {kind!r}")   # not user-reachable

    def const(t):
        def data(*grids):
            shape = np.broadcast(*grids).shape
            return np.broadcast_to(t.reshape(t.shape + (1,) * len(shape)), t.shape + shape).copy()
        return data
    return const(pt), const(k), meaning


def curved_free_slip_operator(u, r, R: float, Pt, K):
    """B(u) = R Pt·dr(u) + K·u, to be taken at r = R (a curved wall of normal e_r): normal
    component u_r, tangential components R dr(u_t) - u_t on curved directions (R dr(u_z) on the
    axis of a pipe)."""
    return R * dot(Pt, diff(u, r)) + dot(K, u)
