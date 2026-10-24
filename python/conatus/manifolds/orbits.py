"""Unstable manifolds of periodic orbits by linear Floquet seeding on a Poincaré section.
Notes: Budanur & Cvitanović, J. Stat. Phys. 167, 636
(2017), arXiv:1509.08133, eqs. (21)-(25).

Order 1 (declared, not a parametrization of the orbit's manifold): with x_p a point of the orbit
on the section, V the Floquet vector(s) of the unstable multiplier Λ of the monodromy and P the
projection **along the flow velocity v onto the section** (P = I − v nᵀ/(nᵀv) for a section of
normal n; P = I − v̂v̂ᵀ when n ∥ v, the choice of Budanur & Cvitanović), P V are the eigenvectors of
the linearized first-return map (the monodromy moves a point along v as well: measured, without P
the stretch of the return map is not Λ). The seeds of a fundamental domain are

* 1D (Λ real): x_p ± ε |Λ|^δ P V, δ ∈ [0, 1) (``n_delta`` points; for Λ < 0 each return flips
  the side, so the two signs are the same branch — stated in the notes);
* 2D (Λ complex): x_p + ε |Λ|^δ P (Re V cos φ + Im V sin φ), δ ∈ [0, 1), φ ∈ [0, 2π).

The error of the linear seeding is O(ε²); the linear consistency ‖P(x_p + εu) − x_p‖ / (ε‖u‖) → |Λ|
is measured by :func:`first_return_stretch`.
"""
from __future__ import annotations

import numpy as np

from ..errors import OptionError

__all__ = ["orbit_manifold_seeds", "first_return_stretch", "section_projector"]


def section_projector(velocity, normal=None) -> np.ndarray:
    """Projection along the flow velocity v onto the hyperplane of normal n (default n = v)."""
    v = np.asarray(velocity, float)
    n = v if normal is None else np.asarray(normal, float)
    den = float(n @ v)
    if abs(den) < 1e-12 * np.linalg.norm(n) * np.linalg.norm(v):
        raise OptionError("section_projector: the flow is tangent to the section (n·v = 0)")
    return np.eye(len(v)) - np.outer(v, n) / den


def orbit_manifold_seeds(x_p, V, Lambda, velocity, *, eps: float, n_delta: int,
                         n_phi: int | None = None, normal=None) -> tuple[np.ndarray, list[str]]:
    """Seeds of the fundamental domain (see module doc) and the notes that declare them;
    ``normal``: normal of the Poincaré section (default: the velocity itself)."""
    x_p = np.asarray(x_p, float)
    P = section_projector(velocity, normal)
    lam = complex(Lambda)
    deltas = np.arange(n_delta) / n_delta
    notes = [f"linear Floquet seeding, eps = {eps:g}, {n_delta} values of delta in [0, 1), "
             "projection along the flow velocity onto the section"
             + (" (normal = velocity)" if normal is None else "")]
    if abs(lam.imag) < 1e-12 * max(1.0, abs(lam)):
        u = P @ np.real(np.asarray(V))
        u = u / np.linalg.norm(u)
        seeds = [x_p + sgn * eps * abs(lam) ** d * u for sgn in (1.0, -1.0) for d in deltas]
        if lam.real < 0:
            notes.append("Λ < 0: each return flips the side (one branch, two signs)")
        return np.array(seeds), notes
    if n_phi is None:
        raise OptionError("orbit_manifold_seeds: complex Λ needs n_phi (angles of the 2D domain)")
    Vc = np.asarray(V, complex)
    Wr, Wi = P @ Vc.real, P @ Vc.imag
    scale = max(np.linalg.norm(Wr), np.linalg.norm(Wi))
    seeds = [x_p + eps * abs(lam) ** d * (np.cos(ph) * Wr + np.sin(ph) * Wi) / scale
             for d in deltas for ph in 2 * np.pi * np.arange(n_phi) / n_phi]
    notes.append(f"2D domain: {n_phi} angles in [0, 2π)")
    return np.array(seeds), notes


def first_return_stretch(return_map, x_p, u, eps: float) -> float:
    """‖P(x_p + εu) − x_p‖ / (ε‖u‖) for the first-return map ``return_map`` (→ |Λ| along an
    unstable Floquet direction as ε → 0)."""
    return float(np.linalg.norm(return_map(x_p + eps * u) - x_p) / (eps * np.linalg.norm(u)))
