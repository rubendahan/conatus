"""Phase-space geometry: trajectories with declared dense output, Poincaré sections and
first-return maps, unstable manifolds of equilibria (parametrization method), connections
between equilibria (shooting with projection boundary conditions), unstable manifolds of periodic
orbits (linear Floquet seeding) and localized-state snaking. Notes:"""
from .connections import (Connection, connection_vs_T, equilibrium_connection, find_connection,
                          projection_gap, scan_connection, unstable_left_basis)
from .ecs_system import ecs_integrator, ecs_system
from .orbits import first_return_stretch, orbit_manifold_seeds, section_projector
from .parametrization import UnstableManifold, globalize, polynomial_degree_defect, unstable_manifold
from .poincare import Crossings, PoincareSection, first_return_map, poincare_crossings
from .snaking import PinningRegion, even_subspace, pinning_region, restrict_system
from .trajectory import ODEIntegrator, SolverIntegrator, Trajectory

__all__ = ["ODEIntegrator", "SolverIntegrator", "Trajectory", "Crossings", "PoincareSection",
           "first_return_map", "poincare_crossings", "UnstableManifold", "globalize",
           "polynomial_degree_defect", "unstable_manifold", "Connection", "connection_vs_T",
           "find_connection", "projection_gap", "scan_connection", "unstable_left_basis",
           "equilibrium_connection",
           "ecs_system", "ecs_integrator", "orbit_manifold_seeds", "first_return_stretch",
           "section_projector",
           "PinningRegion", "even_subspace", "pinning_region", "restrict_system"]
