"""Exact coherent structures: Newton–Krylov–hookstep, equilibria, travelling waves,
relative / pre-periodic orbits and their stability. Notes:"""
from .edge import EdgeTracker, ThresholdBisection, UndecidedTrajectory
from .fourier_ops import FourierOps
from .ks import ks_solver
from .newton import NewtonKrylovHookstep, NewtonKrylovResult, NewtonNotConverged, arnoldi_gmres
from .solutions import (ECSSolution, Equilibrium, Flow, PreperiodicOrbit, RelativePeriodicOrbit,
                        TravellingWave, VectorField, equilibrium_eigenvalues, find_equilibrium,
                        find_relative_periodic_orbit, find_travelling_wave, floquet_multipliers,
                        symmetry_breaking_check)

__all__ = ["FourierOps", "NewtonKrylovHookstep", "NewtonKrylovResult", "NewtonNotConverged",
           "arnoldi_gmres", "ECSSolution", "Equilibrium", "TravellingWave", "RelativePeriodicOrbit",
           "PreperiodicOrbit", "Flow", "VectorField", "equilibrium_eigenvalues",
           "find_equilibrium", "find_travelling_wave", "find_relative_periodic_orbit",
           "floquet_multipliers", "symmetry_breaking_check", "ks_solver",
           "EdgeTracker", "ThresholdBisection", "UndecidedTrajectory"]
