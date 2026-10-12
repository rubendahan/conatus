"""Native bifurcation core on the Conatus IR.

Equilibrium branches with fold / branch-point / Hopf detection, automatic branch switching
(transcritical, pitchfork with declared symmetry, Hopf -> periodic orbits, period doubling),
periodic orbits by orthogonal collocation with Floquet multipliers (fold of cycles, branch point
of cycles, period doubling, Neimark-Sacker), codimension-2 fold and Hopf curves by minimally
augmented systems (cusp, Bogdanov-Takens, Bautin, fold-Hopf, Hopf-Hopf), normal forms on the
centre manifold (pitchfork cubic coefficient, first Lyapunov coefficient) from the IR's exact
multilinear derivatives, Landau and Ginzburg-Landau amplitude equations.
"""
from .amplitude import (GinzburgLandau, LandauEquation, amplitude_equation, cos_mode_observable,
                        ginzburg_landau)
from .branching import branch_switch, branching_directions, hopf_orbit_start
from .codim2 import continue_codim2
from .driver import ArclengthDriver, Curve, CurveEvent, CurvePoint
from .equilibria import continue_equilibria, generalized_eigenvalues, newton_equilibrium
from .normal_form import HopfNormalForm, ZeroEigenvalueNormalForm, normal_form
from .orbits import CollocationMesh, PeriodicOrbitProblem, continue_orbits
from .system import DynamicalSystem, ir_system, ode_problem

__all__ = ["DynamicalSystem", "ir_system", "ode_problem", "ArclengthDriver", "Curve", "CurveEvent",
           "CurvePoint", "continue_equilibria", "newton_equilibrium", "generalized_eigenvalues",
           "normal_form", "ZeroEigenvalueNormalForm", "HopfNormalForm", "CollocationMesh",
           "PeriodicOrbitProblem", "continue_orbits", "branch_switch", "branching_directions",
           "hopf_orbit_start", "continue_codim2", "LandauEquation", "amplitude_equation",
           "GinzburgLandau", "ginzburg_landau", "cos_mode_observable"]
