"""Non-modal stability: transient growth and the continuous adjoint.

Everything is measured in a *declared* energy norm with the exact Chebyshev Gram:

* ``ModalSystem`` / ``parallel_shear_flow``: linear operator of one Fourier mode as a symbolic
  table; ``formal_adjoint`` derives the continuous adjoint (declared, checked BCs;
  ``show_equations``, ``show_boundary_conditions``, ``show_green_identity``);
* ``Descriptor`` (tau pencil M x' + L x = 0 through the IR) and ``StateSpace`` (generator Ã in
  energy-orthonormal coordinates, from an ordered QZ; no eigenvector expansion);
* ``energy_growth`` / ``max_growth`` (optimal transient growth), numerical and spectral
  abscissae;
* ``compare_adjoints`` (continuous vs discrete adjoint of an eigenmode), ``propagator_growth`` (the IMEX
  propagator of ``conatus.growth`` on the same norm).
"""
from .adjoint import (AdjointComparison, compare_adjoints, continuous_adjoint_mode,
                      discrete_adjoint_mode)
from .energy import (energy_growth, exp_norm, galerkin_numerical_abscissa, max_growth,
                            numerical_abscissa, restrict, spectral_abscissa)
from .statespace import StateSpace, state_space
from .system import Descriptor, ModalSystem, Y, parallel_shear_flow
from .transient import SteadyOperator, gram_energy_factor, propagator_growth, steady_operator

__all__ = ["AdjointComparison", "Descriptor", "ModalSystem", "StateSpace", "SteadyOperator", "Y",
           "compare_adjoints", "continuous_adjoint_mode", "discrete_adjoint_mode",
           "energy_growth", "exp_norm", "galerkin_numerical_abscissa", "gram_energy_factor",
           "max_growth", "numerical_abscissa", "parallel_shear_flow",
           "propagator_growth", "restrict",
           "spectral_abscissa", "state_space", "steady_operator"]
