"""Sensitivity of global modes: structural sensitivity, base flow, steady forcing, parameters.

* ``mode_pair``: eigenvalue with its discrete (pencil) left/right vectors and physical direct and
  adjoint modes in the declared product (adjoint kind ``"discrete"`` or ``"continuous"``, no
  default);
* ``structural_sensitivity``: Giannetti & Luchini (2007) tensor, wavemaker, feedback drift;
* ``base_flow_sensitivity`` / ``steady_forcing_sensitivity``: Marquet, Sipp & Jacquin (2008),
  from the symbolic variation of the operator table, boundary terms kept;
* ``eigenvalue_sensitivity`` / ``parameter_sensitivity``: dσ/dp through one entry point over
  ``EVPResult``, ``ParametricEVP`` and ``ModePair``;
* ``linear_ginzburg_landau`` / ``ginzburg_landau_exact``: model with closed-form wavemaker
  (Chomaz 2005) used for validation.
"""
from .baseflow import (BaseFlowSensitivity, ForcingSensitivity, base_flow_sensitivity,
                       steady_forcing_sensitivity)
from .models import GinzburgLandauExact, ginzburg_landau_exact, linear_ginzburg_landau
from .modes import (ADJOINT_KINDS, ModePair, StructuralSensitivity, feedback_system, mode_pair,
                    structural_sensitivity)
from .parameters import EigenvalueSensitivity, eigenvalue_sensitivity, parameter_sensitivity

__all__ = ["ADJOINT_KINDS", "BaseFlowSensitivity", "EigenvalueSensitivity", "ForcingSensitivity",
           "GinzburgLandauExact", "ModePair", "StructuralSensitivity", "base_flow_sensitivity",
           "eigenvalue_sensitivity", "feedback_system", "ginzburg_landau_exact",
           "linear_ginzburg_landau", "mode_pair", "parameter_sensitivity",
           "steady_forcing_sensitivity", "structural_sensitivity"]
