"""Chaos: Lyapunov spectrum on the exact tangent of the step, covariant Lyapunov vectors,
Kaplan–Yorke dimension, finite-time exponents, Poincaré sections and first-return maps.
Notes:"""
from .lyapunov import Domination, LyapunovResult, clv_subspace_angles, kaplan_yorke, lyapunov_spectrum
from .poincare import PoincareResult, poincare_section
from .tangent import rk_tangent_step

__all__ = ["Domination", "LyapunovResult", "clv_subspace_angles", "kaplan_yorke", "lyapunov_spectrum",
           "PoincareResult", "poincare_section", "rk_tangent_step"]
