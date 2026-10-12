"""Spectral discretization: Olver–Townsend operators, generalized tau, per-group plans."""
from . import ultraspherical
from .compiler import DiscretizationError
from .plan import GroupMatrices, GroupRank, LBVPSolution, Plan, RankError, discretize

__all__ = ["DiscretizationError", "GroupMatrices", "GroupRank", "LBVPSolution", "Plan",
           "RankError", "discretize", "ultraspherical"]

# The package called is its main function: cn.discretize(...) == conatus.expert.discretize(...).
from .._callable import make_callable  # noqa: E402

make_callable(__name__, discretize)
