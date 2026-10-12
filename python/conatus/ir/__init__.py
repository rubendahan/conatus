"""Mathematical IR: coordinates, bases, domains, fields, expressions, problems."""
from .bases import (AnnulusBasis, BallBasis, Basis, Chebyshev, ChebyshevT, ChebyshevU, ChebyshevV,
                    ComplexFourier, DiskBasis,
                    CompoundChebyshev, Hermite, Jacobi, Laguerre, Legendre, RealFourier, ShellBasis,
                    SphereBasis, Ultraspherical)
from .coords import (CartesianCoordinates, Coordinate, CoordinateSystem, CylindricalCoordinates,
                     PolarCoordinates, S2Coordinates, SphericalCoordinates)
from .domain import Domain
from .expr import (Add, Component, Conj, Constant, Cross, Curl, Differentiate, Divergence, Dot, Expr, Field,
                   Gradient, Integrate, Interpolate, Laplacian, Lift, Mul, Outer, Parameter, Power,
                   TensorField, TimeDerivative, TransposeResult, UnaryFunction, UnitVector, VectorField,
                   Zero, absolute, add, as_expr, comp, conj, cos, cross, curl, diff, div, dot, dt, exp,
                   grad, has_nonpolynomial, integ, interp, lap, lift, log, mul, neg, outer, pair, power,
                   pointwise, sign, sin, sqrt, tanh, unit_vector)
from .problem import EVP, IVP, LBVP, NLBVP, Equation, NonlinearLHSError, Problem, ProblemError, RankReport
from .symmetry import (EVEN, ODD, SO2, AzimuthalMFold, EquatorialParity, Parity, Reflection, ShiftReflect,
                       SymmetryBreakingError, SymmetryError, SymmetrySubspace, Translation)

__all__ = [
    "Coordinate", "CoordinateSystem", "CartesianCoordinates", "PolarCoordinates",
    "CylindricalCoordinates", "S2Coordinates", "SphericalCoordinates", "Basis", "RealFourier",
    "ComplexFourier", "Jacobi", "Chebyshev", "ChebyshevT", "ChebyshevU", "ChebyshevV", "Legendre", "Ultraspherical", "DiskBasis",
    "AnnulusBasis", "SphereBasis", "ShellBasis", "BallBasis", "Hermite", "Laguerre",
    "CompoundChebyshev", "Domain",
    "Expr", "Field", "VectorField", "TensorField", "Parameter", "Constant", "Zero", "UnitVector",
    "Add", "Mul", "Dot", "Cross", "Outer", "Power", "Conj", "Component", "Differentiate", "Gradient",
    "Divergence", "Laplacian", "Curl", "TimeDerivative", "Integrate", "Interpolate", "Lift",
    "TransposeResult", "as_expr", "add", "mul", "neg", "dot", "cross", "outer", "power", "conj", "comp",
    "diff", "grad", "div", "lap", "curl", "dt", "integ", "interp", "lift", "pair", "unit_vector",
    "UnaryFunction", "exp", "log", "sin", "cos", "tanh", "sqrt", "absolute", "sign", "pointwise",
    "has_nonpolynomial", "Equation", "Problem", "IVP", "LBVP", "EVP", "NLBVP", "NonlinearLHSError", "ProblemError",
    "RankReport", "Parity", "EVEN", "ODD", "Reflection", "SymmetrySubspace", "SymmetryError",
    "SymmetryBreakingError", "Translation", "SO2", "ShiftReflect", "AzimuthalMFold",
    "EquatorialParity",
]

# Mapped Chebyshev bases on (semi-)infinite intervals (gallery gap, BACKLOG wave 1; anticipates)
from .maps import CoordinateMap, MappedChebyshev, RationalMap, TanhMap, TanMap  # noqa: E402

__all__ += ["CoordinateMap", "MappedChebyshev", "RationalMap", "TanhMap", "TanMap"]

# String equations (v0.1 UX): "dt(u) - nu*lap(u) = -u@grad(u)" -> the same IR, no eval
from .parse import parse_equation, parse_expr  # noqa: E402

__all__ += ["parse_equation", "parse_expr"]
