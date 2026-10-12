"""Dedalus translation, part 2: the ``dedalus.public`` names known to the translator.

``_D3`` maps each ``d3.<name>`` (coordinates, distributor, bases, operators, problems) to Conatus,
or refuses it with the reason and the fix; the NumPy / SciPy whitelists for constants and initial
data. Public face and design: ``conatus.translate.dedalus``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..ir import (AnnulusBasis, CartesianCoordinates, Chebyshev,
                  ComplexFourier, Coordinate, CoordinateSystem, DiskBasis, Jacobi, Legendre, PolarCoordinates, RealFourier,
                  Ultraspherical, curl, div, dt,
                  grad, integ, interp, lap)
from ..ir import expr as E
from ..ir.bases import Basis
from ..ir.parse import _LiftBasis
from .dedalus_objects import (_DEDALUS_ONLY_SCHEMES, _SCHEMES, _Bundle, _Dist, _Scheme, _trace,
                              _Unsupported)

if TYPE_CHECKING:
    from .dedalus_interpreter import _Interpreter


# --------------------------------------------------------------------------- dedalus.public


class _D3:
    """The names of ``dedalus.public`` the translator knows, mapped to Conatus."""

    def __init__(self, tr: _Interpreter) -> None:
        self.tr = tr

    # coordinates, distributor --------------------------------------------------------------
    def Coordinate(self, name):
        return CartesianCoordinates(name).coords[0]

    def CartesianCoordinates(self, *names, right_handed=True):
        return CartesianCoordinates(*names)

    def PolarCoordinates(self, azimuth, radius):
        return PolarCoordinates(azimuth, radius)

    def S2Coordinates(self, *a, **k):
        raise _Unsupported(_SPHERE_NOTE)

    def SphericalCoordinates(self, *a, **k):
        raise _Unsupported(_SPHERE_NOTE)

    def Distributor(self, coords, dtype=np.float64, mesh=None, comm=None):
        cs = coords.system if isinstance(coords, Coordinate) else coords
        if not isinstance(cs, CoordinateSystem):
            raise _Unsupported("Distributor of an unknown coordinate object")
        if mesh is not None:
            self.tr.note("process mesh ignored: the parallel layout is not part of the problem")
        return _Dist(cs, np.dtype(dtype))

    # bases -----------------------------------------------------------------------------------
    def _dealias(self, dealias, what):
        if dealias is None:
            self.tr.note(f"{what}: Dedalus default dealias=1 written explicitly (Conatus has no "
                         "implicit dealias)")
            return 1
        if isinstance(dealias, (tuple, list)):
            if len(set(dealias)) != 1:
                raise _Unsupported("per-axis dealias tuple with different values")
            dealias = dealias[0]
        return dealias

    @staticmethod
    def _int(value, what):
        """A size or an index of the script is an integer: ``int()`` truncated 64.5 to 64 without
        a word (a float with an integer value, e.g. ``Nx = 2 * 32.0``, is accepted)."""
        import numbers
        # a number only: float("64") == int("64") would let a string through
        try:
            ok = isinstance(value, numbers.Real) and float(value) == int(value)
        except (TypeError, ValueError, OverflowError):
            ok = False
        if not ok or isinstance(value, bool):
            raise _Unsupported(f"{what} = {value!r} is not an integer (never rounded): write the "
                               "integer value")
        return int(value)

    def _interval(self, cls, coord, size, bounds, dealias, name):
        if not isinstance(coord, Coordinate):
            raise _Unsupported(f"{name}: the first argument must be a coordinate")
        if bounds is None:
            raise _Unsupported(f"{name} without bounds")
        return cls(coord, self._int(size, f"{name} size"), tuple(float(b) for b in bounds),
                   dealias=self._dealias(dealias, f"{name}({coord.name})"))

    def RealFourier(self, coord, size, bounds=None, dealias=None, library=None):
        return self._interval(RealFourier, coord, size, bounds, dealias, "RealFourier")

    def ComplexFourier(self, coord, size, bounds=None, dealias=None, library=None):
        self._nyquist(coord, size, bounds)
        return self._interval(ComplexFourier, coord, size, bounds, dealias, "ComplexFourier")

    def _nyquist(self, coord, size, bounds) -> None:
        """Declared convention difference, found by the Dedalus import tests."""
        L = float(bounds[1]) - float(bounds[0]) if bounds is not None else float("nan")
        self.tr.approx(f"ComplexFourier({getattr(coord, 'name', coord)}, {size}): Dedalus keeps "
                       f"the Nyquist mode k = -{int(size) // 2}*2pi/L (= {-np.pi * int(size) / L:.6g}); "
                       "Conatus ComplexFourier drops it, so that mode (a whole group/subproblem) "
                       "is absent from the translation; all other modes are identical")

    def ChebyshevT(self, coord, size, bounds=None, dealias=None, library=None):
        return self._interval(Chebyshev, coord, size, bounds, dealias, "ChebyshevT")

    Chebyshev = ChebyshevT

    def Legendre(self, coord, size, bounds=None, dealias=None, library=None):
        return self._interval(Legendre, coord, size, bounds, dealias, "Legendre")

    def Jacobi(self, coord, size, bounds=None, a=None, b=None, a0=None, b0=None, dealias=None,
               library=None):
        if a is None or b is None:
            raise _Unsupported("Jacobi without explicit a, b")
        if (a0 is not None and a0 != a) or (b0 is not None and b0 != b):
            raise _Unsupported("Jacobi with a0/b0 different from a/b (grid basis of another "
                               "family): not translated")
        if not isinstance(coord, Coordinate) or bounds is None:
            raise _Unsupported("Jacobi: a coordinate and bounds are required")
        return Jacobi(coord, self._int(size, "Jacobi size"), tuple(float(x) for x in bounds), a, b,
                      dealias=self._dealias(dealias, f"Jacobi({coord.name})"))

    def Ultraspherical(self, coord, size, bounds=None, alpha=None, dealias=None, library=None):
        """Dedalus Ultraspherical(alpha) = C^(alpha): a trial basis
        (discretize/jacobi_family.py); its derivative_basis(k) lifts at level k."""
        if alpha is None:
            raise _Unsupported("Ultraspherical without alpha")
        if not isinstance(coord, Coordinate) or bounds is None:
            raise _Unsupported("Ultraspherical: a coordinate and bounds are required")
        return Ultraspherical(coord, self._int(size, "Ultraspherical size"),
                              tuple(float(x) for x in bounds), lam=float(alpha),
                              dealias=self._dealias(dealias, f"Ultraspherical({coord.name})"))

    def ChebyshevU(self, coord, size, bounds=None, dealias=None, library=None):
        return self.Ultraspherical(coord, size, bounds, 1, dealias)

    def ChebyshevV(self, coord, size, bounds=None, dealias=None, library=None):
        # Dedalus: ChebyshevV = Ultraspherical(alpha=2), i.e. C^(2), not Chebyshev's 3rd kind V_n
        return self.Ultraspherical(coord, size, bounds, 2, dealias)

    def _polar(self, kind, coords, shape, radius=None, radii=None, dealias=None, dtype=None,
               **kw):
        if not isinstance(coords, CoordinateSystem) or coords.kind != "Polar":
            raise _Unsupported(f"{kind} on {coords!r}: only PolarCoordinates are translated")
        if kw:
            raise _Unsupported(f"{kind} options {sorted(kw)} not translated")
        nphi, nr = (self._int(s, f"{kind} shape") for s in shape)
        dl = self._dealias(dealias, kind)
        phi, r = coords.coords
        dtype = np.dtype(dtype if dtype is not None else np.float64)
        four = (ComplexFourier if dtype.kind == "c" else RealFourier)(
            phi, nphi, (0.0, 2 * np.pi), dealias=dl)
        if dtype.kind == "c":
            self._nyquist(phi, nphi, (0.0, 2 * np.pi))
        if kind == "DiskBasis":
            rad = DiskBasis(r, nr, float(1.0 if radius is None else radius), dealias=dl)
            self.tr.note("Dedalus DiskBasis = Zernike polynomials, Conatus DiskBasis = "
                         "r^|m| T_n(2r^2-1): the same polynomial space for each m (pipe-flow "
                         "EVP: eigenvalues equal to Dedalus's to 5e-14, run "
                         ")")
        else:
            rad = AnnulusBasis(r, nr, tuple(float(x) for x in radii), dealias=dl)
            self.tr.approx("Dedalus AnnulusBasis = Chebyshev with Dedalus's r-weighting of the "
                           "equations; Conatus AnnulusBasis multiplies the equations by a "
                           "declared power of r: same trial space, another projection")
        return _Bundle([four, rad], kind, rad)

    def DiskBasis(self, coords, shape, radius=1, **kw):
        return self._polar("DiskBasis", coords, shape, radius=radius, **kw)

    def AnnulusBasis(self, coords, shape, radii=(1, 2), **kw):
        return self._polar("AnnulusBasis", coords, shape, radii=radii, **kw)

    def SphereBasis(self, *a, **k):
        raise _Unsupported(_SPHERE_NOTE)

    ShellBasis = BallBasis = SphereBasis

    # operators -------------------------------------------------------------------------------
    def grad(self, A, coordsys=None):
        return grad(A)

    Gradient = grad

    def div(self, A):
        return div(A)

    Divergence = div

    def lap(self, A, coordsys=None):
        return lap(A)

    Laplacian = lap

    def curl(self, A):
        return curl(A)

    Curl = curl

    def dt(self, A):
        return dt(A)

    TimeDerivative = dt

    def Differentiate(self, A, coord):
        if not isinstance(coord, Coordinate):
            raise _Unsupported("Differentiate: the second argument must be a coordinate")
        return E.diff(A, coord)

    def integ(self, A, *coords):
        if len(coords) == 1 and isinstance(coords[0], (tuple, list)):
            coords = tuple(coords[0])
        if coords and any(isinstance(c, str) for c in coords):
            raise _Unsupported("integ with coordinate names as strings")
        return integ(A, coords or None)

    Integrate = integ

    def Interpolate(self, A, coord, position):
        return interp(A, coord, position)

    interp = Interpolate

    def Lift(self, A, basis, n):
        from ..ir import lift
        if isinstance(basis, _LiftBasis):
            return lift(A, basis.basis, self._int(n, "Lift index"), lam=basis.lam)
        if isinstance(basis, _Bundle):
            return lift(A, basis.radial, self._int(n, "Lift index"))
        if isinstance(basis, Basis):
            return lift(A, basis, self._int(n, "Lift index"))
        raise _Unsupported(f"Lift into {basis!r}")

    lift = Lift

    def dot(self, a, b):
        return E.dot(a, b)

    DotProduct = dot

    def cross(self, a, b):
        return E.cross(a, b)

    CrossProduct = cross

    def trace(self, A):
        return _trace(A, self.tr._notes)

    Trace = trace

    def _refuse(self, name, fix):
        def f(*a, **k):
            raise _Unsupported(f"d3.{name} is not translated: {fix}")
        return f

    # problems, solvers, runtime objects -------------------------------------------------------
    def IVP(self, variables, time=None, namespace=None):
        return self.tr.make_problem("IVP", variables, time=time)

    def LBVP(self, variables, namespace=None):
        return self.tr.make_problem("LBVP", variables)

    def EVP(self, variables, eigenvalue, namespace=None):
        return self.tr.make_problem("EVP", variables, eigenvalue=eigenvalue)

    def NLBVP(self, variables, namespace=None):
        return self.tr.make_problem("NLBVP", variables)

    def __getattr__(self, name: str):
        if name in _SCHEMES:
            return _Scheme(name)
        if name in _DEDALUS_ONLY_SCHEMES:
            raise _Unsupported(f"time stepper {name} is not in Conatus (available: "
                               f"{', '.join(_SCHEMES)}); choose one explicitly, results will "
                               "differ")
        refused = {
            "skew": "2D rotation of a vector; for the vorticity write curl(u)",
            "transpose": "no transpose-of-components node",
            "Average": "write integ(u, x)/Lx", "ave": "write integ(u, x)/Lx",
            "RadialComponent": "use comp(u, index)", "AngularComponent": "use comp(u, index)",
            "AzimuthalComponent": "use comp(u, index)",
            "GeneralFunction": "arbitrary Python functions are not part of the IR; a pointwise "
                               "function of one scalar field is: m.function(name, f, "
                               "derivative=...) or conatus.ir.pointwise",
            "CFL": "adaptive time step: Conatus never changes dt silently; give dt explicitly "
                   "(conatus.CFL(solver, 'u', safety=...) reports the CFL step as an indicator)",
            "GlobalFlowProperty": "runtime diagnostics: sim.properties(every=k, name=f) in the "
                                  "native layer (max/min/grid_average/volume_average), or "
                                  "conatus.runs.track / telemetry",
            "load_from_file": "checkpoints: sim.restart(path) / solver.load_checkpoint(path) "
                              "(cn.io, HDF5 or npz written by Conatus)",
        }
        if name in refused:
            if name in ("CFL", "GlobalFlowProperty"):
                def f(*a, **k):
                    raise _Unsupported(refused[name])
                return f
            return self._refuse(name, refused[name])
        raise _Unsupported(f"d3.{name} is not known to the translator")


_SPHERE_NOTE = ("spherical geometry (S2/Shell/Ball) exists in Conatus but the translator "
                "does not map Dedalus shapes yet (real SphereBasis: Lmax = Ntheta - 2 -> "
                "SphereBasis(theta, Ntheta - 1); radial conventions): write the bases by hand "
                "(the SphereBasis, ShellBasis and BallBasis docstrings give the correspondence)")

# NumPy / SciPy functions allowed on numbers and grid arrays (constants, initial data). On IR
# expressions the pointwise functions map to the Conatus nodes.
_NP_EXPR = {"sin": E.sin, "cos": E.cos, "exp": E.exp, "log": E.log, "tanh": E.tanh,
            "sqrt": E.sqrt, "abs": E.absolute, "absolute": E.absolute, "sign": E.sign,
            "conj": E.conj}
_NP_ALLOWED = {"sin", "cos", "tan", "exp", "log", "tanh", "sinh", "cosh", "arctan", "arcsin",
               "arccos", "sqrt", "abs", "absolute", "sign", "real", "imag", "conj", "pi", "e",
               "float64", "complex128", "float32", "complex64", "linspace", "arange", "array",
               "zeros_like", "ones_like", "zeros", "ones", "where", "heaviside", "maximum",
               "minimum", "exp2", "log10", "power", "inf", "newaxis", "isfinite", "sort",
               "argsort", "max", "min", "round", "floor", "ceil"}
_SCIPY_SPECIAL = {"jv", "yv", "iv", "kv", "erf", "erfc", "gamma", "jn_zeros", "airy"}
