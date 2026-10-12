"""Spectral bases (spec §10, §11).

Every basis carries its coordinate, interval, size, coefficient space and an
**explicit** dealiasing factor. There is no default dealias: omitting it is an
error, so no one inherits a silent 3/2 (spec §4.2, Dedalus pitfall P6).
"""
from __future__ import annotations

import math
import numbers
from fractions import Fraction
from typing import Any

from ..errors import CoordinateError, InputTypeError, OptionError, ResolutionError, did_you_mean
from ..tolerances import REQUIRED as _REQUIRED  # help() shows REQUIRED
from .coords import Coordinate

__all__ = [
    "Basis",
    "RealFourier",
    "ComplexFourier",
    "Jacobi",
    "Chebyshev",
    "ChebyshevT", "ChebyshevU", "ChebyshevV",
    "Legendre",
    "Ultraspherical",
    "DiskBasis",
    "AnnulusBasis",
    "SphereBasis",
    "ShellBasis",
    "BallBasis",
    "Hermite",
    "Laguerre",
    "CompoundChebyshev",
]


def _check_dealias(dealias: Any, cls: str) -> float:
    if dealias is _REQUIRED:
        raise OptionError(
            f"{cls}: dealias must be given explicitly",
            why="the dealiasing factor sets the grid of the nonlinear products; Conatus never "
                "picks it for you",
            fix="dealias=1 (no padding, linear problems) or dealias=3/2 (quadratic "
                "nonlinearities)", doc="docs/site/guide/bases.md")
    if isinstance(dealias, bool):
        raise InputTypeError(
            f"{cls}: dealias=True/False is ambiguous; give the factor (1, 3/2, 2, ...)",
            why="the dealiasing factor is a numerical choice (grid size = factor x N), never "
                "hidden behind a flag",
            fix="dealias=3/2 for quadratic nonlinearities (2 for cubic), dealias=1 for none")
    d = float(dealias)
    if d < 1:
        raise OptionError(f"{cls}: dealias factor must be >= 1, got {d}",
                          fix="dealias=1 (none) or dealias=3/2 (quadratic products)", doc="docs/site/guide/bases.md")
    return d


class Basis:
    """Base class for 1D spectral bases attached to one coordinate."""

    family = "abstract"
    periodic = False
    unbounded = False      # Hermite / Laguerre: infinite bounds are part of the basis

    GRIDS = ("floor", "smooth7")

    def __init__(self, coord: Coordinate, size: int, bounds: tuple[float, float],
                 dealias: Any = _REQUIRED, grid: str = "floor") -> None:
        if grid not in self.GRIDS:
            raise OptionError(f"grid must be one of {self.GRIDS}, got {grid!r}",
                              fix=f"grid={self.GRIDS[0]!r}" + did_you_mean(str(grid), self.GRIDS),
                              doc="docs/site/guide/bases.md")
        if not isinstance(coord, Coordinate):
            raise InputTypeError(f"basis coordinate must be a Coordinate, got {coord!r}",
                                 fix="cs = CartesianCoordinates('x', 'z'); Chebyshev(cs['z'], ...)",
                                 doc="docs/site/guide/bases.md")
        if int(size) != size or size < 1:
            raise ResolutionError(f"size must be a positive integer, got {size!r}",
                                  fix="an integer number of modes, e.g. 64", doc="docs/site/guide/bases.md")
        a, b = (float(bounds[0]), float(bounds[1]))
        if not b > a:
            raise CoordinateError(f"bounds must satisfy a < b, got {bounds!r}",
                                  fix="bounds=(a, b) with a < b, e.g. (0, 2*np.pi)", doc="docs/site/guide/bases.md")
        if not (self.unbounded or getattr(self, "map", None) is not None
                or (math.isfinite(a) and math.isfinite(b))):
            # Chebyshev(z, N, (0, inf)) and RealFourier with an
            # infinite period were accepted (grids of NaN later); unbounded axes are declared
            raise CoordinateError(f"{type(self).__name__}: bounds must be finite, got {bounds!r}",
                                  fix="Hermite (whole line), Laguerre (half line) or MappedChebyshev "
                                      "(TanhMap, TanMap, RationalMap) for an unbounded axis",
                                  doc="docs/site/guide/bases.md")
        self.coord = coord
        self.size = int(size)
        self.bounds = (a, b)
        self.dealias = _check_dealias(dealias, type(self).__name__)
        # "floor": M = floor(N * dealias) points (Dedalus convention, parity tests).
        # "smooth7": smallest M >= floor(N * dealias) whose FFT length (fft_length) has no prime
        # factor > 7 -- dealiasing stays exact (M only grows), FFTs avoid large-prime lengths
        # (e.g. N = 129, 3/2: M = 193, FFT length 386 = 2 * 193 -> M = 200, 400). Declared, never
        # implicit: the default is "floor".
        self.grid = grid

    @property
    def length(self) -> float:
        return self.bounds[1] - self.bounds[0]

    def fft_length(self, M: int) -> int:
        """Length of the FFT that transforms an M-point grid (Fourier: M)."""
        return M

    @property
    def grid_size(self) -> int:
        """Number of points of the dealiased grid: floor(N*d) (Dedalus), or the smallest M above
        it with a 7-smooth FFT length when ``grid="smooth7"`` was declared."""
        M = int(math.floor(self.size * self.dealias))
        if self.grid == "smooth7":
            while not _smooth7(self.fft_length(M)):
                M += 1
        return M

    def _params(self) -> tuple:
        return ()

    @property
    def key(self) -> tuple:
        return (type(self).__name__, self.coord.key, self.size, self.bounds,
                self.dealias) + self._params() + ((self.grid,) if self.grid != "floor" else ())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Basis) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def coeff_space(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f"{type(self).__name__}({self.coord.name}, size={self.size}, "
                f"bounds={self.bounds}, dealias={_frac(self.dealias)}"
                + (f", grid={self.grid!r}" if self.grid != "floor" else "") + ")")

    def explain(self) -> str:
        return "\n".join([
            repr(self),
            f"  family         : {self.family}{' (periodic)' if self.periodic else ''}",
            f"  coefficients   : {self.coeff_space()}",
            f"  dealias        : {_frac(self.dealias)} (explicit; grid size {self.grid_size}"
            + (f", grid='smooth7': floor(N*d) = {int(math.floor(self.size * self.dealias))} raised "
               f"to a 7-smooth FFT length {self.fft_length(self.grid_size)}"
               if self.grid == "smooth7" else ", floor(N*d)") + ")",
        ])


def _smooth7(n: int) -> bool:
    for q in (2, 3, 5, 7):
        while n % q == 0:
            n //= q
    return n == 1


def _frac(x: float) -> str:
    f = Fraction(x).limit_denominator(16)
    return str(f) if abs(float(f) - x) < 1e-14 else repr(x)


class _Fourier(Basis):
    periodic = True

    def __init__(self, coord: Coordinate, size: int, bounds: tuple[float, float],
                 dealias: Any = _REQUIRED, grid: str = "floor") -> None:
        super().__init__(coord, size, bounds, dealias, grid)

    def wavenumbers(self) -> str:
        return f"k = 2*pi*n/{self.length:g}"


class RealFourier(_Fourier):
    """Real Fourier series (cos/sin pairs) for real-valued data."""

    family = "Fourier (real: cos/sin)"

    def coeff_space(self) -> str:
        return (f"{{cos(k x), -sin(k x)}}, n = 0..{self.size // 2 - 1} ({self.size} real "
                f"coefficients), {self.wavenumbers()}")


class ComplexFourier(_Fourier):
    """Complex Fourier series exp(i k x)."""

    family = "Fourier (complex)"

    def coeff_space(self) -> str:
        return (f"exp(i k x), n = {-(self.size // 2)}..{(self.size - 1) // 2}, "
                f"{self.wavenumbers()}")


def _affine(a: float, length: float) -> str:
    """'2(x - a)/L - 1' printed without '- -1' (BACKLOG UX: 't = 2(x - -1)/2 - 1')."""
    if a == -1 and length == 2:
        return "x"
    if a == 0:
        return f"2x/{length:g} - 1"
    shift = f"x + {-a:g}" if a < 0 else f"x - {a:g}"
    return f"2({shift})/{length:g} - 1"


class Jacobi(Basis):
    """Jacobi polynomials P_n^(a,b) on a bounded interval (mapped from [-1, 1])."""

    family = "Jacobi"

    def __init__(self, coord: Coordinate, size: int, bounds: tuple[float, float],
                 a: float, b: float, dealias: Any = _REQUIRED, grid: str = "floor") -> None:
        super().__init__(coord, size, bounds, dealias, grid)
        if a <= -1 or b <= -1:
            raise OptionError(f"Jacobi parameters must satisfy a, b > -1, got a={a!r}, b={b!r}",
                              fix="a = b = -1/2 is Chebyshev, a = b = 0 Legendre", doc="docs/site/guide/bases.md")
        self.a = float(a)
        self.b = float(b)

    def _params(self) -> tuple:
        return (self.a, self.b)

    def coeff_space(self) -> str:
        return (f"P_n^({_frac(self.a)},{_frac(self.b)})(t), n = 0..{self.size - 1}, "
                f"t = {_affine(self.bounds[0], self.length)}")

    def __repr__(self) -> str:
        return (f"{type(self).__name__}({self.coord.name}, size={self.size}, "
                f"bounds={self.bounds}, a={_frac(self.a)}, b={_frac(self.b)}, "
                f"dealias={_frac(self.dealias)})")


class Chebyshev(Jacobi):
    """Chebyshev T_n = Jacobi(-1/2, -1/2) up to normalisation."""

    family = "Chebyshev T (Jacobi -1/2,-1/2)"

    def __init__(self, coord: Coordinate, size: int, bounds: tuple[float, float],
                 dealias: Any = _REQUIRED, grid: str = "floor") -> None:
        super().__init__(coord, size, bounds, -0.5, -0.5, dealias, grid)

    def fft_length(self, M: int) -> int:
        """Gauss–Chebyshev grid: the DCT runs as a real FFT of the even extension, length 2M."""
        return 2 * M

    def coeff_space(self) -> str:
        return (f"T_n(t), n = 0..{self.size - 1}, "
                f"t = {_affine(self.bounds[0], self.length)}")

    def __repr__(self) -> str:
        return Basis.__repr__(self)


ChebyshevT = Chebyshev


class Legendre(Jacobi):
    """Legendre P_n = Jacobi(0, 0)."""

    family = "Legendre (Jacobi 0,0)"

    def __init__(self, coord: Coordinate, size: int, bounds: tuple[float, float],
                 dealias: Any = _REQUIRED) -> None:
        super().__init__(coord, size, bounds, 0.0, 0.0, dealias)

    def coeff_space(self) -> str:
        return (f"P_n(t), n = 0..{self.size - 1}, "
                f"t = {_affine(self.bounds[0], self.length)}")

    def __repr__(self) -> str:
        return Basis.__repr__(self)


class Ultraspherical(Jacobi):
    """Ultraspherical (Gegenbauer) C_n^(lambda) = Jacobi(lambda-1/2, lambda-1/2) up to norm."""

    family = "Ultraspherical (Gegenbauer)"

    def __init__(self, coord: Coordinate, size: int, bounds: tuple[float, float],
                 lam: float, dealias: Any = _REQUIRED) -> None:
        if lam <= -0.5:
            raise OptionError("ultraspherical parameter must satisfy lambda > -1/2",
                              fix="lambda = 0 is Chebyshev T, 1 Chebyshev U, 1/2 Legendre", doc="docs/site/guide/bases.md")
        super().__init__(coord, size, bounds, lam - 0.5, lam - 0.5, dealias)
        self.lam = float(lam)

    def coeff_space(self) -> str:
        return (f"C_n^({_frac(self.lam)})(t), n = 0..{self.size - 1}, "
                f"t = {_affine(self.bounds[0], self.length)}")

    def __repr__(self) -> str:
        return (f"Ultraspherical({self.coord.name}, size={self.size}, bounds={self.bounds}, "
                f"lam={_frac(self.lam)}, dealias={_frac(self.dealias)})")


def ChebyshevU(coord: Coordinate, size: int, bounds: tuple[float, float],
               dealias: Any = _REQUIRED) -> Ultraspherical:
    """Chebyshev polynomials of the second kind U_n = C_n^(1) (Dedalus ``ChebyshevU``)."""
    return Ultraspherical(coord, size, bounds, lam=1, dealias=dealias)


def ChebyshevV(coord: Coordinate, size: int, bounds: tuple[float, float],
               dealias: Any = _REQUIRED) -> Ultraspherical:
    """Gegenbauer C_n^(2), under Dedalus's name ``ChebyshevV`` (= Ultraspherical(alpha=2)).
    Not the Chebyshev polynomials of the third kind V_n of the literature:
    the Dedalus name is kept for scripts that use it; ``Ultraspherical(..., lam=2)`` says it."""
    return Ultraspherical(coord, size, bounds, lam=2, dealias=dealias)


class _Radial(Basis):
    """Radial part of a polar (or cylindrical) basis: its coefficient space depends on the
    azimuthal wavenumber m of each group, so it only makes sense on PolarCoordinates /
    CylindricalCoordinates next to a Fourier basis in phi."""

    def __init__(self, coord: Coordinate, size: int, bounds: tuple[float, float],
                 dealias: Any = _REQUIRED) -> None:
        if not coord.system.curvilinear or coord.system.kind not in ("Polar", "Cylindrical"):
            raise CoordinateError(f"{type(self).__name__} is a radial basis: its coordinate must be "
                                  "the radius of PolarCoordinates or CylindricalCoordinates",
                                  fix="cs = PolarCoordinates('phi', 'r'); "
                                      f"{type(self).__name__}(cs['r'], ...)", doc="docs/site/guide/bases.md")
        if coord.index != 1:
            raise CoordinateError(f"{type(self).__name__} must be attached to the radial coordinate "
                                  f"({coord.system.names[1]}), got {coord.name}",
                                  fix=f"{type(self).__name__}(cs[{coord.system.names[1]!r}], ...)",
                                  doc="docs/site/guide/bases.md")
        if int(size) < 4:
            raise ResolutionError(f"{type(self).__name__}: size must be >= 4, got {size!r}",
                                  fix="a radial size of at least 4 (tau rows need modes)", doc="docs/site/guide/bases.md")
        super().__init__(coord, size, bounds, dealias)


class DiskBasis(_Radial):
    """Radial basis of the full disk r in [0, R], regular at the axis (Vasil et al., JCP 2016).

    A spin component of effective wavenumber k (k = m for a scalar, m +- 1 for u_+- of a vector) is
    rho^|k| w(z), with rho = r/R, z = 2 rho^2 - 1 and w a series of Jacobi polynomials
    P_n^(0,|k|)(z) (Zernike polynomials), n < size - floor(|m|/2) (Dedalus's triangular truncation;
    ``polar_radial.DiskOps``): every representable function is smooth at r = 0 and no axis
    condition is ever imposed. (The standalone toolkit ``discretize.polar.Disk`` keeps a Chebyshev
    T series in z; the IR plans do not use it for the disk.)"""

    family = "disk radial (regular: rho^|k| P_n^(0,|k|)(2 rho^2 - 1), Zernike)"

    def __init__(self, coord: Coordinate, size: int, radius: float = 1.0,
                 dealias: Any = _REQUIRED) -> None:
        if not (radius > 0 and math.isfinite(radius)):
            raise CoordinateError(f"DiskBasis radius must be a finite number > 0, got {radius!r}",
                                  fix="radius=1.0", doc="docs/site/guide/bases.md")
        super().__init__(coord, size, (0.0, float(radius)), dealias)

    @property
    def radius(self) -> float:
        return self.bounds[1]

    def coeff_space(self) -> str:
        return (f"rho^|k| P_n^(0,|k|)(z), n < {self.size} - floor(|m|/2), rho = r/{self.radius:g}, "
                "z = 2 rho^2 - 1, k = m (+-1 per spin index); radial grid: Gauss-Legendre nodes in z")

    def __repr__(self) -> str:
        return (f"DiskBasis({self.coord.name}, size={self.size}, radius={self.radius:g}, "
                f"dealias={_frac(self.dealias)})")


class AnnulusBasis(_Radial):
    """Radial basis of the annulus r in [ri, ro] (0 < ri): Chebyshev T_n(t), r = c + h t.
    Equations are discretized in a declared r-multiplied form (r^2 lap, r grad, ...)."""

    family = "annulus radial (Chebyshev T in t, r-multiplied operators)"

    def __init__(self, coord: Coordinate, size: int, radii: tuple[float, float],
                 dealias: Any = _REQUIRED) -> None:
        ri, ro = float(radii[0]), float(radii[1])
        if not 0 < ri < ro:
            raise CoordinateError(f"AnnulusBasis needs 0 < ri < ro, got {radii!r}",
                                  fix="radii=(ri, ro) with 0 < ri < ro; DiskBasis when the axis is "
                                      "included", doc="docs/site/guide/bases.md")
        super().__init__(coord, size, (ri, ro), dealias)

    def fft_length(self, M: int) -> int:
        return 2 * M

    def coeff_space(self) -> str:
        return (f"T_n(t), n = 0..{self.size - 1}, t = 2(r - {self.bounds[0]:g})/{self.length:g} - 1 "
                "(spin components for vectors); Gauss-Chebyshev radial grid")

    def __repr__(self) -> str:
        return (f"AnnulusBasis({self.coord.name}, size={self.size}, radii={self.bounds}, "
                f"dealias={_frac(self.dealias)})")


# ----------------------------------------------------------------------------------------------
# Spherical bases
# ----------------------------------------------------------------------------------------------

class SphereBasis(Basis):
    """Colatitude part of the spin-weighted spherical harmonics (Vasil et al. 2019), attached to theta
    of S2Coordinates or SphericalCoordinates, next to a Fourier basis in phi.

    ``size`` = Lmax + 1 degrees l = 0..Lmax; a spin-s component of azimuthal mode m holds the degrees
    l >= max(|m|, |s|) (triangular truncation). ``radius`` is the sphere's radius on S2Coordinates
    (ignored in SphericalCoordinates, where the radius is the coordinate r). Grid: Gauss-Legendre nodes
    in cos(theta), floor(size * dealias) of them. Note: Dedalus's real SphereBasis of shape
    (Nphi, Ntheta) has Lmax = Ntheta - 2, i.e. size = Ntheta - 1 here (declared, never inferred)."""

    family = "spin-weighted spherical harmonics (colatitude part, Jacobi in cos theta)"

    def __init__(self, coord: Coordinate, size: int, radius: float = 1.0,
                 dealias: Any = _REQUIRED) -> None:
        if not coord.system.curvilinear or coord.system.kind not in ("S2", "Spherical"):
            raise CoordinateError("SphereBasis is attached to the colatitude of S2Coordinates or "
                                  "SphericalCoordinates",
                                  fix="cs = S2Coordinates('phi', 'theta'); SphereBasis(cs['theta'], ...)",
                                  doc="docs/site/guide/bases.md")
        if coord.index != 1:
            raise CoordinateError(f"SphereBasis must be attached to the colatitude "
                                  f"({coord.system.names[1]}), got {coord.name}",
                                  fix=f"SphereBasis(cs[{coord.system.names[1]!r}], ...)", doc="docs/site/guide/bases.md")
        if not (radius > 0 and math.isfinite(radius)):
            raise CoordinateError(f"SphereBasis radius must be a finite number > 0, got {radius!r}",
                                  fix="radius=1.0", doc="docs/site/guide/bases.md")
        super().__init__(coord, size, (0.0, math.pi), dealias)
        self.radius = float(radius)

    @property
    def Lmax(self) -> int:
        return self.size - 1

    def _params(self) -> tuple:
        return (self.radius,)

    def coeff_space(self) -> str:
        return (f"spin-weighted harmonics sY_lm(theta), l = max(|m|,|s|)..{self.Lmax} (triangular), "
                f"sphere radius {self.radius:g}; Gauss-Legendre grid in cos(theta)")

    def __repr__(self) -> str:
        return (f"SphereBasis({self.coord.name}, size={self.size}, radius={self.radius:g}, "
                f"dealias={_frac(self.dealias)})")


class _SphericalRadial(Basis):
    """Radial part of a spherical basis (attached to r of SphericalCoordinates)."""

    def __init__(self, coord: Coordinate, size: int, bounds: tuple[float, float],
                 dealias: Any = _REQUIRED) -> None:
        if not coord.system.curvilinear or coord.system.kind != "Spherical":
            raise CoordinateError(f"{type(self).__name__} is attached to the radius of "
                                  "SphericalCoordinates",
                                  fix="cs = SphericalCoordinates('phi', 'theta', 'r'); "
                                      f"{type(self).__name__}(cs['r'], ...)", doc="docs/site/guide/bases.md")
        if coord.index != 2:
            raise CoordinateError(f"{type(self).__name__} must be attached to the radius "
                                  f"({coord.system.names[2]}), got {coord.name}",
                                  fix=f"{type(self).__name__}(cs[{coord.system.names[2]!r}], ...)",
                                  doc="docs/site/guide/bases.md")
        if int(size) < 4:
            raise ResolutionError(f"{type(self).__name__}: size must be >= 4, got {size!r}",
                                  fix="a radial size of at least 4 (tau rows need modes)", doc="docs/site/guide/bases.md")
        super().__init__(coord, size, bounds, dealias)


class ShellBasis(_SphericalRadial):
    """Spherical shell ri <= r <= ro (0 < ri): Chebyshev T_n(t), r = c + h t, for every spin component;
    equations in a declared r-multiplied form (each row times the smallest power of r that makes its
    coefficients polynomial, as Dedalus's r^-k Jacobi(k-1/2, k-1/2) derivative bases)."""

    family = "shell radial (Chebyshev T in t, r-multiplied operators)"

    def __init__(self, coord: Coordinate, size: int, radii: tuple[float, float],
                 dealias: Any = _REQUIRED) -> None:
        ri, ro = float(radii[0]), float(radii[1])
        if not 0 < ri < ro:
            raise CoordinateError(f"ShellBasis needs 0 < ri < ro, got {radii!r}",
                                  fix="radii=(ri, ro) with 0 < ri < ro; BallBasis when the centre is "
                                      "included", doc="docs/site/guide/bases.md")
        super().__init__(coord, size, (ri, ro), dealias)

    def fft_length(self, M: int) -> int:
        return 2 * M

    def coeff_space(self) -> str:
        return (f"T_n(t), n = 0..{self.size - 1}, t = 2(r - {self.bounds[0]:g})/{self.length:g} - 1 "
                "(spin components); Gauss-Chebyshev radial grid")

    def __repr__(self) -> str:
        return (f"ShellBasis({self.coord.name}, size={self.size}, radii={self.bounds}, "
                f"dealias={_frac(self.dealias)})")


class BallBasis(_SphericalRadial):
    """Full ball 0 <= r <= R, regular at the centre (Vasil et al. 2019, Lecoanet et al. 2019): the
    regularity component sigma of degree l is rho^k Q_n^(a, k + 1/2)(2 rho^2 - 1), k = l + sigma,
    n < size - floor(l/2) (Dedalus's BallBasis truncation), a = derivative level (0 for fields).
    Gauss-Jacobi (0, 1/2) radial grid in z = 2 rho^2 - 1."""

    family = "ball radial (regular 3D Zernike: rho^k P_n^(a, k+1/2)(2 rho^2 - 1))"

    def __init__(self, coord: Coordinate, size: int, radius: float = 1.0,
                 dealias: Any = _REQUIRED) -> None:
        if not (radius > 0 and math.isfinite(radius)):
            raise CoordinateError(f"BallBasis radius must be a finite number > 0, got {radius!r}",
                                  fix="radius=1.0", doc="docs/site/guide/bases.md")
        super().__init__(coord, size, (0.0, float(radius)), dealias)

    @property
    def radius(self) -> float:
        return self.bounds[1]

    def coeff_space(self) -> str:
        return (f"rho^k Q_n^(a, k+1/2)(z), n < {self.size} - floor(l/2), rho = r/{self.radius:g}, "
                "z = 2 rho^2 - 1, k = l + sigma (regularity components); Gauss-Jacobi radial grid")

    def __repr__(self) -> str:
        return (f"BallBasis({self.coord.name}, size={self.size}, radius={self.radius:g}, "
                f"dealias={_frac(self.dealias)})")


# ----------------------------------------------------------------------------------------------
# Unbounded and compound intervals (docs/mathematics/
# unbounded_compound.md). Their sparse operators and transforms: conatus.discretize.families.
# ----------------------------------------------------------------------------------------------

class Hermite(Basis):
    """Hermite functions on the whole line: f(x) = sum_n c_n psi_n((x - center)/scale), with
    psi_n(y) = H_n(y) exp(-y^2/2) / sqrt(2^n n! sqrt(pi)) (orthonormal). Every representable
    function decays like a Gaussian, so no boundary condition exists or is imposed. ``scale`` is
    the declared width of the envelope (a numerical choice like N, printed by explain()).

    Operators are banded: d/dx and multiplication by x are tridiagonal, a polynomial coefficient
    of degree d has half-bandwidth d. Grid: Gauss-Hermite nodes, floor(N * dealias) of them.
    A product of two Hermite functions is not a finite Hermite series (its envelope is
    exp(-y^2)), so products projected by quadrature are accurate, not exact (declared)."""

    family = "Hermite functions (whole line)"
    unbounded = True

    def __init__(self, coord: Coordinate, size: int, center: float = 0.0, scale: float = 1.0,
                 dealias: Any = _REQUIRED) -> None:
        if not (math.isfinite(scale) and scale > 0):
            raise CoordinateError(f"Hermite: scale must be a finite number > 0, got {scale!r}",
                                  fix="scale=1.0 (functions exp(-(x - center)^2 / (2 scale^2)) H_n)",
                                  doc="docs/site/guide/bases.md")
        if not math.isfinite(center):
            raise CoordinateError(f"Hermite: center must be finite, got {center!r}",
                                  fix="center=0.0", doc="docs/site/guide/bases.md")
        super().__init__(coord, size, (-math.inf, math.inf), dealias)
        self.center, self.scale = float(center), float(scale)

    def _params(self) -> tuple:
        return (self.center, self.scale)

    def coeff_space(self) -> str:
        return (f"psi_n(y), n = 0..{self.size - 1}, y = (x - {self.center:g})/{self.scale:g} "
                "(orthonormal Hermite functions); Gauss-Hermite grid")

    def __repr__(self) -> str:
        return (f"Hermite({self.coord.name}, size={self.size}, center={self.center:g}, "
                f"scale={self.scale:g}, dealias={_frac(self.dealias)})")


class Laguerre(Basis):
    """Laguerre functions on the half line x >= start: f(x) = sum_n c_n L_n(y) exp(-y/2),
    y = (x - start)/scale (orthonormal). Functions decay like exp(-y/2); the wall x = start takes
    boundary conditions through tau terms written in the derivative bases L_n^(k)(y) exp(-y/2)
    (k = derivative level, the analogue of the ultraspherical index: d/dy and the conversion
    k -> k+1 are bidiagonal, multiplication by x tridiagonal). ``scale`` is the declared decay
    length (printed by explain()). Grid: Gauss-Laguerre nodes, floor(N * dealias) of them."""

    family = "Laguerre functions (half line)"
    unbounded = True

    def __init__(self, coord: Coordinate, size: int, start: float = 0.0, scale: float = 1.0,
                 dealias: Any = _REQUIRED) -> None:
        if not (math.isfinite(scale) and scale > 0):
            raise CoordinateError(f"Laguerre: scale must be a finite number > 0, got {scale!r}",
                                  fix="scale=1.0", doc="docs/site/guide/bases.md")
        if not math.isfinite(start):
            raise CoordinateError(f"Laguerre: start must be finite, got {start!r}",
                                  fix="start=0.0 (the half-line [start, inf))", doc="docs/site/guide/bases.md")
        super().__init__(coord, size, (float(start), math.inf), dealias)
        self.start, self.scale = float(start), float(scale)

    def _params(self) -> tuple:
        return (self.scale,)

    def coeff_space(self) -> str:
        return (f"L_n(y) exp(-y/2), n = 0..{self.size - 1}, y = (x - {self.start:g})/{self.scale:g} "
                "(orthonormal Laguerre functions); Gauss-Laguerre grid")

    def __repr__(self) -> str:
        return (f"Laguerre({self.coord.name}, size={self.size}, start={self.start:g}, "
                f"scale={self.scale:g}, dealias={_frac(self.dealias)})")


class CompoundChebyshev(Basis):
    """Chebyshev T series on consecutive subintervals [x_0, x_1], [x_1, x_2], ..., each with its
    own size (``sizes``); coefficients stored piece after piece. Continuity across an interface
    (value and derivatives up to the order of the equation minus one) is a set of ordinary
    conditions with their own tau terms: the native layer writes them (``cn.walls(a, c, b)``); in
    the IR write them with ``interp(f, x, c, side="below"/"above")`` and lift the taus with
    ``lift(tau, basis, -k, piece=p)``. Use: refine the piece holding a thin layer instead of the
    global N. Grid: the Gauss-Chebyshev nodes of each piece, floor(N_p * dealias) per piece."""

    family = "compound Chebyshev T (piecewise, glued at interfaces)"

    def __init__(self, coord: Coordinate, sizes: Any, breakpoints: Any,
                 dealias: Any = _REQUIRED) -> None:
        sizes = tuple(sizes)
        # Same rule as Basis.size: a fractional piece size is refused, never truncated by int().
        bad = [n for n in sizes if isinstance(n, bool) or not isinstance(n, numbers.Real) or int(n) != n]
        if bad:
            raise ResolutionError(f"CompoundChebyshev: piece sizes must be integers, got {bad[0]!r}",
                                  fix="integer sizes per piece, e.g. sizes=(16, 24)",
                                  doc="docs/site/guide/bases.md")
        sizes = tuple(int(n) for n in sizes)
        pts = tuple(float(v) for v in breakpoints)
        if len(pts) < 3:
            raise CoordinateError("CompoundChebyshev needs at least one interface",
                                  fix="breakpoints (a, c, b) with sizes (n1, n2); a single interval "
                                      "is Chebyshev(coord, n, (a, b), ...)", doc="docs/site/guide/bases.md")
        if len(sizes) != len(pts) - 1:
            raise ResolutionError(f"CompoundChebyshev: {len(pts) - 1} subintervals need "
                                  f"{len(pts) - 1} sizes, got {len(sizes)}",
                                  fix="one size per piece, e.g. sizes=(16, 16) for (a, c, b)",
                                  doc="docs/site/guide/bases.md")
        if any(not b > a for a, b in zip(pts, pts[1:], strict=False)):
            raise CoordinateError(f"CompoundChebyshev: breakpoints must increase strictly, got {pts}",
                                  fix="breakpoints (a, c, b) with a < c < b", doc="docs/site/guide/bases.md")
        if any(n < 2 for n in sizes):
            raise ResolutionError("CompoundChebyshev: each piece needs size >= 2",
                                  fix="sizes of at least 2 per piece (more for its tau rows)",
                                  doc="docs/site/guide/bases.md")
        super().__init__(coord, sum(sizes), (pts[0], pts[-1]), dealias)
        self.sizes, self.breakpoints = sizes, pts

    def _params(self) -> tuple:
        return (self.sizes, self.breakpoints)

    @property
    def pieces(self) -> list[tuple[float, float]]:
        return list(zip(self.breakpoints, self.breakpoints[1:], strict=False))

    @property
    def offsets(self) -> list[int]:
        """Index of the first coefficient of each piece."""
        out, o = [], 0
        for n in self.sizes:
            out.append(o)
            o += n
        return out

    @property
    def interfaces(self) -> tuple[float, ...]:
        return self.breakpoints[1:-1]

    def grid_sizes(self) -> list[int]:
        """Grid points per piece: floor(N_p * dealias)."""
        return [int(math.floor(n * self.dealias)) for n in self.sizes]

    @property
    def grid_size(self) -> int:
        return sum(self.grid_sizes())

    def piece_of(self, pos: float, side: str | None = None) -> int:
        """Index of the piece holding ``pos``; at an interface ``side`` ('below'/'above') is
        required (the trace is one-sided there), at the outer ends it is optional."""
        pts = self.breakpoints
        if not pts[0] <= pos <= pts[-1]:
            raise CoordinateError(f"position {pos} outside {self.bounds}",
                                  fix=f"a position in [{self.bounds[0]:g}, {self.bounds[1]:g}]",
                                  doc="docs/site/guide/bases.md")
        for j, c in enumerate(self.interfaces, start=1):
            if abs(pos - c) <= 1e-14 * max(1.0, abs(c)):
                if side not in ("below", "above"):
                    raise CoordinateError(
                        f"interp at the interface {self.coord.name} = {c:g} of {self!r}: the "
                        "trace is one-sided there", fix="give side='below' or side='above'",
                        doc="docs/site/guide/bases.md")
                return j - 1 if side == "below" else j
        if side is not None and pos not in (pts[0], pts[-1]):
            raise CoordinateError(f"side={side!r} is only meaningful at an interface "
                                  f"{self.interfaces}", fix="drop side= away from the interfaces",
                                  doc="docs/site/guide/bases.md")
        for j, (a, b) in enumerate(self.pieces):
            if a <= pos <= b:
                return j
        raise AssertionError(pos)

    def coeff_space(self) -> str:
        parts = ", ".join(f"[{a:g}, {b:g}]: T_n, n < {n}"
                          for (a, b), n in zip(self.pieces, self.sizes, strict=True))
        return f"piecewise Chebyshev T: {parts}; Gauss-Chebyshev nodes per piece"

    def __repr__(self) -> str:
        bp = ", ".join(f"{v:g}" for v in self.breakpoints)
        return (f"CompoundChebyshev({self.coord.name}, sizes={self.sizes}, breakpoints=({bp}), "
                f"dealias={_frac(self.dealias)})")
