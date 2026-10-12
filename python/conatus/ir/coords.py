"""Coordinates and coordinate systems (spec §8, §9, §12, §13).

A coordinate system is a first-class object: it owns its coordinates, its metric
and its regularity conditions. Cartesian systems polar and cylindrical coordinates (regular bases
in conatus.discretize.polar*) and spherical coordinates (S2, ball, shell: conatus.discretize.sphere*)
are implemented. No `1/(r+eps)`.
"""
from __future__ import annotations

from collections.abc import Iterator
from ..errors import CoordinateError

__all__ = [
    "Coordinate",
    "CoordinateSystem",
    "CartesianCoordinates",
    "PolarCoordinates",
    "CylindricalCoordinates",
    "S2Coordinates",
    "SphericalCoordinates",
]


class Coordinate:
    """One coordinate of a coordinate system (e.g. ``x`` of Cartesian (x, y))."""

    def __init__(self, name: str, system: CoordinateSystem, index: int) -> None:
        self.name = name
        self.system = system
        self.index = index

    @property
    def key(self) -> tuple:
        return ("Coordinate", self.name, self.system.key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Coordinate) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return f"Coordinate({self.name!r} in {self.system.name})"

    def __str__(self) -> str:
        return self.name

    def explain(self) -> str:
        return (f"Coordinate '{self.name}': index {self.index} of {self.system.name}"
                f"{tuple(c.name for c in self.system.coords)}")


class CoordinateSystem:
    """Base class: an ordered tuple of coordinates with a metric."""

    kind = "abstract"
    curvilinear = False

    def __init__(self, *names: str) -> None:
        if len(names) == 0:
            raise CoordinateError("a coordinate system needs at least one coordinate name",
                                  fix="CartesianCoordinates('x', 'z')")
        if len(set(names)) != len(names):
            raise CoordinateError(f"duplicate coordinate names: {names}",
                                  fix="give each coordinate a distinct name")
        self.names = tuple(names)
        self.coords = tuple(Coordinate(n, self, i) for i, n in enumerate(names))

    @property
    def dim(self) -> int:
        return len(self.coords)

    @property
    def name(self) -> str:
        return f"{self.kind}({', '.join(self.names)})"

    @property
    def key(self) -> tuple:
        return (self.kind, self.names)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CoordinateSystem) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __iter__(self) -> Iterator[Coordinate]:
        return iter(self.coords)

    def __getitem__(self, i: int | str) -> Coordinate:
        """Coordinate by name (``cs["z"]``) or by index."""
        if isinstance(i, str):
            if i not in self.names:
                from ..errors import CoordinateError
                raise CoordinateError(f"{self.name} has no coordinate {i!r}",
                                      fix="one of " + ", ".join(repr(n) for n in self.names))
            return self.coords[self.names.index(i)]
        return self.coords[i]

    def __repr__(self) -> str:
        return self.name

    def scale_factors(self) -> tuple:
        raise NotImplementedError(f"{self.kind}: scale factors not implemented")

    def metric(self) -> str:
        raise NotImplementedError(f"{self.kind}: metric not implemented")

    def regularity(self) -> str:
        raise NotImplementedError(f"{self.kind}: regularity conditions not implemented")

    def explain(self) -> str:
        return f"{self.name}: dim={self.dim}, curvilinear={self.curvilinear}"


class CartesianCoordinates(CoordinateSystem):
    """Cartesian coordinates in 1, 2 or 3 dimensions (identity metric)."""

    kind = "Cartesian"

    def __init__(self, *names: str) -> None:
        if not 1 <= len(names) <= 3:
            from ..errors import CoordinateError
            raise CoordinateError("CartesianCoordinates supports 1 to 3 coordinates",
                                  where=f"CartesianCoordinates{names!r}",
                                  fix="give 1 to 3 names, e.g. CartesianCoordinates('x', 'y', "
                                      "'z'); curvilinear systems: PolarCoordinates, "
                                      "SphericalCoordinates")
        super().__init__(*names)

    def scale_factors(self) -> tuple:
        return (1,) * self.dim

    def metric(self) -> str:
        return f"identity ({self.dim}x{self.dim}), Lebesgue measure d{' d'.join(self.names)}"

    def regularity(self) -> str:
        return "none (no coordinate singularity)"

    def explain(self) -> str:
        return "\n".join([
            f"{self.name}",
            f"  metric       : {self.metric()}",
            f"  scale factors: {self.scale_factors()}",
            f"  regularity   : {self.regularity()}",
        ])


class _Curvilinear(CoordinateSystem):
    """Declared-only curvilinear system: interface fixed now, implementation later."""

    curvilinear = True
    singularity = "?"

    def _refuse(self, what: str):
        from ..errors import UnsupportedError
        raise UnsupportedError(
            f"{self.kind} coordinates: {what} is not implemented yet. "
            f"Coordinate singularity at {self.singularity} will be handled by regular bases, "
            "never by 1/(r+eps)."
        )

    def scale_factors(self) -> tuple:
        self._refuse("scale factors")

    def metric(self) -> str:
        self._refuse("metric")

    def regularity(self) -> str:
        self._refuse("regularity conditions")

    def explain(self) -> str:
        return (f"{self.name}: curvilinear, singularity at {self.singularity}; "
                "declared only, not implemented yet")


class PolarCoordinates(CoordinateSystem):
    """(phi, r) on the disk or annulus.

    Metric dr^2 + r^2 dphi^2. The axis r = 0 is handled by *regular* bases (conatus.discretize.polar):
    an azimuthal mode m of a scalar is r^|m| w(r^2) with w a polynomial, and the spin components
    u_pm = u_r +- i u_phi of a vector are r^|m pm 1| times polynomials in r^2 (Vasil et al., JCP 325, 2016).
    Nothing is ever divided by r + eps. The native geometries ``cn.Disk`` and ``cn.Annulus`` use these
    coordinates.
    """

    kind = "Polar"
    curvilinear = True
    singularity = "r = 0 (axis)"

    def __init__(self, azimuth: str, radius: str) -> None:
        super().__init__(azimuth, radius)

    def scale_factors(self) -> tuple:
        return ("r", 1)              # (h_phi, h_r)

    def metric(self) -> str:
        return f"d{self.names[1]}^2 + {self.names[1]}^2 d{self.names[0]}^2, measure {self.names[1]} d{self.names[1]} d{self.names[0]}"

    def regularity(self) -> str:
        return ("scalar mode m: r^|m| w(r^2), w polynomial; vector spin components u_+-: r^|m+-1| times "
                "polynomials in r^2 (regular at the axis by construction, never 1/(r+eps))")

    def explain(self) -> str:
        return "\n".join([
            f"{self.name} (disk and annulus: DiskBasis / AnnulusBasis in r, Fourier in phi)",
            f"  metric       : {self.metric()}",
            f"  scale factors: (h_{self.names[0]}, h_{self.names[1]}) = {self.scale_factors()}",
            f"  singularity  : {self.singularity}",
            f"  regularity   : {self.regularity()}",
        ])


class CylindricalCoordinates(CoordinateSystem):
    """(phi, r, z) for pipes and periodic cylinders.

    Metric dr^2 + r^2 dphi^2 + dz^2. Same regular radial bases as PolarCoordinates (the z component of
    a vector is a scalar for the regularity, spin 0); z is expanded in a Fourier series. Nothing is
    ever divided by r + eps."""

    kind = "Cylindrical"
    curvilinear = True
    singularity = "r = 0 (axis)"

    def __init__(self, azimuth: str, radius: str, axial: str) -> None:
        super().__init__(azimuth, radius, axial)

    def scale_factors(self) -> tuple:
        return ("r", 1, 1)           # (h_phi, h_r, h_z)

    def metric(self) -> str:
        p, r, z = self.names
        return f"d{r}^2 + {r}^2 d{p}^2 + d{z}^2, measure {r} d{r} d{p} d{z}"

    def regularity(self) -> str:
        return ("mode (m, k_z): scalar and u_z r^|m| w(r^2); spin components u_+- r^|m+-1| times "
                "polynomials in r^2 (regular at the axis by construction, never 1/(r+eps))")

    def explain(self) -> str:
        return "\n".join([
            f"{self.name} (periodic pipe / cylinder: DiskBasis or AnnulusBasis in r, Fourier in z)",
            f"  metric       : {self.metric()}",
            f"  scale factors: {self.scale_factors()}",
            f"  singularity  : {self.singularity}",
            f"  regularity   : {self.regularity()}",
        ])


class S2Coordinates(CoordinateSystem):
    """(phi, theta) on the sphere of radius R.

    Metric R^2 (dtheta^2 + sin^2 theta dphi^2). The poles theta = 0, pi are handled by *regular* bases:
    a component of spin s (``u_+- = (u_theta +- i u_phi)/sqrt(2)`` for a vector) of azimuthal mode m is a
    sum of spin-weighted spherical harmonics sY_lm, l >= max(|m|, |s|) (Vasil et al., J. Comput. Phys.
    X 3 (2019) 100013). Nothing is ever divided by sin(theta) + eps. The radius is declared by
    SphereBasis."""

    kind = "S2"
    curvilinear = True
    singularity = "theta = 0, pi (poles)"

    def __init__(self, azimuth: str, colatitude: str) -> None:
        super().__init__(azimuth, colatitude)

    def scale_factors(self) -> tuple:
        return ("R sin(theta)", "R")          # (h_phi, h_theta)

    def metric(self) -> str:
        p, t = self.names
        return f"R^2 (d{t}^2 + sin({t})^2 d{p}^2), measure R^2 sin({t}) d{t} d{p}"

    def regularity(self) -> str:
        return ("spin-s component of mode m: sum of spin-weighted harmonics sY_lm, l >= max(|m|, |s|), "
                "i.e. sin(theta/2)^|m+s| cos(theta/2)^|m-s| P_n^(|m+s|,|m-s|)(cos theta): regular at "
                "both poles by construction, never 1/(sin(theta)+eps)")

    def explain(self) -> str:
        return "\n".join([
            f"{self.name} (sphere surface: SphereBasis in theta)",
            f"  metric       : {self.metric()}",
            f"  scale factors: {self.scale_factors()}",
            f"  singularity  : {self.singularity}",
            f"  regularity   : {self.regularity()}",
        ])


class SphericalCoordinates(CoordinateSystem):
    """(phi, theta, r) for the ball and the spherical shell.

    Metric dr^2 + r^2 dtheta^2 + r^2 sin^2 theta dphi^2. Angular dependence: spin-weighted spherical
    harmonics (as S2Coordinates) with spin components ``(u_-, u_+, u_r)``; radial dependence: ShellBasis
    (Chebyshev, rows multiplied by a declared power of r) or BallBasis (regularity components
    r^(l+sigma) times Jacobi polynomials in 2r^2 - 1, regular at the centre; Vasil et al. 2019,
    Lecoanet et al. 2019). Nothing is ever divided by r + eps or sin(theta) + eps."""

    kind = "Spherical"
    curvilinear = True
    singularity = "r = 0 (centre) and theta = 0, pi (poles)"

    def __init__(self, azimuth: str, colatitude: str, radius: str) -> None:
        super().__init__(azimuth, colatitude, radius)

    def scale_factors(self) -> tuple:
        return ("r sin(theta)", "r", 1)       # (h_phi, h_theta, h_r)

    def metric(self) -> str:
        p, t, r = self.names
        return f"d{r}^2 + {r}^2 d{t}^2 + {r}^2 sin({t})^2 d{p}^2, measure {r}^2 sin({t}) d{r} d{t} d{p}"

    def regularity(self) -> str:
        return ("angular: spin-weighted harmonics (regular at the poles); ball: regularity component "
                "sigma of degree l is r^(l+sigma) w(r^2), w polynomial (regular at the centre); shell: "
                "no centre, Chebyshev in r, r-multiplied rows (declared)")

    def explain(self) -> str:
        return "\n".join([
            f"{self.name} (ball (BallBasis) or shell (ShellBasis) in r, SphereBasis in theta)",
            f"  metric       : {self.metric()}",
            f"  scale factors: {self.scale_factors()}",
            f"  singularity  : {self.singularity}",
            f"  regularity   : {self.regularity()}",
        ])
