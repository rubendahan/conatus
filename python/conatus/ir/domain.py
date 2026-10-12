"""Product domains (spec §8, §10).

A domain is a coordinate system together with a basis for *some* of its
coordinates. A coordinate without a basis means "constant along it" (e.g. a tau
field or a boundary value); this is how reduced domains are represented.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..errors import CoordinateError
from .bases import Basis
from .coords import Coordinate, CoordinateSystem

__all__ = ["Domain"]


def _check_curvilinear(coordsys: CoordinateSystem, bases: list[Basis]) -> None:
    """Polar (phi, r) and cylindrical (phi, r, z) domains: Fourier in phi on [0, 2 pi), a regular
    radial basis (DiskBasis / AnnulusBasis) in r, Fourier in z. Anything else is refused with the
    reason; spheres and balls are (coords.py)."""
    from .bases import AnnulusBasis, DiskBasis
    if coordsys.kind in ("S2", "Spherical"):
        _check_spherical(coordsys, bases)
        return
    if coordsys.kind not in ("Polar", "Cylindrical"):
        raise NotImplementedError(
            f"domains on {coordsys.kind} coordinates need regular curvilinear bases "
            "(not implemented yet; see coords.py)")
    phi, r = coordsys.coords[0], coordsys.coords[1]
    for b in bases:
        if b.coord == phi:
            if not b.periodic:
                raise CoordinateError(f"{b!r}: the azimuth {phi.name} needs a Fourier basis")
            if abs(b.bounds[0]) > 1e-14 or abs(b.length - 2 * np.pi) > 1e-12:
                raise CoordinateError(f"{b!r}: the azimuth must span [0, 2 pi) exactly")
        elif b.coord == r:
            if not isinstance(b, (DiskBasis, AnnulusBasis)):
                raise CoordinateError(
                    f"{b!r} along the radius {r.name}: use DiskBasis (axis included, regular at "
                    "r = 0) or AnnulusBasis (0 < ri < ro); a plain Chebyshev basis in r would need "
                    "an axis condition or 1/r factors")
        elif not b.periodic:
            raise CoordinateError(f"{b!r}: the axial direction {b.coord.name} must be Fourier "
                             "(bounded cylinders are not implemented)")


def _check_spherical(coordsys: CoordinateSystem, bases: list[Basis]) -> None:
    """S2 (phi, theta) and spherical (phi, theta, r) domains: Fourier in phi on [0, 2 pi),
    SphereBasis in theta, ShellBasis/BallBasis in r. A phi basis without a theta basis is refused (an
    azimuthal mode m != 0 needs its colatitude harmonics); theta without phi (axisymmetric, m = 0) is
    allowed, e.g. for coefficient fields."""
    from .bases import BallBasis, ShellBasis, SphereBasis
    phi, theta = coordsys.coords[0], coordsys.coords[1]
    r = coordsys.coords[2] if coordsys.kind == "Spherical" else None
    have = {b.coord for b in bases}
    for b in bases:
        if b.coord == phi:
            if not b.periodic:
                raise CoordinateError(f"{b!r}: the azimuth {phi.name} needs a Fourier basis")
            if abs(b.bounds[0]) > 1e-14 or abs(b.length - 2 * np.pi) > 1e-12:
                raise CoordinateError(f"{b!r}: the azimuth must span [0, 2 pi) exactly")
        elif b.coord == theta:
            if not isinstance(b, SphereBasis):
                raise CoordinateError(
                    f"{b!r} along the colatitude {theta.name}: use SphereBasis (spin-weighted spherical "
                    "harmonics, regular at the poles); a Chebyshev/Legendre basis in theta would need "
                    "pole conditions or 1/sin(theta) factors")
        elif b.coord == r:
            if not isinstance(b, (ShellBasis, BallBasis)):
                raise CoordinateError(
                    f"{b!r} along the radius {r.name}: use BallBasis (centre included, regular at r = 0) "
                    "or ShellBasis (0 < ri < ro)")
    if phi in have and theta not in have:
        raise CoordinateError(f"a basis in {phi.name} without a basis in {theta.name}: azimuthal modes on "
                         "the sphere need their colatitude harmonics (SphereBasis)")


class Domain:
    """Tensor-product domain: one coordinate system, bases for a subset of its coords."""

    def __init__(self, coordsys: CoordinateSystem, bases: Iterable[Basis]) -> None:
        bases = list(bases)
        if coordsys.curvilinear:
            _check_curvilinear(coordsys, bases)
        by_coord: dict[Coordinate, Basis] = {}
        for b in bases:
            if b.coord.system != coordsys:
                raise CoordinateError(f"basis {b!r} is not on {coordsys.name}",
                                      why="each basis is attached to a coordinate of one system",
                                      fix=f"build the basis on a coordinate of {coordsys.name}, "
                                          f"e.g. {coordsys.name}['name']")
            if b.coord in by_coord:
                raise CoordinateError(f"two bases for coordinate {b.coord.name}",
                                      where=f"Domain({coordsys.name}, {list(bases)!r})",
                                      why="a domain is a tensor product: one basis per coordinate",
                                      fix="keep one basis per coordinate (composite "
                                          "intervals are not implemented yet)")
            by_coord[b.coord] = b
        self.coordsys = coordsys
        self.bases: tuple[Basis, ...] = tuple(by_coord[c] for c in coordsys.coords if c in by_coord)

    @property
    def coords(self) -> tuple[Coordinate, ...]:
        return tuple(b.coord for b in self.bases)

    @property
    def dim(self) -> int:
        return len(self.bases)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(b.size for b in self.bases)

    def basis(self, coord: Coordinate) -> Basis | None:
        for b in self.bases:
            if b.coord == coord:
                return b
        return None

    def has(self, coord: Coordinate) -> bool:
        return self.basis(coord) is not None

    @property
    def separable_coords(self) -> tuple[Coordinate, ...]:
        """Coordinates whose basis is Fourier: the problem decouples per wavenumber."""
        return tuple(b.coord for b in self.bases if b.periodic)

    @property
    def nonperiodic_coords(self) -> tuple[Coordinate, ...]:
        return tuple(b.coord for b in self.bases if not b.periodic)

    def without(self, *coords: Coordinate) -> Domain:
        return Domain(self.coordsys, [b for b in self.bases if b.coord not in coords])

    def with_basis(self, basis: Basis) -> Domain:
        if self.has(basis.coord):
            raise CoordinateError(f"domain already has a basis for {basis.coord.name}")
        return Domain(self.coordsys, list(self.bases) + [basis])

    @staticmethod
    def union(a: Domain | None, b: Domain | None) -> Domain | None:
        """Broadcast union: a missing coordinate broadcasts; conflicting bases are an error."""
        if a is None:
            return b
        if b is None:
            return a
        if a.coordsys != b.coordsys:
            raise CoordinateError(f"cannot combine domains on {a.coordsys.name} and {b.coordsys.name}")
        bases = {bb.coord: bb for bb in a.bases}
        for bb in b.bases:
            if bb.coord in bases and bases[bb.coord] != bb:
                raise CoordinateError(
                    f"incompatible bases along {bb.coord.name}: {bases[bb.coord]!r} vs {bb!r} "
                    "(Conatus does not convert bases implicitly)")
            bases[bb.coord] = bb
        return Domain(a.coordsys, bases.values())

    @property
    def key(self) -> tuple:
        return ("Domain", self.coordsys.key, tuple(b.key for b in self.bases))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Domain) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        inner = " x ".join(f"{type(b).__name__}({b.coord.name}, {b.size})" for b in self.bases)
        return f"Domain[{inner or 'constant'}]"

    def explain(self) -> str:
        lines = [repr(self), f"  coordinates : {self.coordsys.name}"]
        for b in self.bases:
            lines.append("  " + b.explain().replace("\n", "\n  "))
        missing = [c.name for c in self.coordsys.coords if not self.has(c)]
        if missing:
            lines.append(f"  constant along: {', '.join(missing)}")
        lines.append(f"  separable (Fourier) directions: "
                     f"{', '.join(c.name for c in self.separable_coords) or 'none'}")
        return "\n".join(lines)
