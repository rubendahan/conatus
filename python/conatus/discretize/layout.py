"""Mode groups and coefficient layout for Fourier^k x (Chebyshev | nothing) domains.

A *group* is one Fourier wavenumber per periodic direction: the linear problem
decouples into one sparse system per group (never lose Fourier
separability). Inside a group, an object (variable or equation) that depends on
the Fourier direction i holds ``len(indices_i)`` Fourier coefficients (2 for a
real cos/-sin pair, 1 otherwise); an object constant along i only exists in the
zero-mode group. Vector layout inside a group, per object: tensor components
(C order), then Fourier coefficient multi-index (C order), then Chebyshev index.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np

from ..ir.bases import (Basis, Chebyshev, ComplexFourier, CompoundChebyshev, Hermite, Laguerre,
                        RealFourier, Ultraspherical)
from ..ir.coords import Coordinate
from ..ir.domain import Domain

__all__ = ["FourierDir", "GroupItem", "Layout", "ObjLayout"]


@dataclass(frozen=True)
class GroupItem:
    """One wavenumber of one Fourier direction."""

    kind: str          # 'real' | 'complex'
    index: int         # n (real) or position in fftfreq order (complex)
    k: float           # physical wavenumber
    indices: tuple[int, ...]   # coefficient indices held by a dependent object

    @property
    def zero(self) -> bool:
        return self.k == 0.0

    def size(self, dep: bool) -> int:
        if dep:
            return len(self.indices)
        return 1 if self.zero else 0

    def deriv(self, order: int) -> np.ndarray:
        if self.kind == "complex":
            return np.array([[(1j * self.k) ** order]])
        if len(self.indices) == 1:
            return np.array([[1.0 if order == 0 else 0.0]])
        D = np.array([[0.0, -self.k], [self.k, 0.0]])  # on (cos, -sin)
        return np.linalg.matrix_power(D, order)

    def embed_const(self) -> np.ndarray:
        """Constant (non-dependent) -> dependent: the constant is the cos_0 / k=0 mode."""
        return np.ones((1, 1)) if self.zero else np.zeros((len(self.indices), 0))

    def integrate(self, length: float) -> np.ndarray:
        """Dependent -> constant: only the zero mode survives, times the period."""
        return np.full((1, 1), length) if self.zero else np.zeros((0, len(self.indices)))


class FourierDir:
    def __init__(self, basis: Basis) -> None:
        if basis.size % 2:
            # typed refusal with the fix (a bare ValueError)
            from ..errors import ResolutionError
            raise ResolutionError(f"{basis!r}: Fourier sizes must be even",
                                  why="a Fourier series of N modes holds the wavenumbers up to "
                                      "the Nyquist mode N/2, which needs N even",
                                  fix=f"an even size for {basis.coord.name}, e.g. N = "
                                      f"{basis.size + 1} or {basis.size - 1}",
                                  doc="docs/site/guide/bases.md")
        self.basis = basis
        self.coord = basis.coord
        N, L = basis.size, basis.length
        items = []
        if isinstance(basis, RealFourier):
            for n in range(N // 2):
                items.append(GroupItem("real", n, 2 * np.pi * n / L,
                                       (0,) if n == 0 else (2 * n, 2 * n + 1)))
        elif isinstance(basis, ComplexFourier):
            freq = np.fft.fftfreq(N, 1.0 / N)
            for i in range(N):
                if i == N // 2:
                    continue  # Nyquist: not represented
                items.append(GroupItem("complex", i, 2 * np.pi * freq[i] / L, (i,)))
        else:
            raise TypeError(basis)
        self.items: tuple[GroupItem, ...] = tuple(items)
        self.complex = isinstance(basis, ComplexFourier)


def lam_of(basis: Basis) -> int:
    """Ultraspherical index of a (lift) basis: T -> 0, C^(lam) -> lam. Hermite, Laguerre and
    compound Chebyshev bases lift at the variables' level 0 (discretize/families.py)."""
    if isinstance(basis, (Chebyshev, Hermite, Laguerre, CompoundChebyshev)):
        return 0
    if isinstance(basis, Ultraspherical):
        if float(basis.lam) != int(basis.lam) or basis.lam < 1:
            raise NotImplementedError("lift bases: only Chebyshev T and integer C^(lam)")
        return int(basis.lam)
    raise NotImplementedError(
        f"{type(basis).__name__} as a lift basis of a Chebyshev T problem: use Chebyshev T or an "
        "integer Ultraspherical C^(lam) (Legendre/Jacobi problems lift in their own basis: "
        "lift(tau, basis, n, lam=q))")


@dataclass(frozen=True)
class ObjLayout:
    """Where one object (variable or equation) sits in a group vector."""

    name: str
    ncomp: int
    fdep: tuple[bool, ...]
    zdep: bool
    offset: int
    nf: int
    nz: int

    @property
    def size(self) -> int:
        return self.ncomp * self.nf * self.nz

    @property
    def block(self) -> int:
        return self.nf * self.nz


class Layout:
    """Fourier directions + (at most) one non-periodic Chebyshev direction of a domain."""

    def __init__(self, domain: Domain) -> None:
        self.domain = domain
        self.coordsys = domain.coordsys
        self.fdirs = tuple(FourierDir(b) for b in domain.bases if b.periodic)
        nonper = [b for b in domain.bases if not b.periodic]
        if len(nonper) > 1:
            raise NotImplementedError(
                "more than one non-periodic direction: the system does not separate into banded "
                "blocks (Cheb x Cheb needs a 2D solver, not implemented)")
        self.zbasis = nonper[0] if nonper else None
        from .jacobi_family import is_jacobi_trial
        if self.zbasis is not None and not (isinstance(self.zbasis, (Chebyshev, Hermite, Laguerre,
                                                                     CompoundChebyshev))
                                            or is_jacobi_trial(self.zbasis)):
            raise NotImplementedError(
                f"{type(self.zbasis).__name__} along {self.zbasis.coord.name}: Cartesian plans "
                "discretize Chebyshev T, Legendre, Jacobi, Ultraspherical, compound Chebyshev, "
                "Hermite or Laguerre in the non-periodic direction")
        self.nz = self.zbasis.size if self.zbasis is not None else 1
        # Hermite / Laguerre / compound Chebyshev: operators and transforms from their family
        # object; None for Chebyshev T, whose code path is unchanged.
        from .families import family_of
        self.zfam = family_of(self.zbasis)
        self.groups: tuple[tuple[GroupItem, ...], ...] = tuple(
            product(*[d.items for d in self.fdirs])) if self.fdirs else ((),)
        self.complex = any(d.complex for d in self.fdirs)

    def zconv(self, lam: int):
        """Conversion of z-coefficients from the variables' level 0 to level ``lam`` (the
        right-hand side of an equation written at level lam)."""
        if self.zfam is not None:
            return self.zfam.conversion_chain(0, lam)
        from . import ultraspherical as us
        return us.conversion_chain(self.nz, 0, lam)

    @property
    def zcoord(self) -> Coordinate | None:
        return self.zbasis.coord if self.zbasis is not None else None

    def fdep_of(self, dom: Domain | None) -> tuple[bool, ...]:
        return tuple(dom is not None and dom.has(d.coord) for d in self.fdirs)

    def zdep_of(self, dom: Domain | None) -> bool:
        return self.zbasis is not None and dom is not None and dom.has(self.zbasis.coord)

    def nf(self, fdep: Sequence[bool], group: tuple[GroupItem, ...]) -> int:
        n = 1
        for dep, item in zip(fdep, group, strict=True):
            n *= item.size(dep)
        return n

    def fourier_indices(self, fdep: Sequence[bool], group: tuple[GroupItem, ...]) -> list[tuple]:
        """Coefficient multi-indices (per Fourier dir, 0 for constant dirs) held in a group."""
        per = []
        for dep, item in zip(fdep, group, strict=True):
            per.append(item.indices if dep else ((0,) if item.zero else ()))
        return list(product(*per)) if per else [()]

    def group_label(self, group: tuple[GroupItem, ...]) -> str:
        if not group:
            return "single group (no Fourier direction)"
        return ", ".join(f"{d.coord.name}: k={it.k:g}" for d, it in zip(self.fdirs, group, strict=True))
