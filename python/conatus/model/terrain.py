"""Mapped geometries of the native layer: a channel between two surfaces.

    box = cn.MappedBox(x=cn.periodic(2*np.pi), z=cn.between(bottom=lambda x: 0.1*np.cos(x), top=1))

The vertical axis runs from the bottom surface z = b(x[, y]) to the top surface z = t(x[, y])
(numbers or callables of the periodic coordinates, argument names = coordinate names). The map
is terrain-following (Gal-Chen & Somerville 1975), ``z = b + (t - b)(1 + xi)/2``; the equations
are written in the physical coordinates (``lap(u)``, ``dz(T)``, ``grad(p)`` are the physical
operators) and discretized by the chain rule with the metric as coefficients
(``conatus.ir.terrain``, ``conatus.discretize.terrain_plan``). Boundaries: ``"bottom"``,
``"top"``, ``"walls"``. Everything the map adds is printed by ``explain()``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..errors import InputTypeError, UnsupportedError
from ..ir import RealFourier
from ..ir.terrain import TerrainChebyshev, TerrainMap, surface
from .compiler import _coord_args
from .geometry import Box, Periodic, _g

__all__ = ["Between", "between", "MappedBox"]

_DOC = "docs/site/guide/mapped.md"


@dataclass(frozen=True)
class Between:
    """A bounded vertical axis between two surfaces (numbers or callables of the periodic
    coordinates)."""
    bottom: Any
    top: Any


def between(bottom: Any, top: Any) -> Between:
    """Vertical axis from the surface ``bottom`` up to the surface ``top`` of a ``MappedBox``:
    each a number (a flat wall) or a callable of the periodic coordinates (``lambda x: 0.1*cos(x)``,
    ``lambda x, y: ...``), smooth and periodic; ``top > bottom`` everywhere."""
    for nm, v in (("bottom", bottom), ("top", top)):
        if not (callable(v) or (isinstance(v, (int, float)) and not isinstance(v, bool))):
            raise InputTypeError(f"between({nm}={v!r}): a surface is a number or a callable of "
                                 "the periodic coordinates",
                                 fix=f"{nm}=0.0 or {nm}=lambda x: 0.1*np.cos(x)", doc=_DOC)
    return Between(bottom, top)


class MappedBox(Box):
    """Channel between two surfaces over periodic axes: ``MappedBox(x=periodic(L), z=between(b, t))``
    (2D) or ``MappedBox(x=periodic(Lx), y=periodic(Ly), z=between(b, t))`` (3D).

    Bases (fixed): RealFourier on each periodic axis, Chebyshev T in the terrain-following
    coordinate xi in [-1, 1] of the vertical axis. All Fourier modes are coupled by the metric:
    one sparse system for the whole domain (banded along xi), solved directly. Problems: LBVP
    (``m.solve``), EVP (``m.eigenvalues``) and IVP (``m.simulate``, implicit terms coupled,
    explicit terms on the grid). Boundaries: ``"bottom"`` (z = b), ``"top"`` (z = t), ``"walls"``.
    """

    kind = "mapped box (terrain-following)"

    def __init__(self, **axes: Any) -> None:
        vert = [k for k, v in axes.items() if isinstance(v, Between)]
        if len(vert) != 1:
            raise InputTypeError(f"MappedBox needs exactly one between(...) axis, got {len(vert)}",
                                 fix="MappedBox(x=cn.periodic(2*np.pi), z=cn.between(bottom=..., "
                                     "top=...))", doc=_DOC)
        others = {k: v for k, v in axes.items() if k != vert[0]}
        if not 1 <= len(others) <= 2 or not all(isinstance(v, Periodic) for v in others.values()):
            raise InputTypeError("MappedBox: the other axes are 1 or 2 periodic(...) axes",
                                 fix="MappedBox(x=cn.periodic(L), z=cn.between(...))", doc=_DOC)
        if any(v.complex for v in others.values()):
            raise UnsupportedError(
                "MappedBox with periodic(..., complex=True)",
                why="the mapped discretization couples all Fourier modes of real fields (real "
                    "cos/sin series); complex fields on a mapped domain are not implemented",
                fix="periodic(L) (real series)", doc=_DOC)
        if list(axes)[-1] != vert[0]:
            raise InputTypeError("MappedBox: the between(...) axis is the last one",
                                 fix="MappedBox(x=..., z=cn.between(...))", doc=_DOC)
        # Box validation would refuse a Between axis; build the Box on a placeholder walls axis
        # and keep the real one (the basis table and the build are overridden below)
        from .geometry import walls
        super().__init__(**{**others, vert[0]: walls(-1.0, 1.0)})
        self._axes = dict(axes)
        self._vert = vert[0]
        self._between = axes[vert[0]]
        names = tuple(others)
        surf = []
        for nm in ("bottom", "top"):
            v = getattr(self._between, nm)
            args = _coord_args(v, names, f"between({nm}=...)") if callable(v) else None
            surf.append(surface(v, args, names))
        try:
            self.terrain = TerrainMap(surf[0], surf[1],
                                      [(n, others[n].bounds) for n in names], vertical=vert[0])
        except ValueError as err:
            raise InputTypeError(f"MappedBox: {err}", fix="check the surfaces", doc=_DOC) from None

    # --- Box overrides -------------------------------------------------------------------
    @property
    def wall_coord(self) -> str | None:
        """The vertical (mapped) coordinate."""
        return self._vert

    @property
    def bounded_coord(self) -> str | None:
        """The vertical (mapped) coordinate."""
        return self._vert

    def n_walls(self) -> int:
        """Two walls (bottom and top surfaces)."""
        return 2

    @property
    def sincos_coord(self) -> str | None:
        """None: no sine/cosine axis on a mapped box."""
        return None

    @property
    def interfaces(self) -> tuple[float, ...]:
        """No interface (one piece)."""
        return ()

    def boundaries(self) -> dict[str, list[tuple[str, str]]]:
        """Named boundaries: bottom, top and walls of the vertical coordinate."""
        c = self._vert
        return {"bottom": [(c, "left")], "top": [(c, "right")],
                "walls": [(c, "left"), (c, "right")]}

    def _basis(self, coord, n, dealias):
        ax = self._axes[coord.name]
        if isinstance(ax, Periodic):
            return RealFourier(coord, n, ax.bounds, dealias=dealias)
        return TerrainChebyshev(coord, n, self.terrain, dealias=dealias)

    def basis_table(self) -> list[str]:
        """Coordinate -> basis lines of the mapped box (explain())."""
        out = []
        for k, v in self._axes.items():
            if isinstance(v, Periodic):
                out.append(f"{k}: periodic on [{_g(v.bounds[0])}, {_g(v.bounds[1])}) -> "
                           f"RealFourier (cos/sin)(N_{k})")
            else:
                out.append(f"{k}: between the surfaces {self.terrain.bottom.text} and "
                           f"{self.terrain.top.text} -> Chebyshev T(xi)(N_{k}) in the "
                           f"terrain-following coordinate xi in [-1, 1]")
        out.append("map: " + self.terrain.formula())
        out.append("metric (chain rule, coefficients of the operators): " + self.terrain.metric_text())
        out.append("all Fourier modes coupled by the metric: one sparse system (banded along xi); "
                   f"traces 'bottom'/'top' are at xi = -1/+1; d{self._vert}(f) in a condition is "
                   f"the physical vertical derivative, not the normal one")
        return out

    def __repr__(self) -> str:
        ax = ", ".join(f"{k}=periodic({_g(v.length)})" if isinstance(v, Periodic) else
                       f"{k}=between({self.terrain.bottom.text}, {self.terrain.top.text})"
                       for k, v in self._axes.items())
        return f"MappedBox({ax})"
