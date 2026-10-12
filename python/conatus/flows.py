"""Usual base flows and their domains, stated explicitly (no hidden defaults).

``channel(N, base=...)`` builds the wall-normal Chebyshev domain of a plane channel and the
base-flow field U with its non-constant-coefficient data. Everything it chooses is visible in
``explain()``: basis, size, interval, dealias (1: linear problems have no products to dealias),
and the analytic profile.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import OptionError
from .ir import CartesianCoordinates, Chebyshev, Domain, Field
from ._repr import compact_repr

__all__ = ["ChannelBaseFlow", "channel", "PROFILES"]

PROFILES: dict[str, tuple[str, Callable[[np.ndarray], np.ndarray]]] = {
    "poiseuille": ("U = 1 - y^2 (plane Poiseuille, centreline velocity 1)", lambda Y: 1 - Y ** 2),
    "couette": ("U = y (plane Couette, walls at U = -1, +1)", lambda Y: Y + 0 * Y),
}


@dataclass
class ChannelBaseFlow:
    """Wall-normal coordinate ``y``, its Chebyshev basis and domain, base flow ``U`` and NCC data."""
    __repr__ = compact_repr   # one line, arrays by shape (as)

    y: Any
    basis: Chebyshev
    domain: Domain
    U: Field | None
    ncc: dict
    profile: str

    def field(self, name: str, **kw: Any) -> Field:
        """A new field on the channel domain."""
        return Field(name, self.domain, **kw)

    def explain(self) -> str:
        b = self.basis
        return (f"Channel: y in {b.bounds}, Chebyshev N={b.size}, dealias={b.dealias}; "
                f"base flow {self.profile}")


# Former name (conatus.api.Channel); renamed so that cn.Channel (native geometry) and
# conatus.expert never give the name two meanings.
Channel = ChannelBaseFlow


def channel(N: int, base: str | None, *, interval: tuple[float, float] = (-1.0, 1.0),
            dealias: float = 1) -> ChannelBaseFlow:
    """Plane channel in y with base flow ``base`` ('poiseuille', 'couette', or None)."""
    cs = CartesianCoordinates("y")
    y = cs["y"]
    b = Chebyshev(y, N, interval, dealias=dealias)
    D = Domain(cs, [b])
    if base is None:
        return ChannelBaseFlow(y, b, D, None, {}, "none")
    if base not in PROFILES:
        raise OptionError(f"unknown base flow {base!r}: choose one of {sorted(PROFILES)} or None")
    text, f = PROFILES[base]
    U = Field("U", D)
    return ChannelBaseFlow(y, b, D, U, {U: f}, text)
