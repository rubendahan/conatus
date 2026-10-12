"""Plane channel: the usual Reynolds numbers as a preset.

Walls at y = ±h. Three velocity scales define three Reynolds numbers, all with the half-height h
unless stated otherwise:

* centreline  Re_c = U_c h / ν          (Orr--Sommerfeld convention, Orszag 1971)
* bulk        Re_b = U_b (2h) / ν       (full height, as in Kim, Moin & Moser 1987 and Pope 2000)
* friction    Re_τ = u_τ h / ν,  u_τ = √(τ_w/ρ)

The ratios between the velocity scales depend on the flow state. They are exact only for laminar
Poiseuille flow, U = U_c (1 − y²/h²):  U_b = (2/3) U_c  and  u_τ² = ν |dU/dy|_w = 2 ν U_c / h, hence
Re_b = (4/3) Re_c and Re_τ = √(2 Re_c). ``laminar_poiseuille`` gives these; for turbulent flow a
conversion needs the measured u_τ or U_b and is refused (no empirical correlation is applied
implicitly).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..errors import OptionError, UnsupportedError
from ..units import Group, Scale, UnitSystem

__all__ = ["CHANNEL_SYSTEMS", "ChannelUnits", "laminar_poiseuille"]

_LENGTH = {"centreline": ("h", "half-height", 1.0), "bulk": ("2h", "full height", 2.0),
           "friction": ("h", "half-height", 1.0)}
_SYMBOL = {"centreline": "Re_c", "bulk": "Re_b", "friction": "Re_τ"}
_VELOCITY = {"centreline": "U_c (centreline)", "bulk": "U_b (bulk)", "friction": "u_τ (friction)"}


def _system(based_on: str) -> UnitSystem:
    """Unit system for one Reynolds convention (family 'channel', basis h = ν = 1)."""
    sym, what, L = _LENGTH[based_on]
    return UnitSystem(f"channel/{based_on}", "channel",
                      {"length": Scale(sym, what, L)},
                      Group(_SYMBOL[based_on], f"{_VELOCITY[based_on].split()[0]} {sym} / ν"))


CHANNEL_SYSTEMS = {k: _system(k) for k in _LENGTH}


def laminar_poiseuille(*, Re_c: float) -> dict:
    """Exact Reynolds numbers of laminar Poiseuille flow at centreline Reynolds ``Re_c``."""
    if not Re_c > 0:                    # also NaN
        raise OptionError(f"Re_c must be > 0 (got {Re_c})", fix="laminar_poiseuille(Re_c=5772.22)")
    return {"Re_c": Re_c, "Re_b": 4.0 * Re_c / 3.0, "Re_tau": math.sqrt(2.0 * Re_c),
            "relations": "U_b = 2/3 U_c, u_tau^2 = 2 nu U_c / h (laminar Poiseuille, exact)"}


@dataclass(frozen=True)
class ChannelUnits:
    """A channel Reynolds number with its convention stated (``based_on`` is mandatory)."""

    Re: float
    based_on: str             # "centreline" | "bulk" | "friction"

    def __post_init__(self):
        if self.based_on not in CHANNEL_SYSTEMS:
            raise OptionError(f"based_on must be one of {sorted(CHANNEL_SYSTEMS)} (got '{self.based_on}')",
                              fix='ChannelUnits(Re=180, based_on="friction")')
        if not self.Re > 0:                 # also NaN
            raise OptionError(f"Re must be > 0 (got {self.Re})", fix='ChannelUnits(Re=180, based_on="friction")')

    @property
    def system(self) -> UnitSystem:
        """The unit system of the stated convention."""
        return CHANNEL_SYSTEMS[self.based_on]

    def laminar(self) -> dict:
        """All three Reynolds numbers **if the flow is laminar Poiseuille** (exact relations)."""
        Re_c = {"centreline": self.Re, "bulk": 0.75 * self.Re,
                "friction": 0.5 * self.Re ** 2}[self.based_on]
        return laminar_poiseuille(Re_c=Re_c)

    def turbulent(self, *args, **kw):
        """Refused: turbulent conversions need the measured u_tau or U_b of the run."""
        raise UnsupportedError("turbulent conversions between Re_c, Re_b and Re_tau need the measured "
                                  "u_tau or U_b of the run; no empirical correlation is applied implicitly")

    def explain(self) -> str:
        """The convention, its value and the conversions available."""
        s = self.system
        return (f"Channel: {s.reynolds.symbol} = {s.reynolds.definition} = {self.Re:g} "
                f"(walls y = ±h). Laminar-only conversions: laminar(); turbulent ones need measured "
                "u_tau or U_b.")
