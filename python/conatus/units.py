"""Units and nondimensionalizations.

A study is written in *some* nondimensional system (a boundary-layer thickness or a period, a
channel half-height and the bulk or friction velocity, …), and research codes keep mixing them (a
factor 2√π on Re, 4π² on a stratification parameter, three energy normalizations). Here a system is
an explicit object:

* :class:`UnitSystem` — reference scales (length, time, velocity, …) with their symbols and
  definitions, expressed numerically in the *family basis* shared by related systems, plus named
  dimensionless groups ``prefactor · U L / ν``;
* :func:`convert` — the only way to move a number from one system to another, for a declared
  kind of quantity (wavenumber, growth rate, Reynolds number, …); systems of different families
  are refused;
* :class:`Energy` — an energy tagged with its normalization, converted only explicitly.

Study-specific conventions are *presets* built on this (``conatus.presets``), e.g.
``ChannelUnits`` (plane channel), re-exported here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .errors import InputTypeError, OptionError

if TYPE_CHECKING:  # re-exported lazily below (a module-level import would be circular)
    from .presets.channel import ChannelUnits

__all__ = ["Scale", "Group", "UnitSystem", "QUANTITY_DIMENSIONS", "convert",
           "Energy", "ENERGY_CONVENTIONS", "ChannelUnits"]


@dataclass(frozen=True)
class Scale:
    """A reference scale: its symbol, its definition, its value in the family basis."""

    symbol: str
    definition: str
    value: float

    def explain(self) -> str:
        """The scale in one line."""
        return f"reference scale {self.symbol} = {self.definition} = {self.value:g} (family basis)"

    def __post_init__(self):
        if not self.value > 0:
            raise OptionError(f"scale {self.symbol} = {self.definition} must be > 0 (got {self.value})")


@dataclass(frozen=True)
class Group:
    """A Reynolds-like group: ``prefactor · U_ref L_ref / ν`` (e.g. R = U₀δ/(2ν): prefactor 1/2)."""

    symbol: str
    definition: str
    prefactor: float = 1.0

    def explain(self) -> str:
        """The group in one line."""
        return (f"Reynolds-like group {self.symbol} = {self.definition}"
                + ("" if self.prefactor == 1.0 else f" (prefactor {self.prefactor:g})"))


@dataclass(frozen=True)
class UnitSystem:
    """A nondimensionalization: named reference scales within a family, and its Reynolds group."""

    name: str
    family: str                                   # conversions only inside one family
    scales: dict = field(default_factory=dict)    # "length" | "time" | "velocity" -> Scale
    reynolds: Group | None = None

    def scale(self, kind: str) -> Scale:
        if kind not in self.scales:
            raise KeyError(f"{self.name}: no {kind} scale (has {sorted(self.scales)})")
        return self.scales[kind]

    def explain(self) -> str:
        parts = [f"{k} {s.symbol} = {s.definition}" for k, s in self.scales.items()]
        re = f"; Reynolds {self.reynolds.symbol} = {self.reynolds.definition}" if self.reynolds else ""
        return f"Unit system '{self.name}' ({self.family}): " + ", ".join(parts) + re


# Dimensions of the quantities that can be converted, as exponents of (length, time, velocity).
QUANTITY_DIMENSIONS = {
    "length": (1, 0, 0), "time": (0, 1, 0), "velocity": (0, 0, 1),
    "wavenumber": (-1, 0, 0), "rate": (0, -1, 0), "frequency": (0, -1, 0),
    "frequency_squared": (0, -2, 0), "energy_density": (0, 0, 2),
}


def convert(value: complex, kind: str, frm: UnitSystem, to: UnitSystem) -> complex:
    """Express ``value`` (a ``kind`` quantity written in system ``frm``) in system ``to``.

    ``kind`` is a key of :data:`QUANTITY_DIMENSIONS` or ``"reynolds"`` (the systems' Reynolds
    groups). Refused across families: there is no implicit
    physical assumption linking two different flows.
    """
    if frm.family != to.family:
        raise OptionError(f"no conversion between families '{frm.family}' and '{to.family}'")
    if kind == "reynolds":
        if frm.reynolds is None or to.reynolds is None:
            raise OptionError("both systems need a declared Reynolds group")
        ratio = (to.reynolds.prefactor / frm.reynolds.prefactor) \
            * (to.scale("velocity").value / frm.scale("velocity").value) \
            * (to.scale("length").value / frm.scale("length").value)
        return value * ratio
    if kind not in QUANTITY_DIMENSIONS:
        raise OptionError(f"unknown quantity kind '{kind}' (known: {sorted(QUANTITY_DIMENSIONS)}, reynolds)")
    factor = 1.0
    for dim, p in zip(("length", "time", "velocity"), QUANTITY_DIMENSIONS[kind], strict=True):
        if p:
            factor *= (frm.scale(dim).value / to.scale(dim).value) ** p
    return value * factor


# Energy conventions met in the literature:
# factor that converts each one to the project's total (volume) energy E = ½ ∫_Ω |u|² dV.
ENERGY_CONVENTIONS = {
    "total": "E = ½ ∫_Ω |u|² dV (the reference)",
    "modal": "E = ½ ∫_0^Ly |q̂|² dy for one complex mode amplitude: × Lx Lz / 2",
    "surface": "E = (1/(2 Lx Lz)) ∫_Ω |u|² dV (per unit wall area): × Lx Lz",
}


@dataclass(frozen=True)
class Energy:
    """An energy value tagged with its convention; conversion only by an explicit call."""

    value: float
    convention: str           # key of ENERGY_CONVENTIONS

    def __post_init__(self):
        if self.convention not in ENERGY_CONVENTIONS:
            raise OptionError(f"unknown energy convention '{self.convention}' "
                             f"(known: {sorted(ENERGY_CONVENTIONS)})")

    def to_total(self, Lx: float | None = None, Lz: float | None = None) -> Energy:
        c = self.convention
        if c == "total":
            return self
        if Lx is None or Lz is None:
            raise OptionError(f"converting a '{c}' energy needs the box size Lx, Lz")
        f = Lx * Lz / 2.0 if c == "modal" else Lx * Lz
        return Energy(self.value * f, "total")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Energy) and other.convention != self.convention:
            raise InputTypeError(f"comparing a '{self.convention}' energy with a '{other.convention}' "
                            "one: convert explicitly with to_total()")
        return isinstance(other, Energy) and self.value == other.value

    __hash__ = None

    def explain(self) -> str:
        return f"Energy {self.value:.6g} [{self.convention}: {ENERGY_CONVENTIONS[self.convention]}]"


# Presets (study conventions) live in conatus.presets; re-exported lazily for existing imports
# (a module-level import would be circular: the presets are built on the classes above).
_PRESETS = {"ChannelUnits": "channel"}


def __getattr__(name: str):
    if name in _PRESETS:
        import importlib
        return getattr(importlib.import_module(f"conatus.presets.{_PRESETS[name]}"), name)
    raise AttributeError(f"module 'conatus.units' has no attribute '{name}'")
