"""Study presets built on the generic unit systems of :mod:`conatus.units`.

A preset fixes the conventions of a published study (its reference scales, Reynolds number,
wavenumbers, stratification parameter) and prints them; it never converts implicitly.
"""
from .channel import CHANNEL_SYSTEMS, ChannelUnits, laminar_poiseuille

__all__ = ["CHANNEL_SYSTEMS", "ChannelUnits",
           "laminar_poiseuille"]
