# Units and conventions

Unit systems let you convert between nondimensionalisations with `convert()`, which checks the supplied unit systems and the scales it needs.

Published values are written in many nondimensionalisations: a plane channel alone has three
Reynolds numbers (centreline, bulk, friction), and wavenumbers are scaled by the half-height or the
full height depending on the paper. A missed factor of $2$ or $3/2$ is a common error when
comparing with the literature. `UnitSystem` stores reference scales. Presets provide the
conventions of particular studies. `convert()` translates a quantity between two declared
systems.

## Unit systems and `convert`

A `UnitSystem(name, family, scales, reynolds)` lists reference scales (length, time, velocity)
with their definition and value in the family's base units, and its Reynolds group. `convert`
moves a number between two systems:

```python
from conatus.expert import CHANNEL_SYSTEMS, convert

half, full = CHANNEL_SYSTEMS["centreline"], CHANNEL_SYSTEMS["bulk"]
print(convert(1.020547, "wavenumber", half, full))   # 2.041094: alpha per h -> per 2h
print(convert(2.0, "length", half, full))            # 1.0: two half-heights = one height
```

`kind` is one of length, time, velocity, wavenumber, rate, frequency, frequency squared, energy
density, `reynolds`; the factor is the product of scale ratios to the power of the dimension.
A conversion that needs a scale the system does not define is refused (the bulk system of a
channel has no velocity scale until a flow gives one), as are a conversion between different
families and an unknown kind.

## Channel presets

| Preset | Conventions | Source |
|---|---|---|
| `ChannelUnits(Re, based_on)`, `CHANNEL_SYSTEMS` | $Re_c = U_c h/\nu$, $Re_b = U_b\,2h/\nu$, $Re_\tau = u_\tau h/\nu$ | laminar Poiseuille relations only |

```python
from conatus.expert import ChannelUnits

print(ChannelUnits(5772.22, "centreline").laminar())
# {'Re_c': 5772.22, 'Re_b': 7696.29, 'Re_tau': 107.45, 'relations': 'U_b = 2/3 U_c, ...'}
try:
    ChannelUnits(180.0, "friction").turbulent()
except NotImplementedError as err:
    print(err)
# turbulent conversions between Re_c, Re_b and Re_tau need the measured
#   u_tau or U_b of the run; no empirical correlation is applied implicitly
```

`ChannelUnits` has no default convention: `based_on` is required. `laminar_poiseuille(Re_c=...)`
gives the exact laminar relations.
