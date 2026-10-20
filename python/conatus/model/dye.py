"""Numerical dye: a passive scalar advected by the velocity of a native model, with
its own declared diffusivity, integrated by the simulation's own IMEX scheme.

``m.dye("c", velocity="u", kappa=1e-3)`` declares the scalar unknown ``c`` and its equation in
conservative form,

    dt(c) - kappa*lap(c) = -div(c*u),

with no-flux walls ``dn(c) = 0`` on the wall axis when kappa > 0 (written as ``m.at("walls",
"dz(c) = 0")``; printed by ``m.explain()``). ``kappa=0`` is pure advection: no diffusion, no
filter, no hyperviscosity, no wall condition (the equation has no wall derivative on its left
side); its spectral tail grows as filaments thin, and the dye file records it. The dye is one more
unknown of the coupled state: same scheme, same dt, same dealiasing as the flow (diffusion
implicit, advection explicit), so it is advanced exactly as the rest of the model.

``sim.dye("c", path, every=...)`` writes the dye and its diagnostics to an analysis file of the
self-describing layout (``cn.io.read``, ``.to_xarray()``): ``mass`` (∫c dV, the integral of the
spectral series), ``variance`` (∫(c - c̄)² dV of the dealiased product), ``minimum`` and
``maximum`` on the grid (over- and undershoots of an under-resolved dye), and ``tail`` (largest
relative coefficient among the last two modes of each axis). Mass is conserved by the continuous
equation; what the discretization keeps is measured, never assumed (guide: particles.md).
"""
from __future__ import annotations

import numbers
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import InputTypeError, OptionError, UnsupportedError
from ..io.analysis import AnalysisFile

__all__ = ["declare_dye", "check_dye_parameters", "DyeFile"]

_DOC = "docs/site/guide/particles.md"


def declare_dye(model: Any, name: str, *, velocity: str, kappa: Any) -> Any:
    """``m.dye``: the unknown, its equation and its wall conditions (see the module docstring)."""
    from .geometry import Box, HalfLine, Walls
    from .model import Unknown
    geo = model.geometry
    if not isinstance(geo, Box):
        raise UnsupportedError(
            f"m.dye({name!r}) on {geo!r}",
            why="the dye helper writes the no-flux wall condition of a Cartesian box",
            fix=f"declare it yourself: m.scalar({name!r}) before the equations, the equation "
                f"'dt({name}) - kappa*lap({name}) = -div({name}*{velocity})' and its wall "
                "conditions", doc=_DOC)
    vel = {u.name: u for u in model.unknowns}.get(velocity)
    if vel is None or vel.rank != 1:
        raise InputTypeError(f"m.dye({name!r}, velocity={velocity!r}): not a vector unknown of "
                             "the model", fix="m.velocity('u') first, then m.dye(..., "
                             "velocity='u')", doc=_DOC)
    if isinstance(kappa, str):
        if kappa not in model.params:
            raise InputTypeError(f"m.dye({name!r}, kappa={kappa!r}): not a declared parameter",
                                 fix="a number, or a parameter of cn.Model(..., kappa=1e-3)",
                                 doc=_DOC)
        K, diffusive = kappa, True
    elif isinstance(kappa, numbers.Real) and not isinstance(kappa, bool) and np.isfinite(kappa) \
            and kappa >= 0:
        K, diffusive = repr(float(kappa)), kappa > 0
    else:
        raise InputTypeError(f"m.dye({name!r}, kappa={kappa!r}): a diffusivity >= 0 (a number or "
                             "the name of a declared parameter)", fix="kappa=1e-3; kappa=0 for "
                             "pure advection (no diffusion and no filter, declared)", doc=_DOC)
    model._check_name(name, "unknown")
    # a passive scalar: no earlier equation uses it, so it may be declared after them (the
    # order rule of m.scalar protects equations that would name an undeclared unknown)
    model.unknowns.append(Unknown(name, 0, "dye"))
    text = (f"dt({name}) - {K}*lap({name}) = -div({name}*{velocity})" if diffusive
            else f"dt({name}) = -div({name}*{velocity})")
    model.equations(**{f"dye_{name}": text})
    bc = None
    wc = geo.wall_coord
    if diffusive and wc is not None:
        ax = geo._axes[wc]
        where = "walls" if isinstance(ax, Walls) else ("wall" if isinstance(ax, HalfLine) else None)
        if where is not None:
            bc = f"d{wc}({name}) = 0"
            model.at(where, bc)
    dyes = model.__dict__.setdefault("_dyes", {})
    dyes[name] = dict(velocity=velocity, kappa=kappa, equation=text, wall_condition=bc,
                      walls=(f"no flux: {bc} at the {where}" if bc else
                             "no wall condition (pure advection)" if wc is not None and not diffusive
                             else "no walls"))
    return model.unknowns[-1]


def check_dye_parameters(model: Any, over: dict) -> None:
    """Refuse a dye whose diffusivity is a declared parameter with an inadmissible value at
    compilation: negative (anti-diffusion, blows up), or zero when the dye
    carries the no-flux wall condition written for kappa > 0 (an advection equation with a wall
    condition imposed by the tau method: ill-posed; the dye grew 14-fold in half a time unit,
    silently). ``over``: per-call parameter values (they take precedence)."""
    for name, d in getattr(model, "_dyes", {}).items():
        k = d["kappa"]
        if not isinstance(k, str):
            continue                                   # a number: checked by m.dye itself
        v = over.get(k, model.params.get(k))
        if not isinstance(v, numbers.Real):
            continue                                   # a non-real value is refused elsewhere
        if v < 0 or (v == 0 and d.get("wall_condition")):
            raise OptionError(
                f"dye {name!r}: kappa = {k} = {v!r}",
                why=("a negative diffusivity is anti-diffusion (the dye blows up)" if v < 0 else
                     "m.dye wrote the no-flux wall condition "
                     f"{d['wall_condition']} for a diffusive dye; with kappa = 0 the equation "
                     "has no wall derivative and the condition makes it ill-posed"),
                fix=(f"{k} > 0, or m.dye({name!r}, velocity={d['velocity']!r}, kappa=0) for "
                     "pure advection (no wall condition)"), doc=_DOC)


class DyeFile(AnalysisFile):
    """Analysis file of a dye (``sim.dye``): the field and its conservation diagnostics."""

    def __init__(self, sim: Any, name: str, path: str | Path, *, every: int, mode: str | None,
                 initial: bool, field: bool) -> None:
        sim._check_unknown(name, "dye")
        if sim.compiled.fields[name].rank != 0:
            raise InputTypeError(f"dye({name!r}): a scalar unknown", doc=_DOC)
        if not isinstance(field, bool):
            raise OptionError(f"dye(field={field!r}): True (write the grid values) or False "
                              "(diagnostics only)", doc=_DOC)
        # capabilities checked before any file is created or replaced (review finding 64: on the
        # sphere the file was created, then the first diagnostic failed in evaluate())
        if sim.compiled.coordsys.curvilinear:
            raise UnsupportedError(
                f"dye({name!r}) on {sim.compiled.model.geometry!r}",
                why="the dye diagnostics (mass, variance) are integrals evaluated by "
                    "sim.evaluate(), implemented on Cartesian layouts only",
                fix=f"sim.output(path, tasks=[{name!r}], every=...) writes the field; integrate it "
                    "yourself on the solver grid", doc=_DOC)
        self.dye_name = name
        self.declared = getattr(sim.compiled.model, "_dyes", {}).get(name)
        geo = sim.compiled.model.geometry
        # physical volume (half the doubled period on a sine/cosine axis: review finding 7)
        from .sincos import physical_length
        self.volume = float(np.prod([physical_length(sim.compiled, c) for c in geo.coord_names]))
        c = name
        V = self.volume
        tasks: dict[str, Any] = {name: name} if field else {}

        def mass(s):
            return np.asarray(s.evaluate(f"integ({c})"))

        def variance(s):
            m = float(s.evaluate(f"integ({c})"))
            return np.asarray(float(s.evaluate(f"integ({c}*{c})")) - m * m / V)

        def minimum(s):
            return np.asarray(float(np.min(s.grid(c))))

        def maximum(s):
            return np.asarray(float(np.max(s.grid(c))))

        def tail(s):
            a = np.abs(np.asarray(s.coeffs(c)))
            top = a.max()
            if top == 0:
                return np.asarray(0.0)
            worst = max(float(np.take(a, [-2, -1], axis=ax).max()) for ax in range(a.ndim))
            return np.asarray(worst / top)

        tasks.update(mass=mass, variance=variance, minimum=minimum, maximum=maximum, tail=tail)
        self._dye_sources = {"mass": f"integ({c}) (integral of the spectral series)",
                             "variance": f"integ({c}*{c}) - integ({c})^2 / V, V = {V:g}",
                             "minimum": f"min of the grid values of {c}",
                             "maximum": f"max of the grid values of {c}",
                             "tail": "largest |coefficient| among the last two modes of each "
                                     "axis, relative to the largest"}
        for diag in (mass, variance, minimum, maximum, tail):   # dry run: fail before the file
            diag(sim)
        super().__init__(sim, path, tasks, every=every, mode=mode, initial=initial)

    def _describe(self, src: Any) -> str:
        if callable(src):
            for k, v in getattr(self, "tasks", {}).items():
                if v is src and k in self._dye_sources:
                    return self._dye_sources[k]
        return AnalysisFile._describe(src)

    def explain(self) -> str:
        """The file (as an analysis file), the dye's equation, walls and diffusivity."""
        d = self.declared
        lines = AnalysisFile.explain(self).replace("AnalysisFile", "Dye file", 1).splitlines()
        if d is not None:
            lines.insert(1, f"  dye {self.dye_name} (m.dye): {d['equation']}; {d['walls']}; "
                            f"advected by {d['velocity']}, same scheme and dt as the flow"
                         + ("; kappa = 0: no diffusion and no filter (the tail is recorded)"
                            if d["equation"].startswith(f"dt({self.dye_name}) =") else ""))
        else:
            lines.insert(1, f"  {self.dye_name}: a scalar of the model's own equations (not "
                            "declared by m.dye: conservation depends on how it was written)")
        return "\n".join(lines)
