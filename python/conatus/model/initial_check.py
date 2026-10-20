"""Check of callable initial data.

``sim.initial(T=lambda x: ...)`` samples the callable on the grid and transforms it; nothing used
to measure what the sampling lost. Three mistakes went through without a word:

* **aliasing**: ``cos(15 x)`` on N = 16 (largest kept wavenumber 7) is, on the 16 grid points,
  exactly ``cos(x)``: the run starts from another state;
* **a non-periodic function on a periodic axis**: ``sin(2 pi x)`` on ``periodic(2 pi)`` (the
  "L = 1 or L = 2 pi?" mix-up) jumps at the end of the period; its Fourier coefficients decay
  like 1/k, 9 % of the largest at N = 16;
* **non-finite values**: ``log(x - 1)`` is NaN on part of the grid (an array with a NaN was
  already refused; a callable was not).

The check samples the callable a second time, on a grid **twice as fine along one axis** (the
other axes on their own grids), and transforms along that axis: Fourier axes on 2N equispaced
points (wavenumbers up to N), bounded polynomial axes on 2N Chebyshev--Gauss points (degrees up
to 2N - 1). The **relative tail** is the largest coefficient the resolution N does not keep
(``|k| >= N/2`` on a Fourier axis, degree ``>= N`` on a bounded axis) over the largest
coefficient. It is the part of the data the run never sees. Two levels (a
single refusal at 1e-6 refused the coarse N of every resolution study):

* above ``INITIAL_TAIL_REFUSE`` (1e-2) the data are **clearly wrong** and refused, naming the
  axis: an aliased mode (order 1), or a function that is not periodic on a periodic axis; the
  latter is also refused when the function's values at the two ends of a periodic axis differ by
  more than ``INITIAL_TAIL_REFUSE`` of its largest value (a jump at the end of the period, whose
  tail decays like 1/N and would fall below the refusal level at large N);
* between ``INITIAL_TAIL_TOL`` (1e-6) and ``INITIAL_TAIL_REFUSE`` the data are **not fully
  resolved** (smooth data at a coarse N): an ``InitialDataWarning`` (once per ``initial(...)``
  call) names the axis and the tail, and ``explain()`` repeats it.

``initial(..., check=False)`` silences both (declared by ``explain()``). Nothing is changed: the
data are refused or accepted as given. A callable that does not return the same values twice
(random noise) is not judged on its resolution (it has none; said by ``explain()``); its
finiteness is checked.

Not checked (declared by ``explain()``): curved geometries (disk, annulus, sphere, ...),
sine/cosine axes, mapped and compound axes, Hermite/Laguerre axes; numbers and arrays (an array
of grid values carries no information beyond the grid; its finiteness is checked).
Cost: two extra evaluations of the callable on its grid (determinism, ends of the periodic axes)
and one per checked axis on twice the grid.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import InputTypeError, ResolutionError

#: Relative tail of callable initial data (module docstring) above which they are not fully
#: resolved: warned (InitialDataWarning) and recorded by explain(). The part of the data the
#: resolution cannot hold, relative to the largest coefficient; smooth data resolved to a few
#: parts per million pass silently.
INITIAL_TAIL_TOL = 1e-6
#: Relative tail (and relative jump between the two ends of a periodic axis) above which callable
#: initial data are clearly wrong and refused: an aliased mode (tail of order 1) or a function
#: that is not periodic on a periodic axis. Smooth data at the coarse N of a resolution study
#: stay below it (exp(cos x) at N = 8: 2e-3) and are only warned.
INITIAL_TAIL_REFUSE = 1e-2
_DOC = "docs/site/guide/timesteppers.md"


def _axis_kind(b: Any) -> str | None:
    """'fourier', 'poly' or None (not checked) for a basis."""
    from ..ir.bases import Jacobi, _Fourier
    if getattr(b, "map", None) is not None:
        return None
    if isinstance(b, _Fourier):
        return "fourier"
    if isinstance(b, Jacobi) and all(np.isfinite(b.bounds)):
        return "poly"
    return None


def _native_grid(b: Any) -> np.ndarray:
    from ..reference import transforms as T
    return np.asarray(T.grid(b, dealias=False))


def _fine_points(b: Any, kind: str) -> np.ndarray:
    n2 = 2 * int(b.size)
    a, c = (float(x) for x in b.bounds)
    if kind == "fourier":
        return a + (c - a) * np.arange(n2) / n2
    t = np.cos(np.pi * (np.arange(n2) + 0.5) / n2)[::-1]       # Chebyshev--Gauss, increasing
    return a + (c - a) * (t + 1) / 2


def _lost_fraction(vals: np.ndarray, axis: int, b: Any, kind: str) -> float:
    """Relative amplitude of the coefficients along ``axis`` that N does not keep."""
    N = int(b.size)
    if kind == "fourier":
        c = np.fft.fft(vals, axis=axis)
        k = np.abs(np.fft.fftfreq(2 * N) * 2 * N)
        lost = k >= N // 2
    else:
        import scipy.fft as sfft
        c = sfft.dct(vals[(slice(None),) * axis + (slice(None, None, -1),)], type=2, axis=axis)
        lost = np.arange(2 * N) >= N
    a = np.moveaxis(np.abs(c), axis, -1).reshape(-1, 2 * N)
    top = float(a.max()) if a.size else 0.0
    if not top > 0:
        return 0.0
    return float(a[:, lost].max() / top)


def check_callable(sim: Any, name: str, fn: Any, field: Any,
                   ir_fn: Any) -> tuple[list[str], list[str]]:
    """Check callable data for unknown ``name`` (module docstring). Refuses non-finite values and
    data whose relative tail on some axis exceeds INITIAL_TAIL_REFUSE (or that jump at the ends
    of a periodic axis); returns ``(notes, warned)``: the explain() notes of what could not be
    checked, and the text of each axis whose tail lies between INITIAL_TAIL_TOL and
    INITIAL_TAIL_REFUSE (the caller warns once). ``ir_fn`` is the callable in the IR convention
    (every coordinate grid, in coordinate order)."""
    comp = sim.compiled
    if comp.coordsys.curvilinear or comp.symmetry is not None:
        return ([f"initial data of {name}: not checked on this geometry (curved or sine/cosine "
                 "axis: finiteness and resolution; module conatus.model.initial_check)"], [])
    coords = list(comp.coordsys.coords)
    bases = [field.domain.basis(c) if field.domain is not None else None for c in coords]
    grids = [(_native_grid(b) if b is not None else np.zeros(1)) for b in bases]

    comps = tuple(int(getattr(cs, "dim", len(coords))) for cs in field.tensorsig)

    def sample(i: int | None = None, pts: np.ndarray | None = None) -> np.ndarray:
        """Values of the callable on the grids (axis ``i`` replaced by ``pts``), broadcast to
        the field's shape (components first), as ``Plan._as_grid`` does."""
        use = [pts if j == i else g for j, g in enumerate(grids)]
        gs = []
        for j, g in enumerate(use):
            sh = [1] * len(coords)
            sh[j] = g.size
            gs.append(np.asarray(g).reshape(sh))
        shape = tuple(g.size for g in use)
        with np.errstate(all="ignore"):
            raw = ir_fn(*gs)
        if len(comps) == 1 and isinstance(raw, (tuple, list)):
            # components of different broadcast shapes, e.g. (y, 0*x): each one broadcast to
            # the grid first (np.asarray of the tuple raised a bare ValueError)
            raw = [np.broadcast_to(np.asarray(r), shape) for r in raw]
        v = np.asarray(raw)
        try:
            return np.broadcast_to(v, comps + shape).copy()
        except ValueError:
            # a shape that is not the grid's (audit inputs: a bare ValueError)
            raise InputTypeError(
                f"initial({name}=<callable>): it returned shape {v.shape}, not that of the grid "
                f"{comps + shape}", why="the callable is evaluated on the coordinate grids "
                "(arrays shaped for broadcasting) and must return their broadcast shape",
                fix=f"write it with the coordinates: lambda {', '.join(c.name for c in coords)}: ... (e.g. "
                    "np.sin(x) + 0*z)") from None

    v0 = sample()
    bad = ~np.isfinite(v0)
    if bad.any():
        raise InputTypeError(
            f"initial({name}=<callable>): {int(bad.sum())} non-finite value(s) (NaN or inf) on "
            f"the grid of {name}",
            why="the run would start from a state with NaN/inf (an array with such values is "
                "refused as well); the finite-state check would only stop it after steps",
            fix="check the function on the whole box (e.g. a logarithm or a division by zero at "
                "a grid point)", doc=_DOC)
    if not np.array_equal(v0, sample()):           # random noise: no resolution to measure
        return ([f"initial data of {name}: the callable does not return the same values twice "
                 "(random?): finiteness checked, resolution not judged"], [])
    top = float(np.max(np.abs(v0))) if v0.size else 0.0
    unchecked, warned = [], []
    for i, (c, b) in enumerate(zip(coords, bases, strict=True)):
        if b is None or b.size <= 1:
            continue
        kind = _axis_kind(b)
        if kind is None:
            unchecked.append(c.name)
            continue
        N = int(b.size)
        axis = v0.ndim - len(coords) + i
        if kind == "fourier" and top > 0:          # the two ends of the period
            ends = sample(i, np.asarray([float(x) for x in b.bounds]))
            if np.all(np.isfinite(ends)):
                jump = float(np.max(np.abs(np.take(ends, 1, axis) - np.take(ends, 0, axis))))
                if jump > INITIAL_TAIL_REFUSE * top:
                    L = b.bounds[1] - b.bounds[0]
                    raise ResolutionError(
                        f"initial({name}=<callable>): the data are not periodic on axis "
                        f"{c.name}: the values at its two ends differ by {jump / top:.1e} of the "
                        f"largest value (tolerance INITIAL_TAIL_REFUSE = "
                        f"{INITIAL_TAIL_REFUSE:g})",
                        why="a periodic axis holds periodic data only: the jump at the end of "
                            "the period is spread over every Fourier mode (Gibbs), the run "
                            "starts from another function",
                        fix=f"check that the function has the period {L:g} of the box "
                            f"(sin(2*pi*x) on periodic(2*pi) is not periodic); "
                            "initial(..., check=False) accepts the data as sampled (declared "
                            "by explain())", doc=_DOC)
        pts = _fine_points(b, kind)
        vals = sample(i, pts)
        if not np.all(np.isfinite(vals)):
            raise InputTypeError(
                f"initial({name}=<callable>): non-finite values (NaN or inf) between the grid "
                f"points along {c.name}",
                why="the function is singular inside the box; on the grid it happened to be "
                    "finite, the run would start from a sample of a singular function",
                fix="check the function on the whole box; initial(..., check=False) accepts it "
                    "as sampled (declared by explain())", doc=_DOC)
        lost = _lost_fraction(vals, axis, b, kind)
        if lost <= INITIAL_TAIL_TOL:
            continue
        if kind == "fourier":
            L = b.bounds[1] - b.bounds[0]
            what = (f"aliased or not periodic on axis {c.name}: N = {N} keeps the "
                    f"wavenumbers |k| < {N // 2} (in units of 2 pi / {L:g}), but the "
                    f"function holds {lost:.1e} of its largest coefficient at |k| >= "
                    f"{N // 2}")
            hint = (f"raise N along {c.name} (aliasing: the content above the resolution "
                    f"folds onto resolved wavenumbers), or check that the function has the "
                    f"period {L:g} of the box (sin(2*pi*x) on periodic(2*pi) is not periodic)")
        else:
            what = (f"not resolved on axis {c.name}: N = {N} keeps the polynomial degrees "
                    f"< {N}, but the function holds {lost:.1e} of its largest coefficient at "
                    f"degree >= {N}")
            hint = f"raise N along {c.name}, or smooth the data"
        if lost > INITIAL_TAIL_REFUSE:
            raise ResolutionError(
                f"initial({name}=<callable>): the data are {what} (refusal level "
                f"INITIAL_TAIL_REFUSE = {INITIAL_TAIL_REFUSE:g})",
                why="the run would start from the projection of the data on N modes, not from "
                    "the data: the part above the resolution is lost or aliased, without a word",
                fix=hint + "; initial(..., check=False) accepts the data as sampled (declared by "
                           "explain())", doc=_DOC)
        warned.append(f"{name} along {c.name}: tail {lost:.1e} beyond N = {N} (above "
                      f"INITIAL_TAIL_TOL = {INITIAL_TAIL_TOL:g}, below the refusal level "
                      f"INITIAL_TAIL_REFUSE = {INITIAL_TAIL_REFUSE:g})")
    notes = []
    if unchecked:
        notes.append(f"initial data of {name}: resolution not checked along "
                     f"{', '.join(unchecked)} (mapped, compound, Hermite or Laguerre axis)")
    return notes, warned
