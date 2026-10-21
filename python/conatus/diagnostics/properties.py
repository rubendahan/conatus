"""Global flow properties followed during a run (the role of Dedalus' ``GlobalFlowProperty``).

::

    props = sim.properties(every=10, Re=lambda s: np.sqrt((s.grid("u")**2).sum(0)) / nu,
                           T="T")
    sim.run(until=1.0)
    props.max("Re"), props.min("T"), props.volume_average("T")

Each property is a grid quantity: the name of an unknown or component (``"T"``, ``"u.x"``), an
expression of the state as in the equations (``"0.5*u@u"``, ``Simulation.evaluate``) or a
function ``f(sim)`` returning grid values (or a number). It is evaluated every ``every`` steps of
:meth:`Simulation.run` (a whole number of steps, never rounded); the reductions read the last
evaluation and say when it was made (``evaluated_at``). Reductions:

* ``max`` / ``min``: over the grid points (as Dedalus);
* ``grid_average``: arithmetic mean of the grid values (Dedalus' ``grid_average``; on a Chebyshev
  grid it is *not* the volume average, the points are not uniform);
* ``volume_average`` / ``integral``: quadrature on the grid (trapezoidal on periodic axes, Fejér's
  first rule on Chebyshev–Gauss points, ``conatus.diagnostics.grid``), Cartesian Fourier /
  Chebyshev geometries only (others refused, never approximated by a grid mean).

Nothing here changes the run: no time step, no state is modified.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..errors import DtypeError, InputTypeError, UnsupportedError
from ..tolerances import IMAG_ROUNDOFF

__all__ = ["FlowProperties"]

_DOC = "docs/site/guide/diagnostics.md"


def _host(a: Any) -> np.ndarray:
    if hasattr(a, "get") and not isinstance(a, np.ndarray):     # CuPy array
        a = a.get()
    return np.asarray(a)


class FlowProperties:
    """Named grid quantities evaluated every ``every`` steps; max / min / averages of the last
    evaluation. Built by ``sim.properties(every=k, name=source, ...)``."""

    def __init__(self, sim: Any, every: int, **props: str | Callable[[Any], Any]) -> None:
        from ..io._h5 import positive
        self.sim, self.every = sim, positive(every, "properties()", "every")
        if not props:
            raise InputTypeError("properties(): no property",
                                 fix="properties(every=10, Re=lambda s: ..., T='T')", doc=_DOC)
        names = list(sim.compiled.model.geometry.coord_names)
        unknowns = {u.name: u for u in sim.compiled.model.unknowns}
        from ..io.analysis import _is_name
        self._is_name = _is_name
        for k, src in props.items():
            if isinstance(src, str) and not _is_name(src):
                sim.evaluate(src)                          # an expression: checked now
            elif isinstance(src, str):
                base, _, comp = src.partition(".")
                if base not in unknowns:
                    raise InputTypeError(f"properties({k}={src!r}): {base!r} is not an unknown",
                                         fix="unknowns: " + ", ".join(unknowns) + "; or a function f(sim)",
                                         doc=_DOC)
                rank = sim.compiled.fields[base].rank
                if rank == 1 and comp not in names:
                    raise InputTypeError(f"properties({k}={src!r}): {base!r} is a vector, name a "
                                         "component", fix=", ".join(f"'{base}.{c}'" for c in names),
                                         doc=_DOC)
                if rank != 1 and comp:
                    raise InputTypeError(f"properties({k}={src!r}): components are for vectors", doc=_DOC)
            elif not callable(src):
                raise InputTypeError(f"properties({k}=...): a name or a function f(sim)", doc=_DOC)
        self.props = dict(props)
        self._names = names
        self.values: dict[str, np.ndarray] = {}
        self.evaluated_at: tuple[int, float] | None = None
        self._grid = None

    # ------------------------------------------------------------------ evaluation
    def _value(self, src: Any) -> np.ndarray:
        if callable(src):
            return _host(src(self.sim))
        if not self._is_name(src):
            return _host(self.sim.evaluate(src))
        base, _, comp = src.partition(".")
        g = _host(self.sim.grid(base))
        return g[self._names.index(comp)] if comp else g

    def _real(self, name: str, a: np.ndarray) -> np.ndarray:
        """Grid values as real numbers: a complex array (ComplexFourier problem, complex
        expression or function) is accepted only when its imaginary part is round-off
        (``conatus.tolerances.IMAG_ROUNDOFF`` x max(1, max|a|)); otherwise refused. ``asarray(...,
        dtype=float)`` used to drop it with a mere ComplexWarning, and max/min/averages then read
        the real part of a complex quantity."""
        if np.iscomplexobj(a):
            im = float(np.max(np.abs(a.imag))) if a.size else 0.0
            scale = max(1.0, float(np.max(np.abs(a)))) if a.size else 1.0
            if im > IMAG_ROUNDOFF * scale:
                raise DtypeError(
                    f"property {name!r}: complex grid values (max|Im| = {im:.2e} > "
                    f"{IMAG_ROUNDOFF:.1e} x max(1, max|values|), the declared round-off)",
                    why="max, min and the averages are defined for real quantities",
                    fix=f"declare a real quantity, e.g. {name}=lambda s: abs(...) or "
                        "lambda s: (...).real if the real part is what you mean", doc=_DOC)
            a = a.real
        return np.asarray(a, dtype=float)

    def evaluate(self) -> None:
        """Evaluate every property on the current state (also called by the run)."""
        self.values = {k: self._real(k, np.asarray(self._value(v))) for k, v in self.props.items()}
        self.evaluated_at = (self.sim.iteration, self.sim.t)

    def __call__(self, solver: Any) -> None:              # callback of IVPSolver.run
        if solver.iteration % self.every == 0:
            self.evaluate()

    def _get(self, name: str) -> np.ndarray:
        if name not in self.props:
            raise InputTypeError(f"property {name!r} not declared",
                                 fix="declared: " + ", ".join(self.props), doc=_DOC)
        if self.evaluated_at is None:
            # never evaluated (no run yet, or fewer than `every` steps): evaluate now, as a
            # Dedalus property would read the current state; evaluated_at says so
            self.evaluate()
        return self.values[name]

    # ------------------------------------------------------------------ reductions
    def max(self, name: str) -> float:
        """Maximum over the grid points of the last evaluation."""
        return float(np.max(self._get(name)))

    def min(self, name: str) -> float:
        """Minimum over the grid points of the last evaluation."""
        return float(np.min(self._get(name)))

    def grid_average(self, name: str) -> float:
        """Arithmetic mean of the grid values (not a volume average on non-uniform grids)."""
        return float(np.mean(self._get(name)))

    def _quadrature(self):
        if self._grid is None:
            from ..ir.bases import Chebyshev, ComplexFourier, RealFourier
            from .grid import Grid
            comp = self.sim.compiled
            bases = [comp.bases[c] for c in self._names]
            ok = all(isinstance(b, (RealFourier, ComplexFourier)) or
                     (type(b) is Chebyshev and getattr(b, "map", None) is None) for b in bases)
            if not ok or comp.symmetry is not None:
                raise UnsupportedError(
                    "volume_average / integral: quadrature implemented for Cartesian Fourier and "
                    "Chebyshev axes only", where=f"geometry {comp.model.geometry!r}",
                    why="other bases (curvilinear, mapped, sine/cosine, Hermite, Laguerre, compound) "
                        "need their own weights; a grid mean would be a different quantity",
                    fix="use the integ() of the model, or grid_average() knowingly", doc=_DOC)
            self._grid = Grid.from_bases(bases, dealias=False)
        return self._grid

    def integral(self, name: str) -> float:
        """Integral over the domain of a scalar property (quadrature on the grid)."""
        a = self._get(name)
        g = self._quadrature()
        if a.shape != g.shape:
            raise InputTypeError(f"integral({name!r}): values of shape {a.shape}, grid {g.shape}",
                                 why="only scalar grid quantities are integrated", doc=_DOC)
        return float(g.integrate(a))

    def volume_average(self, name: str) -> float:
        """Integral / volume of a scalar property."""
        return self.integral(name) / self._quadrature().volume()

    def explain(self) -> str:
        when = ("not evaluated yet" if self.evaluated_at is None else
                f"last evaluated at iteration {self.evaluated_at[0]}, t = {self.evaluated_at[1]:g}")
        lines = [f"FlowProperties every {self.every} steps ({when})"]
        for k, v in self.props.items():
            lines.append(f"  {k}: " + ((f"grid values of {v}" if self._is_name(v) else f"expression {v}")
                                       if isinstance(v, str) else
                                       f"function {getattr(v, '__qualname__', type(v).__name__)}"))
        lines.append("  reductions: max, min, grid_average (mean of grid values), volume_average and "
                     "integral (quadrature; Cartesian Fourier/Chebyshev)")
        return "\n".join(lines)

    __str__ = explain
