"""Eigenfunctions on the grid: ``res.mode(i)`` of an eigenvalue result.

An eigenvector of ``eigenvalues(plan, group)`` is one column of the group system: the
coefficients of every variable of the problem, tau variables included, interleaved as the plan
lays them out. :class:`EVPMode` unpacks it into the named unknowns (tau variables left out),
on the physical (non-dealiased) grid of the plan, with the coordinates, under one declared
normalisation:

* **normalisation** (``NORMALIZATION``): the eigenvector is scaled so that the first unknown in
  declaration order that is not zero (largest |coefficient| above ``NORMALIZATION_ZERO_REL``
  times that of the whole eigenvector) has max |q| = 1 on the grid, real and positive
  at the grid point where |q| is largest. The other unknowns get the same complex factor (the
  relative amplitudes and phases of the mode are unchanged). An eigenvector is defined up to such
  a factor: this one makes two runs, two resolutions or two codes comparable point by point.
* the **values are complex**: for a mode exp(s t) (or exp(i alpha (x - c t))) the physical field
  is the real part of q(x) times the time factor. Along a real Fourier axis (cos/sin pairs) the
  group holds the cos and sin coefficients of one wavenumber, so the grid values already carry
  the x dependence; along a complex Fourier axis the grid values are q(z) exp(i k x).

Supported: plans of Cartesian geometries (Fourier and Chebyshev/Legendre/... axes, the native
``m.eigenvalues``, ``lin.eigenvalues`` and expert ``conatus.eigenvalues(plan, ...)``). Other
plans (disk, sphere, sine/cosine axes) refuse with the reason.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import InputTypeError, OptionError, UnsupportedError, did_you_mean

__all__ = ["EVPMode", "NORMALIZATION", "NORMALIZATION_ZERO_REL", "mode_of"]

_DOC = "docs/site/guide/dynamics/eigenvalues.md"

#: The declared normalisation of :meth:`EVPResult.mode` (printed by ``EVPMode.explain()``).
NORMALIZATION = ("max |q| = 1 for the first non-zero unknown (declaration order), real and "
                 "positive at its largest grid value; the other unknowns scaled by the same "
                 "complex factor")
#: An unknown counts as zero for the normalisation when the largest |coefficient| it holds is
#: at most this fraction of the largest |coefficient| of the eigenvector (round-off of a field
#: that the mode does not excite, e.g. by symmetry: a 1e-17 residue was scaled to 1).
NORMALIZATION_ZERO_REL = 1e-10


class EVPMode:
    """One eigenfunction on the grid: ``mode["v"]`` / ``mode.grid("v")`` (complex grid values),
    ``mode.coeffs("v")``, ``mode.coordinate("y")``, ``mode.at("v", y=points)`` (exact
    spectral evaluation, walls included), ``mode.value``, ``mode.residual``, ``mode.names``,
    ``mode.explain()``."""

    def __init__(self, value: complex, residual: float, coeffs: dict[str, np.ndarray],
                 grids: dict[str, np.ndarray], coords: dict[str, np.ndarray], label: str,
                 norm_field: str, index: int) -> None:
        self.value = complex(value)
        self.residual = float(residual)
        self._coeffs, self._grids, self._coords = coeffs, grids, coords
        self.label, self.norm_field, self.index = label, norm_field, index
        self._bases: dict[str, Any] = {}           # coordinate name -> basis (set by mode_of)

    @property
    def names(self) -> list[str]:
        """The unknowns of the mode (tau variables left out)."""
        return list(self._grids)

    def _check(self, name: str) -> None:
        if name not in self._grids:
            raise InputTypeError(f"mode: {name!r} is not an unknown of this problem."
                                 + did_you_mean(name, self._grids),
                                 fix="unknowns: " + ", ".join(self._grids), doc=_DOC)

    def grid(self, name: str) -> np.ndarray:
        """Complex grid values of unknown ``name`` (components first for a vector)."""
        self._check(name)
        return self._grids[name]

    __getitem__ = grid

    def coeffs(self, name: str) -> np.ndarray:
        """Spectral coefficients of unknown ``name`` (the plan's layout, same normalisation)."""
        self._check(name)
        return self._coeffs[name]

    def coordinate(self, name: str) -> np.ndarray:
        """Grid points of coordinate ``name`` matching :meth:`grid`."""
        if name not in self._coords:
            raise InputTypeError(f"mode.coordinate({name!r}): not a coordinate with a basis."
                                 + did_you_mean(name, self._coords),
                                 fix="coordinates: " + ", ".join(self._coords), doc=_DOC)
        return self._coords[name]

    def at(self, name: str, **points: Any) -> np.ndarray:
        """Values of unknown ``name`` at chosen points of the non-periodic coordinate, by exact
        evaluation of its spectral expansion (the grid points of a
        Chebyshev axis do not reach the walls): ``mode.at("w", z=np.linspace(-0.5, 0.5, 101))``.
        The other coordinates stay on the grid of :meth:`grid` (same normalisation, same layout:
        components first, then the coordinates in order)."""
        from ..reference import transforms as T
        self._check(name)
        c = np.asarray(self._coeffs[name])
        names = list(self._coords)
        bad = [k for k in points if k not in names]
        if bad or not points:
            raise InputTypeError(f"mode.at({name!r}, {', '.join(f'{k}=...' for k in points)}): "
                                 "name the non-periodic coordinate" + (did_you_mean(bad[0], names)
                                                                       if bad else ""),
                                 fix=f"mode.at({name!r}, {names[-1]}=np.linspace(a, b, 101))",
                                 doc=_DOC)
        off = c.ndim - len(names)
        for ax, cname in enumerate(names):
            b = self._bases.get(cname)
            if cname in points:
                c = np.moveaxis(np.tensordot(_evaluation_matrix(b, np.atleast_1d(
                    np.asarray(points[cname], dtype=float)), cname), c, axes=(1, off + ax)),
                    0, off + ax)
            else:
                c = T.backward(c, b, axis=off + ax, dealias=False)
        return c

    def explain(self) -> str:
        """Eigenvalue, residual, unknowns, grid sizes and the normalisation."""
        sizes = ", ".join(f"{c}: {g.size}" for c, g in self._coords.items())
        return (f"EVPMode {self.index} [{self.label}]: eigenvalue {self.value:.12g}, residual "
                f"{self.residual:.1e}\n  unknowns: {', '.join(self.names)} (complex grid values "
                f"on the non-dealiased grid, {sizes}; tau variables left out)\n"
                f"  normalisation: {NORMALIZATION} (here: {self.norm_field})")

    def __repr__(self) -> str:
        return f"<EVPMode {self.index}: {self.value:.6g}, unknowns {', '.join(self.names)}>"


def _evaluation_matrix(b: Any, pts: np.ndarray, cname: str) -> np.ndarray:
    """Rows evaluating the expansion of basis ``b`` at the points ``pts`` (exact)."""
    from ..discretize.families import family_of
    from ..ir.bases import Chebyshev
    a, L = b.bounds[0], b.length
    if np.any(pts < a - 1e-12 * abs(L)) or np.any(pts > a + L + 1e-12 * abs(L)):
        raise OptionError(f"mode.at({cname}=...): points outside [{a:g}, {a + L:g}]",
                          fix=f"points in [{a:g}, {a + L:g}]", doc=_DOC)
    fam = family_of(b)
    if fam is not None and hasattr(fam, "interp_row"):
        return np.array([fam.interp_row(float(p)) for p in pts])
    if isinstance(b, Chebyshev) and getattr(b, "map", None) is None:
        t = np.clip(2 * (pts - a) / L - 1, -1.0, 1.0)
        return np.polynomial.chebyshev.chebvander(t, b.size - 1)
    raise UnsupportedError(f"mode.at({cname}=...) on a {type(b).__name__} axis",
                           why="exact evaluation is implemented for the non-periodic bases "
                               "(Chebyshev, Legendre, Jacobi, compound, Hermite, Laguerre)",
                           fix=f"mode.grid(name) and mode.coordinate({cname!r}) (grid values)",
                           doc=_DOC)


def _tau_names(plan: Any) -> set[str]:
    """Names of the tau variables of the plan's problem: every field lifted by a ``lift(...)``
    term of an equation (structural: expert taus may have any name), and the native layer's
    reserved ``tau_*`` names (its gauge taus enter without a lift; ``dtau_*`` once linearized;
    a native unknown cannot take these names)."""
    from ..ir.expr_calculus import Lift
    out = set()
    for eq in getattr(plan.problem, "equations", ()):
        for side in (eq.lhs, eq.rhs):
            for e in side.walk():
                if isinstance(e, Lift):
                    out |= {f.name for f in e.args[0].free_fields()}
    out |= {v.name for v in plan.variables
            if v.name.startswith("tau_") or v.name.startswith("dtau_")}
    return out


def _unpack(plan: Any, group: Any, x: np.ndarray) -> dict[str, np.ndarray]:
    """Full coefficient arrays of every variable from one group vector (full space)."""
    from ..discretize.plan import _comps
    coords = plan.domain.coordsys.coords
    zc = plan.layout.zcoord
    fax = [coords.index(d.coord) for d in plan.layout.fdirs]
    out = {}
    for v, vl in zip(plan.variables, plan.var_layouts(group), strict=True):
        arr = np.zeros(plan._grid_shape(v.domain, v.tensorsig), dtype=complex)
        fidx = plan.layout.fourier_indices(vl.fdep, group)
        for ci, c in enumerate(_comps(v.tensorsig)):
            r0 = vl.offset + ci * vl.block
            for fj, fi in enumerate(fidx):
                idx = list(c) + [0] * len(coords)
                for a, j in zip(fax, fi, strict=True):
                    idx[len(c) + a] = j
                if vl.zdep:
                    idx[len(c) + coords.index(zc)] = slice(None)
                    arr[tuple(idx)] = x[r0 + fj * vl.nz: r0 + (fj + 1) * vl.nz]
                else:
                    arr[tuple(idx)] = x[r0 + fj]
        out[v.name] = arr
    return out


def mode_of(res: Any, i: int) -> EVPMode:
    """The :class:`EVPMode` of mode ``i`` of the result ``res`` (see the module docstring)."""
    ctx = getattr(res, "context", None)
    if res.vectors is None or ctx is None:
        raise UnsupportedError(
            f"mode({i}): this eigenvalue result carries no eigenvectors with their layout",
            why="the eigenfunctions are unpacked from the group system of the plan that produced "
                "them; results gathered over several groups (sine/cosine axes) or built by hand "
                "keep only the eigenvalues",
            fix="m.eigenvalues(N, ...) / lin.eigenvalues(N, k=...) on a Cartesian geometry, or "
                "the expert conatus.eigenvalues(plan, group)", doc=_DOC)
    if not isinstance(i, (int, np.integer)) or isinstance(i, bool):
        raise InputTypeError(f"mode({i!r}): an integer index into res.values",
                             fix="i = int(res.values.imag.argmax())", doc=_DOC)
    n = res.values.size
    if not -n <= i < n:
        raise OptionError(f"mode({i}): the result has {n} mode(s)", fix=f"0 <= i < {n}",
                          doc=_DOC)
    plan, group, reduced, rename = ctx
    if not hasattr(plan, "var_layouts") or not hasattr(plan.layout, "fourier_indices"):
        raise UnsupportedError(f"mode({i}) on a {type(plan).__name__}",
                               why="the eigenfunction unpacking is implemented for Cartesian "
                                   "plans (Fourier x one non-periodic axis)",
                               fix="res.vectors[:, i] holds the raw coefficient column", doc=_DOC)
    x = np.asarray(res.vectors[:, i])
    if reduced:                                    # symmetry subspace -> full group vector
        x = plan.embed(x, group)
    coeffs = {}
    taus = _tau_names(plan)
    for name, c in _unpack(plan, group, x).items():
        if name in taus:
            continue                               # tau variables are not part of the mode
        coeffs[rename.get(name, name)] = c
    inv = {rename.get(n_, n_): n_ for n_ in (v.name for v in plan.variables)}
    grids = {k: plan.to_grid(inv[k], c) for k, c in coeffs.items()}
    # declared normalisation (NORMALIZATION)
    norm_field, scale = "all unknowns zero", 1.0 + 0j
    cmax = max((float(np.abs(c).max()) for c in coeffs.values() if c.size), default=0.0)
    for k, g in grids.items():
        j = int(np.argmax(np.abs(g)))
        peak = g.flat[j]
        if abs(peak) > 0 and float(np.abs(coeffs[k]).max()) > NORMALIZATION_ZERO_REL * cmax:
            norm_field, scale = f"{k}, max |{k}| = 1 at its largest grid value", 1.0 / peak
            break
    coeffs = {k: c * scale for k, c in coeffs.items()}
    grids = {k: g * scale for k, g in grids.items()}
    coords = {}
    from ..reference import transforms as T
    for c in plan.domain.coordsys.coords:
        b = plan.domain.basis(c)
        if b is not None:
            coords[c.name] = np.asarray(T.grid(b, dealias=False))
    md = EVPMode(res.values[i], res.residuals[i], coeffs, grids, coords, res.label,
                 norm_field, int(i))
    md._bases = {c.name: plan.domain.basis(c) for c in plan.domain.coordsys.coords
                 if plan.domain.basis(c) is not None}
    return md

