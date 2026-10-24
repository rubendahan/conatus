"""Declared inner products for snapshot data: ⟨a, b⟩_W = aᴴ W b (data-driven toolkit).

Every decomposition (POD, DMD, SPOD, EDMD) is defined by an inner product on the snapshot space;
the call declares it. Three declared kinds:

* ``DataInnerProduct.quadrature(w)``: diagonal W = diag(w), w the quadrature weight of each entry
  of a flattened snapshot (e.g. ``conatus.diagnostics.Grid.cell_volumes()``, trapezoidal ×
  Fejér: exact for the resolved fields; :meth:`from_grid` stacks it for several components with
  declared component weights);
* ``DataInnerProduct.gram(W)``: a Hermitian positive-definite matrix (e.g. the exact Chebyshev
  Gram of coefficient vectors, ``Descriptor.W``); used through its Cholesky factor W = Fᴴ F;
* ``DataInnerProduct.euclidean(n)``: W = I, *declared* (the unweighted choice is a choice).

``half(X) = F X`` maps data to coordinates where the product is Euclidean, ``unhalf`` inverts it.
Arrays may be NumPy or CuPy (the array module of the data is used; nothing is copied to the host
except the small reduced matrices of each method).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._arrays import array_module
from ..errors import OptionError
from .._repr import compact_repr

__all__ = ["DataInnerProduct", "array_module"]

#: Relative Hermitian defect ||W - W^H|| / ||W|| above which a Gram matrix is refused (declared in
#: ``conatus.tolerances.DECLARED``; the measured defect is in the message).
HERMITIAN_TOL = 1e-13


def to_host(a) -> np.ndarray:
    xp = array_module(a)
    return a.get() if xp is not np else np.asarray(a)


@dataclass
class DataInnerProduct:
    """⟨a, b⟩ = aᴴ W b on flattened snapshots of length n (see module doc)."""
    __repr__ = compact_repr   # one line, arrays by shape (as)

    kind: str                       # "quadrature" | "gram" | "euclidean"
    n: int
    w: Any = None                   # quadrature weights (length n)
    F: Any = None                   # upper Cholesky factor of the Gram (W = Fᴴ F)
    note: str = ""

    # ------------------------------------------------------------------ constructors
    @classmethod
    def quadrature(cls, w, *, note: str = "") -> DataInnerProduct:
        w = np.asarray(to_host(w))
        if w.ndim != 1 or np.iscomplexobj(w):
            # audit inputs: a 2D array was flattened, complex weights cast to float
            raise OptionError(f"quadrature weights: a 1D real array, got shape {w.shape}, dtype "
                              f"{w.dtype}", fix="one weight per entry of a snapshot: w.ravel() "
                                                  "of the grid weights, in snapshot order")
        w = w.astype(float)
        if not np.all(w > 0) or not np.all(np.isfinite(w)):
            raise OptionError("quadrature weights must all be > 0 (a zero weight makes W singular:"
                              " drop those entries from the snapshots instead, explicitly)")
        return cls("quadrature", w.size, w=w, note=note)

    @classmethod
    def gram(cls, W, *, note: str = "") -> DataInnerProduct:
        W = np.asarray(to_host(W))
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise OptionError("gram(W): W must be a square matrix")
        herm = np.linalg.norm(W - W.conj().T) / np.linalg.norm(W)
        if herm > HERMITIAN_TOL:
            raise OptionError(f"gram(W): W is not Hermitian (relative defect {herm:.1e})")
        try:
            F = np.linalg.cholesky(W).conj().T
        except np.linalg.LinAlgError:
            raise OptionError("gram(W): W is not positive definite") from None
        return cls("gram", W.shape[0], F=F, note=note)

    @classmethod
    def euclidean(cls, n: int) -> DataInnerProduct:
        from ..errors import check_count               # audit inputs: int(3.5) = 3, n = 0
        return cls("euclidean", check_count(n, "n", "DataInnerProduct.euclidean()"),
                   note="W = I (declared)")

    @classmethod
    def from_grid(cls, grid, *, components: Sequence[str], weights: dict) -> DataInnerProduct:
        """Stacked components (e.g. u, v, w) on a ``conatus.diagnostics.Grid``: ⟨a, b⟩ =
        Σ_c weights[c] ∫ ā_c b_c dV with the grid's exact quadrature (trapezoid × Fejér).
        Snapshots are flattened component-major (C order of an array ``(ncomp, *grid.shape)``)."""
        missing = [c for c in components if c not in weights]
        if missing:
            raise OptionError(f"from_grid: no weight for components {missing} (every component "
                              "needs an explicit weight)")
        cv = to_host(grid.cell_volumes()).ravel()
        w = np.concatenate([weights[c] * cv for c in components])
        return cls.quadrature(w, note=f"grid quadrature (trapezoid × Fejér), components "
                                       f"{list(components)}, weights {dict(weights)}")

    # ------------------------------------------------------------------ maps
    def check(self, X) -> None:
        # snapshots: a finite 2D array (audit inputs: NaN snapshots ended in a bare
        # LinAlgError of the POD/DMD decomposition)
        if getattr(X, "ndim", None) != 2:
            raise OptionError(f"snapshots: a 2D array (entries x snapshots), got shape "
                              f"{getattr(X, 'shape', None)}")
        if X.shape[0] != self.n:
            raise OptionError(f"snapshots have {X.shape[0]} entries, the inner product {self.n}")

    def half(self, X):
        """F X: coordinates in which the product is Euclidean (X: n × m, NumPy or CuPy)."""
        self.check(X)
        xp = array_module(X)
        if self.kind == "euclidean":
            return X
        if self.kind == "quadrature":
            return xp.asarray(np.sqrt(self.w))[:, None] * X
        return xp.asarray(self.F) @ X

    def unhalf(self, Y):
        """F⁻¹ Y (inverse of :meth:`half`)."""
        xp = array_module(Y)
        if self.kind == "euclidean":
            return Y
        if self.kind == "quadrature":
            return Y / xp.asarray(np.sqrt(self.w))[:, None]
        import scipy.linalg as sla
        if xp is np:
            return sla.solve_triangular(self.F, Y)
        return xp.linalg.solve(xp.asarray(self.F), Y)

    def inner(self, a, b):
        """aᴴ W b for vectors or matrices (columns)."""
        A, B = self.half(a if a.ndim == 2 else a[:, None]), self.half(b if b.ndim == 2 else b[:, None])
        out = A.conj().T @ B
        return out if (a.ndim == 2 or b.ndim == 2) else out[0, 0]

    def explain(self) -> str:
        what = {"quadrature": "diagonal quadrature weights", "gram": "Hermitian positive-definite "
                "Gram matrix (Cholesky factor)", "euclidean": "Euclidean (declared)"}[self.kind]
        return f"DataInnerProduct: ⟨a, b⟩ = aᴴ W b, W = {what}, n = {self.n}. {self.note}"


def check_data(ip: DataInnerProduct, X: Any) -> None:
    """``ip.check(X)`` plus finite values: the user's snapshots at the entry of POD and DMD
    (audit inputs: NaN snapshots ended in a bare LinAlgError of the decomposition)."""
    ip.check(X)
    xp = array_module(X)
    if not bool(xp.all(xp.isfinite(X))):
        raise OptionError("snapshots: non-finite values (NaN or inf)",
                          fix="drop or repair the snapshots that hold them, explicitly")
