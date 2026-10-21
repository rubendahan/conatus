"""``m.eigenvalues(N, ..., region=cn.box(...))``: every eigenvalue of one group in a region, with a
count (argument principle on a sampled boundary walk) and Beyn's method on the pencil ``L + s M``
of the group."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np

from ..solvers.evp import EVPResult, _residuals
from .beyn import RESIDUAL_TOL, BeynResult, beyn
from .operators import Pencil
from .region import ContourBox

__all__ = ["RegionEVPResult", "region_eigenvalues"]


@dataclass(repr=False)
class RegionEVPResult(EVPResult):
    """An :class:`~conatus.solvers.evp.EVPResult` restricted to a region of the eigenvalue plane:
    ``values`` are every eigenvalue of the discrete pencil in ``region`` (one entry per distinct
    eigenvalue, ``multiplicity`` the eigenvalues counted in a small square around it: a multiple
    eigenvalue or a cluster not separated at that size) and ``count`` their number,
    multiplicities included, by the argument principle (``count_method``)."""

    count: int = 0
    count_method: str = ""
    region: ContourBox | None = None
    multiplicity: np.ndarray = dc_field(default_factory=lambda: np.zeros(0, dtype=int))
    contour: BeynResult | None = dc_field(default=None, compare=False)

    def select(self, idx, why: str) -> RegionEVPResult:
        """The modes ``idx`` (a filter), the count of the region kept and the filter logged."""
        idx = np.asarray(idx, dtype=int)
        base = EVPResult.select(self, idx, why)
        return RegionEVPResult(**{f: getattr(base, f) for f in EVPResult.__dataclass_fields__},
                               count=self.count, count_method=self.count_method,
                               region=self.region, multiplicity=self.multiplicity[idx],
                               contour=self.contour)

    def __repr__(self) -> str:
        head = EVPResult.__repr__(self)
        return (head[:-len("; .explain() for the log)")] +
                f"; count {self.count} in {self.region.describe()}; .explain() for the log)")

    def explain(self, n: int = 6, key: str = "growth", *, full: bool = False) -> str:
        """The count of the region, then the usual eigenvalue explanation of the modes."""
        kept = self.values.size
        total = int(self.multiplicity.sum())
        lines = [f"region {self.region.describe()}: count {self.count} (multiplicities included), "
                 f"by {self.count_method}"]
        if total != self.count:
            lines.append(f"  {kept} of the modes kept by the filters below (multiplicities "
                         f"{total} of {self.count})")
        if (self.multiplicity > 1).any():
            lines.append("  several eigenvalues in one square (multiple or not separated): " + ", ".join(
                f"{self.values[i]:.10g} (x{int(self.multiplicity[i])})"
                for i in np.nonzero(self.multiplicity > 1)[0]))
        body = EVPResult.explain(self, n, key, full=full).split("\n")
        body[0] = body[0].replace(f", {self.n_infinite} infinite (M singular on tau/boundary rows)",
                                  " (eigenvalues outside the region not computed)")
        lines += body
        if full and self.contour is not None:
            lines += ["  " + s for s in self.contour.explain(n=n).split("\n")]
        return "\n".join(lines)


def region_eigenvalues(L, M, region: ContourBox, *, label: str = "",
                       residual_tol: float = RESIDUAL_TOL, **beyn_options: Any) -> RegionEVPResult:
    """Every eigenvalue ``s`` of ``(L + s M) x = 0`` in ``region`` (count by the argument
    principle, roots by Beyn's method, each checked by its residual; :func:`conatus.contour.beyn`)."""
    op = Pencil(L, M, label=label)
    b = beyn(op, region, residual_tol=residual_tol, **beyn_options)
    V = b.vectors
    res = _residuals(L, M, b.values, V) if b.values.size else np.zeros(0)
    method = "contour: argument-principle count + Beyn's method (polished, residual-checked)"
    r = RegionEVPResult(b.values, V, res, method, label, 0, count=b.count,
                        count_method=b.count_method, region=region, multiplicity=b.multiplicity,
                        contour=b)
    r.log.append(f"region {region.describe()}: Z = {b.count} eigenvalue(s) of the discrete "
                 f"pencil (spurious ones of the discretisation included), {b.counted.evaluations}"
                 f" determinant evaluations; roots accepted at residual <= {residual_tol:g}")
    r.log += b.log
    return r
