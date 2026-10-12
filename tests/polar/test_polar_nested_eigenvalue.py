"""EVP with the eigenvalue nested in a product, c * (s u + ...): each term must reach its
matrix. The polar and spherical compilers dropped the nested s u from M (silently when another
equation kept M non-zero: the rotating-shell onset came out at Ra_c = 67.4 instead of 55.9); the
Cartesian compiler was right. Checked here on the disk and on the sphere."""
import numpy as np
import pytest

import conatus.expert as C
from conatus.discretize import discretize


def _spectrum(geometry: str, nested: bool) -> np.ndarray:
    s = C.Parameter("s")
    if geometry == "disk":
        pc = C.PolarCoordinates("phi", "r")
        r = pc["r"]
        br = C.DiskBasis(r, 12, 1.0, dealias=1)
        D = C.Domain(pc, [C.ComplexFourier(pc["phi"], 4, (0, 2 * np.pi), dealias=1), br])
        taus = [C.Field("t", D.without(r), dtype=complex)]
    else:
        s2 = C.S2Coordinates("phi", "theta")
        D = C.Domain(s2, [C.ComplexFourier(s2["phi"], 8, (0, 2 * np.pi), dealias=1),
                          C.SphereBasis(s2["theta"], 6, dealias=1)])
        taus = []
    u, v = C.Field("u", D, dtype=complex), C.Field("v", D, dtype=complex)
    pb = C.EVP([u, v] + taus, eigenvalue=s)
    body = s * u + C.lap(u)
    if taus:
        body = body + C.lift(taus[0], br, -1, lam=2)
    pb.add_equation(((3.0 * body) if nested else body, 0))
    pb.add_equation((s * v + 2 * v, 0))            # keeps M non-zero: the old defect was silent
    if taus:
        pb.add_equation((C.interp(u, r, "right"), 0))
    plan = discretize(pb)
    w = np.sort(C.eigenvalues(plan, plan.group(1)).values.real)
    return w[np.isfinite(w)]


@pytest.mark.parametrize("geometry", ["disk", "sphere"])
def test_nested_eigenvalue(geometry):
    plain, nested = _spectrum(geometry, False), _spectrum(geometry, True)
    assert len(plain) == len(nested)
    assert np.abs(plain - nested).max() < 1e-10 * np.abs(plain).max()
