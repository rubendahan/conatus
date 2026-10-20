"""Audit: silent substitutions found mid-one regression test each.

Each test was written before its fix and failed on the code of the day (message in the docstring).
The exact-rate tests that guard against-type defects are in tests/discretize/test_exact_rates.py.
"""
import warnings

import numpy as np
import pytest

from conatus.ir import CartesianCoordinates, RealFourier
from conatus.reference import transforms as RT


def _complex_realfourier_coeffs(N=12, ncol=3):
    rng = np.random.default_rng(7)
    c = rng.standard_normal((N, ncol)) + 1j * rng.standard_normal((N, ncol))
    c[1] = 0                                  # the -sin(0 x) slot is structurally empty
    return c


def _realfourier():
    cs = CartesianCoordinates("x")
    return RealFourier(cs["x"], 12, (0, 2 * np.pi), dealias=3 / 2)


@pytest.mark.parametrize("dealias", [True, False])
def test_reference_realfourier_backward_keeps_imaginary_part(dealias):
    """C1: RealFourier coefficients may be complex (a complex LBVP solution, or a complex
    system on a RealFourier axis); backward() folded Re and Im into one Hermitian spectrum and
    returned a real array (error 2.8 on this case), without error or warning."""
    b, c = _realfourier(), _complex_realfourier_coeffs()
    g = RT.backward(c, b, axis=0, dealias=dealias)
    ref = RT.backward(c.real, b, axis=0, dealias=dealias) + 1j * RT.backward(c.imag, b, axis=0, dealias=dealias)
    assert np.iscomplexobj(g)
    assert np.abs(g - ref).max() < 1e-14 * np.abs(ref).max()


def test_core_realfourier_backward_keeps_imaginary_part():
    """C2: same defect in the C++ path (the former runtime.transforms): the imaginary part was
    dropped by astype(float64) with only a numpy ComplexWarning (error 2.4). That FFTW module was
    removed with FFTW; the check now guards the production CPU path (timestep.xp_transforms)."""
    from conatus.timestep import xp_transforms as XT
    b, c = _realfourier(), _complex_realfourier_coeffs()
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        g = XT.backward(np, c, b, axis=0)
    ref = RT.backward(c.real, b, axis=0) + 1j * RT.backward(c.imag, b, axis=0)
    assert np.iscomplexobj(g)
    assert np.abs(g - ref).max() < 1e-13 * np.abs(ref).max()


# ----------------------------------------------------------------------------------------------
# InnerProduct (linearize/inner_product.py): the product defines the adjoint
# ----------------------------------------------------------------------------------------------

def _two_field_plan(mapped=False):
    from conatus.discretize import discretize
    from conatus.ir import (EVP, Chebyshev, Domain, Field, Parameter, diff)
    from conatus.ir.maps import MappedChebyshev, RationalMap
    cs = CartesianCoordinates("z")
    z = cs["z"]
    b = MappedChebyshev(z, 12, RationalMap(1.0), dealias=1) if mapped else Chebyshev(z, 12, (-1, 1), dealias=1)
    D = Domain(cs, [b])
    u, v, s = Field("u", D), Field("v", D), Parameter("s")
    pb = EVP([u, v], eigenvalue=s)
    pb.add_equation((s * u + diff(u, z) + v, 0))
    pb.add_equation((s * v + u, 0))
    return discretize(pb)


def test_inner_product_partial_weights_refused():
    """C3: InnerProduct(plan, {"u": 1}) gave v the weight 0 without a word (the state, the norm and
    the adjoint W^-1 A^H W silently lost v). A weight must be given for every non-tau unknown
    (0 is allowed, explicitly)."""
    from conatus.linearize.inner_product import InnerProduct
    plan = _two_field_plan()
    with pytest.raises(ValueError, match="'v'"):
        InnerProduct(plan, {"u": 1.0})
    ip = InnerProduct(plan, {"u": 1.0, "v": 0.0})       # explicit zero: accepted and shown
    assert "v: 0" in ip.explain()


def test_inner_product_negative_weight_refused():
    """C4: a negative weight made W indefinite; norm() then clamped a negative |x|^2 to 0."""
    from conatus.linearize.inner_product import InnerProduct
    with pytest.raises(ValueError, match="negative"):
        InnerProduct(_two_field_plan(), {"u": 1.0, "v": -1.0})


def test_inner_product_mapped_chebyshev_refused():
    """C5: on a MappedChebyshev axis the Gram used the affine formula on the physical bounds
    ((0, inf) here: non-finite entries), ignoring the map's metric."""
    from conatus.linearize.inner_product import InnerProduct
    ip = InnerProduct(_two_field_plan(mapped=True))
    with pytest.raises(NotImplementedError, match="MappedChebyshev"):
        ip.gram()


def test_polar_gather_state_refuses_unrepresented_data():
    """C6: PolarGlobalSystem.gather_state accepted check_subspace but ignored it: data in the slots
    dropped by the per-m radial truncation vanished silently (the Cartesian system refuses
    data outside its subspace)."""
    import conatus.expert as C
    PC = C.PolarCoordinates("phi", "r")
    PHI, R = PC["phi"], PC["r"]
    br = C.DiskBasis(R, 12, 1.0, dealias=3 / 2)
    D = C.Domain(PC, [C.RealFourier(PHI, 8, (0, 2 * np.pi), dealias=3 / 2), br])
    u, t1 = C.Field("u", D), C.Field("t1", D.without(R))
    pb = C.IVP([u, t1])
    pb.add_equation((C.dt(u) - C.lap(u) + C.lift(t1, br, -1, lam=2), 0))
    pb.add_equation((C.interp(u, R, "right"), 0))
    sys_ = C.IVPSolver(C.discretize(pb), "RK443", 1e-3).system
    shapes = sys_.var_shapes
    coeffs = [np.zeros(int(np.prod(sh)), dtype=sys_.dtype) for sh in shapes]
    dropped = np.setdiff1d(np.arange(coeffs[0].size), sys_.cols_flat[sys_._col_sel[0]])
    assert dropped.size                                  # the truncation drops some slots
    sys_.gather_state(coeffs)                            # zeros there: accepted
    coeffs[0][dropped[0]] = 1.0
    with pytest.raises(ValueError, match="outside the represented modes"):
        sys_.gather_state(coeffs)


def test_polar_plan_refuses_stray_params():
    """C7: discretize(polar problem, params={"Ra_": ...}) ignored the unknown key and used the value
    declared on the Parameter (a typo changed the physics silently). The Cartesian Plan has the same
    defect (plan.py, forwarded to its owner); stray ncc keys were already refused in both."""
    import conatus.expert as C
    from conatus.discretize import DiscretizationError, discretize
    PC = C.PolarCoordinates("phi", "r")
    PHI, R = PC["phi"], PC["r"]
    br = C.DiskBasis(R, 8, 1.0, dealias=1)
    D = C.Domain(PC, [C.RealFourier(PHI, 8, (0, 2 * np.pi), dealias=1), br])
    u, t1, f = C.Field("u", D), C.Field("t1", D.without(R)), C.Field("f", D)
    k = C.Parameter("k", 2.0)
    pb = C.LBVP([u, t1])
    pb.add_equation((C.lap(u) - k * u + C.lift(t1, br, -1, lam=2), f))
    pb.add_equation((C.interp(u, R, "right"), 0))
    discretize(pb, params={"k": 3.0})                   # a real parameter: accepted
    with pytest.raises(DiscretizationError, match="'kk'"):
        discretize(pb, params={"kk": 3.0})


def test_bessel_modes_refuses_short_spectrum():
    """C8a: Disk.bessel_modes (phase-1 standalone disk) dropped complex / non-positive eigenvalues
    without a count and returned fewer than n_modes values without an error."""
    from conatus.discretize.polar import Disk
    d = Disk(8)
    assert d.bessel_modes(0, 3).size == 3
    with pytest.raises(ValueError, match="n_modes"):
        d.bessel_modes(0, 40)                          # more than N resolved modes


def test_disk_trace_row_refuses_r_power():
    """C8b: Disk.trace_row accepted an r-power p != 0 and ignored it (the other disk rows refuse)."""
    from conatus.discretize.polar import Disk
    d = Disk(8)
    d.trace_row(0, 0, 0, "outer")
    with pytest.raises(ValueError, match="r-multiplier"):
        d.trace_row(0, 0, 1, "outer")
