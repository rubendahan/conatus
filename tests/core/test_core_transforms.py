"""Level B (§70) for the production CPU transforms (``timestep.xp_transforms`` with NumPy, i.e.
pocketfft) against the NumPy oracle ``conatus.reference.transforms``.

Until 0.1.0 this file tested the C++ host transforms of the core (FFTW). No solver called them, and
FFTW (GPL-2.0-or-later) was dropped from every build with them: the same checks now run on
the path the solvers use."""
import numpy as np
import pytest

from conatus.ir import CartesianCoordinates, Chebyshev, ComplexFourier, RealFourier
from conatus.reference import transforms as REF
from conatus.timestep import xp_transforms as XT


def _bases():
    cs = CartesianCoordinates("x")
    x = cs["x"]
    return [Chebyshev(x, 24, (-1, 2), dealias=1), Chebyshev(x, 24, (0, 1), dealias=3 / 2),
            RealFourier(x, 16, (0, 3), dealias=1), RealFourier(x, 16, (0, 3), dealias=3 / 2),
            ComplexFourier(x, 16, (0, 2 * np.pi), dealias=3 / 2)]


@pytest.mark.parametrize("b", _bases(), ids=lambda b: f"{type(b).__name__}-{b.dealias}")
@pytest.mark.parametrize("axis", [0, 1])
def test_basis_transforms_match_reference(b, axis, rng):
    M = REF.grid_size(b)
    assert b.grid_size == M
    shape = [3, 3]
    shape[axis] = M
    f = rng.standard_normal(shape)
    if isinstance(b, ComplexFourier):
        f = f + 1j * rng.standard_normal(shape)
    c_ref, c = REF.forward(f, b, axis), XT.forward(np, f, b, axis)
    assert np.max(abs(c - c_ref)) < 1e-13
    g_ref, g = REF.backward(c_ref, b, axis), XT.backward(np, c_ref, b, axis)
    assert np.max(abs(g - g_ref)) < 1e-13
    # coefficient round trip is the identity (dealiasing projection is exact on coeff data)
    assert np.max(abs(XT.forward(np, XT.backward(np, c, b, axis), b, axis) - c)) < 1e-13


def test_chebyshev_complex_rows(rng):
    b = _bases()[1]
    f = rng.standard_normal((REF.grid_size(b), 4)) + 1j * rng.standard_normal((REF.grid_size(b), 4))
    assert np.max(abs(XT.forward(np, f, b) - REF.forward(f, b))) < 1e-13
    c = REF.forward(f, b)
    assert np.max(abs(XT.backward(np, c, b) - REF.backward(c, b))) < 1e-13


def test_transforms_are_deterministic(rng):
    b = _bases()[0]
    f = rng.standard_normal((REF.grid_size(b), 7))
    assert np.array_equal(XT.forward(np, f, b), XT.forward(np, f, b))     # bitwise


def test_fourier_needs_even_size():
    with pytest.raises(ValueError):
        REF.forward(np.zeros(15), RealFourier(CartesianCoordinates("x")["x"], 15, (0, 1), dealias=1))


@pytest.mark.parametrize("dealias", [1, 3 / 2])
def test_realfourier_complex_data_keeps_imaginary_part(dealias, rng):
    """Complex data or coefficients on a RealFourier axis went through rfft/irfft and the
    imaginary part was lost (backward returned a real array) without a warning; Re and Im are now
    transformed separately, as the reference does."""
    b = RealFourier(CartesianCoordinates("x")["x"], 16, (0, 3), dealias=dealias)
    f = rng.standard_normal((b.grid_size, 3)) + 1j * rng.standard_normal((b.grid_size, 3))
    c = XT.forward(np, f, b)
    assert np.max(abs(c - (REF.forward(f.real, b) + 1j * REF.forward(f.imag, b)))) < 1e-13
    g = XT.backward(np, c, b)
    assert np.iscomplexobj(g)
    assert np.max(abs(g - (REF.backward(c.real, b) + 1j * REF.backward(c.imag, b)))) < 1e-13


def _cupy_device():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001 -- no CuPy or no driver: the GPU twin is skipped
        return False


@pytest.mark.gpu
@pytest.mark.skipif(not _cupy_device(), reason="no CUDA device")
@pytest.mark.parametrize("dealias", [1, 3 / 2])
def test_realfourier_complex_data_keeps_imaginary_part_gpu(dealias, rng):
    """GPU twin of the test above: the same branch of ``xp_transforms`` with CuPy (cuFFT)."""
    import cupy as cp
    b = RealFourier(CartesianCoordinates("x")["x"], 16, (0, 3), dealias=dealias)
    f = rng.standard_normal((b.grid_size, 3)) + 1j * rng.standard_normal((b.grid_size, 3))
    c = cp.asnumpy(XT.forward(cp, cp.asarray(f), b))
    assert np.max(abs(c - (REF.forward(f.real, b) + 1j * REF.forward(f.imag, b)))) < 1e-13
    g = XT.backward(cp, cp.asarray(c), b)
    assert cp.iscomplexobj(g)
    g = cp.asnumpy(g)
    assert np.max(abs(g - (REF.backward(c.real, b) + 1j * REF.backward(c.imag, b)))) < 1e-13
