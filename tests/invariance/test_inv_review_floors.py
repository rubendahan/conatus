"""Critic review of the invariance audit: floors in the unit of the quantity, relative real-part
checks, and the mapped-box stepper's checkpoints refused with a typed error."""
from __future__ import annotations

import numpy as np
import pytest

import conatus as cn
from conatus._scales import pencil_unit, spectral_floor, spectral_scale
from conatus.errors import OptionError, UnsupportedError
from conatus.solvers.evp import drift_ratios


def test_drift_floor_survives_a_spectrum_of_mostly_zero_eigenvalues():
    """The floor was DRIFT_FLOOR x median |sigma|: 0 when most eigenvalues are 0, so a neutral
    mode computed as rounding noise at both resolutions (1e-17 then 2e-17) had ratio 1 and was
    dropped as drifting."""
    lo = np.array([1e-17, 0.0, 1.0, 2.0])
    hi = np.array([2e-17, 0, 0, 0, 0, 0, 1.0, 2.0])
    d = drift_ratios(lo, hi)
    assert d[0] < 1e-12 and d[1] == 0 and d[2] == 0
    # invariant under a change of unit of sigma
    assert np.allclose(drift_ratios(1e4 * lo, 1e4 * hi), d, rtol=1e-12, atol=0)


@pytest.mark.parametrize("unit", [1e-6, 1.0, 1e4])
def test_scales_follow_the_unit(unit):
    w = unit * np.array([0, 0, 1 + 1j, -2, 3j, 1e-3])
    assert spectral_scale(w) == pytest.approx(unit * spectral_scale(w / unit))
    assert spectral_floor(w) == pytest.approx(unit * spectral_floor(w / unit))
    L, M = unit * np.diag([1.0, 2.0]), np.eye(2)
    assert pencil_unit(L, M) == pytest.approx(unit * pencil_unit(L / unit, M))


def test_iterate_imaginary_part_is_judged_against_the_iterate():
    from conatus.solvers.nlbvp import _real_part
    assert np.isrealobj(_real_part(np.array([1.0 + 1e-17j, 2.0]), "u"))
    with pytest.raises(OptionError, match="imaginary part"):
        _real_part(1e-12 * np.array([1.0 + 1e-2j, 2.0]), "u")    # genuine, amplitude 1e-12


def test_polar_data_below_unit_amplitude_is_not_dropped_silently():
    """gather_state compared the unrepresented part with 1e-12 x max(1, |data|): data of
    amplitude 1e-12 (an infinitesimal perturbation) lost it without a word."""
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
    coeffs = [np.zeros(int(np.prod(sh)), dtype=sys_.dtype) for sh in sys_.var_shapes]
    kept = sys_.cols_flat[sys_._col_sel[0]]
    dropped = np.setdiff1d(np.arange(coeffs[0].size), kept)
    coeffs[0][kept[0]] = 1e-12
    coeffs[0][dropped[0]] = 1e-12
    with pytest.raises(ValueError, match="outside the represented modes"):
        sys_.gather_state(coeffs)


@pytest.mark.parametrize("call", ["save", "restart", "checkpoints", "resume_or_start"])
def test_mapped_box_checkpoints_are_refused_with_a_typed_error(tmp_path, call):
    geo = cn.MappedBox(x=cn.periodic(2 * np.pi), z=cn.between(bottom=0.0, top=1.0))
    m = cn.Model(geo)
    m.scalar("u")
    m.equations("dt(u) - dz(dz(u)) = 0")
    m.at("bottom", "u=0")
    m.at("top", "u=0")
    s = m.simulate((8, 8), scheme="RK222", dt=1e-3)
    with pytest.raises(UnsupportedError, match="mapped domain"):
        getattr(s, call)(tmp_path / "x.h5")
