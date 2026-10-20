"""Equivalent solver paths give the same numbers.

The options documented as "exact" or "same numbers" choose another route through the same
discrete problem: they may change the rounding, never the answer.
  IVP  reduced= (symmetry / gauge reduction), ordering= (interleaved or variable-major band),
       complexify= (RealFourier pairs as complex systems), decouple= (block-diagonal groups
       split), f_backend= (reference evaluator or fused kernels);
  EVP  method='dense' (QZ), 'shift-invert' (Arnoldi near a target), a region (Beyn's contour
       method).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import conatus as cn
from conatus._tolerances_internal import INTERNAL

RTOL = INTERNAL["INVARIANCE_ROUNDING"].value
ETOL = INTERNAL["INVARIANCE_EIGENVALUE"].value


def _rel(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), np.finfo(float).tiny))


def _rb_sim(**kw):
    m = cn.Model(cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5)), name="RB", Ra=2500.0, Pr=1.0)
    m.velocity("u"), m.pressure("p"), m.scalar("T")
    m.equations("""
        momentum:   dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)
        continuity: div(u) = 0
        heat:       dt(T) - lap(T) = -u·grad(T)
    """)
    m.at("bottom", u=0, T=0.5)
    m.at("top", u=0, T=-0.5)
    sim = m.simulate(N=(16, 12), scheme=kw.pop("scheme", "RK222"), dt=1e-3, **kw)
    sim.initial(T=lambda x, z: -z + 1e-2 * np.cos(np.pi * x) * np.cos(np.pi * z)
                + 1e-3 * np.sin(2 * np.pi * x) * np.cos(np.pi * z))
    sim.run(steps=40)
    return sim


IVP_PATHS = [
    {"ordering": "variable-major"},
    {"complexify": True},
    {"decouple": True},
    {"f_backend": "fused"},
    {"reduced": False},
    {"scheme": "SBDF3", "startup": "rk443", "ordering": "variable-major"},
]


@pytest.mark.parametrize("opts", IVP_PATHS, ids=[",".join(f"{k}={v}" for k, v in o.items())
                                                 for o in IVP_PATHS])
def test_ivp_paths_rb2d(opts):
    base = {k: v for k, v in opts.items() if k in ("scheme", "startup")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = _rb_sim(**base)
        alt = _rb_sim(**dict(opts))
    for name in ("T", "u"):
        assert _rel(alt.grid(name), ref.grid(name)) < RTOL, (name, _rel(alt.grid(name), ref.grid(name)))


def _os_plan(N=48):
    m = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="OS", Re=1e4, alpha=1.0)
    m.scalar("v")
    m.given(U=lambda y: 1 - y**2)
    m.eigenvalue("c", growth="imag")
    m.define("L2(f) = dy(dy(f)) - alpha**2*f")
    m.equations("OS: L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + c*L2(v) = 0")
    m.at("walls", "v = 0", "dy(v) = 0")
    return m


ORSZAG_C = 0.23752649 + 0.00373967j


def test_evp_dense_shift_invert_contour():
    m = _os_plan()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = m.eigenvalues(N=48)
        s = m.eigenvalues(N=48, method="shift-invert", target=0.24, n_modes=6)
        r = m.eigenvalues(N=48, region=cn.box(re=(0.2, 0.3), im=(-0.05, 0.02)))
    inside = d.values[(d.values.real > 0.2) & (d.values.real < 0.3)
                      & (d.values.imag > -0.05) & (d.values.imag < 0.02)]
    assert inside.size == np.asarray(r.values).size, (inside, r.values)
    for w in np.asarray(r.values):
        assert np.min(np.abs(inside - w)) < ETOL * abs(w)
    for w in s.values:
        assert np.min(np.abs(d.values - w)) < ETOL * abs(w)


def test_polar_dense_vs_shift_invert():
    from conatus.expert import eigenvalues
    drum = cn.Model(cn.Disk(radius=1.0), name="drum")
    drum.scalar("T")
    drum.eigenvalue("s", growth="real")
    drum.equations("s*T = lap(T)")
    drum.at("wall", T=0)
    plan = drum.discretize(N=(8, 32))
    g = plan.groups[1]
    d = eigenvalues(plan, g)
    s = eigenvalues(plan, g, method="shift-invert", target=-15.0, n_modes=3)
    for w in s.values:
        assert np.min(np.abs(d.values - w)) < ETOL * abs(w)
