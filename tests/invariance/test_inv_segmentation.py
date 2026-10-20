"""Run segmentation: N steps in one call = the same N steps cut into pieces, bit for bit.

The development test suite covers every scheme x route (segments, HDF5, npz, series,
registry) on two Cartesian 1-D problems. Here the same relation on what that matrix does not
reach: curvilinear geometries (disk, annulus, pipe, ball, shell), the mapped box, a 2-D
Rayleigh-Benard cell, tracer particles and outputs riding along the run, ``run(until=)``
against ``run(steps=)``, and observers (callbacks, analysis files) that must not change the
state they observe.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import conatus as cn

CURVED = {"disk": (cn.Disk(1.0), (8, 12)),
          "annulus": (cn.Annulus(0.5, 1.5), (8, 12)),
          "pipe": (cn.Pipe(1.0, 2 * np.pi), (8, 12, 4)),
          "ball": (cn.Ball(1.0), (8, 6, 8)),
          "shell": (cn.Shell(0.5, 1.5), (8, 6, 8))}


def _curved(kind: str, scheme: str):
    geo, N = CURVED[kind]
    m = cn.Model(geo, name=kind, nu=0.05)
    m.velocity("u")
    m.pressure("p")
    m.equations("dt(u) + grad(p) - nu*lap(u) = -u@grad(u)", "div(u) = 0")
    m.at("walls" if "walls" in geo.boundaries() else "wall", u=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.simulate(N=N, scheme=scheme, dt=1e-2)


def _seed(sim, kind):
    """A smooth non-trivial state: a few steps of a forced start would do; here the projection of
    a solid-body-like rotation plus a perturbation through initial()."""
    if kind in ("ball", "shell"):
        f = lambda phi, theta, r: np.array([r * np.sin(theta) * (1 + 0.2 * np.cos(phi)),  # noqa: E731
                                            0.1 * np.sin(phi) * np.sin(theta) * r * (1.5 - r),
                                            0 * r * theta * phi])
    elif kind == "pipe":
        f = lambda phi, r, z: np.array([r * (1 + 0.2 * np.cos(phi + z)),  # noqa: E731
                                        0.1 * r * np.sin(phi) + 0 * z, 0.3 + 0 * r * phi * z])
    else:
        f = lambda phi, r: np.array([r * (1 + 0.2 * np.cos(phi)), 0.1 * r * np.sin(phi)])  # noqa: E731
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.initial(u=f, check=False)
    return sim


@pytest.mark.parametrize("scheme", ["RK222", "SBDF2", "CNAB2"])
@pytest.mark.parametrize("kind", list(CURVED))
def test_curved_segments_and_restart(kind, scheme, tmp_path):
    """One run of 12 steps; 5 + 7 steps; 5 steps, save (.h5), a fresh simulation restarted, 7."""
    a = _seed(_curved(kind, scheme), kind)
    a.run(steps=12)
    b = _seed(_curved(kind, scheme), kind)
    b.run(steps=5)
    b.run(steps=7)
    assert np.array_equal(a.solver.X, b.solver.X), "segments differ"
    c = _seed(_curved(kind, scheme), kind)
    c.run(steps=5)
    p = c.save(tmp_path / "ck.h5")
    d = _curved(kind, scheme)
    d.restart(p).run(steps=7)
    assert (d.iteration, d.t) == (a.iteration, a.t)
    assert np.array_equal(a.solver.X, d.solver.X), (
        f"restart differs: max |dX| = {np.abs(a.solver.X - d.solver.X).max():.3g}")


def _mapped(scheme: str):
    m = cn.Model(cn.MappedBox(x=cn.periodic(2 * np.pi),
                              z=cn.between(bottom=lambda x: 0.1 * np.cos(x), top=1.0)), name="mapped")
    m.scalar("u")
    m.equations("dt(u) - lap(u) = -u*u + exp(-t)")
    m.at("bottom", u="cos(t)")
    m.at("top", u=0)
    sim = m.simulate((32, 12), scheme=scheme, dt=1e-2)
    sim.initial(u=lambda x, z: (1 - z) * (1 + 0.1 * np.sin(x)))
    return sim


@pytest.mark.parametrize("scheme", ["SBDF3", "RK443"])
def test_mapped_box_segments(scheme):
    a = _mapped(scheme).run(steps=10)
    b = _mapped(scheme)
    for n in (1, 4, 5):
        b.run(steps=n)
    assert np.array_equal(a.grid("u"), b.grid("u"))


def _rb(scheme="RK222"):
    m = cn.Model(cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5)), name="RB", Ra=2500.0, Pr=1.0)
    m.velocity("u"), m.pressure("p"), m.scalar("T")
    m.equations("""
        momentum:   dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)
        continuity: div(u) = 0
        heat:       dt(T) - lap(T) = -u·grad(T)
    """)
    m.at("bottom", u=0, T=0.5)
    m.at("top", u=0, T=-0.5)
    sim = m.simulate(N=(16, 12), scheme=scheme, dt=1e-3)
    sim.initial(T=lambda x, z: -z + 1e-2 * np.cos(np.pi * x) * np.cos(np.pi * z))
    return sim


@pytest.mark.parametrize("scheme", ["RK443", "SBDF3", "MCNAB2", "CNLF2"])
def test_until_equals_steps_and_observers_do_not_perturb(scheme, tmp_path):
    """run(until=) in three pieces = run(steps=); a callback, an analysis file and a
    checkpoint series riding along leave the state bit-identical."""
    a = _rb(scheme).run(steps=30)
    b = _rb(scheme)
    b.run(until=0.007)
    b.run(until=0.019)
    b.run(until=0.030)
    assert b.iteration == 30 and np.array_equal(a.solver.X, b.solver.X)
    c = _rb(scheme)
    c.output(tmp_path / "a.h5", ["T", "u"], every=4)
    c.checkpoints(tmp_path / "ck", every=7)
    seen = []
    c.run(steps=30, callback=lambda s: seen.append(float(np.max(s.grid("T")))), every=3)
    assert len(seen) == 10 and np.array_equal(a.solver.X, c.solver.X)


@pytest.mark.parametrize("scheme", ["RK443", "SBDF2", "CNAB2", "RK222"])
def test_particles_split_and_with_file(scheme, tmp_path):
    """Tracers in a decaying flow: one run, one-step runs, and one run writing the particle file
    every 3 steps give the same positions (the file is an observer)."""
    def make():
        m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), y=cn.periodic(2 * np.pi)), name="TG", nu=0.1)
        m.velocity("u"), m.pressure("p")
        m.equations("dt(u) + grad(p) - nu*lap(u) = -u@grad(u)", "div(u) = 0")
        s = m.simulate(N=(16, 16), scheme=scheme, dt=0.02)
        s.initial(u=lambda x, y: np.array([np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)]))
        return s
    pos = np.array([[0.3, 0.4], [1.0, 2.0], [4.0, 0.5]])
    a = make()
    pa = a.particles(pos)
    a.run(steps=12)
    b = make()
    pb = b.particles(pos)
    for _ in range(12):
        b.run(steps=1)
    c = make()
    pc = c.particles(pos, path=tmp_path / "p.h5", every=3)
    c.run(steps=12)
    xa, xb, xc = (p.positions(unwrapped=True) for p in (pa, pb, pc))
    assert np.array_equal(xa, xb), f"split: max |dx| = {np.abs(xa - xb).max():.3g}"
    assert np.array_equal(xa, xc), f"file: max |dx| = {np.abs(xa - xc).max():.3g}"
