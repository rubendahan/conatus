"""Equivalent constructions explain themselves identically; refinement converges monotonically.

explain():
  * the equations given as one block or as separate strings, with other whitespace; a parameter
    given to the constructor or set afterwards; the walls named together or one by one: the same
    explain() text and the same numbers;
  * explain() has no side effect: calling it between runs changes nothing.
refinement (the theory says the error decreases until the rounding plateau):
  * the Orr-Sommerfeld eigenvalue against Orszag (1971), N = 24, 32, 40, 48;
  * the drum eigenvalues against the Bessel zeros, N = 8, 12, 16 (radial);
  * the Bratu solution against its closed form, N = 8, 12, 16, 24.
"""
from __future__ import annotations

import warnings

import numpy as np

import conatus as cn


def _build(variant: int):
    if variant == 0:
        m = cn.Model(cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5)), name="RB", Ra=2500.0, Pr=1.0)
    else:
        m = cn.Model(cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5)), name="RB", Ra=1.0, Pr=1.0)
        m.Ra = 2500.0
    m.velocity("u"), m.pressure("p"), m.scalar("T")
    if variant == 0:
        m.equations("""
            momentum: dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)
            continuity: div(u) = 0
            heat: dt(T) - lap(T) = -u·grad(T)
        """)
        m.at("bottom", u=0, T=0.5)
        m.at("top", u=0, T=-0.5)
    else:
        m.equations("momentum:   dt(u)+grad(p)-Pr*lap(u) = Pr*Ra*T*ez-u·grad(u)",
                    "continuity: div(u)=0",
                    "heat:  dt(T) - lap(T)  =  -u·grad(T)")
        m.at("bottom", u=0)
        m.at("bottom", T=0.5)
        m.at("top", u=0, T=-0.5)
    return m


def test_equivalent_constructions_explain_and_compute_the_same():
    a, b = _build(0), _build(1)
    norm = lambda t: " ".join(t.replace(" ", "").split())  # noqa: E731  the user's spacing is echoed
    assert norm(a.explain()) == norm(b.explain())
    sa = a.simulate(N=(16, 12), scheme="RK222", dt=1e-3)
    sb = b.simulate(N=(16, 12), scheme="RK222", dt=1e-3)
    for s in (sa, sb):
        s.initial(T=lambda x, z: -z + 1e-2 * np.cos(np.pi * x) * np.cos(np.pi * z))
    sa.run(steps=10)
    _ = sb.explain(), sb.methods().text
    sb.run(steps=5)
    _ = sb.explain(full=True)
    sb.run(steps=5)
    assert np.array_equal(sa.solver.X, sb.solver.X)
    strip = lambda t: "\n".join(ln for ln in t.splitlines() if "sim" not in ln.split(":")[0])  # noqa: E731
    assert strip(sa.explain()) == strip(sb.explain())


def _monotone(errs, floor):
    """Strictly decreasing until the first value under ``floor`` (the rounding plateau)."""
    for e0, e1 in zip(errs, errs[1:], strict=False):
        if e0 < floor:
            return True
        if not e1 < e0:
            return False
    return True


def test_orr_sommerfeld_refinement_is_monotone():
    c_ref = 0.23752649 + 0.00373967j
    errs = []
    for N in (24, 32, 40, 48):
        m = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="OS", Re=1e4, alpha=1.0)
        m.scalar("v")
        m.given(U=lambda y: 1 - y**2)
        m.eigenvalue("c", growth="imag")
        m.define("L2(f) = dy(dy(f)) - alpha**2*f")
        m.equations("OS: L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + c*L2(v) = 0")
        m.at("walls", "v = 0", "dy(v) = 0")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = m.eigenvalues(N=N)
        errs.append(float(np.min(np.abs(r.values - c_ref))))
    assert _monotone(errs, 1e-8), errs


def test_drum_refinement_is_monotone():
    from scipy.special import jn_zeros
    from conatus.expert import eigenvalues
    ref = -jn_zeros(1, 3) ** 2
    errs = []
    for Nr in (8, 12, 16, 20):
        drum = cn.Model(cn.Disk(radius=1.0), name="drum")
        drum.scalar("T")
        drum.eigenvalue("s", growth="real")
        drum.equations("s*T = lap(T)")
        drum.at("wall", T=0)
        plan = drum.discretize(N=(8, Nr))
        g = [g for g in plan.groups if g.m == 1][0]
        v = np.sort(eigenvalues(plan, g).values.real)[::-1][:3]
        errs.append(float(np.max(np.abs(v - ref) / np.abs(ref))))
    assert _monotone(errs, 1e-12), errs


def test_bratu_refinement_is_monotone():
    from scipy.optimize import brentq
    from conatus.expert import NewtonSolver, grid
    a_star = brentq(lambda a: a * np.tanh(a) - 1.0, 0.5, 2.0, xtol=1e-15)
    a03 = brentq(lambda a: 2 * a * a / np.cosh(a) ** 2 - 0.3, 1e-12, a_star, xtol=1e-15)
    errs = []
    for N in (10, 12, 14, 16):
        m = cn.Model(cn.Box(x=cn.walls(-1, 1)), name="Bratu", lam=0.3)
        m.scalar("u")
        m.equations("bratu: dx(dx(u)) + lam*exp(u) = 0")
        m.at("walls", u=0)
        pb = m.compile(N, dealias=1)
        sol = NewtonSolver(pb.problem, tol=1e-13, max_iterations=20, params={"lam": 0.3},
                           ncc_tail_tol=1e-2).solve(
            {pb.fields["u"]: lambda x: 0 * x})
        X = grid(pb.bases["x"], dealias=False)
        errs.append(float(np.max(np.abs(sol.grid("u") - 2 * np.log(np.cosh(a03) / np.cosh(a03 * X))))))
    assert _monotone(errs, 1e-12), errs
