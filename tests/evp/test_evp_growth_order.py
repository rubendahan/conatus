"""EVPResult ordering by the declared growth convention and the
spurious-mode sentence of an unfiltered result: the least stable mode first (Im c
for a phase speed), said by explain(); "no spurious-mode filter" once, no warning."""
import warnings

import numpy as np

import conatus as cn


def _os():
    m = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="Orr-Sommerfeld", Re=1e4, alpha=1.0)
    m.scalar("v")
    m.given(U=lambda y: 1 - y**2)
    m.eigenvalue("c", growth="imag")
    m.define("L2(f) = dy(dy(f)) - alpha**2*f")
    m.equations("L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + c*L2(v) = 0")
    m.at("walls", "v = 0", "dy(v) = 0")
    return m


def test_order_follows_the_growth_convention_of_c():
    res = _os().eigenvalues(N=48, drift=64, drift_tol=1e-6)
    i = int(res.order()[0])
    assert i == int(np.argmax(res.values.imag))
    assert abs(res.values[i] - (0.23752649 + 0.00373967j)) < 1e-6      # Orszag 1971
    ex = res.explain(3)
    assert "sorted by growth = Im c, least stable first" in ex
    lines = ex.splitlines()
    first = lines[next(j for j, t in enumerate(lines) if "least stable first" in t) + 1]
    assert first.lstrip().startswith("c =  2.3752")
    assert list(res.order("real")) == list(np.argsort(-res.values.real, kind="stable"))
    assert "no spurious-mode filter" not in ex


def test_unfiltered_result_says_so_once_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = _os().eigenvalues(N=48)
    assert res.explain().count("no spurious-mode filter: tau modes may appear") == 1
    assert "no spurious-mode filter" in repr(res) and "\n" not in repr(res)


def test_periodic_problem_has_no_filter_sentence():
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="heat")
    m.scalar("u")
    m.equations("dt(u) - dx(dx(u)) = 0")
    res = m.linearize(about={"u": 0}).eigenvalues(8, k=1)
    assert "spurious" not in res.explain() and "sorted by growth = Re s" in res.explain()
