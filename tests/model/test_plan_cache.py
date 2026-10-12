"""Critique P1: two ``simulate()`` of the same model reuse the assembled operators.

Reuse is declared in ``sim.explain()``, gives bit-for-bit the same run as a fresh build, and is
invalidated by every change that enters the operators: a parameter (attribute or per call), the
resolution, the dealiasing, an equation, a condition, given data (also when a callable's closure
changed). The time stepper (scheme, dt) is always new.
"""
import pickle

import numpy as np
import pytest

import conatus as cn
from conatus.discretize.plan import Plan
from conatus.model import plan_cache

N = (16, 12)


def rb(**kw):
    m = cn.Model(cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5)), name="RB", Ra=2500, Pr=1)
    m.velocity("u"), m.pressure("p"), m.scalar("T")
    m.equations("""
        dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)
        div(u) = 0
        dt(T) - lap(T) = -u·grad(T)
    """)
    m.at("bottom", u=0, T=+0.5)
    m.at("top", u=0, T=-0.5)
    return m


def run(m, steps=5, **kw):
    sim = m.simulate(N=kw.pop("N", N), scheme=kw.pop("scheme", "RK443"), dt=kw.pop("dt", 1e-3), **kw)
    sim.initial(T=lambda x, z: -z + 1e-3 * np.cos(np.pi * x) * np.cos(np.pi * z))
    sim.run(steps=steps)
    return sim, sim.grid("T").copy()


@pytest.fixture
def assemblies(monkeypatch):
    """Number of group assemblies (Plan._assemble_direct calls)."""
    count = {"n": 0}
    real = Plan._assemble_direct

    def counted(self, *a, **k):
        count["n"] += 1
        return real(self, *a, **k)
    monkeypatch.setattr(Plan, "_assemble_direct", counted)
    monkeypatch.delenv("CN_PLAN_CACHE", raising=False)
    return count


def test_second_build_reuses_and_matches_bit_for_bit(assemblies):
    m = rb()
    s1, T1 = run(m)
    n1 = assemblies["n"]
    assert n1 > 0 and "kept for reuse" in s1.explain()
    s2, T2 = run(m)
    assert assemblies["n"] == n1, "the second build assembled again"
    assert "operators reused" in s2.explain() and "operators reused" in s2.explain(full=True)
    np.testing.assert_array_equal(T1, T2)
    fresh = rb()                                   # another model: nothing to reuse
    _, T3 = run(fresh)
    np.testing.assert_array_equal(T1, T3)


def test_new_scheme_and_dt_reuse_operators_not_the_stepper(assemblies):
    m = rb()
    run(m)
    n1 = assemblies["n"]
    s2, T2 = run(m, scheme="SBDF2", dt=5e-4)
    assert assemblies["n"] == n1 and "operators reused" in s2.explain()
    _, T3 = run(rb(), scheme="SBDF2", dt=5e-4)
    np.testing.assert_array_equal(T2, T3)


@pytest.mark.parametrize("change", ["attribute", "per_call", "N", "dealias", "equation",
                                    "condition"])
def test_every_change_of_the_operators_invalidates(assemblies, change):
    m = rb()
    run(m)
    n1 = assemblies["n"]
    kw = {}
    if change == "attribute":
        m.Ra = 3000
    elif change == "per_call":
        kw["Ra"] = 3000
    elif change == "N":
        kw["N"] = (16, 14)
    elif change == "dealias":
        kw["dealias"] = 2
    elif change == "equation":
        m.eqs[2] = ("heat", "dt(T) - 2*lap(T) = -u·grad(T)")
    elif change == "condition":
        m.conditions[1] = type(m.conditions[1])("bottom", None, "T", 0.25, None)
    s2, T2 = run(m, **kw)
    assert assemblies["n"] > n1, f"{change}: operators reused after a change"
    assert "operators reused" not in s2.explain()
    if change in ("attribute", "per_call"):          # same answer as a fresh model at Ra = 3000
        f = rb()
        f.Ra = 3000
        np.testing.assert_array_equal(T2, run(f)[1])


def heat(q):
    m = cn.Model(cn.Box(z=cn.walls(0.0, 1.0)), name="heat", kappa=1.0)
    m.scalar("T")
    m.given("q", q)
    m.equations("dt(T) - kappa*lap(T) + q*T = 0")
    m.at("bottom", T=0)
    m.at("top", T=0)
    return m


def heat_run(m):
    sim = m.simulate(N=16, scheme="SBDF2", dt=1e-3)
    sim.initial(T=lambda z: np.sin(np.pi * z))
    sim.run(steps=4)
    return sim, sim.grid("T").copy()


def test_given_data_changes_invalidate(assemblies):
    """An array mutated in place, and a callable whose closure changed, are seen."""
    A = {"a": 1.0}
    m = heat(lambda z: A["a"] * (1 + z))
    s1, T1 = heat_run(m)
    n1 = assemblies["n"]
    s2, T2 = heat_run(m)
    assert assemblies["n"] == n1 and "operators reused" in s2.explain()
    np.testing.assert_array_equal(T1, T2)
    A["a"] = 3.0                                      # same function object, new data
    s3, T3 = heat_run(m)
    assert assemblies["n"] > n1 and "operators reused" not in s3.explain()
    assert not np.array_equal(T1, T3)
    np.testing.assert_array_equal(T3, heat_run(heat(lambda z: 3.0 * (1 + z)))[1])


def test_two_live_simulations_do_not_share_matrices(assemblies):
    m = rb()
    s1, _ = run(m, steps=0)
    s2, _ = run(m, steps=0)
    g = s1.plan.groups[0]
    a, b = s1.plan.matrices(g), s2.plan.matrices(g)
    assert a.L is not b.L and a.M is not b.M
    assert (a.L != b.L).nnz == 0


def test_cache_is_bounded_disabled_and_not_pickled(assemblies, monkeypatch):
    m = rb()
    for ra in (2000, 2100, 2200):
        run(m, steps=0, Ra=ra)
    assert len(m._plan_cache) == plan_cache.PLAN_CACHE_SIZE
    assert "_plan_cache" not in pickle.loads(pickle.dumps(m)).__dict__
    monkeypatch.setenv("CN_PLAN_CACHE", "0")
    s, _ = run(m, steps=0, Ra=2200)
    assert "reuse disabled: CN_PLAN_CACHE=0" in s.explain()
    plan_cache.clear(m)
    assert not m._plan_cache


def test_store_is_thread_safe_and_keeps_callables_alive(assemblies):
    """Review concurrent builds of one model never corrupt its LRU store, and a declared
    callable enters the key as the object itself (its id cannot be recycled by another function
    while the entry lives)."""
    import threading
    m = rb()
    s, _ = run(m, steps=0)
    errors = []

    def worker(t):
        try:
            for i in range(40):
                plan_cache.remember(s.compiled, s.plan, ("test", t, i))
        except Exception as e:  # noqa: BLE001 -- collected and asserted below
            errors.append(e)
    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(m._plan_cache) == plan_cache.PLAN_CACHE_SIZE

    def f(z):
        return 1 + z
    assert plan_cache._digest(f, "given q") == ("f", f)
