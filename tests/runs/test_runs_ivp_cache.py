"""Registry and solvers: solver.run(record=...), bit-identical numbers, reproduce(), declared cache."""
import json
import os
import warnings

import numpy as np
import pytest

import conatus.runs as R
from conatus.errors import OptionError, UnsupportedError

N = 40


def _plain(make, n=N):
    s = make()
    s.run(n)
    return s.X.copy()


def test_record_returns_the_run_and_changes_no_bit(ks):
    *_, make = ks
    ref = _plain(make)
    s = make()
    run = s.run(N, record=dict(name="ks", tags=["t"], series={"E": lambda s: float(np.mean(s.grid("u") ** 2))},
                               series_every=10))
    assert np.array_equal(s.X, ref) and isinstance(run, R.Run)
    got = R.get(run.id)
    assert got.state == "completed" and got.rec["inputs"] == {"kind": "ivp", "nsteps": N, "segments": 1}
    assert got.rec["numerics"]["time_stepping"]["scheme"] == "RK443" and got.rec["cache_key"]
    assert "IVPSolver(scheme=RK443" in got.rec["explain"]["object"]
    assert list(got.series("E")["step"]) == [10, 20, 30, 40]
    assert (got.path / "state_start.npz").exists() and (got.path / "state_end.npz").exists()
    assert make().run(3) is None                               # no record: no run, as before


def test_unwritable_registry_leaves_the_run_bit_identical(ks, tmp_path):
    """The registry root cannot be created: the computation is unchanged to the bit, the loss is
    reported (RegistryWriteWarning, run.errors), nothing is raised."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    *_, make = ks
    ref = _plain(make)
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        s = make()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run = s.run(N, record=dict(root=ro / "registry", series={"m": lambda s: 1.0}, series_every=5))
        assert np.array_equal(s.X, ref)
        assert run.errors and any(issubclass(x.category, R.RegistryWriteWarning) for x in w)
        assert not (ro / "registry").exists()
    finally:
        ro.chmod(0o700)


def test_record_into_an_open_run_with_segments(ks):
    *_, make = ks
    s = make()
    with R.track("two_legs", params={"note": "demo"}) as run:
        s.run(10, record=run)
        s.run(10, record=run)
        run.result(t_end=s.t)
    got = R.get(run.id)
    assert [g["nsteps"] for g in got.rec["segments"]] == [10, 10]
    assert (got.path / "state_s1_end.npz").exists() and got.results["t_end"] == pytest.approx(1.0)
    with pytest.raises(UnsupportedError, match="single-segment"):
        got.reproduce(make())


def test_reproduce_is_bit_identical_and_checks_numerics_and_code(ks):
    *_, make = ks
    run = make().run(N, record="orig")
    rep = run.reproduce(make())
    assert rep.identical and rep.max_abs_diff == 0.0 and rep.code_identical
    assert rep.new.parent == run.id and rep.new.rec["relation"] == "reproduce"
    with pytest.raises(OptionError, match="numerics differ"):
        run.reproduce(make(dt_=0.04))                          # never adapted silently
    with pytest.raises(OptionError, match="pass the solver"):
        run.reproduce()
    rec = json.loads((run.path / "run.json").read_text())   # pretend the code changed since
    rec["code_fingerprint"]["python_tree_sha256"] = "0" * 64
    (run.path / "run.json").write_text(json.dumps(rec))
    changed = R.get(run.id)
    with pytest.warns(R.CodeChangedWarning, match="python_tree_sha256"):
        rep2 = changed.reproduce(make())
    assert rep2.identical and not rep2.code_identical and "CHANGED" in rep2.explain()
    with pytest.raises(OptionError, match="differs from the recorded"):
        changed.reproduce(make(), require_same_code=True)


def test_cache_is_off_by_default(ks):
    *_, make = ks
    a = make().run(N, record=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", R.CacheReuseWarning)   # no reuse without cache=True
        b = make().run(N, record=True)
    assert a.rec["cache_key"] == R.get(b.id).rec["cache_key"]   # identical computation...
    assert b.rec["reused_from"] is None and b.cache_report is None   # ...computed again
    assert R.get(b.id).rec["segments"][0]["nsteps"] == N


def test_cache_hit_reports_what_was_reused_and_misses_on_any_change(ks):
    *_, make = ks
    ref = _plain(make)
    src = make().run(N, record=True)
    s = make()
    with pytest.warns(R.CacheReuseWarning, match=f"read from run {src.id}"):
        hit = s.run(N, record=True, cache=True)
    assert np.array_equal(s.X, ref) and s.iteration == N and s.t == pytest.approx(N * 0.05)
    rep = hit.cache_report
    assert rep["hit"] and rep["reused_from"] == src.id and "nothing" in rep["computed"]
    assert R.get(hit.id).rec["reused_from"] == src.id and "CACHE" in R.get(hit.id).explain()
    assert "(cache reuse)" in R.lineage(src).explain()
    for other in (lambda: make().run(N + 1, record=True, cache=True),        # other input
                  lambda: make(dt_=0.04).run(N, record=True, cache=True)):  # other numerics
        with warnings.catch_warnings():
            warnings.simplefilter("error", R.CacheReuseWarning)
            miss = other()
        assert not miss.cache_report["hit"] and "no completed" in miss.cache_report["miss_reason"]


def test_retracted_runs_are_never_reused_and_cache_needs_record(ks):
    *_, make = ks
    src = make().run(N, record=True)
    src.retract("bad initial condition")
    with warnings.catch_warnings():
        warnings.simplefilter("error", R.CacheReuseWarning)
        again = make().run(N, record=True, cache=True)
    assert not again.cache_report["hit"]
    with pytest.raises(OptionError, match="needs record"):
        make().run(N, cache=True)
    with pytest.raises(OptionError, match="callback"):
        make().run(N, callback=lambda s: None, every=1, record=True, cache=True)
    with pytest.raises(OptionError, match="series_every"):
        make().run(N, record=dict(series={"a": lambda s: 1.0}))
    with pytest.raises(OptionError, match="unknown option"):
        make().run(N, record=dict(nmae="x"))


def test_parameter_values_enter_the_record_and_the_cache_key():
    """IR Parameter values are not in the structural hash nor (before) in the provenance
    numerics: two runs at different kappa from the same state had identical keys, so the cache
    would have returned the kappa = 0.2 result for kappa = 0.4. Now both are recorded and keyed."""
    from conatus.expert import (IVP, CartesianCoordinates, Domain, Field, IVPSolver, Parameter,
                                RealFourier, discretize, dt, lap)
    cs = CartesianCoordinates("x")
    D = Domain(cs, [RealFourier(cs["x"], 16, (0, 2 * np.pi), dealias=1.5)])

    def heat(kappa):
        u = Field("u", D)
        pb = IVP([u], name="heat")
        pb.add_equation((dt(u) - Parameter("kappa", kappa) * lap(u), 0))
        s = IVPSolver(discretize(pb), "RK222", 1e-3)
        s.set_state({u: lambda X: np.sin(X)})
        return s
    a = heat(0.2).run(20, record=True)
    assert a.params == {"kappa": 0.2}
    assert a.rec["numerics"]["problem"]["parameters"] == {"kappa": 0.2}
    b = heat(0.4)
    with warnings.catch_warnings():
        warnings.simplefilter("error", R.CacheReuseWarning)
        rb = b.run(20, record=True, cache=True)                # must not reuse kappa = 0.2
    assert not rb.cache_report["hit"] and R.get(rb.id).rec["cache_key"] != R.get(a.id).rec["cache_key"]
    assert R.find(kappa=0.4)[0].id == rb.id
    with pytest.raises(OptionError, match="numerics differ"):
        a.reproduce(heat(0.4))                                 # other kappa: refused


CALLS = {"n": 0}


def least_stable(Re: float, N: int) -> dict:
    """A decorated computation (module level: importable by reproduce())."""
    CALLS["n"] += 1
    x = np.linspace(0, 1, N)
    return {"sigma": -1.0 / Re, "profile": np.sin(np.pi * x) / Re, "note": "toy"}


def test_decorator_records_calls_and_caches_only_when_asked():
    plain = R.track("toy")(least_stable)
    cached = R.track("toy", cache=True, compression="gzip")(least_stable)
    CALLS["n"] = 0
    v1 = plain(100.0, 16)
    plain(100.0, 16)                                           # cache off: computed again
    assert CALLS["n"] == 2
    with pytest.warns(R.CacheReuseWarning, match="not called"):
        v3 = cached(100.0, N=16)
    assert CALLS["n"] == 2 and v3.keys() == v1.keys()
    assert v3["sigma"] == v1["sigma"] and np.array_equal(v3["profile"], v1["profile"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", R.CacheReuseWarning)
        cached(200.0, 16)                                      # other argument: computed
    assert CALLS["n"] == 3
    runs = R.find(name="toy", Re=100.0)
    assert len(runs) == 3 and sum(r.rec["reused_from"] is not None for r in runs) == 1


def test_decorator_cache_refuses_inexact_arguments_and_unstored_returns():
    @R.track("objarg", cache=True)
    def f(obj, k: int):
        return k
    with pytest.warns(R.CacheReuseWarning, match="no exact content hash"):
        f(object(), 1)

    @R.track("unstored", cache=True)
    def g(k: int):
        return {1, 2, k}                                        # a set: not storable
    g(1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", R.CacheReuseWarning)
        assert g(1) == {1, 2}                                   # computed again, never a hit


def test_reproduce_a_decorated_call():
    f = R.track("toy_rep")(least_stable)
    f(50.0, 8)
    run = R.find(name="toy_rep")[0]
    rep = run.reproduce()
    assert rep.identical and rep.new.parent == run.id and rep.code_identical

    @R.track("with_block")
    def local(k: int):
        return k
    with R.track("block") as b:
        pass
    with pytest.raises(UnsupportedError, match="re-run the script"):
        R.get(b.id).reproduce()


@pytest.mark.parametrize("bad", [2.5, 0, True])
def test_series_every_and_capacity_are_positive_integers(ks, bad):
    """Quality pass 4: ``series_every=2.5`` was truncated for the callback period (2) but kept in
    the modulo (``iteration % 2.5``): the series was logged every 10 steps, without a word; 0
    divided by zero at the first step. ``capacity=0`` made the writer queue unbounded."""
    *_, make = ks
    with pytest.raises(OptionError, match="series_every"):
        make().run(4, record=dict(series={"a": lambda s: 1.0}, series_every=bad))
    with pytest.raises(OptionError, match="capacity"):
        make().run(4, record=dict(capacity=bad))
