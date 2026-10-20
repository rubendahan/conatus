"""Release 0.1.1, lot A "identity": the external review's findings on what makes two computations
the same (1, 5, 8, 22-24, 45-52, 56-58).

One canonical identity of a computation (``conatus.io._identity.computation_identity``: model,
equations, parameters, every given field by name, domain, shape, dtype and content, user
pointwise functions, numerical choices) is consumed by the cache key, the checkpoint restart and
overwrite guards, the provenance record, the registry and the HDF5 append. Each test reproduces
the reviewer's script (or the finding) and failed on v0.1.0.
"""
import json
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

import conatus as cn
import conatus.runs as R
from conatus.errors import OptionError
from conatus.runs._util import from_jsonable, jsonable
from conatus.runs.integrations import _same
from conatus.runs.keys import ivp_parts


def _forced(amplitude, n=8):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="forced")
    m.scalar("u")
    m.given(f=lambda x: amplitude * np.ones_like(x))
    m.equations("dt(u) = f")
    return m.simulate(n, scheme="RK443", dt=.01).initial(u=0)


def _burgers(nu, n=32):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="burgers", nu=nu)
    m.scalar("u")
    m.equations("dt(u)-nu*dx(dx(u)) = -u*dx(u)")
    return m.simulate(n, scheme="RK443", dt=.002).initial(u=lambda x: np.sin(x) + .3 * np.cos(2 * x))


# ------------------------------------------------------------------ finding 1: given-field data
def test_two_forcings_have_two_cache_keys_and_no_hit(tmp_path):
    a, b = _forced(1), _forced(2)
    assert ivp_parts(a.solver, 5) != ivp_parts(b.solver, 5)
    a.solver.run(5, record={"root": tmp_path})
    with warnings.catch_warnings():
        warnings.simplefilter("error", R.CacheReuseWarning)      # a hit would raise here
        r = b.solver.run(5, record={"root": tmp_path}, cache=True)
    assert r.cache_report["hit"] is False
    assert b.grid("u").mean() == pytest.approx(0.1, rel=1e-12)


def test_same_forcing_rebuilt_is_a_cache_hit(tmp_path):
    """The identity is by content: the same forcing written by another closure hits."""
    a, b = _forced(1), _forced(1)
    a.solver.run(5, record={"root": tmp_path})
    with pytest.warns(R.CacheReuseWarning, match="cache hit"):
        r = b.solver.run(5, record={"root": tmp_path}, cache=True)
    assert r.cache_report["hit"] and np.array_equal(a.solver.X, b.solver.X)


@pytest.mark.parametrize("ext", [".npz", ".h5"])
def test_restart_with_another_forcing_is_refused_unless_allowed(tmp_path, ext):
    a = _forced(1)
    a.run(steps=3)
    p = a.save(tmp_path / f"chk{ext}")
    b = _forced(2)
    with pytest.raises(OptionError, match=r"given field f.*differ"):
        b.restart(p)
    assert b.solver.iteration == 0                                # nothing was loaded
    b.restart(p, allow=("f",))                                    # the deliberate change
    assert "given field f" in b.solver.explain(full=True)
    c = _forced(1)
    c.restart(p)                                                  # same forcing: accepted
    assert c.solver.iteration == 3


def test_restart_is_bit_for_bit_with_a_forcing(tmp_path):
    """Invariance: 6 steps = 3 steps, checkpoint, restart, 3 steps (segmentation of a run)."""
    ref = _forced(1)
    ref.run(steps=6)
    a = _forced(1)
    a.run(steps=3)
    p = a.save(tmp_path / "half.h5")
    b = _forced(1)
    b.restart(p)
    b.run(steps=3)
    assert np.array_equal(ref.solver.X, b.solver.X) and b.solver.t == ref.solver.t


def test_identity_survives_serialisation():
    """Invariance: the identity read back from JSON (checkpoint meta) compares equal."""
    from conatus.io._identity import checkpoint_identity, compare_identity
    s = _forced(1)
    ident = checkpoint_identity(s.solver)
    meta = {"identity": json.loads(json.dumps(ident))}
    assert compare_identity(meta, s.solver, (), "test") == []


def test_unhashable_pointwise_function_refuses_the_cache(tmp_path):
    """A user pointwise function is a dependency of the computation: its code and the values its
    closure reads enter the identity; a closure over an object without a content hash refuses the
    cache (loudly), never a hit."""
    class Box:
        k = 1.0
    box = Box()
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="pw")
    m.scalar("u")
    m.function("g_v011", lambda u: box.k * u)
    m.equations("dt(u) = g_v011(u)")
    s = m.simulate(8, scheme="RK443", dt=.01).initial(u=lambda x: 1 + 0 * x)
    s.solver.run(2, record={"root": tmp_path})
    with pytest.warns(R.CacheReuseWarning, match="cache not used.*g_v011"):
        r = s.solver.run(2, record={"root": tmp_path}, cache=True)
    assert r.cache_report["hit"] is False and r.rec["cache_key"] is None


# ------------------------------------------------------------------ finding 5: HDF5 append
def test_append_refuses_another_computation(tmp_path):
    p = Path(tmp_path) / "analysis.h5"
    a = _burgers(.05)
    a.output(p, tasks={"v": "u"}, every=1)
    a.run(steps=2)
    b = _burgers(.4)
    b.run(steps=2)
    with pytest.raises(OptionError, match="parameter nu"):
        b.output(p, tasks={"v": "dx(u)"}, every=1, mode="a")
    c = _burgers(.05)
    c.run(steps=2)
    with pytest.raises(OptionError, match="task v"):
        c.output(p, tasks={"v": "dx(u)"}, every=1, mode="a")
    d = cn.io.read(p)
    assert d.parameters == {"nu": 0.05} and d.tasks["v"].shape[0] == 3


def test_append_of_the_same_computation_equals_one_file(tmp_path):
    """Invariance: a record written in two segments (append) holds the rows of one segment."""
    one, two = Path(tmp_path) / "one.h5", Path(tmp_path) / "two.h5"
    a = _burgers(.05)
    a.output(one, tasks={"v": "u"}, every=1)
    a.run(steps=4)
    b = _burgers(.05)
    out = b.output(two, tasks={"v": "u"}, every=1)
    b.run(steps=2)
    b.outputs.remove(out)
    b.output(two, tasks={"v": "u"}, every=1, mode="a")
    b.run(steps=2)
    x, y = cn.io.read(one), cn.io.read(two)
    assert np.array_equal(x.tasks["v"], y.tasks["v"])


# ------------------------------------------------------------------ findings 8, 56, 57: checkpoints
def test_failed_overwrite_guard_holds_on_retry(tmp_path):
    root = Path(tmp_path)
    a = _burgers(.05)
    a.checkpoints(root, every=1, keep=2)
    a.run(steps=1)
    b = _burgers(.4)
    b.run(steps=1)
    writer = cn.io.CheckpointSeries(b.solver, root, every=1)
    for _ in range(2):
        with pytest.raises(OptionError, match="already holds"):
            writer.write()
    assert cn.io.read(root / "checkpoint_000000001.h5").parameters == {"nu": 0.05}


def test_latest_skips_files_that_are_not_checkpoints(tmp_path):
    import h5py
    a = _burgers(.05)
    a.run(steps=2)
    a.save(tmp_path / "checkpoint_000000002.h5")
    with h5py.File(tmp_path / "checkpoint_999.h5", "w") as f:       # not a checkpoint
        f.attrs["iteration"] = 999
    np.savez(tmp_path / "checkpoint_998.npz", meta=json.dumps({"iteration": 998}))
    assert cn.io.latest(tmp_path).name == "checkpoint_000000002.h5"


def test_adoption_validates_every_file_before_deleting(tmp_path):
    """A checkpoint of another run in the directory of a chain is refused at adoption, before
    any keep-deletion (it used to be adopted, then deleted by keep)."""
    d = Path(tmp_path)
    a = _burgers(.05)
    a.checkpoints(d, every=1, keep=2)
    a.run(steps=2)
    other = _burgers(.4)
    other.save(d / "checkpoint_other.h5")                        # iteration 0, another nu
    b = _burgers(.05)
    b.restart(cn.io.latest(d))
    b.checkpoints(d, every=1, keep=2)
    with pytest.raises(OptionError, match="checkpoint_other.h5"):
        b.run(steps=1)
    assert (d / "checkpoint_other.h5").exists()
    assert sorted(p.name for p in d.glob("checkpoint_0*.h5")) == [
        "checkpoint_000000001.h5", "checkpoint_000000002.h5"]


# ------------------------------------------------------------------ finding 22: provenance
def test_provenance_records_every_numerical_choice():
    from conatus import provenance
    s = _forced(1)
    ts = provenance.numerics(s.solver)["time_stepping"]
    for k in ("time_parameter", "bc_time", "f_backend", "pivot", "complexify", "decouple"):
        assert k in ts, k
    assert ts["f_backend"] == "evaluator"
    data = provenance.numerics(s.solver)["data"]
    assert set(data) == {"f"} and data["f"]["shape"] == [8] and len(data["f"]["sha256"]) == 64


def test_f_backend_enters_the_cache_key(ks):
    from conatus.expert import IVPSolver
    plan, u, u0, _make = ks
    a = IVPSolver(plan, "RK443", .05)
    b = IVPSolver(plan, "RK443", .05, f_backend="fused")
    for s in (a, b):
        s.set_state({u: u0})
    assert ivp_parts(a, 3)["numerics"] != ivp_parts(b, 3)["numerics"]


# ------------------------------------------------------------------ findings 23, 24: reproduce
def test_bitwise_means_bits():
    assert _same(np.array([0.0]), np.array([-0.0]))[0] is False
    assert _same(np.array([1.0, np.nan]), np.array([1.0, np.nan]))[0] is True
    assert _same(0.0, -0.0)[0] is False
    assert _same({"a": np.float32(1)}, {"a": np.float64(1)})[0] is False


def test_reproduce_compares_history_and_clock(ks):
    *_, make = ks
    s = make(scheme="SBDF2")
    run = s.run(6, record=True)
    end = run.path / "state_end.npz"
    with np.load(end) as z:
        meta, arrays = str(z["meta"]), {k: z[k] for k in z.files if k != "meta"}
    arrays["MX0"] = arrays["MX0"] * (1 + 1e-15) + 1e-300           # history only: X unchanged
    np.savez(end, meta=meta, **arrays)
    rep = R.get(run.id).reproduce(make(scheme="SBDF2"), record=False)
    assert not rep.identical and "MX0" in rep.explain()


# ------------------------------------------------------------------ findings 45-48, 58: registry
def test_series_are_streamed_not_kept_until_close(monkeypatch):
    from conatus.runs import run as runmod
    monkeypatch.setattr(runmod, "SERIES_FLUSH_ROWS", 10)
    with R.track("stream") as run:
        for i in range(55):
            run.log(step=i, e=float(i), label="a" if i % 2 else "b")
            assert len(run._rows) <= 10
    got = R.get(run.id)
    e = got.series("e")
    assert list(e["step"]) == list(range(55)) and list(e["value"]) == [float(i) for i in range(55)]
    assert got.rec["series"]["e"] == {"n": 55, "last": 54.0}
    assert len(got.series("label")["value"]) == 55


def test_close_timeout_is_one_deadline(monkeypatch, tmp_path):
    from conatus.runs import run as runmod
    release = threading.Event()
    real = runmod.shutil.copy2
    monkeypatch.setattr(runmod.shutil, "copy2", lambda *a, **k: (release.wait(10), real(*a, **k)))
    f = tmp_path / "x.txt"
    f.write_text("x")
    run = R.track("slow", capacity=1).__enter__()
    run.save_file(f, name="a.txt")
    time.sleep(0.2)                                    # the writer is now blocked in copy2
    run.save_file(f, name="b.txt")                     # fills the queue
    t0 = time.monotonic()
    with pytest.warns(R.RegistryWriteWarning):
        run.close(timeout=1.0)
    took = time.monotonic() - t0
    release.set()
    assert took < 1.5, took


@pytest.mark.parametrize("name", ["../../../outside.txt", "a/b.txt", "..", ".", "", "x\0y"])
def test_save_file_name_stays_in_the_run(tmp_path, name):
    f = tmp_path / "x.txt"
    f.write_text("x")
    with R.track("files") as run:
        with pytest.raises(OptionError, match="file name"):
            run.save_file(f, name=name)
    assert not (tmp_path / "outside.txt").exists()


def test_one_malformed_record_does_not_break_search(private_registry):
    with R.track("good"):
        pass
    bad = Path(private_registry) / "runs" / "19700101_000000_bad_000000"   # not 20xx: the export scrubs run ids
    bad.mkdir(parents=True)
    (bad / "run.json").write_text("{}")
    other = Path(private_registry) / "runs" / "19700101_000000_list_000000"
    other.mkdir()
    (other / "run.json").write_text("[1, 2]")
    assert [r.name for r in R.find()] == ["good"]


@pytest.mark.parametrize("level", [12, -1, 2.5, True, "3"])
def test_gzip_level_is_checked_at_the_call(level):
    with R.track("lvl") as run:
        with pytest.raises(OptionError, match="level"):
            run.save_field("a", np.zeros(3), compression="gzip", level=level)
    assert not run.errors


# ------------------------------------------------------------------ findings 49-52: function cache
def _factory(a):
    def scaled(x: float):
        return a * x
    return scaled


def test_two_closures_of_one_factory_do_not_share_a_cache():
    f1 = R.track("closure", cache=True)(_factory(1.0))
    f2 = R.track("closure", cache=True)(_factory(2.0))
    assert f1(3.0) == 3.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", R.CacheReuseWarning)
        assert f2(3.0) == 6.0


def test_closure_over_an_object_refuses_the_cache():
    class Cfg:
        k = 2.0
    cfg = Cfg()

    @R.track("objclosure", cache=True)
    def g(x: float):
        return cfg.k * x
    with pytest.warns(R.CacheReuseWarning, match="cfg"):
        assert g(1.0) == 2.0


SCALE = 2.0


def _uses_global(x: float):
    return SCALE * x


def test_a_global_value_read_by_the_function_enters_the_key(monkeypatch):
    f = R.track("glob", cache=True)(_uses_global)
    assert f(1.0) == 2.0
    monkeypatch.setitem(globals(), "SCALE", 3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", R.CacheReuseWarning)
        assert f(1.0) == 3.0


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", np.float32(1.5), np.int16(3),
                                   np.complex64(1 + 2j), 2.5])
def test_a_cache_hit_returns_the_same_type(value):
    @R.track("types", cache=True)
    def h(k: int):
        return {"v": value}
    first = h(1)
    with pytest.warns(R.CacheReuseWarning, match="cache hit"):
        again = h(1)
    assert type(again["v"]) is type(first["v"])
    assert repr(again["v"]) == repr(first["v"])


def test_jsonable_round_trip_keeps_strings_and_tags():
    for v in ["nan", "-inf", {"complex": [1.0, 2.0]}, {"ndarray": 1, "shape": 2, "dtype": 3},
              {"str": "nan"}, float("nan"), 1 + 2j, np.arange(3.0)]:
        back = from_jsonable(jsonable(v))
        assert type(back) is type(v) and repr(back) == repr(v), v


def _posonly(a, /, b, *rest, c=1, **kw):
    return a + b + sum(rest) + c + sum(kw.values())


def test_reproduce_keeps_positional_only_arguments():
    f = R.track("posonly")(_posonly)
    assert f(1, 2, 3, c=4, d=5) == 15
    rep = R.find(name="posonly")[0].reproduce(fn=_posonly)
    assert rep.identical


def test_cache_hit_series_are_reachable_from_the_new_run(ks):
    *_, make = ks
    rec = dict(series={"E": lambda s: float(np.mean(s.grid("u") ** 2))}, series_every=2)
    first = make().run(4, record=rec)
    with pytest.warns(R.CacheReuseWarning):
        hit = make().run(4, record=rec, cache=True)
    assert hit.cache_report["hit"]
    assert np.array_equal(R.get(hit.id).series("E")["value"], R.get(first.id).series("E")["value"])


def test_an_attribute_of_a_user_module_enters_the_key(monkeypatch):
    """Critic review of the identity lot: a user module was identified by its name only, so
    ``cfg.A = 3`` after ``cfg.A = 1`` kept the key and returned the stale value (the harness
    script ``f103_cache_user_module_attribute.py``)."""
    import sys
    import types
    cfg = types.ModuleType("v011_user_cfg")
    cfg.A = 1.0
    monkeypatch.setitem(sys.modules, cfg.__name__, cfg)

    @R.track("modattr", cache=True)
    def scaled(x: float):
        return cfg.A * x
    assert scaled(2.0) == 2.0
    cfg.A = 3.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", R.CacheReuseWarning)
        assert scaled(2.0) == 6.0


def test_an_object_attribute_of_a_user_module_refuses_the_cache(monkeypatch):
    """An attribute of a user module without a content hash is an undeclared dependency."""
    import sys
    import types
    cfg = types.ModuleType("v011_user_cfg_obj")

    class Box:
        k = 2.0
    cfg.box = Box()
    monkeypatch.setitem(sys.modules, cfg.__name__, cfg)

    @R.track("modobj", cache=True)
    def g(x: float):
        return cfg.box.k * x
    with pytest.warns(R.CacheReuseWarning, match="box"):
        assert g(1.0) == 2.0


# ------------------------------------------------------------------ independent review of lot A
def test_series_functions_and_cadence_enter_the_ivp_cache_key(ks):
    """A cache hit returns the source run's series: before the review, another series function
    or another series_every hit the cache and returned the first run's series."""
    *_, make = ks
    energy = {"E": lambda s: float(np.mean(s.grid("u") ** 2))}
    first = make().run(4, record=dict(series=energy, series_every=2))
    peak = {"E": lambda s: float(np.max(np.abs(s.grid("u"))))}
    with warnings.catch_warnings():
        warnings.simplefilter("error", R.CacheReuseWarning)          # a hit would raise here
        other_f = make().run(4, record=dict(series=peak, series_every=2), cache=True)
        other_every = make().run(4, record=dict(series=energy, series_every=1), cache=True)
    assert not other_f.cache_report["hit"] and not other_every.cache_report["hit"]
    assert not np.array_equal(R.get(other_f.id).series("E")["value"],
                              R.get(first.id).series("E")["value"])
    assert len(R.get(other_every.id).series("E")["value"]) > len(R.get(first.id).series("E")["value"])
    with pytest.warns(R.CacheReuseWarning, match="cache hit"):           # same series: a hit
        assert make().run(4, record=dict(series=energy, series_every=2), cache=True) \
            .cache_report["hit"]


def test_a_recursive_container_refuses_the_cache_instead_of_crashing():
    box = [2.0]
    box.append(box)                                                  # contains itself

    @R.track("loop", cache=True)
    def g(x: float):
        return box[0] * x
    with pytest.warns(R.CacheReuseWarning, match="contains itself"):
        assert g(1.0) == 2.0


def _scale(k):
    import functools

    def deco(f):
        @functools.wraps(f)
        def wrapped(x):
            return k * f(x)
        return wrapped
    return deco


def _base(x: float):
    return x


def test_the_users_decorators_are_called_and_enter_the_key():
    """``@track`` over a user decorator: the call skipped the decorator (inspect.unwrap) and the
    key ignored its parameters, so @scale(2) returned x and @scale(3) hit @scale(2)'s cache."""
    f2 = R.track("deco", cache=True)(_scale(2.0)(_base))
    f3 = R.track("deco", cache=True)(_scale(3.0)(_base))
    assert f2(1.5) == 3.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", R.CacheReuseWarning)
        assert f3(1.5) == 4.5
    runs = {r.results["return"]: r for r in R.find(name="deco")}
    assert runs[3.0].reproduce(fn=f2).identical                    # replayed with the decorator


class _Cfg:
    A = 1.0


def _uses_class_attribute(x: float):
    return _Cfg.A * x


def test_a_class_attribute_read_by_the_function_enters_the_key(monkeypatch):
    """A class used as a namespace was identified by its source only: ``_Cfg.A = 3`` kept the
    key of ``_Cfg.A = 1`` (the hole closed for user modules, F103, left open for classes)."""
    f = R.track("clsattr", cache=True)(_uses_class_attribute)
    assert f(2.0) == 2.0
    monkeypatch.setattr(_Cfg, "A", 3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", R.CacheReuseWarning)
        assert f(2.0) == 6.0


def test_an_enum_member_is_hashed_by_value_not_refused():
    import enum

    class Mode(enum.Enum):
        FAST = 2.0

    @R.track("enum", cache=True)
    def g(x: float):
        return Mode.FAST.value * x
    assert g(1.0) == 2.0
    with pytest.warns(R.CacheReuseWarning, match="cache hit"):
        assert g(1.0) == 2.0


def test_a_run_recorded_before_011_is_reproduced_with_a_warning(ks):
    """A 0.1.0 record lacks the keys the identity added: they are named in an IdentityWarning
    and the end state is still compared bit for bit (identity review; the lot refused it)."""
    from conatus.io._identity import IdentityWarning
    *_, make = ks
    run = make().run(4, record=True)
    p = run.path / "run.json"
    rec = json.loads(p.read_text())
    for k in ("identity", "identity_sha256"):
        rec.pop(k, None)
    for k in ("data", "functions", "undeclared"):
        rec["numerics"].pop(k, None)
    for k in ("time_parameter", "bc_time", "f_backend", "complexify", "decouple", "pivot"):
        rec["numerics"]["time_stepping"].pop(k, None)
    p.write_text(json.dumps(rec))
    with pytest.warns(IdentityWarning, match="before 0.1.1.*f_backend"):
        rep = R.get(run.id).reproduce(make(), record=False)
    assert rep.identical
    rec["numerics"]["time_stepping"]["dt"] = 0.01                   # any other difference
    p.write_text(json.dumps(rec))
    with pytest.raises(OptionError, match="numerics differ"):
        R.get(run.id).reproduce(make(), record=False)


def _uses_a_shadowed_builtin(x: float):
    return max * x                                   # the module global below, not the builtin


def test_a_global_shadowing_a_builtin_enters_the_key(monkeypatch):
    """Every name of a builtin was skipped, so a module global ``max = 2.0`` read by the
    function was not part of its key: changing it returned the stale value."""
    monkeypatch.setitem(globals(), "max", 2.0)
    f = R.track("shadow", cache=True)(_uses_a_shadowed_builtin)
    assert f(1.0) == 2.0
    monkeypatch.setitem(globals(), "max", 3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", R.CacheReuseWarning)
        assert f(1.0) == 3.0


def test_a_numpy_string_return_is_stored_and_reused():
    """np.str_ is a NumPy scalar but not a number: stored as a 0-d HDF5 field it was lost in
    the writer thread (no HDF5 type for '<U3'), and the cache could not read it back."""
    @R.track("npstr", cache=True)
    def h(k: int):
        return {"v": np.str_("abc"), "w": np.float32(2.5)}
    first = h(1)
    with pytest.warns(R.CacheReuseWarning, match="cache hit"):
        again = h(1)
    assert again["v"] == first["v"] == "abc" and type(again["w"]) is np.float32
