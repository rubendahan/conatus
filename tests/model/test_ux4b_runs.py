"""Runs that went wrong without a word.

* Y2: a checkpoint of another model, other equations, other parameters, another box or another
  resolution is refused at restart (both formats), unless ``allow=`` names the change, which is
  then recorded; ``cn.io.latest`` and ``resume_or_start`` find ``.npz`` checkpoints too.
* Y4: callable initial data are checked: non-finite, aliased or not periodic data are refused,
  data that are only not fully resolved (coarse N) warn once (review of the lot: two levels).
* Y5: ``explain()`` does not factorize M + L (Allen--Cahn used to raise).
* Y6: an under-resolved run warns once (KS at N = 8); an ill-posed implicit term warns at
  compile (backward heat), naming the term.
"""
import json
import warnings

import numpy as np
import pytest

import conatus as cn
from conatus.errors import (ConatusError, IllPosedWarning, InitialDataWarning, InputTypeError,
                            OptionError, ResolutionError, UnderResolvedWarning)
from conatus.io._identity import CheckpointWarning


def _heat(eq="dt(T) - kappa*lap(T) = 0", L=2 * np.pi, kappa=1.0, name="heat", N=16, dt=1e-3):
    m = cn.Model(cn.Box(x=cn.periodic(L)), name=name, kappa=kappa)
    m.scalar("T")
    m.equations(eq)
    return m.simulate(N=N, scheme="RK443", dt=dt)


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    d = tmp_path_factory.mktemp("ux4b_ck")
    a = _heat()
    a.initial(T=lambda x: np.cos(x))
    a.run(steps=20)
    return {ext: a.save(d / f"chk{ext}") for ext in (".npz", ".h5")}, a


# ---------------------------------------------------------------- Y2: identity of a checkpoint
@pytest.mark.parametrize("ext", [".npz", ".h5"])
@pytest.mark.parametrize("other, words", [
    (dict(kappa=2.0), ["parameter kappa: 1 in the checkpoint, 2 now"]),
    (dict(eq="dt(T) + lap(T) + lap(lap(T)) = -T*dx(T)", name="ks"),
     ["model: 'heat' in the checkpoint, 'ks' now", "equations: hash"]),
    (dict(eq="dt(T) - kappa*lap(T) = -T"), ["equations: hash"]),           # same name, other eq
    (dict(L=22.0), ["geometry along x", "22.0"]),
    (dict(N=32), ["resolution along x: 16 in the checkpoint, 32 now"]),
])
def test_restart_refuses_a_checkpoint_of_another_run(saved, ext, other, words):
    paths, _ = saved
    with pytest.raises(OptionError) as e:
        _heat(**other).restart(paths[ext])
    msg = str(e.value)
    assert "is not a state of this run" in msg
    for w in words:
        assert w in msg, (w, msg)


@pytest.mark.parametrize("ext", [".npz", ".h5"])
def test_restart_of_the_same_run_is_bit_for_bit(saved, ext):
    paths, a = saved
    b = _heat().restart(paths[ext])
    assert b.t == a.t and np.array_equal(b.solver.X, a.solver.X)


def test_allow_names_the_change_and_records_it(saved, tmp_path):
    paths, _ = saved
    b = _heat(kappa=2.0).restart(paths[".npz"], allow=("kappa",))
    text = b.explain()
    assert "restarted with a declared change" in text and "parameter kappa" in text
    b.run(steps=2)
    p = b.save(tmp_path / "next.npz")               # the change travels with later checkpoints
    with np.load(p) as z:
        meta = json.loads(str(z["meta"]))
    assert any("kappa" in c for c in meta["restart_changes"])
    assert meta["identity"]["parameters"]["kappa"] == 2.0
    with pytest.raises(OptionError, match="Did you mean 'kappa'"):
        _heat(kappa=2.0).restart(paths[".npz"], allow=("kapa",))
    with pytest.raises(OptionError, match="resolution"):
        _heat(N=32).restart(paths[".npz"], allow=("resolution",))
    _heat(kappa=2.0).restart(paths[".h5"], allow="parameters")       # a category


def test_old_checkpoint_without_identity_warns_and_checks_the_name(saved, tmp_path):
    paths, _ = saved
    with np.load(paths[".npz"]) as z:
        arrays = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
    meta.pop("identity")
    old = tmp_path / "old.npz"
    np.savez(old, meta=json.dumps(meta), **arrays)
    with pytest.warns(CheckpointWarning, match="only the problem name"):
        _heat(kappa=2.0).restart(old)              # parameters cannot be checked: said so
    with pytest.raises(OptionError, match="state of 'heat'"):
        _heat(name="other").restart(old)


def test_latest_and_resume_or_start_find_npz(saved, tmp_path):
    paths, a = saved
    d = tmp_path / "chain"
    d.mkdir()
    import shutil
    shutil.copy(paths[".npz"], d / f"checkpoint_{a.iteration:09d}.npz")
    assert cn.io.latest(d).suffix == ".npz"
    b = _heat().resume_or_start(d, T=lambda x: 0 * x)
    assert b.t == a.t and "continued from" in b.explain()
    with pytest.raises(OptionError, match="kappa"):
        _heat(kappa=5.0).resume_or_start(d, T=lambda x: 0 * x)   # critique t31
    shutil.copy(paths[".h5"], d / f"checkpoint_{a.iteration:09d}.h5")
    with pytest.raises(OptionError, match="2 checkpoints of iteration"):
        cn.io.latest(d)
    assert cn.io.latest(d, "*.h5").suffix == ".h5"     # another pattern: that glob only


def _two_eq_hash(order):
    from conatus.io._identity import equations_hash
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="two", nu=0.1)
    m.scalar("u")
    m.scalar("v")
    eqs = ["dt(u) - nu*lap(u) = -u*dx(v)", "dt(v)-nu*lap( v ) = u*u"]
    m.equations(*(eqs if order else eqs[::-1]))
    return equations_hash(m.compile(16).problem)


def test_equations_hash_is_canonical_and_process_independent():
    """Review the hash ignores whitespace and the order of terms and equations, and is
    the same in a process with another PYTHONHASHSEED (no salted hash() of strings)."""
    import os
    import subprocess
    import sys
    h = _two_eq_hash(True)
    assert h == _two_eq_hash(False)
    code = ("import sys; sys.path.insert(0, %r); import test_ux4b_runs as t; "
            "print(t._two_eq_hash(True))" % os.path.dirname(__file__))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env=dict(os.environ, PYTHONHASHSEED="12345")).stdout.split()[-1]
    assert out == h


# ---------------------------------------------------------------- Y4: callable initial data
@pytest.mark.parametrize("f, err, words", [
    (lambda x: np.cos(15 * x), ResolutionError, "aliased or not periodic on axis x"),
    (lambda x: np.sin(2 * np.pi * x), ResolutionError, "not periodic on axis x"),
    (lambda x: np.log(x - 1), InputTypeError, "non-finite"),
])
def test_initial_refuses_unresolved_or_nonfinite_callables(f, err, words):
    sim = _heat()
    with pytest.raises(err, match=words):
        sim.initial(T=f)


def test_initial_refuses_a_non_periodic_function_at_large_n():
    # the tail of a jump decays like 1/N (below the refusal level at N = 256): the ends are compared
    with pytest.raises(ResolutionError, match="values at its two ends differ"):
        _heat(N=256).initial(T=lambda x: np.sin(2 * np.pi * x))


def test_initial_warns_between_the_two_levels_and_check_false_silences():
    sim = _heat(N=8)
    with pytest.warns(InitialDataWarning, match="T along x: tail") as w:
        sim.initial(T=lambda x: np.exp(np.cos(x)))      # tail 2e-3 at N = 8: a coarse N, warned
    assert len([x for x in w if issubclass(x.category, InitialDataWarning)]) == 1
    assert "Initial data warning" in sim.explain()
    with warnings.catch_warnings():
        warnings.simplefilter("error", InitialDataWarning)
        sim.initial(T=lambda x: np.exp(np.cos(x)), check=False)
        _heat(N=32).initial(T=lambda x: np.exp(np.cos(x)))   # resolved: silent
    assert "Initial data warning" not in sim.explain()


def test_initial_random_callable_is_not_judged():
    rng = np.random.default_rng(0)
    sim = _heat()
    sim.initial(T=lambda x: 1e-3 * rng.standard_normal(np.shape(x)))
    assert "does not return the same values twice" in sim.explain()


def test_initial_accepts_resolved_data_and_check_false_is_declared():
    sim = _heat()
    sim.initial(T=lambda x: np.exp(np.sin(x)))           # tail ~1e-7 at N = 16
    sim.initial(T=lambda x: np.cos(15 * x), check=False)
    assert "Initial data check: off" in sim.explain()
    with pytest.raises(OptionError):
        sim.initial(T=lambda x: np.cos(x), check="no")


def test_initial_checks_a_wall_axis():
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), z=cn.walls(0, 1)), name="w", kappa=1.0)
    m.scalar("T")
    m.equations("dt(T) - kappa*lap(T) = 0")
    m.at("walls", T=0)
    sim = m.simulate(N={"x": 16, "z": 16}, scheme="RK443", dt=1e-3)
    sim.initial(T=lambda x, z: np.sin(np.pi * z) * np.cos(x))
    with pytest.raises(ResolutionError, match="not resolved on axis z"):
        sim.initial(T=lambda x, z: np.tanh((z - 0.5) / 0.02))


# ---------------------------------------------------------------- Y5: explain without factorizing
def test_explain_of_allen_cahn_does_not_factorize():
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="AC", eps=0.01)
    m.scalar("T")
    m.equations("dt(T) - eps*lap(T) = T - T**3")
    sim = m.simulate(N=32, scheme="RK443", dt=1e-3)
    sim.initial(T=lambda x: 0.1 * np.cos(x))
    sim.run(steps=5)
    assert "Simulation of 'AC'" in sim.explain(full=True)
    from conatus.runtime import BorderedBanded
    rb = cn.problems.native.rayleigh_benard(Ra=2000, Pr=1, Lx=4.0)
    system = rb.simulate(N={"x": 16, "z": 16}, scheme="RK443", dt=1e-3).solver.system
    for (_lab, _n, _r, kl, ku), (idx, _, _) in zip(system.bands(), system._buckets):
        s = system.slots[idx[0]]
        A = (s.M + s.L).tocsr()[s.rperm][:, s.cperm]
        bb = BorderedBanded(A, list(range(s.n_border)), dtype=system.dtype)
        assert (kl, ku) == (bb.kl, bb.ku)          # the structural band is the factor's band


# ---------------------------------------------------------------- Y6: under-resolved, ill-posed
def _ks(N):
    m = cn.Model(cn.Box(x=cn.periodic(22.0)), name="KS")
    m.scalar("u")
    m.equations("dt(u) + lap(u) + lap(lap(u)) = -u*dx(u)")
    sim = m.simulate(N=N, scheme="RK443", dt=0.01)
    sim.initial(u=lambda x: np.cos(2 * np.pi * x / 22) * (1 + np.sin(2 * np.pi * x / 22)))
    return sim


def test_under_resolved_run_warns_once_and_explains():
    sim = _ks(8)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sim.run(steps=300)
    hits = [x for x in w if issubclass(x.category, UnderResolvedWarning)]
    assert len(hits) == 1 and "u along x" in str(hits[0].message)
    assert "under-resolved at t = 1" in sim.explain()


def test_resolved_run_does_not_warn():
    sim = _ks(64)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnderResolvedWarning)
        sim.run(steps=200)
    assert "(none so far)" in sim.explain()


@pytest.mark.parametrize("eq, params, ill", [
    ("dt(T) + kappa*lap(T) = 0", dict(kappa=1.0), "kappa*lap(T)"),
    ("dt(T) = -kappa*lap(T)", dict(kappa=1.0), "kappa*lap(T)"),
    ("dt(T) - kappa*lap(T) = 0", dict(kappa=-1.0), "-kappa*lap(T)"),
    ("dt(T) + lap(T) - lap(lap(T)) = 0", {}, "lap(lap(T))"),
    ("dt(T) - kappa*lap(T) = 0", dict(kappa=1.0), None),
    ("dt(T) + lap(T) + lap(lap(T)) = -T*dx(T)", {}, None),          # KS: well-posed
    ("dt(T) + c*dx(T) = 0", dict(c=1.0), None),
    ("-dt(T) + kappa*lap(T) = 0", dict(kappa=1.0), None),
])
def test_ill_posed_implicit_term_warns_at_compile(eq, params, ill):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="ip", **params)
    m.scalar("T")
    m.equations(eq)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sim = m.simulate(N=16, scheme="RK443", dt=1e-3)
    hits = [str(x.message) for x in w if issubclass(x.category, IllPosedWarning)]
    if ill is None:
        assert not hits
        assert "ill-posed" not in sim.explain()
    else:
        assert len(hits) == 1 and ill in hits[0] and "did you mean" in hits[0]
        assert "ill-posed" in sim.explain()


def test_ill_posed_on_a_wall_axis():
    m = cn.Model(cn.Box(z=cn.walls(0, 1)), name="w", kappa=1.0)
    m.scalar("T")
    m.equations("dt(T) + kappa*dz(dz(T)) = 0")
    m.at("walls", T=0)
    with pytest.warns(IllPosedWarning, match="along z"):
        m.simulate(N=16, scheme="RK443", dt=1e-3)


def test_errors_are_conatus_errors():
    assert issubclass(ResolutionError, ConatusError)
