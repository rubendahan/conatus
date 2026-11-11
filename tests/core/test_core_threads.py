"""Threaded batched banded solves: the systems of a batch are split over std::threads
with the GIL released (core/include/conatus/parallel.hpp). Each system is solved by one thread with
the serial arithmetic, so the solution must be the serial one bit for bit, and correct against a
dense oracle (numpy.linalg.solve)."""
import numpy as np
import pytest
import scipy.sparse as sp

from conatus.errors import OptionError
from conatus.runtime import BorderedBandedBatch
from conatus.runtime import banded as rb

from test_core_banded import almost_banded


def _batch(rng, B, n, dtype, monkeypatch, threads="1"):
    monkeypatch.setenv("CONATUS_SOLVE_THREADS", threads)
    mats = [sp.csr_matrix(almost_banded(rng, n, 3, 2, [0, 1, 2], dtype)) for _ in range(B)]
    b = rng.standard_normal((B, n)) + (1j * rng.standard_normal((B, n)) if dtype == complex else 0)
    return mats, BorderedBandedBatch(mats, [0, 1, 2]), b


@pytest.mark.parametrize("dtype", [float, complex])
@pytest.mark.parametrize("threads", [2, 3, 8, 50])
@pytest.mark.parametrize("transpose", [False, True])
def test_threaded_batch_is_the_serial_solve_bit_for_bit(dtype, threads, transpose, rng, monkeypatch):
    """threads below, at and above the batch size (37 systems: uneven chunks)."""
    mats, batch, b = _batch(rng, 37, 60, dtype, monkeypatch)
    serial = batch._s.solve(b, transpose)
    threaded = batch._s.solve(b, transpose, threads)
    assert np.array_equal(serial, threaded)
    for A, bk, xk in zip(mats, b, threaded, strict=True):       # dense oracle, independent of the core
        Ad = A.toarray().T if transpose else A.toarray()
        ref = np.linalg.solve(Ad, bk)
        assert np.max(abs(xk - ref)) < 1e-12 * np.max(abs(ref)) * 60


def test_wrapper_uses_threads_above_the_work_threshold_only(rng, monkeypatch):
    calls = []
    _, small, b_small = _batch(rng, 4, 20, float, monkeypatch, threads="4")
    _, big, b_big = _batch(rng, 64, 400, float, monkeypatch, threads="4")
    assert small.threads == big.threads == min(4, rb.solve_threads())
    assert small._call_threads == 1 and big._call_threads == rb.threads_for(64, 400, big.threads)

    class Spy:
        def __init__(self, s):
            self.s = s

        def solve(self, *a):
            calls.append(a[2:] if len(a) > 2 else ())
            return self.s.solve(*a)

    text = big.explain()
    for batch, b in ((small, b_small), (big, b_big)):
        ref = batch._s.solve(b, False)
        batch._s = Spy(batch._s)
        assert np.array_equal(batch.solve(b), ref)
    if big.threads > 1:                          # a node with a single CPU caps the setting at 1
        assert calls == [(), (big._call_threads,)]
    assert "threads" in text or big.threads == 1


def test_solve_threads_setting(monkeypatch):
    monkeypatch.delenv("CONATUS_SOLVE_THREADS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert rb.solve_threads() == 1
    monkeypatch.setenv("OMP_NUM_THREADS", "1000000")
    assert 1 <= rb.solve_threads() <= len(__import__("os").sched_getaffinity(0))
    monkeypatch.setenv("CONATUS_SOLVE_THREADS", "1")    # the specific setting wins
    assert rb.solve_threads() == 1
    for bad in ("two", "0", "-3"):
        monkeypatch.setenv("CONATUS_SOLVE_THREADS", bad)
        with pytest.raises(OptionError, match="CONATUS_SOLVE_THREADS"):
            rb.solve_threads()


def test_ivp_state_does_not_depend_on_solve_threads(monkeypatch):
    """Rayleigh-Benard 2D, 128 x 32 (batches above the threshold): 20 RK443 steps with 1 and 4
    solve threads give the same state bit for bit."""
    import conatus as cn
    states = []
    for t in ("1", "4"):
        monkeypatch.setenv("CONATUS_SOLVE_THREADS", t)
        m = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=1e5, Pr=1)
        sim = m.simulate(N=(128, 32), scheme="RK443", dt=1e-4)
        sim.initial(T=lambda x, z: -z + 1e-2 * np.cos(np.pi * x) * np.cos(np.pi * z))
        sim.run(steps=20)
        states.append(sim.solver.X.copy())
    assert np.all(np.isfinite(states[0]))
    assert np.array_equal(states[0], states[1])


@pytest.mark.parametrize("dtype", [float, complex])
def test_threaded_solves_repeat_bit_for_bit_and_python_callers_share_a_batch(dtype, rng):
    """Race check: many threads, repeated, and several Python threads solving one fresh
    batch at once with the GIL released. The first transposed solve builds the transposed border
    data of each system; the core builds it with the GIL held before the threads start
    (prepare_transpose), so every caller reads the same data and gets the serial bits."""
    import threading
    mats = [sp.csr_matrix(almost_banded(rng, 80, 4, 3, [0, 1, 2], dtype)) for _ in range(24)]
    b = rng.standard_normal((24, 80)) + (1j * rng.standard_normal((24, 80)) if dtype == complex else 0)
    for transpose in (True, False):
        ref = BorderedBandedBatch(mats, [0, 1, 2])._s.solve(b, transpose).tobytes()
        fresh = BorderedBandedBatch(mats, [0, 1, 2])
        outs = [None] * 6
        barrier = threading.Barrier(6)

        def work(i, fresh=fresh, transpose=transpose):
            barrier.wait()
            outs[i] = [fresh._s.solve(b, transpose, 1 + i % 4).tobytes() for _ in range(5)]

        ths = [threading.Thread(target=work, args=(i,)) for i in range(6)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        assert all(o == ref for out in outs for o in out)
        for t in (2, 7, 24, 64):
            assert all(fresh._s.solve(b, transpose, t).tobytes() == ref for _ in range(10))


def test_threads_for_rule():
    """The measured rule: t = min(allowed, batch, isqrt(batch n / 900)), 1 below 2."""
    assert rb.threads_for(2, 1024, 8) == 1            # 2048 unknowns: threads lost time (measured)
    assert rb.threads_for(8, 512, 8) == 2             # 4096 unknowns: first gain (1.1x measured)
    assert rb.threads_for(32, 512, 8) == 4
    assert rb.threads_for(128, 512, 8) == 8
    assert rb.threads_for(128, 1024, 16) == 12
    assert rb.threads_for(3, 100000, 16) == 3         # never more threads than systems
    assert rb.threads_for(128, 1024, 1) == 1
