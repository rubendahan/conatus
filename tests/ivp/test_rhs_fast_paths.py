"""Fast paths of the CPU right-hand side give the bits of the code they replaced.

- the Chebyshev derivative recurrence as two sequential accumulations instead of a Python loop;
- the axis moves of the reference transforms without ``np.moveaxis``;
- one cache of sub-expressions and grids shared by all the equations of F (``evaluate_many``).
Every comparison is on the bytes of the arrays (``tobytes``), so signed zeros count.
"""
import numpy as np
import pytest

from conatus.reference import transforms as T
from conatus.timestep import spectral as S


def _loop(c):
    """The loop of 0.3.0 (timestep/spectral.py, SpectralEvaluator._diff, Chebyshev branch)."""
    N = c.shape[0]
    out = np.zeros_like(c)
    if N > 1:
        out[N - 2] = 2 * (N - 1) * c[N - 1]
        for n in range(N - 3, -1, -1):
            out[n] = out[n + 2] + 2 * (n + 1) * c[n + 1]
        out[0] = out[0] / 2
    return out


@pytest.mark.parametrize("dtype", [np.float64, np.complex128, np.float32])
@pytest.mark.parametrize("N", [1, 2, 3, 4, 5, 17, 32, 65])
def test_chebyshev_recurrence_is_the_loop_bit_for_bit(dtype, N, rng):
    for shape in ((N,), (N, 3), (N, 4, 5)):
        c = rng.standard_normal(shape).astype(dtype)
        if np.dtype(dtype).kind == "c":
            c = c + 1j * rng.standard_normal(shape)
        c[rng.random(shape) < 0.2] = 0.0
        c[rng.random(shape) < 0.1] *= -0.0                 # signed zeros
        assert S._cheb_diff_recurrence(c).tobytes() == _loop(c).tobytes()
    z = -np.zeros((N, 2))                                   # all -0.0: the loop gives +0.0 rows
    assert S._cheb_diff_recurrence(z).tobytes() == _loop(z).tobytes()


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
def test_axis_moves_are_moveaxis(ndim, rng):
    a = rng.standard_normal(tuple(range(2, 2 + ndim)))
    for ax in range(-ndim, ndim):
        f = T._to_front(a, ax)
        assert f.shape == np.moveaxis(a, ax, 0).shape and np.shares_memory(f, a)
        assert np.array_equal(f, np.moveaxis(a, ax, 0))
        g = T._to_axis(f, ax)
        assert np.array_equal(g, np.moveaxis(f, 0, ax)) and g.strides == a.strides


def _rb_solver(N=(32, 16)):
    import conatus as cn
    m = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=1e5, Pr=1)
    sim = m.simulate(N=N, scheme="RK443", dt=1e-4)
    sim.initial(T=lambda x, z: -z + 1e-1 * np.cos(np.pi * x) * np.cos(np.pi * z) + 0.05 * np.sin(3 * np.pi * x) * z)
    sim.run(steps=3)
    return sim.solver


def test_shared_cache_gives_the_per_equation_results_bit_for_bit():
    """evaluate_many (one cache for all the equations, grids memoised) against one evaluate per
    expression with the grid memo off: the stacked F must have the same bytes."""
    s = _rb_solver()
    X = s.X.copy()
    shared = s.F(X, 0.0)
    ev = s.evaluator
    data = dict(s._ncc)
    data.update((v, c) for v, c in zip(s.system.variables, s.system.scatter(X, only=s._rhs_vars),
                                       strict=True) if c is not None)
    params = dict(s.plan.params)
    out = []
    for e, shape, eq in zip(s._rhs_exprs, s.system.eq_shapes, s.plan.problem.equations, strict=True):
        if e is None:
            out.append(None)
            continue
        ev._grid_memo = None
        fresh = S.SpectralEvaluator(ev.domain)
        fresh._grid_memo = None
        # a single-expression evaluate without the memo: the 0.3.0 path
        cache = {}

        def one(x, fresh=fresh, cache=cache):
            r = cache.get(x)
            if r is None:
                r = cache[x] = fresh._node(x, one, data, params)
            return r
        out.append(ev._embed(one(e), eq.lhs.rank, shape))
    separate = s.system.gather_rhs(out)
    assert np.any(shared != 0)
    assert shared.tobytes() == separate.tobytes()
    assert ev._grid_memo is None                      # the memo lives for one call only


def test_user_function_gets_a_writable_copy_of_a_shared_grid():
    """A user pointwise function that writes into its argument (u, also used by u*dx(u)) gets a
    writable copy, and the trajectory is the one of the same function without the write."""
    import conatus as cn
    seen = []

    def inplace(a):
        seen.append(a.flags.writeable)
        a *= 2.0                                      # writes into its input
        return np.sin(a)

    def clean(a):
        return np.sin(2.0 * a)
    X = []
    for name, f in (("fw", inplace), ("fc", clean)):
        m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="p", nu=0.1)
        m.scalar("u")
        m.function(name, f)
        m.equations(f"dt(u) - nu*lap(u) = -u*dx(u) + 0.01*{name}(u)")
        sim = m.simulate(N=16, scheme="RK443", dt=1e-3)
        sim.initial(u=lambda x: np.sin(x))
        sim.run(steps=3)
        X.append(sim.solver.X.copy())
    assert seen and all(seen)
    assert np.array_equal(X[0], X[1])
