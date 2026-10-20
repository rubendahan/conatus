"""External review of 0.1.0, findings 38-44 (runtime wrappers and C++ core checks).

38  BorderedBandedBatch took its dtype from the first matrix only: a complex system after a real
    one lost its imaginary part.
39  ImexIntegrator dropped complex parts (an explicit real dtype over complex matrices, a complex
    x0 or a complex F(x, t) into a real integrator).
40  BorderedBanded.solve reshaped a right-hand side of the wrong shape instead of refusing it.
41  A matrix with a NaN/Inf entry factored "fine" and reported cond_1 = 0.
42  A reused SBDF integrator silently kept the history of the previous run.
43  The C++ integrator owned its Python callback: a callback referring to the integrator made a
    reference cycle the Python garbage collector could not see (a leak).
44  LAPACK info codes of the solve calls were ignored in banded.hpp.
"""
from __future__ import annotations

import gc
import re
import weakref
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from conatus.errors import DtypeError, InputTypeError, NumericalError, OptionError
from conatus.runtime import BorderedBanded, BorderedBandedBatch, ImexIntegrator, _core

ROOT = Path(__file__).resolve().parents[2]


def _system(n=12, dtype=float, seed=0):
    """Diagonally dominant tridiagonal matrix with one dense boundary row (row 0)."""
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n), dtype=dtype)
    for i in range(n):
        for j in range(max(0, i - 1), min(n, i + 2)):
            A[i, j] = rng.standard_normal() + (1j * rng.standard_normal() if dtype is complex else 0)
        A[i, i] += 8.0
    A[0] = rng.standard_normal(n) + 1.0
    return A


# ---------------------------------------------------------------------------- 38: batch dtype
def test_batch_dtype_comes_from_every_matrix_not_the_first():
    Ar, Az = _system(dtype=float, seed=1), _system(dtype=complex, seed=2)
    bt = BorderedBandedBatch([sp.csr_matrix(Ar), sp.csr_matrix(Az)], [0])
    assert bt.dtype == np.complex128
    b = np.random.default_rng(3).standard_normal((2, 12)) + 0j
    x = bt.solve(b)
    assert np.allclose(Az @ x[1], b[1], atol=1e-12) and np.allclose(Ar @ x[0], b[0], atol=1e-12)


def test_batch_explicit_real_dtype_over_a_complex_matrix_is_refused():
    with pytest.raises(DtypeError, match="imaginary"):
        BorderedBanded(sp.csr_matrix(_system(dtype=complex)), [0], dtype=np.float64)
    with pytest.raises(DtypeError, match="imaginary"):
        BorderedBandedBatch([_system(), _system(dtype=complex)], [0], dtype=np.float64)


def test_empty_batch_is_refused_with_a_typed_error():
    with pytest.raises(InputTypeError, match="at least one"):
        BorderedBandedBatch([], [0])


# ---------------------------------------------------------------------------- 39: IMEX complex
def _heat(n=10):
    """M = I, L = tridiagonal Laplacian-like, row 0 a Dirichlet row."""
    L = sp.diags([-1.0, 2.5, -1.0], [-1, 0, 1], shape=(n, n)).tolil()
    L[0, :] = 0.0
    L[0, 0] = 1.0
    M = sp.identity(n, format="lil")
    M[0, 0] = 0.0
    return M.tocsr(), L.tocsr(), [0]


def test_imex_refuses_a_complex_state_in_a_real_integrator():
    M, L, b = _heat()
    integ = ImexIntegrator(M, L, b, "SBDF2", 0.01)
    with pytest.raises(DtypeError, match="imaginary"):
        integ.run(np.ones(10) + 1j, 0.0, 2)


def test_imex_refuses_a_complex_rhs_callback_in_a_real_integrator():
    M, L, b = _heat()
    integ = ImexIntegrator(M, L, b, "RK222", 0.01, F=lambda x, t: 1j * x)
    with pytest.raises(DtypeError, match="imaginary"):
        integ.run(np.ones(10), 0.0, 1)


def test_imex_refuses_a_real_dtype_over_complex_matrices():
    M, L, b = _heat()
    with pytest.raises(DtypeError, match="imaginary"):
        ImexIntegrator(M, (1 + 1j) * L, b, "SBDF1", 0.01, dtype=np.float64)


def test_imex_complex_integrator_keeps_the_imaginary_part():
    M, L, b = _heat()
    integ = ImexIntegrator(M, L, b, "SBDF1", 0.01, dtype=np.complex128, F=lambda x, t: 1j * x)
    x, _ = integ.run(np.ones(10) * (1 + 1j), 0.0, 3)
    assert np.abs(x.imag).max() > 0.1


# ---------------------------------------------------------------------------- 40: RHS shape
def test_solve_refuses_a_rhs_of_the_wrong_shape():
    s = BorderedBanded(sp.csr_matrix(_system()), [0])
    for bad in (np.ones(24), np.ones((2, 12)), np.ones((12, 2, 2)), np.ones(()), np.ones((6, 4))):
        with pytest.raises(InputTypeError, match="right-hand side"):
            s.solve(bad)
    x = s.solve(np.ones((12, 2)))                    # (n,) and (n, nrhs) still accepted
    assert x.shape == (12, 2) and s.solve(np.ones(12)).shape == (12,)


def test_batch_solve_refuses_a_rhs_of_the_wrong_shape():
    bt = BorderedBandedBatch([_system(), _system(seed=4)], [0])
    for bad in (np.ones(24), np.ones((3, 12)), np.ones((2, 11)), np.ones((2, 12, 1))):
        with pytest.raises(InputTypeError, match="right-hand side"):
            bt.solve(bad)


# ---------------------------------------------------------------------------- 41: non-finite
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("where", ["band", "border"])
def test_non_finite_matrix_is_refused_not_cond_zero(bad, where):
    A = _system()
    A[(5, 5) if where == "band" else (0, 7)] = bad
    with pytest.raises(NumericalError, match="non-finite"):
        BorderedBanded(sp.csr_matrix(A), [0])
    with pytest.raises(NumericalError, match="non-finite"):
        BorderedBandedBatch([_system(), A], [0])


def test_core_refuses_a_non_finite_matrix_directly():
    """The C++ binding is a public entry point too (conatus.runtime._core)."""
    A = sp.csr_matrix(_system())
    A.data[3] = np.nan
    args = (12, A.indptr.astype(np.int32), A.indices.astype(np.int32), A.data.astype(float))
    with pytest.raises(ValueError, match="non-finite"):
        _core.BorderedBandedD(*args, [0], False)


def test_capacitance_condition_is_finite_and_at_least_one():
    s = BorderedBanded(sp.csr_matrix(_system()), [0])
    assert np.isfinite(s.capacitance_cond) and s.capacitance_cond >= 1.0


# ---------------------------------------------------------------------------- 42: hidden history
def _forced():
    M, L, b = _heat()
    return M, L, b, (lambda x, t: np.sin(x) * np.cos(t))


@pytest.mark.parametrize("scheme", ["SBDF2", "SBDF4"])
def test_reusing_an_sbdf_integrator_on_another_state_is_refused(scheme):
    M, L, b, F = _forced()
    integ = ImexIntegrator(M, L, b, scheme, 0.01, F=F)
    x0 = np.linspace(0, 1, 10)
    integ.run(x0, 0.0, 5)
    with pytest.raises(OptionError, match="reset"):
        integ.run(2 * x0, 0.0, 5)                    # a new trajectory with the old history


@pytest.mark.parametrize("scheme", ["SBDF1", "SBDF3", "RK443"])
def test_continuation_from_the_returned_state_is_the_uninterrupted_run(scheme):
    M, L, b, F = _forced()
    x0 = np.linspace(0, 1, 10)
    ref, tref = ImexIntegrator(M, L, b, scheme, 0.01, F=F).run(x0, 0.0, 10)
    integ = ImexIntegrator(M, L, b, scheme, 0.01, F=F)
    x, t = integ.run(x0, 0.0, 4)
    x, t = integ.run(x, t, 6)
    assert np.array_equal(x, ref) and t == tref and integ.iteration == 10


@pytest.mark.parametrize("scheme", ["SBDF3", "RK222"])
def test_reset_starts_a_new_trajectory_like_a_fresh_integrator(scheme):
    M, L, b, F = _forced()
    x0 = np.linspace(0, 1, 10)
    fresh, _ = ImexIntegrator(M, L, b, scheme, 0.01, F=F).run(2 * x0, 0.5, 7)
    integ = ImexIntegrator(M, L, b, scheme, 0.01, F=F)
    integ.run(x0, 0.0, 5)
    nfac = integ.n_factorizations
    integ.reset()
    assert integ.iteration == 0
    x, _ = integ.run(2 * x0, 0.5, 7)
    assert np.array_equal(x, fresh)
    assert integ.n_factorizations == nfac            # factorizations are kept, only history goes
    assert "history" in integ.explain()


def test_core_integrator_refuses_a_hidden_history_too():
    """The C++ class itself (conatus.runtime._core.ImexD) checks the continuation."""
    M, L, b, _ = _forced()
    from conatus.runtime.banded import csr_args
    n, mp, mi, md = csr_args(M)
    _, lp, li, ld = csr_args(L)
    s = _core.ImexD(n, mp, mi, md, lp, li, ld, b, "SBDF2", 0.01)
    x, t = s.run(np.ones(10), 0.0, 3)
    s.run(np.asarray(x), t, 1)                       # continuation: accepted
    with pytest.raises(ValueError, match="reset"):
        s.run(np.ones(10), 0.0, 1)
    s.reset()
    s.run(np.ones(10), 0.0, 1)


# ---------------------------------------------------------------------------- 43: GC cycle
def _wrapper_cycle():
    """An ImexIntegrator whose callback refers to the integrator; returns a weak reference only
    (the names live in this frame, so the cycle is the only thing left when it returns)."""
    M, L, b = _heat()
    box: dict = {}

    def F(x, t):
        return np.zeros_like(x) * box["integ"].dt

    integ = ImexIntegrator(M, L, b, "SBDF1", 0.01, F=F)
    box["integ"] = integ
    integ.run(np.ones(10), 0.0, 1)
    return weakref.ref(integ)


def _core_cycle():
    """The same cycle on the C++ class directly: holder -> ImexD -> callback -> holder."""
    M, L, b = _heat()
    from conatus.runtime.banded import csr_args
    n, mp, mi, md = csr_args(M)
    _, lp, li, ld = csr_args(L)

    class Holder:
        pass

    h = Holder()
    h.s = _core.ImexD(n, mp, mi, md, lp, li, ld, b, "SBDF1", 0.01)
    h.s.set_rhs(lambda x, t: np.zeros(10) * (h is not None))
    h.s.run(np.ones(10), 0.0, 1)
    return weakref.ref(h)


@pytest.mark.parametrize("make", [_wrapper_cycle, _core_cycle], ids=["wrapper", "core"])
def test_callback_referring_to_its_integrator_is_collected(make):
    ref = make()
    gc.collect()
    assert ref() is None, "the integrator <-> callback cycle was not collected (leak)"


# ---------------------------------------------------------------------------- 44: LAPACK info
def test_every_lapack_call_in_the_host_core_checks_its_info_code():
    """A bare ``lapack::gbtrs(...);`` statement discards the info code (an illegal argument would
    leave the right-hand side untouched and return it as the solution)."""
    for name in ("banded.hpp", "imex.hpp"):
        src = (ROOT / "core" / "include" / "conatus" / name).read_text()
        bare = [line.strip() for line in src.splitlines() if re.match(r"\s*lapack::\w+\(", line)]
        assert not bare, f"{name}: LAPACK calls with the info code ignored: {bare}"
