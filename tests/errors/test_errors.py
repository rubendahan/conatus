"""Pedagogical errors (v0.1): type, compatibility with the historical classes, and content.

Every error below is one a beginner is likely to hit. Each test checks the class (and that the
old ``except`` clauses still catch it) and that the message says what, where, why, a fix, and
points to the documentation. The fix is a suggestion: nothing is changed in the problem.
"""
import numpy as np
import pytest

from conatus import errors as ce
from conatus.discretize import discretize
from conatus.discretize.compiler import DiscretizationError
from conatus.discretize.plan import RankError
from conatus.ir import (EVP, IVP, LBVP, CartesianCoordinates, Chebyshev, Domain, Equation, Field,
                        Parameter, RealFourier, VectorField, diff, dt, interp, lap)
from conatus.ir.problem import NonlinearLHSError, ProblemError
from conatus.reference import transforms as T
from conatus.solvers.nlbvp import NewtonNotConverged
from conatus.timestep.integrator import IVPSolver


def cheb(N=16):
    cs = CartesianCoordinates("x")
    x = cs["x"]
    b = Chebyshev(x, N, (-1, 1), dealias=1.5)
    return x, b, Domain(cs, [b])


def fourier(N=16):
    cs = CartesianCoordinates("x")
    x = cs["x"]
    b = RealFourier(x, N, (0, 2 * np.pi), dealias=1.5)
    return x, b, Domain(cs, [b])


def assert_pedagogical(err, *needles, doc=None):
    """what (first line) + where + why + fix + docs link, and the given substrings."""
    text = str(err.value)
    assert isinstance(err.value, ce.ConatusError)
    for key in ("where:", "why:", "fix:", "docs:  guide/", "(python -m conatus docs "):
        assert key in text, (key, text)
    assert "suggestion only" in text
    for n in needles:
        assert n in text, (n, text)
    if doc:
        assert err.value.doc.endswith(doc)


# -- hierarchy and compatibility -------------------------------------------------------------
def test_historical_classes_are_the_public_ones():
    assert ProblemError is ce.EquationError is ce.ProblemError
    assert NonlinearLHSError is ce.NonlinearLHSError
    assert DiscretizationError is ce.DiscretizationError and RankError is ce.RankError
    assert NewtonNotConverged is ce.NewtonNotConverged
    from conatus.ir.symmetry import SymmetryBreakingError, SymmetryError
    from conatus.timestep.spectral import SpectralError
    assert SymmetryError is ce.SymmetryError and SymmetryBreakingError is ce.SymmetryBreakingError
    assert SpectralError is ce.SpectralError


@pytest.mark.parametrize("cls, builtin", [
    (ce.EquationError, ValueError), (ce.DiscretizationError, ValueError),
    (ce.RankError, ValueError), (ce.UndeclaredVariableError, KeyError),
    (ce.ParameterError, KeyError), (ce.InputTypeError, TypeError), (ce.DtypeError, TypeError),
    (ce.ConvergenceError, RuntimeError), (ce.BackendError, RuntimeError),
    (ce.MissingDependencyError, ImportError), (ce.UnsupportedError, NotImplementedError),
    (ce.OptionError, ValueError)])
def test_builtin_compatibility(cls, builtin):
    assert issubclass(cls, ce.ConatusError) and issubclass(cls, builtin)


def test_rank_error_is_both_discretization_and_bc_error():
    assert issubclass(ce.RankError, ce.DiscretizationError)
    assert issubclass(ce.RankError, ce.BoundaryConditionError)


def test_one_argument_raise_keeps_message_and_adds_doc_link():
    e = ce.DiscretizationError("old message")
    assert str(e).splitlines()[0] == "old message"
    assert "guide/errors#discretizationerror  (python -m conatus docs guide/errors)" in str(e)
    assert e.doc == "docs/site/guide/errors.md#discretizationerror"      # the source path, for tools
    assert "docs/site" not in str(e)                     # not a path of the source repository
    assert "fix:" not in str(e)


def test_key_error_subclass_is_not_quoted():
    e = ce.ParameterError("no value for parameter nu", fix="give it a value")
    assert str(e).startswith("no value for parameter nu\n")


def test_explain_and_catalogue():
    tree = ce.explain()
    for name in ("ConatusError", "EquationError", "RankError", "BackendError", "UnsupportedError"):
        assert name in tree
    cat = ce.catalogue()
    assert "## RankError" in cat and "## ParameterError" in cat


def test_did_you_mean():
    assert "'RK443'" in ce.did_you_mean("RK433", ("RK222", "RK443"))
    assert ce.did_you_mean("zzz", ("RK222",)) == ""


# -- boundary conditions and rank -------------------------------------------------------------
def test_missing_bc_names_the_direction_and_the_count():
    x, b, D = cheb()
    u = Field("u", D)
    pb = LBVP([u])
    pb.add_equation((diff(diff(u, x), x), 1))
    pb.add_bc(interp(u, x, -1))
    with pytest.raises(RankError, match="FAIL") as err:
        discretize(pb).solve()
    assert_pedagogical(err, "1 boundary condition(s) missing along x", "total differential order 2",
                       "add_bc", doc="#rankerror")
    assert isinstance(err.value, ProblemError)          # also a BoundaryConditionError


def test_extra_bc_is_refused_before_a_silently_wrong_solution():
    """ (v0.1): three conditions on u'' = 1 got three taus; the square system solved to a
    7e-15 discrete residual and the solution was off by 0.94 (the PDE carried an extra tau)."""
    x, b, D = cheb()
    u = Field("u", D)
    pb = LBVP([u])
    pb.add_equation((diff(diff(u, x), x), 1))
    with pytest.raises(ce.BoundaryConditionError) as err:
        pb.add_bc(interp(u, x, -1), interp(u, x, 1), interp(diff(u, x), x, 1))
    assert_pedagogical(err, "too many boundary conditions along x", "keep 2 condition")
    assert len(pb.equations) == 1 and len(pb.variables) == 1   # nothing half-added
    # the two correct conditions give the exact solution (x^2 - 1) / 2
    pb.add_bc(interp(u, x, -1), interp(u, x, 1))
    X = T.grid(b, dealias=False)
    assert np.abs(discretize(pb).solve().grid("u") - (X ** 2 - 1) / 2).max() < 1e-12


def test_extra_bc_counted_across_calls():
    x, b, D = cheb()
    u = Field("u", D)
    pb = LBVP([u])
    pb.add_equation((diff(diff(u, x), x), 1))
    pb.add_bc(interp(u, x, -1), interp(u, x, 1))
    with pytest.raises(ce.BoundaryConditionError, match="already declared along x: 2"):
        pb.add_bc(interp(diff(u, x), x, 1))


def test_bc_without_tau_says_add_bc():
    x, b, D = cheb()
    u = Field("u", D)
    pb = LBVP([u])
    pb.add_equation((diff(diff(u, x), x), 1))
    pb.add_equation((interp(u, x, -1), 0))
    pb.add_equation((interp(u, x, 1), 0))
    with pytest.raises(RankError) as err:
        discretize(pb).solve()
    assert_pedagogical(err, "0 tau DOFs", "lift(tau, basis, -k)")


def test_pure_neumann_suggests_a_gauge():
    x, b, D = cheb()
    u = Field("u", D)
    pb = LBVP([u])
    pb.add_equation((diff(diff(u, x), x), 1))
    pb.add_bc(interp(diff(u, x), x, -1), interp(diff(u, x), x, 1))
    with pytest.raises(RankError) as err:
        discretize(pb).solve()
    assert_pedagogical(err, "structurally singular", "gauge")


def test_bc_on_periodic_direction():
    x, b, D = fourier()
    u = Field("u", D)
    pb = LBVP([u])
    pb.add_equation((lap(u), 1))
    with pytest.raises(ce.BoundaryConditionError) as err:
        pb.add_bc(interp(u, x, 0))
    assert_pedagogical(err, "periodic coordinate x", "RealFourier(x) is periodic", "integ(u) = 0")
    assert isinstance(err.value, ValueError)


def test_ambiguous_bc_target_lists_the_equations():
    x, b, D = cheb()
    u, v = Field("u", D), Field("v", D)
    pb = LBVP([u, v])
    pb.add_equation((lap(u) + v, 1))
    pb.add_equation((lap(v) - u, 0))
    with pytest.raises(ProblemError, match="into=") as err:
        pb.add_bc(interp(u, x, -1), interp(u, x, 1))
    assert_pedagogical(err, "[0] v + lap(u) = 1", "[1]")


def test_too_few_modes_for_the_conditions_is_a_resolution_error():
    x, b, D = cheb(3)
    u = Field("u", D)
    pb = LBVP([u])
    pb.add_equation((lap(lap(u)), 1))
    with pytest.raises(ce.ResolutionError) as err:
        pb.add_bc(interp(u, x, -1), interp(u, x, 1), interp(diff(u, x), x, -1),
                  interp(diff(u, x), x, 1))
    assert_pedagogical(err, "N=3", "4 tau terms", "increase the resolution")


# -- equations --------------------------------------------------------------------------------
def test_nonlinear_lhs_in_ivp_proposes_the_rhs_form():
    x, b, D = fourier()
    u = Field("u", D)
    pb = IVP([u])
    with pytest.raises(NonlinearLHSError, match="not linear") as err:
        pb.add_equation((dt(u) + u * diff(u, x), 0))
    assert_pedagogical(err, "term 'dx(u)*u'", "add_equation((dt(u), -(dx(u)*u)))")
    assert pb.equations == []                         # nothing was moved for the user


def test_nonlinear_lhs_in_lbvp_proposes_linearize_or_nlbvp():
    x, b, D = cheb()
    u = Field("u", D)
    pb = LBVP([u])
    with pytest.raises(NonlinearLHSError) as err:
        pb.add_equation((lap(u) + u * u, 1))
    assert_pedagogical(err, "NLBVP", "linearize")


def test_undeclared_variable_in_time_derivative():
    x, b, D = fourier()
    u, v = Field("u", D), Field("v", D)
    pb = IVP([u])
    with pytest.raises(ProblemError, match="does not depend") as err:
        pb.add_equation((dt(u) - lap(u) + dt(v), 0))
    assert isinstance(err.value, ce.UndeclaredVariableError)
    assert_pedagogical(err, "'v' is not a variable", "IVP([u, v])")


def test_source_term_in_lhs():
    x, b, D = fourier()
    u = Field("u", D)
    pb = IVP([u])
    with pytest.raises(ProblemError, match="does not depend") as err:
        pb.add_equation((dt(u) - lap(u) + 1, 0))
    assert_pedagogical(err, "opposite sign")


def test_dt_in_evp():
    x, b, D = cheb()
    u, s = Field("u", D), Parameter("s", dtype=complex)
    pb = EVP([u], eigenvalue=s)
    with pytest.raises(ProblemError, match="dt") as err:
        pb.add_equation((dt(u) - lap(u), 0))
    assert_pedagogical(err, "replace dt(f) by s*f", "linearize")


def test_eigenvalue_must_be_a_parameter():
    x, b, D = cheb()
    with pytest.raises(TypeError) as err:
        EVP([Field("u", D)], eigenvalue="s")
    assert_pedagogical(err, "Parameter('s', dtype=complex)")


def test_equation_sides_with_different_tensor_signatures():
    cs = CartesianCoordinates("x", "z")
    x, z = cs.coords
    D = Domain(cs, [RealFourier(x, 8, (0, 1), dealias=1), Chebyshev(z, 8, (-1, 1), dealias=1)])
    with pytest.raises(TypeError) as err:
        Equation(VectorField("u", D), Field("p", D))
    assert isinstance(err.value, ce.TensorError)
    assert_pedagogical(err, "vector", "scalar", "dot(ez, u)")


# -- parameters and data ----------------------------------------------------------------------
def test_parameter_without_value_in_lhs():
    x, b, D = fourier()
    u, nu = Field("u", D), Parameter("nu")
    pb = IVP([u])
    pb.add_equation((dt(u) - nu * lap(u), 0))
    with pytest.raises(ValueError) as err:     # was DiscretizationError before v0.1 (CHANGELOG)
        IVPSolver(discretize(pb), "RK222", 1e-3).step()
    assert isinstance(err.value, ce.ParameterError)
    assert_pedagogical(err, "Parameter('nu', 1e-2)", "params={'nu': ...}")


def test_parameter_without_value_in_rhs():
    x, b, D = fourier()
    u, nu = Field("u", D), Parameter("nu")
    pb = IVP([u])
    pb.add_equation((dt(u) - lap(u), nu * u))
    solver = IVPSolver(discretize(pb), "RK222", 1e-3)
    with pytest.raises(KeyError) as err:
        solver.step()
    assert isinstance(err.value, ce.ParameterError)
    assert_pedagogical(err, "no value for parameter nu", "time_parameter")


def test_rhs_field_without_data():
    x, b, D = fourier()
    u, v = Field("u", D), Field("v", D)
    pb = IVP([u])
    pb.add_equation((dt(u) - lap(u), v * u))
    solver = IVPSolver(discretize(pb), "RK222", 1e-3)
    with pytest.raises(KeyError) as err:
        solver.step()
    assert isinstance(err.value, ce.UndeclaredVariableError)
    assert_pedagogical(err, "field 'v'", "never assumed to be zero")


def test_complex_parameter_value_needs_complex_dtype():
    with pytest.raises(TypeError, match="declare dtype=complex"):
        Parameter("c", 1 + 2j)


# -- options, backends, declarations ----------------------------------------------------------
def _heat():
    x, b, D = fourier()
    u = Field("u", D)
    pb = IVP([u])
    pb.add_equation((dt(u) - lap(u), 0))
    return discretize(pb)


def test_unknown_scheme_suggests_close_names():
    with pytest.raises(ValueError) as err:
        IVPSolver(_heat(), "RK4", 1e-3)
    assert isinstance(err.value, ce.OptionError)
    assert_pedagogical(err, "Did you mean", "'RK443'")


def test_backend_gpu_is_spelled_cuda():
    with pytest.raises(ValueError) as err:
        IVPSolver(_heat(), "RK222", 1e-3, backend="gpu")
    assert_pedagogical(err, "spelled 'cuda' here (not 'gpu')")


def test_cuda_backend_without_device():
    from conatus.runtime import cuda
    if cuda.available():
        pytest.skip("a CUDA device is visible: the error cannot be triggered here")
    pytest.importorskip("cupy")
    with pytest.raises(RuntimeError) as err:
        IVPSolver(_heat(), "RK222", 1e-3, backend="cuda")
    assert isinstance(err.value, ce.BackendError)
    assert_pedagogical(err, "no CUDA device", "backend='cpu'")


def test_missing_cupy_is_a_missing_dependency(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "cupy", None)          # import cupy -> ImportError
    with pytest.raises(ImportError) as err:
        ce.require_cupy("test")
    assert isinstance(err.value, ce.MissingDependencyError)
    # the install section, not one recipe (".[gpu]" is wrong for conda)
    assert_pedagogical(err, "install.md#gpu-cuda-install")
    assert ".[gpu]" not in str(err.value) and "CONATUS_CUDA=ON" in str(err.value)


def test_dealias_flag_is_refused_with_the_factor_to_use():
    cs = CartesianCoordinates("x")
    with pytest.raises(TypeError) as err:
        Chebyshev(cs["x"], 16, (-1, 1), dealias=True)
    assert isinstance(err.value, ce.InputTypeError)
    assert "dealias=3/2" in str(err.value) and "docs:" in str(err.value)


def test_cartesian_coordinates_count():
    with pytest.raises(ValueError) as err:
        CartesianCoordinates("x", "y", "z", "w")
    assert isinstance(err.value, ce.CoordinateError)
    assert "CartesianCoordinates('x', 'y', 'z', 'w')" in str(err.value)


def test_two_bases_for_one_coordinate():
    cs = CartesianCoordinates("x")
    x = cs["x"]
    with pytest.raises(ValueError) as err:
        Domain(cs, [Chebyshev(x, 8, (-1, 1), dealias=1), Chebyshev(x, 8, (-1, 1), dealias=1)])
    assert_pedagogical(err, "one basis per coordinate")
