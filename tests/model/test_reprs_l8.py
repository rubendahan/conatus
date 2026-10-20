"""One-line reprs of the objects a user prints and the expected accuracy stated by the
spectral-tail warning of a selected eigenvalue."""
import numpy as np

import conatus as cn
from conatus.expert import NewtonKrylovHookstep
from conatus.solvers.spurious import expected_accuracy, unresolved_sentence


def _sim():
    m = cn.Model(cn.Box(x=cn.periodic(6.0)), name="heat")
    m.scalar("u")
    m.equations("dt(u) - dx(dx(u)) = 0")
    return m.simulate(N=8, scheme="RK222", dt=1e-3)


def test_simulation_and_solver_reprs_are_one_line():
    sim = _sim()
    sim.initial(u=lambda x: np.cos(2 * np.pi * x / 6.0))
    sim.run(steps=2)
    r = repr(sim)
    assert r.startswith("Simulation('heat', N=(x=8), scheme='RK222', dt=0.001, backend='cpu'")
    assert "iteration=2" in r and r.endswith("; .explain())") and "\n" not in r
    s = repr(sim.solver)
    assert s.startswith("IVPSolver(scheme='RK222', dt=0.001, backend='cpu', unknowns=")
    assert "iteration=2" in s and "\n" not in s


def test_half_built_objects_still_print():
    from conatus.model.run import Simulation
    from conatus.timestep import IVPSolver
    assert repr(Simulation.__new__(Simulation)) == "Simulation(<not constructed>)"
    assert repr(object.__new__(IVPSolver)).startswith("IVPSolver(scheme=None")


def test_newton_krylov_repr_lists_every_control():
    r = repr(NewtonKrylovHookstep(tol=1e-12, max_iterations=7))
    assert r.startswith("NewtonKrylovHookstep(tol=1e-12, max_iterations=7, gmres_rtol=")
    assert "radii='absolute'" in r and r.endswith("; .explain())")


def test_tail_warning_states_the_expected_eigenvalue_accuracy():
    """Z4: the selected-mode tail (an energy) at 1e-8 means an eigenvalue good to about 1e-4
    (Orr-Sommerfeld at N = 32 is right to 4e-5): the sentence says so."""
    assert abs(expected_accuracy(1e-8) - 1e-4) < 1e-18
    text = unresolved_sentence("c = 0.2375+0.0037j", 1e-8)
    assert "about sqrt(tail) = 1e-04" in text and "not a bound" in text
