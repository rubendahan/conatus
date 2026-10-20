"""Frictions of grid arrays as initial state, a full
integral evaluated as a number, a typed refusal of an unknown name, the agent CLI."""
import json

import numpy as np
import pytest

import conatus as cn
from conatus.errors import InputTypeError


def _sim():
    m = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=2000.0, Pr=1.0)
    sim = m.simulate(N=(16, 12), scheme="RK443", dt=1e-3)
    sim.initial(T=lambda x, z: -z + 1e-3 * np.cos(np.pi * x))
    return m, sim


def test_initial_accepts_grid_arrays_of_the_exact_shape():
    m, a = _sim()
    b = m.simulate(N=(16, 12), scheme="RK443", dt=1e-3)
    b.initial(T=a.grid("T"), u=a.grid("u"))
    assert np.abs(b.grid("T") - a.grid("T")).max() < 1e-14
    with pytest.raises(InputTypeError, match=r"the grid of T has shape \(16, 12\)"):
        b.initial(T=a.grid("T")[:, :5])


def test_full_integral_is_a_number():
    _, sim = _sim()
    v = sim.evaluate("integ(T*T)/2")
    assert isinstance(v, float) and abs(v - 2.0 / 24) < 1e-6        # int z^2/2 over 2 x 1
    assert sim.evaluate("dz(T)").shape == sim.grid("T").shape


def test_unknown_name_is_a_typed_refusal():
    _, sim = _sim()
    with pytest.raises(InputTypeError, match=r"a component of the vector u: sim.grid\('u'\)\[i\]"):
        sim.grid("w")
    with pytest.raises(InputTypeError, match="not an unknown"):
        sim.coeffs("w")


def test_agent_cli_default_options_and_explain_never_runs_a_script(tmp_path, capsys):
    from conatus.agent import main
    assert main(["--no-probe", "--no-runs"]) == 0
    assert json.loads(capsys.readouterr().out)["command"] == "info"
    script = tmp_path / "s.py"
    script.write_text(f"open({str(tmp_path / 'ran')!r}, 'w').close()\n")
    assert main(["explain", str(script)]) == 2
    out = json.loads(capsys.readouterr().out)
    assert "never runs a script" in out["error"] and "agent check" in out["hint"]
    assert not (tmp_path / "ran").exists()
