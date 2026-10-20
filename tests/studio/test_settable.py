"""``sim.watch(settable={"dt": (lo, hi)})``: the time step changed from the Studio or an agent,
only when declared, within its bounds, between two steps, recorded; ``run(until=T)`` stays exact
(a step that would not land on T is refused with the nearest one that does), and a watch with a
settable dt but no command is bit-identical to an unwatched run."""
from __future__ import annotations

import json

import numpy as np
import pytest

import conatus as cn
from conatus.errors import OptionError
from conatus.studio.client import read_run
from test_watch import L, _ks


def _cmds(spool, run_id, *cmds):
    with (spool / f"{run_id}.control.jsonl").open("a") as fh:
        for i, (cmd, params) in enumerate(cmds):
            fh.write(json.dumps({"cmd": cmd, "seq": i + 1, "params": params, "from": "test"}) + "\n")


def test_settable_watch_without_commands_is_bit_identical(tmp_path):
    plain, _ = _ks()
    plain.run(steps=137)
    sim, _ = _ks()
    w = sim.watch("ks", poll=5, spool=tmp_path, settable={"dt": (1e-3, 0.1)})
    assert "settable from the Studio: dt in [0.001, 0.1]" in w.explain()
    sim.run(steps=137)                              # 137: not a multiple of poll (last chunk short)
    sim.unwatch()
    assert np.array_equal(plain.solver.X, sim.solver.X) and sim.iteration == 137
    head = read_run(tmp_path, "ks").header
    assert head["controls"] == [{"name": "dt", "kind": "number", "min": 1e-3, "max": 0.1,
                                 "label": "time step"}]


def test_set_dt_lands_exactly_on_until_and_is_recorded(tmp_path):
    sim, _ = _ks()
    sim.watch("ks", poll=5, spool=tmp_path, settable={"dt": (1e-3, 0.1)}, controls_every_s=0.0)
    _cmds(tmp_path, "ks", ("set", {"name": "dt", "value": 0.025}),     # 9.75 / 0.025 = 390 steps
          ("set", {"name": "dt", "value": 0.0333}),                     # does not divide: refused
          ("set", {"name": "dt", "value": 0.5}),                        # outside the bounds
          ("set", {"name": "nu", "value": 0.01}))                       # not declared
    sim.run(until=10.0)
    sim.unwatch()
    assert sim.t == pytest.approx(10.0, abs=1e-12) and sim.dt == 0.025
    assert sim.iteration == 5 + 390
    ev = read_run(tmp_path, "ks").events
    done = [e for e in ev if e["name"] == "param_set"]
    refused = [e["reason"] for e in ev if e["name"] == "set_refused"]
    assert [(e["param"], e["value"], e["old"], e["iteration"]) for e in done] == [("dt", 0.025, 0.05, 5)]
    assert f"nearest one that does: dt = {9.75 / 293:.12g} (293 steps)" in refused[0]
    assert "outside the declared bounds" in refused[1] and "not declared settable" in refused[2]


def test_set_dt_keeps_the_step_count_of_run_steps(tmp_path):
    sim, _ = _ks()
    sim.watch("ks", poll=4, spool=tmp_path, settable={"dt": (1e-3, 0.1)}, controls_every_s=0.0)
    _cmds(tmp_path, "ks", ("set", {"name": "dt", "value": 0.02}))
    sim.run(steps=40)
    sim.unwatch()
    assert sim.iteration == 40 and sim.t == pytest.approx(4 * 0.05 + 36 * 0.02, abs=1e-12)


def test_settable_is_declared_explicitly_and_refused_where_it_cannot_hold(tmp_path):
    sim, _ = _ks()
    for bad, match in [({"nu": (0, 1)}, "only"), ({"dt": 0.01}, "bounds"), ({"dt": (0.1, 0.01)}, "lo < hi")]:
        with pytest.raises(OptionError, match=match):
            sim.watch("ks", spool=tmp_path, settable=bad)
    box = cn.Box(x=cn.periodic(L))
    m = cn.Model(box, name="KS")
    u = m.scalar("u")
    m.equations("dt(u) + lap(u) + lap(lap(u)) = -u*dx(u)")
    sbdf = m.simulate(N=32, scheme="SBDF2", dt=0.05)
    with pytest.raises(OptionError, match="constant-step"):
        sbdf.watch("ks2", spool=tmp_path, settable={"dt": (1e-3, 0.1)})
    del u
    sim.watch("ks3", spool=tmp_path, settable={"dt": (1e-3, 0.1)})
    with pytest.raises(OptionError, match="registry"):
        sim.run(steps=5, record=True)
    sim.unwatch()
