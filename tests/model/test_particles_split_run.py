"""Finding 14: splitting a run into one-step calls changed the particle
convergence order (begin_run cleared the flow history at every call)."""
import numpy as np

import conatus as cn


def _error(dt, split):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
    m.velocity("u")
    m.equations("dt(u)+u=0")
    s = m.simulate(8, scheme="RK443", dt=dt).initial(u=lambda x: (np.ones_like(x),))
    p = s.particles([[0.]])
    n = round(1 / dt)
    if split:
        for _ in range(n):
            s.run(steps=1)
    else:
        s.run(steps=n)
    return abs(p.positions(unwrapped=True)[0, 0] - (1 - np.exp(-1)))


def test_split_run_same_as_single_run():
    for dt in (0.1, 0.05):
        assert abs(_error(dt, True) - _error(dt, False)) <= 1e-14


def test_state_change_between_runs_resets_history():
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
    m.velocity("u")
    m.equations("dt(u)+u=0")
    s = m.simulate(8, scheme="RK443", dt=0.1).initial(u=lambda x: (np.ones_like(x),))
    p = s.particles([[0.]])
    s.run(steps=2)
    s.initial(u=lambda x: (2 * np.ones_like(x),))
    p.begin_run()
    assert len(p._hist) == 1
