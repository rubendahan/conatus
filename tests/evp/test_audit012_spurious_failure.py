"""Audit of 0.1.2, F37: an unexpected failure to identify the tau variables is reported on the
result (spurious_note, filter_note) with its cause, instead of looking like "no tau"."""
from types import SimpleNamespace

import numpy as np

from conatus.solvers import evp_modes, plan_evp, spurious
from conatus.solvers.evp import EVPResult


def _plan():
    return SimpleNamespace(var_layouts=lambda g: [], layout=SimpleNamespace(fourier_indices=None),
                           problem=SimpleNamespace(eigenvalue=None, growth=None, growth_why=None))


def _boom(plan):
    raise RuntimeError("boom in tau identification")


def test_tail_layout_reports_an_unexpected_failure(monkeypatch):
    monkeypatch.setattr(evp_modes, "_tau_names", _boom)
    layout, why = spurious.tail_layout(_plan(), None, False)
    assert layout is None
    assert "unavailable" in why and "RuntimeError" in why and "boom" in why


def test_tail_layout_plan_without_ir_layout_is_still_no_tau():
    plan = SimpleNamespace(var_layouts=lambda g: [], layout=SimpleNamespace(fourier_indices=None))
    assert spurious.tail_layout(plan, None, False) == (None, "")     # no .problem: AttributeError


def test_annotate_reports_an_unexpected_failure(monkeypatch):
    monkeypatch.setattr(evp_modes, "_tau_names", _boom)
    res = EVPResult(np.zeros(0), None, np.zeros(0), "test")
    plan_evp._annotate(res, _plan(), filtered=False)
    assert res.filter_note is not None and "boom" in res.filter_note
