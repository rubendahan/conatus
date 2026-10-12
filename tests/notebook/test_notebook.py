"""Rich Jupyter display (conatus.notebook): well-formed HTML, escaping, SVG, optional widgets.

Runs without IPython and without ipywidgets (neither is a dependency of Conatus)."""
import builtins
import sys
import types
from html.parser import HTMLParser

import numpy as np
import pytest

from conatus import notebook as nb
from conatus.expert import (EVP, IVP, NLBVP, CartesianCoordinates, Chebyshev, Domain, Field,
                            Parameter, channel, diff, dt, interp, lift)
from conatus.discretize import discretize
from conatus.solvers import eigenvalues
from conatus.solvers.evp import EVPResult
from conatus.solvers.nlbvp import NewtonSolver
from conatus.verify.report import Check, VerificationReport

VOID = {"br", "hr", "img", "input", "meta", "link", "col", "wbr", "area", "base", "source"}


class _Balance(HTMLParser):
    """Tag-balance checker: every non-void start tag is closed in order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.tags: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):     # <rect .../>: self-closed
        self.tags.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> with open {self.stack[-3:]}")
        else:
            self.stack.pop()


def well_formed(h: str) -> _Balance:
    p = _Balance()
    p.feed(h)
    p.close()
    assert not p.errors, p.errors
    assert not p.stack, f"unclosed tags {p.stack}"
    return p


def string_evp(N=16, name="strings<b>"):
    ch = channel(N, base=None)
    u, lam = ch.field("u"), Parameter("lam", dtype=complex)
    pb = EVP([u], eigenvalue=lam, name=name)
    pb.add_equation((diff(diff(u, ch.y), ch.y) + lam * u, 0))
    pb.add_bc(interp(u, ch.y, -1), interp(u, ch.y, 1))
    return pb


def test_problem_html_balanced_and_escaped():
    pb = string_evp()
    h = pb._repr_html_()
    p = well_formed(h)
    assert "strings&lt;b&gt;" in h and "strings<b>" not in h      # the name is escaped
    assert "table" in p.tags and "style" in p.tags
    assert "Tau terms" in h and "lam" in h and "eigenvalue" in h
    assert "CONSISTENT" in h                                       # the report's own status
    assert pb.explain() == pb.explain()                            # terminal path untouched
    assert pb.rank_report()._repr_html_().count("<table") == 2


def test_parameter_name_escaped():
    cs = CartesianCoordinates("y")
    b = Chebyshev(cs["y"], 8, (-1, 1), dealias=1)
    D, D0 = Domain(cs, [b]), Domain(cs, [b]).without(cs["y"])
    u, t1 = Field("u", D), Field("t1", D0)
    k = Parameter("k<x>&", 2.0)
    pb = NLBVP([u, t1], name="p")
    pb.add_equation((diff(u, cs["y"]) - k * u + lift(t1, b, -1), 0))
    pb.add_equation((interp(u, cs["y"], -1), 1))
    h = nb.problem_html(pb)
    well_formed(h)
    assert "k&lt;x&gt;&amp;" in h and "k<x>" not in h


def test_plan_and_evp_html():
    pb = string_evp()
    plan = discretize(pb)
    h = plan._repr_html_()
    well_formed(h)
    assert "PASS" in h and "Discrete rank per group" in h and "plan.explain()" in h
    res = eigenvalues(plan)
    h = res._repr_html_()
    p = well_formed(h)
    assert "svg" in p.tags and "circle" in p.tags and "residual" in h
    assert h.count("<tr>") >= 1 + min(10, len(res))                # header + leading modes


def test_spectrum_svg_edge_cases():
    well_formed(nb.spectrum_svg(np.array([])))
    well_formed(nb.spectrum_svg(np.array([1.0 + 0j])))             # one value: finite window
    s = nb.spectrum_svg(np.array([1 + 1j, np.inf, np.nan]), highlight=[0])
    assert s.count("<circle") == 1                                 # non-finite not plotted
    r = EVPResult(np.array([1 + 2j, np.inf + 0j]), None, np.array([1e-14, np.nan]), "test<m>")
    h = nb.evp_html(r)
    well_formed(h)
    assert "test&lt;m&gt;" in h and "1 non-finite not plotted" in h


def test_verification_and_check():
    rep = VerificationReport([Check("a<b", 1e-14, 1e-12, "x & y"), Check("c", 1.0, 1e-12, "z")])
    h = rep._repr_html_()
    well_formed(h)
    assert "a&lt;b" in h and "x &amp; y" in h and "PASS" in h and "FAIL" in h
    assert "1 to investigate by hand" in h
    well_formed(rep.checks[0]._repr_html_())


@pytest.mark.filterwarnings("ignore::conatus.discretize.ultraspherical.NCCResolutionWarning")
def test_newton_and_solver_explain_html():
    cs = CartesianCoordinates("y")
    b = Chebyshev(cs["y"], 12, (0, 1), dealias=1)
    D = Domain(cs, [b])
    D0 = D.without(cs["y"])
    u, t1, t2 = Field("u", D), Field("t1", D0), Field("t2", D0)
    pb = NLBVP([u, t1, t2], name="bratu-ish")
    pb.add_equation((diff(diff(u, cs["y"]), cs["y"]) + lift(t1, b, -1) + lift(t2, b, -2), -u * u))
    pb.add_equation((interp(u, cs["y"], 0), 0))
    pb.add_equation((interp(u, cs["y"], 1), 0.1))
    ns = NewtonSolver(pb, tolerance=1e-10, damping=1.0, max_iterations=20)
    well_formed(ns._repr_html_())
    sol = ns.solve({u: lambda y: 0.1 * y})
    h = sol._repr_html_()
    well_formed(h)
    assert "CONVERGED" in h and "|F| / scale" in h


def test_ivp_solver_html():
    from conatus.timestep import IVPSolver
    cs = CartesianCoordinates("y")
    b = Chebyshev(cs["y"], 12, (-1, 1), dealias=1)
    D = Domain(cs, [b])
    D0 = D.without(cs["y"])
    u, t1, t2 = Field("u", D), Field("t1", D0), Field("t2", D0)
    pb = IVP([u, t1, t2], name="heat")
    pb.add_equation((dt(u) - diff(diff(u, cs["y"]), cs["y"]) + lift(t1, b, -1) + lift(t2, b, -2), 0))
    pb.add_equation((interp(u, cs["y"], -1), 0))
    pb.add_equation((interp(u, cs["y"], 1), 0))
    s = IVPSolver(discretize(pb), "RK222", 1e-3)
    h = s._repr_html_()
    well_formed(h)
    assert "IVPSolver(scheme=RK222" in h


def test_explain_html_generic():
    class Obj:
        def explain(self):
            return "Title <1>\n  line & 2"
    h = nb.explain_html(Obj())
    well_formed(h)
    assert "Title &lt;1&gt;" in h and "line &amp; 2" in h
    well_formed(nb.explain_html("single line"))


def _block_import(monkeypatch, *names):
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.split(".")[0] in names:
            raise ImportError(f"blocked {name}")
        return real(name, *a, **k)
    for n in names:
        monkeypatch.delitem(sys.modules, n, raising=False)
    monkeypatch.setattr(builtins, "__import__", fake)


def test_widgets_none_without_ipywidgets(monkeypatch):
    pb = string_evp()
    _block_import(monkeypatch, "ipywidgets", "IPython")
    with pytest.warns(UserWarning, match="ipywidgets is not installed"):
        assert nb.parameter_widgets(pb) is None
    well_formed(pb._repr_html_())                                  # display needs no IPython


def test_widgets_with_stub_ipywidgets(monkeypatch):
    """A minimal stand-in for ipywidgets: checks the callback contract and that nothing mutates."""
    class _W:
        def __init__(self, value=None, description="", placeholder=""):
            self.value, self.description, self._obs = value, description, []

        def observe(self, fn, names):
            self._obs.append(fn)

        def set(self, v):
            self.value = v
            for fn in self._obs:
                fn({"new": v})

    class _Box:
        def __init__(self, children):
            self.children = children
    stub = types.ModuleType("ipywidgets")
    stub.FloatText, stub.Text, stub.HTML, stub.VBox = _W, _W, _W, _Box
    monkeypatch.setitem(sys.modules, "ipywidgets", stub)

    cs = CartesianCoordinates("y")
    b = Chebyshev(cs["y"], 8, (-1, 1), dealias=1)
    D = Domain(cs, [b])
    u, lam = Field("u", D), Parameter("lam", dtype=complex)
    Re, c = Parameter("Re", 100.0), Parameter("c", 1 + 1j, dtype=complex)
    pb = EVP([u], eigenvalue=lam)
    pb.add_equation((diff(u, cs["y"]) / Re + c * u + lam * u, 0))
    seen = []
    w = nb.parameter_widgets(pb, on_change=seen.append)
    assert set(w._fields) == {"Re", "c"}                           # eigenvalue excluded
    w._fields["Re"].set(200.0)
    assert seen[-1] == {"Re": 200.0, "c": 1 + 1j}
    assert Re.value == 100.0                                       # the IR is not mutated
    w._fields["c"].set("bad")
    assert len(seen) == 1 and "cannot read" in w._status.value     # refused, callback not called
