"""F808 (external review of 0.2.3, finding 9): the kind of a zero-eigenvalue point is verified.

Oracles, each by the equilibria of the scalar equation. x' = eps r - x^2: x = +-sqrt(eps r), a
fold for every eps > 0. x' = eps r x - x^2: x = 0 and x = eps r, transcritical.
x' = eps r x - x^3: pitchfork. x' = -x^2 with an unused parameter: no unfolding. Rescaling r by
a positive constant must not change the kind.
"""
import numpy as np
from _acceptance import check, passed, refused, start

start(808, "normal_form verifies each kind under parameter rescaling")
import conatus.expert as C  # noqa: E402


def nf(rhs, tol=1e-8, parameter="r", x0=0.0):
    pb = C.ode_problem(["x"], {"r": 0.0}, lambda x, p: [rhs(x["x"], p["r"])])
    return C.normal_form(C.ir_system(pb, ["r"]), np.array([x0]), kind="zero", parameter=parameter,
                         kernel_tol=tol)


for eps in (1.0, 1e-3, 1e-9, 1e-14, 1e6):
    for tol in (1e-8, 1e-12):
        a = nf(lambda x, r, e=eps: e * r - x ** 2, tol)
        check(a.kind == "fold" and not a.degenerate, f"fold, eps {eps:g}, tol {tol:g}: {a.kind}")
        b = nf(lambda x, r, e=eps: e * r * x - x ** 2, tol)
        check(b.kind == "transcritical", f"transcritical, eps {eps:g}: {b.kind}")
        c = nf(lambda x, r, e=eps: e * r * x - x ** 3, tol)
        check(c.kind == "pitchfork" and not c.degenerate, f"pitchfork, eps {eps:g}: {c.kind}")
pb = C.ode_problem(["x", "y"], {"r": 0.0}, lambda x, p: [-x["x"] ** 2, p["r"] - x["y"]])
n = C.normal_form(C.ir_system(pb, ["r"]), np.zeros(2), kind="zero", parameter="r")
check(n.kind == "unresolved" and n.degenerate, f"no unfolding: {n.kind}")
n = nf(lambda x, r: r - x ** 2, parameter=None)
check(n.kind == "unresolved", f"no parameter: {n.kind}")
check("unresolved" in n.explain() or "not established" in n.explain(), n.explain())
n = nf(lambda x, r: r - x ** 3)
check(n.kind == "fold" and n.degenerate, f"cusp-type fold: {n.kind}, degenerate {n.degenerate}")
# Hopf: a real double zero is no Hopf point
pb = C.ode_problem(["x", "y"], {"r": 0.0}, lambda x, p: [-x["x"] ** 2 + p["r"], -x["y"]])
hit, _ = refused(lambda: C.normal_form(C.ir_system(pb, ["r"]), np.zeros(2), kind="hopf", parameter="r"))
check(hit, "Hopf normal form at a real zero root")
passed("fold, transcritical, pitchfork stable from eps 1e-14 to 1e6; missing conditions are unresolved")
