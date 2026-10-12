"""conj() in a linear LHS: the equivalent doubled system (X, conj X), declared (gallery, BACKLOG).

An LHS such as ``A v + B conj(v)`` is R-linear, not C-linear: along a Fourier direction conj couples
the wavenumbers k and -k, so it has no matrix in one group. The standard remedy is to treat
``w = conj(v)`` as an unknown of its own and to append the conjugate of every equation:

    A v + B w = f,            conj(A) w + conj(B) v = conj(f),

where conj(A) is A with its coefficients conjugated (derivatives, traces at real points, integrals
and lifts are real operators). This system is C-linear; in a ComplexFourier group k it holds
(v_k, conj(v_{-k})), i.e. exactly the R-linear pair (k, -k). Its solutions with w = conj(v) are the
solutions of the original problem (the real system in (Re v, Im v) is the same system after the
change of variables (v, w) = (X + iY, X - iY)).

EVP: the eigenvalue is *not* conjugated in the appended equations. It stands for d/dt with real
time (sigma v comes from dt(v), and conj(dt v) = dt conj(v)), which is how linearize() produces it;
this convention is declared by explain(). An eigenvalue that is not a time derivative (e.g. a
complex wavenumber in a spatial problem) must not use this construction.

Refused (explicit): IVP (the explicit RHS and the state would have to carry conj(v) consistently),
symmetry subspaces, a conj() applied to an unknown-free complex parameter that is the eigenvalue.
"""
from __future__ import annotations

from ..ir import expr as E
from ..ir.problem import EVP, IVP, LBVP, Problem, ProblemError

__all__ = ["has_lhs_conj", "conjugate_doubling", "SUFFIX"]

SUFFIX = "_conj"


def has_lhs_conj(problem: Problem) -> bool:
    """True if some LHS applies conj() to an expression of the unknowns."""
    V = frozenset(problem.variables)
    return any(isinstance(n, E.Conj) and n.args[0].depends_on(V)
               for eq in problem.equations for n in eq.lhs.walk())


def _conj(x: E.Expr, vmap: dict, ev: E.Parameter | None) -> E.Expr:
    """conj(x) pushed down to the leaves: unknowns -> their partners, coefficients conjugated."""
    if isinstance(x, E.Field):
        return vmap[x] if x in vmap else E.conj(x)          # real coefficient: conj(f) = f
    if isinstance(x, E.Parameter):
        if ev is not None and x == ev:
            return x                                         # eigenvalue = d/dt (declared)
        return E.conj(x)
    if isinstance(x, E.Constant):
        return E.conj(x)
    if isinstance(x, E.Conj):
        return x.args[0]                                     # conj(conj(a)) = a (a has no unknown)
    if not x.args:
        return x                                             # unit vectors, other real leaves
    return x.rebuild([_conj(a, vmap, ev) for a in x.args])


def _replace_conj(x: E.Expr, vmap: dict, ev: E.Parameter | None) -> E.Expr:
    """Replace every conj(a) with a depending on unknowns by the pushed-down conjugate of a."""
    V = frozenset(vmap)
    mapping = {n: _conj(n.args[0], vmap, ev) for n in x.walk()
               if isinstance(n, E.Conj) and n.args[0].depends_on(V)}
    return x.substitute(mapping) if mapping else x


def conjugate_doubling(problem: Problem) -> Problem:
    """The doubled C-linear problem in (X, conj X), with the construction in its explain()."""
    if isinstance(problem, IVP):
        raise ProblemError(
            "conj() of an unknown in an IVP LHS is not supported: the explicit RHS and the state "
            "would have to carry conj(u) consistently. Write the equation in real variables, or "
            "linearize() to an EVP (doubled system (u, conj u) there)")
    if not isinstance(problem, (EVP, LBVP)):
        raise ProblemError(f"conj() in the LHS: {problem.kind} not supported (EVP, LBVP only)")
    if problem.symmetry is not None:
        raise ProblemError("conj() in the LHS with a symmetry subspace: not supported")
    names = {v.name for v in problem.variables}
    bars = {v: E.Field(v.name + SUFFIX, v.domain, v.tensorsig, v.dtype) for v in problem.variables}
    clash = {b.name for b in bars.values()} & names
    if clash:
        raise ProblemError(f"conj() doubling: variable names {sorted(clash)} already exist")
    vmap = {**bars, **{b: v for v, b in bars.items()}}
    ev = problem.eigenvalue if isinstance(problem, EVP) else None
    if ev is not None and any(isinstance(n, E.Conj) and ev in n.free_parameters()
                              for eq in problem.equations for n in eq.lhs.walk()):
        raise ProblemError(f"conj() of the eigenvalue {ev.name} in the LHS: not a linear pencil")
    note = (f"conj() in the LHS: solved as the equivalent doubled C-linear system in "
            f"({', '.join(v.name for v in problem.variables)}) and their conjugates "
            f"({', '.join(b.name for b in bars.values())}); every equation is followed by its "
            f"conjugate (coefficients conjugated). In a ComplexFourier group k the unknowns are "
            f"(X_k, conj X_(-k)): the R-linear pair (k, -k)."
            + (f" The eigenvalue {ev.name} is not conjugated: it stands for d/dt with real time "
               "(as produced by linearize()); each eigenvalue appears in the groups k and -k."
               if ev is not None else " Solutions satisfy X_conj = conj(X) (reported by solve()).")
            + f" System size doubled ({len(problem.equations)} -> {2 * len(problem.equations)} "
            "equations).")
    base = EVP if ev is not None else LBVP          # plain classes (LinearizedEVP has extra state)
    orig = problem._extra_explain()                 # e.g. the linearization's base state, kept
    cls = type(f"Doubled{base.__name__}", (base,), {
        "_extra_explain": lambda self: "\n".join(t for t in (orig, note) if t)})
    new_vars = list(problem.variables) + list(bars.values())
    name = f"{problem.name} [doubled: (X, conj X)]"
    out = cls(new_vars, ev, name=name) if ev is not None else cls(new_vars, name=name)
    first = [(_replace_conj(eq.lhs, vmap, ev), eq.rhs) for eq in problem.equations]
    second = [(_conj(lhs, vmap, ev), _conj(rhs, {}, None)) for lhs, rhs in first]
    for lhs, rhs in first + second:
        out.add_equation((lhs, rhs))
    out._tau_log = list(problem._tau_log) + ["conjugate equations reuse the taus' partners (X_conj)"]
    out.doubling_note = note
    out.doubled_from = problem
    out.conj_partner = {v.name: b.name for v, b in bars.items()}
    return out
