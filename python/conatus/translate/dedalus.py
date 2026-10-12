"""``conatus.from_dedalus(script)``: assisted translation of a Dedalus 3 script (v0.1 UX).

How it works
------------
The script is parsed with Python's ``ast`` module and walked statement by statement by a
*restricted interpreter*: only literals, arithmetic, tuples, lambdas, the ``dedalus.public``
names listed in ``_D3``, a whitelist of NumPy / SciPy special functions (for constants and
initial data), attribute access on the objects the translator itself created, and the equation
strings (parsed by ``conatus.ir.parse``, never evaluated) are understood. Nothing is imported from
Dedalus and no statement of the script is executed by Python itself.

Every source line gets a status in the report:

* ``translated``    the Conatus object is the same mathematics and the same method (a declared
                     rewrite such as ``trace(grad(u)) -> div(u)`` or an explicit Dedalus default
                     such as ``dealias=1`` is listed in the note);
* ``approximated``  translated, but the Conatus method differs in a way that changes digits
                     (the note says how: e.g. disk radial basis, same trial space, other tau
                     projection) -- never silent;
* ``unsupported``   not translated (the note says why and what to write instead); nothing that
                     depends on it is translated either;
* ``skipped``       output only (logging, plots, file handlers are not simulation content).

The result, :class:`DedalusTranslation`, holds the Conatus ``problem``, the declared time stepping
(``scheme``, ``timestep``, ``stop_sim_time``), the initial / given data as grid functions
(``data``), the report, ``explain()``, ``_repr_html_()`` and ``to_script()`` (a readable Conatus
script with the report as comments).

Layout (pure split; every name importable from here as before): ``dedalus_objects`` (report
lines, stand-ins of the Dedalus objects, helpers), ``dedalus_public`` (the ``d3.<name>`` map),
``dedalus_interpreter`` (the restricted interpreter), ``dedalus_result``
(``DedalusTranslation``), this module (``from_dedalus``, the entry point).
"""
from __future__ import annotations

import ast
import os

from ..errors import InputTypeError, UnsupportedError
from .dedalus_objects import (  # noqa: F401  (re-exported: public face)
    STATUSES, _SCHEMES, _DEDALUS_ONLY_SCHEMES, ReportLine, _Unsupported, _Skip, _Opaque, _Module,
    _Dist, _Bundle, _Grid, _Scheme, _ProblemDecl, _SolverDecl, _Lambda, _FieldData, _is_num,
    _trace, _rewrite_div_outer, _outer_unit, _is_main_guard, _is_field_data, _field_data_target,
    _code_of)

from .dedalus_public import (  # noqa: F401  (re-exported: public face)
    _D3, _SPHERE_NOTE, _NP_EXPR, _NP_ALLOWED, _SCIPY_SPECIAL)

from .dedalus_interpreter import (  # noqa: F401  (re-exported: public face)
    _Interpreter, _NpFunc, _DistMethod)

from .dedalus_result import (  # noqa: F401  (re-exported: public face)
    DedalusTranslation, _data_function, _GridEval)

__all__ = ["from_dedalus", "DedalusTranslation", "ReportLine"]


# --------------------------------------------------------------------------- entry point


def from_dedalus(script: str | os.PathLike, *, entry: tuple[str, dict] | None = None,
                 overrides: dict | None = None) -> DedalusTranslation:
    """Translate a Dedalus 3 script (a path, or the source text) into Conatus.

    ``entry=('max_growth_rate', dict(Rayleigh=1710, ...))`` translates the body of that function
    with these arguments (scripts that build the problem inside a function). ``overrides``
    replaces top-level ``name = value`` assignments (e.g. a smaller resolution); each override
    is listed in the report. Nothing of the script is executed by Python: see the module doc.
    """
    if isinstance(script, os.PathLike) or (isinstance(script, str) and "\n" not in script
                                           and script.endswith(".py")):
        path = os.fspath(script)
        with open(path) as fh:
            source = fh.read()
        name = os.path.basename(path)
    elif isinstance(script, str):
        source, name = script, "<string>"
    else:
        raise InputTypeError("from_dedalus needs a path or the source text")
    tree = ast.parse(source, filename=name)
    tr = _Interpreter(source, name, overrides)
    tr.run(tree.body, tr.g)
    scope = tr.g
    if entry is not None:
        fname, args = entry
        fn = tr.functions.get(fname)
        if fn is None:
            raise InputTypeError(f"entry function '{fname}' not found in {name}",
                                 fix="available: " + ", ".join(sorted(tr.functions)))
        scope = dict(tr.g)
        a = fn.args
        names = [x.arg for x in a.args]
        defaults = dict(zip(names[len(names) - len(a.defaults):], a.defaults, strict=True))
        for n in names:
            if n in args:
                scope[n] = args[n]
            elif n in defaults:
                scope[n] = tr.eval(defaults[n], tr.g)
            else:
                raise InputTypeError(f"entry: argument '{n}' of {fname} is required")
        tr._scopes = (scope,)
        tr.report.append(ReportLine(fn.lineno, f"def {fname}(...)", "translated",
                                    f"body translated with {args}"))
        tr.run(fn.body, scope)
    unused = sorted(set(tr.overrides) - tr.overrides_used)
    if unused:
        # an override that matches no top-level assignment would leave the script's value in
        # place without a word: refused with the names that can be overridden
        top = sorted({t.id for st in tree.body if isinstance(st, ast.Assign) for tt in st.targets
                      for t in ([tt] if isinstance(tt, ast.Name) else getattr(tt, "elts", []))
                      if isinstance(t, ast.Name)})
        raise InputTypeError(f"from_dedalus(overrides=...): {unused} match no top-level "
                             f"assignment of {name}",
                             why="an override replaces the value of a top-level `name = value` "
                                 "statement; nothing else is changed (arguments of an entry "
                                 "function go in entry=(name, {...}))",
                             fix="top-level names of the script: " + ", ".join(top))
    # data functions (replay the data statements with NumPy grids)
    data: dict = {}
    for f, fd in tr.data.items():
        data[f] = _data_function(fd, f, tr, data)
    decl = tr.problems[-1] if tr.problems else None
    if decl is None and not any(r.status in ("translated", "approximated") for r in tr.report):
        # nothing of the script translates ("import os; os.system(...)" came
        # back as an empty translation without a word; it is still never executed)
        shown = [f"line {r.lineno}: {r.source.strip()[:60]} ({r.status}: {r.note})"
                 for r in sorted(tr.report, key=lambda r: r.lineno)[:5]]
        raise UnsupportedError(
            f"from_dedalus({name}): nothing of the script translates (no Dedalus problem, no "
            f"statement translated; {len(tr.report)} statement(s) read)",
            where="\n".join(shown) if shown else "the script is empty",
            why="a Dedalus script declares a problem (d3.IVP, d3.EVP, d3.LBVP, d3.NLBVP) on "
                "fields of a distributor; the script was read, not run",
            fix="pass a Dedalus 3 script (a path ending in .py, or its text); the supported "
                "subset is listed in the guide",
            doc="docs/site/guide/dedalus.md")
    if len(tr.problems) > 1:
        tr.report.append(ReportLine(0, "", "approximated",
                                    f"{len(tr.problems)} problems in the script: the last one is "
                                    "returned (tr.problem)"))
    scheme = None
    if tr.solver is not None:
        scheme = tr.solver.scheme
    time_param = decl.time.name if decl is not None and decl.time is not None else None
    tr.report.sort(key=lambda r: r.lineno)
    return DedalusTranslation(name, decl.problem if decl else None, decl.kind if decl else None,
                              scheme, tr.timestep, tr.stop_sim_time, tr.stop_iteration,
                              time_param, data, tr.report,
                              {k: v for k, v in scope.items() if not k.startswith("__")},
                              tr.evp_requests, tr.code)
