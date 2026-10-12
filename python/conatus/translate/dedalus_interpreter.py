"""Dedalus translation, part 3: the restricted interpreter of a Dedalus script.

``_Interpreter`` walks the ``ast`` statement by statement (assignments, imports, ``add_equation``,
solver calls, the time loop, field data statements), evaluating only the understood expression
nodes and recording a status for every source line; ``_NpFunc`` and ``_DistMethod`` are the callables
it hands to the script. Nothing of the script is executed by Python (see ``conatus.translate.dedalus``).
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import UnsupportedError
from ..ir import (EVP, IVP, LBVP, NLBVP, Coordinate, CoordinateSystem, Domain, Field,
                  Parameter, unit_vector)
from ..ir import expr as E
from ..ir.bases import Basis
from ..ir.parse import Namespace, _LiftBasis, parse_equation, to_string
from .dedalus_objects import (ReportLine, _Bundle, _code_of, _Dist, _field_data_target, _FieldData,
                              _Grid, _is_field_data, _is_main_guard, _Lambda, _Module, _Opaque,
                              _ProblemDecl, _rewrite_div_outer, _Scheme, _Skip, _SolverDecl, _trace,
                              _Unsupported)
from .dedalus_public import _D3, _NP_ALLOWED, _NP_EXPR, _SCIPY_SPECIAL

# --------------------------------------------------------------------------- literal bounds
# The interpreter evaluates the script's arithmetic on Python numbers: `N = 10**10**10` used to
# hang the translation (killed by a 10 s alarm), and `[0]*10**12` or np.ones(10**11)
# would exhaust memory. Declared bounds, far above any resolution or parameter a script holds:
LITERAL_MAX_BITS = 4096              # an integer result of at most 4096 bits (about 1e1233)
LITERAL_MAX_ITEMS = 10**8            # a repeated sequence or a created array of at most 1e8 items


class LiteralTooLargeError(UnsupportedError):
    """A literal computation of the script exceeds the declared bounds (never evaluated)."""


def _too_large(what: str, size: str) -> LiteralTooLargeError:
    return LiteralTooLargeError(
        f"from_dedalus: {what} would give {size}",
        why=f"the translator evaluates script arithmetic itself and refuses results beyond "
            f"{LITERAL_MAX_BITS}-bit integers or {LITERAL_MAX_ITEMS:.0e} items "
            "(conatus.translate.dedalus_interpreter.LITERAL_MAX_BITS / LITERAL_MAX_ITEMS) instead "
            "of hanging or exhausting memory",
        fix="check the expression (a typo such as 10**10**10?), or pass the intended value with "
            "from_dedalus(..., overrides={name: value})")


def _check_binop(op: ast.operator, a: Any, b: Any) -> None:
    """Refuse, before computing it, an integer power/product or a sequence repetition beyond
    the declared bounds (cheap estimates from bit lengths: nothing large is ever built)."""
    def is_int(x: Any) -> bool:
        return isinstance(x, int)                     # Python int (bool included)
    if isinstance(op, ast.Pow) and is_int(a) and is_int(b) and b > 0 and abs(a) > 1:
        bits = b * math.log2(abs(a))
        if bits > LITERAL_MAX_BITS:
            raise _too_large(f"the integer power {a}**{b}" if b.bit_length() < 64 else
                             f"an integer power with exponent of {b.bit_length()} bits",
                             f"about {bits:.3g} bits")
    if isinstance(op, ast.Mult):
        if is_int(a) and is_int(b) and a.bit_length() + b.bit_length() > LITERAL_MAX_BITS + 1:
            raise _too_large("an integer product", f"about {a.bit_length() + b.bit_length()} bits")
        for seq, n in ((a, b), (b, a)):
            # a NumPy integer repeats a sequence too (np.int64(10**12) * [0])
            if isinstance(seq, (str, bytes, list, tuple)) and isinstance(n, (int, np.integer)) \
                    and len(seq) * max(int(n), 0) > LITERAL_MAX_ITEMS:
                raise _too_large(f"a {type(seq).__name__} repeated {n} times",
                                 f"{len(seq) * n:.3g} items")


def _check_array_size(name: str, args: tuple, kw: dict) -> None:
    """Element count of a NumPy array creation, refused beyond LITERAL_MAX_ITEMS."""
    n = None
    if name in ("zeros", "ones") and (args or "shape" in kw):
        shape = args[0] if args else kw["shape"]
        dims = shape if isinstance(shape, (tuple, list)) else (shape,)
        if all(isinstance(d, (int, np.integer)) for d in dims):
            n = math.prod(int(d) for d in dims)
    elif name == "linspace":
        num = args[2] if len(args) > 2 else kw.get("num", 50)
        n = int(num) if isinstance(num, (int, np.integer)) else None
    elif name == "arange" and args and all(isinstance(x, (int, float, np.number)) for x in args):
        start, stop, step = (0, args[0], 1) if len(args) == 1 else \
            (args[0], args[1], args[2] if len(args) > 2 else kw.get("step", 1))
        if step:
            n = max(0, math.ceil((stop - start) / step))
    if n is not None and n > LITERAL_MAX_ITEMS:
        raise _too_large(f"numpy.{name}{args!r}", f"{n:.3g} elements")


# --------------------------------------------------------------------------- interpreter


class _Interpreter:
    """Restricted interpreter of a Dedalus script (see the module docstring)."""

    def __init__(self, source: str, filename: str, overrides: dict | None) -> None:
        self.source, self.filename = source, filename
        self.lines = source.splitlines()
        self.overrides = dict(overrides or {})
        self.overrides_used: set[str] = set()    # an override that matches nothing is refused
        self.report: list[ReportLine] = []
        self.g: dict[str, Any] = {"__name__": "__main__", "True": True, "False": False,
                                  "None": None}
        self.d3 = _D3(self)
        self.problems: list[_ProblemDecl] = []
        self.solver: _SolverDecl | None = None
        self.timestep: Any = None
        self.stop_sim_time: Any = None
        self.stop_iteration: Any = None
        self.data: dict[Field, _FieldData] = {}
        self.grids: dict[str, Coordinate] = {}
        self.fields: dict[str, Field] = {}
        self.functions: dict[str, ast.FunctionDef] = {}
        self.evp_requests: list[str] = []
        self.code: list[tuple[int, str]] = []    # generated Conatus code, per source line
        self._notes: list[str] = []
        self._approx: list[str] = []
        self.line = 0

    # notes ------------------------------------------------------------------------------------
    def note(self, text: str) -> None:
        self._notes.append(text)

    def approx(self, text: str) -> None:
        self._approx.append(text)

    def _record(self, node: ast.AST, status: str, note: str = "") -> None:
        src = ast.get_source_segment(self.source, node) or self.lines[node.lineno - 1]
        notes = [n for n in dict.fromkeys(self._notes)]
        approx = [n for n in dict.fromkeys(self._approx)]
        if status == "translated" and approx:
            status = "approximated"
        full = "; ".join(x for x in [note] + approx + notes if x)
        self.report.append(ReportLine(node.lineno, src, status, full))
        self._notes, self._approx = [], []

    # statements -------------------------------------------------------------------------------
    def run(self, body: list[ast.stmt], scope: dict) -> None:
        for st in body:
            self.line = st.lineno
            try:
                self.stmt(st, scope)
            except _Skip as s:
                self._poison(st, scope, f"output / post-processing (line {st.lineno}): {s}",
                             skip=True)
                self._record(st, "skipped", str(s))
            except _Unsupported as u:
                self._poison(st, scope, str(u))
                self._record(st, "unsupported", str(u))
            except LiteralTooLargeError as err:      # refused, never recorded and continued
                seg = (ast.get_source_segment(self.source, st) or "").strip()[:80]
                err.where = f"{self.filename}, line {st.lineno}: {seg}"
                raise
            # noqa below: the translated statement is arbitrary user code (numpy, Dedalus API);
            # any failure is recorded in the translation report with its type, never hidden.
            except Exception as err:  # noqa: BLE001 -- arbitrary user code, recorded (see above)
                msg = f"{type(err).__name__}: {str(err).splitlines()[0] if str(err) else ''}"
                self._poison(st, scope, msg)
                self._record(st, "unsupported", msg)

    def _poison(self, st: ast.stmt, scope: dict, reason: str, skip: bool = False) -> None:
        """Names assigned by an untranslated statement become opaque (their uses are reported)."""
        targets = []
        if isinstance(st, ast.Assign):
            targets = st.targets
        elif isinstance(st, (ast.AugAssign, ast.AnnAssign)):
            targets = [st.target]
        for t in targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Name):
                    scope[n.id] = _Opaque(reason, st.lineno, skip)

    def stmt(self, st: ast.stmt, scope: dict) -> None:
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) \
                and isinstance(st.value.value, str):
            return                                            # docstring
        if isinstance(st, (ast.Import, ast.ImportFrom)):
            return self.imports(st, scope)
        if isinstance(st, ast.FunctionDef):
            self.functions[st.name] = st
            scope[st.name] = _Opaque(f"function {st.name} (translate its body with "
                                     f"from_dedalus(..., entry=('{st.name}', {{args}})))",
                                     st.lineno)
            return self._record(st, "skipped", f"function definition; its body is translated "
                                               f"only with entry=('{st.name}', {{...}})")
        if isinstance(st, ast.If):
            if _is_main_guard(st.test):
                return self.run(st.body, scope)
            if "rank" in ast.unparse(st.test) and not st.orelse:
                self._record(st, "skipped", "MPI rank guard: its body is read as serial code")
                return self.run(st.body, scope)
            raise _Unsupported("conditional statement (the translator follows no branches)")
        if isinstance(st, ast.Try):
            self.run(st.body, scope)
            self.run(st.finalbody, scope)
            return
        if isinstance(st, ast.While):
            return self.time_loop(st, scope)
        if isinstance(st, ast.For):
            self.eval(st.iter, scope)         # raises _Skip for a loop over results / output
            raise _Unsupported("Python loop (parameter sweep or post-processing): write the loop "
                               "around Conatus calls (e.g. ParametricEVP for eigenvalue sweeps)")
        if isinstance(st, ast.With):
            raise _Unsupported("'with' block")
        if isinstance(st, ast.Assign):
            return self.assign(st, scope)
        if isinstance(st, ast.AugAssign):
            return self.augassign(st, scope)
        if isinstance(st, ast.Expr):
            return self.expr_stmt(st, scope)
        if isinstance(st, (ast.Pass, ast.Return)):
            return self._record(st, "skipped", "no effect on the problem")
        raise _Unsupported(f"statement {type(st).__name__}")

    def imports(self, st, scope) -> None:
        names = []
        if isinstance(st, ast.Import):
            for a in st.names:
                mod = a.name
                alias = a.asname or a.name.split(".")[0]
                if mod in ("dedalus.public", "dedalus.public as d3"):
                    scope[alias] = self.d3
                elif mod == "numpy":
                    scope[alias] = _Module("numpy")
                elif mod == "scipy.special":
                    scope[alias] = _Module("scipy.special")
                else:
                    scope[alias] = _Module(f"ignored:{mod}")
                names.append(mod)
        else:
            mod = st.module or ""
            for a in st.names:
                alias = a.asname or a.name
                if mod == "dedalus" and a.name == "public":
                    scope[alias] = self.d3
                elif mod == "scipy.special" and a.name in _SCIPY_SPECIAL:
                    scope[alias] = _NpFunc("scipy.special", a.name)
                elif mod == "dedalus.public":
                    scope[alias] = getattr(self.d3, a.name)
                else:
                    scope[alias] = _Module(f"ignored:{mod}.{a.name}")
            names.append(mod)
        used = any(n.startswith(("dedalus", "numpy", "scipy.special")) for n in names)
        self._record(st, "translated" if used else "skipped",
                     "" if used else "not needed by the problem (plots, logs, MPI, timing)")

    def assign(self, st: ast.Assign, scope: dict) -> None:
        # field data: u['g'] = ..., u['g'][0] = ...
        tgt = st.targets[0]
        if len(st.targets) == 1 and _is_field_data(tgt):
            return self.field_data(st, tgt, "=", scope)
        if len(st.targets) == 1 and isinstance(tgt, ast.Attribute):
            return self.set_attribute(st, tgt, scope)
        # overrides by the caller (declared in the report)
        if (len(st.targets) == 1 and isinstance(tgt, ast.Name) and tgt.id in self.overrides
                and scope is self.g):
            val = self.overrides[tgt.id]
            scope[tgt.id] = val
            self.overrides_used.add(tgt.id)
            self._code(st, f"{tgt.id} = {val!r}")
            return self._record(st, "translated", f"value overridden by the caller: "
                                                  f"{tgt.id} = {val!r}")
        if (len(st.targets) == 1 and isinstance(tgt, (ast.Tuple, ast.List)) and scope is self.g
                and any(isinstance(e, ast.Name) and e.id in self.overrides for e in tgt.elts)):
            return self._override_unpacking(st, tgt, scope)
        val = self.eval(st.value, scope)
        for t in st.targets:
            self._bind(t, val, scope)
        status, note = self._describe(st, val, scope)
        self._record(st, status, note)

    def _override_unpacking(self, st: ast.Assign, tgt: ast.AST, scope: dict) -> None:
        """``Nx, Nz = 256, 64`` with ``overrides={'Nx': 64}``: the overridden names take the
        caller's values, the others the script's (the override of a name bound
        by tuple unpacking used to be ignored, silently)."""
        names = [e.id if isinstance(e, ast.Name) else None for e in tgt.elts]
        if any(n is None for n in names):
            raise _Unsupported("override of a name inside a nested unpacking target")
        if isinstance(st.value, (ast.Tuple, ast.List)) and len(st.value.elts) == len(names):
            vals = [self.overrides[n] if n in self.overrides else self.eval(v, scope)
                    for n, v in zip(names, st.value.elts, strict=True)]
        else:
            val = self.eval(st.value, scope)
            if not isinstance(val, (tuple, list)) or len(val) != len(names):
                raise _Unsupported("tuple unpacking of a value of another length")
            vals = [self.overrides.get(n, v) for n, v in zip(names, val, strict=True)]
        done = [n for n in names if n in self.overrides]
        for n, v in zip(names, vals, strict=True):
            self._bind(ast.Name(id=n), v, scope)
        self.overrides_used.update(done)
        text = ", ".join(f"{n} = {self.overrides[n]!r}" for n in done)
        self._code(st, "; ".join(f"{n} = {v!r}" for n, v in zip(names, vals, strict=True)))
        self._record(st, "translated", f"value overridden by the caller: {text}")

    def _bind(self, t: ast.AST, val: Any, scope: dict) -> None:
        if isinstance(t, ast.Name):
            if isinstance(val, Field) and val.name.startswith("_unnamed"):
                val = self._rename(val, t.id)
            scope[t.id] = val
            if isinstance(val, _Grid):
                self.grids[t.id] = val.coord
        elif isinstance(t, (ast.Tuple, ast.List)):
            if not isinstance(val, (tuple, list)) or len(val) != len(t.elts):
                raise _Unsupported("tuple unpacking of a value of another length")
            for tt, vv in zip(t.elts, val, strict=True):
                self._bind(tt, vv, scope)
        else:
            raise _Unsupported(f"assignment target {type(t).__name__}")

    def _rename(self, f: Field, name: str) -> Field:
        """``f = dist.Field(bases=...)`` without name=: Dedalus leaves it anonymous; Conatus
        fields have names: the assigned variable's name is used (declared)."""
        g = Field(name, f.domain, f.tensorsig, f.dtype)
        self.note(f"unnamed Dedalus field named '{name}' after its variable")
        self.fields[name] = g
        return g

    def _describe(self, st: ast.Assign, val: Any, scope: dict) -> tuple[str, str]:
        name = ast.unparse(st.targets[0])
        if isinstance(val, _Opaque):
            return "unsupported", f"depends on line {val.line}: {val.reason}"
        if isinstance(val, (tuple, list)) and any(isinstance(v, _Grid) for v in val):
            return "translated", "grid coordinates: used for initial / given data only"
        if isinstance(val, _Grid):
            return "translated", "grid coordinate: used for initial / given data only"
        if isinstance(val, _Module):
            return "skipped", "output only"
        if isinstance(val, _SolverDecl):
            s = val.scheme
            return "translated", (f"solver: IVPSolver(plan, '{s}', dt) (dt from the time loop)"
                                  if s else "solver of the problem")
        self._code(st, f"{name} = {_code_of(val, name)}")
        return "translated", ""

    def augassign(self, st: ast.AugAssign, scope: dict) -> None:
        if _is_field_data(st.target):
            op = {ast.Mult: "*=", ast.Add: "+=", ast.Sub: "-=", ast.Div: "/="}.get(type(st.op))
            if op is None:
                raise _Unsupported("in-place operation on field data")
            return self.field_data(st, st.target, op, scope)
        if isinstance(st.target, ast.Name):
            cur = self.eval(st.target, scope)
            val = self._binop(st.op, cur, self.eval(st.value, scope))
            scope[st.target.id] = val
            return self._record(st, "translated")
        raise _Unsupported("augmented assignment")

    def set_attribute(self, st: ast.Assign, tgt: ast.Attribute, scope: dict) -> None:
        obj = self.eval(tgt.value, scope)
        val = self.eval(st.value, scope)
        if isinstance(obj, _SolverDecl):
            if tgt.attr == "stop_sim_time":
                self.stop_sim_time = val
                return self._record(st, "translated", "run to this time: solver.run(n) with "
                                                      "n = stop_sim_time / dt")
            if tgt.attr == "stop_iteration":
                self.stop_iteration = val
                return self._record(st, "translated", "number of steps: solver.run(n)")
            if tgt.attr == "stop_wall_time":
                raise _Skip("wall-clock limit (a run-time setting, not the problem)")
        raise _Unsupported(f"attribute assignment .{tgt.attr}")

    def expr_stmt(self, st: ast.Expr, scope: dict) -> None:
        v = st.value
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute):
            owner_node, meth = v.func.value, v.func.attr
            if isinstance(owner_node, ast.Name) and owner_node.id in ("logger", "plt", "print",
                                                                     "logging"):
                raise _Skip("logging / plotting")
            owner = self.eval(owner_node, scope)
            if isinstance(owner, _Module) and owner.name.startswith("ignored"):
                raise _Skip("output only")
            if isinstance(owner, _ProblemDecl) and meth in ("add_equation", "add_bc"):
                return self.add_equation(st, owner, v, scope)
            if isinstance(owner, _SolverDecl):
                return self.solver_call(st, owner, meth, v, scope)
            if isinstance(owner, Field):
                if meth == "fill_random":
                    self.data.setdefault(owner, _FieldData()).unsupported = (
                        f"line {st.lineno}: fill_random (Dedalus's random layout is not "
                        "reproducible elsewhere)")
                    raise _Unsupported("fill_random: Dedalus's random field (its RNG and "
                                       "layout) cannot be reproduced; write an explicit "
                                       "initial condition")
                if meth in ("low_pass_filter", "high_pass_filter"):
                    self.data.setdefault(owner, _FieldData()).unsupported = (
                        f"line {st.lineno}: {meth}")
                    raise _Unsupported(f"{meth}: spectral filtering of data (Dedalus-specific "
                                       "layout): not translated")
                if meth in ("change_scales", "require_scales", "require_grid_space",
                            "require_coeff_space"):
                    raise _Skip("layout / output setting")
            if isinstance(owner, _Opaque):
                raise _Unsupported(f"depends on line {owner.line}: {owner.reason}")
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "print":
            raise _Skip("print")
        self.eval(v, scope)
        self._record(st, "translated")

    def add_equation(self, st, decl: _ProblemDecl, call: ast.Call, scope: dict) -> None:
        if len(call.args) != 1 or call.keywords:
            raise _Unsupported("add_equation with several arguments")
        arg = call.args[0]
        pb = decl.problem
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            text = arg.value
            ns = self._namespace(scope)
            eq = parse_equation(text, ns)
            self._notes += ns.notes
            if ns.shadowed:
                self.note("namespace functions used: " + ", ".join(sorted(ns.shadowed)))
        else:
            val = self.eval(arg, scope)
            if not isinstance(val, tuple) or len(val) != 2:
                raise _Unsupported("add_equation argument is neither a string nor (lhs, rhs)")
            from ..ir import Equation
            eq = Equation(val[0], val[1])
        from ..ir import Equation
        lhs = _rewrite_div_outer(eq.lhs, self._notes)
        rhs = _rewrite_div_outer(eq.rhs, self._notes)
        eq = Equation(lhs, rhs)
        pb.add_equation(eq)
        self._code(st, f"problem.add_equation({to_string(eq)!r})")
        self._record(st, "translated")

    def _namespace(self, scope: dict) -> Namespace:
        vals: dict[str, Any] = {}
        for k, v in scope.items():
            if isinstance(v, (_Opaque, _Module, _D3, _ProblemDecl, _SolverDecl, _Dist, _Scheme)):
                continue
            if isinstance(v, _Bundle):
                continue
            vals[k] = v
        vals.setdefault("trace", lambda A: _trace(A, self._notes))
        return Namespace(vals, tensor_product="outer")

    def solver_call(self, st, solver: _SolverDecl, meth: str, call: ast.Call, scope) -> None:
        if meth == "step":
            self.timestep = self.eval(call.args[0], scope) if call.args else None
            return self._record(st, "translated", "fixed time step")
        if meth == "solve":
            return self._record(st, "translated", "LBVP/NLBVP solve: plan.solve(...) / "
                                                  "NewtonSolver")
        if meth in ("solve_dense", "solve_sparse"):
            self.evp_requests.append(ast.unparse(call))
            return self._record(st, "approximated",
                                "eigenvalues(plan, group, method='dense'|'sparse', k=..., "
                                "target=...): the Dedalus subproblem index is not a Conatus "
                                "group index: select the group by its wavenumbers "
                                "(plan.group(m, k))")
        if meth in ("log_stats", "print_subproblem_ranks", "evaluate_handlers_now"):
            raise _Skip("run statistics / output")
        raise _Unsupported(f"solver.{meth}")

    def time_loop(self, st: ast.While, scope: dict) -> None:
        test = ast.unparse(st.test)
        if "proceed" not in test:
            raise _Unsupported("while loop that is not the solver's time loop")
        dt_expr = None
        for n in ast.walk(st):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "step" and n.args):
                dt_expr = n.args[0]
        if dt_expr is None:
            raise _Unsupported("time loop without solver.step(dt)")
        if isinstance(dt_expr, ast.Name):          # timestep = CFL.compute_timestep() in the loop
            for n in ast.walk(st):
                if (isinstance(n, ast.Assign) and len(n.targets) == 1
                        and isinstance(n.targets[0], ast.Name) and n.targets[0].id == dt_expr.id):
                    dt_expr = n.value
                    break
        try:
            self.timestep = self.eval(dt_expr, scope)
        except _Unsupported as u:
            self.timestep = None
            self._record(st, "unsupported", f"time step {ast.unparse(dt_expr)}: {u}")
            return
        if isinstance(self.timestep, _Opaque):
            reason = self.timestep.reason
            self.timestep = None
            return self._record(st, "unsupported", f"time step {ast.unparse(dt_expr)}: {reason}")
        self._record(st, "translated", f"time loop -> solver.run(n) with a fixed dt = "
                                       f"{self.timestep!r} (no Python loop); the loop body "
                                       "(logging, analysis) is not translated")

    def field_data(self, st, tgt, op: str, scope: dict) -> None:
        f, comp = _field_data_target(tgt, self, scope)
        if not isinstance(f, Field):
            raise _Unsupported("data of an object that is not a translated field")
        d = self.data.setdefault(f, _FieldData())
        if d.unsupported:
            raise _Unsupported(f"data of {f.name} already untranslatable ({d.unsupported})")
        # a copy of the scope: later rebinding of a grid name (x = basis.global_grid(...) for
        # plots) must not change the data statement written here
        d.ops.append((op, comp, st.value, dict(scope)))
        self._record(st, "translated", f"grid data of {f.name}"
                                       + (f"[{comp}]" if comp is not None else "")
                                       + ": initial condition / given field (tr.data)")

    # problems ----------------------------------------------------------------------------------
    def make_problem(self, kind: str, variables, time=None, eigenvalue=None) -> _ProblemDecl:
        vs = list(variables)
        for v in vs:
            if not isinstance(v, Field):
                raise _Unsupported(f"problem variable {v!r} is not a translated field")
        if kind == "EVP":
            s = self._to_parameter(eigenvalue, complex)
            pb = EVP(vs, eigenvalue=s)
        elif kind == "IVP":
            pb = IVP(vs)
            if time is not None:
                self._to_parameter(time, float)
                self.note(f"time field '{time.name}' -> Parameter, "
                          f"IVPSolver(..., time_parameter='{time.name}')")
        else:
            pb = {"LBVP": LBVP, "NLBVP": NLBVP}[kind](vs)
        decl = _ProblemDecl(pb, kind, time)
        self.problems.append(decl)
        return decl

    def _to_parameter(self, f: Any, dtype) -> Parameter:
        """A 0-dimensional Dedalus field used as eigenvalue / time -> a Parameter, everywhere."""
        if isinstance(f, Parameter):
            return f
        if not isinstance(f, Field) or f.domain.bases:
            raise _Unsupported("eigenvalue / time must be a field without bases")
        p = Parameter(f.name, dtype=dtype)
        for scope in (self.g,) + tuple(self._scopes):
            for k, v in list(scope.items()):
                if v is f:
                    scope[k] = p
                elif isinstance(v, E.Expr) and f in v.free_fields():
                    scope[k] = v.substitute({f: p})
        self.note(f"0-dimensional field '{f.name}' -> Parameter('{f.name}', dtype="
                  f"{np.dtype(dtype).name})")
        return p

    _scopes: tuple = ()

    def build_solver(self, decl: _ProblemDecl, *args, **kw) -> _SolverDecl:
        """``problem.build_solver(timestepper, **options)``: the declared scheme."""
        scheme = None
        if args:
            s = args[0]
            if not isinstance(s, _Scheme):
                raise _Unsupported("time stepper is not a known Dedalus scheme")
            scheme = s.name
        for k in kw:
            if k == "entry_cutoff":
                self.note("entry_cutoff: Conatus does not zero matrix entries")
            elif k == "ncc_cutoff":
                self.approx(f"ncc_cutoff={kw[k]}: pass ncc_cutoff to discretize (Conatus "
                            "truncates NCC expansions only when declared)")
            elif k in ("matrix_coupling", "matsolver", "enforce_real_cadence",
                       "warmup_iterations"):
                self.note(f"{k}: internal Dedalus solver option, no Conatus equivalent needed")
            else:
                raise _Unsupported(f"build_solver option {k}")
        if decl.kind == "IVP" and scheme is None:
            raise _Unsupported("IVP solver without a time stepper")
        self.solver = _SolverDecl(decl, scheme, self.line)
        return self.solver

    # expressions ------------------------------------------------------------------------------
    def eval(self, node: ast.AST, scope: dict) -> Any:
        m = getattr(self, "_e_" + type(node).__name__, None)
        if m is None:
            raise _Unsupported(f"Python construct {type(node).__name__} "
                               f"('{ast.unparse(node)[:40]}')")
        return m(node, scope)

    def _e_Constant(self, n, scope):
        return n.value

    def _e_Name(self, n, scope):
        if n.id in scope:
            v = scope[n.id]
        elif n.id in self.g:
            v = self.g[n.id]
        elif n.id in ("abs", "min", "max", "float", "int", "complex", "len", "range", "round"):
            return _NpFunc("builtins", n.id)
        elif n.id in ("locals", "globals"):
            return lambda: None          # namespace=locals(): the translator's own scope is used
        else:
            raise _Unsupported(f"name '{n.id}' is not defined at this point")
        if isinstance(v, _Opaque):
            if v.skip:
                raise _Skip(f"uses '{n.id}' ({v.reason})")
            raise _Unsupported(f"'{n.id}' comes from line {v.line}: {v.reason}")
        return v

    def _e_Tuple(self, n, scope):
        return tuple(self.eval(e, scope) for e in n.elts)

    def _e_List(self, n, scope):
        return [self.eval(e, scope) for e in n.elts]

    def _e_Lambda(self, n, scope):
        return _Lambda(n, scope, self)

    def _e_UnaryOp(self, n, scope):
        v = self.eval(n.operand, scope)
        if isinstance(n.op, ast.USub):
            return -v
        if isinstance(n.op, ast.UAdd):
            return v
        if isinstance(n.op, ast.Not):
            return not v
        raise _Unsupported("unary operator")

    def _e_BinOp(self, n, scope):
        return self._binop(n.op, self.eval(n.left, scope), self.eval(n.right, scope))

    def _binop(self, op, a, b):
        if isinstance(a, _Grid) or isinstance(b, _Grid):
            raise _Unsupported("arithmetic on grids outside field data (a grid is not a "
                               "symbol; declare a Field for a coefficient)")
        _check_binop(op, a, b)
        if isinstance(op, ast.Add):
            return a + b
        if isinstance(op, ast.Sub):
            return a - b
        if isinstance(op, ast.Mult):
            if isinstance(a, E.Expr) and isinstance(b, E.Expr) and a.rank and b.rank:
                self.note(f"'{a}*{b}' of two tensors -> outer (Dedalus convention)")
                return E.outer(a, b)
            return a * b
        if isinstance(op, ast.Div):
            return a / b
        if isinstance(op, ast.Pow):
            return a ** b
        if isinstance(op, ast.MatMult):
            return E.dot(a, b)
        if isinstance(op, ast.FloorDiv):
            return a // b
        if isinstance(op, ast.Mod):
            return a % b
        raise _Unsupported("binary operator")

    def _e_Subscript(self, n, scope):
        obj = self.eval(n.value, scope)
        key = self.eval(n.slice, scope)
        if isinstance(obj, CoordinateSystem) and isinstance(key, str):
            return obj[key]
        if isinstance(obj, (tuple, list)) and isinstance(key, int):
            return obj[key]
        if isinstance(obj, dict):
            return obj[key]
        if isinstance(obj, Field) and (key in ("g", "c") or (isinstance(key, tuple) and key
                                                              and key[0] in ("g", "c"))):
            raise _Skip("reads field data (output / post-processing)")
        if isinstance(obj, E.Expr) and isinstance(key, int):
            return E.comp(obj, key)
        if isinstance(obj, _SolverDecl):
            raise _Unsupported("solver internals")
        raise _Unsupported(f"subscript of {type(obj).__name__}")

    def _e_Attribute(self, n, scope):
        obj = self.eval(n.value, scope)
        a = n.attr
        if isinstance(obj, _D3):
            return getattr(obj, a)
        if isinstance(obj, _Module):
            if obj.name == "numpy":
                if a not in _NP_ALLOWED:
                    raise _Unsupported(f"numpy.{a} is not in the translator's whitelist")
                v = getattr(np, a)
                return v if not callable(v) or isinstance(v, type) else _NpFunc("numpy", a)
            if obj.name == "scipy.special":
                if a not in _SCIPY_SPECIAL:
                    raise _Unsupported(f"scipy.special.{a} is not in the whitelist")
                return _NpFunc("scipy.special", a)
            return _Module(f"{obj.name}.{a}")        # still ignored (e.g. MPI.COMM_SELF)
        if isinstance(obj, _Bundle):
            if a in ("edge", "radial_basis", "outer_edge", "inner_edge"):
                if a in ("outer_edge", "inner_edge"):
                    return obj.edge
                return getattr(obj, a)
            if a == "derivative_basis":
                return obj.derivative_basis
            raise _Unsupported(f"{obj.kind}.{a}")
        if isinstance(obj, Basis):
            if a == "derivative_basis":
                return lambda lam=1: _LiftBasis(obj, int(lam))
            if a == "clone_with":
                raise _Unsupported("basis.clone_with")
            if a in ("global_grid", "local_grid"):
                raise _Skip("output grid")
            raise _Unsupported(f"basis attribute .{a}")
        if isinstance(obj, CoordinateSystem):
            if a == "unit_vector_fields":
                return lambda dist=None: tuple(unit_vector(c) for c in obj.coords)
            if a == "coords":
                return obj.coords
            if a == "cartesian":
                raise _Skip("output grid")
            raise _Unsupported(f"coordinate system attribute .{a}")
        if isinstance(obj, _Dist):
            return _DistMethod(self, obj, a)
        if isinstance(obj, _ProblemDecl):
            if a == "build_solver":
                return lambda *args, **kw: self.build_solver(obj, *args, **kw)
            raise _Unsupported(f"problem.{a} outside add_equation")
        if isinstance(obj, _SolverDecl):
            if a in ("sim_time", "iteration", "proceed"):
                raise _Skip("solver run-time state")
            if a in ("eigenvalues", "eigenvectors", "state", "subsystems"):
                raise _Skip("results read after the solve (post-processing)")
            if a == "evaluator":
                raise _Unsupported("analysis output (file handlers, tasks): use "
                                   "conatus.runs.track / solver.run(record=...) and the "
                                   "telemetry")
            if a in ("subproblems", "subproblems_by_group"):
                raise _Unsupported("Dedalus subproblems: select a Conatus group with "
                                   "plan.group(m, k)")
            raise _Unsupported(f"solver.{a}")
        if isinstance(obj, E.Expr):
            if a in ("allgather_data", "gather_data", "change_scales", "copy", "data"):
                raise _Skip("field data for output")
            if a == "evaluate":
                raise _Skip("evaluation for output")
            raise _Unsupported(f"expression attribute .{a}")
        if isinstance(obj, (complex, float, int, np.number)) and a in ("real", "imag"):
            return getattr(obj, a)
        raise _Unsupported(f"attribute .{a} of {type(obj).__name__}")

    def _e_Call(self, n, scope):
        f = self.eval(n.func, scope)
        args = []
        for a in n.args:
            if isinstance(a, ast.Starred):
                args.extend(self.eval(a.value, scope))
            else:
                args.append(self.eval(a, scope))
        kw = {k.arg: self.eval(k.value, scope) for k in n.keywords if k.arg is not None}
        if isinstance(f, _Module):
            raise _Skip("output only")
        if isinstance(f, _Opaque):
            raise _Unsupported(f"call of an untranslated object (line {f.line}): {f.reason}")
        if not callable(f):
            raise _Unsupported(f"call of a {type(f).__name__}")
        # every callable here was created by the translator (or given by the caller in
        # overrides): d3 names, whitelisted NumPy/SciPy, script lambdas (interpreted), methods
        return f(*args, **kw)

    def _e_JoinedStr(self, n, scope):
        raise _Skip("formatted string")

    def _e_Compare(self, n, scope):
        raise _Unsupported("comparison")

    def _e_Dict(self, n, scope):
        return {self.eval(k, scope): self.eval(v, scope) for k, v in zip(n.keys, n.values, strict=True)}

    def _e_Slice(self, n, scope):
        raise _Unsupported("slice")

    # code generation --------------------------------------------------------------------------
    def _code(self, st, text: str) -> None:
        self.code.append((st.lineno, text))


@dataclass
class _NpFunc:
    """A whitelisted numerical function: NumPy / SciPy on numbers and arrays, IR node on Exprs."""
    module: str
    name: str

    def __call__(self, *args, **kw):
        if any(isinstance(a, E.Expr) for a in args):
            if self.name in _NP_EXPR and len(args) == 1 and not kw:
                return _NP_EXPR[self.name](args[0])
            raise _Unsupported(f"{self.module}.{self.name} of a symbolic expression")
        if any(isinstance(a, _Grid) for a in args):
            raise _Unsupported("function of a grid outside field data")
        if self.module == "numpy":
            _check_array_size(self.name, args, kw)
            return getattr(np, self.name)(*args, **kw)
        if self.module == "builtins":
            return {"abs": abs, "min": min, "max": max, "float": float, "int": int,
                    "complex": complex, "len": len, "range": range, "round": round}[self.name](
                *args, **kw)
        import scipy.special as sp
        return getattr(sp, self.name)(*args, **kw)


class _DistMethod:
    """Methods of a Dedalus Distributor: fields and grids."""

    def __init__(self, tr: _Interpreter, dist: _Dist, name: str) -> None:
        self.tr, self.dist, self.name = tr, dist, name

    def _domain(self, bases) -> Domain:
        if bases is None:
            bases = ()
        if not isinstance(bases, (tuple, list)):
            bases = (bases,)
        out = []
        for b in bases:
            if isinstance(b, _Bundle):
                out += b.bases
            elif isinstance(b, Basis):
                out.append(b)
            else:
                raise _Unsupported(f"field basis {b!r}")
        return Domain(self.dist.coordsys, out)

    def __call__(self, *args, **kw):
        n = self.name
        dtype = kw.pop("dtype", self.dist.dtype)
        if n in ("Field", "VectorField", "TensorField"):
            if n == "Field" and args:
                raise _Unsupported("Field with positional arguments")
            if n != "Field":
                cs = args[0] if args else kw.pop("coordsys", None)
                if cs is not self.dist.coordsys and not (isinstance(cs, CoordinateSystem)
                                                         and cs.key == self.dist.coordsys.key):
                    raise _Unsupported("vector/tensor field on another coordinate system")
            name = kw.pop("name", None)
            D = self._domain(kw.pop("bases", None))
            tensorsig = kw.pop("tensorsig", None)
            if kw:
                raise _Unsupported(f"{n} options {sorted(kw)}")
            if tensorsig is not None:
                raise _Unsupported("explicit tensorsig")
            sig = {"Field": (), "VectorField": (self.dist.coordsys,)}.get(n)
            if sig is None:
                raise _Unsupported("TensorField")
            f = Field(name or f"_unnamed{len(self.tr.fields)}", D, sig, dtype)
            self.tr.fields[f.name] = f
            return f
        if n == "local_grid":
            b = args[0]
            if isinstance(b, _Bundle):
                raise _Unsupported("local_grid of a curvilinear basis: use local_grids")
            return _Grid(b.coord)
        if n == "local_grids":
            coords = []
            for b in args:
                coords += [x.coord for x in (b.bases if isinstance(b, _Bundle) else [b])]
            return tuple(_Grid(c) for c in coords)
        raise _Unsupported(f"dist.{n}")
