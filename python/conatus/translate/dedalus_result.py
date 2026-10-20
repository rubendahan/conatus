"""Dedalus translation, part 4: the result, :class:`DedalusTranslation`.

The translated problem, declared time stepping, grid functions of the initial / given data (the data
statements replayed with NumPy grids by ``_GridEval``), the report, ``explain()``, ``_repr_html_()``
and ``to_script()``. Public face and entry point: ``conatus.translate.dedalus.from_dedalus``.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import UnsupportedError
from ..ir import (Field)
from .dedalus_interpreter import _NpFunc
from .dedalus_objects import (STATUSES, ReportLine, _field_data_target, _FieldData, _Grid,
                              _is_field_data, _is_num, _Opaque)

if TYPE_CHECKING:
    from .dedalus_interpreter import _Interpreter


# --------------------------------------------------------------------------- result


@dataclass
class DedalusTranslation:
    """Result of :func:`from_dedalus`: the Conatus problem, declared settings, data, report."""

    source_name: str
    problem: Any
    kind: str | None
    scheme: str | None
    timestep: Any
    stop_sim_time: Any
    stop_iteration: Any
    time_parameter: str | None
    data: dict
    report: list[ReportLine]
    namespace: dict
    evp_requests: list[str]
    code: list[tuple[int, str]]
    # the translated source and the call's arguments: to_script() replays them
    source: str = ""
    overrides: dict | None = None
    entry: tuple | None = None

    # counts -----------------------------------------------------------------------------------
    def counts(self) -> dict[str, int]:
        return {s: sum(1 for r in self.report if r.status == s) for s in STATUSES}

    @property
    def complete(self) -> bool:
        """True when a problem was translated with no known semantic loss: no line unsupported
        and none approximated (an approximated line is a declared difference of method, e.g.
        a dropped Nyquist mode or another tau projection, finding 6)."""
        c = self.counts()
        return self.problem is not None and c["unsupported"] == 0 and c["approximated"] == 0

    def lines(self, status: str) -> list[ReportLine]:
        return [r for r in self.report if r.status == status]

    # use --------------------------------------------------------------------------------------
    def numbers(self) -> dict:
        """The script's numeric names (kappa, nu, Lz, ...): a namespace for string equations."""
        return {k: v for k, v in self.namespace.items() if _is_num(v)}

    def ncc(self) -> dict:
        """Grid functions of the non-variable fields that have translated data (NCC / RHS data)."""
        pv = set(self.problem.variables) if self.problem is not None else set()
        return {f: g for f, g in self.data.items() if f not in pv and g is not None}

    def initial_state(self) -> dict:
        """Grid functions of the problem variables with translated data (for set_state)."""
        pv = set(self.problem.variables) if self.problem is not None else set()
        return {f: g for f, g in self.data.items() if f in pv and g is not None}

    def discretize(self, **kw):
        """``discretize(problem, ncc=<translated data of the coefficient fields>, **kw)``."""
        from ..discretize import discretize
        if self.problem is None:
            raise UnsupportedError("no problem was translated", fix="see explain()")
        ncc = self.ncc()
        ncc.update(kw.pop("ncc", {}) or {})
        return discretize(self.problem, ncc=ncc or None, **kw)

    def ivp_solver(self, plan=None, *, dt: float | None = None, scheme: str | None = None,
                   **kw):
        """IVPSolver with the scheme and fixed dt declared by the script (explicit arguments win,
        and are required when the script's dt was not translatable, e.g. an adaptive CFL).

        Time-dependent boundary data follow the Dedalus convention, ``bc_time="step"``: the translation reproduces the script, and the Conatus
        default (``"auto"`` = ``"rk"``: the order of the scheme kept) would change the
        digits of every scheme that loses order with moving boundary data. Pass
        ``bc_time="auto"`` (or ``"stage"``) to get a Conatus convention; ``explain()`` states
        which one is used."""
        from ..timestep import IVPSolver
        scheme = scheme or self.scheme
        dt = dt if dt is not None else self.timestep
        if scheme is None or dt is None:
            raise UnsupportedError(
                "the script does not declare a translatable scheme and fixed dt",
                why="Conatus never picks a time stepper or a time step for you",
                fix="tr.ivp_solver(dt=..., scheme='RK222')")
        plan = plan if plan is not None else self.discretize()
        if self.time_parameter:
            kw.setdefault("time_parameter", self.time_parameter)
        kw.setdefault("bc_time", "step")           # Dedalus convention (docstring)
        s = IVPSolver(plan, scheme, float(dt), **kw)
        init = self.initial_state()
        if init:
            s.set_state(init)
        return s

    def steps(self) -> int | None:
        """Number of steps: stop_iteration, or stop_sim_time / dt when it is an integer."""
        if self.stop_iteration is not None:
            return int(self.stop_iteration)
        if self.stop_sim_time is None or not self.timestep:
            return None
        n = self.stop_sim_time / self.timestep
        return int(round(n)) if abs(n - round(n)) < 1e-9 * max(1, n) else None

    # display ----------------------------------------------------------------------------------
    def explain(self) -> str:
        c = self.counts()
        lines = [f"Dedalus -> Conatus: {self.source_name}",
                 f"  {c['translated']} translated, {c['approximated']} approximated, "
                 f"{c['unsupported']} unsupported, {c['skipped']} skipped (output only)"]
        if self.problem is not None:
            lines.append(f"  problem: {self.kind} with {len(self.problem.equations)} equations, "
                         f"variables {', '.join(v.name for v in self.problem.variables)}")
        if self.scheme:
            lines.append(f"  time stepping: {self.scheme}, dt = {self.timestep!r}, "
                         f"stop_sim_time = {self.stop_sim_time!r}; ivp_solver() imposes "
                         "time-dependent boundary data as Dedalus does (bc_time='step'; "
                         "bc_time='auto' for the Conatus default, which keeps the order of "
                         "the scheme)")
        if self.data:
            lines.append("  data: " + ", ".join(
                f"{f.name} ({'translated' if g is not None else 'untranslatable'})"
                for f, g in self.data.items()))
        lines.append("Line-by-line report:")
        lines += [str(r) for r in self.report]
        return "\n".join(lines)

    def __repr__(self) -> str:
        c = self.counts()
        return (f"<DedalusTranslation {self.source_name}: {c['translated']} translated, "
                f"{c['approximated']} approximated, {c['unsupported']} unsupported>")

    def _repr_html_(self) -> str:
        import html
        colors = {"translated": "#2e7d32", "approximated": "#b26a00", "unsupported": "#c62828",
                  "skipped": "#757575"}
        c = self.counts()
        rows = "".join(
            f"<tr><td style='text-align:right'>{r.lineno}</td>"
            f"<td style='color:{colors[r.status]};font-weight:600'>{r.status}</td>"
            f"<td><code>{html.escape(r.source.strip().splitlines()[0][:90] if r.source.strip() else '')}"
            f"</code></td><td>{html.escape(r.note)}</td></tr>" for r in self.report)
        head = (f"<b>Dedalus &rarr; Conatus</b>: {html.escape(self.source_name)} &mdash; "
                + ", ".join(f"{c[s]} {s}" for s in STATUSES))
        return (f"<div>{head}<table style='border-collapse:collapse;font-size:90%'>"
                f"<tr><th>line</th><th>status</th><th>Dedalus</th><th>note</th></tr>{rows}"
                "</table></div>")

    def to_script(self) -> str:
        """A runnable Conatus program for this translation (finding 55: it
        used to be a skeleton that did not run).

        The program embeds the Dedalus source, replays ``conatus.from_dedalus`` on it (with the
        same ``overrides`` / ``entry``; nothing of the script is executed by Python), refuses to
        run if any line is unsupported, and then: IVP -> ``tr.ivp_solver()`` and
        ``solver.run(tr.steps())``; LBVP -> ``tr.discretize().solve()`` with its residual;
        EVP / NLBVP -> the discretized plan (choosing the eigenvalue group, or the Newton initial
        guess, is the user's decision, stated in the program). The Conatus form of each
        translated statement and the line-by-line report follow as comments, for reading."""
        import pprint
        for what, val in (("overrides", self.overrides), ("entry", self.entry)):
            if val is None:
                continue
            try:
                same = ast.literal_eval(repr(val)) == val
            except (ValueError, SyntaxError):
                same = False
            if not same:
                raise UnsupportedError(
                    f"to_script(): the {what} {val!r} have no literal form a program can hold",
                    fix=f"pass {what} made of numbers, strings, tuples, lists and dicts")
        call = "SOURCE"
        if self.overrides:
            call += f", overrides={self.overrides!r}"
        if self.entry is not None:
            call += f", entry={self.entry!r}"
        out = [f'"""Conatus program translated from {self.source_name} (conatus.from_dedalus).',
               "",
               "Runnable: the Dedalus source below is translated again (never executed) and the",
               "translated problem is solved with the settings the script declares. The program",
               "stops with UnsupportedError if a line of the script is not translated. The",
               "commented block at the end shows the Conatus form of each statement.",
               '"""',
               "import conatus as cn",
               "from conatus.errors import UnsupportedError",
               "",
               "SOURCE = " + pprint.pformat(self.source, width=96),
               "",
               f"tr = cn.from_dedalus({call})",
               "print(tr.explain())",
               "bad = tr.lines('unsupported')",
               "if bad:",
               "    raise UnsupportedError(",
               "        f'{len(bad)} line(s) of the Dedalus script are not translated',",
               "        where='; '.join(f'line {r.lineno}: {r.note}' for r in bad),",
               "        fix='adapt these lines (tr.explain() gives the reason of each)')",
               ""]
        if self.kind == "IVP":
            out += ["# time-dependent boundary data imposed as Dedalus does (bc_time='step'); "
                    "pass bc_time='auto' for the Conatus default",
                    "solver = tr.ivp_solver()",
                    "steps = tr.steps()",
                    "if steps is None:",
                    "    raise UnsupportedError('the script declares no translatable number of "
                    "steps', fix='call solver.run(n) with your n')",
                    "solver.run(steps)",
                    "print(f'IVP: {steps} steps, t = {solver.t}')"]
        elif self.kind == "LBVP":
            out += ["solution = tr.discretize().solve()",
                    "print(f'LBVP solved: max residual {solution.max_residual:.3e}')"]
        elif self.kind in ("EVP", "NLBVP"):
            nxt = ("eigenvalues(plan, plan.group(...), ...) on the group of your choice"
                   if self.kind == "EVP" else
                   "a Newton solve from an initial guess of your choice")
            out += ["plan = tr.discretize()",
                    f"print('{self.kind} discretized: next step is {nxt} (the program stops "
                    "here: Conatus never chooses it for you)')"]
        out += ["", "# " + "-" * 94,
                "# Translation, line by line (Conatus form of each statement; not executed):", ""]
        by_line: dict[int, list[str]] = {}
        for ln, text in self.code:
            by_line.setdefault(ln, []).append(text)
        for r in self.report:
            first = r.source.strip().splitlines()[0] if r.source.strip() else ""
            out.append(f"# [{r.lineno}] {r.status.upper()}: {first}")
            if r.note:
                out.append(f"#     {r.note}")
            if r.status in ("translated", "approximated"):
                out += [f"#   -> {t}" for t in by_line.get(r.lineno, [])]
        return "\n".join(out) + "\n"

def _data_function(fd: _FieldData, field: Field, tr: _Interpreter, all_data: dict):
    """Grid function (one array per coordinate of the field's coordinate system, as Conatus
    passes them to set_state / ncc) that replays the script's data statements with NumPy, or
    None if untranslatable."""
    if fd.unsupported or not fd.ops:
        return None
    coords = field.domain.coordsys.coords

    def f(*grids):
        arrays = {c: g for c, g in zip(coords, grids, strict=True)}
        shape = np.broadcast(*[np.asarray(g) for g in grids]).shape if grids else ()
        dims = [c for c in coords if field.domain.has(c)]
        # a field without some coordinates is constant along them: evaluate on the full grid
        ncomp = field.tensorsig[0].dim if field.rank else None
        val = (np.zeros((ncomp,) + shape, dtype=field.dtype) if ncomp
               else np.zeros(shape, dtype=field.dtype))
        for op, comp, node, scope in fd.ops:
            local = dict(scope)
            for gname, c in tr.grids.items():
                if isinstance(local.get(gname), _Grid):
                    local[gname] = arrays[c]
            rhs = _GridEval(tr, local, all_data, grids).eval(node)
            tgt = val if comp is None else val[comp]
            rhs = np.broadcast_to(rhs, tgt.shape) if np.ndim(rhs) else rhs
            if op == "=":
                tgt[...] = rhs
            elif op == "*=":
                tgt *= rhs
            elif op == "+=":
                tgt += rhs
            elif op == "-=":
                tgt -= rhs
            elif op == "/=":
                tgt /= rhs
        del dims
        return val
    return f


class _GridEval:
    """Evaluates a data statement with NumPy arrays for the grids (initial / given data)."""

    def __init__(self, tr: _Interpreter, scope: dict, all_data: dict, grids: tuple) -> None:
        self.tr, self.scope, self.all_data, self.grids = tr, scope, all_data, grids

    def eval(self, node):
        if isinstance(node, ast.Subscript) and _is_field_data(node):
            f, comp = _field_data_target(node, self.tr, self.scope)
            g = self.all_data.get(f)
            if g is None:
                raise UnsupportedError(f"data of {getattr(f, 'name', f)} is read but was not "
                                       "translated")
            v = g(*self.grids)
            return v if comp is None else v[comp]
        if isinstance(node, ast.Name):
            v = self.scope.get(node.id, self.tr.g.get(node.id))
            if isinstance(v, (_Opaque,)):
                raise UnsupportedError(f"'{node.id}': {v.reason}")
            return v
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            a, b = self.eval(node.left), self.eval(node.right)
            return {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
                    ast.Div: np.divide, ast.Pow: np.power}[type(node.op)](a, b)
        if isinstance(node, ast.UnaryOp):
            v = self.eval(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.Call):
            f = self.tr.eval(node.func, self.scope)
            if not isinstance(f, _NpFunc):
                raise UnsupportedError(f"call of {ast.unparse(node.func)} in field data")
            # positional and keyword arguments, starred ones expanded: keywords used to be
            # dropped (finding 53: np.linspace(0, 1, num=8) gave 50 points)
            args: list = []
            for a in node.args:
                if isinstance(a, ast.Starred):
                    args.extend(self.eval(a.value))
                else:
                    args.append(self.eval(a))
            kw: dict = {}
            for k in node.keywords:
                val = self.eval(k.value)
                if k.arg is None:
                    if not (isinstance(val, dict) and all(isinstance(x, str) for x in val)):
                        raise UnsupportedError(f"'**{ast.unparse(k.value)}' in field data is not "
                                               "a mapping of argument names")
                    kw.update(val)
                else:
                    kw[k.arg] = val
            return f(*args, **kw)
        if isinstance(node, ast.Attribute):
            return self.tr.eval(node, self.scope)
        if isinstance(node, ast.Tuple):
            return tuple(self.eval(e) for e in node.elts)
        raise UnsupportedError(f"construct {type(node).__name__} in field data")
