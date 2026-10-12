"""Rich Jupyter display of Conatus objects (v0.1 UX, "notebook first").

Every HTML string of the package is built here; the classes only carry one-line
``_repr_html_`` hooks that call these functions, so the terminal path (``__repr__``,
``explain()``) is unchanged. Rules:

* every piece of text coming from an object is escaped with :func:`html.escape`;
* the CSS is inline and scoped to ``.cn-nb`` (no external asset, no JavaScript); colours are
  translucent tints over ``currentColor`` so the output is readable in light and dark themes;
* nothing is computed that the object does not already report (no threshold is invented: a
  PASS / FAIL badge only appears where the object itself defines the status);
* IPython and ipywidgets are optional and imported lazily, only by :func:`parameter_widgets`.

Public functions: :func:`explain_html`, :func:`problem_html`, :func:`rank_report_html`,
:func:`plan_html`, :func:`group_rank_html`, :func:`evp_html`, :func:`spectrum_svg`,
:func:`verification_html`, :func:`check_html`, :func:`newton_html`, :func:`parameter_widgets`.
"""
from __future__ import annotations

import html
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

__all__ = ["explain_html", "problem_html", "rank_report_html", "plan_html", "group_rank_html",
           "evp_html", "spectrum_svg", "verification_html", "check_html", "newton_html",
           "parameter_widgets", "ParameterWidgets"]

# --------------------------------------------------------------------------- style
# Scoped CSS. Backgrounds are rgba tints so they sit on either a white or a dark notebook
# background; text keeps the theme colour (inherit). Status colours carry a word as well
# (PASS/FAIL/CHECK...), never colour alone.
_CSS = """<style>
.cn-nb{font-family:inherit;font-size:13px;line-height:1.4;color:inherit;max-width:100%;
 overflow-x:auto;margin:2px 0}
.cn-nb h4{margin:4px 0 6px;font-size:14px;font-weight:600}
.cn-nb h5{margin:10px 0 4px;font-size:12px;font-weight:600;text-transform:uppercase;
 letter-spacing:.04em;opacity:.75}
.cn-nb table{border-collapse:collapse;margin:2px 0 6px;font-size:12px}
.cn-nb th,.cn-nb td{border:1px solid rgba(128,128,128,.35);padding:2px 8px;text-align:left;
 vertical-align:top}
.cn-nb th{background:rgba(128,128,128,.12);font-weight:600}
.cn-nb td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.cn-nb code,.cn-nb pre{font-family:SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}
.cn-nb pre{margin:2px 0 6px;padding:6px 8px;background:rgba(128,128,128,.08);
 border-left:3px solid rgba(128,128,128,.4);white-space:pre-wrap;color:inherit}
.cn-nb .muted{opacity:.7}
.cn-nb .badge{display:inline-block;padding:0 6px;border-radius:3px;font-weight:600;
 font-size:11px;border:1px solid}
.cn-nb .ok{color:#1a7f37;background:rgba(26,127,55,.12);border-color:rgba(26,127,55,.5)}
.cn-nb .bad{color:#cf222e;background:rgba(207,34,46,.12);border-color:rgba(207,34,46,.5)}
.cn-nb .warn{color:#9a6700;background:rgba(191,135,0,.14);border-color:rgba(191,135,0,.55)}
.cn-nb .info{color:inherit;background:rgba(128,128,128,.12);border-color:rgba(128,128,128,.5)}
.cn-nb details summary{cursor:pointer;opacity:.8}
.cn-nb svg{display:block;margin:4px 0}
@media (prefers-color-scheme:dark){
 .cn-nb .ok{color:#4ac26b}.cn-nb .bad{color:#ff7b72}.cn-nb .warn{color:#d4a72c}}
body[data-jp-theme-light="false"] .cn-nb .ok{color:#4ac26b}
body[data-jp-theme-light="false"] .cn-nb .bad{color:#ff7b72}
body[data-jp-theme-light="false"] .cn-nb .warn{color:#d4a72c}
</style>"""

# status word -> badge class (only words the objects themselves produce)
_STATUS_CLASS = {"PASS": "ok", "CONSISTENT": "ok", "CONVERGED": "ok", "OK": "ok",
                 "FAIL": "bad", "NOT CONVERGED": "bad", "MISMATCH": "bad",
                 "CHECK": "warn"}


def _e(x: Any) -> str:
    """Escaped text of anything."""
    return html.escape(str(x), quote=True)


def _badge(status: str) -> str:
    cls = _STATUS_CLASS.get(status.upper(), "info")
    return f'<span class="badge {cls}">{_e(status)}</span>'


def _num(x: Any, fmt: str = ".3e") -> str:
    """Formatted number (plain text, escaped by _table; '-' for None)."""
    if x is None:
        return "-"
    try:
        return format(x, fmt)
    except (TypeError, ValueError):
        return str(x)


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]], *,
           raw: Iterable[int] = (), num: Iterable[int] = ()) -> str:
    """HTML table. Cells are escaped except columns listed in ``raw`` (already-built HTML);
    columns in ``num`` are right-aligned."""
    raw, num = set(raw), set(num)
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = []
    for r in rows:
        cells = []
        for j, c in enumerate(r):
            cls = ' class="num"' if j in num else ""
            cells.append(f"<td{cls}>{c if j in raw else _e(c)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _wrap(title: str, body: str) -> str:
    """Scoped container: style + title + body."""
    return f'{_CSS}<div class="cn-nb"><h4>{title}</h4>{body}</div>'


def _pre(text: str) -> str:
    return f"<pre>{_e(text)}</pre>"


def _details(summary: str, inner: str) -> str:
    return f"<details><summary>{_e(summary)}</summary>{inner}</details>"


# --------------------------------------------------------------------------- generic explain()
def explain_html(obj: Any, title: str | None = None) -> str:
    """HTML view of ``obj.explain()`` (or of a string): the first line is the title, the rest is
    a preformatted block; indentation is kept. Works for any object with ``explain()``."""
    text = obj if isinstance(obj, str) else obj.explain()
    lines = str(text).splitlines() or [""]
    head = title if title is not None else lines[0]
    rest = "\n".join(lines[1:] if title is None else lines)
    return _wrap(_e(head), _pre(rest) if rest.strip() else "")



# --------------------------------------------------------------------------- IR
def _sig(tensorsig) -> str:
    from .ir.expr import _sig_str
    return _sig_str(tensorsig)


def _problem_parameters(pb) -> list:
    """Parameters of the equations (LHS and RHS), sorted by name."""
    params: set = set()
    for eq in pb.equations:
        params |= eq.lhs.free_parameters() | eq.rhs.free_parameters()
    return sorted(params, key=lambda p: p.name)


def rank_report_html(rr, *, standalone: bool = True) -> str:
    """IR count analysis (:class:`conatus.ir.problem.RankReport`): totals, one row per
    non-periodic direction, the status the report computes (CONSISTENT / CHECK / FAIL)."""
    body = _table(["", "components"],
                  [("unknown field components", rr.variable_components),
                   ("equation components", rr.equation_components),
                   ("gauge / integral constraints", rr.gauge_components)], num=[1])
    if rr.directions:
        rows = [(_e(d.coord.name), d.order_sum, d.lifted_tau_components, d.boundary_components,
                 d.tau_variable_components, _badge("CONSISTENT" if d.consistent else "MISMATCH"))
                for d in rr.directions]
        body += _table(["direction", "sum of orders (bulk)", "lifted taus", "BC components",
                        "tau variables", "status"], rows, raw=[0, 5], num=[1, 2, 3, 4])
    note = ""
    if rr.status == "FAIL":
        note = (f"the system is not square: {rr.equation_components} equation vs "
                f"{rr.variable_components} unknown components")
    elif rr.status == "CHECK":
        note = ("counts differ in some direction: legitimate for gauges (pressure) or mixed-order "
                "systems, otherwise a missing/extra BC or tau")
    body += f"<div>status {_badge(rr.status)} <span class='muted'>{_e(note)}</span></div>"
    body += ("<div class='muted'>IR level count; the discrete rank is checked by "
             "discretize() / plan.check()</div>")
    return _wrap("System count analysis (IR)", body) if standalone else body


def problem_html(pb) -> str:
    """Problem card: variables, parameters, equations (LHS | RHS), taus, rank count."""
    title = f"{_e(pb.kind)} problem <code>{_e(pb.name)}</code>"
    parts = ["<h5>Variables</h5>",
             _table(["name", "tensor", "domain"],
                    [(f"<code>{_e(v.name)}</code>", _sig(v.tensorsig), repr(v.domain))
                     for v in pb.variables], raw=[0])]
    params = _problem_parameters(pb)
    eig = getattr(pb, "eigenvalue", None)
    if params or eig is not None:
        rows = [(f"<code>{_e(p.name)}</code>", "-" if p.value is None else p.value,
                 str(np.dtype(p.dtype)) if getattr(p, "dtype", None) is not None else "-",
                 "eigenvalue" if (eig is not None and p is eig) else "")
                for p in params]
        if eig is not None and all(p is not eig for p in params):
            rows.append((f"<code>{_e(eig.name)}</code>", "-", str(np.dtype(eig.dtype)),
                         "eigenvalue (not in any equation yet)"))
        parts += ["<h5>Parameters</h5>", _table(["name", "value", "dtype", "role"], rows, raw=[0])]
    ncc = pb.ncc_fields()
    if ncc:
        parts.append("<div>Non-constant coefficients (LHS, fixed): "
                     + ", ".join(f"<code>{_e(f.name)}</code>" for f in sorted(ncc, key=lambda f: f.name))
                     + "</div>")
    bc = getattr(pb, "_bc_index", set())
    rows = [(i, "boundary" if i in bc else "", f"<code>{_e(eq.lhs)}</code>",
             f"<code>{_e(eq.rhs)}</code>") for i, eq in enumerate(pb.equations)]
    parts += ["<h5>Equations <span class='muted'>(LHS implicit / linear | RHS explicit)</span></h5>",
              _table(["#", "kind", "LHS", "RHS"], rows, raw=[2, 3], num=[0])
              if rows else "<div class='muted'>no equation yet</div>"]
    taus = getattr(pb, "_tau_log", [])
    if taus:
        parts += ["<h5>Tau terms (declared)</h5>", _pre("\n".join(taus))]
    from .ir.expr import has_nonpolynomial
    nonpoly = sorted({f for eq in pb.equations
                      for f in has_nonpolynomial(eq.lhs) + has_nonpolynomial(eq.rhs)})
    if nonpoly:
        parts.append(f"<div>{_badge('CHECK')} non-polynomial pointwise functions "
                     f"{_e(', '.join(nonpoly))}: pseudospectral evaluation is not alias-free for any "
                     "dealias factor (check convergence in N)</div>")
    extra = pb._extra_explain()
    if extra:
        parts.append(_pre(extra))
    parts.append("<h5>Rank count</h5>")
    try:
        parts.append(rank_report_html(pb.rank_report(), standalone=False))
    except Exception as err:  # noqa: BLE001 -- display only: say why, never hide it
        parts.append(f"<div>{_badge('FAIL')} rank report unavailable: {_e(err)}</div>")
    if pb.symmetry is not None:
        parts += ["<h5>Symmetry</h5>", _pre(pb.symmetry_report())]
    return _wrap(title, "".join(parts))


# --------------------------------------------------------------------------- discrete plan
def group_rank_html(gr, *, standalone: bool = True) -> str:
    """Discrete rank of one group (:class:`conatus.discretize.plan.GroupRank`)."""
    body = _table(_GROUP_HEAD, [_group_row(gr)], raw=[0, 10], num=range(1, 8))
    if gr.null_hint:
        body += f"<div class='muted'>null space: {_e(gr.null_hint)}</div>"
    return _wrap("System rank analysis", body) if standalone else body


_GROUP_HEAD = ["group", "field DOFs", "tau DOFs", "gauge DOFs", "PDE eqs", "boundary eqs",
               "constraint eqs", "structural rank", "numerical rank", "unknowns / equations",
               "status"]


def _group_row(gr) -> tuple:
    u, q = gr.unknowns, gr.equations
    return (f"<code>{_e(gr.label)}</code>", u.get("field", 0), u.get("tau", 0),
            u.get("constant", 0), q.get("bulk", 0), q.get("boundary", 0), q.get("constraint", 0),
            f"{gr.structural_rank} / {gr.n_unknowns}", gr.numerical,
            f"{gr.n_unknowns} / {gr.n_equations}", _badge("PASS" if gr.ok else "FAIL"))


def plan_html(plan, max_groups: int = 4) -> str:
    """Plan card: explain() text, IR count, and the discrete rank of the first ``max_groups``
    groups (the table says how many of how many; ``plan.check()`` checks every group).
    Plans without ``rank_report`` (polar, sphere) show their explain() only."""
    if not hasattr(plan, "rank_report") or not hasattr(plan, "groups"):
        return explain_html(plan)
    pb = plan.problem
    title = f"Discretization plan for {_e(pb.kind)} <code>{_e(pb.name)}</code>"
    parts = []
    groups = list(plan.groups)
    shown = groups[:max_groups]
    rows, errors = [], []
    for g in shown:
        try:
            rows.append(_group_row(plan.rank_report(g)))
        except Exception as err:  # noqa: BLE001 -- display only: the error is printed in the card
            errors.append(f"group {g}: {err}")
    parts.append(f"<h5>Discrete rank per group <span class='muted'>({len(shown)} of "
                 f"{len(groups)} group(s) shown; plan.check() checks all)</span></h5>")
    parts.append(_table(_GROUP_HEAD, rows, raw=[0, 10], num=range(1, 8)))
    for m in errors:
        parts.append(f"<div>{_badge('FAIL')} {_e(m)}</div>")
    params = getattr(plan, "params", None)
    if params:
        parts += ["<h5>Parameter values of this plan</h5>",
                  _table(["name", "value"], [(f"<code>{_e(k)}</code>", v)
                                             for k, v in sorted(params.items())], raw=[0])]
    parts.append("<h5>Direction counts (IR)</h5>")
    try:
        parts.append(rank_report_html(pb.rank_report(), standalone=False))
    except Exception as err:  # noqa: BLE001 -- display only, the error is shown
        parts.append(f"<div>{_badge('FAIL')} IR count unavailable: {_e(err)}</div>")
    try:
        parts.append(_details("plan.explain()", _pre(plan.explain())))
    except Exception as err:  # noqa: BLE001 -- display only, the error is shown
        parts.append(f"<div>{_badge('FAIL')} explain() failed: {_e(err)}</div>")
    return _wrap(title, "".join(parts))


# --------------------------------------------------------------------------- EVP
def _nice(v: float) -> str:
    return f"{v:.3g}"


def spectrum_svg(values: np.ndarray, highlight: Sequence[int] = (), *, width: int = 360,
                 height: int = 240) -> str:
    """Inline SVG scatter of eigenvalues, Re sigma (x) against Im sigma (y). Non-finite values
    are not plotted (the caption counts them). ``highlight``: indices drawn as leading modes.
    Each dot has a <title> tooltip with its value."""
    v = np.asarray(values, dtype=complex).ravel()
    fin = np.isfinite(v)
    pts = v[fin]
    idx = np.nonzero(fin)[0]
    ml, mr, mt, mb = 56, 12, 10, 34               # margins (room for tick labels)
    pw, ph = width - ml - mr, height - mt - mb
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" role="img" aria-label="eigenvalue spectrum" '
           'style="font:10px sans-serif;color:inherit">']
    if pts.size == 0:
        out.append(f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" '
                   'fill="currentColor">no finite eigenvalue</text></svg>')
        return "".join(out)
    xr, yi = pts.real, pts.imag

    def span(a: np.ndarray) -> tuple[float, float]:
        lo, hi = float(a.min()), float(a.max())
        if hi == lo:                              # one value: open a window around it
            d = abs(lo) * 0.1 or 1.0
            return lo - d, hi + d
        d = 0.06 * (hi - lo)
        return lo - d, hi + d
    x0, x1 = span(xr)
    y0, y1 = span(yi)
    X = lambda x: ml + (x - x0) / (x1 - x0) * pw          # noqa: E731
    Y = lambda y: mt + (y1 - y) / (y1 - y0) * ph          # noqa: E731
    grid = 'stroke="currentColor" stroke-opacity="0.25" stroke-width="1"'
    out.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" {grid}/>')
    if x0 < 0 < x1:                               # Re sigma = 0: neutral line
        out.append(f'<line x1="{X(0):.1f}" y1="{mt}" x2="{X(0):.1f}" y2="{mt + ph}" '
                   'stroke="currentColor" stroke-opacity="0.45" stroke-dasharray="3,3"/>')
    if y0 < 0 < y1:
        out.append(f'<line x1="{ml}" y1="{Y(0):.1f}" x2="{ml + pw}" y2="{Y(0):.1f}" '
                   'stroke="currentColor" stroke-opacity="0.45" stroke-dasharray="3,3"/>')
    for t in (x0, (x0 + x1) / 2, x1):             # three ticks per axis, labels in text ink
        out.append(f'<text x="{X(t):.1f}" y="{mt + ph + 13}" text-anchor="middle" '
                   f'fill="currentColor" fill-opacity="0.75">{_e(_nice(t))}</text>')
    for t in (y0, (y0 + y1) / 2, y1):
        out.append(f'<text x="{ml - 4}" y="{Y(t) + 3:.1f}" text-anchor="end" '
                   f'fill="currentColor" fill-opacity="0.75">{_e(_nice(t))}</text>')
    out.append(f'<text x="{ml + pw / 2}" y="{height - 4}" text-anchor="middle" '
               'fill="currentColor">Re σ</text>')
    out.append(f'<text x="12" y="{mt + ph / 2}" text-anchor="middle" fill="currentColor" '
               f'transform="rotate(-90 12 {mt + ph / 2})">Im σ</text>')
    hl = set(int(i) for i in highlight)
    # other modes first (muted), leading modes on top with a surface-coloured ring
    for i, s in zip(idx, pts, strict=True):
        if int(i) in hl:
            continue
        out.append(f'<circle cx="{X(s.real):.1f}" cy="{Y(s.imag):.1f}" r="2.5" fill="#6e8fbf" '
                   f'fill-opacity="0.65"><title>{_e(f"{s:.6g}")}</title></circle>')
    for i, s in zip(idx, pts, strict=True):
        if int(i) not in hl:
            continue
        out.append(f'<circle cx="{X(s.real):.1f}" cy="{Y(s.imag):.1f}" r="4.5" fill="#2a6fdb" '
                   f'stroke="white" stroke-opacity="0.8" stroke-width="1.5">'
                   f'<title>{_e(f"mode {i}: {s:.10g}")}</title></circle>')
    out.append("</svg>")
    return "".join(out)


def evp_html(res, n: int = 10, key: str = "growth") -> str:
    """EVP result card: counts, log, table of the ``n`` leading modes (order ``key``, as in
    ``explain``) with their residuals (and drift ratio if a drift test ran), spectrum SVG."""
    title = f"EVP result <code>{_e(res.label)}</code>" if res.label else "EVP result"
    parts = [f"<div>{_e(res.method)}: <b>{res.values.size}</b> finite mode(s), "
             f"<b>{res.n_infinite}</b> infinite (M singular on tau/boundary rows)</div>"]
    if res.log:
        parts.append("<ul>" + "".join(f"<li class='muted'>{_e(s)}</li>" for s in res.log) + "</ul>")
    order = res.order(key)[:n] if res.values.size else np.array([], dtype=int)
    head = ["#", "mode", "Re σ", "Im σ", "residual"] + (["drift"] if res.drift is not None else [])
    rows = []
    for r, i in enumerate(order):
        s = res.values[i]
        row = [r, int(i), _num(s.real, ".12e"), _num(s.imag, ".12e"), _num(res.residuals[i], ".1e")]
        if res.drift is not None:
            row.append(_num(res.drift[i], ".1e"))
        rows.append(row)
    parts.append(f"<h5>Leading modes <span class='muted'>(by {_e(key)}, {len(order)} of "
                 f"{res.values.size})</span></h5>")
    parts.append(_table(head, rows, num=range(len(head))))
    nonfin = int((~np.isfinite(res.values)).sum())
    cap = f"spectrum: {res.values.size - nonfin} finite value(s) plotted"
    cap += f", {nonfin} non-finite not plotted" if nonfin else ""
    parts += ["<h5>Spectrum</h5>", spectrum_svg(res.values, order),
              f"<div class='muted'>{_e(cap)}; large dots = leading modes of the table</div>"]
    return _wrap(title, "".join(parts))


# --------------------------------------------------------------------------- verification
def _check_table(checks) -> str:
    """Rows: status badge, name, value, threshold, what (the Check's own status)."""
    rows = [(_badge(c.status), c.name, _num(c.value, ".2e"), _num(c.threshold, ".0e"), c.what)
            for c in checks]
    return _table(["status", "check", "value", "threshold", "what"], rows, raw=[0], num=[2, 3])


def check_html(c) -> str:
    """One :class:`conatus.verify.report.Check` as a one-row table."""
    return _wrap("Check", _check_table([c]))


def verification_html(rep) -> str:
    """Verification report: one row per check with value, threshold and PASS / FAIL."""
    n, k = len(rep.checks), rep.manual_steps
    body = (f"<div>{n} check(s), {_badge('PASS' if k == 0 else 'FAIL')} "
            f"{k} to investigate by hand</div>" + _check_table(rep.checks))
    return _wrap("Verification report", body)


# --------------------------------------------------------------------------- Newton (NLBVP)
def newton_html(sol) -> str:
    """Newton solution (:class:`conatus.solvers.nlbvp.NLBVPSolution`): status, settings,
    residual history, observed orders."""
    st = "CONVERGED" if sol.converged else "NOT CONVERGED"
    title = f"Newton on NLBVP <code>{_e(sol.plan.problem.name)}</code>"
    parts = [f"<div>{_badge(st)} in {len(sol.iterations) - 1} iteration(s)</div>"]
    if sol.settings:
        parts.append(_table(["setting", "value"], sorted(sol.settings.items())))
    rows = [(it.k, _num(it.residual), _num(it.rel_residual), _num(it.step)) for it in sol.iterations]
    parts += ["<h5>Iterations</h5>", _table(["k", "|F|", "|F| / scale", "|dx|"], rows, num=range(4))]
    q = sol.observed_orders()
    if q:
        parts.append("<div>observed orders: " + _e(", ".join(f"{x:.2f}" for x in q)) + "</div>")
    return _wrap(title, "".join(parts))


# --------------------------------------------------------------------------- widgets
class ParameterWidgets:
    """Parameter editor built by :func:`parameter_widgets`. ``box`` is the ipywidgets container;
    ``values()`` returns the current {name: value}. Nothing in the IR or the plan is changed: the
    values only reach ``on_change`` (called at each user edit) or the caller through
    ``values()``."""

    def __init__(self, box: Any, fields: dict, kinds: dict, status: Any) -> None:
        self.box, self._fields, self._kinds, self._status = box, fields, kinds, status

    def values(self) -> dict[str, Any]:
        """Current values; a text that does not parse raises ParameterError (never guessed)."""
        from .errors import ParameterError
        out = {}
        for name, w in self._fields.items():
            raw = w.value
            if self._kinds[name] == "float":
                out[name] = float(raw)
                continue
            try:
                out[name] = complex(str(raw).replace(" ", "")) if self._kinds[name] == "complex" \
                    else float(raw)
            except ValueError as err:
                raise ParameterError(f"parameter '{name}': cannot read {raw!r} as a number",
                                     where="conatus.notebook.parameter_widgets",
                                     fix="type a number (complex: 1+2j)") from err
        return out

    def _ipython_display_(self) -> None:          # displays the widget box in a notebook
        from IPython.display import display  # present whenever ipywidgets is
        display(self.box)


def parameter_widgets(problem_or_plan: Any, on_change: Callable[[dict], Any] | None = None
                      ) -> ParameterWidgets | None:
    """Text boxes for the Parameters of a problem (or of ``plan.problem``), if ipywidgets exists.

    Returns a :class:`ParameterWidgets` (display it in a cell), or **None when ipywidgets is not
    importable**, in which case a ``UserWarning`` says so (ipywidgets is optional and never
    installed by Conatus). Initial values: the plan's ``params`` if given, else
    ``Parameter.value``; a parameter without value starts empty. The EVP eigenvalue is excluded.

    Nothing is mutated: IR objects are immutable and the plan is frozen. At every edit the whole
    dict {name: value} is passed to ``on_change`` (e.g. to rebuild a plan with
    ``discretize(pb, params=values)``); ``.values()`` returns the same dict on demand. Plain
    text boxes are used, not sliders: a slider range would be an invented bound.
    """
    try:
        import ipywidgets as W
    except ImportError:
        warnings.warn("conatus.notebook.parameter_widgets: ipywidgets is not installed, no widget "
                      "built (returns None). Pass parameter values explicitly "
                      "(discretize(pb, params={...})) or install ipywidgets (pip install \".[notebook]\" "
                      "from the source tree).",
                      UserWarning, stacklevel=2)
        return None
    pb = getattr(problem_or_plan, "problem", problem_or_plan)
    given = dict(getattr(problem_or_plan, "params", None) or {})
    eig = getattr(pb, "eigenvalue", None)
    fields, kinds = {}, {}
    for p in _problem_parameters(pb):
        if eig is not None and p is eig:
            continue
        v = given.get(p.name, p.value)
        cplx = np.dtype(p.dtype).kind == "c"
        if v is not None and not cplx and np.isrealobj(v):
            w = W.FloatText(value=float(v), description=p.name)
            kinds[p.name] = "float"
        else:
            w = W.Text(value="" if v is None else str(v), description=p.name,
                       placeholder="complex, e.g. 1+2j" if cplx else "number")
            kinds[p.name] = "complex" if cplx else "text"
        fields[p.name] = w
    status = W.HTML(value="" if eig is None else
                    f"<span>eigenvalue <code>{_e(eig.name)}</code> is solved for, not edited</span>")
    box = W.VBox(list(fields.values()) + [status])
    pw = ParameterWidgets(box, fields, kinds, status)

    def _changed(_change: Any) -> None:
        try:
            vals = pw.values()
        except Exception as err:  # noqa: BLE001 -- shown in the box, on_change not called
            status.value = f"<span style='color:#cf222e'>{_e(err).splitlines()[0]}</span>"
            return
        status.value = ""
        if on_change is not None:
            on_change(vals)

    for w in fields.values():
        w.observe(_changed, names="value")
    return pw

