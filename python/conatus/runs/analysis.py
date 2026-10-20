"""Reading several runs: lists, tables, comparisons, lineage.

Nothing is interpolated, aligned or averaged: a comparison shows each run's own values (series keep
their own abscissae, as in the Studio's multi-run curves), and a difference is reported with both
values next to it.
"""
from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..errors import MissingDependencyError, OptionError
from ._util import diff_dicts
from .run import Run
from .store import Registry


class RunList(list):
    """A list of :class:`Run` (what ``find`` returns), with a table view."""

    def to_dataframe(self):
        return to_dataframe(self)

    def explain(self) -> str:
        if not self:
            return "no run"
        return "\n".join(f"{r.id}  {r.state:<9} {r.status:<11} "
                         + ", ".join(f"{k}={v}" for k, v in sorted(r.results.items())[:4]) for r in self)

    def _repr_html_(self) -> str:
        import html
        rows = "".join(f"<tr><td><code>{html.escape(r.id)}</code></td><td>{r.state}</td>"
                       f"<td>{r.status}</td><td>{html.escape(str(r.params))[:120]}</td>"
                       f"<td>{html.escape(str(r.results))[:120]}</td></tr>" for r in self)
        return ("<table><tr><th>id</th><th>state</th><th>status</th><th>params</th><th>results</th>"
                f"</tr>{rows}</table>")


def to_dataframe(runs: Iterable[Run]):
    """pandas DataFrame, one row per run: id, name, state, status, created, problem, then
    ``param.<k>`` and ``result.<k>`` columns (pandas is optional: imported here only)."""
    try:
        import pandas as pd
    except ImportError as err:
        raise MissingDependencyError("conatus.runs.to_dataframe needs pandas",
                                     fix="pip install pandas (or use find(...) and Run.params)") from err
    rows = []
    for r in runs:
        row = {"id": r.id, "name": r.name, "state": r.state, "status": r.status,
               "created_utc": r.rec.get("created_utc"), "duration_s": r.rec.get("duration_s"),
               "problem": (r.rec.get("problem") or {}).get("name"), "parent": r.parent,
               "tags": ",".join(r.tags)}
        row.update({f"param.{k}": v for k, v in r.params.items()})
        row.update({f"result.{k}": v for k, v in r.results.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _rel(a: Any, b: Any) -> float | None:
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) \
            or not isinstance(b, (int, float)):
        return None
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0 else abs(a - b) / scale


@dataclass
class Comparison:
    """Differences between two runs (``a`` is the reference)."""
    a: Run
    b: Run
    params: dict
    results: dict
    numerics: dict
    code: dict
    problem_same: bool
    series_common: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"a": self.a.id, "b": self.b.id, "params": self.params, "results": self.results,
                "numerics": self.numerics, "code": self.code, "problem_same": self.problem_same,
                "series_common": self.series_common}

    def explain(self) -> str:
        L = [f"compare {self.a.id} (a) vs {self.b.id} (b)",
             f"  problem: {'same structural hash' if self.problem_same else 'DIFFERENT problems'}"]
        for title, d in (("params", self.params), ("numerics", self.numerics)):
            if not (d["changed"] or d["only_a"] or d["only_b"]):
                L.append(f"  {title}: identical")
                continue
            for k, (x, y) in d["changed"].items():
                L.append(f"  {title} {k}: {x} -> {y}")
            for k, v in d["only_a"].items():
                L.append(f"  {title} {k}: {v} (only a)")
            for k, v in d["only_b"].items():
                L.append(f"  {title} {k}: {v} (only b)")
        for k, v in self.results.items():
            rel = "" if v["rel"] is None else f" (rel. diff {v['rel']:.3e})"
            L.append(f"  result {k}: a={v['a']} b={v['b']}{rel}")
        c = self.code
        L.append(f"  code: {'identical executed tree' if c['tree_same'] else 'DIFFERENT executed tree'}"
                 f"; git {str(c['git_head'][0])[:10]} -> {str(c['git_head'][1])[:10]}"
                 + (f"; dirty a={len(c['dirty'][0])} b={len(c['dirty'][1])} paths"
                    if any(c['dirty']) else ""))
        if c.get("diff_stat"):
            L.append("  git diff --stat (python/, core/):\n    " + c["diff_stat"].replace("\n", "\n    "))
        if self.series_common:
            L.append(f"  series in both: {', '.join(self.series_common)} (plot(key) overlays them)")
        return "\n".join(L)

    def plot(self, key: str, x: str | None = None, ax: Any = None):
        """Overlay series ``key`` of both runs (each on its own abscissae; matplotlib optional)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise MissingDependencyError("Comparison.plot needs matplotlib") from err
        if ax is None:
            _, ax = plt.subplots()
        for r, style in ((self.a, "-"), (self.b, "--")):
            s = r.series(key, x=x)
            ax.plot(s["x"] if x else s["row"], s["value"], style, label=r.id)
        ax.set_xlabel(x or "row")
        ax.set_ylabel(key)
        ax.legend()
        return ax


def _git_diff_stat(a: str | None, b: str | None) -> str | None:
    if not a or not b or a == b:
        return None
    # commits read from run records: hexadecimal only (a value starting with "-" would be a git
    # option, e.g. --output=<file>), and only in a checkout of Conatus (provenance._git)
    if not all(re.fullmatch(r"[0-9a-fA-F]{4,64}", c) for c in (a, b)):
        return None
    from ..provenance import REPO_ROOT, _in_git_checkout
    if not _in_git_checkout():
        return None
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "diff", "--stat", a, b, "--", "python", "core"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def compare(a: Run, b: Run) -> Comparison:
    """Parameters, numerics, results (both values + relative difference), code and series."""
    ra, rb = a.rec, b.rec
    res = {}
    for k in sorted(set(a.results) | set(b.results)):
        va, vb = a.results.get(k), b.results.get(k)
        if va != vb:
            res[k] = {"a": va, "b": vb, "rel": _rel(va, vb)}
    ca, cb = ra.get("code") or {}, rb.get("code") or {}
    code = {"git_head": [ca.get("git_head"), cb.get("git_head")],
            "tree_same": ca.get("python_tree_sha256") == cb.get("python_tree_sha256")
            and (ra.get("code_fingerprint") or {}).get("core_sha256")
            == (rb.get("code_fingerprint") or {}).get("core_sha256"),
            "dirty": [ca.get("git_dirty_paths") or [], cb.get("git_dirty_paths") or []],
            "diff_stat": _git_diff_stat(ca.get("git_head"), cb.get("git_head"))}
    pa, pb = ra.get("problem") or {}, rb.get("problem") or {}
    return Comparison(a, b, diff_dicts(a.params, b.params), res,
                      diff_dicts(ra.get("numerics") or {}, rb.get("numerics") or {}), code,
                      bool(pa) and pa.get("hash") == pb.get("hash"),
                      sorted(set(ra.get("series") or {}) & set(rb.get("series") or {})))


@dataclass
class Lineage:
    """Ancestors (parent chain, oldest first) and descendants (tree) of one run."""
    run: str
    ancestors: list
    descendants: dict
    relations: dict = field(default_factory=dict)     # child id -> "fork", "reproduce", "cache"...

    def explain(self) -> str:
        L = [f"{rid} (ancestor)" for rid in self.ancestors] + [f"{self.run} (this run)"]

        def walk(tree: dict, depth: int) -> None:
            for rid, sub in tree.items():
                L.append("  " * depth + f"└─ {rid} ({self.relations.get(rid, '?')})")
                walk(sub, depth + 1)
        walk(self.descendants, 1)
        return "\n".join(L)


def lineage(run: Any, registry: Registry) -> Lineage:
    rid = registry.resolve_id(run)
    anc, seen, cur = [], {rid}, registry.load(rid).get("parent")
    while cur and cur not in seen:
        anc.append(cur)
        seen.add(cur)
        try:
            cur = registry.load(cur).get("parent")
        except (OSError, ValueError):
            anc[-1] += " (record missing)"
            break
    anc.reverse()
    relations: dict = {}

    def tree(r: str, depth: int) -> dict:
        if depth > 1000:
            raise OptionError("conatus.runs.lineage: parent links deeper than 1000 runs")
        out = {}
        for c in registry.children(r):
            if c not in seen:                      # a cycle of parent links is cut, not followed
                seen.add(c)
                try:
                    rc = registry.load(c)
                    relations[c] = "cache reuse" if rc.get("reused_from") == r else rc.get("relation")
                except (OSError, ValueError):
                    relations[c] = "record missing"
                out[c] = tree(c, depth + 1)
        return out
    desc = tree(rid, 0)
    return Lineage(rid, anc, desc, relations)
