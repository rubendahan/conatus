"""Registry storage: one directory per run (the truth) + a SQLite index (derived, rebuildable).

Layout under the declared root (``$CN_RUNS``, else ``$CN_TMP/runs``, else ``<repo>/scratch/runs``;
never the home directory by default)::

    <root>/index.sqlite            index for queries; rebuilt from the run directories at any time
    <root>/runs/<id>/run.json      the record (schema conatus.runs/1), replaced atomically
    <root>/runs/<id>/data.h5       series and fields (HDF5), renamed into place when complete
    <root>/runs/<id>/state_*.npz   solver checkpoints (start / end of an IVP run), if recorded
    <root>/runs/<id>/files/        files attached by the user (copies)
    <root>/runs/<id>/.lock         flock for read-modify-write of run.json (status changes)

Why the directory is the truth: several jobs on several nodes write concurrently to a shared
file system (CephFS). Each run only ever writes its own directory (no contention); the index is a
cache that every reader reconciles with the directories (:meth:`Registry.sync`), so a failed or
contended index write loses nothing. SQLite runs in rollback-journal mode (WAL needs shared memory,
unsafe across nodes) with a busy timeout; if the index cannot be opened at all (read-only root), an
in-memory index is built from the run directories.
"""
from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..errors import OptionError, did_you_mean
from ._util import FileLock, atomic_write_json

SCHEMA = "conatus.runs/1"
STATUSES = ("provisional", "certified", "retracted")
STATES = ("running", "completed", "failed")
#: seconds SQLite waits for a lock held by another process before giving up (the writer thread
#: waits, never the computation; a give-up only delays the index, see Registry.sync)
BUSY_TIMEOUT_S = 60.0

# Index columns read from the record (name -> function of the record). Filterable by find().
_COLUMNS: dict[str, Any] = {
    "name": lambda r: r.get("name"),
    "created_utc": lambda r: r.get("created_utc"),
    "ended_utc": lambda r: r.get("ended_utc"),
    "duration_s": lambda r: r.get("duration_s"),
    "state": lambda r: r.get("state"),
    "status": lambda r: r.get("status"),
    "problem_hash": lambda r: (r.get("problem") or {}).get("hash"),
    "problem_name": lambda r: (r.get("problem") or {}).get("name"),
    "problem_kind": lambda r: (r.get("problem") or {}).get("kind"),
    "parent": lambda r: r.get("parent"),
    "relation": lambda r: r.get("relation"),
    "cache_key": lambda r: r.get("cache_key"),
    "reused_from": lambda r: r.get("reused_from"),
    "git_head": lambda r: (r.get("code") or {}).get("git_head"),
    "code_clean": lambda r: (None if (r.get("code") or {}).get("code_matches_commit") is None
                             else int(bool(r["code"]["code_matches_commit"]))),
    "tree_sha": lambda r: (r.get("code") or {}).get("python_tree_sha256"),
    "host": lambda r: (r.get("hardware") or {}).get("host"),
    "gpu": lambda r: (r.get("hardware") or {}).get("gpu"),
    "slurm_job": lambda r: (r.get("slurm") or {}).get("job_id"),
    "slurm_run_id": lambda r: (r.get("slurm") or {}).get("run_id"),
    "telemetry_id": lambda r: r.get("telemetry_id"),
    "has_state": lambda r: int(bool((r.get("outputs") or {}).get("state_end"))),
}
_OPS = {"eq": "=", "ne": "!=", "gt": ">", "ge": ">=", "lt": "<", "le": "<=", "in": "IN",
        "contains": "LIKE"}


def default_root() -> Path:
    """Declared registry root: ``$CN_RUNS`` (any path, the user's explicit choice), else
    ``$CN_TMP/runs``, else ``<repository>/scratch/runs``. A default (not ``$CN_RUNS``) that
    resolves under the home directory is refused: the home is small and backed up."""
    env = os.environ.get("CN_RUNS")
    if env:
        return Path(env)
    base = os.environ.get("CN_TMP")
    if base:
        root = Path(base) / "runs"
    else:
        from .._paths import REPO_ROOT, in_checkout
        # development checkout only: an installed package never writes next to itself (P2)
        repo_tmp = REPO_ROOT / "scratch"
        if not in_checkout() or not repo_tmp.exists():
            raise OptionError("conatus.runs: no registry root declared",
                              why="the registry never writes to a guessed place",
                              fix="set CN_RUNS=/path/on/scratch (or pass root=... explicitly, e.g. "
                                  "cn.runs.track(..., root='runs') for ./runs)")
        root = repo_tmp / "runs"
    try:
        home = Path.home().resolve()
        if root.resolve().is_relative_to(home):
            raise OptionError(f"conatus.runs: default root {root} resolves under the home directory",
                              why="the home is for code and docs only; runs go to scratch storage",
                              fix="set CN_RUNS to a scratch path (CN_RUNS accepts any path you choose)")
    except RuntimeError:                               # no home directory: nothing to protect
        pass
    return root


def _num_text(v: Any) -> tuple[float | None, str | None]:
    """(num, text) index columns of a JSON-safe value: real numbers and booleans in ``num``,
    strings in ``text``; anything else only in the JSON column."""
    if isinstance(v, bool):
        return float(v), None
    if isinstance(v, (int, float)):
        return float(v), None
    if isinstance(v, str):
        if v in ("nan", "inf", "-inf"):
            return float(v), None
        return None, v
    return None, None


class Registry:
    """A registry root: run directories + index. Cheap to create (nothing is opened until used)."""

    def __init__(self, root: str | os.PathLike | None = None) -> None:
        self.root = Path(root) if root is not None else default_root()
        self.runs_dir = self.root / "runs"
        self.index_path = self.root / "index.sqlite"
        self._mem: sqlite3.Connection | None = None     # fallback index (read-only root)
        self.index_note: str | None = None

    def __repr__(self) -> str:
        return f"Registry({str(self.root)!r})"

    def explain(self) -> str:
        n = len(self._ids_on_disk())
        return (f"Registry at {self.root}: {n} run directories (the truth), index {self.index_path}"
                f" (derived, reconciled at every query; SQLite rollback journal, busy timeout "
                f"{BUSY_TIMEOUT_S:g} s)" + (f"; {self.index_note}" if self.index_note else ""))

    # ------------------------------------------------------------------ index connection
    @staticmethod
    def _create(con: sqlite3.Connection) -> None:
        cols = ", ".join(f"{c}" for c in _COLUMNS)
        con.executescript(
            f"CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, {cols}, json_mtime INTEGER);"
            "CREATE TABLE IF NOT EXISTS params (run_id TEXT, key TEXT, num REAL, text TEXT, js TEXT);"
            "CREATE TABLE IF NOT EXISTS results (run_id TEXT, key TEXT, num REAL, text TEXT, js TEXT);"
            "CREATE TABLE IF NOT EXISTS tags (run_id TEXT, tag TEXT);"
            "CREATE INDEX IF NOT EXISTS p_key ON params(key, num);"
            "CREATE INDEX IF NOT EXISTS r_key ON results(key, num);"
            "CREATE INDEX IF NOT EXISTS t_tag ON tags(tag);"
            "CREATE INDEX IF NOT EXISTS runs_cache ON runs(cache_key);"
            "CREATE INDEX IF NOT EXISTS runs_parent ON runs(parent);")

    def _connect(self) -> sqlite3.Connection:
        """The on-disk index, or (read-only / unavailable root) an in-memory one."""
        if self._mem is not None:
            return self._mem
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            con = sqlite3.connect(str(self.index_path), timeout=BUSY_TIMEOUT_S, isolation_level=None)
            con.execute("PRAGMA journal_mode=DELETE")     # no WAL on a network file system
            self._create(con)
            return con
        except (OSError, sqlite3.Error) as err:
            self.index_note = f"on-disk index unavailable ({err}); in-memory index from run directories"
            self._mem = sqlite3.connect(":memory:", isolation_level=None)
            self._create(self._mem)
            return self._mem

    def _close(self, con: sqlite3.Connection) -> None:
        if con is not self._mem:
            con.close()

    # ------------------------------------------------------------------ run files
    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def _ids_on_disk(self) -> list[str]:
        try:
            return sorted(p.name for p in self.runs_dir.iterdir()
                          if p.is_dir() and not p.name.startswith("."))
        except OSError:
            return []

    def load(self, run_id: str) -> dict:
        """The record of one run (``run.json``)."""
        return json.loads((self.run_dir(run_id) / "run.json").read_text())

    # ------------------------------------------------------------------ indexing
    @staticmethod
    def _upsert(con: sqlite3.Connection, rec: Mapping[str, Any], mtime: int) -> None:
        rid = rec["id"]
        con.execute("DELETE FROM runs WHERE id=?", (rid,))
        for t in ("params", "results", "tags"):
            con.execute(f"DELETE FROM {t} WHERE run_id=?", (rid,))
        vals = [rid] + [f(rec) for f in _COLUMNS.values()] + [mtime]
        con.execute(f"INSERT INTO runs VALUES ({','.join('?' * len(vals))})", vals)
        for table, block in (("params", rec.get("params") or {}), ("results", rec.get("results") or {})):
            for k, v in block.items():
                num, text = _num_text(v)
                con.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?)",
                            (rid, k, num, text, json.dumps(v, sort_keys=True)))
        for tag in rec.get("tags") or []:
            con.execute("INSERT INTO tags VALUES (?,?)", (rid, str(tag)))

    def index(self, rec: Mapping[str, Any]) -> None:
        """Insert / replace one run in the index (called by the run's writer thread)."""
        path = self.run_dir(rec["id"]) / "run.json"
        mtime = path.stat().st_mtime_ns if path.exists() else 0
        con = self._connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            self._upsert(con, rec, mtime)
            con.execute("COMMIT")
        except BaseException:
            if con.in_transaction:
                con.execute("ROLLBACK")
            raise
        finally:
            self._close(con)

    def sync(self) -> dict:
        """Reconcile the index with the run directories: add / refresh runs whose ``run.json``
        changed, drop entries whose directory is gone. Returns counts; never raises on one bad
        record (it is reported in ``skipped``)."""
        on_disk = {}
        for rid in self._ids_on_disk():
            try:
                on_disk[rid] = (self.run_dir(rid) / "run.json").stat().st_mtime_ns
            except OSError:
                continue                                 # a run being created: no record yet
        con = self._connect()
        added, removed, skipped = 0, 0, []
        try:
            known = dict(con.execute("SELECT id, json_mtime FROM runs").fetchall())
            stale = [rid for rid, m in on_disk.items() if known.get(rid) != m]
            gone = [rid for rid in known if rid not in on_disk]
            if stale or gone:
                con.execute("BEGIN IMMEDIATE")
                for rid in stale:
                    try:
                        rec = self.load(rid)
                    except (OSError, ValueError) as err:
                        skipped.append(f"{rid}: {err}")
                        continue
                    self._upsert(con, rec, on_disk[rid])
                    added += 1
                for rid in gone:
                    for t, col in (("runs", "id"), ("params", "run_id"), ("results", "run_id"),
                                   ("tags", "run_id")):
                        con.execute(f"DELETE FROM {t} WHERE {col}=?", (rid,))
                    removed += 1
                con.execute("COMMIT")
        except BaseException:
            if con.in_transaction:
                con.execute("ROLLBACK")
            raise
        finally:
            self._close(con)
        return {"indexed": added, "removed": removed, "skipped": skipped}

    def reindex(self) -> dict:
        """Drop the index and rebuild it from the run directories."""
        con = self._connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            for t in ("runs", "params", "results", "tags"):
                con.execute(f"DELETE FROM {t}")
            con.execute("COMMIT")
        finally:
            self._close(con)
        return self.sync()

    # ------------------------------------------------------------------ queries
    def resolve_id(self, run_id: Any) -> str:
        """Full id from an id, a unique prefix, or a Run object."""
        rid = getattr(run_id, "id", run_id)
        if not isinstance(rid, str) or not rid:
            raise OptionError(f"conatus.runs: bad run id {run_id!r}")
        if self.run_dir(rid).is_dir():
            return rid
        matches = [i for i in self._ids_on_disk() if i.startswith(rid)]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise OptionError(f"conatus.runs: no run {rid!r} in {self.root}",
                              fix="list runs with cn.runs.find()" + did_you_mean(rid, self._ids_on_disk()))
        raise OptionError(f"conatus.runs: run id prefix {rid!r} is ambiguous ({len(matches)} runs)",
                          fix="give more characters: " + ", ".join(matches[:5]))

    def _known_keys(self, con: sqlite3.Connection, table: str) -> set[str]:
        return {k for (k,) in con.execute(f"SELECT DISTINCT key FROM {table}")}

    def query_ids(self, *, problem: Any = None, name: str | None = None,
                  status: str | Iterable[str] | None = None, state: str | Iterable[str] | None = None,
                  tags: str | Iterable[str] | None = None, parent: Any = None,
                  since: Any = None, until: Any = None, limit: int | None = None,
                  order: str = "-created", filters: Mapping[str, Any] | None = None) -> list[str]:
        """Ids matching every criterion (AND). See :func:`conatus.runs.find` for the key rules."""
        self.sync()
        where, args = [], []

        def any_of(col: str, v: Any, allowed: tuple | None = None) -> None:
            vals = [v] if isinstance(v, str) else list(v)
            if allowed is not None:
                bad = [x for x in vals if x not in allowed]
                if bad:
                    raise OptionError(f"conatus.runs.find: unknown {col} {bad}; one of {allowed}")
            where.append(f"r.{col} IN ({','.join('?' * len(vals))})")
            args.extend(vals)

        if problem is not None:
            if isinstance(problem, str):
                is_hex = len(problem) >= 8 and all(c in "0123456789abcdef" for c in problem)
                if is_hex:
                    where.append("r.problem_hash LIKE ?")
                    args.append(problem + "%")
                else:
                    where.append("r.problem_name = ?")
                    args.append(problem)
            else:
                from .run import problem_info
                info = problem_info(problem)
                if info is None:
                    raise OptionError(f"conatus.runs.find: problem= must be a Problem, Plan, solver, "
                                      f"hash prefix or name, not {type(problem).__name__}")
                where.append("r.problem_hash = ?")
                args.append(info["hash"])
        if name is not None:
            where.append("r.name = ?")
            args.append(name)
        if status is not None:
            any_of("status", status, STATUSES)
        if state is not None:
            any_of("state", state, STATES)
        if parent is not None:
            where.append("r.parent = ?")
            args.append(getattr(parent, "id", parent))
        if since is not None:
            where.append("r.created_utc >= ?")
            args.append(since.isoformat() if hasattr(since, "isoformat") else str(since))
        if until is not None:
            where.append("r.created_utc <= ?")
            args.append(until.isoformat() if hasattr(until, "isoformat") else str(until))
        for tag in ([tags] if isinstance(tags, str) else list(tags or [])):
            where.append("EXISTS (SELECT 1 FROM tags t WHERE t.run_id = r.id AND t.tag = ?)")
            args.append(tag)
        con = self._connect()
        try:
            if filters:
                pkeys, rkeys = self._known_keys(con, "params"), self._known_keys(con, "results")
                empty = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
                for raw, value in filters.items():
                    clause, cargs = self._filter(raw, value, pkeys, rkeys, empty)
                    where.append(clause)
                    args.extend(cargs)
            if order not in ("created", "-created"):
                raise OptionError("conatus.runs.find: order must be 'created' or '-created'")
            sql = ("SELECT r.id FROM runs r" + (" WHERE " + " AND ".join(where) if where else "")
                   + f" ORDER BY r.created_utc {'DESC' if order.startswith('-') else 'ASC'}, r.id")
            if limit is not None:
                sql += f" LIMIT {int(limit)}"
            return [i for (i,) in con.execute(sql, args)]
        finally:
            self._close(con)

    @staticmethod
    def _filter(raw: str, value: Any, pkeys: set, rkeys: set, empty: bool) -> tuple[str, list]:
        """SQL clause of one ``key__op=value`` filter (key rules in conatus.runs.find)."""
        key, op = raw, "eq"
        if "__" in raw:
            key, op = raw.rsplit("__", 1)
            if op not in _OPS:
                raise OptionError(f"conatus.runs.find: unknown operator {op!r} in {raw!r}",
                                  fix=f"operators: {', '.join(_OPS)}" + did_you_mean(op, _OPS))
        table = None
        if key.startswith("param_") and key[6:] in pkeys:
            table, key = "params", key[6:]
        elif key.startswith("result_") and key[7:] in rkeys:
            table, key = "results", key[7:]
        elif key in _COLUMNS:
            table = "runs"
        elif key in pkeys and key in rkeys:
            raise OptionError(f"conatus.runs.find: {key!r} is both a parameter and a result",
                              why="the registry does not choose between them for you",
                              fix=f"write param_{key}__... or result_{key}__...")
        elif key in pkeys:
            table = "params"
        elif key in rkeys:
            table = "results"
        elif empty:
            return "0", []                               # empty registry: nothing matches
        else:
            raise OptionError(f"conatus.runs.find: no run has a parameter or result named {key!r}",
                              fix="check the name" + did_you_mean(key, sorted(pkeys | rkeys | set(_COLUMNS))))
        sql_op = _OPS[op]
        vals = list(value) if op == "in" else [value]
        if op == "in" and not vals:
            return "0", []
        if op == "contains":
            vals = [f"%{value}%"]
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, complex) for v in vals)
        if table == "runs":
            col = f"r.{key}"
            conv = [float(v) if isinstance(v, bool) else v for v in vals]
            if op == "in":
                return f"{col} IN ({','.join('?' * len(conv))})", conv
            return f"{col} {sql_op} ?", conv
        col = "num" if numeric else "text"
        conv = [float(v) for v in vals] if numeric else [str(v) for v in vals]
        cmp = (f"x.{col} IN ({','.join('?' * len(conv))})" if op == "in" else f"x.{col} {sql_op} ?")
        return (f"EXISTS (SELECT 1 FROM {table} x WHERE x.run_id = r.id AND x.key = ? AND {cmp})",
                [key] + conv)

    def children(self, run_id: str) -> list[str]:
        self.sync()
        con = self._connect()
        try:
            return [i for (i,) in con.execute(
                "SELECT id FROM runs WHERE parent = ? OR reused_from = ? ORDER BY created_utc",
                (run_id, run_id))]
        finally:
            self._close(con)

    def cache_lookup(self, key: str) -> str | None:
        """Newest completed, non-retracted run with this cache key and a stored output."""
        self.sync()
        con = self._connect()
        try:
            row = con.execute(
                "SELECT id FROM runs WHERE cache_key = ? AND state = 'completed' AND status != "
                "'retracted' AND reused_from IS NULL ORDER BY created_utc DESC LIMIT 1", (key,)).fetchone()
            return None if row is None else row[0]
        finally:
            self._close(con)

    # ------------------------------------------------------------------ status changes
    def update_status(self, run_id: str, status: str, reason: str, evidence: Any, utc: str) -> dict:
        """Read-modify-write of one run's status under its lock; history kept (never rewritten)."""
        d = self.run_dir(run_id)
        with FileLock(d / ".lock"):
            rec = self.load(run_id)
            rec["status"] = status
            rec.setdefault("status_history", []).append(
                {"status": status, "reason": reason, "evidence": evidence, "utc": utc,
                 "by": os.environ.get("USER")})
            atomic_write_json(d / "run.json", rec)
        try:
            self.index(rec)
        except (OSError, sqlite3.Error):
            pass                                          # reconciled by the next sync()
        return rec
