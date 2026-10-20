"""Incremental reader of a telemetry spool (used by the Studio server; see protocol.py).

Keeps one byte offset per ``<run_id>.jsonl`` and returns only complete new lines, so it can be
polled cheaply. Frames are re-read only when their file changes. Malformed lines are counted, not
hidden. Bounded histories keep the server's memory constant whatever the run length. Films
(``<run_id>.film.ndjson``) are indexed by :meth:`SpoolReader.poll_film` (offsets only, see
:mod:`conatus.telemetry.film`).
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from .film import FILM_SUFFIX, FilmIndex
from .protocol import FRAME_KINDS, run_file_id


class RunState:
    """What the Studio knows about one run (bounded)."""

    def __init__(self, run_id: str, max_scalars: int, max_logs: int) -> None:
        self.run_id = run_id
        self.header: dict | None = None
        self.scalars: deque = deque(maxlen=max_scalars)
        self.newton: deque = deque(maxlen=max_scalars)
        self.logs: deque = deque(maxlen=max_logs)
        self.events: deque = deque(maxlen=max_logs)
        self.status: dict | None = None
        self.frames: dict[str, dict] = {}
        self.last_wall = 0.0
        self.malformed = 0

    def summary(self) -> dict:
        """Header, status, last scalars, counters and frame names of the run."""
        last = self.scalars[-1] if self.scalars else {}
        return dict(run_id=self.run_id, header=self.header, status=self.status,
                    last=last, last_wall=self.last_wall, malformed=self.malformed,
                    frames=sorted(self.frames))

    def snapshot(self) -> dict:
        """The summary plus every bounded history and the latest frames."""
        return dict(summary=self.summary(), scalars=list(self.scalars), newton=list(self.newton),
                    logs=list(self.logs), events=list(self.events), frames=self.frames)

    def ingest(self, rec: dict[str, Any]) -> None:
        """File one telemetry-plane record (frames go to ``frames``). Unknown kinds and malformed
        records are counted, never raised: on the hub an exception here left the relay's cursor
        behind, and the same line was resent forever."""
        try:
            wall = float(rec.get("wall", 0.0) or 0.0)
        except (TypeError, ValueError, OverflowError):
            self.malformed += 1
            return
        if wall == wall:                            # a NaN wall never becomes last_wall
            self.last_wall = max(self.last_wall, wall)
        kind = rec.get("kind")
        if kind == "header":
            self.header = rec
        elif kind == "scalars":
            self.scalars.append(rec)
        elif kind == "newton":
            self.newton.append(rec)
        elif kind == "log":
            self.logs.append(rec)
        elif kind == "event":
            self.events.append(rec)
        elif kind == "status":
            self.status = rec
        elif kind in FRAME_KINDS and isinstance(rec.get("name"), str):   # every frame kind, 2D,
            self.frames[rec["name"]] = rec          # 3D, curves (the hub used to count 3D frames
                                                    # as malformed)
        else:
            self.malformed += 1


class SpoolReader:
    """Incremental reader of all runs in a telemetry spool directory (bounded histories)."""

    def __init__(self, spool: str | Path, *, max_scalars: int, max_logs: int) -> None:
        self.spool = Path(spool)
        self.max_scalars, self.max_logs = int(max_scalars), int(max_logs)
        self.runs: dict[str, RunState] = {}
        self._pos: dict[str, int] = {}
        self._frame_mtime: dict[Path, float] = {}
        self.films: dict[str, FilmIndex] = {}

    def poll(self) -> list[dict]:
        """Read what is new; return the increments (records and frames) for broadcasting."""
        out: list[dict] = []
        if not self.spool.is_dir():
            return out
        for path in sorted(self.spool.glob("*.jsonl")):
            run_id = run_file_id(path.name)
            if run_id is None:                      # control file, spool audit, not a run id
                continue
            st = self.runs.setdefault(run_id, RunState(run_id, self.max_scalars, self.max_logs))
            pos = self._pos.get(run_id, 0)
            try:
                size = path.stat().st_size
                if size < pos:                      # file truncated/replaced: start over
                    pos = 0
                if size == pos:
                    continue
                with path.open("rb") as fh:
                    fh.seek(pos)
                    chunk = fh.read(size - pos)
            except OSError:
                continue
            chunk = chunk[:chunk.rfind(b"\n") + 1]  # complete lines only
            self._pos[run_id] = pos + len(chunk)
            for line in chunk.decode("utf-8", "replace").splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    st.malformed += 1
                    continue
                if not isinstance(rec, dict):       # valid JSON, not a record: counted
                    st.malformed += 1
                    continue
                self._ingest(st, rec)
                out.append(rec)
        for path in self.spool.glob("*.frame.*.json"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if self._frame_mtime.get(path) == mtime:
                continue
            try:
                fr = json.loads(path.read_text())
            except (OSError, ValueError):
                continue                            # being replaced: next poll
            self._frame_mtime[path] = mtime
            if not (isinstance(fr, dict) and isinstance(fr.get("run_id"), str)
                    and isinstance(fr.get("name"), str)):
                continue                            # not a frame: never a KeyError in the poll
            st = self.runs.setdefault(fr["run_id"], RunState(fr["run_id"], self.max_scalars, self.max_logs))
            st.frames[fr["name"]] = fr
            out.append(fr)
        return out

    def poll_film(self) -> list[dict]:
        """Index what is new in every film file; returns ``{"type": "film", "run_id", "name",
        "i", "t", "iteration"}`` notices (one per new film frame) for broadcasting."""
        out: list[dict] = []
        if not self.spool.is_dir():
            return out
        for path in sorted(self.spool.glob("*" + FILM_SUFFIX)):
            run_id = path.name[:-len(FILM_SUFFIX)]
            idx = self.films.get(run_id)
            if idx is None:
                idx = self.films[run_id] = FilmIndex(run_id, path=path)
            out += [{"type": "film", "run_id": run_id, **e} for e in idx.poll_file()]
        return out

    def _ingest(self, st: RunState, rec: dict[str, Any]) -> None:
        st.ingest(rec)
