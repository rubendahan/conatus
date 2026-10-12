"""The film of a run, on the reading side: an index of ``<run_id>.film.ndjson``.

The computation's writer thread appends one line per frame kept for replay
(:func:`conatus.telemetry.protocol.film_record`, budget in :class:`conatus.telemetry.Telemetry`).
Readers never load the whole film: :class:`FilmIndex` keeps, per frame name, one small entry
``(i, t, iteration, wall, offset, length)`` and reads the frames asked for on demand.

Two storage modes, one interface:

* **file** (local server): the bytes stay in the file, entries hold byte offsets; a film file that
  shrinks (replaced, truncated) resets the index;
* **memory** (hub, fed by relays that push the film lines): entries hold the line itself, with a
  bounded number of frames per name and a global byte budget; past them the film is **thinned**
  (every second frame of the name dropped, uniformly over its whole length, so the replay still
  covers the whole run; when no name has 3 frames left, the oldest frame of the largest name, down
  to removing a name) and the thinning is counted and declared in :meth:`FilmIndex.summary`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FILM_SUFFIX = ".film.ndjson"
MAX_FRAMES_PER_REQUEST = 64          # one HTTP answer carries at most this many frames
MAX_BYTES_PER_REQUEST = 8 << 20      # ... and at most about this many bytes (one frame at least)
READ_CHUNK = 16 << 20                # file mode: bytes indexed per poll (a 256 MiB film: 16 polls)


@dataclass
class FilmEntry:
    """One frame of the film: where it is (file offset or the line itself) and when it was."""
    i: int
    t: float | None
    iteration: int | None
    wall: float | None
    kind: str
    offset: int = 0
    length: int = 0
    line: bytes | None = None


class FilmIndex:
    """Index of the film of one run (see the module docstring for the two storage modes)."""

    def __init__(self, run_id: str, *, path: str | Path | None = None,
                 max_frames: int | None = None) -> None:
        self.run_id = run_id
        self.path = Path(path) if path is not None else None     # None: memory mode
        self.max_frames = int(max_frames) if max_frames else None
        self.names: dict[str, list[FilmEntry]] = {}
        self.pos = 0                    # bytes of the source already indexed (file or stream)
        self.level = 0                  # latest thinning level written by the computation
        self.malformed = 0
        self.thinned = 0                # frames dropped here (memory mode), on top of the source
        self.bytes = 0                  # bytes held (memory mode) or indexed (file mode)

    # ------------------------------------------------------------------ ingestion
    def reset(self) -> None:
        """Forget everything (the source was truncated or replaced)."""
        self.names.clear()
        self.pos = self.bytes = 0
        self.level = 0

    def poll_file(self) -> list[dict]:
        """File mode: index the complete new lines of the film file; returns the new entries
        as ``{"name", "i", "t", "iteration"}`` (what a monitor broadcasts)."""
        if self.path is None:
            raise ValueError("poll_file() needs a film file (memory-mode index)")
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self.pos:                         # replaced/truncated: index again
            self.reset()
        if size == self.pos:
            return []
        try:
            with self.path.open("rb") as fh:
                fh.seek(self.pos)
                chunk = fh.read(min(size - self.pos, READ_CHUNK))
        except OSError:
            return []
        if len(chunk) == READ_CHUNK and b"\n" not in chunk:    # a line longer than a chunk:
            self.malformed += 1                                  # skipped, counted (never stuck)
            self.pos += len(chunk)
            return []
        return self.ingest(chunk, self.pos)

    def ingest(self, chunk: bytes, start: int) -> list[dict]:
        """Index the complete lines of ``chunk`` (bytes ``[start, ...)`` of the film stream).
        An incomplete last line is left for the next call. Returns the new entries."""
        cut = chunk.rfind(b"\n") + 1
        new: list[dict] = []
        off = start
        for line in chunk[:cut].splitlines(keepends=True):
            n = len(line)
            try:          # everything parsed before the entry is added (a bad field never
                rec = json.loads(line)              # leaves the index half-updated)
                name, i = str(rec["name"]), int(rec["i"])
                level = int(rec.get("level", 0) or 0)
            except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
                self.malformed += 1
                off += n
                continue
            e = FilmEntry(i=i, t=_num(rec.get("t")), iteration=_int(rec.get("iteration")),
                          wall=_num(rec.get("wall")), kind=str(rec.get("kind", "frame")),
                          offset=off, length=n, line=None if self.path is not None else line)
            self.names.setdefault(name, []).append(e)
            self.level = max(self.level, level)
            self.bytes += n
            new.append({"name": name, "i": i, "t": e.t, "iteration": e.iteration})
            off += n
            if self.path is None and self.max_frames and len(self.names[name]) > self.max_frames:
                self.thin(name)
        self.pos = start + cut
        return new

    def thin(self, name: str) -> int:
        """Memory mode: drop every second frame of ``name`` (uniformly, first and last kept);
        returns the bytes freed. Counted in ``thinned`` and declared in the summary."""
        es = self.names.get(name, [])
        if len(es) < 3:
            return 0
        keep = es[::2] if len(es) % 2 == 1 else es[::2] + [es[-1]]
        drop = len(es) - len(keep)
        freed = sum(e.length for e in es) - sum(e.length for e in keep)
        self.names[name] = keep
        self.thinned += drop
        self.bytes -= freed
        return freed

    # ------------------------------------------------------------------ queries
    def summary(self) -> dict:
        """Names with count and time span, thinning level and counters."""
        return {"run": self.run_id, "level": self.level, "bytes": self.bytes,
                "thinned_here": self.thinned, "malformed": self.malformed,
                "names": {n: {"kind": es[-1].kind, "count": len(es),
                              "first_t": es[0].t, "last_t": es[-1].t}
                          for n, es in self.names.items() if es}}

    def index(self, name: str) -> list[dict]:
        """``[{i, t, iteration, wall}]`` of the frames of ``name`` in the film (time order)."""
        return [{"i": e.i, "t": e.t, "iteration": e.iteration, "wall": e.wall}
                for e in self.names.get(name, [])]

    def frames(self, name: str, start: int = 0, stop: int | None = None, step: int = 1,
               limit: int = MAX_FRAMES_PER_REQUEST) -> tuple[list[dict], int | None]:
        """The film frames of ``name`` whose film index ``i`` lies in ``[start, stop)``, every
        ``step``-th of them; at most ``limit`` per call and about MAX_BYTES_PER_REQUEST bytes
        (always one frame at least), with the ``i`` to ask next (or None)."""
        if step < 1:
            raise ValueError("step must be >= 1")
        sel = [e for e in self.names.get(name, []) if e.i >= start and (stop is None or e.i < stop)]
        sel = sel[::step]
        cap, total = 0, 0
        for e in sel[:limit]:
            if cap and total + e.length > MAX_BYTES_PER_REQUEST:
                break
            total += e.length
            cap += 1
        more = sel[cap:]
        sel = sel[:cap]
        out = []
        fh = None
        try:
            for e in sel:
                if e.line is not None:
                    raw = e.line
                else:
                    if fh is None:
                        fh = self.path.open("rb")                  # type: ignore[union-attr]
                    fh.seek(e.offset)
                    raw = fh.read(e.length)
                try:
                    out.append(json.loads(raw))
                except ValueError:
                    self.malformed += 1
        except OSError:
            pass                                            # file gone: what was read is returned
        finally:
            if fh is not None:
                fh.close()
        return out, (more[0].i if more else None)


def thin_to_budget(films: dict[Any, FilmIndex], max_bytes: int) -> int:
    """Memory mode, several runs: thin the name holding the most bytes until the total fits
    ``max_bytes``; when no name has 3 frames left, drop the oldest frame of the largest name (a
    name of one frame goes entirely). The budget is therefore always met (50 names of 2 frames
    each used to stay 10x over it). Returns the number of passes; every dropped frame
    is counted in its index's ``thinned_here``."""
    passes = 0
    while sum(f.bytes for f in films.values()) > max_bytes:
        sized = [(f, n, sum(e.length for e in es), len(es)) for f in films.values()
                 for n, es in f.names.items() if es]
        if not sized:
            break
        thick = [x for x in sized if x[3] >= 3]
        f, n, _, _ = max(thick or sized, key=lambda x: x[2])
        if thick:
            f.thin(n)
        else:
            e = f.names[n].pop(0)
            f.bytes -= e.length
            f.thinned += 1
            if not f.names[n]:
                del f.names[n]
        passes += 1
    return passes


def _num(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError, OverflowError):     # 10**400 overflows float()
        return None


def _int(v: Any) -> int | None:
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError, OverflowError):     # int(inf): a JSON Infinity iteration
        return None
