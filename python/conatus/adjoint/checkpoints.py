"""Where the Revolve checkpoints of :class:`conatus.adjoint.FlowMap` live, and how a long reverse
sweep is suspended and resumed by another job (step 2).

Stores (``FlowMap(..., checkpoint_store=...)``, explicit, shown by ``explain()``):

* ``"device"`` (default, unchanged behaviour): a checkpoint is a reference to the device arrays
  of the augmented state; device memory grows with the number of slots;
* ``"host"``: a copy in host memory (page-locked "pinned" buffers on the GPU, so the copies run
  at full PCIe speed; plain NumPy copies on the CPU backend); the device keeps only the working
  states;
* ``"disk"``: one ``.npy`` file per array in ``checkpoint_dir`` (written to a temporary name,
  then renamed: a file is either complete or absent). Survives the job: a suspended sweep
  resumes from it.

A copy moves bits unchanged, so the three stores give bit-identical results (tested). Only the
place and the cost of a checkpoint change, never the arithmetic.

Journal (``FlowMap.vjp(..., journal=dir, budget_s=...)``): when the time budget is reached, the
sweep stops at a step boundary and writes, atomically, its position in the Revolve action list
(:func:`conatus.adjoint.flowmap.revolve_actions`), the partial cotangents (the "partial
gradient"), the working state of an interrupted forward recomputation, and the live checkpoints
(hard links for the disk store, copies otherwise), then raises :class:`VJPSuspended`. The same
call in a new job (same X_0, same map) resumes from it; the result is bit-identical to an
uninterrupted sweep because every recomputed step is the same deterministic function of the same
bits. Atomicity: every file of a new journal generation is written first (temporary name, then
rename), then ``state.json`` is replaced, then the files of older generations are removed; a job
killed at any point leaves a complete previous generation.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np

from ..errors import ConatusError, OptionError

__all__ = ["CheckpointStore", "make_store", "VJPSuspended", "Journal", "STORES"]

STORES = ("device", "host", "disk")
_DOC = "docs/site/guide/dynamics/adjoint.md"


class VJPSuspended(ConatusError, RuntimeError):
    """The reverse sweep reached its time budget and was written to its journal (not an error of
    the computation: re-run the same call to resume)."""

    def __init__(self, message: str, *, journal: str, position: dict) -> None:
        super().__init__(message, why="budget_s reached: the sweep stopped at a step boundary and "
                                      "its exact state was written to the journal",
                         fix=f"re-run the same call (same X0, same FlowMap) with journal={journal!r}, "
                             "e.g. in the next job: it resumes bit for bit", doc=_DOC)
        self.journal, self.position = journal, position


# ---------------------------------------------------------------------------- stores
class CheckpointStore:
    """Checkpoints of augmented states (lists of arrays or None), keyed by step index."""

    kind = "device"

    def __init__(self, xp) -> None:
        self.xp = xp
        self._d: dict[int, list] = {}
        self.stats = dict(puts=0, gets=0, bytes_out=0, bytes_in=0, seconds=0.0, peak_held=0)

    def held(self) -> list[int]:
        """Steps whose checkpoint is stored, in increasing order."""
        return sorted(self._d)

    def _count(self, hist, t0: float, key: str) -> None:
        nb = sum(int(a.nbytes) for a in hist if a is not None)
        self.stats["bytes_out" if key == "puts" else "bytes_in"] += nb
        self.stats[key] += 1
        self.stats["seconds"] += time.perf_counter() - t0
        self.stats["peak_held"] = max(self.stats["peak_held"], len(self._d))

    def put(self, k: int, hist: list) -> None:
        """Store the history ``hist`` of step ``k`` (references to the arrays) and update the
        counters."""
        self._d[k] = list(hist)                  # references: the arrays are never modified
        self.stats["puts"] += 1
        self.stats["peak_held"] = max(self.stats["peak_held"], len(self._d))

    def get(self, k: int) -> list:
        """The stored history of step ``k`` (a new list of the same arrays)."""
        self.stats["gets"] += 1
        return list(self._d[k])

    def drop(self, k: int) -> None:
        """Forget the checkpoint of step ``k`` (no error if absent)."""
        self._d.pop(k, None)

    def to_host(self, k: int) -> list:
        """Host (NumPy) copies of checkpoint k (journal)."""
        return [None if a is None else (a if isinstance(a, np.ndarray) else a.get())
                for a in self._d[k]]

    def files(self, k: int) -> list | None:
        """Files already holding checkpoint k (disk store), else None."""
        return None

    def put_host(self, k: int, arrays: list) -> None:
        """Checkpoint k from host arrays (journal restore)."""
        self.put(k, [None if a is None else self.xp.asarray(a) for a in arrays])

    def clear(self) -> None:
        """Drop every stored checkpoint."""
        for k in list(self._d):
            self.drop(k)

    def describe(self) -> str:
        """Where the checkpoints are kept, one line."""
        return "device memory (references to the state arrays; default)"


class HostStore(CheckpointStore):
    """Copies in host memory: pinned buffers on the GPU, NumPy copies on the CPU backend."""

    kind = "host"

    def put(self, k: int, hist: list) -> None:
        """Copy the history ``hist`` of step ``k`` to host memory (pinned buffers on the GPU)."""
        t0 = time.perf_counter()
        out = []
        for a in hist:
            if a is None:
                out.append(None)
            elif isinstance(a, np.ndarray):
                out.append(a.copy())
            else:
                import cupyx
                buf = cupyx.empty_pinned(a.shape, dtype=a.dtype)
                a.get(out=buf)                   # synchronous device -> pinned host copy
                out.append(buf)
        self._d[k] = out
        self._count(hist, t0, "puts")

    def get(self, k: int) -> list:
        """Fresh device (or NumPy) copies of the history of step ``k``."""
        t0 = time.perf_counter()
        hist = [None if a is None else self.xp.array(a) for a in self._d[k]]  # fresh copies
        self._count(hist, t0, "gets")
        return hist

    def to_host(self, k: int) -> list:
        """The stored host copies of step ``k``."""
        return list(self._d[k])

    def put_host(self, k: int, arrays: list) -> None:
        """Store host arrays for step ``k`` (same as put)."""
        self.put(k, arrays)

    def describe(self) -> str:
        """Where the checkpoints are kept (pinned host buffers or NumPy copies), one line."""
        return ("host memory (pinned buffers, copies device <-> host at every store and fetch)"
                if self.xp is not np else "host memory (NumPy copies)")


class DiskStore(CheckpointStore):
    """One ``.npy`` file per array of a checkpoint in ``directory`` (atomic writes)."""

    kind = "disk"

    def __init__(self, xp, directory) -> None:
        super().__init__(xp)
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, k: int, j: int) -> Path:
        return self.dir / f"ck_{k:09d}_{j}.npy"

    def put(self, k: int, hist: list) -> None:
        """Write the history ``hist`` of step ``k``, one .npy file per array, each written
        atomically."""
        t0 = time.perf_counter()
        names = []
        for j, a in enumerate(hist):
            if a is None:
                names.append(None)
                continue
            p = self._path(k, j)
            _save_npy(p, a if isinstance(a, np.ndarray) else a.get())
            names.append(p)
        self._d[k] = names
        self._count(hist, t0, "puts")

    def get(self, k: int) -> list:
        """Load the history of step ``k`` into the array module of the store."""
        t0 = time.perf_counter()
        hist = [None if p is None else self.xp.asarray(np.load(p)) for p in self._d[k]]
        self._count(hist, t0, "gets")
        return hist

    def drop(self, k: int) -> None:
        """Delete the files of step ``k`` (no error if absent)."""
        for p in self._d.pop(k, None) or []:
            if p is not None:
                p.unlink(missing_ok=True)

    def to_host(self, k: int) -> list:
        """The history of step ``k`` as NumPy arrays read from disk."""
        return [None if p is None else np.load(p) for p in self._d[k]]

    def files(self, k: int) -> list | None:
        """Paths of the files of step ``k`` (None for an absent array)."""
        return list(self._d[k])

    def put_host(self, k: int, arrays: list) -> None:
        """Store host arrays for step ``k`` (same as put)."""
        self.put(k, arrays)

    def adopt(self, k: int, files: list) -> None:
        """Checkpoint k from journal files: hard links (no copy; a copy across file systems)."""
        names = []
        for j, src in enumerate(files):
            if src is None:
                names.append(None)
                continue
            p = self._path(k, j)
            _link_or_copy(Path(src), p)
            names.append(p)
        self._d[k] = names

    def describe(self) -> str:
        """Where the checkpoints are kept (the directory), one line."""
        return f"disk, one .npy file per array in {self.dir} (atomic writes)"


def make_store(kind: str, xp, directory=None) -> CheckpointStore:
    """The checkpoint store of kind 'device', 'host' or 'disk' (the last needs ``directory``);
    OptionError otherwise."""
    if kind == "device":
        return CheckpointStore(xp)
    if kind == "host":
        return HostStore(xp)
    if kind == "disk":
        if directory is None:
            raise OptionError("checkpoint_store='disk' needs checkpoint_dir",
                              fix="FlowMap(..., checkpoint_store='disk', checkpoint_dir="
                                  "'<a directory on the project store, e.g. scratch/ck/run1>')",
                              doc=_DOC)
        return DiskStore(xp, directory)
    raise OptionError(f"checkpoint_store must be one of {STORES}, got {kind!r}",
                      fix="checkpoint_store='device' (default), 'host' (pinned host memory) or "
                          "'disk' (with checkpoint_dir=...)", doc=_DOC)


# ---------------------------------------------------------------------------- journal
# Files owned by a journal: only these are ever removed from its directory (a user file such as
# "grid_x.npy" in the same directory is never touched).
_GEN_FILE = re.compile(r"^g\d{6}_(bar|work|ck)\d+(_\d+)?\.npy(\.tmp)?$")


def _save_npy(path: Path, a: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, np.ascontiguousarray(a))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _fsync_dir(d: Path) -> None:
    """Make the renames in ``d`` durable (a node crash after a rename must not lose it)."""
    try:
        fd = os.open(d, os.O_RDONLY)
    except OSError:                            # platform without directory descriptors
        return
    try:
        os.fsync(fd)
    except OSError:                            # some file systems refuse fsync on a directory
        pass
    finally:
        os.close(fd)


def _link_or_copy(src: Path, dst: Path) -> None:
    """dst holds the bytes of src: a hard link if possible, else an atomic copy. Any previous dst
    is removed first (a leftover of a crashed save may be a hard link to a live checkpoint: it must
    never be written through)."""
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
    except OSError:                            # other file system, or links not supported
        tmp = dst.with_name(dst.name + ".tmp")
        tmp.unlink(missing_ok=True)
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)


def array_digest(a) -> str:
    """SHA-256 of the bytes of an array (host copy for a device array)."""
    h = a if isinstance(a, np.ndarray) else a.get()
    return hashlib.sha256(np.ascontiguousarray(h).tobytes()).hexdigest()


class Journal:
    """State of a suspended reverse sweep in ``directory`` (module docstring)."""

    def __init__(self, directory, fingerprint: dict) -> None:
        self.dir = Path(directory)
        self.fp = dict(fingerprint)
        self.state_path = self.dir / "state.json"

    def load(self) -> dict | None:
        """The saved state if it belongs to this computation, None if there is none; a journal
        of another computation is refused (never mixed in silently)."""
        if not self.state_path.exists():
            return None
        try:
            st = json.loads(self.state_path.read_text())
            old = st["fingerprint"]
        except (ValueError, KeyError) as err:
            raise OptionError(f"journal {self.state_path} is unreadable ({err})",
                              why="state.json is replaced atomically and synced, so this file was "
                                  "not written by a FlowMap journal or was altered",
                              fix="inspect it; remove the journal directory knowingly to restart "
                                  "the sweep from scratch", doc=_DOC) from err
        diff = sorted(k for k in set(old) | set(self.fp)
                      if old.get(k) != self.fp.get(k) and not (k == "W" and self.fp.get(k) is None))
        if diff == ["adjoint"]:
            # the same sweep with the other shell adjoint (planned vs node-by-node walk): same adjoint up to rounding, not to the bit, so never mixed; say which
            # path wrote it and how to finish it
            def path(fp: dict) -> str:
                return ("the planned adjoint" if fp.get("adjoint") == "planned"
                        else "the node-by-node walk (planned=False, or written before the planned "
                             "shell adjoint)")
            raise OptionError(
                f"journal {self.dir} was written by {path(old)}; this FlowMap uses {path(self.fp)}",
                why="both are the exact adjoint of the same discrete step but differ in the last "
                    "bits (summation order): a sweep resumed with the other one would mix them",
                fix="restart the sweep in a new journal directory, or finish this one with the path "
                    "that wrote it: fm.rhs_lin = conatus.adjoint.sphere_rhs.SphereLinearizedRHS("
                    "solver, planned=" + ("True" if old.get("adjoint") == "planned" else "False")
                    + ") before fm.vjp(X0, None, journal=...)", doc=_DOC)
        if diff:
            raise OptionError(f"journal {self.dir} belongs to another computation (differs in "
                              f"{', '.join(diff)})",
                              why="resuming it would mix two different sweeps",
                              fix="pass the same X0, W and FlowMap as the suspended call, or another "
                                  "journal directory (or delete this one knowingly)", doc=_DOC)
        self.fp = old                          # keeps the digest of W when resumed with W=None
        # the names come from a file: each must be a generation file of this directory (security
        # "../x" or an absolute path made the resume read, hard-link and later
        # delete files elsewhere)
        names = list(st.get("bars") or []) + list(st.get("work") or [])
        for v in (st.get("checkpoints") or {}).values():
            names += list(v or [])
        bad = [n for n in names if n is not None and not (isinstance(n, str) and _GEN_FILE.match(n))]
        if bad:
            raise OptionError(f"journal {self.state_path} names files that are not its own: "
                              f"{bad[:3]}", why="a journal only references its generation files",
                              fix="inspect it; remove the journal directory knowingly to restart "
                                  "the sweep from scratch", doc=_DOC)
        return st

    def read_arrays(self, names: list) -> list:
        """Load the named .npy files of the journal directory (None stays None)."""
        return [None if n is None else np.load(self.dir / n) for n in names]

    def save(self, *, cursor: int, progress: int | None, bars: list, work: list | None,
             store: CheckpointStore, extra: dict | None = None) -> dict:
        """Write a new generation (arrays first, then state.json, then remove the old one)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        gen = 0
        if self.state_path.exists():
            gen = int(json.loads(self.state_path.read_text())["generation"]) + 1
        pre = f"g{gen:06d}_"

        def put(name: str, a) -> str:
            _save_npy(self.dir / (pre + name), a if isinstance(a, np.ndarray) else a.get())
            return pre + name

        for p in self._owned():                # leftovers of a save of this generation that
            if p.name.startswith(pre):         # crashed before its state.json
                p.unlink()
        bar_names = [None if b is None else put(f"bar{j}.npy", b) for j, b in enumerate(bars)]
        work_names = (None if work is None else
                      [None if a is None else put(f"work{j}.npy", a) for j, a in enumerate(work)])
        cks = {}
        for k in store.held():
            files = store.files(k)
            names = []
            if files is not None:              # disk store: hard links (same bytes, no copy)
                for j, p in enumerate(files):
                    if p is None:
                        names.append(None)
                        continue
                    dst = self.dir / f"{pre}ck{k}_{j}.npy"
                    _link_or_copy(p, dst)
                    names.append(dst.name)
            else:
                names = [None if a is None else put(f"ck{k}_{j}.npy", a)
                         for j, a in enumerate(store.to_host(k))]
            cks[str(k)] = names
        st = dict(fingerprint=self.fp, generation=gen, cursor=int(cursor),
                  progress=None if progress is None else int(progress), bars=bar_names,
                  work=work_names, checkpoints=cks, extra=extra or {},
                  saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        _fsync_dir(self.dir)                   # the array renames are durable before the state
        tmp = self.state_path.with_name("state.json.tmp")
        with open(tmp, "w") as f:
            f.write(json.dumps(st, indent=1))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.state_path)
        _fsync_dir(self.dir)                   # the new state is durable before the old files go
        for p in self._owned():
            if not p.name.startswith(pre):
                p.unlink(missing_ok=True)
        return st

    def _owned(self) -> list[Path]:
        """Files of this journal (all generations, temporary files included), nothing else."""
        if not self.dir.is_dir():
            return []
        return [p for p in self.dir.iterdir() if _GEN_FILE.match(p.name)]

    def clear(self) -> None:
        """Remove the journal of a completed sweep (state first: never a state without files),
        temporary files included; other files in the directory are left alone."""
        self.state_path.unlink(missing_ok=True)
        self.state_path.with_name("state.json.tmp").unlink(missing_ok=True)
        for p in self._owned():
            p.unlink(missing_ok=True)
