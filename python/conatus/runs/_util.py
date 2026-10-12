"""Helpers of the run registry: JSON-safe values, canonical hashing, atomic writes, file locks.

Nothing here touches the computation: these functions only transform values that the caller has
already handed to the registry, and every conversion is lossless or declared (a value that cannot
be stored exactly is stored as a tagged summary, never silently rounded).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

#: arrays up to this many elements are stored inline in ``run.json`` (as lists); larger arrays in
#: parameters are replaced by a tagged summary {"array": {shape, dtype, sha256}} (declared)
INLINE_ARRAY_MAX = 64


def jsonable(v: Any) -> Any:
    """JSON-safe copy of ``v``, exact for numbers (floats kept as Python floats: JSON round-trips
    them exactly via repr). Non-finite floats become the strings "nan", "inf", "-inf"; complex
    numbers {"complex": [re, im]}; big arrays a hashed summary; unknown objects {"repr": ...}."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else ("nan" if math.isnan(f) else ("inf" if f > 0 else "-inf"))
    if isinstance(v, (complex, np.complexfloating)):
        c = complex(v)
        return {"complex": [jsonable(c.real), jsonable(c.imag)]}
    if isinstance(v, np.ndarray):
        if v.size <= INLINE_ARRAY_MAX and v.dtype.kind in "biufc":
            return {"ndarray": [jsonable(x) for x in v.ravel().tolist()], "shape": list(v.shape),
                    "dtype": str(v.dtype)}
        return {"array": {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": array_sha256(v)}}
    if isinstance(v, Mapping):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, Path):
        return str(v)
    return {"repr": repr(v)}


def from_jsonable(v: Any) -> Any:
    """Inverse of :func:`jsonable` where it is exact (numbers, complex, small arrays)."""
    if isinstance(v, str) and v in ("nan", "inf", "-inf"):
        return float(v)
    if isinstance(v, list):
        return [from_jsonable(x) for x in v]
    if isinstance(v, dict):
        if set(v) == {"complex"}:
            re_, im = (from_jsonable(x) for x in v["complex"])
            return complex(re_, im)
        if set(v) == {"ndarray", "shape", "dtype"}:
            flat = [from_jsonable(x) for x in v["ndarray"]]
            return np.array(flat, dtype=v["dtype"]).reshape(v["shape"])
        return {k: from_jsonable(x) for k, x in v.items()}
    return v


def is_exact(v: Any) -> bool:
    """True if ``from_jsonable(jsonable(v))`` gives back ``v`` (used before a function is re-called
    by ``reproduce()``: arguments stored as summaries or repr cannot be replayed)."""
    j = jsonable(v)

    def ok(x: Any) -> bool:
        if isinstance(x, dict):
            if "repr" in x and len(x) == 1 or "array" in x and len(x) == 1:
                return False
            return all(ok(y) for y in x.values())
        if isinstance(x, list):
            return all(ok(y) for y in x)
        return True
    return ok(j)


def canonical(v: Any) -> bytes:
    """Deterministic bytes of ``v`` for hashing: sorted keys, floats by ``float.hex`` (exact),
    arrays by (shape, dtype, sha256 of the bytes)."""
    def enc(x: Any) -> Any:
        if isinstance(x, (float, np.floating)):
            return {"f": float(x).hex()}
        if isinstance(x, (complex, np.complexfloating)):
            return {"c": [complex(x).real.hex(), complex(x).imag.hex()]}
        if isinstance(x, np.ndarray):
            return {"a": [list(x.shape), str(x.dtype), array_sha256(x)]}
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, Mapping):
            return {str(k): enc(y) for k, y in sorted(x.items(), key=lambda kv: str(kv[0]))}
        if isinstance(x, (list, tuple)):
            return [enc(y) for y in x]
        if x is None or isinstance(x, (bool, int, str)):
            return x
        return {"r": repr(x)}
    return json.dumps(enc(v), sort_keys=True, separators=(",", ":")).encode()


def sha256_of(v: Any) -> str:
    """sha256 hex digest of :func:`canonical` bytes."""
    return hashlib.sha256(canonical(v)).hexdigest()


def array_sha256(a: np.ndarray) -> str:
    """sha256 of the raw bytes of a C-contiguous copy (shape and dtype hashed by the caller)."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def file_sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def slug(name: str) -> str:
    """File-name-safe short form of a run name (the id keeps it readable)."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return (s or "run")[:40]


def atomic_write_json(path: Path, obj: Any) -> None:
    """Write JSON next to ``path`` then ``os.replace`` (atomic on POSIX, also on CephFS): a reader
    sees the old file or the new one, never a half-written one."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)


class FileLock:
    """Exclusive advisory lock (``fcntl.flock``) on ``path`` for read-modify-write of one run's
    record by several processes (status changes). Blocking, but only ever taken outside the
    computation (status changes after a run, or by the writer thread)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> FileLock:
        import fcntl
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc: Any) -> None:
        import fcntl
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


def flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """{"a": {"b": 1}} -> {"a.b": 1} (lists kept as values). Used by diffs of nested records."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping) and v and not ({"complex"} == set(v) or "ndarray" in v):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def diff_dicts(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """{changed: {k: [a, b]}, only_a: {k: a}, only_b: {k: b}} on flattened mappings."""
    fa, fb = flatten(a), flatten(b)
    return {"changed": {k: [fa[k], fb[k]] for k in sorted(fa.keys() & fb.keys()) if fa[k] != fb[k]},
            "only_a": {k: fa[k] for k in sorted(fa.keys() - fb.keys())},
            "only_b": {k: fb[k] for k in sorted(fb.keys() - fa.keys())}}
