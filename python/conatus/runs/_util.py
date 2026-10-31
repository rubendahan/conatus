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
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

#: arrays up to this many elements are stored inline in ``run.json`` (as lists); larger arrays in
#: parameters are replaced by a tagged summary {"array": {shape, dtype, sha256}} (declared)
INLINE_ARRAY_MAX = 64
#: strings that stand for non-finite floats in the JSON record: a *string* with one of these
#: values is stored tagged, {"str": ...}, so that it reads back as a string
_FLOAT_WORDS = ("nan", "inf", "-inf")
#: key sets of the tagged encodings: a user dict with one of these key sets is stored wrapped,
#: {"dict": {...}}, so that it reads back as a dict (not as a complex number or an array)
#: dtypes of the inline arrays jsonable writes (numbers and booleans; never object)
_INLINE_DTYPES = {"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
                  "float16", "float32", "float64", "complex64", "complex128"}
_TAGS = ({"complex"}, {"ndarray", "shape", "dtype"}, {"array"}, {"repr"}, {"str"}, {"dict"})


def jsonable(v: Any) -> Any:
    """JSON-safe copy of ``v``, exact for numbers (floats kept as Python floats: JSON round-trips
    them exactly via repr). Non-finite floats become the strings "nan", "inf", "-inf"; complex
    numbers {"complex": [re, im]}; big arrays a hashed summary; unknown objects {"repr": ...}.
    A string "nan", "inf" or "-inf" is stored {"str": ...} and a dict whose keys are those of a
    tag {"dict": ...}, so that :func:`from_jsonable` gives back the same type."""
    if isinstance(v, str):
        return {"str": v} if v in _FLOAT_WORDS else v
    if v is None or isinstance(v, bool):
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
        d = {str(k): jsonable(x) for k, x in v.items()}
        return {"dict": d} if set(d) in _TAGS else d
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
        if set(v) == {"str"} and isinstance(v["str"], str):
            return v["str"]
        if set(v) == {"dict"} and isinstance(v["dict"], dict):
            return {k: from_jsonable(x) for k, x in v["dict"].items()}
        if set(v) == {"complex"}:
            re_, im = (from_jsonable(x) for x in v["complex"])
            return complex(re_, im)
        if set(v) == {"ndarray", "shape", "dtype"}:
            # a record is data another account may have written (security):
            # the shape must be that of the values, and an inline array is never longer than
            # what jsonable writes (INLINE_ARRAY_MAX); anything else is refused, not built
            shape, vals = v["shape"], v["ndarray"]
            if (not isinstance(shape, list) or not isinstance(vals, list)
                    or not all(isinstance(n, int) and not isinstance(n, bool) and 0 <= n for n in shape)
                    or len(vals) > INLINE_ARRAY_MAX or int(np.prod(shape, dtype=np.int64)) != len(vals)
                    or str(v["dtype"]) not in _INLINE_DTYPES):
                raise ValueError(f"inline array of shape {shape!r} and dtype {v['dtype']!r} with "
                                 f"{len(vals) if isinstance(vals, list) else '?'} values: not "
                                 "an array this library writes")
            flat = [from_jsonable(x) for x in vals]
            return np.array(flat, dtype=v["dtype"]).reshape(shape)
        return {k: from_jsonable(x) for k, x in v.items()}
    return v


def typed_jsonable(v: Any, _active: set[int] | None = None) -> Any:
    """JSON form of an argument that keeps its Python type (format 2 of ``inputs.args``): a tuple
    {"tuple": [...]}, a list a list, a dict with string keys {"items": [[k, v], ...]} in its own
    order, a NumPy scalar {"npscalar": [dtype, value]}, bytes {"bytes": hex}. Subclasses and other
    objects are {"repr": ...}, which :func:`is_typed_exact` calls inexact."""
    t = type(v)
    if v is None or t in (bool, int, float, complex, str):
        return jsonable(v)
    if isinstance(v, np.generic):
        if v.dtype.kind in "biufc" and str(v.dtype) in _INLINE_DTYPES:
            return {"npscalar": [str(v.dtype), jsonable(v.item())]}
        return {"repr": repr(v)}
    if t is bytes:
        return {"bytes": v.hex()}
    if t is np.ndarray:
        return jsonable(v)
    if t in (list, tuple) or (t is dict and all(type(k) is str for k in v)):
        _active = set() if _active is None else _active
        if id(v) in _active:
            return {"repr": "<a container that contains itself>"}
        _active.add(id(v))
        try:
            if t is dict:
                return {"items": [[k, typed_jsonable(x, _active)] for k, x in v.items()]}
            items = [typed_jsonable(x, _active) for x in v]
        finally:
            _active.discard(id(v))
        return items if t is list else {"tuple": items}
    return {"repr": repr(v)}


def from_typed(v: Any) -> Any:
    """Inverse of :func:`typed_jsonable`. A record is untrusted data: the only values built are
    numbers, strings, containers and inline arrays with a checked dtype and shape."""
    if isinstance(v, list):
        return [from_typed(x) for x in v]
    if isinstance(v, dict):
        if set(v) == {"tuple"}:
            return tuple(from_typed(x) for x in v["tuple"])
        if set(v) == {"items"}:
            return {str(k): from_typed(x) for k, x in v["items"]}
        if set(v) == {"bytes"}:
            return bytes.fromhex(v["bytes"])
        if set(v) == {"npscalar"}:
            dtype, value = v["npscalar"]
            if str(dtype) not in _INLINE_DTYPES:
                raise ValueError(f"NumPy scalar of dtype {dtype!r}: not a value this library writes")
            return np.dtype(str(dtype)).type(from_jsonable(value))
    return from_jsonable(v)


def is_typed_exact(v: Any) -> bool:
    """True if ``from_typed(typed_jsonable(v))`` gives back ``v`` with its types."""
    def ok(x: Any) -> bool:
        if isinstance(x, list):
            return all(ok(y) for y in x)
        if isinstance(x, dict):
            if len(x) == 1 and ("repr" in x or "array" in x):
                return False
            if set(x) == {"items"}:
                return all(ok(y) for _, y in x["items"])
            return all(ok(y) for y in x.values())
        return True
    return ok(typed_jsonable(v))


def is_exact(v: Any) -> bool:
    """True if ``from_jsonable(jsonable(v))`` gives back ``v`` (used before a function is re-called
    by ``reproduce()``: arguments stored as summaries or repr cannot be replayed)."""
    j = jsonable(v)

    def ok(x: Any) -> bool:
        if isinstance(x, dict):
            if set(x) == {"dict"} and isinstance(x["dict"], dict):   # a wrapped user dict
                return all(ok(y) for y in x["dict"].values())
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


def typed_sha256(v: Any, *, ordered: bool = True) -> str:
    """sha256 of a value by content and type: a list and a tuple, a Python and a NumPy scalar, bytes
    and text differ. ``ordered=True`` also separates two orders of a dict. A value made of lists,
    Python numbers and strings hashes as :func:`sha256_of` when ``ordered`` is False, so identities
    without tuples keep the digest stored in existing checkpoints."""
    def enc(x: Any) -> Any:
        if isinstance(x, np.generic) and not isinstance(x, (str, bytes)):
            item = x.item()      # a long double has no Python value: item() returns it unchanged
            return {"n": [str(x.dtype), x.tobytes().hex() if isinstance(item, np.generic) else enc(item)]}
        if isinstance(x, (float,)):
            return {"f": float(x).hex()}
        if isinstance(x, complex):
            return {"c": [x.real.hex(), x.imag.hex()]}
        if isinstance(x, np.ndarray):
            return {"a": [list(x.shape), str(x.dtype), array_sha256(x)]}
        if isinstance(x, bytes):
            return {"b": x.hex()}
        if isinstance(x, Mapping):
            if ordered:
                return {"o": [[str(k), enc(y)] for k, y in x.items()]}
            return {str(k): enc(y) for k, y in sorted(x.items(), key=lambda kv: str(kv[0]))}
        if isinstance(x, tuple):
            return {"t": [enc(y) for y in x]}
        if isinstance(x, list):
            return [enc(y) for y in x]
        if x is None or isinstance(x, (bool, int, str)):
            return x
        return {"r": repr(x)}
    data = json.dumps(enc(v), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


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
    # O_EXCL with a random name: a link planted at a predictable name is never followed
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(obj, indent=1, sort_keys=True, default=str))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


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
        if isinstance(v, Mapping) and v and not ({"complex"} == set(v) or "ndarray" in v
                                                 or {"str"} == set(v)):
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
