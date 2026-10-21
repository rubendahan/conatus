"""Files that only their owner may read: tokens, audit logs, run telemetry (review F12, 0.1.2).

A file opened with ``open(path, "a")`` gets ``0666 & ~umask``: ``0644`` under the common umask
022, so other accounts on a shared machine could read a run's log, scalars and films, and the
Studio audit (user names, addresses, commands). The helpers below create directories ``0700``
and files ``0600`` whatever the umask, and bring an existing file of ours back to ``0600`` when
it is opened again. An existing directory keeps its mode; files written there from now on
are ``0600``, and older files keep their mode until they are written again.

Stdlib only.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import IO, Any

FILE_MODE = 0o600
DIR_MODE = 0o700


def private_dir(path: str | os.PathLike) -> Path:
    """Create ``path`` (and missing parents) if needed; a directory created here is ``0700``.

    ``Path.mkdir(mode=...)`` is masked by the umask, which can only remove bits, so ``0700``
    stays ``0700``. Parents created on the way get the same mode."""
    p = Path(path)
    missing = []
    q = p
    while not q.exists() and q != q.parent:
        missing.append(q)
        q = q.parent
    for d in reversed(missing):
        try:
            d.mkdir(mode=DIR_MODE)
        except FileExistsError:             # created meanwhile by another process
            continue
    if not p.is_dir():
        raise NotADirectoryError(f"{p} exists and is not a directory")
    return p


def open_private(path: str | os.PathLike, mode: str = "a", encoding: str | None = "utf-8") -> IO[Any]:
    """Open ``path`` for writing (``mode`` among ``a``, ``ab``, ``w``, ``wb``) as a ``0600`` file.

    A new file is created ``0600``. An existing file owned by this account is set to ``0600``
    (a log written by an older version could be ``0644``); a file owned by someone else is left
    as it is (``fchmod`` would fail; the caller writes to it as before)."""
    if mode not in ("a", "ab", "w", "wb"):
        raise ValueError(f"open_private: mode must be a, ab, w or wb, got {mode!r}")
    flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if mode[0] == "a" else os.O_TRUNC)
    flags |= getattr(os, "O_NOFOLLOW", 0)    # a symbolic link planted in a shared spool is refused
    fd = os.open(path, flags, FILE_MODE)
    try:
        st = os.fstat(fd)
        if st.st_mode & 0o077 and st.st_uid == os.getuid():
            os.fchmod(fd, FILE_MODE)
    except (OSError, AttributeError):       # no getuid/fchmod (Windows): the mode at creation
        pass
    if "b" in mode:
        return os.fdopen(fd, mode)
    return os.fdopen(fd, mode, encoding=encoding)
