"""Authentication for the Studio hub (stdlib only: hashlib.scrypt, hmac, secrets).

Nothing secret is ever stored in clear:

* user passwords  -> ``scrypt$n$r$p$salt$hash`` (hashlib.scrypt, random 16-byte salt);
* user API tokens, agent tokens and relay tokens -> SHA-256 hex digest (the tokens are 256-bit
  random strings, so a fast hash is enough; the clear token is shown once at creation, or written
  straight into a 0600 file, and never again).

Agent tokens (the agent channel, :mod:`conatus.studio.agentapi`) carry a **scope**: ``read`` (state,
scalars, log, frames; every command refused with 403) or ``command`` (also the declared run
commands; never job submission, never the audit of others), and optionally the run keys they may
reach (``fnmatch`` patterns such as ``mycluster:kh_*``). A user's own API token is a full user.

The auth file (JSON, written 0600) holds only these digests, so it can also be passed to a hosted
hub as an environment secret (``CN_HUB_AUTH_JSON``). Sessions live in the hub's memory (a restart
logs everyone out). Every login attempt and every command goes to the append-only audit log.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from conatus._private import open_private, private_dir

# scrypt cost: 16 MiB of memory per hash, ~50 ms on a laptop; stored with the hash so it can change.
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 14, 8, 1
TOKEN_BYTES = 32                     # 256-bit tokens (relay and user API tokens, session cookies)
MIN_PASSWORD = 12                    # refuse short passwords at creation
SCOPES = ("read", "command")         # agent token scopes


@dataclass(frozen=True)
class Principal:
    """Who is calling: a signed-in user (session or own token) or an agent token with a scope."""
    name: str
    kind: str = "user"               # "user" | "agent"
    scope: str = "command"           # "read" | "command"
    runs: tuple[str, ...] | None = None    # agents: fnmatch patterns of the run keys reachable
    session: bool = False            # True: browser session cookie (same-origin rules apply)

    @property
    def label(self) -> str:
        """The name written in the audit log and shown to others (``agent:<name>`` for agents)."""
        return self.name if self.kind == "user" else f"agent:{self.name}"

    def can_see(self, key: str) -> bool:
        """May this principal read the run ``key``?"""
        return self.runs is None or any(fnmatchcase(key, p) or fnmatchcase(key.partition(":")[2], p)
                                        for p in self.runs)

    def can_command(self, key: str) -> bool:
        """May this principal send a declared command to the run ``key``?"""
        return self.scope == "command" and self.can_see(key)


def hash_password(password: str, *, n: int = SCRYPT_N, r: int = SCRYPT_R, p: int = SCRYPT_P) -> str:
    """Salted scrypt hash, self-describing (cost parameters included)."""
    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, maxmem=128 * n * r * p + 2 ** 20)
    b64 = lambda b: base64.b64encode(b).decode("ascii")  # noqa: E731
    return f"scrypt${n}${r}${p}${b64(salt)}${b64(dk)}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check of ``password`` against a :func:`hash_password` string."""
    try:
        algo, n, r, p, salt, want = stored.split("$")
        if algo != "scrypt":
            return False
        n, r, p = int(n), int(r), int(p)
        dk = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=n, r=r, p=p,
                            maxmem=128 * n * r * p + 2 ** 20)
        expected = base64.b64decode(want)       # inside the try: a corrupt entry is a refusal
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(dk, expected)


def new_token() -> str:
    """A fresh 256-bit URL-safe token (shown once to the operator, stored only as a digest)."""
    return secrets.token_urlsafe(TOKEN_BYTES)


def token_digest(token: str) -> str:
    """SHA-256 hex digest of a token (what the auth file stores instead of the token)."""
    return hashlib.sha256(token.encode()).hexdigest()


class AuthDB:
    """Users (password hash + API token digests) and relays (token digest).

    Layout: ``{"users": {name: {"password": str|None, "tokens": [digest, ...]}},
    "relays": {name: digest}, "agents": {name: {"digest", "scope", "runs": [pattern]|null,
    "created"}}}``.
    """

    def __init__(self, data: dict | None = None, path: Path | None = None) -> None:
        data = data or {}
        self.users: dict[str, dict] = dict(data.get("users", {}))
        self.relays: dict[str, str] = dict(data.get("relays", {}))
        self.agents: dict[str, dict] = dict(data.get("agents", {}))
        self.path = path

    # ------------------------------------------------------------------ persistence
    @classmethod
    def load(cls, path: str | Path | None = None, *, env: str = "CN_HUB_AUTH_JSON") -> AuthDB:
        """From a file, else from the environment variable ``env`` (hosted hub secrets)."""
        if path is not None and Path(path).exists():
            return cls(json.loads(Path(path).read_text()), Path(path))
        if os.environ.get(env):
            return cls(json.loads(os.environ[env]), None)
        return cls({}, Path(path) if path is not None else None)

    def to_json(self) -> str:
        """The users and relays as indented, key-sorted JSON (hashes and digests only)."""
        return json.dumps({"users": self.users, "relays": self.relays, "agents": self.agents},
                          indent=1, sort_keys=True)

    def save(self) -> None:
        """Write atomically to ``path`` with mode 0600; ValueError if loaded from the environment."""
        if self.path is None:
            raise ValueError("this AuthDB came from the environment; nowhere to save it")
        private_dir(self.path.parent)
        # a fresh file, created exclusively and without following links (security audit
        #): the fixed name auth.tmp, opened O_TRUNC, followed a link planted there
        # (overwriting its target) and kept the mode of an existing auth.tmp (0644: hashes readable)
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(self.to_json())
            os.replace(tmp, self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    # ------------------------------------------------------------------ management
    def add_user(self, name: str, password: str | None) -> None:
        """Add a user or reset their password (None: token-only user); checks name and length."""
        _check_name(name)
        if password is not None and len(password) < MIN_PASSWORD:
            raise ValueError(f"password too short (< {MIN_PASSWORD} characters)")
        entry = self.users.setdefault(name, {"password": None, "tokens": []})
        entry["password"] = hash_password(password) if password is not None else None

    def add_user_token(self, name: str) -> str:
        """Issue a new API token for an existing user; returns it (only its digest is kept)."""
        if name not in self.users:
            raise KeyError(f"unknown user {name!r}")
        tok = new_token()
        self.users[name]["tokens"].append(token_digest(tok))
        return tok

    def add_relay(self, name: str) -> str:
        """Add or re-key a relay; returns its new token (only its digest is kept)."""
        _check_name(name)
        tok = new_token()
        self.relays[name] = token_digest(tok)
        return tok

    def add_agent(self, name: str, scope: str, runs: list[str] | None = None) -> str:
        """Add or re-key an agent token of scope ``read`` or ``command``, optionally limited to
        the run keys matching ``runs``; returns the token (only its digest is kept)."""
        _check_name(name)
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
        if runs is not None and (not runs or not all(isinstance(r, str) and 0 < len(r) <= 200
                                                     for r in runs)):
            raise ValueError("runs: a non-empty list of patterns such as 'mycluster:kh_*'")
        tok = new_token()
        self.agents[name] = {"digest": token_digest(tok), "scope": scope,
                             "runs": list(runs) if runs is not None else None,
                             "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        return tok

    # ------------------------------------------------------------------ checks (constant time)
    def check_password(self, name: str, password: str) -> bool:
        """True iff the password matches; unknown users cost the same hashing time."""
        stored = self.users.get(name, {}).get("password")
        if not stored:
            verify_password(password, _dummy_hash())        # same cost whether the user exists or not
            return False
        return verify_password(password, stored)

    def user_for_token(self, token: str) -> str | None:
        """The user owning this API token, else None (compares every digest, constant time)."""
        d = token_digest(token)
        found = None
        for name, entry in self.users.items():
            for want in entry.get("tokens", []):
                if hmac.compare_digest(d, want):
                    found = name
        return found

    def principal_for_token(self, token: str) -> Principal | None:
        """Who owns this bearer token: a user (full rights) or an agent (its scope and runs);
        None if nobody (every digest is compared, constant time)."""
        user = self.user_for_token(token)
        if user is not None:
            return Principal(user)
        d = token_digest(token)
        found = None
        for name, entry in self.agents.items():
            if hmac.compare_digest(d, str(entry.get("digest", ""))):
                found = Principal(name, "agent", entry.get("scope", "read"),
                                  tuple(entry["runs"]) if entry.get("runs") else None)
        return found

    def relay_for_token(self, token: str) -> str | None:
        """The relay owning this token, else None (compares every digest, constant time)."""
        d = token_digest(token)
        found = None
        for name, want in self.relays.items():
            if hmac.compare_digest(d, want):
                found = name
        return found


_DUMMY: list[str] = []


def _dummy_hash() -> str:
    """A throw-away hash, computed once, so unknown users cost the same time as known ones."""
    if not _DUMMY:
        _DUMMY.append(hash_password(secrets.token_hex(8)))
    return _DUMMY[0]


def _check_name(name: str) -> None:
    ok = name and len(name) <= 64 and all(c.isalnum() or c in "-_." for c in name)
    if not ok:
        raise ValueError(f"name must be 1-64 characters of [A-Za-z0-9._-], got {name!r}")


class Sessions:
    """In-memory browser sessions: random cookie value -> (user, expiry, origin).

    ``origin`` is the ``Host`` (name and port) the session was opened at; the session answers only
    there (review F11, 0.1.2). This keeps a cookie that a browser sends to another port or name
    from opening a session there. It does not stop a program that has captured the cookie and
    sends it with the right ``Host``: see docs/site/guide/studio.md (F11)."""

    def __init__(self, ttl_s: float = 12 * 3600) -> None:
        self.ttl_s = float(ttl_s)
        self._s: dict[str, tuple[str, float, str | None]] = {}

    def create(self, user: str, origin: str | None = None) -> str:
        """Open a session for ``user`` at ``origin`` (a ``Host`` value, or None: any); returns
        the cookie value (only its digest is kept). Expired sessions are dropped here too (a
        session never used again was kept forever)."""
        now = time.time()
        self._s = {d: v for d, v in self._s.items() if v[1] >= now}
        tok = new_token()
        self._s[token_digest(tok)] = (user, now + self.ttl_s, origin.lower() if origin else None)
        return tok

    def user(self, tok: str | None, origin: str | None = None) -> str | None:
        """The user of a live session cookie presented at ``origin``, else None (expired
        sessions are dropped; a session opened at another ``Host`` answers None)."""
        if not tok:
            return None
        hit = self._s.get(token_digest(tok))
        if hit is None:
            return None
        if hit[1] < time.time():
            self._s.pop(token_digest(tok), None)
            return None
        if hit[2] is not None and (origin or "").lower() != hit[2]:
            return None
        return hit[0]

    def drop(self, tok: str | None) -> None:
        """End the session of this cookie value, if any."""
        if tok:
            self._s.pop(token_digest(tok), None)


class LoginThrottle:
    """At most ``max_failures`` failed logins per client key in a sliding ``window_s``."""

    def __init__(self, max_failures: int = 5, window_s: float = 900.0) -> None:
        self.max_failures, self.window_s = int(max_failures), float(window_s)
        self._f: dict[str, list[float]] = {}

    def blocked(self, key: str) -> bool:
        """True if ``key`` has reached ``max_failures`` within the window (prunes old failures)."""
        now = time.time()
        recent = [t for t in self._f.get(key, []) if now - t < self.window_s]
        if recent:
            self._f[key] = recent
        else:               # no empty entry per address ever seen (unbounded growth otherwise)
            self._f.pop(key, None)
        return len(recent) >= self.max_failures

    def failure(self, key: str) -> None:
        """Record one failed login for ``key``."""
        self._f.setdefault(key, []).append(time.time())

    def success(self, key: str) -> None:
        """Forget the failures of ``key`` after a successful login."""
        self._f.pop(key, None)


class AuditLog:
    """Append-only JSONL log (who, when, what, result); also echoed to stdout for hosted logs."""

    def __init__(self, path: str | Path | None, echo: bool = True) -> None:
        self.path = Path(path) if path is not None else None
        self.echo = echo
        self.error: str | None = None

    def write(self, **entry: Any) -> dict:
        """Append one timestamped entry (file and stdout; stdout only if the file fails); returns it."""
        entry = {"wall": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **entry}
        line = json.dumps(entry, sort_keys=True)
        if self.path is not None:
            try:
                private_dir(self.path.parent)
                with open_private(self.path, "a") as fh:  # 0600: names, addresses (F12)
                    fh.write(line + "\n")
            except OSError as err:              # never lose the entry: it still goes to stdout
                if self.error is None:
                    print(f"AUDIT FILE UNWRITABLE ({err}); audit continues on stdout only", flush=True)
                self.error = str(err)
                print("AUDIT " + line, flush=True)
                return entry
        if self.echo:
            print("AUDIT " + line, flush=True)
        return entry
