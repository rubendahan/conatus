"""Review F11 (external review 0.1.1): the scope of the Studio session cookie.

Browsers key cookies by host name, not by port. The local hub set ``cn_session`` on 127.0.0.1,
so every service on 127.0.0.1 (any port) received it, and a second Studio on another port used
the same cookie name. Expected after the fix:

* local hub: the cookie is ``cn_session_<port>`` (port of the ``Host`` the browser used);
* hosted hub (HTTPS): ``__Host-cn_session`` (host-only, Secure, ``Path=/``, enforced by browsers);
* a session answers only at the ``Host`` it was opened at;
* names under ``.localhost`` are accepted as loopback, so a Studio can be given a host name
  of its own; other names (DNS rebinding) stay refused.

What these measures do not stop (a program that captures the cookie on the same host name and
replays it) is stated in docs/audits/security.md; the last test pins that statement.
"""
from __future__ import annotations

import asyncio
from http.cookies import SimpleCookie
from pathlib import Path

import pytest

pytest.importorskip("tornado")
from tornado.httpclient import AsyncHTTPClient  # noqa: E402
from tornado.httpserver import HTTPServer  # noqa: E402
from tornado.testing import bind_unused_port  # noqa: E402

from conatus.studio import webapp  # noqa: E402
from conatus.studio.auth import AuditLog, AuthDB, Sessions  # noqa: E402
from conatus.studio.hub import Hub, make_app  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FORM = {"Content-Type": "application/x-www-form-urlencoded"}


def test_cookie_names():
    assert webapp.session_cookie_name("127.0.0.1:8765", secure=False) == "cn_session_8765"
    assert webapp.session_cookie_name("localhost:9000", secure=False) == "cn_session_9000"
    assert webapp.session_cookie_name("[::1]:8765", secure=False) == "cn_session_8765"
    assert webapp.session_cookie_name("localhost", secure=False) == "cn_session"
    assert webapp.session_cookie_name("studio.example.org", secure=True) == "__Host-cn_session"


def test_loopback_names_include_dot_localhost_only():
    for host in ("studio.localhost", "studio.localhost:8765", "a-1.b.localhost:1", "LOCALHOST:2"):
        assert webapp.loopback_host(host), host
    for host in ("localhost.evil.example", "evil.example", "studio.localhost.evil.example:80",
                 ".localhost", "-x.localhost", "a..localhost", "node07"):
        assert not webapp.loopback_host(host), host


def test_sessions_are_bound_to_their_host():
    s = Sessions()
    tok = s.create("u", origin="127.0.0.1:8765")
    assert s.user(tok, origin="127.0.0.1:8765") == "u"
    assert s.user(tok, origin="127.0.0.1:8766") is None
    assert s.user(tok, origin="localhost:8765") is None
    assert s.user(tok, origin=None) is None
    assert s.user(Sessions().create("v"), origin="anything") is None    # another table


def _login_and_probe(hosted: bool = False):
    async def go():
        db = AuthDB({})
        db.add_user("u", None)
        tok = db.add_user_token("u")
        hub = Hub(db, audit=AuditLog(None, echo=False), local=not hosted)
        server = HTTPServer(make_app(hub), xheaders=hosted)
        sock, port = bind_unused_port()
        server.add_sockets([sock])
        c, base = AsyncHTTPClient(), f"http://127.0.0.1:{port}"
        extra = {"X-Forwarded-Proto": "https"} if hosted else {}
        try:
            r = await c.fetch(base + "/login", method="POST", body=f"token={tok}", raise_error=False,
                              follow_redirects=False, headers={**FORM, **extra})
            jar = SimpleCookie()
            for line in r.headers.get_list("Set-Cookie"):
                jar.load(line)
            out = {"login": r.code, "cookies": {k: dict(v) for k, v in jar.items()}, "port": port}
            if not hosted:
                (name, morsel), = jar.items()
                cookie = f"{name}={morsel.value}"

                async def runs(host):
                    rr = await c.fetch(base + "/api/runs", raise_error=False,
                                       headers={"Cookie": cookie, "Host": host})
                    return rr.code
                out["same"] = await runs(f"127.0.0.1:{port}")
                out["other_name"] = await runs(f"localhost:{port}")
                # another port, with the cookie under that port's name: the Host binding refuses it
                rr = await c.fetch(base + "/api/runs", raise_error=False,
                                   headers={"Cookie": f"cn_session_{port + 1}={morsel.value}",
                                            "Host": f"127.0.0.1:{port + 1}"})
                out["other_port"] = rr.code
                # the old name, as a second service would replay it under its own port's name
                rr = await c.fetch(base + "/api/runs", raise_error=False,
                                   headers={"Cookie": f"cn_session={morsel.value}"})
                out["old_name"] = rr.code
        finally:
            server.stop()
        return out
    return asyncio.run(asyncio.wait_for(go(), 60))


def test_local_hub_sets_a_per_port_cookie_bound_to_its_host():
    out = _login_and_probe()
    assert out["login"] == 302
    assert list(out["cookies"]) == [f"cn_session_{out['port']}"]
    attrs = out["cookies"][f"cn_session_{out['port']}"]
    assert attrs["httponly"] and attrs["samesite"] == "Strict" and attrs["path"] == "/"
    assert out["same"] == 200
    assert out["other_name"] == 401 and out["other_port"] == 401 and out["old_name"] == 401


def test_hosted_hub_uses_the_host_prefix():
    out = _login_and_probe(hosted=True)
    assert out["login"] == 302 and list(out["cookies"]) == ["__Host-cn_session"]
    attrs = out["cookies"]["__Host-cn_session"]
    assert attrs["secure"] and attrs["path"] == "/" and not attrs["domain"]


def test_residual_risk_is_documented():
    audit = REPO / "docs" / "audits" / "security.md"
    if not audit.exists():                      # the public tree does not ship docs/audits
        pytest.skip("docs/audits/security.md is not in this tree")
    text = audit.read_text()
    assert "F11" in text and "same host name" in text and ".localhost" in text
