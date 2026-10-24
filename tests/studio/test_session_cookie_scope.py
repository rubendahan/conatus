"""Review F11 (external review 0.1.1): the scope of the Studio session cookie.

Browsers key cookies by host name, not by port. The local hub set ``cn_session`` on 127.0.0.1,
so every service on 127.0.0.1 (any port) received it, and a second Studio on another port used
the same cookie name. Expected after the fix:

* local hub: the cookie is ``cn_session_<port>`` (port of the ``Host`` the browser used);
* hosted hub (HTTPS): ``__Host-cn_session`` (host-only, Secure, ``Path=/``, enforced by browsers);
* a session answers only at the ``Host`` it was opened at;
* names under ``.localhost`` are accepted as loopback, so a Studio can be given a host name
  of its own; other names (DNS rebinding) stay refused.

The cookie alone is not enough (closing F11): a program that captures the cookie on the same
host name and replays it with the right ``Host`` is refused, because data, commands and ``/ws``
also need the session key that the app keeps in the storage of its own origin. The tests below
replay the attack of the review and get 401.
"""
from __future__ import annotations

import asyncio
import json
import re
from http.cookies import SimpleCookie
from pathlib import Path

import pytest

pytest.importorskip("tornado")
from tornado.httpclient import AsyncHTTPClient, HTTPClientError, HTTPRequest  # noqa: E402
from tornado.httpserver import HTTPServer  # noqa: E402
from tornado.testing import bind_unused_port  # noqa: E402
from tornado.websocket import websocket_connect  # noqa: E402

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
    tok, key = s.create("u", origin="127.0.0.1:8765")
    assert s.user(tok, origin="127.0.0.1:8765", key=key) == "u"
    assert s.user(tok, origin="127.0.0.1:8766", key=key) is None
    assert s.user(tok, origin="localhost:8765", key=key) is None
    assert s.user(tok, origin=None, key=key) is None
    tok2, key2 = Sessions().create("v")
    assert s.user(tok2, origin="anything", key=key2) is None            # another table


def test_a_session_needs_its_key_except_for_pages():
    s = Sessions()
    tok, key = s.create("u", origin="127.0.0.1:8765")
    _, other_key = s.create("w", origin="127.0.0.1:8765")
    assert key != tok and len(key) >= 43                                # 256 bits, not the cookie
    assert s.user(tok, origin="127.0.0.1:8765") is None                 # the cookie alone
    assert s.user(tok, origin="127.0.0.1:8765", key="") is None
    assert s.user(tok, origin="127.0.0.1:8765", key=tok) is None        # the cookie as the key
    assert s.user(tok, origin="127.0.0.1:8765", key=other_key) is None  # another session's key
    assert s.user(tok, origin="127.0.0.1:8765", page=True) == "u"       # pages: no data
    assert s.user(tok, origin="127.0.0.1:8766", page=True) is None      # still bound to the Host


def test_ws_key_helpers():
    assert webapp.ws_session_key("cn.v1, cn-key.abc_DEF-1") == "abc_DEF-1"
    assert webapp.ws_session_key(["cn.v1"]) is None and webapp.ws_session_key(None) is None
    assert webapp.ws_session_key("cn-key.") is None
    assert webapp.ws_select_protocol(["cn.v1", "cn-key.secret"]) == "cn.v1"
    assert webapp.ws_select_protocol(["other"]) is None


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
            m = re.search(rb'<meta name="cn-session-key" content="([A-Za-z0-9_-]+)">', r.body)
            out["key_in_page"] = bool(m)
            out["login_script"] = b'<script src="/login-done.js">' in r.body
            if not hosted:
                (name, morsel), = jar.items()
                cookie, key = f"{name}={morsel.value}", m.group(1).decode()
                out["key_elsewhere"] = key in (r.headers.get("Location", "")
                                               + "".join(r.headers.get_list("Set-Cookie")))

                async def get(path, host, with_key=True):
                    h = {"Cookie": cookie, "Host": host}
                    if with_key:
                        h["X-CN-Session-Key"] = key
                    rr = await c.fetch(base + path, raise_error=False, follow_redirects=False,
                                       headers=h)
                    return rr.code
                out["same"] = await get("/api/runs", f"127.0.0.1:{port}")
                out["other_name"] = await get("/api/runs", f"localhost:{port}")
                # another port, with the cookie under that port's name: the Host binding refuses it
                rr = await c.fetch(base + "/api/runs", raise_error=False,
                                   headers={"Cookie": f"cn_session_{port + 1}={morsel.value}",
                                            "Host": f"127.0.0.1:{port + 1}",
                                            "X-CN-Session-Key": key})
                out["other_port"] = rr.code
                # the old name, as a second service would replay it under its own port's name
                rr = await c.fetch(base + "/api/runs", raise_error=False,
                                   headers={"Cookie": f"cn_session={morsel.value}"})
                out["old_name"] = rr.code
                # review F11, the attack itself: a service on the same host name received the
                # cookie and replays it with the hub's own Host, without the key
                own = f"127.0.0.1:{port}"
                out["replay"] = {p: await get(p, own, with_key=False)
                                 for p in ("/api/runs", "/api/whoami", "/api/v1/runs",
                                           "/api/film/x", "/api/compare?runs=x&key=e")}
                rr = await c.fetch(base + "/api/control/local:x", method="POST", raise_error=False,
                                   body='{"cmd": "pause"}',
                                   headers={"Cookie": cookie, "Content-Type": "application/json"})
                out["replay_command"] = rr.code
                rr = await c.fetch(base + "/api/v1/runs/x/commands", method="POST", raise_error=False,
                                   body='{"cmd": "stop", "confirm": true}',
                                   headers={"Cookie": cookie, "Content-Type": "application/json"})
                out["replay_v1_command"] = rr.code
                try:
                    ws = await websocket_connect(HTTPRequest(f"ws://{own}/ws",
                                                             headers={"Cookie": cookie}))
                    ws.close()
                    out["replay_ws"] = 101
                except HTTPClientError as err:
                    out["replay_ws"] = err.code
                # the cookie still opens the app's pages (static code, no data)
                out["page_cookie_only"] = await get("/", own, with_key=False)
                # the app's own WebSocket: the key in the subprotocol list, "cn.v1" answered
                ws = await websocket_connect(HTTPRequest(
                    f"ws://{own}/ws", headers={"Cookie": cookie}), subprotocols=["cn.v1", f"cn-key.{key}"])
                out["ws_app"] = (ws.selected_subprotocol, json.loads(await ws.read_message())["type"])
                ws.close()
                # sign-out needs the key too: the replayed cookie cannot end the session
                await c.fetch(base + "/logout", method="POST", body="", raise_error=False,
                              follow_redirects=False, headers={"Cookie": cookie})
                out["after_replayed_logout"] = await get("/api/runs", own)
                await c.fetch(base + "/logout", method="POST", body="", raise_error=False,
                              follow_redirects=False, headers={"Cookie": cookie, "X-CN-Session-Key": key})
                out["after_logout"] = await get("/api/runs", own)
                js = await c.fetch(base + "/login-done.js", raise_error=False)
                out["login_js"] = (js.code, js.headers.get("Content-Type", ""),
                                   b"localStorage" in js.body and b"location.replace" in js.body)
        finally:
            server.stop()
        return out
    return asyncio.run(asyncio.wait_for(go(), 60))


def test_local_hub_sets_a_per_port_cookie_bound_to_its_host():
    out = _login_and_probe()
    assert out["login"] == 200 and out["key_in_page"] and out["login_script"]
    assert list(out["cookies"]) == [f"cn_session_{out['port']}"]
    attrs = out["cookies"][f"cn_session_{out['port']}"]
    assert attrs["httponly"] and attrs["samesite"] == "Strict" and attrs["path"] == "/"
    assert out["same"] == 200
    assert out["other_name"] == 401 and out["other_port"] == 401 and out["old_name"] == 401


def test_replayed_cookie_without_the_key_is_refused():
    """Review F11 replayed: before, the captured cookie sent with the hub's ``Host`` got 200 on
    /api/runs. Now every data route, both command routes and /ws answer 401 without the key."""
    out = _login_and_probe()
    assert not out["key_elsewhere"]                     # the key is in the page only
    assert out["replay"] == dict.fromkeys(out["replay"], 401), out["replay"]
    assert out["replay_command"] == 401 and out["replay_v1_command"] == 401
    assert out["replay_ws"] == 401
    assert out["page_cookie_only"] == 200
    assert out["ws_app"] == ("cn.v1", "snapshot")
    assert out["after_replayed_logout"] == 200 and out["after_logout"] == 401
    assert out["login_js"][0] == 200 and out["login_js"][1].startswith("text/javascript")
    assert out["login_js"][2]


def test_hosted_hub_uses_the_host_prefix():
    out = _login_and_probe(hosted=True)
    assert out["login"] == 200 and list(out["cookies"]) == ["__Host-cn_session"]
    attrs = out["cookies"]["__Host-cn_session"]
    assert attrs["secure"] and attrs["path"] == "/" and not attrs["domain"]


def test_session_key_is_documented():
    audit = REPO / "docs" / "audits" / "security.md"
    if not audit.exists():                      # the public tree does not ship docs/audits
        pytest.skip("docs/audits/security.md is not in this tree")
    text = audit.read_text()
    assert "F11" in text and "session key" in text and "X-CN-Session-Key" in text
