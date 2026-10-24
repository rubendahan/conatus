"""F541 (external review of 0.1.1, F11): the local Studio's session cookie alone does not
authenticate.

Browsers send a cookie to every port of its host name. The local hub set ``cn_session`` on
127.0.0.1, and that cookie, received by another local service and replayed with the hub's
``Host``, authenticated at the hub. Expected: the cookie is named ``cn_session_<port>``; the
session answers only at the ``Host`` it was opened at; data needs the session key handed to the
page at sign-in (``X-CN-Session-Key``), so the replayed cookie alone gets 401; ``*.localhost``
names are accepted as loopback. SKIP where the Studio hub is not shipped.
"""
import asyncio
import re
from http.cookies import SimpleCookie

from _acceptance import check, need, passed, start

start(541, "Studio session cookie per port, bound to its Host, useless without the session key")
need("conatus.studio.hub")
need("tornado")
from tornado.httpclient import AsyncHTTPClient  # noqa: E402
from tornado.httpserver import HTTPServer  # noqa: E402
from tornado.testing import bind_unused_port  # noqa: E402

from conatus.studio.auth import AuditLog, AuthDB  # noqa: E402
from conatus.studio.hub import Hub, make_app  # noqa: E402


async def go():
    db = AuthDB({})
    db.add_user("u", None)
    tok = db.add_user_token("u")
    hub = Hub(db, audit=AuditLog(None, echo=False), local=True)
    server = HTTPServer(make_app(hub))
    sock, port = bind_unused_port()
    server.add_sockets([sock])
    c, base = AsyncHTTPClient(), f"http://127.0.0.1:{port}"
    try:
        r = await c.fetch(base + "/login", method="POST", body=f"token={tok}", raise_error=False,
                          follow_redirects=False,
                          headers={"Content-Type": "application/x-www-form-urlencoded"})
        jar = SimpleCookie()
        for line in r.headers.get_list("Set-Cookie"):
            jar.load(line)
        names = list(jar)
        value = jar[names[0]].value if names else ""
        m = re.search(rb'name="cn-session-key" content="([A-Za-z0-9_-]+)"', r.body or b"")
        key = m.group(1).decode() if m else ""
        codes = {}
        # the other port's probe carries the cookie under that port's name, so the Host binding
        # (not the cookie name) is what refuses it; "replay" is the review's attack: the cookie
        # alone, with the hub's own Host
        for label, host, name, k in (("own", f"127.0.0.1:{port}", names[0], key),
                                     ("replay", f"127.0.0.1:{port}", names[0], None),
                                     ("other port", f"127.0.0.1:{port + 1}", f"cn_session_{port + 1}", key),
                                     ("other name", f"localhost:{port}", names[0], key)):
            h = {"Cookie": f"{name}={value}", "Host": host}
            if k:
                h["X-CN-Session-Key"] = k
            rr = await c.fetch(base + "/api/runs", raise_error=False, headers=h)
            codes[label] = rr.code
        alias = (await c.fetch(base + "/login", raise_error=False,
                               headers={"Host": f"studio.localhost:{port}"})).code
    finally:
        server.stop()
    return port, names, codes, alias


port, names, codes, alias = asyncio.run(go())
check(names == [f"cn_session_{port}"], f"cookie names {names}, expected cn_session_{port}")
check(codes == {"own": 200, "replay": 401, "other port": 401, "other name": 401},
      f"replay codes {codes}")
check(alias == 200, f"studio.localhost refused ({alias})")
passed(f"cookie {names[0]}; with key 200; cookie alone 401; other port/name 401; studio.localhost accepted")
