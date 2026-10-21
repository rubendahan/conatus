"""F541 (external review of 0.1.1, F11): the local Studio's session cookie is per port and bound
to its Host.

Browsers send a cookie to every port of its host name. The local hub set ``cn_session`` on
127.0.0.1, and that cookie, received by another local service, authenticated at the hub.
Expected: the cookie is named ``cn_session_<port>``; the session answers only at the
``Host`` it was opened at; ``*.localhost`` names are accepted as loopback, so the Studio can
have a host name of its own. SKIP where the Studio hub is not shipped.
"""
import asyncio
from http.cookies import SimpleCookie

from _acceptance import check, need, passed, start

start(541, "Studio session cookie named per port and bound to its Host")
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
        codes = {}
        # the other port's probe carries the cookie under that port's name, so the Host binding
        # (not the cookie name) is what refuses it
        for label, host, name in (("own", f"127.0.0.1:{port}", names[0]),
                                  ("other port", f"127.0.0.1:{port + 1}", f"cn_session_{port + 1}"),
                                  ("other name", f"localhost:{port}", names[0])):
            rr = await c.fetch(base + "/api/runs", raise_error=False,
                               headers={"Cookie": f"{name}={value}", "Host": host})
            codes[label] = rr.code
        alias = (await c.fetch(base + "/login", raise_error=False,
                               headers={"Host": f"studio.localhost:{port}"})).code
    finally:
        server.stop()
    return port, names, codes, alias


port, names, codes, alias = asyncio.run(go())
check(names == [f"cn_session_{port}"], f"cookie names {names}, expected cn_session_{port}")
check(codes == {"own": 200, "other port": 401, "other name": 401}, f"replay codes {codes}")
check(alias == 200, f"studio.localhost refused ({alias})")
passed(f"cookie {names[0]}; own Host 200, other port/name 401; studio.localhost accepted")
