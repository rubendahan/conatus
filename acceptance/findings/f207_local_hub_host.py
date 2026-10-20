"""F207 (security review 2026-10-20): `hub serve --local` answers only a loopback Host.

DNS rebinding: a page of evil.example whose name is rebound to 127.0.0.1 reached the local hub
with Host: evil.example. Expected: 403 for it, 200 for 127.0.0.1 / localhost. SKIP where the
Studio hub is not shipped (the public wheel).
"""
import asyncio

from _acceptance import check, need, passed, start

start(207, "local hub refuses a non-loopback Host")
need("conatus.studio.hub")
need("tornado")
from tornado.httpclient import AsyncHTTPClient  # noqa: E402
from tornado.httpserver import HTTPServer  # noqa: E402
from tornado.testing import bind_unused_port  # noqa: E402

from conatus.studio.auth import AuditLog, AuthDB  # noqa: E402
from conatus.studio.hub import Hub, make_app  # noqa: E402


async def go():
    hub = Hub(AuthDB({}), audit=AuditLog(None, echo=False), local=True)
    server = HTTPServer(make_app(hub))
    sock, port = bind_unused_port()
    server.add_sockets([sock])
    c, base = AsyncHTTPClient(), f"http://127.0.0.1:{port}"
    try:
        evil = (await c.fetch(base + "/login", raise_error=False, headers={"Host": f"evil.example:{port}"})).code
        own = (await c.fetch(base + "/login", raise_error=False, headers={"Host": f"localhost:{port}"})).code
    finally:
        server.stop()
    return evil, own


evil, own = asyncio.run(go())
check(evil == 403 and own == 200, f"Host evil.example -> {evil}, localhost -> {own}")
passed("rebound Host refused")
