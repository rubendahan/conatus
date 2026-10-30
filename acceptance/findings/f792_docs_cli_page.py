"""F792 (audit of 0.2.1, N06). Without a copy of the guide, the docs command gives the page address.

Oracle: the address follows the page name; the Studio errors name a page that the export keeps."""
from _acceptance import check, passed, start

start(792, "docs command without a guide")
import contextlib  # noqa: E402
import io  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import conatus.__main__ as cli  # noqa: E402
from conatus.model import watch  # noqa: E402

cli.__file__ = str(Path(tempfile.mkdtemp()) / "a/b/c/__main__.py")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    code = cli._docs("guide/studio")
check(code == 0, f"exit {code}")
check(buf.getvalue().splitlines()[0].endswith("/docs/guide/studio.html"), buf.getvalue())
check("no public site" not in buf.getvalue(), "stale statement")
check(watch._DOC == "docs/site/guide/studio.md", watch._DOC)
passed("page address printed")
