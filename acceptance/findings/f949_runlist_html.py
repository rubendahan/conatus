"""F949 (review G15): no field of a run record reaches the HTML view of find() unescaped.

Oracle: html.parser. A record whose state or status holds an <img> tag is refused on load or
shown as text; a parameter holding markup is shown as text. Control: a valid record shows its
state and status.
"""
import json
from html.parser import HTMLParser

from _acceptance import check, passed, refused, start

start(949, "RunList HTML escapes every field; bad states refused on load")
import conatus as cn  # noqa: E402


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)


root = "registry"
r = cn.runs.record(name="probe", params={"p": "<img src=x>"}, results={"value": 1.0},
                   compression=None, root=root)
html = cn.runs.find(root=root)._repr_html_()
t = Tags()
t.feed(html)
check("img" not in t.tags, "a parameter's markup reached the HTML")
check("completed" in html and "provisional" in html, "control: state and status not shown")
path = r.path / "run.json"
data = json.loads(path.read_text())
for field in ("state", "status"):
    bad = dict(data)
    bad[field] = '<img src="invalid:">'
    path.write_text(json.dumps(bad))
    try:
        html = cn.runs.find(root=root)._repr_html_()
    except Exception as exc:  # noqa: BLE001 -- classified by refused() below
        ok, _ = refused(lambda e=exc: (_ for _ in ()).throw(e))
        check(ok, f"{field}: untyped error")
        continue
    t = Tags()
    t.feed(html)
    check("img" not in t.tags, f"{field}: markup reached the HTML")
path.write_text(json.dumps(data))
passed("markup in params, state and status never reaches the HTML")
