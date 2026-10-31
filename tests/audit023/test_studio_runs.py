"""One odd run never hides the others, and a list cut by the hub's bound says so. The oracle for a
message is a strict JSON parser: JSON.parse in a browser refuses a bare NaN, which Python accepts."""
import json
import types

import pytest

pytest.importorskip("tornado")

from conatus.studio import agentapi  # noqa: E402
from conatus.studio.auth import AuthDB  # noqa: E402
from conatus.studio.hub import Hub  # noqa: E402
from conatus.telemetry.protocol import strict_dumps, strict_loads  # noqa: E402
from conatus.telemetry.sink import Telemetry  # noqa: E402


def browser_parse(text):
    """Refuses NaN and Infinity as JSON.parse does."""
    def no(c):
        raise ValueError(c)
    return json.loads(text, parse_constant=no)


def hub(**kw):
    return Hub(AuthDB({"users": {}, "agents": {}}, None), local=True,
               audit=types.SimpleNamespace(write=lambda **_: None), **kw)


def lines(h, run, *records):
    data = "\n".join(r if isinstance(r, str) else json.dumps(r) for r in records) + "\n"
    start = h.cursors.get(("a", run), 0)
    h.ingest_lines("a", {"run_id": run, "start": start, "end": start + len(data.encode()),
                         "data": data})


def test_bare_nan_in_a_header_leaves_every_run_listed():
    h = hub()
    lines(h, "ok", {"kind": "header", "run_id": "ok", "meta": {}})
    lines(h, "bad", '{"kind":"header","run_id":"bad","meta":{"params":{"Ra":NaN,"Pr":Infinity}}}')
    snap = browser_parse(strict_dumps(h.snapshot("u"), separators=(",", ":")))
    assert len(snap["runs"]) == 2
    meta = next(r for k, r in snap["runs"].items() if "bad" in k)["summary"]["header"]["meta"]
    assert meta["params"] == {"Ra": "nan", "Pr": "inf"}          # shown as such


def test_writer_stores_a_non_finite_meta_as_a_string(tmp_path):
    t = Telemetry("r1", spool=tmp_path, capacity=16, scalar_interval=0.0, frame_interval=0.0,
                  max_px=8, meta={"params": {"Ra": float("nan"), "Q": [float("-inf"), 1.0]}})
    t.close()
    first = next(p for p in tmp_path.iterdir() if p.suffix in (".jsonl", ".ndjson")
                 and "control" not in p.name and "film" not in p.name)
    head = browser_parse(first.read_text().splitlines()[0])
    assert head["meta"]["params"] == {"Ra": "nan", "Q": ["-inf", 1.0]}


def test_huge_number_in_a_line_is_not_an_infinity_downstream():
    assert strict_loads('{"a": 1e999, "b": [-1e999, 2]}') == {"a": "inf", "b": ["-inf", 2]}
    browser_parse(strict_dumps({"x": float("inf")}))


def test_malformed_status_keeps_the_listing():
    """L1: a counter that is not a number breaks neither the run's own view nor the others."""
    h = hub()
    lines(h, "ok", {"kind": "header", "run_id": "ok"})
    lines(h, "bad", {"kind": "status", "run_id": "bad", "dropped_records": "x", "dropped_frames": []})
    views = [h.run_view(k, brief=True) for k in h.runs]
    assert len(views) == 2
    assert all(v["state"] != "unreadable" for v in views)
    for k in h.runs:
        browser_parse(strict_dumps(h.run_view(k)))


def test_run_view_failure_is_confined_to_its_run(monkeypatch):
    h = hub()
    lines(h, "ok", {"kind": "header", "run_id": "ok"})
    lines(h, "bad", {"kind": "header", "run_id": "bad"})
    real = agentapi.run_view

    def boom(key, st, **kw):
        if "bad" in key:
            raise TypeError("unreadable")
        return real(key, st, **kw)
    monkeypatch.setattr(agentapi, "run_view", boom)
    states = [h.run_view(k, brief=True)["state"] for k in h.runs]
    assert sorted(v == "unreadable" for v in states) == [False, True]


def test_run_limit_is_stated():
    h = hub(max_runs=3)
    for i in range(5):
        lines(h, f"r{i}", {"kind": "header", "run_id": f"r{i}"})
    assert len(h.runs) == 3
    assert h.snapshot("u")["runs_limit"] == {"max": 3, "dropped": 2}
