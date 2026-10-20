"""The film of a run (Studio rebuild): encoding, geometric budget, index, curves, declarations.

The film keeps frames for replay in ``<run_id>.film.ndjson``: the 8-bit frames are re-encoded
losslessly (Up filter + deflate), the file never exceeds its budget, the thinning is geometric and
declared, and readers index it by offsets without loading it.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from conatus.telemetry import (FILM_ENCODING, FilmIndex, SpoolReader, Telemetry, decode_curve,
                               decode_frame, decode_frame3d, encode_curve, encode_frame,
                               encode_frame3d, film_frame, film_record)
from conatus.telemetry.film import MAX_FRAMES_PER_REQUEST, thin_to_budget
from conatus.telemetry.sink import FILM_GROWTH


def _tel(tmp_path, **kw):
    args = dict(spool=tmp_path, capacity=10000, scalar_interval=0.0, frame_interval=0.0, max_px=64)
    args.update(kw)
    return Telemetry("r", **args)


def test_film_encoding_is_lossless_and_smaller_on_smooth_fields():
    x = np.linspace(0, 2 * np.pi, 200)
    a = np.sin(x[:, None]) * np.cos(2 * x[None, :])
    fr = encode_frame("r", "w", a, max_px=128)
    rec = film_record(fr, i=3, level=1)
    assert rec["encoding"] == FILM_ENCODING and rec["i"] == 3 and rec["level"] == 1
    assert len(rec["data"]) < len(fr["data"]) / 3              # smooth field: >3x smaller
    back = film_frame(rec)
    assert back["data"] == fr["data"] and back["encoding"] == fr["encoding"]   # bit for bit
    assert np.array_equal(decode_frame(back), decode_frame(fr))
    v = np.random.default_rng(0).standard_normal((10, 12, 14))      # 3D, worst case (noise)
    f3 = encode_frame3d("r", "q", v, max_px=48, extent=[[0, 1]] * 3, scale="signed")
    assert np.array_equal(decode_frame3d(film_frame(film_record(f3, i=0, level=0))),
                          decode_frame3d(f3))
    c = encode_curve("r", "E", [1, 2, 3], [1.0, 0.5, 0.25], logy=True)
    assert film_record(c, i=0, level=0)["x"] == [1.0, 2.0, 3.0]   # curves kept as they are


def test_curves_are_declared_and_never_hide_nan():
    c = encode_curve("r", "E", np.arange(10000.0), np.ones(10000), max_points=1000, xlabel="k")
    x, y = decode_curve(c)
    assert c["stride"] == 10 and x[-1] == 9999.0 and "stride 10" in c["lossy"]
    with pytest.raises(ValueError, match="non-finite"):
        encode_curve("r", "E", [1, 2], [1.0, np.nan])


def test_film_budget_is_never_exceeded_and_thinning_is_geometric(tmp_path):
    budget = 60_000
    tel = _tel(tmp_path, film_max_bytes=budget)
    rng = np.random.default_rng(1)
    for k in range(2000):                   # noisy frames (~6 kB each): the budget binds
        # what the writer thread does with each frame it writes (called directly: deterministic)
        tel._film_append("w", encode_frame("r", "w", rng.standard_normal((64, 64)), max_px=64,
                                           t=0.01 * k, iteration=k))
    tel.close()
    size = (tmp_path / "r.film.ndjson").stat().st_size
    assert size <= budget and size == tel.film_bytes
    lines = [json.loads(x) for x in (tmp_path / "r.film.ndjson").read_text().splitlines()]
    levels = [r["level"] for r in lines]
    assert levels == sorted(levels) and levels[-1] >= 2          # the level went up, in order
    assert [r["i"] for r in lines] == list(range(len(lines)))    # film indices are consecutive
    names = [r["name"] for r in json.loads("[" + ",".join(
        (tmp_path / "r.jsonl").read_text().splitlines()) + "]") if r["kind"] == "event"]
    assert "film_level" in names
    # level j holds frames FILM_GROWTH**j apart in the sequence the writer saw (the iteration
    # numbers increase by at least that much between two kept frames of one level)
    for j in range(1, levels[-1] + 1):
        its = [r["iteration"] for r in lines if r["level"] == j]
        assert all(b - a >= FILM_GROWTH ** j for a, b in zip(its, its[1:], strict=False))


def test_full_budget_is_declared(tmp_path):
    tel = _tel(tmp_path, film_max_bytes=3000)          # smaller than two noisy frames
    rng = np.random.default_rng(2)
    for k in range(20):
        tel._film_append("w", encode_frame("r", "w", rng.standard_normal((64, 64)), max_px=64,
                                           t=k, iteration=k))
    tel.close()
    assert tel.film_skipped > 0
    events = [json.loads(x) for x in (tmp_path / "r.jsonl").read_text().splitlines()]
    assert any(e.get("name") == "film_full" for e in events)
    assert "film <= " in tel.explain() and "skipped" in tel.explain()


def test_no_film_unless_declared(tmp_path):
    tel = _tel(tmp_path)
    tel.frame("w", np.ones((4, 4)), force=True)
    tel.close()
    assert not (tmp_path / "r.film.ndjson").exists() and "no film" in tel.explain()


def test_views_and_stats_are_checked_and_written_in_the_header(tmp_path):
    with pytest.raises(ValueError, match="kind"):
        _tel(tmp_path, views=[{"name": "a", "kind": "movie"}])
    with pytest.raises(ValueError, match="scale"):
        _tel(tmp_path, views=[{"name": "a", "kind": "film2d", "scale": "rainbow"}])
    with pytest.raises(ValueError, match="unique"):
        _tel(tmp_path, stats=[{"name": "E", "kind": "series"}, {"name": "E", "kind": "series"}])
    tel = _tel(tmp_path, film_max_bytes=10**6, views=[{"name": "T", "kind": "film2d"}],
               stats=[{"name": "E", "kind": "series"}], header_extra={"watch_explain": "x"})
    tel.close()
    head = json.loads((tmp_path / "r.jsonl").read_text().splitlines()[0])
    assert head["views"][0]["name"] == "T" and head["stats"][0]["kind"] == "series"
    assert head["film"]["max_bytes"] == 10**6 and head["watch_explain"] == "x"


def test_index_reads_offsets_pages_and_resets(tmp_path):
    path = tmp_path / "r.film.ndjson"
    recs = [film_record(encode_frame("r", "w", np.full((3, 3), float(k)), max_px=8, t=0.5 * k,
                                     iteration=k), i=k, level=0) for k in range(150)]
    text = "".join(json.dumps(r) + "\n" for r in recs).encode()
    path.write_bytes(text[:-7])                         # last line incomplete
    idx = FilmIndex("r", path=path)
    new = idx.poll_file()
    assert len(new) == 149 and new[-1]["i"] == 148
    path.write_bytes(text)
    assert [e["i"] for e in idx.poll_file()] == [149]
    frames, nxt = idx.frames("w", 10, None, 2)
    assert len(frames) == MAX_FRAMES_PER_REQUEST and frames[0]["i"] == 10 and frames[1]["i"] == 12
    assert nxt == 10 + 2 * MAX_FRAMES_PER_REQUEST
    assert decode_frame(film_frame(frames[0]))[0, 0] == 10.0
    assert idx.frames("w", 140, 145)[0][-1]["i"] == 144
    path.write_bytes(text[: len(text) // 3])            # replaced by a shorter file: re-index
    idx.poll_file()
    assert idx.summary()["names"]["w"]["count"] < 60


def test_memory_index_thins_uniformly_and_declares_it():
    recs = [json.dumps({"name": "w", "i": k, "t": float(k), "kind": "frame", "data": "x" * 50})
            + "\n" for k in range(100)]
    idx = FilmIndex("relay:r", max_frames=40)
    pos = 0
    for line in recs:
        idx.ingest(line.encode(), pos)
        pos += len(line)
    s = idx.summary()
    assert s["names"]["w"]["count"] <= 40 and s["thinned_here"] > 0
    ts = [e["t"] for e in idx.index("w")]
    assert ts[0] == 0.0 and ts[-1] == 99.0             # still covers the whole run
    other = FilmIndex("relay:q", max_frames=1000)
    other.ingest("".join(recs).encode(), 0)
    films = {"a": idx, "b": other}
    thin_to_budget(films, 4000)
    assert sum(f.bytes for f in films.values()) <= 4000


def test_spool_reader_indexes_films_and_all_frame_kinds(tmp_path):
    tel = _tel(tmp_path, film_max_bytes=10**6)
    tel.frame("w", np.ones((4, 4)), t=0.0, iteration=0, force=True)
    tel.curve("E(k)", [1, 2], [1.0, 0.5], t=0.0, iteration=0, force=True)
    tel.frame3d("q", np.ones((4, 4, 4)), extent=[[0, 1]] * 3, scale="smoke", force=True)
    tel.close()
    r = SpoolReader(tmp_path, max_scalars=10, max_logs=10)
    r.poll()
    notices = r.poll_film()
    assert {n["name"] for n in notices} == {"w", "E(k)", "q"} and notices[0]["type"] == "film"
    assert set(r.runs["r"].frames) == {"w", "E(k)", "q"}
    # RunState.ingest stored only 2D frames; a 3D frame arriving as a record (hub) was counted
    # malformed and missing from the snapshot
    st = r.runs["r"]
    st.ingest({**st.frames["q"], "name": "q2"})
    assert "q2" in st.frames and st.malformed == 0


# ---------------------------------------------------------------------- Studio review
def test_bad_names_are_refused_before_they_silence_the_spool(tmp_path):
    """A frame/curve name with '/' made the writer's file write fail: telemetry went silent for
    the rest of the run (error only in the status). Names are checked up front; a statistic
    named like a built-in number (t, dt, ...) and a view/stat name clash are refused too."""
    with pytest.raises(ValueError, match="cannot name a spool file"):
        _tel(tmp_path, stats=[{"name": "a/b", "kind": "profile"}])
    with pytest.raises(ValueError, match="both a view and a statistic"):
        _tel(tmp_path, views=[{"name": "E", "kind": "film2d"}], stats=[{"name": "E", "kind": "spectrum"}])
    tel = _tel(tmp_path)
    for bad in ("a/b", "", ".hidden", "x\x00y"):
        with pytest.raises(ValueError):
            tel.frame(bad, np.zeros((4, 4)))
    with pytest.raises(ValueError, match="built-in"):
        tel.ivp_callback(scalars={"t": lambda s: 1.0})
    with pytest.raises(ValueError, match="cannot name"):
        tel.ivp_callback(curves={"x/y": (lambda s: ([0], [0]), {})})
    tel.close()
    assert tel.error is None
    for bad in ("my run", "../x", ""):                  # the relay and the hub refuse these ids
        with pytest.raises(ValueError, match="run_id"):
            Telemetry(bad, spool=tmp_path, capacity=10, scalar_interval=0, frame_interval=0, max_px=8)


def test_reused_run_id_starts_a_new_film_within_budget(tmp_path):
    """Re-running a script with the same run_id appended a second film to the first: indices
    restarted at 0 inside one file and the budget no longer held. The old film is moved aside."""
    budget = 20_000
    rng = np.random.default_rng(2)
    for _ in range(2):
        tel = _tel(tmp_path, film_max_bytes=budget)
        for k in range(40):
            tel._film_append("w", encode_frame("r", "w", rng.standard_normal((32, 32)), max_px=64,
                                               t=k, iteration=k))
        tel.close()
    film = tmp_path / "r.film.ndjson"
    lines = [json.loads(x) for x in film.read_text().splitlines()]
    assert film.stat().st_size <= budget
    assert [r["i"] for r in lines] == list(range(len(lines)))
    assert (tmp_path / "r.film.previous.ndjson").exists()
    assert "was used before" in (tmp_path / "r.jsonl").read_text()


def test_bad_level_field_is_counted_not_jamming_the_index(tmp_path):
    """A film line with a non-integer 'level' raised after its entry was added: pos never moved,
    every poll re-added the same entries (memory grew without bound)."""
    good = json.dumps({"name": "w", "i": 0, "t": 0.0, "kind": "frame", "level": 0}) + "\n"
    bad = json.dumps({"name": "w", "i": 1, "t": 1.0, "kind": "frame", "level": "x"}) + "\n"
    path = tmp_path / "r.film.ndjson"
    path.write_text(good + bad + good.replace('"i": 0', '"i": 2'))
    idx = FilmIndex("r", path=path)
    for _ in range(3):
        idx.poll_file()
    assert len(idx.index("w")) == 2 and idx.malformed == 1 and idx.pos == path.stat().st_size


def test_hub_film_budget_holds_with_many_small_names():
    """thin_to_budget only thinned names of >= 3 frames: 50 names x 2 frames stayed 10x over."""
    films = {}
    for r in range(5):
        idx = FilmIndex(f"relay:r{r}")
        pos = 0
        for n in range(10):
            for k in range(2):
                line = json.dumps({"name": f"n{n}", "i": k, "t": float(k), "data": "x" * 900}) + "\n"
                idx.ingest(line.encode(), pos)
                pos += len(line)
        films[r] = idx
    thin_to_budget(films, 10_000)
    assert sum(f.bytes for f in films.values()) <= 10_000
    assert sum(f.thinned for f in films.values()) > 0                  # declared, per index


def test_film_answer_is_capped_in_bytes(tmp_path, monkeypatch):
    import conatus.telemetry.film as film_mod
    monkeypatch.setattr(film_mod, "MAX_BYTES_PER_REQUEST", 2500)
    idx = FilmIndex("relay:r")
    pos = 0
    for k in range(10):
        line = json.dumps({"name": "w", "i": k, "t": float(k), "data": "x" * 1000}) + "\n"
        idx.ingest(line.encode(), pos)
        pos += len(line)
    out, nxt = idx.frames("w", 0, None)
    assert len(out) == 2 and nxt == 2
