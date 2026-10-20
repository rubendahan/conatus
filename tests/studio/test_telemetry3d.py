""" telemetry frames (frame3d, points3d, vec2d): lossy only where declared, bounded, never
harmful to the run (section "Trames 3D")."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from conatus.telemetry import (FRAME_KINDS, SpoolReader, Telemetry, decode_frame3d, decode_points,
                               decode_vec2d, encode_frame3d, encode_points, encode_vec2d)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ivp"))
EXT = [[0.0, 2 * np.pi], [-1.0, 1.0], [0.0, np.pi]]


def _tel(tmp_path, **kw):
    kw.setdefault("capacity", 1000)
    kw.setdefault("scalar_interval", 0.0)
    kw.setdefault("frame_interval", 0.0)
    kw.setdefault("max_px", 64)
    return Telemetry("run3d", spool=tmp_path, **kw)


def _tg(n0, n1, n2):
    """Taylor–Green-like analytic field on the EXT box."""
    x = np.linspace(*EXT[0], n0)[:, None, None]
    y = np.linspace(*EXT[1], n1)[None, :, None]
    z = np.linspace(*EXT[2], n2)[None, None, :]
    return np.sin(x) * np.cos(np.pi * y) * np.cos(2 * z)


def test_frame3d_roundtrip_bound_and_stride():
    f = _tg(100, 33, 20)
    fr = encode_frame3d("r", "q", f, max_px=40, extent=EXT, scale="signed", t=1.0, iteration=7)
    assert fr["kind"] == "frame3d" and fr["kind"] in FRAME_KINDS
    assert fr["stride"] == [3, 1, 1] and fr["shape"] == [34, 33, 20] and fr["full_shape"] == [100, 33, 20]
    g = decode_frame3d(fr)
    ref = f[::3]
    assert np.max(np.abs(g - ref)) <= (fr["max"] - fr["min"]) / 510 + 1e-12   # 8-bit rounding
    assert "8-bit" in fr["lossy"] and "stride 3x1x1" in fr["lossy"]
    assert len(json.dumps(fr)) < 4 * 34 * 33 * 20 / 3 + 6000      # base64 of uint8 + node positions
    assert len(fr["coords"][0]) == 34 and fr["coords"][0][1] == pytest.approx(3 * 2 * np.pi / 100)


def test_frame3d_coords_and_refusals():
    y = -np.cos(np.pi * (np.arange(33) + 0.5) / 33)
    fr = encode_frame3d("r", "u", _tg(8, 33, 8), max_px=16, extent=EXT, scale="smoke",
                        coords=[np.arange(8.0), y, np.arange(8.0)])
    assert fr["coords"][1] == y[::3].tolist()
    bad = _tg(4, 4, 4)
    bad[1, 1, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        encode_frame3d("r", "u", bad, max_px=8, extent=EXT, scale="smoke")
    with pytest.raises(ValueError, match="scale"):
        encode_frame3d("r", "u", _tg(4, 4, 4), max_px=8, extent=EXT, scale="jet")
    with pytest.raises(ValueError, match="3D"):
        encode_frame3d("r", "u", np.zeros((4, 4)), max_px=8, extent=EXT, scale="smoke")


def test_points_roundtrip_missing_clipping_and_budget():
    rng = np.random.default_rng(0)
    lines = rng.uniform(0.1, 0.9, (50, 40, 3))
    lines[3, :10] = np.nan                                # not yet emitted
    lines[4, -1] = [1.5, 0.5, 0.5]                        # outside the extent: clipped, counted
    ext = [[0, 1]] * 3
    fr = encode_points("r", "streak", lines, max_points=10_000, extent=ext, mode="streaklines")
    assert fr["stride"] == [1, 1] and fr["clipped"] == 1 and fr["missing"] == 10
    p = decode_points(fr)
    ok = np.isfinite(lines).all(axis=2) & (lines <= 1).all(axis=2)
    assert np.all(np.isnan(p[3, :10])) and np.max(np.abs(p[ok] - lines[ok])) <= 0.5 / 65534 + 1e-12
    small = encode_points("r", "s", lines, max_points=500, extent=ext, mode="pathlines")
    n, m = small["shape"]
    assert n * m <= 500 + n and small["full_shape"] == [50, 40]           # last point kept per line
    parts = encode_points("r", "p", rng.uniform(0, 1, (5000, 2)), max_points=1000,
                          extent=[[0, 1], [0, 1]], mode="particles", values=np.arange(5000.0))
    assert parts["dim"] == 2 and parts["shape"][0] <= 1000 and "values" in parts
    with pytest.raises(ValueError, match="mode"):
        encode_points("r", "p", lines, max_points=10, extent=ext, mode="smoke")


def test_vec2d_roundtrip():
    x = np.linspace(0, 2 * np.pi, 80)[:, None]
    y = np.linspace(0, 2 * np.pi, 60)[None, :]
    u, v = np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)
    fr = encode_vec2d("r", "uv", u, v, max_px=64, extent=[[0, 6.3], [0, 6.3]])
    uu, vv = decode_vec2d(fr)
    assert fr["stride"] == [2, 1] and uu.shape == (40, 60)
    assert max(np.abs(uu - u[::2]).max(), np.abs(vv - v[::2]).max()) <= fr["amp"] / 255 + 1e-12


def test_sink_throttle_want_frame_and_errors_never_raise(tmp_path):
    tel = _tel(tmp_path, frame_interval=100.0, max_px3d=16, max_points=64)
    assert tel.want_frame("q") and tel.frame3d("q", _tg(32, 32, 32), extent=EXT, scale="signed")
    assert not tel.want_frame("q")                                # throttled now
    assert not tel.frame3d("q", _tg(32, 32, 32), extent=EXT, scale="signed")
    bad = _tg(8, 8, 8)
    bad[0, 0, 0] = np.inf
    assert not tel.frame3d("bad", bad, extent=EXT, scale="smoke")       # logged, not raised
    assert not tel.points3d("p", np.zeros((4, 4)), extent=EXT, mode="particles")  # d = 4
    assert tel.points3d("p2", np.random.default_rng(1).uniform(0, 1, (400, 3)),
                        extent=[[0, 1]] * 3, mode="particles")
    tel.close()
    rd = SpoolReader(tmp_path, max_scalars=10, max_logs=50)
    rd.poll()
    st = rd.runs["run3d"]
    assert st.frames["q"]["shape"] == [16, 16, 16] and st.frames["p2"]["shape"][0] <= 64
    assert st.header["settings"]["max_px3d"] == 16 and st.header["settings"]["max_points"] == 64
    msgs = " ".join(r["message"] for r in st.logs)
    assert "'bad' not sent" in msgs and "'p' not sent" in msgs
    assert "3D frames strided to <= 16" in tel.explain()


def test_ivp_run_is_bit_identical_with_3d_frames(tmp_path):
    """S0 rule kept for the frames: telemetry with 3D, point and vec2d frames (plus a failing
    3D callable) leaves the state bit-identical."""
    from ivp_cases import burgers_1d
    from conatus.timestep import IVPSolver
    plan, u = burgers_1d(N=32)[:2]
    runs = []
    for with_tel in (False, True):
        s = IVPSolver(plan, "RK443", 1e-3)
        s.set_state({u: lambda x: 0.5 * np.sin(x)})
        if with_tel:
            tel = _tel(tmp_path)
            ext = [[0, 2 * np.pi], [0, 1], [0, 1]]

            def f3(s):
                g = s.grid("u")
                return g[:, None, None] * np.ones((1, 5, 6))
            cb = tel.ivp_callback(frames3d={"u3": (f3, dict(extent=ext, scale="signed")),
                                            "boom": (lambda s: 1 / 0, dict(extent=ext, scale="smoke"))})

            def cb2(s):
                cb(s)
                g = s.grid("u")
                tel.points3d("pts", np.stack([np.linspace(0, 6, g.size), g, 0 * g], 1),
                             extent=[[0, 2 * np.pi], [-1, 1], [-1, 1]], mode="particles")
                tel.vec2d("uv", np.tile(g, (4, 1)), np.tile(g, (4, 1)), extent=[[0, 1], [0, 1]])
            s.run(40, callback=cb2, every=5)
            tel.close()
        else:
            s.run(40)
        runs.append(np.array(s.X))
    assert np.array_equal(runs[0], runs[1])
    rd = SpoolReader(tmp_path, max_scalars=100, max_logs=100)
    rd.poll()
    st = rd.runs["run3d"]
    assert {st.frames[k]["kind"] for k in ("u3", "pts", "uv")} == {"frame3d", "points3d", "vec2d"}
    assert any("'boom' failed" in r["message"] for r in st.logs)


def test_page_inlines_the_3d_view(tmp_path):
    """The served page carries view3d.js inline (one document for server and hub); a missing
    script is reported in the page, not dropped silently."""
    from conatus.studio.page import TAG, render_page
    from conatus.studio.webapp import classic_page
    web = classic_page()                      # the checkout's studio/web, or the copy in the package
    html = render_page(web).decode()
    assert TAG not in html and "window.View3D" in html and 'id="view3dPanel"' in html
    (tmp_path / "index.html").write_text(f"<html>{TAG}</html>")
    assert "view3d.js unavailable" in render_page(tmp_path / "index.html").decode()
