"""Self-describing HDF5 outputs: plain h5py, xarray, XDMF, old files.

Analysis files and checkpoints name every dimension (``dims`` attribute + HDF5 dimension scales
linked to the coordinate datasets), name vector components, and carry the Conatus version, the
parameters and the coordinates; ``to_xarray()`` builds an ``xarray.Dataset``; a Cartesian 2D/3D
analysis file gets an XDMF sidecar for ParaView; files written before keep reading."""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import conatus as cn

h5py = pytest.importorskip("h5py")


def _rb(N=(16, 12)):
    box = cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5))
    m = cn.Model(box, name="RB", Ra=2500, Pr=1)
    m.velocity("u"), m.pressure("p"), m.scalar("T")
    m.equations("dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)", "div(u) = 0",
                "dt(T) - lap(T) = -u·grad(T)")
    m.at("bottom", u=0, T=+0.5)
    m.at("top", u=0, T=-0.5)
    sim = m.simulate(N=N, scheme="RK443", dt=1e-3)
    sim.initial(T=lambda x, z: -z + 1e-3 * np.cos(np.pi * x) * np.cos(np.pi * z))
    return sim


def _run(tmp_path):
    sim = _rb()
    out = sim.output(tmp_path / "a.h5", tasks={"T": "T", "u": "u", "ux": "u.x", "dzT": "dz(T)",
                                               "KE": lambda s: 1.0}, every=2)
    sim.run(steps=6)
    return sim, out


def test_plain_h5py_sees_dimensions_scales_components_and_metadata(tmp_path):
    sim, out = _run(tmp_path)
    with h5py.File(tmp_path / "a.h5", "r") as f:
        assert f.attrs["conatus_version"] == cn.__version__
        assert json.loads(f.attrs["parameters"]) == {"Pr": 1.0, "Ra": 2500.0}
        assert json.loads(f.attrs["coordinates"]) == ["x", "z"] and bool(f.attrs["cartesian"])
        T, u = f["tasks/T"], f["tasks/u"]
        assert json.loads(T.attrs["dims"]) == ["t", "x", "z"]
        assert [d.label for d in u.dims] == ["t", "component", "x", "z"]
        assert json.loads(u.attrs["component_names"]) == ["x", "z"]
        # each labelled dimension is attached to its coordinate dataset (HDF5 dimension scales)
        assert u.dims[0][0].name == "/scales/sim_time" and u.dims[2][0].name == "/scales/x"
        assert np.array_equal(T.dims[2][0][()], sim.coordinate("z"))
        assert [c.decode() for c in u.dims[1][0][()]] == ["x", "z"]
        assert json.loads(f["tasks/KE"].attrs["dims"]) == ["t"]
        assert json.loads(f["tasks/dzT"].attrs["dims"]) == ["t", "x", "z"]
    assert "XDMF sidecar (ParaView)" in out.explain()


def test_to_xarray(tmp_path):
    xr = pytest.importorskip("xarray")
    sim, _out = _run(tmp_path)
    ds = cn.io.read(tmp_path / "a.h5").to_xarray()
    assert isinstance(ds, xr.Dataset)
    assert ds["u"].dims == ("t", "component", "x", "z") and list(ds["component"].values) == ["x", "z"]
    # row of the initial state first (output(initial=True) by default)
    assert np.allclose(ds["t"].values, [0, 0.002, 0.004, 0.006]) and list(ds["iteration"].values) == [0, 2, 4, 6]
    assert np.array_equal(ds["z"].values, sim.coordinate("z"))
    assert np.array_equal(ds["T"].isel(t=-1).values, sim.grid("T"))
    assert np.array_equal(ds["u"].sel(component="x").values, ds["ux"].values)
    assert json.loads(ds.attrs["parameters"])["Ra"] == 2500.0 and ds.attrs["conatus_version"]


def test_checkpoint_is_self_describing_and_restarts(tmp_path):
    xr = pytest.importorskip("xarray")
    sim = _rb()
    sim.run(steps=3)
    sim.save(tmp_path / "c.h5", grids=True)
    with h5py.File(tmp_path / "c.h5", "r") as f:
        assert json.loads(f["grids/u"].attrs["dims"]) == ["component", "x", "z"]
        assert json.loads(f["fields/T"].attrs["dims"]) == ["x_mode", "z_mode"]
        assert f["grids/T"].dims[1][0].name == "/scales/z"
    ds = cn.io.read(tmp_path / "c.h5").to_xarray()
    assert isinstance(ds, xr.Dataset) and ds.attrs["iteration"] == 3
    assert np.array_equal(ds["T"].values, sim.grid("T")) and ds["u_coeffs"].dims[0] == "component"
    again = _rb()
    again.restart(tmp_path / "c.h5")                    # the added metadata change no restart
    sim.run(steps=2)
    again.run(steps=2)
    assert np.array_equal(again.solver.X, sim.solver.X)


def test_xdmf_sidecar_points_into_the_file(tmp_path):
    _run(tmp_path)
    root = ET.parse(tmp_path / "a.xmf").getroot()
    grids = root.findall("./Domain/Grid/Grid")
    assert len(grids) == 4 and grids[-1].find("Time").get("Value") == "0.006"   # t = 0 included
    topo = grids[0].find("Topology")
    assert topo.get("TopologyType") == "2DRectMesh" and topo.get("Dimensions") == "16 12"
    names = [a.get("Name") for a in grids[0].findall("Attribute")]
    assert names == ["T", "dzT", "u.x", "u.z", "ux"]             # KE (a scalar) not on the grid
    ux = grids[1].findall("Attribute")[2]
    sel, data = ux.find("DataItem").findall("DataItem")
    assert sel.text.split()[:4] == ["1", "0", "0", "0"]          # row 1, component x
    assert data.text == "a.h5:/tasks/u" and data.get("Dimensions") == "4 2 16 12"
    geo = [d.text for d in grids[0].find("Geometry").findall("DataItem")]
    assert geo == ["a.h5:/scales/z", "a.h5:/scales/x"]           # XDMF X = last coordinate
    assert "ParaView X = z" in (tmp_path / "a.xmf").read_text()


def test_curvilinear_has_no_sidecar_and_says_why(tmp_path):
    m = cn.Model(cn.Disk(1.0))
    m.scalar("T")
    m.equations("heat: dt(T) - lap(T) = 0")
    m.at("wall", T=0)
    sim = m.simulate(N=(8, 8), scheme="RK443", dt=1e-3)
    out = sim.output(tmp_path / "d.h5", tasks=["T"], every=1)
    sim.run(steps=2)
    assert not (tmp_path / "d.xmf").exists()
    assert "no XDMF sidecar" in out.explain() and "not a Cartesian box" in out.explain()
    with pytest.raises(cn.io.xdmf.XDMFNotCovered):
        cn.io.write_xdmf(tmp_path / "d.h5")
    with h5py.File(tmp_path / "d.h5", "r") as f:              # still self-describing in HDF5
        assert json.loads(f["tasks/T"].attrs["dims"]) == ["t", "phi", "r"]


def test_files_written_before_still_read(tmp_path):
    """A file of the earlier layout (no dims, no scales attached, no metadata) reads, and
    to_xarray() names only what the coordinate lengths identify."""
    pytest.importorskip("xarray")
    _run(tmp_path)
    with h5py.File(tmp_path / "a.h5", "a") as f:
        for k in ("conatus_version", "parameters", "coordinates", "cartesian", "geometry", "model"):
            del f.attrs[k]
        for d in f["tasks"].values():
            for i in range(d.ndim):
                for sc in list(d.dims[i].values()):
                    d.dims[i].detach_scale(sc)
            for k in ("dims", "component_names"):
                if k in d.attrs:
                    del d.attrs[k]
        del f["scales/component"]
    data = cn.io.read(tmp_path / "a.h5")
    assert data.dims == {} and data.conatus_version == "" and "not stored" in data.explain()
    ds = data.to_xarray()
    assert ds["T"].dims == ("t", "x", "z")                      # lengths 16 and 12: unique
    assert ds["u"].dims == ("t", "u_dim0", "x", "z")
    with pytest.raises(cn.io.xdmf.XDMFNotCovered, match="no coordinates"):
        cn.io.write_xdmf(tmp_path / "a.h5")


def test_to_xarray_without_xarray_says_how_to_install(tmp_path, monkeypatch):
    _run(tmp_path)
    data = cn.io.read(tmp_path / "a.h5")
    monkeypatch.setitem(sys.modules, "xarray", None)
    from conatus.errors import MissingDependencyError
    with pytest.raises(MissingDependencyError, match="pip install xarray"):
        data.to_xarray()
