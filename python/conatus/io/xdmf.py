"""XDMF sidecar of an analysis file: ParaView (and VisIt) open Conatus outputs directly.

``write_xdmf("snapshots.h5")`` writes ``snapshots.xmf`` next to the HDF5 file: a temporal
collection with one grid per written row (``Time`` = ``sim_time``), whose arrays point into the
HDF5 file (no data copied). ``sim.output(...)`` rewrites it at the end of every ``sim.run`` call;
for a file of a job that was killed, call ``cn.io.write_xdmf(path)`` (also works on files of the
self-describing layout written by any later run).

What it covers (declared):

* **Cartesian grids in 2D and 3D** (``cn.Box``, ``cn.Channel``): ``RectMesh`` topology with the
  coordinate arrays of the file (``/scales/<coord>``; Chebyshev points are not uniform, so the
  mesh is rectilinear, not regular).
* **Real scalar tasks** on the full grid (an unknown, a component ``"u.x"``, a scalar expression)
  and **real vector tasks**, written as one scalar array per component (``u.x``, ``u.z``):
  XDMF vectors need an interleaved last axis, the file stores components first.
* **Axis order**: HDF5 arrays are row-major in coordinate order (x, [y,] z), and XDMF names the
  fastest axis X: ParaView's X axis is therefore the **last** coordinate (z), its Y (2D) or Z (3D)
  axis the first one (x). The sidecar says so in a comment; rotate in ParaView if needed.

Not covered (the function raises :class:`XDMFNotCovered` with the reason, and ``explain()`` of
the output says it): 1D boxes, **curvilinear geometries** (disk, annulus, pipe, sphere, ball,
shell: their grids are polar/spherical, an XDMF structured mesh of their Cartesian points is not
written yet), mapped boxes (curved walls), complex tasks (XDMF has no complex type), function
tasks whose shape is not the grid, files without coordinates (expert solvers).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from ..errors import UnsupportedError
from ._h5 import DOC, open_h5
from ._describe import COMPONENT, read_dims

__all__ = ["write_xdmf", "XDMFNotCovered"]


class XDMFNotCovered(UnsupportedError):
    """This analysis file cannot be described by the XDMF sidecar (the message says why)."""


def _attr(f, k, default=None):
    v = f.attrs.get(k, default)
    return v.decode() if isinstance(v, bytes) else v


def write_xdmf(path: str | os.PathLike) -> Path:
    """Write ``<path>.xmf`` (same name, suffix ``.xmf``) describing the analysis file ``path``
    for ParaView; returns its path. Atomic (temporary file renamed over the sidecar)."""
    from .analysis import FORMAT
    path = Path(path)
    with open_h5(path, "r") as f:
        if _attr(f, "format") != FORMAT:
            raise XDMFNotCovered(f"write_xdmf({str(path)!r}): not a Conatus analysis file",
                                 fix="a file written by sim.output(...)", doc=DOC)
        coords = json.loads(_attr(f, "coordinates") or "[]")
        if not coords:
            raise XDMFNotCovered("the file has no coordinates (written before the "
                                 "self-describing layout, or by an expert solver)",
                                 fix="sim.output(...) of a native simulation", doc=DOC)
        if not bool(f.attrs.get("cartesian", False)):
            raise XDMFNotCovered(f"geometry {_attr(f, 'geometry', '?')} is not a Cartesian "
                                 "box: XDMF RectMesh describes rectilinear grids only "
                                 "(curvilinear sidecars are not written yet)", doc=DOC)
        if len(coords) not in (2, 3):
            raise XDMFNotCovered(f"{len(coords)}D grid: the sidecar covers 2D and 3D grids",
                                 doc=DOC)
        n = int(f.attrs["writes"])
        times = f["scales/sim_time"][:n]
        sizes = [int(f["scales"][c].shape[0]) for c in coords]
        arrays = []                                       # (name, dataset path, component index)
        skipped = []
        for k, d in f["tasks"].items():
            dims = read_dims(d)
            if d.dtype.kind not in "fiu":
                skipped.append(f"{k} (complex)")
            elif dims == ["t", *coords]:
                arrays.append((k, f"/tasks/{k}", None))
            elif dims == ["t", COMPONENT, *coords]:
                arrays += [(f"{k}.{c}", f"/tasks/{k}", i) for i, c in enumerate(coords)]
            else:
                skipped.append(f"{k} (dims {dims})")
        if not arrays:
            raise XDMFNotCovered("no real task on the full grid" + (f" (skipped: "
                                 f"{', '.join(skipped)})" if skipped else ""), doc=DOC)
        dtypes = {k: f["tasks"][k].dtype for k in f["tasks"]}
        shapes = {k: tuple(f["tasks"][k].shape) for k in f["tasks"]}   # rows beyond `writes`
        # (a job killed inside a write) exist in the dataset: its real shape is declared
    h5 = path.name                                        # relative: the pair moves together
    dimstr = " ".join(str(s) for s in sizes)
    topo = "2DRectMesh" if len(coords) == 2 else "3DRectMesh"
    geom = "VXVY" if len(coords) == 2 else "VXVYVZ"
    order = list(reversed(coords))                        # XDMF X = fastest = last coordinate
    lines = ['<?xml version="1.0" ?>', '<Xdmf Version="3.0">', "<Domain>",
             f"<!-- written by conatus.io.xdmf for {escape(h5)}: ParaView X = {order[0]}, "
             f"Y = {order[1]}" + (f", Z = {order[2]}" if len(order) > 2 else "")
             + " (row-major arrays in coordinate order " + ", ".join(coords) + ") -->",
             '<Grid Name="conatus" GridType="Collection" CollectionType="Temporal">']
    if skipped:
        lines.insert(3, "<!-- not described: " + escape(", ".join(skipped)) + " -->")
    for w in range(n):
        lines.append(f'  <Grid Name="row{w}" GridType="Uniform">')
        lines.append(f'    <Time Value="{float(times[w])!r}"/>')
        lines.append(f'    <Topology TopologyType="{topo}" Dimensions="{dimstr}"/>')
        lines.append(f'    <Geometry GeometryType="{geom}">')
        for c in order:
            sz = sizes[coords.index(c)]
            lines.append(f'      <DataItem Dimensions="{sz}" NumberType="Float" Precision="8" '
                         f'Format="HDF">{escape(h5)}:/scales/{c}</DataItem>')
        lines.append("    </Geometry>")
        for name, ds, comp in arrays:
            key = ds.rsplit("/", 1)[1]
            dt = dtypes[key]
            num = "Float" if dt.kind == "f" else "Int"
            full = list(shapes[key])
            start = [w, *([comp] if comp is not None else []), *[0] * len(sizes)]
            count = [1, *([1] if comp is not None else []), *sizes]
            rank = len(full)
            lines += [f"    <Attribute Name={quoteattr(name)} AttributeType=\"Scalar\" Center=\"Node\">",
                      f'      <DataItem ItemType="HyperSlab" Dimensions="{dimstr}" Type="HyperSlab">',
                      f'        <DataItem Dimensions="3 {rank}" Format="XML">'
                      + " ".join(map(str, start)) + "  " + " ".join(["1"] * rank) + "  "
                      + " ".join(map(str, count)) + "</DataItem>",
                      f'        <DataItem Dimensions="{" ".join(map(str, full))}" NumberType="{num}" '
                      f'Precision="{dt.itemsize}" Format="HDF">{escape(h5)}:{ds}</DataItem>',
                      "      </DataItem>", "    </Attribute>"]
        lines.append("  </Grid>")
    lines += ["</Grid>", "</Domain>", "</Xdmf>", ""]
    out = path.with_suffix(".xmf")
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    tmp.write_text("\n".join(lines))
    os.replace(tmp, out)
    return out
