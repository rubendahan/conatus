"""F709 (audit of 0.2.1, N34): XDMF of accepted names is XML.

Oracle: xml.etree parses the sidecar and returns the task name."""
from _acceptance import check, passed, start

start(709, "XDMF of accepted names is XML")
import tempfile  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from pathlib import Path  # noqa: E402

import conatus as cn  # noqa: E402

for base, task in (("two--hyphens", "u"), ("amp", "u&v"), ("angle", "u<2")):
    m = cn.Model(cn.Box(x=cn.periodic(1.), z=cn.periodic(2.)))
    m.scalar("u")
    m.equations("dt(u)=0")
    s = m.simulate(N=(4, 6), scheme="RK222", dt=.1)
    s.initial(u=1.)
    p = Path(tempfile.mkdtemp()) / f"{base}.h5"
    s.output(p, tasks={task: "u"}, every=1, mode="w")
    s.run(steps=1)
    root = ET.parse(cn.io.write_xdmf(p)).getroot()
    check(root.find("./Domain/Grid/Grid/Attribute").attrib["Name"] == task, f"task {task}")
passed("three sidecars parse")
