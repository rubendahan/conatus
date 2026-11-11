"""F945 (review H11): save_field refuses what data.h5 cannot store, in the caller's thread.

Oracle: the array passed in. Datetime, timedelta, unicode, object and masked arrays raise a
typed error at the call (they used to be accepted, then lost in the writer thread, or stored
without the mask). Control: bytes, big-endian, structured, zero-size, bool and complex arrays
round-trip with the same dtype and bytes.
"""
import numpy as np
from _acceptance import check, passed, refused, start

start(945, "save_field refuses unstorable arrays at the call")
import conatus as cn  # noqa: E402

bad = {"datetime": np.array(["2020-01-01"], dtype="M8[D]"), "timedelta": np.array([1], dtype="m8[s]"),
       "unicode": np.array(["a"]), "object": np.array([1, "a"], dtype=object),
       "masked": np.ma.array([1.0, 2.0], mask=[False, True])}
good = {"bytes": np.array([b"a"]), "big": np.array([1.0], dtype=">f8"),
        "struct": np.array([(1, 2.0)], dtype=[("a", "i4"), ("b", "f8")]),
        "zero": np.zeros(0), "bool": np.array([True, False]), "complex": np.array([1j])}
with cn.runs.track("fields", root="registry") as run:
    for name, a in bad.items():
        ok, _ = refused(run.save_field, name, a, compression=None)
        check(ok, f"{name}: accepted by save_field")
    for name, a in good.items():
        run.save_field(name, a, compression=None)
check(run.state == "completed", f"run state {run.state}")
for name, a in good.items():
    b = run.load_field(name)
    check(b.dtype == a.dtype and b.tobytes() == a.tobytes(), f"{name}: {a!r} came back as {b!r}")
passed("five unstorable kinds refused at the call; six kinds round-trip exactly")
