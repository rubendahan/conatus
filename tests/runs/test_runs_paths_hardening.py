"""Run registry hardening (security the "fragile" list of the audit): path
components from callers or records stay inside the registry, forged inline arrays are refused,
the writer's directory is writable by its owner only, temporary names are unpredictable."""
from __future__ import annotations

import stat

import numpy as np
import pytest

import conatus.runs as R
from conatus.errors import OptionError
from conatus.runs._util import from_jsonable, jsonable


@pytest.mark.parametrize("rid", ["../x", "..", ".", "a/b", "/etc", "x\0y", ""])
def test_run_dir_refuses_path_components(tmp_path, rid):
    reg = R.registry(tmp_path / "reg")
    with pytest.raises(OptionError):
        reg.run_dir(rid)


@pytest.mark.parametrize("name", ["../run.json", "..", "a/b", "/etc/passwd", ""])
def test_file_path_refuses_path_components(tmp_path, name):
    with R.track("paths", root=tmp_path / "reg") as run:
        pass
    with pytest.raises(OptionError):
        run.file_path(name)


def test_forged_inline_arrays_are_refused():
    good = jsonable(np.arange(6.0).reshape(2, 3))
    assert np.array_equal(from_jsonable(good), np.arange(6.0).reshape(2, 3))
    for forged in ({"ndarray": [1.0], "shape": [10**12, 10**12], "dtype": "float64"},
                   {"ndarray": [1.0] * 65, "shape": [65], "dtype": "float64"},
                   {"ndarray": [1.0, 2.0], "shape": [-1], "dtype": "float64"},
                   {"ndarray": ["x"], "shape": [1], "dtype": "object"}):
        with pytest.raises(ValueError):
            from_jsonable(forged)


def test_run_directory_is_not_writable_by_others(tmp_path):
    import os
    old = os.umask(0o002)                       # a group-writable umask, as on shared clusters
    try:
        with R.track("modes", root=tmp_path / "reg") as run:
            pass
    finally:
        os.umask(old)
    mode = stat.S_IMODE(run.path.stat().st_mode)
    assert not mode & 0o022, oct(mode)
