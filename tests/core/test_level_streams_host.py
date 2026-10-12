"""The level-scheduled streams of banded.cu, forward and transposed, checked on the host.

``core/bench/check_level_streams.cu`` includes banded.cu, writes the streams with the builder the
device uses (``LvlBuilder::build`` / ``build_T``, merged and unmerged levels) from LAPACK band LU
factors, executes them by a host emulation of the warp sweeps and compares with B y = b and
B^T y = b (backward error <= 1e-14, real and complex). No GPU needed; nvcc compiles it (slow).
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.slow


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="needs nvcc (host code only)")
def test_level_streams_forward_and_transposed(tmp_path):
    env = os.environ.get("CN_ENV") or os.environ.get("CONDA_PREFIX") or ""
    exe = tmp_path / "check_lvl"
    cmd = ["nvcc", "-std=c++17", "-O1", "-Wno-deprecated-gpu-targets", "-I", str(ROOT / "core/include"),
           str(ROOT / "core/bench/check_level_streams.cu"), str(ROOT / "core/src/cuda/memory.cu"),
           "-o", str(exe), f"-L{env}/lib", "-llapack", "-lblas", "-lcublas", "-Xlinker", f"-rpath={env}/lib"]
    subprocess.run(cmd, check=True, capture_output=True, timeout=900)
    out = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300)
    print(out.stdout[-600:])
    assert out.returncode == 0, out.stdout[-2000:]
    assert "0 failure(s)" in out.stdout
