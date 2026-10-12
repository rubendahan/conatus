"""Is there a usable GPU here? ``gpu_available()`` and ``gpu_report()``.

``backend='cuda'`` needs two things, checked separately and reported by name (nothing is
guessed, nothing falls back to the CPU silently):

* **CuPy** importable, with at least one visible CUDA device (the state lives in CuPy arrays);
* the **compiled core** built with CUDA (``CONATUS_CUDA=ON``) and seeing a device
  (``conatus.runtime.cuda``: batched banded solves, transforms, IMEX kernels).
"""
from __future__ import annotations

__all__ = ["gpu_available", "gpu_report"]


def _cupy_status() -> tuple[bool, str]:
    """(usable, reason) for CuPy with a visible device."""
    try:
        import cupy
    except ImportError as err:
        from .errors import GPU_INSTALL_FIX
        return False, f"CuPy not installed ({err}); {GPU_INSTALL_FIX}"
    try:
        n = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as err:  # noqa: BLE001 -- no driver/device: CuPy raises its own runtime errors
        return False, f"CuPy sees no CUDA device ({type(err).__name__}: {err})"
    return (n > 0), f"CuPy {cupy.__version__}, {n} device(s)"


def _core_status() -> tuple[bool, str]:
    """(usable, reason) for the CUDA part of the compiled core."""
    try:
        from .runtime import cuda
    except ImportError as err:
        return False, f"compiled core not importable ({err})"
    if cuda.available():
        return True, f"compiled core with CUDA, device {cuda.device_name()}"
    return False, ("compiled core has no CUDA part (build with CONATUS_CUDA=ON)"
                   if cuda._cu is None else "compiled core has CUDA but sees no device")


def gpu_available() -> bool:
    """True when ``backend='cuda'`` can run here: CuPy sees a device **and** the compiled core
    has CUDA with a device. ``gpu_report()`` says which part is missing."""
    return _cupy_status()[0] and _core_status()[0]


def gpu_report() -> str:
    """One line per requirement of ``backend='cuda'``, each marked ok/missing with its reason."""
    lines = [f"GPU (backend='cuda'): {'available' if gpu_available() else 'NOT available'}"]
    for name, (ok, why) in (("CuPy", _cupy_status()), ("compiled core", _core_status())):
        lines.append(f"  {name:14s} {'ok     ' if ok else 'missing'}  {why}")
    return "\n".join(lines)
