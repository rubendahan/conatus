"""Where Conatus writes by default: development checkout or installed package.

Two install modes, told apart without importing the compiled core:

* **development checkout**: this package is the repository's ``python/conatus`` (the repository
  has ``core/CMakeLists.txt``); scratch data goes to ``$CN_TMP`` (``source env/env.sh``) or the
  checkout's ``scratch/``, as before;
* **installed package** (``pip install .``, a wheel, conda): the package sits in
  ``site-packages`` of an environment. Nothing is ever written next to it (the
  telemetry spool used to land in ``<env>/lib/python3.x/scratch/telemetry``). The defaults are
  the XDG user directories, declared here and printed by ``python -m conatus doctor``:

  ========================  =====================================================================
  user cache directory      ``$CN_CACHE``, else ``$XDG_CACHE_HOME/conatus``, else
                            ``~/.cache/conatus`` (the XDG Base Directory default)
  telemetry spool           ``$CN_TELEMETRY``, else ``$CN_TMP/telemetry``, else (checkout)
                            ``scratch/telemetry``, else ``<user cache directory>/telemetry``
  ========================  =====================================================================

The run registry (``cn.runs``) is data, not a cache: it keeps its own explicit rule
(``$CN_RUNS``, ``$CN_TMP/runs``, or ``root=``; ``conatus.runs.store.default_root``).

No third-party package (platformdirs) is needed: Conatus supports Linux only (pyproject
classifiers), where the XDG variables are the convention. A default that would resolve inside
the Python environment (``sys.prefix``) or the package directory is refused, never used.
Standard library only (a monitor hub image copies this file without the rest of the library).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["PACKAGE_DIR", "REPO_ROOT", "in_checkout", "install_mode", "user_cache_dir",
           "telemetry_spool", "describe"]

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]


def in_checkout() -> bool:
    """True when this package is the ``python/conatus`` of a source checkout (development mode)."""
    return PACKAGE_DIR.parent.name == "python" and (REPO_ROOT / "core" / "CMakeLists.txt").is_file()


def install_mode() -> str:
    """``"development checkout"`` or ``"installed package"`` (see the module docstring)."""
    return "development checkout" if in_checkout() else "installed package"


def _inside_environment(p: Path) -> bool:
    """True if ``p`` lies in the Python environment or in the package itself (never written)."""
    try:
        rp = p.expanduser().resolve()
    except OSError:
        return False
    roots = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(), PACKAGE_DIR}
    return any(rp == r or rp.is_relative_to(r) for r in roots)


def user_cache_dir() -> Path:
    """Declared user cache directory: ``$CN_CACHE``, else ``$XDG_CACHE_HOME/conatus``, else
    ``~/.cache/conatus``. Refused (``OptionError``) if it resolves inside the environment."""
    env = os.environ.get("CN_CACHE")
    if env:
        p, src = Path(env), "$CN_CACHE"
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        # the XDG specification: a relative XDG_CACHE_HOME is invalid and ignored (and then not
        # named as the source in a refusal)
        use = bool(xdg) and os.path.isabs(xdg)
        base = Path(xdg) if use else Path.home() / ".cache"
        p, src = base / "conatus", ("$XDG_CACHE_HOME/conatus" if use else "~/.cache/conatus")
    if _inside_environment(p):
        from .errors import OptionError       # lazy: the hub image ships this module alone
        raise OptionError(f"user cache directory {p} ({src}) lies inside the Python environment "
                          f"{sys.prefix}", why="nothing is written into an installed environment",
                          fix="export CN_CACHE=/a/writable/path (or XDG_CACHE_HOME)")
    return p


def telemetry_spool() -> Path:
    """Default telemetry spool (``Telemetry``, ``sim.watch``): ``$CN_TELEMETRY``,
    else ``$CN_TMP/telemetry``, else the checkout's ``scratch/telemetry``, else
    ``<user_cache_dir()>/telemetry``."""
    env = os.environ.get("CN_TELEMETRY")
    if env:
        return Path(env)
    tmp = os.environ.get("CN_TMP")
    if tmp:
        return Path(tmp) / "telemetry"
    if in_checkout():
        return REPO_ROOT / "scratch" / "telemetry"
    return user_cache_dir() / "telemetry"


def describe() -> dict[str, str]:
    """Install mode and the default directories, as strings (for ``doctor``)."""
    from .errors import OptionError
    out = {"mode": install_mode(), "package": str(PACKAGE_DIR)}
    for k, fn in (("cache", user_cache_dir), ("telemetry", telemetry_spool)):
        try:
            out[k] = str(fn())
        except OptionError as err:
            out[k] = f"refused: {str(err).splitlines()[0]}"
    return out
