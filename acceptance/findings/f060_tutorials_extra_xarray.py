"""F060: the [tutorials] extra installs everything the tutorials import (tutorial 16: xarray).

Read from the installed distribution's metadata (Requires-Dist with extra == "tutorials").
0.1.0's extra lacked xarray, so tutorial 16 failed after `pip install conatus[tutorials]`.
SKIP when conatus is not installed as a distribution (a source checkout on the path).
"""
import re
from importlib import metadata

from _acceptance import check, passed, skipped, start

start(60, "conatus[tutorials] includes xarray")
try:
    requires = metadata.requires("conatus") or []
except metadata.PackageNotFoundError:
    skipped("conatus is not an installed distribution (no metadata)")
extra = [r for r in requires if re.search(r"extra\s*==\s*['\"]tutorials['\"]", r)]
names = {re.split(r"[\s<>=!~;\[(]", r, maxsplit=1)[0].lower() for r in extra}
check("xarray" in names, f"[tutorials] requires {sorted(names)}: no xarray")
passed(f"[tutorials]: {sorted(names)}")
