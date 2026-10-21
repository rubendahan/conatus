"""F543 (external review of 0.1.1, F28): `hub serve --local` keeps its token where it says.

An explicit ``--state hub_state`` was replaced by a temporary directory (the code compared the
value with the default string), and the guide named that fixed path. Expected: ``--state`` is
used as given; without it, ``--local`` uses a new 0700 temporary directory; the printed line
names the token file. SKIP where the Studio hub is not shipped.
"""
import argparse
import os
import stat
from pathlib import Path

from _acceptance import check, need, passed, start

start(543, "hub --state is used as given; the token path is printed")
need("conatus.studio.hub")
need("tornado")
from conatus.studio import hub  # noqa: E402

os.environ.pop("CN_HUB_STATE", None)
check(hub.state_dir("hub_state", local=True) == "hub_state", "explicit --state hub_state replaced")
tmp = Path(hub.state_dir(None, local=True))
check(tmp.name.startswith("conatus_hub_") and stat.S_IMODE(tmp.stat().st_mode) == 0o700,
      f"default local state {tmp} mode {oct(stat.S_IMODE(tmp.stat().st_mode))}")
tmp.rmdir()
a = argparse.Namespace(state="hub_state", user_token_out=None, print_token=False)
line = hub._local_user_token(a, 8765, "t" * 43)
f = Path("hub_state") / "local_user_token"
check(str(f) in line and "t" * 43 not in line, f"printed line {line!r}")
check(stat.S_IMODE(f.stat().st_mode) == 0o600, "token file not 0600")
passed(f"--state kept; default {tmp.name}; printed {f}")
