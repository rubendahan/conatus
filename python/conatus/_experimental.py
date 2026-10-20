"""The experimental features of the release (shipped, tested, not promised).

 (§2, scope) lists what 0.2 ships without promising it: the Studio and its
hub, more than one GPU, shells beyond the tested grid sizes, channel and pipe orbits, and resolvent
analysis. Their first use in a process emits one :class:`~conatus.errors.ExperimentalWarning`
naming the feature; the same list is printed in the guide ("Experimental features"), README,
AGENTS.md and ``llms.txt`` (the development test suite keeps them in step).
"""
from __future__ import annotations

import warnings

from .errors import ExperimentalWarning

__all__ = ["EXPERIMENTAL", "experimental", "SHELL_TESTED_SIZE"]

#: feature key -> what is experimental (the sentence of the warning and of the docs)
EXPERIMENTAL: dict[str, str] = {
    "studio": "the Studio (the web page, its server and its client, python -m conatus studio)",
    "hub": "the Studio hub and relay (several runs, remote control, accounts)",
    "distributed": "runs on more than one GPU (DistributedIVPSolver with backend='cuda')",
    "shell_resolution": "spherical-shell simulations larger than the largest shell grid the test "
                        "suite runs",
    "orbits": "periodic orbits and travelling waves of 3-D channel and pipe flow (ShootingRPO, "
              "PipeTravellingWave)",
    "resolvent": "resolvent analysis (lin.resolvent, resolvent_modes)",
}

#: The largest spherical-shell grid (N_phi, N_theta, N_r) that the test suite runs: the GPU example
#: of docs/site/guide/shells.md (the CPU benchmark of Christensen et al. case 0 runs at 64 x 32 x 24,
#: inside it). Larger shells run, but no test covers them and their GPU acceptance runs are not done yet.
SHELL_TESTED_SIZE = (64, 48, 48)

_warned: set[str] = set()


def experimental(feature: str, detail: str = "") -> None:
    """Emit the ExperimentalWarning of ``feature`` the first time it is used in this process."""
    what = EXPERIMENTAL[feature]
    if feature in _warned:
        return
    _warned.add(feature)
    warnings.warn(f"experimental: {what}{f' ({detail})' if detail else ''}. It is tested but not "
                  "promised: its API and defaults may change in a minor release without a "
                  "deprecation cycle (guide: Experimental features)", ExperimentalWarning,
                  stacklevel=3)


def check_shell_size(sizes: tuple[int, ...]) -> None:
    """ExperimentalWarning for a shell grid beyond :data:`SHELL_TESTED_SIZE` in any direction."""
    if any(int(n) > m for n, m in zip(sizes, SHELL_TESTED_SIZE, strict=False)):
        tested = " x ".join(map(str, SHELL_TESTED_SIZE))
        experimental("shell_resolution", f"grid {' x '.join(map(str, sizes))}, largest tested "
                                         f"{tested}")


def _reset() -> None:
    """Forget which features have warned (tests)."""
    _warned.clear()
