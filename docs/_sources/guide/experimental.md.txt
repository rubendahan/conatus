# Experimental features

Most of Conatus is supported: its API changes only through a declared deprecation, and what the
guide says about it is measured. A few parts are shipped and tested but **not promised**: their
API and defaults may change in a minor release without a deprecation cycle, and some of their
claims rest on fewer measurements. The first use of each in a process emits one
`ExperimentalWarning` that names it:

- **Orbits of 3-D channel and pipe flow**: `conatus.ecs.shooting.ShootingRPO` (relative periodic
  orbits of a channel) and `conatus.ecs.pipe.PipeTravellingWave` ({doc}`dynamics/invariant_solutions`).
- **Large spherical shells**: a shell simulation whose grid is larger, in any direction, than
  64 × 48 × 48 ($N_\phi \times N_\theta \times N_r$), the largest shell grid the test suite runs
  ({doc}`shells`); `conatus._experimental.SHELL_TESTED_SIZE`.
- **The Studio**: the web page, its server and the Python client `conatus.studio.client.connect`
  ({doc}`studio`).
- **The Studio hub and relay**: several runs, remote control and accounts ({doc}`studio_agents`).

The warning is a `UserWarning` (shown under Python's default filters), raised once per feature and
process. To silence it once you have read it:

```python
import warnings

import conatus as cn

warnings.filterwarnings("ignore", category=cn.errors.ExperimentalWarning)
```

Experimental never means unchecked: the same rules hold (residuals reported, nothing substituted
silently), and every number that the documentation gives for these features points to the run
that measured it.
