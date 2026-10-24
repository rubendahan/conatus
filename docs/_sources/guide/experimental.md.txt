# Experimental features

Most of the Conatus API changes only through a declared deprecation. The experimental APIs below
may change in a minor release without a deprecation cycle. Some of their documented results rest
on fewer measurements than the rest of the library. The first use of each in a process emits one
`ExperimentalWarning` that names it:

- **Orbits of 3-D channel and pipe flow**: `conatus.ecs.shooting.ShootingRPO` (relative periodic
  orbits of a channel) and `conatus.ecs.pipe.PipeTravellingWave` ({doc}`dynamics/invariant_solutions`).
- **Large spherical shells**: a shell simulation whose grid is larger, in any direction, than
  64 × 48 × 48 ($N_\phi \times N_\theta \times N_r$), the largest shell grid the test suite runs
  ({doc}`shells`); `conatus._experimental.SHELL_TESTED_SIZE`.
- **The Studio**: the web page, its server and the Python client `conatus.studio.client.connect`
  ({doc}`studio`).
- **The Studio hub** (`conatus.studio.hub serve --local`, which starts a relay in the same
  process): several runs, accounts and the agent channel ({doc}`studio_agents`). A hub hosted
  outside 127.0.0.1 with remote relays is not supported in this release.

The warning is a `UserWarning` (shown under Python's default filters), raised once per feature and
process. To silence it once you have read it:

```python
import warnings

import conatus as cn

warnings.filterwarnings("ignore", category=cn.errors.ExperimentalWarning)
```

The same checks apply to these features as to the rest of the library: residuals are reported and
numerical choices are declared.
