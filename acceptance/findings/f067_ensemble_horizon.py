"""F067: ensemble_fate() must not run past max_steps.

check_every=4, max_steps=10, nobody ever decided: 0.1.0 ran 12 steps (three blocks of 4).
Expected: exactly 10 steps (last block shortened) or a refusal of a max_steps that is not a
multiple of check_every.
"""
import numpy as np
from _acceptance import check, passed, refused, start

start(67, "ensemble horizon is respected")
from conatus.ensembles import ensemble_fate  # noqa: E402


class Ensemble:
    """The duck type ensemble_fate() drives: B members, run(n), t, dt."""
    B, dt, t, steps = 3, 0.1, 0.0, 0

    def run(self, n):
        self.steps += int(n)
        self.t = self.steps * self.dt


ens = Ensemble()
was_refused, fate = refused(ensemble_fate, ens, lambda e: np.zeros(e.B, int), check_every=4,
                            min_steps=0, max_steps=10, classifier="never decides")
if was_refused:
    passed("max_steps not a multiple of check_every refused")
check(ens.steps == 10, f"ran {ens.steps} steps with max_steps=10, expected exactly 10 (reported "
      f"{fate.settings.get('steps_run')})")
check(fate.settings.get("steps_run") == ens.steps, f"reported {fate.settings.get('steps_run')}, ran {ens.steps}")
passed(f"ran {ens.steps} steps")
