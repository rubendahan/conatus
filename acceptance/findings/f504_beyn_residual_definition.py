"""F504 (external review of 0.1.1, F18): the documented nonlinear residual was not the one reported.

For f(z) = z^2 - 2 the guide said ||T(z) v|| / (||T(z)|| ||v||), which is 1 for a scalar; the
reported value used a reference scale from the boundary. Correct behaviour: the result states
the formula it uses, and the value matches it (computed here independently).
"""
import numpy as np
from _acceptance import check, passed, start

import conatus as cn

start(504, "the reported Beyn residual is the stated one")
r = cn.contour.beyn(lambda z: z * z - 2, cn.box(re=(1, 2), im=(-1, 1)), poles=0)
z = r.values[0]
pts = [1 - 1j, 2 - 1j, 2 + 1j, 1 + 1j, 1.5 - 1j, 2 + 0j, 1.5 + 1j, 1 + 0j]
s_ref = float(np.median([abs(p * p - 2) for p in pts]))
expected = abs(z * z - 2) / max(abs(z * z - 2), s_ref)
definition = getattr(r, "residual_definition", "")
check("median" in definition, f"no stated residual scale (residual_definition={definition!r})")
check("||T(z) v|| / (||T(z)|| ||v||)" not in r.explain(), "explain() still states the unscaled formula")
check(abs(r.residuals[0] - expected) <= 1e-6 * expected + 1e-300,
      f"residual {r.residuals[0]:.3e}, stated formula gives {expected:.3e}")
passed(f"residual {r.residuals[0]:.3e} = |f| / max(|f|, s_ref = {s_ref:.4g}); {definition[:60]}...")
