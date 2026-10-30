"""F773 (audit of 0.2.1, N12): the Gaussian RBF uses |x - c|^2 for complex states.
Oracle: exp(-1) at x = i and exp(-5) at x = 1 + 2i, center 0."""
from _acceptance import check, passed, start

start(773, "Gaussian RBF of complex states")
import numpy as np  # noqa: E402

from conatus.datadriven import rbf  # noqa: E402

D = rbf(np.array([[0.0]]), epsilon=1.0)
for x, want in ((1j, np.exp(-1)), (1 + 2j, np.exp(-5)), (1.0, np.exp(-1))):
    got = D(np.array([[x]]))[0, -1]
    check(abs(got - want) < 1e-14, f"x = {x}: {got}, expected {want}")
passed("values exp(-|x|^2)")
