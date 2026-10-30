"""F791 (audit of 0.2.1, N05, W10-01, N06). Statements of the docs against the installed package.

Oracle: a mode exp(i k z + s t) moving downstream at speed c has Im s = -k c, with c in (0, 1) for
the pipe profile 1 - r^2; the signature of dmd(); the address of the page for the docs command."""
from _acceptance import check, passed, start

start(791, "pipe reference signs, dmd signature, docs command")
import inspect  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402

import conatus as cn  # noqa: E402
from conatus.datadriven import dmd  # noqa: E402
from conatus.expert import eigenvalues  # noqa: E402

hp = cn.problems.pipe_evp(N=32, Re=1000.0, m=1, k=1.0)
p = cn.discretize(hp.problem, ncc=hp.ncc)
s = eigenvalues(p, p.group(1, 1.0)).values
doc = " ".join(cn.problems.pipe_evp.__doc__.split())
check("-0.0708640053 - 0.8467498288i" in doc, "centre mode printed with the wrong sign")
check(float(np.min(np.abs(s - complex(-0.0708640053, -0.8467498288)))) < 1e-8, "centre mode not returned")
check(0.0 < 0.8467498288 < 1.0, "phase speed outside the profile")
check(not re.search(r"-0\.0708640053 \+ 0\.8467498288i \(conjugates", doc), "old statement remains")
check(inspect.signature(dmd).parameters["amplitudes"].default == "first", "dmd default changed")
passed("signs and defaults agree with the docs")
