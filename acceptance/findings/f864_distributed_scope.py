"""F864: the distributed IVP solver refuses the combinations it does not support, with a
typed error, instead of approximating them.

Oracle: the scope stated on the benchmarks page (Fourier x Chebyshev x Fourier, right-hand side
-u.grad(u) or the poloidal-toroidal form). Refused here: a 2D (Fourier x Chebyshev) domain.
Control: the 3D channel is accepted on
one rank (COMM_SELF) and agrees with the single-process solver to 1e-12 after 3 steps.
"""
import sys
from pathlib import Path

import numpy as np
from _acceptance import check, need, passed, refused, start

start(864, "distributed solver: unsupported combinations refused, supported one agrees")
need("mpi4py")
from mpi4py import MPI  # noqa: E402

import conatus as cn  # noqa: E402
from conatus.distributed import PencilDecomposition  # noqa: E402
from conatus.distributed.imex import DistributedIVPSolver  # noqa: E402
from conatus.timestep import IVPSolver  # noqa: E402

root = Path(cn.__file__).resolve().parents[2]
sys.path.insert(0, str(root / "tests" / "distributed"))
one = PencilDecomposition(MPI.COMM_SELF, (1, 1))
m2 = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=1e4, Pr=1)
p2 = m2.simulate(N=(16, 16), scheme="RK443", dt=1e-3).solver.plan
hit, exc = refused(lambda: DistributedIVPSolver(p2, one, "RK443", 1e-3, backend="cpu",
                                                velocity=p2.problem.variables[0]))
check(hit, f"2D domain not refused: {exc}")
try:
    from dist_cases import channel, initial_velocity
except ImportError:
    from _acceptance import skipped
    skipped("tests/distributed not shipped with this tree")
plan, f = channel(8, 13, 8)
s = DistributedIVPSolver(plan, one, "RK443", 5e-3, backend="cpu", velocity=f["u"])
ref = IVPSolver(plan, "RK443", 5e-3, f_backend="fused")
st = {f["u"]: initial_velocity()}
for x in (s, ref):
    x.set_state(st)
    x.run(3)
d = float(np.abs(s.coeffs("u") - ref.coeffs("u")).max())
check(d < 1e-12, f"channel on one rank vs single process: {d:.1e}")
passed(f"2D refused; channel agrees to {d:.1e}")
