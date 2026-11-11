"""F850 (CUDA risk items of v0.1.1): the device IMEX integrator does not carry the SBDF
history of one trajectory into another.

``conatus.runtime.cuda.Imex`` kept the M X and F history between ``run`` calls; a run on a new
state used the old history silently (the host integrator was fixed for this in 0.1.1, finding 42).
Oracle: a fresh integrator started on the new state. Checks: a run on another state is refused
with a reason naming reset(); after reset() the result equals the fresh integrator bit for bit;
two calls that continue one trajectory equal one call of the summed steps bit for bit.
SKIP without a GPU.
"""
import numpy as np
import scipy.sparse as sp
from _acceptance import check, passed, skipped, start

start(850, "device IMEX: no SBDF history carried into another trajectory")
from conatus.runtime import cuda  # noqa: E402

if not cuda.available():
    skipped("no CUDA device (conatus.runtime.cuda.available() is False)")


def system(N, kappa):
    """u_t - u_xx + kappa u + taus = 0 on Chebyshev with u(+-1) = 0 (tau form)."""
    from conatus.discretize.ultraspherical import conversion_chain, differentiation
    S, D2 = conversion_chain(N, 0, 2), differentiation(N, 0, 2)
    M, L = sp.lil_matrix((N + 2, N + 2)), sp.lil_matrix((N + 2, N + 2))
    M[:N, :N] = S
    L[:N, :N] = -D2 + kappa * S
    L[N - 1, N] = L[N - 2, N + 1] = 1.0
    L[N, :N] = (-1.0) ** np.arange(N)
    L[N + 1, :N] = 1.0
    return M.tocsr(), L.tocsr()


N, dt = 24, 2e-3
sy = [system(N, k) for k in (0.0, 1.0, 2.0)]
rng = np.random.default_rng(850)
xa, xb = np.zeros((3, N + 2)), np.zeros((3, N + 2))
xa[:, :N] = rng.standard_normal((3, N)) * np.exp(-0.3 * np.arange(N))
xb[:, :N] = rng.standard_normal((3, N)) * np.exp(-0.3 * np.arange(N))


def imex():
    return cuda.Imex([s[0] for s in sy], [s[1] for s in sy], [N, N + 1], "SBDF3", dt)


im = imex()
im.run(cuda.to_device(xa), 0.0, 20)
db = cuda.to_device(xb)
try:
    im.run(db, 0.0, 20)
    check(False, "a run on another state with the old SBDF history was accepted")
except ValueError as exc:
    check("reset" in str(exc), f"refusal without the remedy: {exc}")
im.reset()
im.run(db, 0.0, 20)
fresh = cuda.to_device(xb)
imex().run(fresh, 0.0, 20)
check(np.array_equal(db.to_host(), fresh.to_host()), "after reset(): differs from a fresh integrator")
one, two = cuda.to_device(xa), cuda.to_device(xa)
imex().run(one, 0.0, 30)
im2 = imex()
im2.run(two, im2.run(two, 0.0, 12), 18)
check(np.array_equal(one.to_host(), two.to_host()), "12 + 18 continued steps differ from 30 steps")
passed("refused with reset(); reset() = fresh integrator; continuation = one run (bit for bit)")
