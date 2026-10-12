"""Sub-step timings of the channel's batched transforms (xp_transforms.FourierFourierChebyshev)
on the GPU, and of candidate replacements (profiling only). Usage: python
profile_ffc_gpu.py [NxxNyxNz ...]; JSON in $CN_OUT/profile_ffc.json."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


def main():
    import cupy as cp
    import cupyx.scipy.fft as cfft
    from conatus.ir import CartesianCoordinates, Chebyshev, RealFourier
    from conatus.timestep.xp_transforms import FourierFourierChebyshev
    out = Path(os.environ.get("CN_OUT", "."))
    sizes = [tuple(map(int, a.split("x"))) for a in sys.argv[1:]] or [(128, 128, 128)]
    res = []
    for Nx, Ny, Nz in sizes:
        cs = CartesianCoordinates("x", "y", "z")
        x, y, z = cs.coords
        bases = [RealFourier(x, Nx, (0, 4 * np.pi), dealias=1.5, grid="smooth7"),
                 Chebyshev(y, Ny, (-1, 1), dealias=1.5, grid="smooth7"),
                 RealFourier(z, Nz, (0, 2 * np.pi), dealias=1.5, grid="smooth7")]
        T = FourierFourierChebyshev(cp, bases, 1)
        _ = T._gpu                     # build the GPU kernels before timing (was an unused K)
        Mc, Ma, Mb = T.M[1], T.M[0], T.M[2]
        row = dict(size=f"{Nx}x{Ny}x{Nz}", grid=[Ma, Mc, Mb])
        ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()

        def t(fn, reps=10):
            fn()
            ev0.record()
            for _ in range(reps):
                fn()
            ev1.record()
            ev1.synchronize()
            return cp.cuda.get_elapsed_time(ev0, ev1) / reps

        for n in (1, 3, 6):
            rng = np.random.default_rng(0)
            c = cp.asarray(rng.standard_normal((n, Nx, Ny, Nz)))  # no curand in the env
            g = cp.asarray(rng.standard_normal((n, Mc, Ma, Mb)))
            a = cp.empty((n, Nx, Nz, Mc + 1), dtype=complex)
            f = cp.fft.irfft(a, n=2 * Mc, axis=-1)
            F = cp.empty((n, Mc, Ma, Mb // 2 + 1), dtype=complex)
            r = dict(n=n)
            r["backward_ms"] = t(lambda: T.backward(c))
            r["forward_ms"] = t(lambda: T.forward(g))
            r["cheb_irfft_2M_ms"] = t(lambda: cp.fft.irfft(a, n=2 * Mc, axis=-1))
            r["cheb_rfft_2M_ms"] = t(lambda: cp.fft.rfft(f, axis=-1))
            r["irfft2_ms"] = t(lambda: cp.fft.irfft2(F, s=(Ma, Mb), axes=(-2, -1)))
            r["rfft2_ms"] = t(lambda: cp.fft.rfft2(g, axes=(-2, -1)))
            plan = cfft.get_fft_plan(F, shape=(Ma, Mb), axes=(-2, -1), value_type="C2R")
            r["irfft2_cupyx_plan_ms"] = t(lambda: cfft.irfft2(F, s=(Ma, Mb), axes=(-2, -1), plan=plan))
            plan2 = cfft.get_fft_plan(g, axes=(-2, -1), value_type="R2C")
            r["rfft2_cupyx_plan_ms"] = t(lambda: cfft.rfft2(g, axes=(-2, -1), plan=plan2))
            Fw = F.copy()
            r["irfft2_cupyx_overwrite_ms"] = t(lambda: cfft.irfft2(Fw, s=(Ma, Mb), axes=(-2, -1),
                                                                   overwrite_x=True, plan=plan))
            aM = cp.zeros((n, Nx, Nz, Mc // 2 + 1), dtype=complex)
            r["cheb_irfft_M_ms"] = t(lambda: cp.fft.irfft(aM, n=Mc, axis=-1))
            r["cheb_irfft_M_overwrite_ms"] = t(lambda: cfft.irfft(aM, n=Mc, axis=-1, overwrite_x=True))
            r["copy_grid_ms"] = t(lambda: g.copy())
            res.append(dict(row, **r))
            print(json.dumps(res[-1]), flush=True)
    (out / "profile_ffc.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
