"""Stacked right-hand side of a 2D RealFourier(x) x Chebyshev(z) problem in a few GPU kernels.

Same numbers as ``IVPSolver.F`` with ``f_backend="fused"`` (the polynomial :class:`FusedRHS`
followed by ``gather_rhs``), same algorithms, fewer passes over memory and ~10 kernels instead of
~80 per evaluation. From the stacked state ``X`` to the stacked RHS vector ``b``:

1. ``fc_gather`` / ``fc_zsteps`` / ``fc_zin``: every factor ``D^a f``: gathered from ``X`` (index
   map), Fourier derivatives (cos/-sin swap times k), Chebyshev derivatives (backward recurrence,
   one thread per column), in the order of ``fused._peel``; then the input of the Chebyshev
   backward transform;
2. cuFFT Z2D (Chebyshev backward): length Mz with the DCT reordering of Makhoul (default,
   ``chebyshev="dct"``), or length 2 Mz exactly as ``xp_transforms.backward`` (``"fft2m"``);
3. ``fc_zx``: scaling and the RealFourier half spectrum (cos, -sin) -> F_k, tiled transpose to
   rows of x;
4. cuFFT Z2D of length Mx (RealFourier backward): the factors on the dealiased grid;
5. ``fc_products``: the generated pointwise kernel of the FusedRHS monomials;
6. cuFFT D2Z of length Mx; 7. ``fc_xz``: (cos, -sin) coefficients, first N modes kept (the
   dealiasing projection), tiled transpose to columns of z (zero padding to 2 Mz for "fft2m");
8. cuFFT D2Z (Chebyshev forward, same choice); 9. ``fc_out``: twiddle, first N modes kept,
   conversion to each row's C^(lam) basis (banded, ``conversion_chain``), written to the rows of
   ``b``; the rows of constant right-hand sides (boundary values) come from a vector computed
   once (parameters are frozen, as in FusedRHS).

Applicability is checked (:meth:`FourierChebyshev2DRHS.reason`); when it does not apply, the
caller keeps the general FusedRHS path and ``explain()`` says why. Nothing is approximated: the
difference with the general path is FFT rounding (checked by the tests).
"""
from __future__ import annotations

import numpy as np

from ..ir.bases import Chebyshev, RealFourier
from ..errors import OptionError

__all__ = ["FourierChebyshev2DRHS"]

_SRC = r"""
#include <cupy/complex.cuh>
typedef complex<double> cd;

// Factor columns, in three passes (a column-sequential kernel was latency-bound):
//   fc_scatter, fc_gather : X -> V (per used variable component, (n, m) layout), then thread
//               (f, n, m): Cf[f][n][m] = V[src], times the x-steps before the first z-step;
//   fc_zsteps : thread (f, m): the remaining steps (Chebyshev recurrences, x-steps between and
//               after them), Cf <-> Cs; nothing to do for factors without a z-step;
//   fc_zin    : thread (f, m, k): the Chebyshev backward-transform input from the final column
//               (in Cf if the factor has an even number of z-steps, else in Cs):
//     dct = 0: Zin[n] = tw_n c_n (n < Nz), 0 up to zlen (length-2Mz real FFT, xp_transforms);
//     dct = 1: Zin[k] = tw_k (X_k - i X_{Mz-k}), k < zlen, X_0 = c_0, X_n = (-1)^n c_n / 2
//              (DCT-III by a length-Mz real FFT, Makhoul; the grid comes out even/odd permuted).
// Steps (0 = x, 1 = z) replay fused._peel in order; d/dx maps column c to c ^ 1 times k (sign -
// for an even (cos) target), the recurrence is the one of the fused evaluator.
__device__ inline int fc_npre(const int* st, int ns) {
    int k = 0;
    while (k < ns && st[k] == 0) ++k;
    return k;
}

// X (stacked, group-major) -> V[v][n][m] for the variable components used by the factors: one
// coalesced read of X; V entries without a source (e.g. the -sin 0 column) stay zero.
extern "C" __global__ void fc_scatter(const double* X, const int* xvar, const int* xpos, double* V,
        long long nX, long long NzNx) {
    long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nX) return;
    const int v = xvar[j];
    if (v >= 0) V[v * NzNx + xpos[j]] = X[j];
}

extern "C" __global__ void fc_gather(const double* V, const int* fvar,
        const int* fsteps, const int* fnsteps, const double* kx, double* Cf, int nf, int Nx,
        int Nz, int maxs) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)nf * Nz * Nx) return;
    const int m = (int)(t % Nx);
    const int n = (int)((t / Nx) % Nz);
    const int f = (int)(t / ((long long)Nx * Nz));
    const int ns = fnsteps[f];
    const int* st = fsteps + f * maxs;
    int nx = 0;
    for (int s = 0; s < ns; ++s) nx += st[s] == 0;
    const int npre = fc_npre(st, ns);
    const int col = (nx & 1) ? (m ^ 1) : m;             // source column of the whole chain
    double v = V[((long long)fvar[f] * Nz + n) * Nx + col];
    int c = col;
    for (int s = 0; s < npre; ++s) { c ^= 1; v = kx[c] * ((c & 1) == 0 ? -v : v); }
    Cf[t] = v;
}

extern "C" __global__ void fc_zsteps(const int* fsteps, const int* fnsteps, const double* kx,
        double sz, double* Cf, double* Cs, int nf, int Nx, int Nz, int maxs) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)nf * Nx) return;
    const int f = (int)(t / Nx), m = (int)(t % Nx);
    const int ns = fnsteps[f];
    const int* st = fsteps + f * maxs;
    const int npre = fc_npre(st, ns);
    if (npre == ns) return;
    int nx = 0;
    for (int s = 0; s < ns; ++s) nx += st[s] == 0;
    int c = (nx & 1) ? (m ^ 1) : m;
    for (int s = 0; s < npre; ++s) c ^= 1;
    const long long NS = (long long)Nx;
    double* cur = Cf + (long long)f * Nz * NS + m;
    double* oth = Cs + (long long)f * Nz * NS + m;
    for (int s = npre; s < ns; ++s) {
        if (st[s] == 0) {
            c ^= 1;
            const double k = kx[c];
            const bool neg = (c & 1) == 0;
#pragma unroll 8
            for (int n = 0; n < Nz; ++n) { double v = cur[n * NS]; cur[n * NS] = k * (neg ? -v : v); }
        } else {
            double p2 = 0.0, p1 = 0.0;
            oth[(long long)(Nz - 1) * NS] = 0.0;
#pragma unroll 8
            for (int n = Nz - 2; n >= 0; --n) {
                double v = p2 + (2.0 * (n + 1)) * cur[(long long)(n + 1) * NS];
                p2 = p1; p1 = v;
                oth[(long long)n * NS] = (n == 0 ? 0.5 * v : v) * sz;
            }
            double* tmp = cur; cur = oth; oth = tmp;
        }
    }
}

extern "C" __global__ void fc_zin(const int* fsteps, const int* fnsteps, const double* Cf,
        const double* Cs, cd* Zin, const cd* tw, int nf, int Nx, int Nz, int Mz, int zlen, int maxs,
        int dct) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)nf * zlen * Nx) return;
    const int m = (int)(t % Nx);
    const int k = (int)((t / Nx) % zlen);
    const int f = (int)(t / ((long long)Nx * zlen));
    const int ns = fnsteps[f];
    const int* st = fsteps + f * maxs;
    int nz = 0;
    for (int s = 0; s < ns; ++s) nz += st[s] == 1;
    const double* c = ((nz & 1) ? Cs : Cf) + (long long)f * Nz * Nx + m;
    cd* z = Zin + ((long long)f * Nx + m) * zlen + k;
    if (!dct) { *z = k < Nz ? tw[k] * c[(long long)k * Nx] : cd(0.0, 0.0); return; }
    auto Xn = [&](int j) {
        if (j >= Nz) return 0.0;
        const double v = c[(long long)j * Nx];
        return j == 0 ? v : ((j & 1) ? -0.5 * v : 0.5 * v);
    };
    *z = tw[k] * cd(Xn(k), k > 0 ? -Xn(Mz - k) : 0.0);
}

// One block per Fourier group k (the columns m = 2k, 2k + 1 of every factor): the group's entries
// of X (a contiguous slice of the stacked vector) into shared memory, one thread per (factor,
// column) for the derivative chain (in place: the recurrence keeps c_{n+1} in a register), then
// the whole block writes the Chebyshev backward-transform input of the 2 nf columns (coalesced).
// Same operations as fc_gather / fc_zsteps / fc_zin. Shared memory: (nu 2 Nz + 2 nf (Nz + 1))
// doubles (the column stride Nz + 1 avoids bank conflicts between the chains).
extern "C" __global__ void fc_group(const double* X, const int* gmap, const int* fvar,
        const int* fsteps, const int* fnsteps, const double* kx, double sz, cd* Zin, const cd* tw,
        int nf, int nu, int Nx, int Nz, int Mz, int zlen, int maxs, int dct) {
    extern __shared__ double shm[];
    double* raw = shm;                                  // (u, cs, n)
    double* col = raw + nu * 2 * Nz;                    // (f, cs) columns, stride Nz + 1
    const int CS = Nz + 1;
    const int k = blockIdx.x;
    const int* g = gmap + (long long)k * nu * 2 * Nz;
    for (int i = threadIdx.x; i < nu * 2 * Nz; i += blockDim.x) {
        const int q = g[i];
        raw[i] = q >= 0 ? X[q] : 0.0;
    }
    __syncthreads();
    // columns, and the x-steps before the first z-step, in parallel over (column, n)
    for (int idx = threadIdx.x; idx < 2 * nf * Nz; idx += blockDim.x) {
        const int jj = idx / Nz, n = idx % Nz;
        const int f = jj >> 1, m = 2 * k + (jj & 1);
        const int ns = fnsteps[f];
        const int* st = fsteps + f * maxs;
        int nx = 0, npre = 0;
        for (int s = 0; s < ns; ++s) nx += st[s] == 0;
        while (npre < ns && st[npre] == 0) ++npre;
        int c = (nx & 1) ? (m ^ 1) : m;                 // source column of the chain
        double v = raw[(fvar[f] * 2 + (c & 1)) * Nz + n];
        for (int s = 0; s < npre; ++s) { c ^= 1; v = kx[c] * ((c & 1) == 0 ? -v : v); }
        col[jj * CS + n] = v;
    }
    __syncthreads();
    const int j = threadIdx.x;
    if (j < 2 * nf) {                                   // the rest of the chain, sequential
        const int f = j >> 1, m = 2 * k + (j & 1);
        const int ns = fnsteps[f];
        const int* st = fsteps + f * maxs;
        int nx = 0, npre = 0;
        for (int s = 0; s < ns; ++s) nx += st[s] == 0;
        while (npre < ns && st[npre] == 0) ++npre;
        int c = (nx & 1) ? (m ^ 1) : m;
        for (int s = 0; s < npre; ++s) c ^= 1;
        double* cur = col + j * CS;
        for (int s = npre; s < ns; ++s) {
            if (st[s] == 0) {                           // d/dx: column c -> c ^ 1, times k
                c ^= 1;
                const double kk = kx[c];
                const bool neg = (c & 1) == 0;
                for (int n = 0; n < Nz; ++n) { double v = cur[n]; cur[n] = kk * (neg ? -v : v); }
            } else {                                    // d/dz in place
                double p2 = 0.0, p1 = 0.0, cn1 = cur[Nz - 1];
                cur[Nz - 1] = 0.0;
#pragma unroll 4
                for (int n = Nz - 2; n >= 0; --n) {
                    const double cn = cur[n];
                    double v = p2 + (2.0 * (n + 1)) * cn1;
                    p2 = p1; p1 = v;
                    cur[n] = (n == 0 ? 0.5 * v : v) * sz;
                    cn1 = cn;
                }
            }
        }
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 2 * nf * zlen; idx += blockDim.x) {
        const int jj = idx / zlen, kk = idx % zlen;
        const int f = jj >> 1, m = 2 * k + (jj & 1);
        const double* cc = col + jj * CS;
        cd* z = Zin + ((long long)f * Nx + m) * zlen + kk;
        if (!dct) { *z = kk < Nz ? tw[kk] * cc[kk] : cd(0.0, 0.0); continue; }
        auto Xn = [&](int q) {
            if (q >= Nz) return 0.0;
            const double v = cc[q];
            return q == 0 ? v : ((q & 1) ? -0.5 * v : 0.5 * v);
        };
        *z = tw[kk] * cd(Xn(kk), kk > 0 ? -Xn(Mz - kk) : 0.0);
    }
}

// R1 (f, m, rl) real -> Xin (f, jz, Kx1) complex: F_0 = c_0, F_k = (c_2k + i c_2k+1) / 2
// (k < K), 0 beyond; times Mx. nrm > 0: c = R1 / nrm * nrm (the irfft normalisation, then the
// transform's factor, as xp_transforms.backward). Tile: 32 k x 32 jz.
extern "C" __global__ void fc_zx(const double* R1, cd* Xin, int nf, int Nx, int Mz, int rl, int Kx1,
                                  int K, double Mx, double nrm) {
    __shared__ double re[32][33], im[32][33];
    const int f = blockIdx.z, k0 = blockIdx.x * 32, j0 = blockIdx.y * 32;
    const int tx = threadIdx.x, ty = threadIdx.y;       // 32 x 8
    for (int r = ty; r < 32; r += 8) {                  // load: row k = k0 + r, jz = j0 + tx
        const int k = k0 + r, j = j0 + tx;
        double a = 0.0, b = 0.0;
        if (k < K && j < Mz) {
            a = R1[((long long)f * Nx + 2 * k) * rl + j];
            if (k > 0) b = R1[((long long)f * Nx + 2 * k + 1) * rl + j];
            if (nrm > 0.0) { a = a / nrm * nrm; b = b / nrm * nrm; }
        }
        re[r][tx] = a; im[r][tx] = b;
    }
    __syncthreads();
    for (int r = ty; r < 32; r += 8) {                  // store: jz = j0 + r, k = k0 + tx
        const int j = j0 + r, k = k0 + tx;
        if (j >= Mz || k >= Kx1) continue;
        cd F(0.0, 0.0);
        if (k == 0) F = cd(re[tx][r], 0.0);
        else if (k < K) F = cd(re[tx][r], im[tx][r]) / 2.0;
        Xin[((long long)f * Mz + j) * Kx1 + k] = cd(F.real() * Mx, F.imag() * Mx);
    }
}

// S (o, jz, Kx1) complex (rfft along x) -> R2 (o, m, rl) real: c_0 = Re F_0, c_2k = 2 Re F_k,
// c_2k+1 = 2 Im F_k (k < K, F = S / Mx), c_1 = 0; zero padding for Mz <= jz < rl. Tile 32 x 32.
extern "C" __global__ void fc_xz(const cd* S, double* R2, int no, int Nx, int Mz, int rl, int Kx1,
                                  int K, double Mx) {
    __shared__ double re[32][33], im[32][33];
    const int o = blockIdx.z, k0 = blockIdx.x * 32, j0 = blockIdx.y * 32;
    const int tx = threadIdx.x, ty = threadIdx.y;
    for (int r = ty; r < 32; r += 8) {                  // load: jz = j0 + r, k = k0 + tx
        const int j = j0 + r, k = k0 + tx;
        double a = 0.0, b = 0.0;
        if (j < Mz && k < K) {
            const cd F = S[((long long)o * Mz + j) * Kx1 + k];
            a = F.real() / Mx; b = F.imag() / Mx;
        }
        re[tx][r] = a; im[tx][r] = b;
    }
    __syncthreads();
    const int j = j0 + threadIdx.x;
    for (int r = ty; r < 32; r += 8) {                  // store: column m = 2k, 2k + 1, row jz
        const int k = k0 + r;
        if (k >= K || j >= Mz) continue;
        double* c0 = R2 + ((long long)o * Nx + 2 * k) * rl;
        double* c1 = c0 + rl;
        const double a = re[r][threadIdx.x], b = im[r][threadIdx.x];
        c0[j] = k == 0 ? a : 2.0 * a;
        c1[j] = k == 0 ? 0.0 : 2.0 * b;
        for (int jj = Mz + j; jj < rl; jj += Mz) { c0[jj] = 0.0; c1[jj] = 0.0; }
    }
}

// Q (o, m, ql) complex (forward real FFT of each column) -> Chebyshev coefficients
//   dct = 0: c_n = Re(tw_n Q_n) * s2, halved at n = 0 (length-2Mz FFT, as xp_transforms.forward);
//   dct = 1: c_n = Re(tw_n V_n), V_n = Q_n (n <= Mz/2) or conj(Q_{Mz-n}), halved at n = 0 (tw
//            includes (-1)^n 2/Mz: DCT-II by a length-Mz real FFT of the permuted grid);
// then converted to C^(lam) (CSR rows of conversion_chain per lam id) and written to b.
extern "C" __global__ void fc_out(const cd* Q, const cd* twf, double s2, const int* rowsrc,
        const double* b0, const signed char* lamid, const int* cptr, const int* cind,
        const double* cval, double* b, long long nrow, int Nz, int Mz, int ql, int dct) {
    // one thread per stacked row: rows of a field-dependent right-hand side are computed from
    // Q, the others take the constant vector b0 (boundary values, zeros)
    const long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrow) return;
    const int t = rowsrc[row];
    if (t < 0) { b[row] = b0[row]; return; }
    const int n = t % Nz;
    const long long om = t / Nz;
    const cd* q = Q + om * ql;
    auto coef = [&](int j) {
        double c;
        if (!dct) {
            c = (twf[j] * q[j]).real() * s2;
        } else {
            const cd V = 2 * j <= Mz ? q[j] : conj(q[Mz - j]);
            c = (twf[j] * V).real();
        }
        return j == 0 ? 0.5 * c : c;
    };
    const int L = lamid[t];
    if (L < 0) { b[row] = coef(n); return; }
    const int* ptr = cptr + (long long)L * (Nz + 1);     // CSR row pointers of lam id L
    double acc = 0.0;
    for (int p = ptr[n]; p < ptr[n + 1]; ++p) acc += cval[p] * coef(cind[p]);
    b[row] = acc;
}
"""
_mod = None


def _module(cp):
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=_SRC, options=("-std=c++17",))
    return _mod


def host_maps(fused, host) -> dict:
    """Index maps of the pipeline (NumPy): factor sources in X, derivative steps, output rows and
    C^(lam) conversions. Host-only, so the logic is testable without a GPU."""
    bx, bz = fused.bases
    Nx, Nz = bx.size, bz.size
    fl = fused.factor_list
    nf, no = len(fl), len(fused.outputs)
    # variable components -> index maps into X
    comps: dict = {}
    srcmaps = []
    for vk, v in enumerate(host.variables):
        sel = np.nonzero(host.cols_obj == vk)[0]
        flat = host.cols_flat[sel]
        shape = host.var_shapes[vk]
        if tuple(shape[-2:]) != (Nx, Nz):
            continue                                  # taus, gauges: never a factor (checked)
        inv = np.full(int(np.prod(shape)), -1, np.int64)
        inv[flat] = sel
        inv = inv.reshape(shape)
        lead = shape[:-2]
        for comp in np.ndindex(*lead) if lead else [()]:
            comps[(v.name, tuple(comp))] = len(srcmaps)
            srcmaps.append(inv[comp] if comp else inv)
    if host.n >= 2**31:
        raise ValueError("FourierChebyshev2DRHS: system too large for int32 maps")
    if any((f.name, tuple(c)) not in comps for f, c, _ in fl):
        raise ValueError("FourierChebyshev2DRHS: a factor is not a full-domain unknown")
    fvar = [comps[(f.name, tuple(c))] for f, c, _ in fl]
    from .fused import _peel
    steps = [[ax for _, _, ax in _peel(tuple(a))] for _, _, a in fl]
    maxs = max(1, max(len(s) for s in steps))
    fst = np.zeros((nf, maxs), np.int32)
    for j, s in enumerate(steps):
        fst[j, :len(s)] = s
    # output rows: (o, m, n) -> row of b and conversion id
    rowmap = np.full((no, Nx, Nz), -1, np.int64)
    lamid = np.full((no, Nx, Nz), -1, np.int8)
    lams: list = []
    eq_out = {(k, tuple(comp)): o for o, (k, comp, _) in enumerate(fused.outputs)}
    for (k, lam), sel in host._row_sel.items():
        eq = host.plan.problem.equations[k]
        shape = host.eq_shapes[k]
        idx = np.unravel_index(host.rows_flat[sel], shape)
        if len(shape) > 2:
            lead = np.stack(idx[:-2], axis=1)
            ucomp, inv = np.unique(lead, axis=0, return_inverse=True)
            omap = np.array([eq_out.get((k, tuple(int(a) for a in c)), -1) for c in ucomp])
            o = omap[np.asarray(inv).reshape(-1)]
        else:
            o = np.full(len(sel), eq_out.get((k, ()), -1))
        if np.all(o < 0):
            continue
        if np.any(o < 0):
            raise ValueError(f"FourierChebyshev2DRHS: equation {k} rows only partly covered")
        lid = -1
        if lam > 0 and shape[eq.lhs.rank + 1] > 1:       # z axis (bases checked: (x, z))
            if lam not in lams:
                lams.append(lam)
            lid = lams.index(lam)
        rowmap[o, idx[-2], idx[-1]] = sel
        lamid[o, idx[-2], idx[-1]] = lid
    ptrs, inds, vals = [], [], []
    nnz = 0
    for lam in lams:
        S = host.plan.layout.zconv(lam).tocsr()      # the layout's conversion (family-aware)
        S.sort_indices()
        ptrs.append(S.indptr.astype(np.int64) + nnz)
        inds.append(S.indices)
        vals.append(S.data)
        nnz += S.nnz
    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(1, dt)  # noqa: E731
    # scatter map of the device path: X[j] -> V[vu][n][m] for the components used by factors
    used = sorted(set(fvar))
    srcs = np.stack(srcmaps)
    xvar = np.full(host.n, -1, np.int32)
    xpos = np.zeros(host.n, np.int32)
    for u, vc in enumerate(used):
        m_, n_ = np.nonzero(srcs[vc] >= 0)
        j = srcs[vc][m_, n_]
        xvar[j] = u
        xpos[j] = n_ * Nx + m_
    # per Fourier group k: X indices of the used components, (k, u, cs, n) (fc_group)
    gmap = np.stack([srcs[vc] for vc in used])                     # (u, m, n)
    gmap = gmap.reshape(len(used), Nx // 2, 2, Nz).transpose(1, 0, 2, 3)
    rowsrc = np.full(host.n, -1, np.int64)                       # stacked row -> (o, m, n)
    ok = rowmap >= 0
    rowsrc[rowmap[ok]] = np.nonzero(ok.reshape(-1))[0]
    return dict(srcmap=srcs.astype(np.int32), fvar=np.array(fvar, np.int32),
                rowsrc=rowsrc.astype(np.int32),
                gmap=np.ascontiguousarray(gmap).astype(np.int32),
                fvar_used=np.array([used.index(v) for v in fvar], np.int32), nused=len(used),
                xvar=xvar, xpos=xpos,
                fsteps=fst, fnsteps=np.array([len(s) for s in steps], np.int32), maxs=maxs,
                kx=np.repeat(2 * np.pi * np.arange(Nx // 2) / bx.length, 2), sz=2.0 / bz.length,
                rowmap=rowmap.astype(np.int32), lamid=lamid, cptr=cat(ptrs, np.int32),
                cind=cat(inds, np.int32), cval=cat(vals, np.float64))


class FourierChebyshev2DRHS:
    """Stacked RHS of a 2D RealFourier x Chebyshev polynomial FusedRHS on the GPU (module doc)."""

    @staticmethod
    def reason(fused, system, assume_gpu: bool = False) -> str | None:
        """None if the fast path applies, else why not (shown by IVPSolver.explain).
        ``assume_gpu``: judge a host-built FusedRHS as if it were the GPU one (memory planner)."""
        if fused is None or not (assume_gpu or getattr(fused, "_gpu", False)):
            return "needs the fused RHS on the GPU"
        if fused.mode != "polynomial" or fused.reflection is not None or fused.cplx:
            return "needs a real polynomial RHS in the full space"
        if len(fused.bases) != 2 or not isinstance(fused.bases[0], RealFourier) \
                or not isinstance(fused.bases[1], Chebyshev):
            return "needs bases (RealFourier x, Chebyshev z)"
        if np.dtype(system.dtype) != np.float64:
            return "needs a real stacked system"
        names = {v.name for v in system.variables}
        if any(f.name not in names for f, _, _ in fused.factor_list):
            return "a factor is not an unknown (non-constant coefficient data)"
        if not fused.factor_list:
            return "no field-dependent factor"
        shape = tuple(b.size for b in fused.bases)
        for k, _comp, _ in fused.outputs:
            if tuple(system.eq_shapes[k][-2:]) != shape:
                return f"equation {k} is not on the full (x, z) domain"
        if system.plan.layout.nz != fused.bases[1].size:
            return "Chebyshev size differs from the layout"
        bx = fused.bases[0]
        if bx.size % 2 or bx.grid_size % 2:
            # checked here, not only in __init__: an odd dealiased x grid (Nx = 18 -> Mx = 27)
            # used to raise from IVPSolver(...) instead of keeping the general path
            return f"needs even Nx and x grid size (Nx = {bx.size}, Mx = {bx.grid_size})"
        return None

    def __init__(self, fused, system, const_rhs, chebyshev: str = "dct") -> None:
        import cupy as cp
        self.cp, self.fused, self.system = cp, fused, system
        host = getattr(system, "host", system)
        bx, bz = fused.bases
        Nx, Nz, Mx, Mz = bx.size, bz.size, bx.grid_size, bz.grid_size
        if Mx % 2 or Nx % 2:
            raise OptionError("FourierChebyshev2DRHS: even Nx and grid size required",
                              fix="an even Nx and an even grid size")
        self.Nx, self.Nz, self.Mx, self.Mz = Nx, Nz, Mx, Mz
        self.K, self.Kx1 = Nx // 2, Mx // 2 + 1
        if chebyshev not in ("fft2m", "dct"):
            raise OptionError("chebyshev must be 'fft2m' (the evaluator's transform) or 'dct'",
                              fix="chebyshev='fft2m' or chebyshev='dct'")
        self.chebyshev = chebyshev
        dct = chebyshev == "dct"
        # Chebyshev transforms: length-2Mz real FFTs (zlen = Mz + 1 complex inputs, rows of 2Mz)
        # or length-Mz real FFTs (DCT by Makhoul's reordering: Mz/2 + 1 inputs, rows of Mz)
        self.flen = Mz if dct else 2 * Mz
        self.zlen = self.flen // 2 + 1
        self._src = fused._src
        fl = fused.factor_list
        nf, no = len(fl), len(fused.outputs)
        self.nf, self.no = nf, no
        mp = host_maps(fused, host)
        self._maxs = mp["maxs"]
        self._sz = mp["sz"]
        for key in ("gmap", "fvar_used", "xvar", "xpos", "fsteps", "fnsteps", "kx", "rowmap", "rowsrc", "lamid", "cptr",
                    "cind", "cval"):
            setattr(self, "_" + key, cp.asarray(mp[key]))
        from . import xp_transforms as XT
        if dct:
            k = np.arange(self.zlen)
            self._twb = cp.asarray(np.exp(1j * np.pi * k / (2 * Mz)))
            n = np.arange(Nz)
            self._twf = cp.asarray((-1.0) ** n * (2.0 / Mz) * np.exp(-1j * np.pi * n / (2 * Mz)))
        else:                                  # exactly as xp_transforms (same device arithmetic)
            self._twb = cp.ascontiguousarray(XT._twiddle(cp, Mz, Nz, 1.0) * XT._half(cp, Nz))
            self._twf = cp.ascontiguousarray(XT._twiddle(cp, Mz, Nz, -1.0))
        self._s2 = 2.0 / Mz
        self._b0 = cp.ascontiguousarray(cp.asarray(const_rhs, dtype=np.float64))
        # persistent work buffers (never allocated during a capture)
        self._V = cp.zeros((mp["nused"], Nz, Nx))       # entries without a source stay zero
        self._Cf = cp.empty((nf, Nz, Nx))
        self._Cs = cp.empty((nf, Nz, Nx))
        self._Zin = cp.empty((nf * Nx, self.zlen), dtype=np.complex128)
        self._R1 = cp.empty((nf * Nx, self.flen))
        self._Xin = cp.empty((nf * Mz, self.Kx1), dtype=np.complex128)
        self._G = cp.empty((nf, Mz, Mx))
        self._O = cp.empty((no, Mz, Mx))
        self._S = cp.empty((no * Mz, self.Kx1), dtype=np.complex128)
        self._R2 = cp.zeros((no * Nx, self.flen))
        self._Q = cp.empty((no * Nx, self.zlen), dtype=np.complex128)
        from cupy.cuda import cufft
        self._cufft = cufft
        self._p1 = cufft.Plan1d(self.flen, cufft.CUFFT_Z2D, nf * Nx)
        self._p2 = cufft.Plan1d(Mx, cufft.CUFFT_Z2D, nf * Mz)
        self._p3 = cufft.Plan1d(Mx, cufft.CUFFT_D2Z, no * Mz)
        self._p4 = cufft.Plan1d(self.flen, cufft.CUFFT_D2Z, no * Nx)
        mod = _module(cp)
        self._kg = mod.get_function("fc_group")
        self._nused = mp["nused"]
        self._gsmem = 8 * (self._nused * 2 * Nz + 2 * nf * (Nz + 1))
        optin = cp.cuda.Device().attributes["MaxSharedMemoryPerBlockOptin"]
        self.factor_kernel = "group" if self._gsmem <= optin else "passes"
        # The limit is an attribute of the kernel, shared by every instance of the process: only
        # raise it (setting it to this instance's size broke an earlier, larger one).
        if self.factor_kernel == "group" and self._gsmem > self._kg.max_dynamic_shared_size_bytes:
            self._kg.max_dynamic_shared_size_bytes = self._gsmem
        self._k1s = mod.get_function("fc_scatter")
        self._k1a = mod.get_function("fc_gather")
        self._k1b = mod.get_function("fc_zsteps")
        self._k1c = mod.get_function("fc_zin")
        self._zsteps = bool(np.any(mp["fsteps"][np.arange(mp["fsteps"].shape[1])[None, :]
                                               < mp["fnsteps"][:, None]] == 1))
        self._k2 = mod.get_function("fc_zx")
        self._k4 = mod.get_function("fc_xz")
        self._k5 = mod.get_function("fc_out")
        self._k3 = cp.RawKernel(self._products_source(), "fc_products", options=("-std=c++17",))
        self._k3.compile()
        self.stage_hook = None

    def _products_source(self) -> str:
        """FusedRHS monomials; each factor read as irfft output (raw / Mx), as cupy.fft.irfft."""
        f = self.fused
        lines = ["#include <cupy/complex.cuh>",
                 'extern "C" __global__ void fc_products(const double* g, double* out, long long P, double M) {',
                 "  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;",
                 "  if (i >= P) return;"]
        for j in range(self.nf):
            lines.append(f"  const double f{j} = g[{j}LL * P + i] / M;")
        for o, (_, _, monos) in enumerate(f.outputs):
            terms = ["*".join([f"{float(c)!r}"] + [f"f{j}" for j in fs]) for c, fs in monos]
            lines.append(f"  out[{o}LL * P + i] = " + (" + ".join(terms) if terms else "0.0") + ";")
        lines.append("}")
        return "\n".join(lines)

    def __call__(self, X):
        return self.evaluate(X)

    def evaluate(self, X):
        """Stacked RHS vector F(X) (a new device array)."""
        cp, cf = self.cp, self._cufft
        Nx, Nz, Mz, nf = self.Nx, self.Nz, self.Mz, self.nf
        X = cp.ascontiguousarray(X)
        hook = self.stage_hook or (lambda name: None)    # timing probe (benchmarks), off by default
        dct = np.int32(self.chebyshev == "dct")
        if self.factor_kernel == "group":
            self._kg((Nx // 2,), (max(256, ((2 * nf + 31) // 32) * 32),),
                     (X, self._gmap, self._fvar_used, self._fsteps, self._fnsteps, self._kx,
                      np.float64(self._sz), self._Zin, self._twb, np.int32(nf), np.int32(self._nused),
                      np.int32(Nx), np.int32(Nz), np.int32(Mz), np.int32(self.zlen),
                      np.int32(self._maxs), dct), shared_mem=self._gsmem)
            hook("f_group")
        else:
            self._factor_passes(X, dct, hook)
        self._p1.fft(self._Zin, self._R1, cf.CUFFT_INVERSE)
        return self._after_factors(hook)

    def _factor_passes(self, X, dct, hook) -> None:
        """The same factors in four grid-wide passes (fallback when a group does not fit the
        shared memory of a block)."""
        Nx, Nz, Mz, nf = self.Nx, self.Nz, self.Mz, self.nf
        ints = (np.int32(nf), np.int32(Nx), np.int32(Nz))
        nX = X.size
        self._k1s(((nX + 255) // 256,), (256,),
                  (X, self._xvar, self._xpos, self._V, np.int64(nX), np.int64(Nz * Nx)))
        hook("f_scatter")
        tot = nf * Nz * Nx
        self._k1a(((tot + 255) // 256,), (256,),
                  (self._V, self._fvar_used, self._fsteps, self._fnsteps, self._kx, self._Cf, *ints,
                   np.int32(self._maxs)))
        hook("f_gather")
        if self._zsteps:
            tot = nf * Nx
            self._k1b(((tot + 127) // 128,), (128,),
                      (self._fsteps, self._fnsteps, self._kx, np.float64(self._sz), self._Cf, self._Cs,
                       *ints, np.int32(self._maxs)))
        hook("f_zsteps")
        tot = nf * self.zlen * Nx
        self._k1c(((tot + 255) // 256,), (256,),
                  (self._fsteps, self._fnsteps, self._Cf, self._Cs, self._Zin, self._twb, *ints,
                   np.int32(Mz), np.int32(self.zlen), np.int32(self._maxs), dct))
        hook("f_zin")

    def _after_factors(self, hook):
        cf = self._cufft
        Nx, Nz, Mx, Mz = self.Nx, self.Nz, self.Mx, self.Mz
        nf, no = self.nf, self.no
        dct = np.int32(self.chebyshev == "dct")
        hook("fft_z_backward")
        self._k2(((self.Kx1 + 31) // 32, (Mz + 31) // 32, nf), (32, 8),
                 (self._R1, self._Xin, np.int32(nf), np.int32(Nx), np.int32(Mz), np.int32(self.flen),
                  np.int32(self.Kx1), np.int32(self.K), np.float64(Mx),
                  np.float64(0.0 if dct else 2.0 * Mz)))
        hook("zx_transpose")
        self._p2.fft(self._Xin, self._G, cf.CUFFT_INVERSE)
        hook("fft_x_backward")
        P = Mz * Mx
        self._k3(((P + 255) // 256,), (256,), (self._G, self._O, np.int64(P), np.float64(Mx)))
        hook("products")
        self._p3.fft(self._O, self._S, cf.CUFFT_FORWARD)
        hook("fft_x_forward")
        self._k4(((self.K + 31) // 32, (Mz + 31) // 32, no), (32, 8),
                 (self._S, self._R2, np.int32(no), np.int32(Nx), np.int32(Mz), np.int32(self.flen),
                  np.int32(self.Kx1), np.int32(self.K), np.float64(Mx)))
        hook("xz_transpose")
        self._p4.fft(self._R2, self._Q, cf.CUFFT_FORWARD)
        hook("fft_z_forward")
        b = self.cp.empty_like(self._b0)
        nrow = b.size
        self._k5(((nrow + 255) // 256,), (256,),
                 (self._Q, self._twf, np.float64(self._s2), self._rowsrc, self._b0, self._lamid,
                  self._cptr, self._cind, self._cval, b, np.int64(nrow), np.int32(Nz), np.int32(Mz),
                  np.int32(self.zlen), dct))
        hook("out")
        return b

    def explain(self) -> str:
        return (f"  F: 2D RealFourier x Chebyshev pipeline ({self.nf} factors -> {self.no} outputs, "
                f"grid {self.Mx} x {self.Mz}): factors from X ("
                + ("one block per Fourier pair" if self.factor_kernel == "group" else
                   "four grid-wide passes: a Fourier pair does not fit the shared memory of a block")
                + "), cuFFT (Chebyshev "
                + ("DCT by length-M real FFTs (Makhoul)" if self.chebyshev == "dct" else
                   "as length-2M real FFTs, the evaluator's algorithm")
                + ", then x), fused products, cuFFT back, C^(lam) conversion into the stacked "
                "rows; ~10 kernels per evaluation; same numbers as the fused evaluator to FFT "
                "rounding")
