"""Array-module generic grid <-> coefficient transforms (NumPy or CuPy), dealiased grid.

Same conventions as ``conatus.reference.transforms`` (the oracle; see its docstring), written
with FFTs only so that the identical code runs on the GPU through CuPy:

- Chebyshev: with theta_j = pi (2j+1) / (2M), T_n(t_j) = (-1)^n cos(n theta_j). Both
  directions are length-2M FFTs of zero-padded data (no DCT routine needed).
- RealFourier / ComplexFourier: rfft / fft of length M, first N modes kept.

Complex data on a Chebyshev axis is transformed as real + i imag.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..ir.bases import Basis, Chebyshev, ComplexFourier, RealFourier
from ..errors import OptionError

__all__ = ["forward", "backward", "FourierFourierChebyshev"]

_cache: dict = {}


class _ThreadedFFT:
    """scipy.fft with a declared number of worker threads, same call signatures as the
    ``numpy.fft`` functions used below (both pocketfft: same transforms to rounding, not
    bitwise; tests/ecs3d/test_channel_rpo.py)."""

    def __init__(self, workers: int) -> None:
        import scipy.fft as sfft
        self.workers, self._f = int(workers), sfft

    def __getattr__(self, name):
        f = getattr(self._f, name)
        return lambda *a, **k: f(*a, workers=self.workers, **k)


def cpu_fft(workers: int | None = None):
    """FFT module of the CPU path: ``numpy.fft`` by default; with ``workers`` (or the declared
    environment variable CN_FFT_WORKERS) scipy.fft on that many threads (opt-in, reported by
    the caller; channel orbits)."""
    import os
    if workers is None:
        raw = os.environ.get("CN_FFT_WORKERS", "").strip()
        try:
            workers = int(raw or 0)
        except ValueError:
            raise OptionError(f"CN_FFT_WORKERS = {raw!r}: expected a whole number of FFT threads "
                              "(unset or ≤ 1 keeps numpy.fft)",
                              fix="export CN_FFT_WORKERS=4 (a whole number), or unset it") from None
    return _ThreadedFFT(workers) if workers > 1 else np.fft


def _twiddle(xp, M: int, N: int, sign: float):
    key = (id(xp), M, N, sign)
    w = _cache.get(key)
    if w is None:
        n = np.arange(N)
        w = _cache[key] = xp.asarray(np.exp(sign * 1j * np.pi * n / (2 * M)) * (-1.0) ** n)
    return w


def _half(xp, N: int):
    key = (id(xp), "half", N)
    w = _cache.get(key)
    if w is None:
        h = np.full(N, 0.5)
        h[0] = 1.0
        w = _cache[key] = xp.asarray(h)
    return w


def _ashape(v, ndim, axis):
    shp = [1] * ndim
    shp[axis] = v.shape[0]
    return v.reshape(shp)


def _ix(ndim, axis, s):
    idx = [slice(None)] * ndim
    idx[axis] = s
    return tuple(idx)


_FAM_MATS: dict = {}


def _family_apply(xp: Any, x, b: Basis, axis: int, which: int):
    """Hermite / Laguerre / compound Chebyshev on the device: dense transform matrices
    (discretize/families.py, matrix-multiply transforms; same numbers as the NumPy reference up
    to the summation order), cached per (basis, backend)."""
    from ..discretize.families import family_of
    key = (b.key, which, id(xp))
    A = _FAM_MATS.get(key)
    if A is None:
        A = _FAM_MATS[key] = xp.asarray(family_of(b).matrices(True)[which])
    xm = xp.moveaxis(x, axis, 0)
    sh = xm.shape
    out = A.astype(xp.result_type(A, xm)) @ xm.reshape(sh[0], -1)
    return xp.moveaxis(out.reshape((A.shape[0],) + sh[1:]), 0, axis)


def _is_family(b: Basis) -> bool:
    from ..discretize.families import has_family
    return has_family(b)


def forward(xp: Any, f, b: Basis, axis: int = 0):
    """Grid values (M = b.grid_size points along ``axis``) -> N coefficients.
    Works along ``axis`` directly (no moveaxis): cuFFT runs on the strided batch."""
    M, N = b.grid_size, b.size
    nd = f.ndim
    axis = axis % nd
    if f.shape[axis] != M:
        raise ValueError(f"expected {M} grid points along axis {axis}, got {f.shape[axis]}")
    if _is_family(b):
        return _family_apply(xp, f, b, axis, 1)
    if isinstance(b, Chebyshev):
        if xp.iscomplexobj(f):
            return forward(xp, f.real, b, axis) + 1j * forward(xp, f.imag, b, axis)
        # c_n = (-1)^n (2/M) sum_j f_j cos(n theta_j), halved at n = 0; f real: rfft of length
        # 2M gives the M + 1 >= N needed modes at half the cost of a complex FFT
        F = xp.fft.rfft(f, n=2 * M, axis=axis)[_ix(nd, axis, slice(0, N))]
        c = (_ashape(_twiddle(xp, M, N, -1.0), nd, axis) * F).real * (2.0 / M)
        c[_ix(nd, axis, 0)] *= 0.5
    elif isinstance(b, RealFourier):
        if xp.iscomplexobj(f):
            # complex data on a RealFourier axis (a complex system): Re and Im separately, as the
            # reference does; rfft would drop or refuse the imaginary part
            return forward(xp, f.real, b, axis) + 1j * forward(xp, f.imag, b, axis)
        F = xp.fft.rfft(f, axis=axis) / M
        K = N // 2
        shp = list(f.shape)
        shp[axis] = N
        c = xp.zeros(shp, dtype=f.dtype)
        c[_ix(nd, axis, 0)] = F[_ix(nd, axis, 0)].real
        c[_ix(nd, axis, slice(2, N, 2))] = 2 * F[_ix(nd, axis, slice(1, K))].real
        c[_ix(nd, axis, slice(3, N, 2))] = 2 * F[_ix(nd, axis, slice(1, K))].imag
    elif isinstance(b, ComplexFourier):
        F = xp.fft.fft(f, axis=axis) / M
        K = N // 2
        shp = list(f.shape)
        shp[axis] = N
        c = xp.zeros(shp, dtype=complex)
        c[_ix(nd, axis, slice(0, K))] = F[_ix(nd, axis, slice(0, K))]
        c[_ix(nd, axis, slice(K + 1, None))] = F[_ix(nd, axis, slice(M - K + 1, None))]
    else:
        raise NotImplementedError(f"xp transforms: {type(b).__name__} not supported yet")
    return c


def backward(xp: Any, c, b: Basis, axis: int = 0):
    """N coefficients -> grid values on M = b.grid_size points (zero-padded spectrum)."""
    M, N = b.grid_size, b.size
    nd = c.ndim
    axis = axis % nd
    if c.shape[axis] != N:
        raise ValueError(f"expected {N} coefficients along axis {axis}, got {c.shape[axis]}")
    if _is_family(b):
        return _family_apply(xp, c, b, axis, 0)
    if isinstance(b, Chebyshev):
        if xp.iscomplexobj(c):
            return backward(xp, c.real, b, axis) + 1j * backward(xp, c.imag, b, axis)
        # f_j = Re sum_n (-1)^n c_n e^{i n theta_j} = Re[(2M) ifft_{2M}(c_n (-1)^n e^{i pi n/(2M)})]_j
        # Re[ifft_{2M}(a)] = irfft_{2M}(b) with b_0 = a_0, b_n = a_n / 2 (n >= 1; N <= M so the
        # Nyquist bin is empty): a real inverse FFT, half the cost of the complex one
        a = _ashape(_twiddle(xp, M, N, 1.0) * _half(xp, N), nd, axis) * c
        f = xp.fft.irfft(a, n=2 * M, axis=axis)[_ix(nd, axis, slice(0, M))] * (2 * M)
    elif isinstance(b, RealFourier):
        if xp.iscomplexobj(c):
            # complex coefficients: irfft of the folded spectrum returned a REAL array, the
            # imaginary part silently lost (the production twin): Re and Im separately
            return backward(xp, c.real, b, axis) + 1j * backward(xp, c.imag, b, axis)
        K = N // 2
        shp = list(c.shape)
        shp[axis] = M // 2 + 1
        F = xp.zeros(shp, dtype=complex)
        F[_ix(nd, axis, 0)] = c[_ix(nd, axis, 0)]
        F[_ix(nd, axis, slice(1, K))] = (c[_ix(nd, axis, slice(2, N, 2))]
                                         + 1j * c[_ix(nd, axis, slice(3, N, 2))]) / 2
        f = xp.fft.irfft(F * M, n=M, axis=axis)
    elif isinstance(b, ComplexFourier):
        K = N // 2
        shp = list(c.shape)
        shp[axis] = M
        F = xp.zeros(shp, dtype=complex)
        F[_ix(nd, axis, slice(0, K))] = c[_ix(nd, axis, slice(0, K))]
        F[_ix(nd, axis, slice(M - K + 1, None))] = c[_ix(nd, axis, slice(K + 1, None))]
        f = xp.fft.ifft(F * M, axis=axis)
    else:
        raise NotImplementedError(f"xp transforms: {type(b).__name__} not supported yet")
    return f


class FourierFourierChebyshev:
    """Batched 3D transforms for two RealFourier axes and one Chebyshev axis (the
    channel): same numbers as applying :func:`backward` / :func:`forward` axis by axis (tested),
    in far fewer memory passes. The per-axis path moves every non-last axis to the end and back
    (CuPy FFTs), i.e. ~3 copies per axis and per direction; here:

    backward: (n, coefficients) --[twiddle, transpose: Chebyshev axis last]--> irfft of length 2M
      (contiguous batch) --[RealFourier pairs -> complex half spectrum, zero padding, transpose]-->
      irfft2 over the two Fourier axes (contiguous) -> grid values, layout (n, My, Ma, Mb).
    forward: the reverse (rfft2, unpack to cos/sin pairs, Chebyshev rfft of length 2M).

    The grid layout (n, My, Ma, Mb) (Chebyshev axis first, Fourier axes in domain order) is
    internal: products are pointwise, only the coefficient layout is public. The RealFourier
    pair (cos, -sin) products of the two axes are the bicomplex amplitudes of the modes
    exp(i(ka a +- kb b)) (system.py), F(ka, kb) = C+/4, F(-ka, kb) = conj(C-)/4."""

    def __init__(self, xp, bases, cheb_axis: int) -> None:
        fa = [i for i in range(3) if i != cheb_axis]
        for i in fa:
            if not isinstance(bases[i], RealFourier):
                raise NotImplementedError("FourierFourierChebyshev: two RealFourier axes needed")
        if not isinstance(bases[cheb_axis], Chebyshev):
            raise NotImplementedError("FourierFourierChebyshev: one Chebyshev axis needed")
        self.xp, self.bases, self.cy, self.fa = xp, bases, cheb_axis, fa
        self.N = tuple(b.size for b in bases)
        self.M = tuple(b.grid_size for b in bases)
        cy = cheb_axis
        self.tw_b = _twiddle(xp, self.M[cy], self.N[cy], 1.0) * _half(xp, self.N[cy])
        self.tw_f = _twiddle(xp, self.M[cy], self.N[cy], -1.0)
        # coefficient layout (n, N0, N1, N2) -> Chebyshev last, Fourier axes in order
        self.perm_c = (0, 1 + fa[0], 1 + fa[1], 1 + cy)
        self.inv_c = tuple(int(i) for i in np.argsort(self.perm_c))
        self._neg(self.M[fa[0]])                  # device constants built now (graph capture)
        # GPU: the packing steps as generated single-pass kernels (_FFCKernels, same numbers)
        self._gpu = _FFCKernels(self) if xp is not np else None
        self.fft = xp.fft if self._gpu is not None else cpu_fft()

    @property
    def grid_shape(self) -> tuple:
        """Grid shape (non-periodic axis first, then the two Fourier axes)."""
        a, b = self.fa
        return (self.M[self.cy], self.M[a], self.M[b])

    def backward(self, c):
        """Grid values of the coefficients ``c`` (3-D batched transform)."""
        if self._gpu is not None:
            return self._gpu.backward(c)
        xp = self.xp
        cy, (ia, ib) = self.cy, self.fa
        Mc, Ma, Mb = self.M[cy], self.M[ia], self.M[ib]
        Na, Nb = self.N[ia], self.N[ib]
        n = c.shape[0]
        a = xp.transpose(c, self.perm_c) * self.tw_b            # (n, Na, Nb, Ny) complex
        f = self.fft.irfft(a, n=2 * Mc, axis=-1)[..., :Mc] * (2 * Mc)  # (n, Na, Nb, My) real
        Ka, Kb = Na // 2, Nb // 2
        # RealFourier pair products a_pq[ma, mb], p/q = 0 cos, 1 -sin
        a00, a01 = f[:, 0::2, 0::2], f[:, 0::2, 1::2]
        a10, a11 = f[:, 1::2, 0::2], f[:, 1::2, 1::2]
        Cp = (a00 - a11) + 1j * (a01 + a10)                    # (n, Ka, Kb, My)
        Cm = (a00 + a11) + 1j * (a10 - a01)
        F = xp.zeros((n, Mc, Ma, Mb // 2 + 1), dtype=complex)
        s = 0.25 * Ma * Mb
        F[:, :, :Ka, :Kb] = xp.transpose(Cp, (0, 3, 1, 2)) * s
        # negative ka (rows Ma - ma, ma >= 1); ma = 0: same row, contributions added
        Fm = xp.transpose(xp.conj(Cm), (0, 3, 1, 2)) * s
        F[:, :, Ma - Ka + 1:, :Kb] += Fm[:, :, :0:-1, :]
        F[:, :, 0, :Kb] += Fm[:, :, 0, :]
        # kb = 0 column: the (ka, -0) contributions land on (ka, 0) too (Hermitian completion)
        c0 = F[..., 0]
        F[..., 0] = c0 + xp.conj(c0[..., self._neg(Ma)])
        return self.fft.irfft2(F, s=(Ma, Mb), axes=(-2, -1))    # (n, My, Ma, Mb)

    def _neg(self, M: int):
        key = ("neg", M)
        v = self.__dict__.setdefault("_cache", {}).get(key)
        if v is None:
            v = self._cache[key] = self.xp.asarray((-np.arange(M)) % M)
        return v

    def forward(self, g):
        """Coefficients of the grid values ``g`` (3-D batched transform, dealiasing projection)."""
        if self._gpu is not None:
            return self._gpu.forward(g)
        xp = self.xp
        cy, (ia, ib) = self.cy, self.fa
        Mc, Ma, Mb = self.M[cy], self.M[ia], self.M[ib]
        Na, Nb, Nc = self.N[ia], self.N[ib], self.N[cy]
        n = g.shape[0]
        Ka, Kb = Na // 2, Nb // 2
        F = self.fft.rfft2(g, axes=(-2, -1)) / (Ma * Mb)        # (n, My, Ma, Mb/2+1)
        Fp = F[:, :, :Ka, :Kb]
        Fm = xp.empty_like(Fp)
        Fm[:, :, 1:, :] = F[:, :, Ma - Ka + 1:, :Kb][:, :, ::-1, :]
        Fm[:, :, 0, :] = F[:, :, 0, :Kb]
        Cp, Cm = 4 * Fp, 4 * xp.conj(Fm)
        # edges: the zero mode of a Fourier axis carries both contributions (halve each)
        w = xp.ones((Ka, Kb))
        w[0, :] *= 0.5
        w[:, 0] *= 0.5
        Cp, Cm = Cp * w, Cm * w
        f = xp.empty((n, Na, Nb, Mc))
        # (n, My, Ka, Kb) -> (n, Ka, Kb, My), pairs back into the interleaved layout
        t = lambda z: xp.transpose(z, (0, 2, 3, 1))  # noqa: E731
        f[:, 0::2, 0::2] = t((Cp + Cm).real / 2)
        f[:, 1::2, 1::2] = t((Cm - Cp).real / 2)
        f[:, 0::2, 1::2] = t((Cp - Cm).imag / 2)
        f[:, 1::2, 0::2] = t((Cp + Cm).imag / 2)
        # unused slots (-sin of the zero modes) are exactly zero
        f[:, 1, :] = 0.0
        f[:, :, 1] = 0.0
        Fc = self.fft.rfft(f, n=2 * Mc, axis=-1)[..., :Nc]
        cc = (self.tw_f * Fc).real * (2.0 / Mc)
        cc[..., 0] *= 0.5
        return xp.ascontiguousarray(xp.transpose(cc, self.inv_c))


_FFC_SRC = r"""
#include <cupy/complex.cuh>
typedef complex<double> cd;
// Coefficient layout: (n, N0, N1, N2) C order, strides s0, s1, s2 per element of the axes
// (a = first Fourier axis, b = second, c = Chebyshev). Work layouts put the Chebyshev axis last.

// A: backward Chebyshev input for a length-M DCT-III (Makhoul): with d_k = (-1)^k c_k (0 for
// k >= Nc), H_0 = d_0 and H_k = 1/2 e^{i pi k / 2M} (d_k - i d_{M-k}), k = 1..M/2; then
// v = M irfft(H, M) and the grid values are x_{2m} = v_m, x_{2m+1} = v_{M-1-m}.
extern "C" __global__ void ffc_bw_cheb(const double* c, const cd* e, cd* out, long long n,
        int Na, int Nb, int Nc, int M, long long sq, long long sa, long long sb, long long sc) {
    const int H = M / 2 + 1;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long tot = n * Na * Nb * (long long)H;
    if (i >= tot) return;
    long long r = i;
    int k = r % H; r /= H;
    int ib = r % Nb; r /= Nb;
    int ia = r % Na; long long q = r / Na;
    const double* base = c + q * sq + ia * sa + ib * sb;
    double dk = (k < Nc) ? ((k & 1) ? -base[(long long)k * sc] : base[(long long)k * sc]) : 0.0;
    if (k == 0) { out[i] = cd(dk, 0.0); return; }
    int km = M - k;
    double dm = (km < Nc) ? ((km & 1) ? -base[(long long)km * sc] : base[(long long)km * sc]) : 0.0;
    out[i] = 0.5 * e[k] * cd(dk, -dm);
}

// D: Chebyshev forward output -> coefficient layout. Fc: (n, Na, Nb, Mc1 = M/2+1) complex (rfft of
// the Makhoul-ordered values); c[q, ia, ib, k] = Re(tw[k] Y_k) * 2 / M (halved at k = 0),
// tw[k] = (-1)^k e^{-i pi k / 2M}, strided store.
extern "C" __global__ void ffc_fw_cheb(const cd* Fc, const cd* tw, double* c, long long n, int Na,
        int Nb, int Nc, int Mc1, int M, double sc2, long long sq, long long sa, long long sb,
        long long sk) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long tot = n * Na * (long long)Nb * Nc;
    if (t >= tot) return;
    long long r = t;
    int k = r % Nc; r /= Nc;
    int ib = r % Nb; r /= Nb;
    int ia = r % Na; long long q = r / Na;
    const cd* col = Fc + ((q * Na + ia) * (long long)Nb + ib) * Mc1;
    cd Y = (k < Mc1) ? col[k] : conj(col[M - k]);                 // real input: Y_{M-k} = conj Y_k
    double v = (tw[k] * Y).real() * sc2;
    if (k == 0) v *= 0.5;
    c[q * sq + ia * sa + ib * sb + (long long)k * sk] = v;
}

// B: RealFourier pairs -> zero-padded half spectrum, F(ka, kb) = s Cp, F(-ka, kb) = s conj(Cm);
// f: (n, Na, Nb, L = M) real in Makhoul order, scaled by fs. Tiled: coalesced reads (along the
// Chebyshev axis) and writes (along kb) through a 32 x 32 shared-memory tile.
// grid (ceil(Kb/32), ceil(M/32), n*Ka), block (32, 8); F must be zeroed beforehand; the
// kb = 0 Hermitian completion is a separate pass (ffc_bw_herm).
extern "C" __global__ void ffc_bw_pack_t(const double* f, cd* F, int Na, int Nb, int L, int Mc,
        int Ma, int Mb2, int Ka, int Kb, double fs, double s) {
    __shared__ double t[4][32][33];
    const int mb0 = blockIdx.x * 32, y0 = blockIdx.y * 32;
    const long long q = blockIdx.z / Ka;
    const int ma = blockIdx.z % Ka;
    const double* base = f + q * (long long)Na * Nb * L;
    // load: x -> y (fast, Makhoul order runs of 16), y -> mb
    const int yl = threadIdx.x;
    const int y = y0 + yl;
    if (y < Mc) {
        const int yy = (y & 1) ? L - 1 - (y >> 1) : (y >> 1);
        for (int ml = threadIdx.y; ml < 32; ml += blockDim.y) {
            const int mb = mb0 + ml;
            if (mb >= Kb) continue;
            for (int c = 0; c < 4; ++c) {
                const int p = c >> 1, pp = c & 1;
                t[c][ml][yl] = base[((long long)(2 * ma + p) * Nb + 2 * mb + pp) * L + yy] * fs;
            }
        }
    }
    __syncthreads();
    // store: x -> mb (fast)
    const int ml = threadIdx.x;
    const int mb = mb0 + ml;
    if (mb >= Kb) return;
    for (int yl2 = threadIdx.y; yl2 < 32; yl2 += blockDim.y) {
        const int y2 = y0 + yl2;
        if (y2 >= Mc) break;
        const double a00 = t[0][ml][yl2], a01 = t[1][ml][yl2], a10 = t[2][ml][yl2], a11 = t[3][ml][yl2];
        const cd cp_(a00 - a11, a01 + a10), cmc(a00 + a11, -(a10 - a01));   // Cp, conj(Cm)
        cd* row = F + ((q * Mc + y2) * (long long)Ma) * Mb2;
        if (ma == 0) row[mb] = (cp_ + cmc) * s;
        else {
            row[(long long)ma * Mb2 + mb] = cp_ * s;
            row[(long long)(Ma - ma) * Mb2 + mb] = cmc * s;
        }
    }
}

// kb = 0 column: F(ka, 0) += conj(F(-ka, 0)) for all ka (pairs handled by one thread).
extern "C" __global__ void ffc_bw_herm(cd* F, long long nplanes, int Ma, int Mb2) {
    const int half = Ma / 2 + 1;
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nplanes * half) return;
    const long long pl = t / half;
    const int i = t % half, j = (Ma - i) % Ma;
    cd* col = F + pl * (long long)Ma * Mb2;
    const cd a = col[(long long)i * Mb2], b = col[(long long)j * Mb2];
    if (i == j) col[(long long)i * Mb2] = a + conj(a);
    else { col[(long long)i * Mb2] = a + conj(b); col[(long long)j * Mb2] = b + conj(a); }
}

// C: half spectrum -> pairs (Makhoul order along the Chebyshev axis, v_m = x_{2m},
// v_{M-1-m} = x_{2m+1}), same tiling. grid (ceil(Kb/32), ceil(M/32), n*Ka), block (32, 8). out: (n, Na, Nb, L = Mc) real.
extern "C" __global__ void ffc_fw_unpack_t(const cd* F, double* out, int Na, int Nb, int Mc,
        int Ma, int Mb2, int Ka, int Kb, double inv) {
    __shared__ double t[4][32][33];
    const int mb0 = blockIdx.x * 32, p0 = blockIdx.y * 32;
    const long long q = blockIdx.z / Ka;
    const int ma = blockIdx.z % Ka;
    const int mneg = (Ma - ma) % Ma;
    const double wa = (ma == 0 ? 0.5 : 1.0);
    // load: x -> mb (fast), y -> p
    {
        const int ml = threadIdx.x, mb = mb0 + ml;
        if (mb < Kb) {
            const double w = 4.0 * wa * (mb == 0 ? 0.5 : 1.0) * inv;
            for (int pl = threadIdx.y; pl < 32; pl += blockDim.y) {
                const int p = p0 + pl;
                if (p >= Mc) break;
                const int y = (p < (Mc + 1) / 2) ? 2 * p : 2 * (Mc - 1 - p) + 1;
                const cd* row = F + ((q * Mc + y) * (long long)Ma) * Mb2;
                const cd Cp = row[(long long)ma * Mb2 + mb] * w;
                const cd Cm = conj(row[(long long)mneg * Mb2 + mb]) * w;
                t[0][pl][ml] = (Cp + Cm).real() / 2;      // a00
                t[1][pl][ml] = (Cp - Cm).imag() / 2;      // a01
                t[2][pl][ml] = (Cp + Cm).imag() / 2;      // a10
                t[3][pl][ml] = (Cm - Cp).real() / 2;      // a11
            }
        }
    }
    __syncthreads();
    // store: x -> p (fast)
    const int pl = threadIdx.x, p = p0 + pl;
    if (p >= Mc) return;
    double* base = out + q * (long long)Na * Nb * Mc;
    for (int ml = threadIdx.y; ml < 32; ml += blockDim.y) {
        const int mb = mb0 + ml;
        if (mb >= Kb) break;
        for (int c = 0; c < 4; ++c) {
            const int pa = c >> 1, pb = c & 1;
            const int ia = 2 * ma + pa, ib = 2 * mb + pb;
            base[((long long)ia * Nb + ib) * Mc + p] = (ia == 1 || ib == 1) ? 0.0 : t[c][pl][ml];
        }
    }
}
"""
_ffc_mod = None


class _FFCKernels:
    """GPU path of FourierFourierChebyshev: each packing step is one generated pass (no
    temporaries): Chebyshev pre-twiddle and transpose, pair packing with zero padding and
    Hermitian completion, unpacking, post-twiddle and transpose back."""

    def __init__(self, T: FourierFourierChebyshev) -> None:
        global _ffc_mod
        cp = T.xp
        if _ffc_mod is None:
            _ffc_mod = cp.RawModule(code=_FFC_SRC, options=("-std=c++17",))
        self.k = {n: _ffc_mod.get_function(n) for n in
                  ("ffc_bw_cheb", "ffc_fw_cheb", "ffc_bw_pack_t", "ffc_bw_herm", "ffc_fw_unpack_t")}
        self.T = T
        import cupyx.scipy.fft as cfft
        self.fft = cfft
        M = T.M[T.cy]
        self.e_b = cp.asarray(np.exp(1j * np.pi * np.arange(M // 2 + 1) / (2 * M)))
        self.tw_f = cp.ascontiguousarray(T.tw_f.astype(complex))

    @staticmethod
    def _launch(k, tot, args):
        k(((int(tot) + 255) // 256,), (256,), args)

    def _strides(self, n):
        """Element strides (q, a, b, c) of the coefficient layout (n, N0, N1, N2)."""
        T = self.T
        N = T.N
        st = [N[1] * N[2], N[2], 1]
        return (N[0] * N[1] * N[2], st[T.fa[0]], st[T.fa[1]], st[T.cy])

    def backward(self, c):
        cp, T = self.T.xp, self.T
        c = cp.ascontiguousarray(c)
        n = c.shape[0]
        cy, (ia, ib) = T.cy, T.fa
        Mc, Ma, Mb = T.M[cy], T.M[ia], T.M[ib]
        Na, Nb, Nc = T.N[ia], T.N[ib], T.N[cy]
        sq, sa, sb, sc = self._strides(n)
        # Chebyshev: length-M DCT-III by Makhoul's reordering (half the FFT length of the
        # zero-padded 2M transform of the NumPy path; same numbers to rounding)
        a = cp.empty((n, Na, Nb, Mc // 2 + 1), dtype=complex)
        self._launch(self.k["ffc_bw_cheb"], a.size,
                     (c, self.e_b, a, np.int64(n), np.int32(Na), np.int32(Nb), np.int32(Nc),
                      np.int32(Mc), np.int64(sq), np.int64(sa), np.int64(sb), np.int64(sc)))
        # C2R transforms overwrite their input in cuFFT: the inputs here are private temporaries,
        # so overwrite_x=True spares CuPy's defensive copy (-17 to -25 % on these calls)
        f = self.fft.irfft(a, n=Mc, axis=-1, overwrite_x=True)   # (n, Na, Nb, Mc), Makhoul order
        Mb2 = Mb // 2 + 1
        Ka, Kb = Na // 2, Nb // 2
        F = cp.zeros((n, Mc, Ma, Mb2), dtype=complex)
        # tiled pack (coalesced through shared memory), then the kb = 0 Hermitian completion
        self.k["ffc_bw_pack_t"](((Kb + 31) // 32, (Mc + 31) // 32, n * Ka), (32, 8),
                                (f, F, np.int32(Na), np.int32(Nb), np.int32(Mc), np.int32(Mc),
                                 np.int32(Ma), np.int32(Mb2), np.int32(Ka), np.int32(Kb),
                                 np.float64(Mc), np.float64(0.25 * Ma * Mb)))
        half = Ma // 2 + 1
        self._launch(self.k["ffc_bw_herm"], n * Mc * half,
                     (F, np.int64(n * Mc), np.int32(Ma), np.int32(Mb2)))
        return self.fft.irfft2(F, s=(Ma, Mb), axes=(-2, -1), overwrite_x=True)

    def forward(self, g):
        cp, T = self.T.xp, self.T
        g = cp.ascontiguousarray(g)
        n = g.shape[0]
        cy, (ia, ib) = T.cy, T.fa
        Mc, Ma, Mb = T.M[cy], T.M[ia], T.M[ib]
        Na, Nb, Nc = T.N[ia], T.N[ib], T.N[cy]
        F = cp.fft.rfft2(g, axes=(-2, -1))                       # (n, Mc, Ma, Mb/2+1)
        Mb2 = F.shape[-1]
        f = cp.empty((n, Na, Nb, Mc))
        Ka, Kb = Na // 2, Nb // 2
        self.k["ffc_fw_unpack_t"](((Kb + 31) // 32, (Mc + 31) // 32, n * Ka), (32, 8),
                                  (F, f, np.int32(Na), np.int32(Nb), np.int32(Mc), np.int32(Ma),
                                   np.int32(Mb2), np.int32(Ka), np.int32(Kb),
                                   np.float64(1.0 / (Ma * Mb))))
        Fc = cp.fft.rfft(f, axis=-1)                             # (n, Na, Nb, Mc/2 + 1)
        out = cp.empty((n,) + tuple(T.N))
        sq, sa, sb, sk = self._strides(n)
        self._launch(self.k["ffc_fw_cheb"], n * Na * Nb * Nc,
                     (Fc, self.tw_f, out, np.int64(n), np.int32(Na), np.int32(Nb), np.int32(Nc),
                      np.int32(Fc.shape[-1]), np.int32(Mc), np.float64(2.0 / Mc), np.int64(sq), np.int64(sa),
                      np.int64(sb), np.int64(sk)))
        return out
