"""Fast transforms of real fields for the planned spherical right-hand side (items 4, 7 and 8 of
:mod:`conatus.timestep.sphere_fused`): real GEMMs on interleaved complex data (direct cuBLAS
during a CUDA-graph capture), angular syntheses and projections restricted to the existing
degrees per m-chunk (``ang_chunks``), and :class:`_FastSpace`, the spherical space with these
transforms for real fields. Split out of :mod:`conatus.timestep.sphere_fused`, which re-exports
every name."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..discretize._xp import array_module
from ..discretize.sphere_eval import _R2, _flip, comps
from ..discretize.sphere_ops import spin_of
from ..discretize.spin import canon as _canon


# ----------------------------------------------------------------------------------------------
# real-matrix products on complex data (interleaved re/im, real GEMM)
# ----------------------------------------------------------------------------------------------
def _direct_cublas() -> bool:
    from ..discretize._cublas import available
    return available()


def _gemm(A, B, xp):
    """A @ B for float64 arrays, (P, Q) @ (Q, R) or batched (n, P, Q) @ (n, Q, R) (or (P, Q)
    shared by the batch). GPU: the direct cuBLAS call of ``discretize._cublas`` (usable inside
    a CUDA graph capture, where CuPy refuses cuBLAS) or, if that library cannot be loaded,
    ``cupy.matmul`` (declared by ``explain()``; no graph replay then); CPU: NumPy."""
    from ..discretize._cublas import available, gemm_rm
    if xp is np or not available():
        return xp.matmul(A, B)
    A, B = xp.ascontiguousarray(A, dtype=np.float64), xp.ascontiguousarray(B, dtype=np.float64)
    P, Q = A.shape[-2:]
    R = B.shape[-1]
    n = B.shape[0] if B.ndim == 3 else 1
    if (A.ndim == 3 and A.shape[0] != n) or B.shape[-2] != Q or A.ndim > 3 or B.ndim > 3:
        raise ValueError(f"_gemm: shapes {A.shape} @ {B.shape}")
    C = xp.empty((n, P, R) if B.ndim == 3 else (P, R), dtype=np.float64)
    gemm_rm(xp, P, R, Q, A.data.ptr, Q, P * Q if A.ndim == 3 else 0, B.data.ptr, R, Q * R,
            C.data.ptr, R, P * R, n)
    return C


def _left_real(B, y, xp):
    """B @ y for a real (batched) matrix B (..., P, Q) and complex y (..., Q, R): one real GEMM
    on the interleaved data (..., Q, 2R) (y viewed as float64)."""
    y = xp.ascontiguousarray(y)
    z = _gemm(B, y.view(np.float64), xp)
    return z.view(np.complex128)


def _right_real(y, M, xp):
    """y @ M for complex y (..., Q) and a real matrix M (Q, P): one real GEMM on (2 K, Q) rows
    (real and imaginary parts as separate rows)."""
    sh = y.shape
    K, Q = int(np.prod(sh[:-1])), sh[-1]
    yr = xp.ascontiguousarray(xp.ascontiguousarray(y).view(np.float64).reshape(K, Q, 2).transpose(0, 2, 1))
    z = _gemm(yr.reshape(2 * K, Q), M, xp)                       # (2K, P)
    z = xp.ascontiguousarray(z.reshape(K, 2, M.shape[1]).transpose(0, 2, 1))
    return z.view(np.complex128).reshape(sh[:-1] + (M.shape[1],))


def default_ang_chunks(space, xp) -> int:
    """Number of m-chunks of the angular GEMMs (``ang_chunks``); the rule and its reason are
    :func:`ang_chunks_rule` (declared in explain())."""
    return ang_chunks_rule(space, xp)[0]


# at most 4 chunks: 8 was never faster on A100 (both B2 grids) nor in the L40S micro-benchmark;
# on L40S at 256x127x64, 8 chunks beat 1 by 5 %, 4 unmeasured
# there: cap kept at 4, to be confirmed by a 1/2/4/8 measurement.
_MAX_CHUNKS = 4


def ang_chunks_rule(space, xp) -> tuple:
    """(number of m-chunks, reason) for the angular GEMMs. Chunks drop the exactly-zero terms
    (about 40 % of the flops at nm >= 64 with 4 chunks) for more, smaller GEMMs: worth it where
    FP64 GEMMs bound F -- the CPU and GPUs with reduced FP64 throughput (compute capability
    other than 6.0, 7.0, 8.0, 9.0, 10.0: L40S) -- and, on full-FP64 GPUs (A100, H100), only for
    large GEMMs (>= 1 GFLOP per call; at 128x63x48 on A100 one chunk is faster). Measured on
    A100; the reduced-FP64 branch (L40S, graph step): 1/2/4/8 chunks within 7 % at 128x63x48 (26.0/25.5/26.5/27.3 ms);
    at 256x127x64 4 chunks are the fastest, 9 % faster than 1 and 4 % faster than 8
    on one L40S, hence the cap of 4. Same numbers either way up to
    rounding (the dropped terms are exact zeros; the summation blocking of the GEMM changes)."""
    nm = len(space.ms)
    if nm < 16 or not space.real:
        return 1, f"nm = {nm} < 16 or complex fields: one GEMM"
    many = min(_MAX_CHUNKS, nm // 16)
    if xp is None or xp is np:
        return many, f"CPU: min({_MAX_CHUNKS}, nm // 16) with nm = {nm}"
    try:
        cc = int(xp.cuda.Device().compute_capability)
    except Exception:                                  # noqa: BLE001 -- no device query: default
        return many, f"GPU of unknown compute capability: min({_MAX_CHUNKS}, nm // 16)"
    if cc not in (60, 70, 80, 90, 100):              # reduced FP64 throughput
        return many, (f"GPU sm_{cc} with reduced FP64 throughput: min({_MAX_CHUNKS}, nm // 16) "
                      "(L40S: 4 chunks fastest at 256x127x64, 9 % faster than 1)")
    Mt = space.sizes(True)[1]
    Mr = space.sizes(True)[2] if not space.s2 else 1
    flops = 2.0 * nm * Mt * (space.L + 1) * 2 * Mr
    if flops >= 1e9:
        return many, f"full-FP64 GPU sm_{cc}, {flops / 1e9:.2f} GFLOP per angular GEMM >= 1 (measured on A100)"
    return 1, f"full-FP64 GPU sm_{cc}, {flops / 1e9:.2f} GFLOP per angular GEMM < 1: one GEMM (measured on A100)"


def ang_chunks(ms: np.ndarray, s: int, L: int, nchunk: int) -> list:
    """Chunks (m0, m1, l0) of the stored wavenumbers for the angular GEMMs restricted to the
    degrees that exist: Phi_l^{m,s} = 0 exactly for l < max(|m|, |s|) (sphere_ops.values: zero
    columns), so chunk [m0, m1) of increasing |m| only needs l >= l0 = max(|m_m0|, |s|).
    Same products, the exactly-zero terms dropped (about half the flops for many chunks)."""
    nm = len(ms)
    if nchunk <= 1 or nm < 2 or np.any(np.diff(np.abs(ms)) < 0):
        return [(0, nm, 0)]
    edges = sorted(set(np.linspace(0, nm, min(nchunk, nm) + 1).astype(int).tolist()))
    out = []
    for m0, m1 in zip(edges[:-1], edges[1:]):
        out.append((m0, m1, min(max(int(abs(ms[m0])), abs(s)), L + 1)))
    return out


def _ang_real(which: str, A, y, chunks: list, xp):
    """Angular transform with the real (nm, P, Q) stack A on complex y (nm, Q, R), as
    ``_left_real`` but chunk by chunk over m on the existing degrees (``ang_chunks``):
    'bwd' (synthesis, Q = L+1): out[m] = A[m][:, l0:] @ y[m][l0:]; 'fwd' (projection, P = L+1):
    out[m][l0:] = A[m][l0:] @ y[m], rows l < l0 exactly zero (as the full product gives).
    GPU: cuBLAS strided-batched GEMMs on the views (no copy); CPU (or no direct cuBLAS):
    matmul per chunk."""
    from ..discretize._cublas import available, gemm_rm
    y = xp.ascontiguousarray(y)
    nm, Q, R = y.shape
    P = A.shape[1]
    yf = y.view(np.float64)                                   # (nm, Q, 2R)
    W = 2 * R
    if len(chunks) == 1:
        return _gemm(A, yf, xp).view(np.complex128)
    out = (xp.empty if which == "bwd" else xp.zeros)((nm, P, W), dtype=np.float64)
    if xp is np or not available():
        for m0, m1, l0 in chunks:
            if which == "bwd":
                out[m0:m1] = xp.matmul(A[m0:m1, :, l0:], yf[m0:m1, l0:, :])
            else:
                out[m0:m1, l0:, :] = xp.matmul(A[m0:m1, l0:, :], yf[m0:m1])
        return out.view(np.complex128)
    A = xp.ascontiguousarray(A, dtype=np.float64)
    a0, y0, c0 = A.data.ptr, yf.data.ptr, out.data.ptr
    for m0, m1, l0 in chunks:
        k = (Q if which == "fwd" else Q - l0)
        rows = P - l0 if which == "fwd" else P
        if k <= 0 or rows <= 0:
            if which == "bwd":
                out[m0:m1] = 0
            continue
        # row-major C' (rows x W) = A' (rows x k) @ Y' (k x W) on views of the full arrays
        if which == "bwd":
            pa, pc = a0 + 8 * (m0 * P * Q + l0), c0 + 8 * (m0 * P * W)
            py = y0 + 8 * (m0 * Q * W + l0 * W)
        else:
            pa, pc = a0 + 8 * (m0 * P * Q + l0 * Q), c0 + 8 * (m0 * P * W + l0 * W)
            py = y0 + 8 * (m0 * Q * W)
        gemm_rm(xp, rows, W, k, pa, Q, P * Q, py, W, Q * W, pc, W, P * W, m1 - m0)
    return out.view(np.complex128)


def _dot_fast(a, ra: int, b, rb: int, G):
    """sphere_eval._dot (last index of a with the first of b, spin metric G) with the same
    products and the same order of the sums, without the multiplications by G = 1 and with the
    sums accumulated in place (same values; signed zeros aside)."""
    xp = array_module(a, b)
    A = xp.moveaxis(a, ra - 1, 0)
    ar, br = ra - 1, rb - 1
    out = None
    for s in range(G.shape[0]):
        for t in range(G.shape[1]):
            if not G[s, t]:
                continue
            x, y = A[s], b[t]
            x = x.reshape(x.shape[:ar] + (1,) * br + x.shape[ar:])
            y = y.reshape((1,) * ar + y.shape)
            term = x * y if G[s, t] == 1 else G[s, t] * (x * y)
            if out is None:
                out = term
            else:
                out += term
    return out


_SPEC_SRC = {
    # spin pair: spectrum of comp c (phi last) from its angular synthesis V and the partner's W,
    # both (nm, Mt, Mr) contiguous: m = 0 symmetrized, -m by conjugation, the rest zero
    "pair_in": ("raw complex128 V, raw complex128 W, int64 nm, int64 Mp, int64 TR",
                "complex128 S",
                "const long long k = i % Mp, q = i / Mp;"
                "if (k == 0) S = 0.5 * (V[q] + conj(W[q]));"
                "else if (k < nm) S = V[k * TR + q];"
                "else if (k > Mp - nm) S = conj(W[(Mp - k) * TR + q]);"
                "else S = complex<double>(0.0, 0.0);"),
    # radial / scalar component: half spectrum for the real inverse FFT
    "real_in": ("raw complex128 V, int64 nm, int64 Mh, int64 TR", "complex128 H",
                "const long long k = i % Mh, q = i / Mh;"
                "if (k == 0) H = complex<double>(V[q].real(), 0.0);"
                "else if (k < nm) H = V[k * TR + q];"
                "else H = complex<double>(0.0, 0.0);"),
    # forward: stored modes of comp c and of its partner (conjugates of the modes -m), (nm, Mt, Mr)
    "pair_out": ("raw complex128 F, int64 Mp, int64 TR", "complex128 A, complex128 B",
                 "const long long m = i / TR, q = i % TR;"
                 "A = F[q * Mp + m];"
                 "B = conj(F[q * Mp + (m == 0 ? 0 : Mp - m)]);"),
    "real_out": ("raw complex128 F, int64 Mh, int64 TR", "complex128 A",
                 "const long long m = i / TR, q = i % TR;"
                 "A = F[q * Mh + m];"),
}
_SPEC_K: dict = {}


def _spec_kernel(xp, name: str):
    """Generated elementwise kernel of the spectrum assembly / extraction (GPU path of
    ``mixed_to_parts`` / ``parts_to_mixed``; the array code is its CPU oracle)."""
    k = _SPEC_K.get(name)
    if k is None:
        ins, outs, body = _SPEC_SRC[name]
        k = _SPEC_K[name] = xp.ElementwiseKernel(ins, outs, body, "cn_sphere_spec_" + name)
    return k


class _FastSpace:
    """SphereSpace with faster transforms for real fields in the full space (no fold, spin
    components on the grid). Everything else is delegated to the space (the reference)."""

    def __init__(self, S, ang_chunks: int = 1) -> None:
        self._S = S
        self._cache: dict = {}
        self.nchunk = int(ang_chunks)                # m-chunks of the angular GEMMs (ang_chunks)

    def _ang(self, which: str, ms, s: int, Mt: int, y, xp):
        """Angular synthesis ('bwd') / projection ('fwd') of spin s on complex y, restricted to
        the existing degrees l >= max(|m|, |s|) per m-chunk (exactly-zero terms dropped)."""
        key = ("chunks", tuple(ms), s)
        ch = self._cache.get(key)
        if ch is None:
            ch = self._cache[key] = ang_chunks(np.asarray(ms), s, self._S.L, self.nchunk)
        return _ang_real(which, self._S._ang_stack(which, ms, s, Mt, xp), y, ch, xp)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._S, name)

    def _fast(self, dealias, physical, fold) -> bool:
        return self._S.real and dealias is True and not physical and fold is None

    # -------------------------------------------------------------- radial (shell)
    def _rad(self, whats: tuple, Mr: int, xp):
        key = ("rad", whats, Mr, xp is np)
        if key not in self._cache:
            M = np.concatenate([self._S._shell_mat("bwd", w, Mr) for w in whats], axis=1)
            self._cache[key] = xp.asarray(np.ascontiguousarray(M))
        return self._cache[key]

    def coeff_to_mixed_many(self, c, dom, rank: int, whats: tuple) -> list:
        """coeff_to_mixed for several ``what`` at once (one real GEMM, shell, dealiased grid)."""
        S = self._S
        xp = array_module(c)
        if not (S.shell and S.has(dom)[2]):
            return [S.coeff_to_mixed(c, dom, rank, True, w) for w in whats]
        Mr = S.sizes(True)[2]
        z = _right_real(xp.asarray(c, dtype=complex), self._rad(whats, Mr, xp), xp)
        return [z[..., i * Mr:(i + 1) * Mr] for i in range(len(whats))]

    def coeff_to_mixed(self, c, dom, rank: int, dealias, what: str = "val", fold=None):
        S = self._S
        if fold is not None or dealias is not True or not (S.shell and S.has(dom)[2]):
            return S.coeff_to_mixed(c, dom, rank, dealias, what, fold=fold)
        return self.coeff_to_mixed_many(c, dom, rank, (what,))[0]

    def mixed_to_coeff(self, x, dom, rank: int, dealias, fold=None):
        S = self._S
        if fold is not None or dealias is not True or not (S.shell and S.has(dom)[2]):
            return S.mixed_to_coeff(x, dom, rank, dealias, fold=fold)
        xp = array_module(x)
        Mr = x.shape[-1]
        key = ("fwd", Mr, xp is np)
        if key not in self._cache:
            self._cache[key] = xp.asarray(np.ascontiguousarray(S._shell_mat("fwd", "val", Mr)))
        return _right_real(xp.asarray(x, dtype=complex), self._cache[key], xp)

    # -------------------------------------------------------------- angular + azimuthal
    def mixed_to_grid(self, x, dom, rank: int, dealias, physical: bool = True, fold=None):
        S = self._S
        if not self._fast(dealias, physical, fold):
            return S.mixed_to_grid(x, dom, rank, dealias, physical, fold)
        hp, ht, hr = S.has(dom)
        Mp, Mt = S.sizes(True)[:2]
        xp = array_module(x)
        x = xp.asarray(x)
        ms = S._m_of(dom)
        spec = {}
        for c in comps(rank, S.nc):
            y = x[c] if c else x
            y = y if not S.s2 else y[..., None]                     # (nm, L+1|1, Mr|1)
            if ht:
                spec[c] = self._ang("bwd", ms, spin_of(c), Mt, y, xp)
            else:                          # no theta: the single entry is the l = 0 coefficient
                spec[c] = y * _R2 if spin_of(c) == 0 else 0 * y
        out = xp.empty(S.grid_shape(dom, rank, True), dtype=complex)
        done: set = set()
        for c in comps(rank, S.nc):
            if c in done:
                continue
            V, f = spec[c], _flip(c)
            if not hp:                     # no phi: m = 0 only, the values themselves
                g = V
            elif f == c:                   # radial / scalar component: real on the grid
                H = xp.zeros((Mp // 2 + 1,) + V.shape[1:], dtype=complex)
                H[S.ms] = V
                H[0] = H[0].real
                g = xp.fft.irfft(H, n=Mp, axis=0, norm="forward")
            else:                          # spin pair (s, -s): one FFT, the partner by conjugation
                Sx = xp.zeros((Mp,) + V.shape[1:], dtype=complex)
                W = spec[f]
                Sx[S.ms[1:]] = V[1:]
                Sx[0] = 0.5 * (V[0] + xp.conj(W[0]))         # m = 0 of a real field: symmetrized
                Sx[(-S.ms[1:]) % Mp] = xp.conj(W[1:])
                g = xp.fft.ifft(Sx, axis=0, norm="forward")
                xp.conjugate(g[..., 0] if S.s2 else g, out=out[f])
                done.add(f)
            g = g[..., 0] if S.s2 else g
            if c:
                out[c] = g
            else:
                out[...] = g
        return out

    def grid_to_mixed(self, g, dom, rank: int, dealias, physical: bool = True, fold=None):
        S = self._S
        if not self._fast(dealias, physical, fold):
            return S.grid_to_mixed(g, dom, rank, dealias, physical, fold)
        hp, ht, hr = S.has(dom)
        Mp, Mt = S.sizes(True)[:2]
        xp = array_module(g)
        g = xp.asarray(g)
        ms = S._m_of(dom)
        Mr = (S.sizes(True)[2] if hr else 1) if not S.s2 else 1
        out = xp.empty((S.nc,) * rank + (len(ms), S.L + 1 if ht else 1, Mr), dtype=complex)
        az: dict = {}
        for c in comps(rank, S.nc):
            if c in az:
                continue
            y = g[c] if c else g
            y = y if not S.s2 else y[..., None]
            f = _flip(c)
            if not hp:
                az[c] = xp.asarray(y, dtype=complex)
            elif f == c:                   # real component: half-size real FFT
                az[c] = xp.fft.rfft(y.real, axis=0, norm="forward")[S.ms]
            else:                          # spin pair: one FFT, the partner's modes by conjugation
                Fy = xp.fft.fft(y, axis=0, norm="forward")
                az[c] = Fy[S.ms]
                az[f] = xp.conj(Fy[(-S.ms) % Mp])
        for c in comps(rank, S.nc):
            s = spin_of(c)
            if ht:
                y = self._ang("fwd", ms, s, Mt, az[c], xp)
            else:
                y = az[c] / _R2 if s == 0 else 0 * az[c]
            if c:
                out[c] = y
            else:
                out[...] = y
        return out[..., 0] if S.s2 else out

    def mixed_to_parts(self, x, dom, rank: int) -> dict:
        """Mixed space -> canonical grid components of a real tensor (``_PG`` parts, layout
        ``_plast``: phi last, contiguous): one inverse FFT per spin pair, real inverse FFTs for
        radial / scalar components, no conjugate copies."""
        S = self._S
        hp, ht, hr = S.has(dom)
        Mp, Mt = S.sizes(True)[:2]
        xp = array_module(x)
        x = xp.asarray(x)
        ms = S._m_of(dom)
        nm = len(S.ms)
        out = {}
        for c in comps(rank, S.nc):
            if _canon(c) != c:
                continue
            f = _flip(c)
            spec = []
            for cc in ((c,) if f == c else (c, f)):
                y = x[cc] if cc else x
                y = y if not S.s2 else y[..., None]                  # (nm, L+1|1, Mr|1)
                if ht:
                    y = self._ang("bwd", ms, spin_of(cc), Mt, y, xp)
                else:
                    y = y * _R2 if spin_of(cc) == 0 else 0 * y
                spec.append(xp.moveaxis(y, 0, -1))                  # (Mt|1, Mr|1, nm)
            V = spec[0]
            if hp and xp is not np:          # GPU: assembly in one kernel from (nm, Mt|1, Mr|1)
                Vc = xp.ascontiguousarray(xp.moveaxis(V, -1, 0))
                TR = int(np.prod(Vc.shape[1:]))
                if f == c:
                    H = xp.empty(V.shape[:-1] + (Mp // 2 + 1,), dtype=complex)
                    _spec_kernel(xp, "real_in")(Vc, nm, Mp // 2 + 1, TR, H)
                    g = xp.fft.irfft(H, n=Mp, axis=-1, norm="forward")
                else:
                    Wc = xp.ascontiguousarray(xp.moveaxis(spec[1], -1, 0))
                    Sx = xp.empty(V.shape[:-1] + (Mp,), dtype=complex)
                    _spec_kernel(xp, "pair_in")(Vc, Wc, nm, Mp, TR, Sx)
                    g = xp.fft.ifft(Sx, axis=-1, norm="forward")
            elif not hp:                     # no phi: m = 0 only, the values themselves
                g = xp.ascontiguousarray(V.real) if f == c else xp.ascontiguousarray(V)
            elif f == c:                     # radial / scalar component: real on the grid
                H = xp.empty(V.shape[:-1] + (Mp // 2 + 1,), dtype=complex)
                H[..., :nm] = V                              # stored m = 0 .. nm - 1 (RealFourier)
                H[..., nm:] = 0
                H[..., 0] = H[..., 0].real
                g = xp.fft.irfft(H, n=Mp, axis=-1, norm="forward")
            else:                            # spin pair: one FFT; m = 0 symmetrized (real field)
                W = spec[1]
                Sx = xp.empty(V.shape[:-1] + (Mp,), dtype=complex)
                Sx[..., 1:nm] = V[..., 1:]
                Sx[..., 0] = 0.5 * (V[..., 0] + xp.conj(W[..., 0]))
                Sx[..., nm:Mp - nm + 1] = 0
                Sx[..., Mp - 1:Mp - nm:-1] = xp.conj(W[..., 1:])       # modes -1 .. -(nm - 1)
                g = xp.fft.ifft(Sx, axis=-1, norm="forward")
            out[c] = g[:, 0] if S.s2 else g
        return out

    def parts_to_mixed(self, parts: dict, dom, rank: int):
        """Canonical grid components (layout ``_plast``) -> mixed space: one forward FFT per
        spin pair (the partner's modes by conjugation), real FFTs for radial / scalar ones."""
        S = self._S
        hp, ht, hr = S.has(dom)
        Mp, Mt = S.sizes(True)[:2]
        ms = S._m_of(dom)
        Mr = (S.sizes(True)[2] if hr else 1) if not S.s2 else 1
        first = next(iter(parts.values()))
        xp = array_module(first)
        out = xp.empty((S.nc,) * rank + (len(ms), S.L + 1 if ht else 1, Mr), dtype=complex)
        az: dict = {}
        for c, y in parts.items():
            y = y[:, None, :] if S.s2 else y                         # (Mt|1, Mr|1, Mp|1)
            f = _flip(c)
            if not hp:
                az[c] = xp.moveaxis(xp.asarray(y, dtype=complex), -1, 0)
                if f != c:
                    az[f] = xp.conj(az[c])
            elif xp is not np:               # GPU: extraction in one kernel -> (nm, Mt|1, Mr|1)
                TR = int(np.prod(y.shape[:-1]))
                shp = (len(S.ms),) + y.shape[:-1]
                if f == c:
                    Fh = xp.fft.rfft(y, axis=-1, norm="forward")
                    az[c] = xp.empty(shp, dtype=complex)
                    _spec_kernel(xp, "real_out")(Fh, Mp // 2 + 1, TR, az[c])
                else:
                    Fy = xp.fft.fft(y, axis=-1, norm="forward")
                    az[c], az[f] = xp.empty(shp, dtype=complex), xp.empty(shp, dtype=complex)
                    _spec_kernel(xp, "pair_out")(Fy, Mp, TR, az[c], az[f])
            elif f == c:
                az[c] = xp.moveaxis(xp.fft.rfft(y, axis=-1, norm="forward")[..., S.ms], -1, 0)
            else:
                Fy = xp.fft.fft(y, axis=-1, norm="forward")
                az[c] = xp.moveaxis(Fy[..., S.ms], -1, 0)
                az[f] = xp.moveaxis(xp.conj(Fy[..., (-S.ms) % Mp]), -1, 0)
        for c in comps(rank, S.nc):
            s = spin_of(c)
            if ht:
                y = self._ang("fwd", ms, s, Mt, az[c], xp)
            else:
                y = az[c] / _R2 if s == 0 else 0 * az[c]
            if c:
                out[c] = y
            else:
                out[...] = y
        return out[..., 0] if S.s2 else out

    def backward(self, c, dom, rank: int, dealias, physical: bool = True, fold=None):
        return self.mixed_to_grid(self.coeff_to_mixed(c, dom, rank, dealias, fold=fold), dom, rank,
                                  dealias, physical, fold)

    def forward(self, g, dom, rank: int, dealias, physical: bool = True, fold=None):
        return self.mixed_to_coeff(self.grid_to_mixed(g, dom, rank, dealias, physical, fold), dom,
                                   rank, dealias, fold=fold)
