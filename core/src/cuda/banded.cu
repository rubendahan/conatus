// Batched bordered banded solves on the device.
// Same decomposition as the host solver (conatus::BorderedBanded, shared BorderedLayout):
//   P A = B + U V^T,  B banded (LAPACK band storage), U = first r unit vectors.
// One thread block (one warp) per mode:
//   factor : banded LU of B, LAPACK dgbtf2 algorithm (partial pivoting inside the band, or none when
//            the caller declares B diagonally dominant), then W = B^{-1} U and the r x r capacitance
//            S = I + V^T W, LU-factored with partial pivoting; cond_1(S) reported;
//   solve  : y = P b, y = B^{-1} y, z = S^{-1} V^T y, x = y - W z. The work vector lives in shared
//            memory when it fits.
// Column border (col_border; banded.hpp, "Column border"): B = diag(s) (+) A11 is
// block diagonal, the border is solved by its Schur complement S = A00 - A01 A11^{-1} A10 in the
// same storage (V^T = [diag(s), -A01], W = [-I; A11^{-1} A10]); the only kernel differences: the
// final x = y - W z starts from 0 on the first r entries, and a transposed solve corrects its top
// entries by y_k -= (A10^T y)_k / s_k (xcorr_T) before the border.
#include <cublas_v2.h>
#include <cuda_pipeline.h>

#include <cstring>
#include <type_traits>

#include <algorithm>
#include <chrono>

#include "common.cuh"
#include "conatus/cuda.hpp"

namespace conatus::cuda {

namespace {

constexpr int WARP = 32;
constexpr int MAXR = 64;                  // largest border rank handled on the device (3D channel: 24)
constexpr std::size_t SHARED_Y_MAX = 40 * 1024;

template <class T>
struct Band {
    T* a;
    int ldab, kv;
    __device__ T& operator()(int i, int c) const { return a[std::size_t(c) * ldab + kv + i - c]; }
};

__device__ inline double shfl_down(double v, int o) { return __shfl_down_sync(0xffffffffu, v, o); }
__device__ inline double warp_sum(double v) {
    for (int o = WARP / 2; o > 0; o >>= 1) v += shfl_down(v, o);
    return v;
}
__device__ inline ::cuda::std::complex<double> warp_sum(::cuda::std::complex<double> v) {
    return {warp_sum(v.real()), warp_sum(v.imag())};
}

// Cooperative banded solve B y = y ('N'), B factored by k_factor. Block = one warp, so warp-level
// barriers (__syncwarp) suffice. Fast path (kl <= 32, kl + ku <= 64): each lane keeps at most one
// (forward) / two (backward) band entries in registers and the next column is prefetched while the
// current one is applied, so the n-step dependency chain no longer waits on global loads.
template <class T>
__device__ void band_solve(const Band<T>& A, const int* ip, T* y, int n, int kl) {
    const int lane = threadIdx.x, kv = A.kv;
    if (kl <= WARP && kv <= 2 * WARP) {
        // forward: y[j] <-> y[ip[j]], y[j+1..j+lm] -= A(j+1.., j) y[j]
        T a = T(0.0);
        int p = n > 1 ? ip[0] : 0;
        if (n > 1 && lane + 1 <= min(kl, n - 1)) a = A(lane + 1, 0);
        for (int j = 0; j < n - 1; ++j) {
            const int lm = min(kl, n - 1 - j);
            T an = T(0.0);
            int pn = 0;
            if (j + 1 < n - 1) {
                pn = ip[j + 1];
                if (lane + 1 <= min(kl, n - 2 - j)) an = A(j + 2 + lane, j + 1);
            }
            if (lane == 0 && p != j) { T t = y[p]; y[p] = y[j]; y[j] = t; }
            __syncwarp();
            const T yj = y[j];
            if (lane + 1 <= lm) y[j + 1 + lane] -= a * yj;
            __syncwarp();
            a = an;
            p = pn;
        }
        // backward: y[j] /= A(j,j), y[lo..j-1] -= A(.., j) y[j]
        auto load = [&](int j, T& d, T& b1, T& b2) {
            const int lo = max(0, j - kv);
            d = A(j, j);
            const int i1 = j - 1 - lane, i2 = j - 1 - lane - WARP;
            b1 = i1 >= lo ? A(i1, j) : T(0.0);
            b2 = i2 >= lo ? A(i2, j) : T(0.0);
        };
        T d, b1, b2;
        load(n - 1, d, b1, b2);
        for (int j = n - 1; j >= 0; --j) {
            T dn = T(1.0), b1n = T(0.0), b2n = T(0.0);
            if (j > 0) load(j - 1, dn, b1n, b2n);
            const T yj = y[j] / d;
            __syncwarp();
            if (lane == 0) y[j] = yj;
            const int lo = max(0, j - kv);
            const int i1 = j - 1 - lane, i2 = j - 1 - lane - WARP;
            if (i1 >= lo) y[i1] -= b1 * yj;
            if (i2 >= lo) y[i2] -= b2 * yj;
            __syncwarp();
            d = dn;
            b1 = b1n;
            b2 = b2n;
        }
        return;
    }
    for (int j = 0; j < n - 1; ++j) {
        int lm = min(kl, n - 1 - j);
        if (lane == 0) {
            int l = ip[j];
            if (l != j) { T t = y[l]; y[l] = y[j]; y[j] = t; }
        }
        __syncwarp();
        T yj = y[j];
        for (int i = 1 + lane; i <= lm; i += WARP) y[j + i] -= A(j + i, j) * yj;
        __syncwarp();
    }
    for (int j = n - 1; j >= 0; --j) {
        if (lane == 0) y[j] /= A(j, j);
        __syncwarp();
        T yj = y[j];
        int lo = max(0, j - kv);
        for (int i = j - 1 - lane; i >= lo; i -= WARP) y[i] -= A(i, j) * yj;
        __syncwarp();
    }
}

// Small dense LU (column-major r x r) with partial pivoting, single thread. Returns false if singular.
template <class T>
__device__ bool small_lu(T* S, int* piv, int r) {
    for (int j = 0; j < r; ++j) {
        int p = j;
        double best = absval(S[j * r + j]);
        for (int i = j + 1; i < r; ++i) {
            double v = absval(S[j * r + i]);
            if (v > best) { best = v; p = i; }
        }
        piv[j] = p;
        if (best == 0.0) return false;
        if (p != j)
            for (int c = 0; c < r; ++c) { T t = S[c * r + j]; S[c * r + j] = S[c * r + p]; S[c * r + p] = t; }
        for (int i = j + 1; i < r; ++i) S[j * r + i] /= S[j * r + j];
        for (int c = j + 1; c < r; ++c)
            for (int i = j + 1; i < r; ++i) S[c * r + i] -= S[j * r + i] * S[c * r + j];
    }
    return true;
}

template <class T>
__device__ void small_solve(const T* S, const int* piv, T* z, int r) {
    for (int j = 0; j < r; ++j)
        if (piv[j] != j) { T t = z[j]; z[j] = z[piv[j]]; z[piv[j]] = t; }
    for (int j = 0; j < r; ++j)
        for (int i = j + 1; i < r; ++i) z[i] -= S[j * r + i] * z[j];
    for (int j = r - 1; j >= 0; --j) {
        z[j] /= S[j * r + j];
        for (int i = 0; i < j; ++i) z[i] -= S[j * r + i] * z[j];
    }
}

// Column border, transposed solves: y_k -= (A10^T y)_k / s_k for k < r (warp sum per column;
// xp: r + 1 absolute offsets of this mode, s_k = B(k, k), untouched by the LU of B).
template <class T>
__device__ void xcorr_T(T* y, const long long* xp, const int* xr, const T* xv, const Band<T>& A, int r) {
    const int lane = threadIdx.x;
    for (int k = 0; k < r; ++k) {
        const long long q0 = xp[k], q1 = xp[k + 1];
        if (q0 == q1) continue;                      // uniform over the warp
        T part(0.0);
        for (long long q = q0 + lane; q < q1; q += WARP) part += xv[q] * y[xr[q]];
        const T tot = warp_sum(part);
        if (lane == 0) y[k] -= tot / A(k, k);
        __syncwarp();
    }
}

template <class T>
__global__ void k_factor(T* ab, int* ipiv, const T* Vt, T* W, T* S, int* spiv, int* info,
                         double* cond, int n, int kl, int ku, int ldab, int r, bool pivot,
                         const long long* xptr, const int* xrow, const T* xval, const T* A00) {
    const int m = blockIdx.x, lane = threadIdx.x;
    Band<T> A{ab + std::size_t(m) * ldab * n, ldab, kl + ku};
    int* ip = ipiv + std::size_t(m) * n;
    __shared__ int s_jp, s_ju, s_info;
    if (lane == 0) { s_ju = 0; s_info = 0; }
    __syncthreads();
    for (int j = 0; j < n; ++j) {
        int km = min(kl, n - 1 - j);
        if (lane == 0) {
            int jp = 0;
            if (pivot) {
                double best = absval(A(j, j));
                for (int i = 1; i <= km; ++i) {
                    double v = absval(A(j + i, j));
                    if (v > best) { best = v; jp = i; }
                }
            }
            ip[j] = j + jp;
            if (absval(A(j + jp, j)) == 0.0) {
                if (!s_info) s_info = j + 1;
                s_jp = -1;
            } else {
                s_ju = max(s_ju, min(j + ku + jp, n - 1));
                s_jp = jp;
            }
        }
        __syncthreads();
        const int jp = s_jp, ju = s_ju;
        if (jp >= 0) {
            if (jp > 0) {
                for (int c = j + lane; c <= ju; c += WARP) {
                    T t = A(j, c); A(j, c) = A(j + jp, c); A(j + jp, c) = t;
                }
                __syncthreads();
            }
            if (km > 0) {
                T inv = T(1.0) / A(j, j);
                for (int i = 1 + lane; i <= km; i += WARP) A(j + i, j) *= inv;
                __syncthreads();
                const int ncol = ju - j, total = km * ncol;
                for (int t = lane; t < total; t += WARP) {
                    int i = 1 + t % km, c = j + 1 + t / km;
                    A(j + i, c) -= A(j + i, j) * A(j, c);
                }
            }
        }
        __syncthreads();
    }
    if (lane == 0) info[m] = s_info;
    if (s_info || r == 0) { if (lane == 0) cond[m] = 1.0; return; }

    // W = B^{-1} U (rows of W are the r solution vectors), S = I + V^T W (column-major).
    // Column border: W = [-I; A11^{-1} A10], S = A00 + V^T[:, r:] W[r:] (Schur complement).
    T* Wm = W + std::size_t(m) * r * n;
    const T* Vm = Vt + std::size_t(m) * r * n;
    const long long* xp = xptr ? xptr + std::size_t(m) * (r + 1) : nullptr;
    for (int k = 0; k < r; ++k) {
        T* w = Wm + std::size_t(k) * n;
        for (int i = lane; i < n; i += WARP) w[i] = (!xp && i == k) ? T(1.0) : T(0.0);
        __syncthreads();
        if (xp) {
            for (long long q = xp[k] + lane; q < xp[k + 1]; q += WARP) w[xrow[q]] = xval[q];
            __syncthreads();
        }
        band_solve(A, ip, w, n, kl);
        __syncthreads();
        if (xp) for (int i = lane; i < r; i += WARP) w[i] = (i == k) ? T(-1.0) : T(0.0);
    }
    __syncthreads();
    T* Sm = S + std::size_t(m) * r * r;
    const T* A0 = xp ? A00 + std::size_t(m) * r * r : nullptr;
    for (int a = 0; a < r; ++a)
        for (int b = 0; b < r; ++b) {
            T part(0.0);
            for (int i = (xp ? r : 0) + lane; i < n; i += WARP) part += Vm[std::size_t(a) * n + i] * Wm[std::size_t(b) * n + i];
            T tot = warp_sum(part);
            if (lane == 0) Sm[b * r + a] = tot + (xp ? A0[a * r + b] : ((a == b) ? T(1.0) : T(0.0)));
        }
    __syncthreads();
    if (lane == 0) {
        double nS = 0.0;
        for (int b = 0; b < r; ++b) {
            double s = 0.0;
            for (int a = 0; a < r; ++a) s += absval(Sm[b * r + a]);
            nS = fmax(nS, s);
        }
        int* sp = spiv + std::size_t(m) * r;
        if (!small_lu(Sm, sp, r)) {
            info[m] = -1;
            cond[m] = INFINITY;
        } else {
            double nInv = 0.0;
            for (int b = 0; b < r; ++b) {
                T e[MAXR];
                for (int a = 0; a < r; ++a) e[a] = (a == b) ? T(1.0) : T(0.0);
                small_solve(Sm, sp, e, r);
                double s = 0.0;
                for (int a = 0; a < r; ++a) s += absval(e[a]);
                nInv = fmax(nInv, s);
            }
            cond[m] = nS * nInv;
        }
    }
}

// Right-hand side gather / solution scatter of the solve kernels (int64 (batch, n) maps):
//   plain : b_m[i] = rhs[rows[m n + i]], x[cols[m n + i]] = x_m[i]; null maps = contiguous layout;
//   split : complex systems of real data (complexified RealFourier pairs):
//           b_m[i] = re[rows[q]] + i re[rows2[q]], x[cols[q]] = Re x_m[i], x[cols2[q]] = Im x_m[i].
template <class T>
struct Io {
    const T* rhs;
    const long long* rows;
    T* x;
    const long long* cols;
    const double* rre = nullptr;
    const long long* rows2 = nullptr;
    double* xre = nullptr;
    const long long* cols2 = nullptr;
    __device__ T load(std::size_t q) const {
        if constexpr (!std::is_same_v<T, double>) {
            if (rre) return T(rre[rows[q]], rre[rows2[q]]);
        }
        return rhs[rows ? rows[q] : q];
    }
    __device__ void store(std::size_t q, const T& v) const {
        if constexpr (!std::is_same_v<T, double>) {
            if (xre) { xre[cols[q]] = v.real(); xre[cols2[q]] = v.imag(); return; }
        }
        x[cols ? cols[q] : q] = v;
    }
};

// Plain / split gather-scatter (see Io); null maps mean the contiguous (batch, n) layout.
template <class T>
__global__ void k_solve(const T* ab, const int* ipiv, const T* Vt, const T* W, const T* S,
                        const int* spiv, const int* perm, Io<T> io, T* scratch, int n, int kl,
                        int ku, int ldab, int r, bool use_shared, bool schur) {
    extern __shared__ unsigned char smem_raw[];
    const int m = blockIdx.x, lane = threadIdx.x;
    T* y = use_shared ? reinterpret_cast<T*>(smem_raw) : scratch + std::size_t(m) * n;
    Band<T> A{const_cast<T*>(ab) + std::size_t(m) * ldab * n, ldab, kl + ku};
    const int* ip = ipiv + std::size_t(m) * n;
    const std::size_t mo = std::size_t(m) * n;
    for (int i = lane; i < n; i += WARP) y[i] = io.load(mo + perm[i]);
    __syncthreads();
    band_solve(A, ip, y, n, kl);
    if (r > 0) {
        __shared__ alignas(16) unsigned char zraw[MAXR * sizeof(T)];
        T* z = reinterpret_cast<T*>(zraw);
        const T* Vm = Vt + std::size_t(m) * r * n;
        for (int k = 0; k < r; ++k) {
            T part(0.0);
            for (int i = lane; i < n; i += WARP) part += Vm[std::size_t(k) * n + i] * y[i];
            T tot = warp_sum(part);
            if (lane == 0) z[k] = tot;
        }
        __syncthreads();
        if (lane == 0) small_solve(S + std::size_t(m) * r * r, spiv + std::size_t(m) * r, z, r);
        __syncthreads();
        const T* Wm = W + std::size_t(m) * r * n;
        for (int i = lane; i < n; i += WARP) {
            T acc = (schur && i < r) ? T(0.0) : y[i];        // column border: x0 = z (W top = -I)
            for (int k = 0; k < r; ++k) acc -= Wm[std::size_t(k) * n + i] * z[k];
            y[i] = acc;
        }
        __syncthreads();
    }
    for (int i = lane; i < n; i += WARP) io.store(mo + i, y[i]);
}

// Transposed banded solve B^T y = y in place, B factored by k_factor (LAPACK dgbtrs 'T'):
// U^T w = y (forward, one warp reduction per row), then L^T with the interchanges in reverse.
template <class T>
__device__ void band_solve_T(const Band<T>& A, const int* ip, T* y, int n, int kl) {
    const int lane = threadIdx.x, kv = A.kv;
    for (int j = 0; j < n; ++j) {
        const int lo = max(0, j - kv);
        T part(0.0);
        for (int i = lo + lane; i < j; i += WARP) part += A(i, j) * y[i];
        const T tot = warp_sum(part);
        if (lane == 0) y[j] = (y[j] - tot) / A(j, j);
        __syncwarp();
    }
    for (int j = n - 2; j >= 0; --j) {
        const int lm = min(kl, n - 1 - j);
        T part(0.0);
        for (int i = 1 + lane; i <= lm; i += WARP) part += A(j + i, j) * y[j + i];
        const T tot = warp_sum(part);
        if (lane == 0) {
            y[j] -= tot;
            const int l = ip[j];
            if (l != j) { T t = y[l]; y[l] = y[j]; y[j] = t; }
        }
        __syncwarp();
    }
}

// Transposed Woodbury data per mode: Wt = B^{-T} V (rows = r vectors), St = I + U^T Wt, LU-factored.
template <class T>
__global__ void k_prepare_T(const T* ab, const int* ipiv, const T* Vt, T* Wt, T* St, int* stpiv,
                            int* info, int n, int kl, int ku, int ldab, int r,
                            const long long* xptr, const T* W, const T* A00) {
    const int m = blockIdx.x, lane = threadIdx.x;
    Band<T> A{const_cast<T*>(ab) + std::size_t(m) * ldab * n, ldab, kl + ku};
    const int* ip = ipiv + std::size_t(m) * n;
    T* Wm = Wt + std::size_t(m) * r * n;
    const T* Vm = Vt + std::size_t(m) * r * n;
    // column border (xptr set): Wt = [-I; A11^{-T} A01^T] (A01 = -V^T[:, r:]), St = diag(1/s) S^T
    // with S = A00 + V^T[:, r:] W[r:] recomputed from the forward W (the factored S is in place)
    for (int k = 0; k < r; ++k) {
        T* w = Wm + std::size_t(k) * n;
        for (int i = lane; i < n; i += WARP)
            w[i] = xptr ? (i < r ? T(0.0) : -Vm[std::size_t(k) * n + i]) : Vm[std::size_t(k) * n + i];
        __syncwarp();
        band_solve_T(A, ip, w, n, kl);
        __syncwarp();
        if (xptr) for (int i = lane; i < r; i += WARP) w[i] = (i == k) ? T(-1.0) : T(0.0);
    }
    __syncwarp();
    T* Sm = St + std::size_t(m) * r * r;
    if (xptr) {
        const T* Wf = W + std::size_t(m) * r * n;
        const T* A0 = A00 + std::size_t(m) * r * r;
        for (int a = 0; a < r; ++a)
            for (int b = 0; b < r; ++b) {        // S(b, a) = A00(b, a) + V^T[b, r:] . W[a][r:]
                T part(0.0);
                for (int i = r + lane; i < n; i += WARP) part += Vm[std::size_t(b) * n + i] * Wf[std::size_t(a) * n + i];
                const T tot = warp_sum(part);
                if (lane == 0) Sm[b * r + a] = (tot + A0[b * r + a]) / A(a, a);
            }
        __syncwarp();
    }
    if (lane == 0) {
        if (!xptr)
            for (int a = 0; a < r; ++a)
                for (int b = 0; b < r; ++b)
                    Sm[b * r + a] = Wm[std::size_t(b) * n + a] + ((a == b) ? T(1.0) : T(0.0));
        info[m] = small_lu(Sm, stpiv + std::size_t(m) * r, r) ? 0 : -1;
    }
}

// A^T x = b (A = P^T (B + U V^T)): y = B^{-T} b, z = St^{-1} U^T y, y -= Wt z, x = P^T y.
template <class T>
__global__ void k_solve_T(const T* ab, const int* ipiv, const T* Wt, const T* St, const int* stpiv,
                          const int* perm, T* b, T* scratch, int n, int kl, int ku, int ldab, int r,
                          bool use_shared, const long long* xptr, const int* xrow, const T* xval) {
    // column border (xptr set): top entries corrected by xcorr_T, then x0 = z (Wt top = -I)
    extern __shared__ unsigned char smem_raw[];
    const int m = blockIdx.x, lane = threadIdx.x;
    T* y = use_shared ? reinterpret_cast<T*>(smem_raw) : scratch + std::size_t(m) * n;
    Band<T> A{const_cast<T*>(ab) + std::size_t(m) * ldab * n, ldab, kl + ku};
    const int* ip = ipiv + std::size_t(m) * n;
    T* bm = b + std::size_t(m) * n;
    for (int i = lane; i < n; i += WARP) y[i] = bm[i];
    __syncwarp();
    band_solve_T(A, ip, y, n, kl);
    if (xptr) {
        __syncwarp();
        xcorr_T(y, xptr + std::size_t(m) * (r + 1), xrow, xval, A, r);
    }
    if (r > 0) {
        __shared__ alignas(16) unsigned char zraw[MAXR * sizeof(T)];
        T* z = reinterpret_cast<T*>(zraw);
        if (lane == 0) {
            for (int k = 0; k < r; ++k) z[k] = y[k];
            small_solve(St + std::size_t(m) * r * r, stpiv + std::size_t(m) * r, z, r);
        }
        __syncwarp();
        const T* Wm = Wt + std::size_t(m) * r * n;
        for (int i = lane; i < n; i += WARP) {
            T acc = (xptr && i < r) ? T(0.0) : y[i];
            for (int k = 0; k < r; ++k) acc -= Wm[std::size_t(k) * n + i] * z[k];
            y[i] = acc;
        }
        __syncwarp();
    }
    for (int i = lane; i < n; i += WARP) bm[perm[i]] = y[i];
}

// ---------------------------------------------------------------------------------------------
// Level-scheduled solves. The band LU of B written by k_factor is ~88 % exact
// zeros for the spectral operators (RB 2D: 4.8 nonzeros per column of A, band 84 wide), and its
// band solve is a chain of n dependent steps per mode (latency bound, 10-35 % of the DRAM roof).
// On the host, after the factorization, each mode's factors are rewritten as
//   P' B = L' U   (L' unit lower: the row interchanges of gbtrf applied to the earlier multiplier
//                  columns, as in getrf; U the band upper factor; exact zeros dropped),
// and the rows of L' (forward) and of U (backward) are grouped by level: a row depends only on
// rows of earlier levels. One warp per mode processes one level per step, one row per lane, the
// work vector y in shared memory. The arithmetic of every entry is the band kernel's: row i
// receives the same updates in the same order (increasing column for L', decreasing column for
// U, then the division by U_ii), so the sweeps are the same to the rounding of fused multiply-adds.
// Woodbury as in k_solve (x = y - W S^{-1} V^T y), with V^T read without its zeros. The diagonal
// of U is stored inverted (a multiply instead of a division on the chain: rounding differs by at
// most one unit per row from the band kernel). (Applying the
// border first, x = B^{-1}(P b - U z) with V^T B^{-1} precomputed, reads less but was 10x less
// accurate on ill-conditioned random bands: rejected.)
// The factor stream of a mode (chunks: header, row ids, column ids, values [, diagonals]) is one
// contiguous word array in execution order, copied into a shared-memory ring by cp.async ahead of
// use: every factor word is read once, coalesced.
//
// Chunk layout (64-bit words): header = nrows | K << 16 | flags << 32; ceil(nrows/4) words of
// int16 row ids; K groups of ceil(nrows/4) words of int16 column ids; K * nrows values (slot-major:
// value k of row slot s at k * nrows + s); backward chunks then nrows diagonal values. A value is
// one word (double) or two (complex: re, im). Padded slots have value 0 and column = the row.
constexpr int LVL_SEG = 256;                        // words per cp.async segment (2 KiB)
#ifndef CN_LVL_NSEG
#define CN_LVL_NSEG 4
#endif
constexpr int LVL_NSEG = CN_LVL_NSEG;               // ring of 4 segments (8 KiB: 4 blocks per SM)
constexpr int LVL_RING = LVL_SEG * LVL_NSEG;
// Header flags: END_LEVEL closes a step (warp barrier after it), LAST ends the sweep, MERGED marks
// a chunk of a merged step (every lane reads before any lane writes; see "Merged levels" below).
constexpr unsigned long long LVL_END_LEVEL = 1ull, LVL_LAST = 2ull, LVL_MERGED = 4ull;

// Words of one chunk of `rows` rows with K entries each (layout above); host and device agree.
__host__ __device__ constexpr long long lvl_chunk_words(int rows, int K, int TW, bool backward) {
    return 1 + (rows + 3) / 4 + static_cast<long long>(K) * ((rows + 3) / 4) +
           static_cast<long long>(K) * rows * TW + (backward ? rows * TW : 0);
}

template <class T> struct LvlWords;
template <> struct LvlWords<double> {
    static constexpr int W = 1;
    __device__ static double get(const unsigned long long* r, long long p) {
        return __longlong_as_double(static_cast<long long>(r[p & (LVL_RING - 1)]));
    }
};
template <> struct LvlWords<::cuda::std::complex<double>> {
    static constexpr int W = 2;
    __device__ static ::cuda::std::complex<double> get(const unsigned long long* r, long long p) {
        return {__longlong_as_double(static_cast<long long>(r[p & (LVL_RING - 1)])),
                __longlong_as_double(static_cast<long long>(r[(p + 1) & (LVL_RING - 1)]))};
    }
};

// Stream of one mode's words through the shared ring (segment s lives in slot s % LVL_NSEG).
struct LvlRing {
    unsigned long long* ring;
    const unsigned long long* src;
    long long issued, nseg;
    long long ready = 0;                 // words below `ready` (and at or after the current chunk) are resident
    // Make words [a, b) resident; keep prefetching up to LVL_NSEG segments from a's segment on.
    // Called only when [a, b) is not known resident: most chunks need no wait and no barrier.
    __device__ __forceinline__ void ensure(long long a, long long b, int lane) {
        const long long sa = a / LVL_SEG, sb = (b - 1) / LVL_SEG;
        const long long lim = min(nseg, sa + LVL_NSEG);
        if (issued < lim) {
            __syncwarp();                // all lanes are done with the slots about to be refilled
            for (; issued < lim; ++issued) {
                unsigned long long* dst = ring + (issued % LVL_NSEG) * LVL_SEG;
                const unsigned long long* s = src + issued * LVL_SEG;
                for (int w = 2 * lane; w < LVL_SEG; w += 2 * WARP) __pipeline_memcpy_async(dst + w, s + w, 16);
                __pipeline_commit();
            }
        }
        const long long pend = issued - 1 - sb;       // groups allowed to stay in flight (<= 7)
        __pipeline_wait_prior(pend < 0 ? 0 : static_cast<std::size_t>(pend));
        __syncwarp();
        ready = (sb + 1) * LVL_SEG;
    }
};

// One sweep (forward: y_i -= sum_k L'_ik y_k; backward: y_i = (y_i - sum_j U_ij y_j) / U_ii).
template <class T, bool BACKWARD>
__device__ __forceinline__ long long lvl_sweep(LvlRing& R, long long p, T* y, int lane, int& nchunk, int& nens,
                               long long& tens) {
    using V = LvlWords<T>;
    constexpr int TW = V::W;
    constexpr unsigned MASK = LVL_RING - 1;
    if (p + 1 > R.ready) { ++nens; R.ensure(p, p + 1, lane); }
    unsigned long long h = R.ring[p & MASK];
    while (true) {
        ++nchunk;
        const int nrows = static_cast<int>(h & 0xFFFFull), K = static_cast<int>((h >> 16) & 0xFFFFull);
        const unsigned long long flags = h >> 32;
        const int q = (nrows + 3) / 4;
        const int size = 1 + q + K * q + K * nrows * TW + (BACKWARD ? nrows * TW : 0);
        // this chunk, and the next header (read now: off the dependency chain of the next level)
        const bool last = (flags & LVL_LAST) != 0;
        const long long need = p + size + (last ? 0 : 1);
        if (need > R.ready) {
            ++nens;
            const long long t0 = clock64();
            R.ensure(p, need, lane);
            tens += clock64() - t0;
        }
        const unsigned long long hn = last ? 0ull : R.ring[(p + size) & MASK];
        int i = 0;
        T acc(0.0);
        if (lane < nrows) {
            // 32-bit offsets inside the ring (wrapped by the mask)
            const unsigned b = static_cast<unsigned>(p & MASK);
            const unsigned wr = R.ring[(b + 1 + (lane >> 2)) & MASK] >> (16 * (lane & 3)) & 0xFFFFull;
            i = static_cast<int>(wr);
            acc = y[i];
            const unsigned pc = b + 1 + q, pv = b + 1 + q + K * q;
            // Sequential dot product over the K entries of this row, one chain of fused
            // multiply-adds in a fixed order (deterministic rounding, bit-identical replays).
            // Note (run): a "loads-first" variant for
            // K <= 8 (all column ids and values, then all y[c], then two interleaved partial
            // sums) was measured SLOWER on A100 (RK443 1024x256: 5.66 vs 4.96 ms/step, run
            //) and changed the summation order; it was reverted.
#pragma unroll 4
            for (int k = 0; k < K; ++k) {
                const int cc = static_cast<int>(R.ring[(pc + k * q + (lane >> 2)) & MASK] >> (16 * (lane & 3)) & 0xFFFFull);
                acc -= V::get(R.ring, pv + (k * nrows + lane) * TW) * y[cc];
            }
            if constexpr (BACKWARD) acc = acc * V::get(R.ring, pv + (K * nrows + lane) * TW);
        }
        // A merged step reads values that rows of the same step overwrite (their inputs): all
        // reads first. Unmerged levels never read a row of their own level: no extra barrier.
        if (flags & LVL_MERGED) __syncwarp();
        if (lane < nrows) y[i] = acc;
        if (flags & LVL_END_LEVEL) __syncwarp();
        p += size;
        if (last) return p;
        h = hn;
    }
}

template <class T>
__global__ void __launch_bounds__(32) k_solve_lvl(const unsigned long long* blob, const long long* off, const int* gperm,
                            const long long* vptr, const int* vcol, const T* vval, const T* W,
                            const T* S, const int* spiv, Io<T> io, int n, int r,
                            long long* probe, bool schur) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const int m = blockIdx.x, lane = threadIdx.x;
    // probe (null in normal use): per block clock64 stamps and chunk / wait counts (diagnostics)
    long long* pr = probe ? probe + std::size_t(m) * 10 : nullptr;
    if (pr && lane == 0) pr[0] = clock64();
    int nch_f = 0, nen_f = 0, nch_b = 0, nen_b = 0;
    long long tens = 0;
    unsigned long long* ring = reinterpret_cast<unsigned long long*>(smem_raw);
    T* y = reinterpret_cast<T*>(smem_raw + LVL_RING * sizeof(unsigned long long));
    const long long o0 = off[m], len = off[m + 1] - o0;
    LvlRing R{ring, blob + o0, 0, (len + LVL_SEG - 1) / LVL_SEG};
    R.ensure(0, 1, lane);                              // start the factor stream first
    // border data staged now (its loads overlap the sweeps): V^T row pointers, the LU factors
    // and pivots of the r x r capacitance matrix (r <= 16; larger r reads them from global memory)
    constexpr int RS = 16;
    __shared__ long long svp[RS + 1];
    __shared__ alignas(16) unsigned char sraw[RS * RS * sizeof(T)];
    __shared__ int sp[RS];
    T* Ss = reinterpret_cast<T*>(sraw);
    if (r > 0 && r <= RS) {
        const T* Sm = S + std::size_t(m) * r * r;
        for (int e = lane; e < r * r; e += WARP) Ss[e] = Sm[e];
        if (lane < r) sp[lane] = spiv[std::size_t(m) * r + lane];
        if (lane <= r) svp[lane] = vptr[std::size_t(m) * (r + 1) + lane];
    }
    const std::size_t mo = std::size_t(m) * n;
    const int* gp = gperm + mo;
    for (int t = lane; t < n; t += WARP) y[t] = io.load(mo + gp[t]);   // y = P' P b
    __syncwarp();
    if (pr && lane == 0) pr[1] = clock64();
    long long p = lvl_sweep<T, false>(R, 0, y, lane, nch_f, nen_f, tens);
    if (pr && lane == 0) pr[2] = clock64();
    lvl_sweep<T, true>(R, p, y, lane, nch_b, nen_b, tens);
    __syncwarp();
    if (pr && lane == 0) { pr[3] = clock64(); pr[5] = nch_f; pr[6] = nen_f; pr[7] = nch_b; pr[8] = nen_b; pr[9] = tens; }
    if (r > 0) {                                        // Woodbury, as in k_solve
        __shared__ alignas(16) unsigned char zraw[MAXR * sizeof(T)];
        T* z = reinterpret_cast<T*>(zraw);
        // z = V^T y with V^T stored sparse (its non-zeros only; lane-strided over them, then
        // the warp sum: the partial sums differ from the dense loop of k_solve by rounding only)
        const long long* vp = vptr + std::size_t(m) * (r + 1);
        constexpr int RB = 16;
        if (r > RB) {
            for (int k = 0; k < r; ++k) {
                T part(0.0);
                for (long long q = vp[k] + lane; q < vp[k + 1]; q += WARP) part += vval[q] * y[vcol[q] & 0xFFFF];
                T tot = warp_sum(part);
                if (lane == 0) z[k] = tot;
            }
        } else {   // one lane-strided pass over the non-zeros of all r rows (independent loads),
            // partial sums per (row, lane) in shared memory, then a fixed-order sum per row
            __shared__ alignas(16) unsigned char praw[RB * WARP * sizeof(T)];
            T* part = reinterpret_cast<T*>(praw);
            for (int k = 0; k < r; ++k) part[k * WARP + lane] = T(0.0);
            // entries carry their row in bits 16..23 of the column word: a branch-free loop,
            // eight entries in flight per lane (independent global loads; each lane still adds
            // its entries in ascending order, so the partial sums do not depend on the unroll)
            constexpr int QU = 8;
            const long long q0 = svp[0], q1 = svp[r];
            for (long long q = q0 + lane; q < q1; q += QU * WARP) {
                int cw[QU];
                T vv[QU];
#pragma unroll
                for (int u = 0; u < QU; ++u) {
                    const long long qq = q + u * WARP;
                    cw[u] = qq < q1 ? vcol[qq] : -1;
                    vv[u] = qq < q1 ? vval[qq] : T(0.0);
                }
#pragma unroll
                for (int u = 0; u < QU; ++u)
                    if (cw[u] >= 0) part[(cw[u] >> 16) * WARP + lane] += vv[u] * y[cw[u] & 0xFFFF];
            }
            __syncwarp();
            for (int kk = 0; kk < r; ++kk) {
                const T tot = warp_sum(part[kk * WARP + lane]);
                if (lane == 0) z[kk] = tot;
            }
        }
        __syncwarp();
        if (r <= RS) {
            // S z' = z by the LU of S, one lane per entry: entry i receives the same updates in
            // the same order as in small_solve (bitwise the serial result), r + r warp steps
            // instead of r^2 dependent shared-memory updates on one lane (512)
            if (lane == 0)
                for (int j = 0; j < r; ++j)
                    if (sp[j] != j) { T t = z[j]; z[j] = z[sp[j]]; z[sp[j]] = t; }
            __syncwarp();
            for (int j = 0; j < r; ++j) {
                const T zj = z[j];
                if (lane > j && lane < r) z[lane] -= Ss[j * r + lane] * zj;
                __syncwarp();
            }
            for (int j = r - 1; j >= 0; --j) {
                if (lane == 0) z[j] /= Ss[j * r + j];
                __syncwarp();
                const T zj = z[j];
                if (lane < j) z[lane] -= Ss[j * r + lane] * zj;
                __syncwarp();
            }
        } else {
            if (lane == 0) small_solve(S + std::size_t(m) * r * r, spiv + std::size_t(m) * r, z, r);
            __syncwarp();
        }
        const T* Wm = W + std::size_t(m) * r * n;
        if (r <= RS) {
            // x = y - W z': for two columns per lane, all r entries of W loaded first (up to 32
            // independent global loads in flight; W is r x n per mode and read once: this loop
            // was latency bound with four), then acc -= w_k z_k in increasing k as before
            for (int i0 = lane; i0 < n; i0 += 2 * WARP) {
                const bool two = i0 + WARP < n;
                T w0[RS], w1[RS];
#pragma unroll
                for (int k = 0; k < RS; ++k)
                    if (k < r) {
                        w0[k] = Wm[std::size_t(k) * n + i0];
                        w1[k] = two ? Wm[std::size_t(k) * n + i0 + WARP] : T(0.0);
                    }
                // column border: x0 = z (W top = -I), the final x starts from 0 there
                T a0 = (schur && i0 < r) ? T(0.0) : y[i0];
                T a1 = two ? ((schur && i0 + WARP < r) ? T(0.0) : y[i0 + WARP]) : T(0.0);
#pragma unroll
                for (int k = 0; k < RS; ++k)
                    if (k < r) {
                        const T zk = z[k];
                        a0 -= w0[k] * zk;
                        a1 -= w1[k] * zk;
                    }
                io.store(mo + i0, a0);
                if (two) io.store(mo + i0 + WARP, a1);
            }
        } else {
            for (int i0 = lane; i0 < n; i0 += 4 * WARP) {   // four columns in flight per lane
                T acc[4];
#pragma unroll
                for (int u = 0; u < 4; ++u)
                    acc[u] = (i0 + u * WARP < n && !(schur && i0 + u * WARP < r)) ? y[i0 + u * WARP] : T(0.0);
                for (int k = 0; k < r; ++k) {
                    const T zk = z[k];
                    const T* wk = Wm + std::size_t(k) * n;
#pragma unroll
                    for (int u = 0; u < 4; ++u)
                        if (i0 + u * WARP < n) acc[u] -= wk[i0 + u * WARP] * zk;
                }
#pragma unroll
                for (int u = 0; u < 4; ++u)
                    if (i0 + u * WARP < n) io.store(mo + i0 + u * WARP, acc[u]);
            }
        }
    } else {
        for (int i = lane; i < n; i += WARP) io.store(mo + i, y[i]);
    }
    if (pr && lane == 0) pr[4] = clock64();
}

// Transposed level-scheduled solve: A^T x = b, A = P^T (B + U V^T), with
// P' B = L' U. B^T = U^T L'^T P', so y0 = B^{-T} b is: U^T w = b (rows of U^T in increasing order,
// with the diagonal: the "backward" chunk format), L'^T v = w (unit rows, decreasing order: the
// "forward" format), y0[rho] = v[fpos[rho]]; then the transposed Woodbury as in k_solve_T
// (z = St^{-1} y0[0:r], y = y0 - Wt z) and x[perm[i]] = y[i]. Same factors as the band kernel
// k_solve_T, same updates per row up to their summation order (warp sums there): equal to rounding.
// Built only on request (build_transpose_schedule); it lets the band storage be released.
template <class T>
__global__ void __launch_bounds__(32) k_solve_lvl_T(const unsigned long long* blob, const long long* off,
                            const int* fpos, const int* perm, const T* Wt, const T* St,
                            const int* stpiv, T* b, int n, int r) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const int m = blockIdx.x, lane = threadIdx.x;
    int nch = 0, nen = 0;
    long long tens = 0;
    unsigned long long* ring = reinterpret_cast<unsigned long long*>(smem_raw);
    T* y = reinterpret_cast<T*>(smem_raw + LVL_RING * sizeof(unsigned long long));
    const long long o0 = off[m], len = off[m + 1] - o0;
    LvlRing R{ring, blob + o0, 0, (len + LVL_SEG - 1) / LVL_SEG};
    R.ensure(0, 1, lane);
    const std::size_t mo = std::size_t(m) * n;
    T* bm = b + mo;
    for (int t = lane; t < n; t += WARP) y[t] = bm[t];
    __syncwarp();
    long long p = lvl_sweep<T, true>(R, 0, y, lane, nch, nen, tens);     // U^T w = b
    lvl_sweep<T, false>(R, p, y, lane, nch, nen, tens);                  // L'^T v = w
    __syncwarp();
    const int* fp = fpos + mo;
    if (r > 0) {
        __shared__ alignas(16) unsigned char zraw[MAXR * sizeof(T)];
        T* z = reinterpret_cast<T*>(zraw);
        if (lane == 0) {
            for (int k = 0; k < r; ++k) z[k] = y[fp[k]];
            small_solve(St + std::size_t(m) * r * r, stpiv + std::size_t(m) * r, z, r);
        }
        __syncwarp();
        const T* Wm = Wt + std::size_t(m) * r * n;
        for (int i = lane; i < n; i += WARP) {
            T acc = y[fp[i]];
            for (int k = 0; k < r; ++k) acc -= Wm[std::size_t(k) * n + i] * z[k];
            bm[perm[i]] = acc;
        }
    } else {
        for (int i = lane; i < n; i += WARP) bm[perm[i]] = y[fp[i]];
    }
}

// Host side: the level-scheduled stream of one mode from its band LU (see above).
template <class T>
struct LvlBuilder {
    int n, kl, ku, ldab;
    struct Row { int i; std::vector<std::pair<int, T>> e; T d; };
    static void put_value(std::vector<unsigned long long>& out, const T& v) {
        if constexpr (std::is_same_v<T, double>) {
            unsigned long long w;
            std::memcpy(&w, &v, 8);
            out.push_back(w);
        } else {
            double re = v.real(), im = v.imag();
            unsigned long long a, b;
            std::memcpy(&a, &re, 8);
            std::memcpy(&b, &im, 8);
            out.push_back(a);
            out.push_back(b);
        }
    }
    static void put_ids(std::vector<unsigned long long>& out, const std::vector<int>& ids) {
        for (std::size_t w = 0; w < (ids.size() + 3) / 4; ++w) {
            unsigned long long word = 0;
            for (int s = 0; s < 4; ++s) {
                std::size_t j = 4 * w + s;
                if (j < ids.size()) word |= static_cast<unsigned long long>(ids[j] & 0xFFFF) << (16 * s);
            }
            out.push_back(word);
        }
    }
    // Merged levels (512). A level holds ~4 rows of RB 2D (one warp of 32 lanes), so
    // the sweep is a chain of ~n/4 steps whose cost is mostly per-step latency. Consecutive levels
    // are merged into one step of up to WARP rows: a row of a later level is rewritten in terms of
    // the values y holds at the start of the step (its own input, the inputs of the step's
    // earlier rows, the results of earlier steps), by substituting the expressions of the step's
    // earlier rows:
    //   forward   y_i -= sum_j L'_ij y_j,       y_j = y_j0 - sum_k v_jk y_k
    //             -> y_i -= L'_ij y_j0 - sum_k L'_ij v_jk y_k
    //   backward  y_i = d_i (y_i - sum_j U_ij y_j), y_j = d_j (y_j0 - sum_k v_jk y_k)
    //             -> y_i = d_i (y_i - U_ij d_j y_j0 + sum_k U_ij d_j v_jk y_k)
    // so the stream format and the per-row arithmetic of the kernel are unchanged (acc -= v y[c],
    // then the diagonal); only the kernel's write order is (MERGED chunks: all lanes read, then
    // write). The coefficients of a merged row are products of factor entries formed on the host
    // in double: the result differs from the unmerged sweep by rounding (substitution is the same
    // elimination; measured <= 2e-13 relative on RB 2D, tests/cross_backend/test_banded_level.py).
    // A step is closed when the next level would exceed WARP rows or one ring chunk; a level that
    // alone exceeds them stays a step of its own (chunked as before). RB 2D 512x128, k != 0:
    // 147 + 226 levels -> 26 + 34 steps, 1.7x the stream words.
    using Step = std::pair<std::vector<Row>, bool>;      // rows, merged (more than one level)
    static std::vector<Step> merge_levels(std::vector<std::vector<Row>>& levels, bool backward, int n,
                                          bool merge) {
        constexpr int TW = std::is_same_v<T, double> ? 1 : 2;
        const long long cap = static_cast<long long>(LVL_NSEG - 1) * LVL_SEG - 1;
        auto kmax = [](const std::vector<Row>& rows) {
            std::size_t k = 0;
            for (const auto& r : rows) k = std::max(k, r.e.size());
            return static_cast<int>(k);
        };
        auto fits = [&](const std::vector<Row>& rows) {
            return static_cast<int>(rows.size()) <= WARP &&
                   lvl_chunk_words(static_cast<int>(rows.size()), kmax(rows), TW, backward) <= cap;
        };
        std::vector<Step> steps;
        std::vector<int> where(n, -1);                  // row -> slot in the open step
        std::vector<T> acc(n, T(0.0));                  // dense accumulator of one expanded row
        std::vector<char> used(n, 0);
        std::vector<int> touched;
        std::vector<Row> cur;
        int nlev = 0;                                   // levels in the open step
        bool open = false;                              // the open step may take more levels
        auto close = [&]() {
            if (cur.empty()) return;
            for (const auto& r : cur) where[r.i] = -1;
            steps.emplace_back(std::move(cur), nlev > 1);
            cur.clear();
            nlev = 0;
        };
        auto start = [&](std::vector<Row>& L) {
            cur = std::move(L);
            nlev = 1;
            for (int s = 0; s < static_cast<int>(cur.size()); ++s) where[cur[s].i] = s;
            open = merge && fits(cur);
        };
        for (auto& L : levels) {
            if (L.empty()) continue;
            if (!open) { close(); start(L); continue; }
            // expand the rows of L against the open step
            std::vector<Row> ex;
            ex.reserve(L.size());
            for (const auto& R : L) {
                auto add = [&](int c, const T& v) {
                    if (!used[c]) { used[c] = 1; touched.push_back(c); acc[c] = T(0.0); }
                    acc[c] += v;
                };
                for (const auto& [j, v] : R.e) {
                    if (where[j] < 0) { add(j, v); continue; }
                    const Row& Rj = cur[where[j]];
                    const T f = backward ? v * Rj.d : v;
                    add(j, f);
                    for (const auto& [k, vk] : Rj.e) add(k, -(f * vk));
                }
                Row E{R.i, {}, R.d};
                std::sort(touched.begin(), touched.end());
                if (backward) std::reverse(touched.begin(), touched.end());
                for (int c : touched) {
                    if (acc[c] != T(0.0)) E.e.push_back({c, acc[c]});
                    used[c] = 0;
                }
                touched.clear();
                ex.push_back(std::move(E));
            }
            std::vector<Row> trial = cur;
            for (auto& E : ex) trial.push_back(E);
            if (fits(trial)) {
                for (auto& E : ex) { where[E.i] = static_cast<int>(cur.size()); cur.push_back(std::move(E)); }
                ++nlev;
            } else {
                close();
                start(L);
            }
        }
        close();
        return steps;
    }
    // steps -> chunks of <= 32 rows (fewer if a chunk would not fit the ring); a merged step is
    // always one chunk (merge_levels checked it)
    static bool emit(std::vector<unsigned long long>& out, std::vector<Step>& steps, bool backward) {
        constexpr int TW = std::is_same_v<T, double> ? 1 : 2;
        bool any = false;
        std::size_t last_header = 0;
        for (auto& [L, merged] : steps) {
            if (L.empty()) continue;
            std::stable_sort(L.begin(), L.end(), [](const Row& a, const Row& b) { return a.e.size() > b.e.size(); });
            std::size_t pos = 0;
            while (pos < L.size()) {
                int nr = static_cast<int>(std::min<std::size_t>(WARP, L.size() - pos));
                const int K = static_cast<int>(L[pos].e.size());   // sorted: first row has the most
                // a chunk plus the next header must fit LVL_NSEG - 1 segments (lvl_sweep reads
                // the next header early; the ring holds LVL_NSEG segments from the chunk's start)
                while (nr > 1 && lvl_chunk_words(nr, K, TW, backward) + 1 > (LVL_NSEG - 1) * LVL_SEG) nr /= 2;
                if (lvl_chunk_words(nr, K, TW, backward) + 1 > (LVL_NSEG - 1) * LVL_SEG) return false;
                if (merged && nr != static_cast<int>(L.size())) return false;   // cannot happen
                const bool end = pos + nr == L.size();
                last_header = out.size();
                out.push_back(static_cast<unsigned long long>(nr) | (static_cast<unsigned long long>(K) << 16) |
                              (((end ? LVL_END_LEVEL : 0ull) | (merged ? LVL_MERGED : 0ull)) << 32));
                std::vector<int> ids(nr);
                for (int s = 0; s < nr; ++s) ids[s] = L[pos + s].i;
                put_ids(out, ids);
                for (int k = 0; k < K; ++k) {
                    for (int s = 0; s < nr; ++s) {
                        const auto& e = L[pos + s].e;
                        ids[s] = k < static_cast<int>(e.size()) ? e[k].first : L[pos + s].i;
                    }
                    put_ids(out, ids);
                }
                for (int k = 0; k < K; ++k)
                    for (int s = 0; s < nr; ++s) {
                        const auto& e = L[pos + s].e;
                        put_value(out, k < static_cast<int>(e.size()) ? e[k].second : T(0.0));
                    }
                if (backward)
                    for (int s = 0; s < nr; ++s) put_value(out, L[pos + s].d);
                pos += nr;
                any = true;
            }
        }
        if (!any) {                                     // nothing to do in this sweep: empty chunk
            last_header = out.size();
            out.push_back(LVL_END_LEVEL << 32);
        }
        out[last_header] |= LVL_LAST << 32;
        return true;
    }
    // Returns false if a chunk cannot fit the ring. depth_f / depth_b: number of levels;
    // steps_f / steps_b: warp steps after merging (= depth when merge is false).
    bool build(const T* ab, const int* ip, const int* perm, std::vector<unsigned long long>& out,
               int* gperm, int* fpos, int& depth_f, int& depth_b, bool merge, int& steps_f,
               int& steps_b) const {
        const int kv = kl + ku;
        auto A = [&](int i, int j) -> const T& { return ab[std::size_t(j) * ldab + kv + i - j]; };
        // L': entries (final row, column k, value), final row = F_k(k + i) (later interchanges)
        std::vector<int> F(n);
        for (int i = 0; i < n; ++i) F[i] = i;
        std::vector<std::vector<std::pair<int, T>>> lrow(n);
        for (int k = n - 1; k >= 0; --k) {
            const int lm = std::min(kl, n - 1 - k);
            for (int i = 1; i <= lm; ++i) {
                const T& v = A(k + i, k);
                if (v != T(0.0)) lrow[F[k + i]].push_back({k, v});
            }
            std::swap(F[k], F[ip[k]]);
        }
        // y[F[rho]] = (P b)[rho] = b[perm[rho]]
        for (int rho = 0; rho < n; ++rho) { gperm[F[rho]] = perm[rho]; fpos[rho] = F[rho]; }
        std::vector<int> lev(n, 0);
        int dmax = 0;
        std::vector<std::vector<Row>> levels(1);
        for (int i = 0; i < n; ++i) {
            if (lrow[i].empty()) continue;
            std::sort(lrow[i].begin(), lrow[i].end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            int l = 0;
            for (auto& e : lrow[i]) l = std::max(l, lev[e.first]);
            lev[i] = l + 1;
            dmax = std::max(dmax, l + 1);
            if (static_cast<int>(levels.size()) <= l + 1) levels.resize(l + 2);
            levels[l + 1].push_back(Row{i, std::move(lrow[i]), T(0.0)});
        }
        depth_f = dmax;
        {
            auto steps = merge_levels(levels, false, n, merge);
            steps_f = static_cast<int>(steps.size());
            if (!emit(out, steps, false)) return false;
        }
        // U: row i, columns j = i+1..i+kv in decreasing order, then the diagonal
        std::fill(lev.begin(), lev.end(), 0);
        levels.assign(1, {});
        dmax = 0;
        for (int i = n - 1; i >= 0; --i) {
            Row R{i, {}, T(1.0) / A(i, i)};             // reciprocal: a multiply on the device
            int l = 0;
            for (int j = std::min(n - 1, i + kv); j > i; --j) {
                const T& v = A(i, j);
                if (v == T(0.0)) continue;
                R.e.push_back({j, v});
                l = std::max(l, lev[j]);
            }
            lev[i] = l + 1;
            dmax = std::max(dmax, l + 1);
            if (static_cast<int>(levels.size()) <= l + 1) levels.resize(l + 2);
            levels[l + 1].push_back(std::move(R));
        }
        depth_b = dmax;
        auto steps = merge_levels(levels, true, n, merge);
        steps_b = static_cast<int>(steps.size());
        return emit(out, steps, true);
    }
    // Transposed stream (k_solve_lvl_T): rows of U^T (with the diagonal, increasing order), then
    // rows of L'^T (unit, decreasing order); fpos[rho] = position of row rho after the
    // interchanges (the F of build()). Returns false if a chunk cannot fit the ring.
    bool build_T(const T* ab, const int* ip, std::vector<unsigned long long>& out, int* fpos,
                 int& depth_u, int& depth_l, bool merge, int& steps_u, int& steps_l) const {
        const int kv = kl + ku;
        auto A = [&](int i, int j) -> const T& { return ab[std::size_t(j) * ldab + kv + i - j]; };
        std::vector<int> F(n);
        for (int i = 0; i < n; ++i) F[i] = i;
        std::vector<std::vector<std::pair<int, T>>> lcol(n);   // L'^T row k: (row r of L', L'_rk)
        for (int k = n - 1; k >= 0; --k) {
            const int lm = std::min(kl, n - 1 - k);
            for (int i = 1; i <= lm; ++i) {
                const T& v = A(k + i, k);
                if (v != T(0.0)) lcol[k].push_back({F[k + i], v});
            }
            std::swap(F[k], F[ip[k]]);
        }
        for (int rho = 0; rho < n; ++rho) fpos[rho] = F[rho];
        // U^T: row j = column j of U above the diagonal, then 1 / U_jj
        std::vector<int> lev(n, 0);
        std::vector<std::vector<Row>> levels(1);
        int dmax = 0;
        for (int j = 0; j < n; ++j) {
            Row R{j, {}, T(1.0) / A(j, j)};
            int l = 0;
            for (int i = std::max(0, j - kv); i < j; ++i) {
                const T& v = A(i, j);
                if (v == T(0.0)) continue;
                R.e.push_back({i, v});
                l = std::max(l, lev[i]);
            }
            lev[j] = l + 1;
            dmax = std::max(dmax, l + 1);
            if (static_cast<int>(levels.size()) <= l + 1) levels.resize(l + 2);
            levels[l + 1].push_back(std::move(R));
        }
        depth_u = dmax;
        {
            auto steps = merge_levels(levels, true, n, merge);
            steps_u = static_cast<int>(steps.size());
            if (!emit(out, steps, true)) return false;
        }
        // L'^T: row k = column k of L' (rows r > k), unit diagonal; rows without entries skipped
        std::fill(lev.begin(), lev.end(), 0);
        levels.assign(1, {});
        dmax = 0;
        for (int k = n - 1; k >= 0; --k) {
            if (lcol[k].empty()) continue;
            std::sort(lcol[k].begin(), lcol[k].end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            int l = 0;
            for (auto& e : lcol[k]) l = std::max(l, lev[e.first]);
            lev[k] = l + 1;
            dmax = std::max(dmax, l + 1);
            if (static_cast<int>(levels.size()) <= l + 1) levels.resize(l + 2);
            levels[l + 1].push_back(Row{k, std::move(lcol[k]), T(0.0)});
        }
        depth_l = dmax;
        auto steps = merge_levels(levels, false, n, merge);
        steps_l = static_cast<int>(steps.size());
        return emit(out, steps, false);
    }
};

struct EventTimer {
    cudaEvent_t a, b;
    EventTimer() { CN_CUDA(cudaEventCreate(&a)); CN_CUDA(cudaEventCreate(&b)); CN_CUDA(cudaEventRecord(a)); }
    double stop() {
        CN_CUDA(cudaEventRecord(b));
        CN_CUDA(cudaEventSynchronize(b));
        float ms = 0;
        CN_CUDA(cudaEventElapsedTime(&ms, a, b));
        return ms;
    }
    ~EventTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }
};

}  // namespace

template <class T>
BandedBatch<T>::BandedBatch(const std::vector<Csr<T>>& mats, const std::vector<int>& border_rows,
                            bool pivot, bool level_schedule, bool level_merge, bool col_border)
    : pivot_(pivot), lvl_merge_(level_merge), col_border_(col_border) {
    if (mats.empty()) throw std::invalid_argument("BandedBatch: empty batch");
    n_ = mats[0].n;
    batch_ = static_cast<int>(mats.size());
    r_ = static_cast<int>(border_rows.size());
    if (r_ > MAXR)
        throw std::invalid_argument("BandedBatch: border rank " + std::to_string(r_) +
                                    " > " + std::to_string(MAXR) + " not supported on the device");
    int kl = 0, ku = 0;
    for (const auto& A : mats) {
        if (A.n != n_) throw std::invalid_argument("BandedBatch: every mode must have the same size n");
        int a, b;
        measure_band(A, border_rows, a, b, col_border_);
        kl = std::max(kl, a);
        ku = std::max(ku, b);
    }
    kl_ = kl;
    ku_ = ku;
    ldab_ = 2 * kl_ + ku_ + 1;
    const std::size_t nb = std::size_t(ldab_) * n_, nv = std::size_t(r_) * n_;
    std::vector<T> ab(nb * batch_), vt(nv * batch_);
    std::vector<int> perm;
    std::vector<T> hxval, ha00;
    if (col_border_) hxptr_.assign(std::size_t(batch_) * (r_ + 1), 0);
    for (int m = 0; m < batch_; ++m) {
        auto Lo = make_bordered_layout(mats[m], border_rows, kl_, ku_, col_border_);
        std::copy(Lo.ab.begin(), Lo.ab.end(), ab.begin() + m * nb);
        std::copy(Lo.Vt.begin(), Lo.Vt.end(), vt.begin() + m * nv);
        if (col_border_) {
            const long long base = static_cast<long long>(hxrow_.size());
            for (int k = 0; k <= r_; ++k) hxptr_[std::size_t(m) * (r_ + 1) + k] = base + Lo.xptr[k];
            hxrow_.insert(hxrow_.end(), Lo.xrow.begin(), Lo.xrow.end());
            hxval.insert(hxval.end(), Lo.xval.begin(), Lo.xval.end());
            ha00.insert(ha00.end(), Lo.A00.begin(), Lo.A00.end());
        }
        if (m == 0) perm = std::move(Lo.perm);
    }
    const auto D = MemorySpace::CUDA_DEVICE;
    ab_ = Buffer<T>(ab.size(), D);
    copy_from_host(ab_, ab.data(), ab.size());
    perm_ = Buffer<int>(n_, D);
    copy_from_host(perm_, perm.data(), perm.size());
    ipiv_ = Buffer<int>(std::size_t(batch_) * n_, D);
    if (r_ > 0) {
        Vt_ = Buffer<T>(vt.size(), D);
        copy_from_host(Vt_, vt.data(), vt.size());
        W_ = Buffer<T>(nv * batch_, D);
        S_ = Buffer<T>(std::size_t(r_) * r_ * batch_, D);
        spiv_ = Buffer<int>(std::size_t(r_) * batch_, D);
    }
    if (col_border_ && r_ > 0) {
        x_nnz_ = static_cast<int>(hxrow_.size());
        if (hxrow_.empty()) { hxrow_.push_back(0); hxval.push_back(T(0.0)); }   // non-null buffers
        xptr_ = Buffer<long long>(hxptr_.size(), D);
        copy_from_host(xptr_, hxptr_.data(), hxptr_.size());
        xrow_ = Buffer<int>(hxrow_.size(), D);
        copy_from_host(xrow_, hxrow_.data(), hxrow_.size());
        xval_ = Buffer<T>(hxval.size(), D);
        copy_from_host(xval_, hxval.data(), hxval.size());
        A00_ = Buffer<T>(ha00.size(), D);
        copy_from_host(A00_, ha00.data(), ha00.size());
    } else {
        col_border_ = false;                         // nothing to move: plain row border
    }
    Buffer<int> info(batch_, D);
    Buffer<double> cond(batch_, D);
    {
        EventTimer tm;
        k_factor<dev_t<T>><<<batch_, WARP>>>(dptr(ab_.data()), ipiv_.data(), dptr(Vt_.data()),
                                             dptr(W_.data()), dptr(S_.data()), spiv_.data(),
                                             info.data(), cond.data(), n_, kl_, ku_, ldab_, r_, pivot_,
                                             xptr_ptr(), xrow_.data(), dptr(xval_.data()),
                                             dptr(A00_.data()));
        CN_LAUNCH_CHECK("k_factor");
        factor_ms_ = tm.stop();
    }
    std::vector<int> hinfo(batch_);
    std::vector<double> hcond(batch_);
    copy_to_host(hinfo.data(), info, batch_);
    copy_to_host(hcond.data(), cond, batch_);
    for (int m = 0; m < batch_; ++m) {
        if (hinfo[m] > 0)
            throw std::runtime_error(
                "BandedBatch: banded part of mode " + std::to_string(m) + " is singular at pivot " +
                std::to_string(hinfo[m]) + (pivot_ ? "" : " (no pivoting was requested)") +
                "; the non-border rows must form an invertible banded operator");
        if (hinfo[m] < 0)
            throw std::runtime_error(
                "BandedBatch: capacitance matrix of mode " + std::to_string(m) +
                " is singular: the boundary rows are not independent of the banded operator");
        cond_max_ = std::max(cond_max_, hcond[m]);
    }
    if (std::size_t(n_) * sizeof(T) > SHARED_Y_MAX) scratch_ = Buffer<T>(std::size_t(batch_) * n_, D);
    if (level_schedule) build_level_schedule();
    else lvl_note_ = "level schedule not requested";
}

// Host rewrite of the band factors into the level-scheduled streams (see k_solve_lvl). Falls back
// to the band kernel (and says why in level_note()) when it does not apply.
template <class T>
void BandedBatch<T>::build_level_schedule() {
    lvl_ = false;
    lvl_bytes_ = 0.0;
    if (n_ >= 32768) { lvl_note_ = "n >= 32768 (int16 row ids)"; return; }
    int dev = 0, optin = 0;
    CN_CUDA(cudaGetDevice(&dev));
    CN_CUDA(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    const std::size_t smem = LVL_RING * sizeof(unsigned long long) + std::size_t(n_) * sizeof(T);
    // The kernel's static shared memory (Woodbury and border staging arrays, several KiB) counts
    // against the same per-block limit: checking with MAXR * sizeof(T) only let n through for
    // which cudaFuncSetAttribute below then failed (an exception instead of the band kernel;
    //).
    cudaFuncAttributes fa{};
    CN_CUDA(cudaFuncGetAttributes(&fa, k_solve_lvl<dev_t<T>>));
    if (smem + fa.sharedSizeBytes > std::size_t(optin)) {
        lvl_note_ = "work vector does not fit the shared memory of a block (" + std::to_string(smem) +
                    " + " + std::to_string(fa.sharedSizeBytes) + " static > " + std::to_string(optin) +
                    " bytes)";
        return;
    }
    const std::size_t nb = std::size_t(ldab_) * n_;
    std::vector<T> ab(nb * batch_);
    std::vector<int> ip(std::size_t(batch_) * n_), perm(n_);
    copy_to_host(ab.data(), ab_, ab.size());
    copy_to_host(ip.data(), ipiv_, ip.size());
    copy_to_host(perm.data(), perm_, perm.size());
    std::vector<unsigned long long> blob;
    std::vector<long long> off(batch_ + 1, 0);
    std::vector<int> gperm(std::size_t(batch_) * n_), fpos(std::size_t(batch_) * n_);
    LvlBuilder<T> B{n_, kl_, ku_, ldab_};
    int df = 0, db = 0, sf = 0, sb = 0;
    for (int m = 0; m < batch_; ++m) {
        int f = 0, b = 0, f2 = 0, b2 = 0;
        off[m] = static_cast<long long>(blob.size());
        if (!B.build(ab.data() + m * nb, ip.data() + std::size_t(m) * n_, perm.data(), blob,
                     gperm.data() + std::size_t(m) * n_, fpos.data() + std::size_t(m) * n_, f, b,
                     lvl_merge_, f2, b2)) {
            lvl_note_ = "a level chunk does not fit the shared-memory ring";
            return;
        }
        df = std::max(df, f);
        db = std::max(db, b);
        sf = std::max(sf, f2);
        sb = std::max(sb, b2);
        blob.resize((blob.size() + 1) & ~std::size_t(1));   // 16-byte aligned mode streams
    }
    off[batch_] = static_cast<long long>(blob.size());
    blob.resize(blob.size() + LVL_RING, 0ull);          // the last segment may be read whole
    const auto D = MemorySpace::CUDA_DEVICE;
    lvl_blob_ = Buffer<unsigned long long>(blob.size(), D);
    copy_from_host(lvl_blob_, blob.data(), blob.size());
    lvl_off_ = Buffer<long long>(off.size(), D);
    copy_from_host(lvl_off_, off.data(), off.size());
    lvl_gperm_ = Buffer<int>(gperm.size(), D);
    copy_from_host(lvl_gperm_, gperm.data(), gperm.size());
    if (r_ > 0) {
        // V^T of every mode without its zeros (boundary rows touch one variable: ~n/8 entries)
        const std::size_t nv = std::size_t(r_) * n_;
        std::vector<T> vt(nv * batch_);
        copy_to_host(vt.data(), Vt_, vt.size());
        std::vector<long long> vp(std::size_t(batch_) * (r_ + 1));
        std::vector<int> vc;
        std::vector<T> vv;
        for (int m = 0; m < batch_; ++m)
            for (int k = 0; k < r_; ++k) {
                vp[std::size_t(m) * (r_ + 1) + k] = static_cast<long long>(vc.size());
                for (int i = 0; i < n_; ++i) {
                    const T& v = vt[m * nv + std::size_t(k) * n_ + i];
                    if (v != T(0.0)) { vc.push_back(i | (k << 16)); vv.push_back(v); }
                }
                vp[std::size_t(m) * (r_ + 1) + k + 1] = static_cast<long long>(vc.size());
            }
        if (vc.empty()) { vc.push_back(0); vv.push_back(T(0.0)); }
        lvl_vptr_ = Buffer<long long>(vp.size(), D);
        copy_from_host(lvl_vptr_, vp.data(), vp.size());
        lvl_vcol_ = Buffer<int>(vc.size(), D);
        copy_from_host(lvl_vcol_, vc.data(), vc.size());
        lvl_vval_ = Buffer<T>(vv.size(), D);
        copy_from_host(lvl_vval_, vv.data(), vv.size());
        lvl_bytes_ += static_cast<double>(vv.size()) * (sizeof(T) + 4);
    }
    lvl_smem_ = smem;
    // The limit is an attribute of the kernel, shared by every BandedBatch<T> of the process: only
    // ever raise it. Setting it to this instance's size lowered it below the size of an instance
    // built earlier with a larger n, whose next solve then failed to launch.
    if (static_cast<int>(smem) > fa.maxDynamicSharedSizeBytes)
        CN_CUDA(cudaFuncSetAttribute(k_solve_lvl<dev_t<T>>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(smem)));
    lvl_depth_f_ = df;
    lvl_depth_b_ = db;
    lvl_steps_f_ = sf;
    lvl_steps_b_ = sb;
    lvl_bytes_ += static_cast<double>(off[batch_]) * 8.0;
    lvl_ = true;
    lvl_note_.clear();
}

template <class T>
std::vector<long long> BandedBatch<T>::probe_level_solve() const {
    if (!lvl_) return {};
    const auto D = MemorySpace::CUDA_DEVICE;
    Buffer<long long> pb(std::size_t(batch_) * 10, D);
    Buffer<T> b(std::size_t(batch_) * n_, D);
    std::vector<T> ones(std::size_t(batch_) * n_, T(1.0));
    copy_from_host(b, ones.data(), ones.size());
    // the probe pointer is reset on every exit: a failed launch used to leave lvl_probe_ pointing
    // at the freed buffer, and the next solve wrote its clock stamps there
    struct Reset { long long*& p; ~Reset() { p = nullptr; } } reset{lvl_probe_};
    lvl_probe_ = pb.data();
    solve(b.data(), nullptr);
    CN_CUDA(cudaDeviceSynchronize());
    std::vector<long long> out(std::size_t(batch_) * 10);
    copy_to_host(out.data(), pb, out.size());
    return out;
}

template <class T>
void BandedBatch<T>::solve(T* d_b, void* stream) const {
    solve_mapped(d_b, nullptr, d_b, nullptr, stream);
}

template <class T>
void BandedBatch<T>::solve_mapped(const T* d_rhs, const long long* d_rows, T* d_x,
                                  const long long* d_cols, void* stream) const {
    launch_solve(Io<dev_t<T>>{dptr(d_rhs), d_rows, dptr(d_x), d_cols}, stream);
}

template <class T>
void BandedBatch<T>::solve_split(const double* d_re, const long long* d_rows_c,
                                 const long long* d_rows_s, double* d_x, const long long* d_cols_c,
                                 const long long* d_cols_s, void* stream) const {
    if constexpr (std::is_same_v<T, double>) {
        throw std::invalid_argument("BandedBatch.solve_split: complex systems only");
    } else {
        if (!d_re || !d_rows_c || !d_rows_s || !d_x || !d_cols_c || !d_cols_s)
            throw std::invalid_argument("BandedBatch.solve_split: every pointer and map is required");
        launch_solve(Io<dev_t<T>>{nullptr, d_rows_c, nullptr, d_cols_c, d_re, d_rows_s, d_x, d_cols_s},
                     stream);
    }
}

template <class T>
template <class IO>
void BandedBatch<T>::launch_solve(const IO& io, void* stream) const {
    auto st = static_cast<cudaStream_t>(stream);
    if (lvl_) {
        k_solve_lvl<dev_t<T>><<<batch_, WARP, lvl_smem_, st>>>(
            lvl_blob_.data(), lvl_off_.data(), lvl_gperm_.data(), lvl_vptr_.data(), lvl_vcol_.data(),
            dptr(lvl_vval_.data()), dptr(W_.data()), dptr(S_.data()), spiv_.data(), io, n_, r_,
            lvl_probe_, col_border_);
        CN_LAUNCH_CHECK("k_solve_lvl");
        return;
    }
    if (band_released_) throw std::runtime_error("BandedBatch: band factors released");
    bool shared = std::size_t(n_) * sizeof(T) <= SHARED_Y_MAX;
    std::size_t smem = shared ? std::size_t(n_) * sizeof(T) : 0;
    k_solve<dev_t<T>><<<batch_, WARP, smem, st>>>(dptr(ab_.data()), ipiv_.data(), dptr(Vt_.data()),
                                              dptr(W_.data()), dptr(S_.data()), spiv_.data(),
                                              perm_.data(), io, dptr(scratch_.data()), n_, kl_, ku_,
                                              ldab_, r_, shared, col_border_);
    CN_LAUNCH_CHECK("k_solve");
}

template <class T>
void BandedBatch<T>::prepare_transpose() const {
    if (tr_ready_) return;
    if (band_released_) throw std::runtime_error("BandedBatch: band factors released");
    if (r_ > 0) {
        const auto D = MemorySpace::CUDA_DEVICE;
        Wt_ = Buffer<T>(std::size_t(r_) * n_ * batch_, D);
        St_ = Buffer<T>(std::size_t(r_) * r_ * batch_, D);
        stpiv_ = Buffer<int>(std::size_t(r_) * batch_, D);
        Buffer<int> info(batch_, D);
        k_prepare_T<dev_t<T>><<<batch_, WARP>>>(dptr(ab_.data()), ipiv_.data(), dptr(Vt_.data()),
                                                dptr(Wt_.data()), dptr(St_.data()), stpiv_.data(),
                                                info.data(), n_, kl_, ku_, ldab_, r_, xptr_ptr(),
                                                dptr(W_.data()), dptr(A00_.data()));
        CN_LAUNCH_CHECK("k_prepare_T");
        std::vector<int> hinfo(batch_);
        copy_to_host(hinfo.data(), info, batch_);
        for (int m = 0; m < batch_; ++m)
            if (hinfo[m])
                throw std::runtime_error("BandedBatch: transposed capacitance matrix of mode " +
                                         std::to_string(m) + " is singular");
    }
    tr_ready_ = true;
}

template <class T>
void BandedBatch<T>::build_transpose_schedule() {
    if (lvl_t_) return;
    if (!lvl_) { lvl_t_note_ = "no forward level schedule (" + lvl_note_ + ")"; return; }
    if (col_border_) { lvl_t_note_ = "column border (transposed correction on the band kernel only)"; return; }
    if (band_released_) throw std::runtime_error("BandedBatch: band factors already released");
    prepare_transpose();                                 // Wt, St from the band factors
    const std::size_t nb = std::size_t(ldab_) * n_;
    std::vector<T> ab(nb * batch_);
    std::vector<int> ip(std::size_t(batch_) * n_);
    copy_to_host(ab.data(), ab_, ab.size());
    copy_to_host(ip.data(), ipiv_, ip.size());
    std::vector<unsigned long long> blob;
    std::vector<long long> off(batch_ + 1, 0);
    std::vector<int> fpos(std::size_t(batch_) * n_);
    LvlBuilder<T> B{n_, kl_, ku_, ldab_};
    int du = 0, dl = 0, su = 0, sl = 0;
    for (int m = 0; m < batch_; ++m) {
        int a = 0, b = 0, a2 = 0, b2 = 0;
        off[m] = static_cast<long long>(blob.size());
        if (!B.build_T(ab.data() + m * nb, ip.data() + std::size_t(m) * n_, blob,
                       fpos.data() + std::size_t(m) * n_, a, b, lvl_merge_, a2, b2)) {
            lvl_t_note_ = "a transposed level chunk does not fit the shared-memory ring";
            return;
        }
        du = std::max(du, a);
        dl = std::max(dl, b);
        su = std::max(su, a2);
        sl = std::max(sl, b2);
        blob.resize((blob.size() + 1) & ~std::size_t(1));
    }
    off[batch_] = static_cast<long long>(blob.size());
    blob.resize(blob.size() + LVL_RING, 0ull);
    const auto D = MemorySpace::CUDA_DEVICE;
    lvl_t_blob_ = Buffer<unsigned long long>(blob.size(), D);
    copy_from_host(lvl_t_blob_, blob.data(), blob.size());
    lvl_t_off_ = Buffer<long long>(off.size(), D);
    copy_from_host(lvl_t_off_, off.data(), off.size());
    lvl_fpos_ = Buffer<int>(fpos.size(), D);
    copy_from_host(lvl_fpos_, fpos.data(), fpos.size());
    // same shared memory as the forward kernel (ring + work vector)
    cudaFuncAttributes fa{};
    CN_CUDA(cudaFuncGetAttributes(&fa, k_solve_lvl_T<dev_t<T>>));
    if (static_cast<int>(lvl_smem_) > fa.maxDynamicSharedSizeBytes)
        CN_CUDA(cudaFuncSetAttribute(k_solve_lvl_T<dev_t<T>>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(lvl_smem_)));
    lvl_t_depth_u_ = du;
    lvl_t_depth_l_ = dl;
    lvl_t_steps_u_ = su;
    lvl_t_steps_l_ = sl;
    lvl_t_bytes_ = static_cast<double>(off[batch_]) * 8.0;
    lvl_t_ = true;
    lvl_t_note_.clear();
}

template <class T>
void BandedBatch<T>::release_band() {
    if (band_released_) return;
    if (!lvl_ || !lvl_t_)
        throw std::runtime_error("BandedBatch.release_band: needs the forward and the transposed level "
                                 "schedules (the band kernels would have no factors left)");
    ab_ = Buffer<T>();
    ipiv_ = Buffer<int>();
    Vt_ = Buffer<T>();
    scratch_ = Buffer<T>();
    band_released_ = true;
}

template <class T>
void BandedBatch<T>::solve_transpose(T* d_b, void* stream) const {
    if (lvl_t_) {
        k_solve_lvl_T<dev_t<T>><<<batch_, WARP, lvl_smem_, static_cast<cudaStream_t>(stream)>>>(
            lvl_t_blob_.data(), lvl_t_off_.data(), lvl_fpos_.data(), perm_.data(), dptr(Wt_.data()),
            dptr(St_.data()), stpiv_.data(), dptr(d_b), n_, r_);
        CN_LAUNCH_CHECK("k_solve_lvl_T");
        return;
    }
    if (band_released_) throw std::runtime_error("BandedBatch: band factors released");
    prepare_transpose();
    bool shared = std::size_t(n_) * sizeof(T) <= SHARED_Y_MAX;
    std::size_t smem = shared ? std::size_t(n_) * sizeof(T) : 0;
    k_solve_T<dev_t<T>><<<batch_, WARP, smem, static_cast<cudaStream_t>(stream)>>>(
        dptr(ab_.data()), ipiv_.data(), dptr(Wt_.data()), dptr(St_.data()), stpiv_.data(),
        perm_.data(), dptr(d_b), dptr(scratch_.data()), n_, kl_, ku_, ldab_, r_, shared, xptr_ptr(),
        xrow_.data(), dptr(xval_.data()));
    CN_LAUNCH_CHECK("k_solve_T");
}

template class BandedBatch<double>;
template class BandedBatch<cplx>;

DenseRefResult dense_lu_reference(const cplx* A, const cplx* b, int batch, int n, int repeats) {
    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed");
    const std::size_t nn = std::size_t(n) * n;
    // column-major copies of the row-major host matrices
    std::vector<cplx> Ac(nn * batch);
    for (int m = 0; m < batch; ++m)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) Ac[m * nn + std::size_t(j) * n + i] = A[m * nn + std::size_t(i) * n + j];
    const auto D = MemorySpace::CUDA_DEVICE;
    Buffer<cplx> dA(nn * batch, D), dB(std::size_t(n) * batch, D), dB0(std::size_t(n) * batch, D);
    copy_from_host(dA, Ac.data(), Ac.size());
    copy_from_host(dB0, b, std::size_t(n) * batch);
    std::vector<cuDoubleComplex*> pa(batch), pb(batch);
    for (int m = 0; m < batch; ++m) {
        pa[m] = reinterpret_cast<cuDoubleComplex*>(dA.data() + m * nn);
        pb[m] = reinterpret_cast<cuDoubleComplex*>(dB.data() + std::size_t(m) * n);
    }
    Buffer<cuDoubleComplex*> dpa(batch, D), dpb(batch, D);
    copy_from_host(dpa, pa.data(), batch);
    copy_from_host(dpb, pb.data(), batch);
    Buffer<int> piv(std::size_t(n) * batch, D), info(batch, D);
    DenseRefResult res;
    {
        EventTimer tm;
        if (cublasZgetrfBatched(h, n, dpa.data(), n, piv.data(), info.data(), batch) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasZgetrfBatched failed");
        res.factor_ms = tm.stop();
    }
    int hinfo = 0;
    double total = 0.0;
    for (int k = 0; k < std::max(1, repeats); ++k) {
        copy(dB, dB0);
        EventTimer tm;
        if (cublasZgetrsBatched(h, CUBLAS_OP_N, n, 1, dpa.data(), n, piv.data(), dpb.data(), n,
                                &hinfo, batch) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasZgetrsBatched failed");
        total += tm.stop();
    }
    res.solve_ms = total / std::max(1, repeats);
    res.x.resize(std::size_t(n) * batch);
    copy_to_host(res.x.data(), dB, res.x.size());
    cublasDestroy(h);
    return res;
}

}  // namespace conatus::cuda
