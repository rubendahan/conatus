// Batched grid <-> coefficient transforms on the device (cuFFT), spec §49–50.
// Conventions are exactly those of the host backend (core/src/transforms/transforms.cpp) and of
// python/conatus/reference/transforms.py; results agree to rounding (not bitwise: different FFTs).
// Chebyshev DCT-II / DCT-III go through a real FFT of length 2M on the even extension (valid for any
// M), with the pre/post twiddles in one kernel each. cuFFT plans are cached by (kind, n, batch);
// cuFFT is run-to-run deterministic for a fixed plan and device.
#include <cufft.h>

#include <map>
#include <mutex>
#include <tuple>

#include "common.cuh"
#include "conatus/cuda.hpp"

namespace conatus::cuda {

namespace {

using dz = ::cuda::std::complex<double>;

enum class Kind { Z2Z, D2Z, Z2D };
using Key = std::tuple<Kind, std::size_t, std::size_t>;

std::mutex g_mutex;
std::map<Key, cufftHandle>& cache() {
    static auto* c = new std::map<Key, cufftHandle>();   // leaked: no teardown after the driver
    return *c;
}

cufftHandle get_plan(Kind kind, std::size_t n, std::size_t batch) {
    std::lock_guard<std::mutex> lock(g_mutex);
    Key key{kind, n, batch};
    auto it = cache().find(key);
    if (it != cache().end()) return it->second;
    cufftHandle p;
    int ni = static_cast<int>(n);
    cufftType t = kind == Kind::Z2Z ? CUFFT_Z2Z : (kind == Kind::D2Z ? CUFFT_D2Z : CUFFT_Z2D);
    CN_CUFFT(cufftPlanMany(&p, 1, &ni, nullptr, 1, 0, nullptr, 1, 0, t, static_cast<int>(batch)));
    cache()[key] = p;
    return p;
}

void check_sizes(std::size_t M, std::size_t N, const char* who) {
    if (N > M) throw std::invalid_argument(std::string(who) + ": need M >= N");
}
void check_even(std::size_t N, const char* who) {
    if (N % 2) throw std::invalid_argument(std::string(who) + ": Fourier sizes must be even");
}

__device__ inline double sgn(std::size_t k) { return (k & 1) ? -1.0 : 1.0; }

// ---- Chebyshev --------------------------------------------------------------------------------
// Component access: for real T the value itself, for complex T its real (c=0) or imaginary (c=1)
// part. Complex rows are transformed as 2*batch real rows (the DCT is real-linear).
__device__ inline double comp(double x, int) { return x; }
__device__ inline double comp(dz z, int c) { return c ? z.imag() : z.real(); }

template <class T>
__global__ void k_cheb_extend(const T* f, double* e, std::size_t batch, std::size_t M, int ncomp) {
    std::size_t idx = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    std::size_t total = batch * ncomp * M;
    if (idx >= total) return;
    std::size_t j = idx % M, row = idx / M;             // row in [0, ncomp*batch)
    int c = static_cast<int>(row / batch);
    std::size_t b = row % batch;
    double v = comp(f[b * M + j], c);
    double* er = e + row * 2 * M;
    er[j] = v;
    er[2 * M - 1 - j] = v;
}

// coef_k = Re(e^{-i pi k / 2M} Y_k) * sgn(k) / M, halved at k = 0.
__global__ void k_cheb_post(const dz* Y, double* out, std::size_t rows, std::size_t M,
                            std::size_t N) {
    std::size_t idx = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (idx >= rows * N) return;
    std::size_t k = idx % N, row = idx / N;
    double th = -M_PI * double(k) / (2.0 * double(M));
    dz y = Y[row * (M + 1) + k];
    double v = (y.real() * cos(th) - y.imag() * sin(th)) * sgn(k) / double(M);
    if (k == 0) v *= 0.5;
    out[row * N + k] = v;
}

// Z_k = y_k e^{+i pi k / 2M}, y_k = c_k sgn(k) (doubled at k = 0), zero for N <= k <= M.
template <class T>
__global__ void k_cheb_pre(const T* c, dz* Z, std::size_t batch, std::size_t N, std::size_t M,
                           int ncomp) {
    std::size_t idx = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    std::size_t total = batch * ncomp * (M + 1);
    if (idx >= total) return;
    std::size_t k = idx % (M + 1), row = idx / (M + 1);
    int cc = static_cast<int>(row / batch);
    std::size_t b = row % batch;
    dz z(0.0, 0.0);
    if (k < N) {
        double y = comp(c[b * N + k], cc) * sgn(k);
        if (k == 0) y *= 2.0;
        double th = M_PI * double(k) / (2.0 * double(M));
        z = dz(y * cos(th), y * sin(th));
    }
    Z[row * (M + 1) + k] = z;
}

__device__ inline void store(double* out, std::size_t i, int, double v) { out[i] = v; }
__device__ inline void store(dz* out, std::size_t i, int c, double v) {
    // two separate kernels never race on the same element: component c is written by rows of
    // block c only; write through the underlying doubles.
    reinterpret_cast<double*>(out)[2 * i + c] = v;
}

template <class T>
__global__ void k_cheb_gather(const double* src, T* out, std::size_t batch, std::size_t len,
                              std::size_t src_len, double scale, int ncomp) {
    std::size_t idx = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    std::size_t total = batch * ncomp * len;
    if (idx >= total) return;
    std::size_t j = idx % len, row = idx / len;
    int c = static_cast<int>(row / batch);
    std::size_t b = row % batch;
    store(out, b * len + j, c, scale * src[row * src_len + j]);
}

template <class T> constexpr int ncomp_of() { return std::is_same_v<T, double> ? 1 : 2; }

// ---- Fourier ------------------------------------------------------------------------------------
__global__ void k_scale_real(double* x, std::size_t n, double s) {
    std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < n) x[i] *= s;
}
__global__ void k_scale_cplx(dz* x, std::size_t n, double s) {
    std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < n) x[i] *= s;
}

__global__ void k_realfourier_post(const dz* F, double* c, std::size_t batch, std::size_t M,
                                   std::size_t N) {
    std::size_t idx = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    std::size_t K = N / 2, nc = M / 2 + 1;
    if (idx >= batch * K) return;
    std::size_t k = idx % K, b = idx / K;
    dz Fk = F[b * nc + k];
    double* cb = c + b * N;
    if (k == 0) { cb[0] = Fk.real() / double(M); cb[1] = 0.0; }
    else { cb[2 * k] = 2.0 * Fk.real() / double(M); cb[2 * k + 1] = 2.0 * Fk.imag() / double(M); }
}

__global__ void k_realfourier_pre(const double* c, dz* F, std::size_t batch, std::size_t N,
                                  std::size_t M) {
    std::size_t idx = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    std::size_t K = N / 2, nc = M / 2 + 1;
    if (idx >= batch * nc) return;
    std::size_t k = idx % nc, b = idx / nc;
    const double* cb = c + b * N;
    dz v(0.0, 0.0);
    if (k == 0) v = dz(cb[0] * double(M), 0.0);
    else if (k < K) v = dz(cb[2 * k], cb[2 * k + 1]) * (0.5 * double(M));
    F[b * nc + k] = v;
}

__global__ void k_complexfourier_post(const dz* F, dz* c, std::size_t batch, std::size_t M,
                                      std::size_t N) {
    std::size_t idx = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    std::size_t K = N / 2;
    if (idx >= batch * N) return;
    std::size_t k = idx % N, b = idx / N;
    const dz* Fb = F + b * M;
    dz v(0.0, 0.0);
    if (k < K) v = Fb[k] / double(M);
    else if (k > K) v = Fb[M - K + (k - K)] / double(M);
    c[b * N + k] = v;
}

__global__ void k_complexfourier_pre(const dz* c, dz* F, std::size_t batch, std::size_t N,
                                     std::size_t M) {
    std::size_t idx = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    std::size_t K = N / 2;
    if (idx >= batch * M) return;
    std::size_t j = idx % M, b = idx / M;
    const dz* cb = c + b * N;
    dz v(0.0, 0.0);
    if (j < K) v = cb[j] * double(M);
    else if (j > M - K) v = cb[K + (j - (M - K))] * double(M);
    F[b * M + j] = v;
}

constexpr unsigned TPB = 256;

}  // namespace

std::size_t plan_cache_size() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return cache().size();
}

void fft_c2c(const cplx* d_in, cplx* d_out, std::size_t batch, std::size_t n, bool forward) {
    // cuFFT may overwrite its input for some sizes: work on a copy (explicit device-to-device).
    Buffer<cplx> a(batch * n, MemorySpace::CUDA_DEVICE);
    CN_CUDA(cudaMemcpy(a.data(), d_in, sizeof(cplx) * batch * n, cudaMemcpyDeviceToDevice));
    auto* ai = reinterpret_cast<cufftDoubleComplex*>(a.data());
    auto* ao = reinterpret_cast<cufftDoubleComplex*>(d_out);
    CN_CUFFT(cufftExecZ2Z(get_plan(Kind::Z2Z, n, batch), ai, ao, forward ? CUFFT_FORWARD : CUFFT_INVERSE));
    if (!forward) {
        k_scale_cplx<<<blocks_for(batch * n, TPB), TPB>>>(dptr(d_out), batch * n, 1.0 / double(n));
        CN_LAUNCH_CHECK("k_scale_cplx");
    }
}

void rfft(const double* d_in, cplx* d_out, std::size_t batch, std::size_t n) {
    Buffer<double> a(batch * n, MemorySpace::CUDA_DEVICE);
    CN_CUDA(cudaMemcpy(a.data(), d_in, sizeof(double) * batch * n, cudaMemcpyDeviceToDevice));
    CN_CUFFT(cufftExecD2Z(get_plan(Kind::D2Z, n, batch), a.data(),
                          reinterpret_cast<cufftDoubleComplex*>(d_out)));
}

void irfft(const cplx* d_in, double* d_out, std::size_t batch, std::size_t n) {
    std::size_t nc = n / 2 + 1;
    Buffer<cplx> a(batch * nc, MemorySpace::CUDA_DEVICE);   // C2R destroys its input
    CN_CUDA(cudaMemcpy(a.data(), d_in, sizeof(cplx) * batch * nc, cudaMemcpyDeviceToDevice));
    CN_CUFFT(cufftExecZ2D(get_plan(Kind::Z2D, n, batch),
                          reinterpret_cast<cufftDoubleComplex*>(a.data()), d_out));
    k_scale_real<<<blocks_for(batch * n, TPB), TPB>>>(d_out, batch * n, 1.0 / double(n));
    CN_LAUNCH_CHECK("k_scale_real");
}

template <class T>
void cheb_forward(const T* d_grid, T* d_coef, std::size_t batch, std::size_t M, std::size_t N) {
    check_sizes(M, N, "cheb_forward");
    constexpr int nc = ncomp_of<T>();
    std::size_t rows = batch * nc;
    Buffer<double> e(rows * 2 * M, MemorySpace::CUDA_DEVICE);
    Buffer<cplx> Y(rows * (M + 1), MemorySpace::CUDA_DEVICE);
    Buffer<double> out(rows * N, MemorySpace::CUDA_DEVICE);
    k_cheb_extend<<<blocks_for(rows * M, TPB), TPB>>>(dptr(d_grid), e.data(), batch, M, nc);
    CN_LAUNCH_CHECK("k_cheb_extend");
    CN_CUFFT(cufftExecD2Z(get_plan(Kind::D2Z, 2 * M, rows), e.data(),
                          reinterpret_cast<cufftDoubleComplex*>(Y.data())));
    k_cheb_post<<<blocks_for(rows * N, TPB), TPB>>>(dptr(Y.data()), out.data(), rows, M, N);
    CN_LAUNCH_CHECK("k_cheb_post");
    k_cheb_gather<<<blocks_for(rows * N, TPB), TPB>>>(out.data(), dptr(d_coef), batch, N, N, 1.0, nc);
    CN_LAUNCH_CHECK("k_cheb_gather");
}

template <class T>
void cheb_backward(const T* d_coef, T* d_grid, std::size_t batch, std::size_t N, std::size_t M) {
    check_sizes(M, N, "cheb_backward");
    constexpr int nc = ncomp_of<T>();
    std::size_t rows = batch * nc;
    Buffer<cplx> Z(rows * (M + 1), MemorySpace::CUDA_DEVICE);
    Buffer<double> z(rows * 2 * M, MemorySpace::CUDA_DEVICE);
    k_cheb_pre<<<blocks_for(rows * (M + 1), TPB), TPB>>>(dptr(d_coef), dptr(Z.data()), batch, N, M, nc);
    CN_LAUNCH_CHECK("k_cheb_pre");
    CN_CUFFT(cufftExecZ2D(get_plan(Kind::Z2D, 2 * M, rows),
                          reinterpret_cast<cufftDoubleComplex*>(Z.data()), z.data()));
    k_cheb_gather<<<blocks_for(rows * M, TPB), TPB>>>(z.data(), dptr(d_grid), batch, M, 2 * M, 0.5, nc);
    CN_LAUNCH_CHECK("k_cheb_gather");
}

template void cheb_forward<double>(const double*, double*, std::size_t, std::size_t, std::size_t);
template void cheb_forward<cplx>(const cplx*, cplx*, std::size_t, std::size_t, std::size_t);
template void cheb_backward<double>(const double*, double*, std::size_t, std::size_t, std::size_t);
template void cheb_backward<cplx>(const cplx*, cplx*, std::size_t, std::size_t, std::size_t);

void realfourier_forward(const double* d_grid, double* d_coef, std::size_t batch, std::size_t M,
                         std::size_t N) {
    check_sizes(M, N, "realfourier_forward");
    check_even(N, "realfourier_forward");
    std::size_t nc = M / 2 + 1;
    Buffer<cplx> F(batch * nc, MemorySpace::CUDA_DEVICE);
    rfft(d_grid, F.data(), batch, M);
    k_realfourier_post<<<blocks_for(batch * (N / 2), TPB), TPB>>>(dptr(F.data()), d_coef, batch, M, N);
    CN_LAUNCH_CHECK("k_realfourier_post");
}

void realfourier_backward(const double* d_coef, double* d_grid, std::size_t batch, std::size_t N,
                          std::size_t M) {
    check_sizes(M, N, "realfourier_backward");
    check_even(N, "realfourier_backward");
    std::size_t nc = M / 2 + 1;
    Buffer<cplx> F(batch * nc, MemorySpace::CUDA_DEVICE);
    k_realfourier_pre<<<blocks_for(batch * nc, TPB), TPB>>>(d_coef, dptr(F.data()), batch, N, M);
    CN_LAUNCH_CHECK("k_realfourier_pre");
    irfft(F.data(), d_grid, batch, M);
}

void complexfourier_forward(const cplx* d_grid, cplx* d_coef, std::size_t batch, std::size_t M,
                            std::size_t N) {
    check_sizes(M, N, "complexfourier_forward");
    check_even(N, "complexfourier_forward");
    Buffer<cplx> F(batch * M, MemorySpace::CUDA_DEVICE);
    fft_c2c(d_grid, F.data(), batch, M, true);
    k_complexfourier_post<<<blocks_for(batch * N, TPB), TPB>>>(dptr(F.data()), dptr(d_coef), batch, M, N);
    CN_LAUNCH_CHECK("k_complexfourier_post");
}

void complexfourier_backward(const cplx* d_coef, cplx* d_grid, std::size_t batch, std::size_t N,
                             std::size_t M) {
    check_sizes(M, N, "complexfourier_backward");
    check_even(N, "complexfourier_backward");
    Buffer<cplx> F(batch * M, MemorySpace::CUDA_DEVICE);
    k_complexfourier_pre<<<blocks_for(batch * M, TPB), TPB>>>(dptr(d_coef), dptr(F.data()), batch, N, M);
    CN_LAUNCH_CHECK("k_complexfourier_pre");
    fft_c2c(F.data(), d_grid, batch, M, false);
}

}  // namespace conatus::cuda
