// IMEX timesteppers on the device, batched over modes.
// Same schemes, coefficients and factor caching as the host integrator (conatus/imex.hpp):
// SBDF1..4 (constant dt, startup orders 1..q) and RK111 / RK222 / RK443 (one factorization for
// RK443, trap P1; L X_0 skipped when H[:,0] = 0, trap P2). The state stays on the device; only the
// scheme coefficients live on the host. F is zero or the pointwise model F = g .* X + h.
#include <algorithm>

#include "common.cuh"
#include "conatus/cuda.hpp"
#include "conatus/imex.hpp"

namespace conatus::cuda {

namespace {
constexpr unsigned TPB = 256;

// y[m, i] = alpha * sum_p A_m[i, p] x[m, p]  (+ y if accumulate); one thread per row, fixed order.
template <class T>
__global__ void k_spmv(const int* rowptr, const int* idx, const T* val, const T* x, T* y, int n,
                       int batch, double alpha, bool accumulate) {
    std::size_t g = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (g >= std::size_t(n) * batch) return;
    std::size_t m = g / n, i = g % n;
    const int* rp = rowptr + m * (n + 1);
    const T* xm = x + m * n;
    T s(0.0);
    for (int p = rp[i]; p < rp[i + 1]; ++p) s += val[p] * xm[idx[p]];
    y[g] = accumulate ? y[g] + T(alpha) * s : T(alpha) * s;
}

template <class T>
__global__ void k_axpy(T* y, const T* x, double a, std::size_t N) {
    std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < N) y[i] += T(a) * x[i];
}

// rhs += c * f - a * mx   (same association as the host SBDF loop)
template <class T>
__global__ void k_sbdf_acc(T* rhs, const T* f, const T* mx, double c, double a, std::size_t N) {
    std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < N) rhs[i] += T(c) * f[i] - T(a) * mx[i];
}

template <class T>
__global__ void k_pointwise(T* out, const T* x, const T* g, const T* h, std::size_t N) {
    std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i >= N) return;
    T v(0.0);
    if (g) v = g[i] * x[i];
    if (h) v += h[i];
    out[i] = v;
}
}  // namespace

template <class T>
struct Imex<T>::DevCsr {
    Buffer<int> rowptr, idx;
    Buffer<T> val;
    DevCsr(const std::vector<Csr<T>>& mats) {
        const int n = mats[0].n;
        std::vector<int> rp, ix;
        std::vector<T> v;
        rp.reserve(mats.size() * (n + 1));
        for (const auto& A : mats) {
            int off = static_cast<int>(ix.size());
            for (int i = 0; i <= n; ++i) rp.push_back(off + A.indptr[i]);
            ix.insert(ix.end(), A.indices.begin(), A.indices.end());
            v.insert(v.end(), A.data.begin(), A.data.end());
        }
        const auto D = MemorySpace::CUDA_DEVICE;
        rowptr = Buffer<int>(rp.size(), D);
        copy_from_host(rowptr, rp.data(), rp.size());
        idx = Buffer<int>(std::max<std::size_t>(ix.size(), 1), D);
        val = Buffer<T>(std::max<std::size_t>(v.size(), 1), D);
        if (!ix.empty()) {
            detail::space_copy(D, idx.data(), MemorySpace::HOST, ix.data(), sizeof(int) * ix.size());
            detail::space_copy(D, val.data(), MemorySpace::HOST, v.data(), sizeof(T) * v.size());
        }
    }
    void matvec(const T* x, T* y, int n, int batch, double alpha = 1.0, bool acc = false) const {
        std::size_t N = std::size_t(n) * batch;
        k_spmv<dev_t<T>><<<blocks_for(N, TPB), TPB>>>(rowptr.data(), idx.data(), dptr(val.data()),
                                                      dptr(x), dptr(y), n, batch, alpha, acc);
        CN_LAUNCH_CHECK("k_spmv");
    }
};

template <class T>
Imex<T>::Imex(const std::vector<Csr<T>>& M, const std::vector<Csr<T>>& L,
              const std::vector<int>& border_rows, const std::string& scheme, double dt, bool pivot)
    : Mh_(M), Lh_(L), border_(border_rows), scheme_(scheme), dt_(dt), pivot_(pivot) {
    if (M.empty() || M.size() != L.size()) throw std::invalid_argument("Imex: M and L batches differ");
    if (!(dt_ > 0)) throw std::invalid_argument("dt must be > 0");
    n_ = M[0].n;
    batch_ = static_cast<int>(M.size());
    for (std::size_t k = 0; k < M.size(); ++k)
        if (M[k].n != n_ || L[k].n != n_) throw std::invalid_argument("Imex: all modes need the same n");
    if (scheme_.rfind("SBDF", 0) == 0) {
        order_ = sbdf_order(scheme_);   // validates (SBDF1..4)
    } else {
        rk_tableau(scheme_);         // validates
    }
    M_ = std::make_unique<DevCsr>(Mh_);
    L_ = std::make_unique<DevCsr>(Lh_);
}

template <class T>
Imex<T>::~Imex() = default;

template <class T>
void Imex<T>::set_pointwise_rhs(const T* host_g, const T* host_h) {
    const std::size_t N = std::size_t(n_) * batch_;
    has_g_ = host_g != nullptr;
    has_h_ = host_h != nullptr;
    if (has_g_) { g_ = Buffer<T>(N, MemorySpace::CUDA_DEVICE); copy_from_host(g_, host_g, N); }
    if (has_h_) { h_ = Buffer<T>(N, MemorySpace::CUDA_DEVICE); copy_from_host(h_, host_h, N); }
}

template <class T>
const BandedBatch<T>& Imex<T>::factor(double alpha, double beta) {
    auto key = std::make_pair(alpha, beta);
    auto it = factors_.find(key);
    if (it != factors_.end()) return *it->second;
    std::vector<Csr<T>> lhs;
    lhs.reserve(batch_);
    for (int m = 0; m < batch_; ++m) lhs.push_back(csr_axpby(T(alpha), Mh_[m], T(beta), Lh_[m]));
    auto f = std::make_unique<BandedBatch<T>>(lhs, border_, pivot_);
    auto& ref = *f;
    factors_.emplace(key, std::move(f));
    return ref;
}

template <class T>
void Imex<T>::eval_F(const T* d_x, T* d_out) const {
    const std::size_t N = std::size_t(n_) * batch_;
    k_pointwise<dev_t<T>><<<blocks_for(N, TPB), TPB>>>(dptr(d_out), dptr(d_x),
                                                       has_g_ ? dptr(g_.data()) : nullptr,
                                                       has_h_ ? dptr(h_.data()) : nullptr, N);
    CN_LAUNCH_CHECK("k_pointwise");
}

template <class T>
double Imex<T>::run(T* d_x, double t, long nsteps) {
    for (long s = 0; s < nsteps; ++s) {
        if (order_) step_sbdf(d_x); else step_rk(d_x);
        t += dt_;
        ++iteration_;
    }
    return t;
}

template <class T>
void Imex<T>::step_sbdf(T* d_x) {
    const std::size_t N = std::size_t(n_) * batch_;
    const auto D = MemorySpace::CUDA_DEVICE;
    Buffer<T> MX(N, D), FX(N, D);
    M_->matvec(d_x, MX.data(), n_, batch_);
    eval_F(d_x, FX.data());
    MXh_.insert(MXh_.begin(), std::move(MX));
    Fh_.insert(Fh_.begin(), std::move(FX));
    if (static_cast<int>(MXh_.size()) > order_) { MXh_.pop_back(); Fh_.pop_back(); }
    int q = std::min<int>(order_, static_cast<int>(MXh_.size()));
    auto [a, c] = sbdf_coefficients(q);
    Buffer<T> rhs(N, D);   // zeroed
    for (int j = 1; j <= q; ++j) {
        k_sbdf_acc<dev_t<T>><<<blocks_for(N, TPB), TPB>>>(dptr(rhs.data()), dptr(Fh_[j - 1].data()),
                                                          dptr(MXh_[j - 1].data()), c[j], a[j] / dt_, N);
        CN_LAUNCH_CHECK("k_sbdf_acc");
    }
    factor(a[0] / dt_, 1.0).solve(rhs.data());
    CN_CUDA(cudaMemcpy(d_x, rhs.data(), sizeof(T) * N, cudaMemcpyDeviceToDevice));
}

template <class T>
void Imex<T>::step_rk(T* d_x) {
    const RKTableau tab = rk_tableau(scheme_);
    const int s = tab.stages();
    const auto& A = tab.A;
    const auto& H = tab.H;
    const std::size_t N = std::size_t(n_) * batch_;
    const auto D = MemorySpace::CUDA_DEVICE;
    std::vector<Buffer<T>> X(s + 1), F(s + 1), LX(s + 1);
    X[0] = Buffer<T>(N, D);
    CN_CUDA(cudaMemcpy(X[0].data(), d_x, sizeof(T) * N, cudaMemcpyDeviceToDevice));
    Buffer<T> MX0(N, D);
    M_->matvec(d_x, MX0.data(), n_, batch_);
    auto needF = [&](int j) { for (int i = j + 1; i <= s; ++i) if (A[i][j] != 0) return true; return false; };
    auto needLX = [&](int j) { for (int i = j + 1; i <= s; ++i) if (H[i][j] != 0) return true; return false; };
    for (int i = 1; i <= s; ++i) {
        int j = i - 1;
        if (needF(j)) { F[j] = Buffer<T>(N, D); eval_F(X[j].data(), F[j].data()); }
        if (needLX(j)) { LX[j] = Buffer<T>(N, D); L_->matvec(X[j].data(), LX[j].data(), n_, batch_); }
        Buffer<T> rhs(N, D);
        copy(rhs, MX0);
        for (int k = 0; k < i; ++k) {
            if (A[i][k] != 0) {
                k_axpy<dev_t<T>><<<blocks_for(N, TPB), TPB>>>(dptr(rhs.data()), dptr(F[k].data()), dt_ * A[i][k], N);
                CN_LAUNCH_CHECK("k_axpy F");
            }
            if (H[i][k] != 0) {
                k_axpy<dev_t<T>><<<blocks_for(N, TPB), TPB>>>(dptr(rhs.data()), dptr(LX[k].data()), -dt_ * H[i][k], N);
                CN_LAUNCH_CHECK("k_axpy LX");
            }
        }
        factor(1.0, dt_ * H[i][i]).solve(rhs.data());
        X[i] = std::move(rhs);
    }
    CN_CUDA(cudaMemcpy(d_x, X[s].data(), sizeof(T) * N, cudaMemcpyDeviceToDevice));
}

template class Imex<double>;
template class Imex<cplx>;

}  // namespace conatus::cuda
