// CUDA backend: plain C++ declarations, implemented in core/src/cuda/*.cu.
// All pointers named d_* are device pointers. Nothing here copies between host and device
// implicitly: callers own device Buffers and use conatus::copy_* explicitly.
#pragma once

#include <complex>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "conatus/banded.hpp"
#include "conatus/memory.hpp"

namespace conatus::cuda {

using cplx = std::complex<double>;

// ---- device / pool -------------------------------------------------------------------------
bool available();                 // compiled with CUDA and at least one device visible
std::string device_name();
void synchronize();

struct PoolStats {
    std::size_t live_bytes = 0, cached_bytes = 0, n_alloc = 0, n_reuse = 0;
};
PoolStats pool_stats();
void pool_trim();                 // release cached (free) device blocks to the driver

// ---- transforms (same conventions as conatus::transforms on the host) --------------------------
// Rows are contiguous (batch, len); cuFFT plans are cached by (kind, n, batch).
void fft_c2c(const cplx* d_in, cplx* d_out, std::size_t batch, std::size_t n, bool forward);
void rfft(const double* d_in, cplx* d_out, std::size_t batch, std::size_t n);
void irfft(const cplx* d_in, double* d_out, std::size_t batch, std::size_t n);
template <class T>
void cheb_forward(const T* d_grid, T* d_coef, std::size_t batch, std::size_t M, std::size_t N);
template <class T>
void cheb_backward(const T* d_coef, T* d_grid, std::size_t batch, std::size_t N, std::size_t M);
void realfourier_forward(const double* d_grid, double* d_coef, std::size_t batch, std::size_t M,
                         std::size_t N);
void realfourier_backward(const double* d_coef, double* d_grid, std::size_t batch, std::size_t N,
                          std::size_t M);
void complexfourier_forward(const cplx* d_grid, cplx* d_coef, std::size_t batch, std::size_t M,
                            std::size_t N);
void complexfourier_backward(const cplx* d_coef, cplx* d_grid, std::size_t batch, std::size_t N,
                             std::size_t M);
std::size_t plan_cache_size();

// ---- bordered banded solves, batched over modes -------------------------------------------
// Same decomposition as conatus::BorderedBanded (shared BorderedLayout): P A = B + U V^T.
// One thread block per mode: banded LU of B (partial pivoting inside the band, or none when the
// caller declares B diagonally dominant), W = B^{-1} U and the r x r capacitance S factored on the
// device; solves are fully on the device. All modes share n, the border rows and a common band.
template <class T>
class BandedBatch {
public:
    // level_schedule: also build the level-scheduled sparse form of the factors (see
    // "Level-scheduled solves" in banded.cu); used by solve() when it applies, else the band kernel.
    // level_merge: process several consecutive levels per warp step, the rows of a step expanded
    // in terms of the values present at its start ("Merged levels" in banded.cu); fewer, fuller
    // steps, same factors, different rounding (products of factor entries formed on the host).
    // col_border: the first r columns leave the band (column border, exact Schur complement on
    // the border; see banded.hpp "Column border").
    BandedBatch(const std::vector<Csr<T>>& mats, const std::vector<int>& border_rows,
                bool pivot = true, bool level_schedule = true, bool level_merge = true,
                bool col_border = false);
    void solve(T* d_b, void* stream = nullptr) const;  // d_b: (batch, n) on the device, in place;
                                                       // stream: cudaStream_t (null = legacy default)
    // Out-of-place solve with index maps fused into the kernel: b_m = d_rhs[d_rows[m*n + i]]
    // (d_rows null: d_rhs[m*n + i]); x_m[i] is written to d_x[d_cols[m*n + i]] (null: d_x[m*n + i]).
    // d_rhs may equal d_x (every read happens before the first write of a mode; modes must not
    // share entries). Maps are int64 (batch, n) device arrays.
    void solve_mapped(const T* d_rhs, const long long* d_rows, T* d_x, const long long* d_cols,
                      void* stream = nullptr) const;
    // Complex systems of real data (complexified RealFourier pairs): b_m[i] = re[rows_c[q]] +
    // i re[rows_s[q]], x[cols_c[q]] = Re x_m[i], x[cols_s[q]] = Im x_m[i] (q = m n + i; maps
    // int64 (batch, n) on the device). Complex T only.
    void solve_split(const double* d_re, const long long* d_rows_c, const long long* d_rows_s,
                     double* d_x, const long long* d_cols_c, const long long* d_cols_s,
                     void* stream = nullptr) const;
    bool level_scheduled() const { return lvl_; }       // solve() uses the level-scheduled kernel
    int level_depth_forward() const { return lvl_depth_f_; }   // max over modes
    int level_depth_backward() const { return lvl_depth_b_; }
    int level_steps_forward() const { return lvl_steps_f_; }   // warp steps executed (max over
    int level_steps_backward() const { return lvl_steps_b_; }  // modes; = depth without merging)
    bool level_merged() const { return lvl_merge_; }
    double level_stream_bytes() const { return lvl_bytes_; }  // factor stream, all modes
    std::string level_note() const { return lvl_note_; }     // why the band kernel is used, if so
    // Diagnostics: one level-scheduled solve of b = 1 with per-block clock64 stamps (start, after
    // the gather, forward, backward, end) and chunk / wait counts; (batch x 10) values.
    std::vector<long long> probe_level_solve() const;
    // A^T x = b in place (plain transpose, no conjugation), same factors: the discrete adjoint path.
    // The transposed Woodbury data (Wt = B^{-T} V, St = I + U^T Wt) are built on first use, or
    // explicitly by prepare_transpose() (required before capturing a CUDA graph).
    void solve_transpose(T* d_b, void* stream = nullptr) const;
    void prepare_transpose() const;
    // Transposed level-scheduled solves: streams of U^T and L'^T without their zeros,
    // built on request after the forward schedule (prepare_transpose included); solve_transpose
    // then uses them. release_band() frees the band LU (and V^T, pivots, scratch) once both
    // schedules exist: the band kernels are refused afterwards.
    void build_transpose_schedule();
    void release_band();
    bool level_transposed() const { return lvl_t_; }
    bool band_released() const { return band_released_; }
    double level_t_stream_bytes() const { return lvl_t_bytes_; }
    std::string level_t_note() const { return lvl_t_note_; }
    int level_t_steps_u() const { return lvl_t_steps_u_; }
    int level_t_steps_l() const { return lvl_t_steps_l_; }
    int n() const { return n_; }
    int kl() const { return kl_; }
    int ku() const { return ku_; }
    int rank() const { return r_; }
    bool col_border() const { return col_border_; }
    int col_border_nnz() const { return x_nnz_; }        // entries of X, all modes
    int batch() const { return batch_; }
    bool pivot() const { return pivot_; }
    double factor_ms() const { return factor_ms_; }   // device time of the factorization kernel
    double capacitance_cond_max() const { return cond_max_; }

private:
    int n_ = 0, r_ = 0, kl_ = 0, ku_ = 0, ldab_ = 1, batch_ = 0;
    bool pivot_ = true;
    double factor_ms_ = 0.0, cond_max_ = 1.0;
    Buffer<T> ab_, Vt_, W_, S_;
    Buffer<int> perm_, ipiv_, spiv_;
    // column border, per mode: A10 by column (offsets xptr_[m (r + 1) + k], absolute; rows in B
    // order; raw values) and A00 (r x r, row-major) for the Schur complement
    bool col_border_ = false;
    int x_nnz_ = 0;
    std::vector<long long> hxptr_;
    std::vector<int> hxrow_;
    Buffer<long long> xptr_;
    Buffer<int> xrow_;
    Buffer<T> xval_, A00_;
    const long long* xptr_ptr() const { return col_border_ ? xptr_.data() : nullptr; }
    mutable Buffer<T> scratch_;
    mutable Buffer<T> Wt_, St_;          // transposed Woodbury data (lazy)
    // level-scheduled sparse factors: one contiguous word stream per mode (offsets lvl_off_),
    // gather permutation (y[t] = b[lvl_gperm_[m*n + t]]), shared memory per block
    bool lvl_ = false, lvl_merge_ = true;
    int lvl_depth_f_ = 0, lvl_depth_b_ = 0, lvl_steps_f_ = 0, lvl_steps_b_ = 0;
    std::size_t lvl_smem_ = 0;
    double lvl_bytes_ = 0.0;
    std::string lvl_note_;
    Buffer<unsigned long long> lvl_blob_;
    Buffer<long long> lvl_off_;
    Buffer<int> lvl_gperm_;
    Buffer<long long> lvl_vptr_;         // V^T without zeros: row pointers (r + 1 per mode),
    Buffer<int> lvl_vcol_;               // columns
    Buffer<T> lvl_vval_;                 // and values
    mutable long long* lvl_probe_ = nullptr;
    void build_level_schedule();
    template <class IO>
    void launch_solve(const IO& io, void* stream) const;
    mutable Buffer<int> stpiv_;
    mutable bool tr_ready_ = false;
    bool lvl_t_ = false, band_released_ = false;
    int lvl_t_depth_u_ = 0, lvl_t_depth_l_ = 0, lvl_t_steps_u_ = 0, lvl_t_steps_l_ = 0;
    double lvl_t_bytes_ = 0.0;
    std::string lvl_t_note_ = "not requested";
    Buffer<unsigned long long> lvl_t_blob_;
    Buffer<long long> lvl_t_off_;
    Buffer<int> lvl_fpos_;
};

// Dense batched LU reference (cuBLAS getrf/getrsBatched), for performance comparison only.
// A: host (batch, n, n) row-major, b: host (batch, n); returns x (host) and device times in ms.
struct DenseRefResult {
    std::vector<cplx> x;
    double factor_ms = 0.0, solve_ms = 0.0;
};
DenseRefResult dense_lu_reference(const cplx* A, const cplx* b, int batch, int n, int repeats);

// ---- IMEX on the device, batched over modes -------------------------------------------------
// M_k dX_k/dt + L_k X_k = F(X_k, t) for every mode k, same scheme / dt / border for all modes.
// F is either zero or the pointwise model F = g .* X + h (g, h: (batch, n) device arrays), enough
// for cross-backend tests; nonlinear F arrives with the IVP work.
template <class T>
class Imex {
public:
    Imex(const std::vector<Csr<T>>& M, const std::vector<Csr<T>>& L,
         const std::vector<int>& border_rows, const std::string& scheme, double dt, bool pivot = true);
    ~Imex();
    Imex(const Imex&) = delete;
    Imex& operator=(const Imex&) = delete;
    void set_pointwise_rhs(const T* host_g, const T* host_h);   // (batch, n) each; nullptr = zero
    double run(T* d_x, double t, long nsteps);                    // d_x: (batch, n) device
    int n() const { return n_; }
    int batch() const { return batch_; }
    int n_factorizations() const { return static_cast<int>(factors_.size()); }
    long iteration() const { return iteration_; }
    const std::string& scheme() const { return scheme_; }

private:
    struct DevCsr;
    const BandedBatch<T>& factor(double alpha, double beta);
    void eval_F(const T* d_x, T* d_out) const;
    void step_sbdf(T* d_x);
    void step_rk(T* d_x);

    std::vector<Csr<T>> Mh_, Lh_;
    std::vector<int> border_;
    std::string scheme_;
    double dt_;
    bool pivot_;
    int n_ = 0, batch_ = 0, order_ = 0;
    long iteration_ = 0;
    std::unique_ptr<DevCsr> M_, L_;
    Buffer<T> g_, h_;
    bool has_g_ = false, has_h_ = false;
    std::vector<Buffer<T>> MXh_, Fh_;   // SBDF history ring (most recent first)
    std::map<std::pair<double, double>, std::unique_ptr<BandedBatch<T>>> factors_;
};

}  // namespace conatus::cuda
