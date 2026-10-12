// Bordered banded solves (technical spec §19–20, DEDALUS.md §4, pitfall P8 of).
//
// A tau/BC system is "almost banded": after moving the r boundary rows to the top, every other
// row has its nonzeros inside a band [i-kl, i+ku]. We write P A = B + U V^T with
//   B     = P A with its r first rows replaced by s_i e_i^T (banded, kl/ku from the other rows),
//   U     = [e_0 ... e_{r-1}],   V^T = (first r rows of P A) - diag(s),
// factor B once with LAPACK gbtrf and solve with the r x r capacitance (Schur) matrix
//   S = I + V^T B^{-1} U  (Woodbury). Factor once, solve many times; batch over modes.
//
// Column border (opt-in, ``col_border``). In the first-order spherical shell the
// tau of the gradient, lifted and multiplied by r, fills whole columns: kept in B they make the
// band as wide as the system (kl ~ n/2). The caller then moves those columns among the first r
// ("pinned") columns, with as many extra bulk rows among the border rows, and asks for the column
// border. With the partition P A = [[A00, A01], [A10, A11]] (A00 r x r), every entry of a bulk row
// in a column < r (A10, sparse, stored as X by column) leaves the band: B = diag(s) (+) A11 is
// block diagonal and its band is the band of the core A11. The solve is the Schur complement
//   S = A00 - A01 W1,  W1 = A11^{-1} A10            (factor time: r band solves)
//   x0 = S^{-1} (b0 - A01 A11^{-1} b1),  x1 = A11^{-1} b1 - W1 x0   (one band solve)
//   transpose: x0 = S^{-T} (b0 - A10^T A11^{-T} b1),  x1 = A11^{-T} b1 - Wt1 x0,
//              Wt1 = A11^{-T} A01^T.
// It is written in the storage of the row border (one code path): V^T = [diag(s), -A01] (so
// V^T B^{-1} b = b0 - A01 A11^{-1} b1), W = [-I; W1], S as above, and the final x = y - W z
// starts from 0 instead of y on the first r entries; transposed: B^{-T} b, then the top entries
// y_k -= (A10^T y)_k / s_k, St = diag(1/s) S^T, Wt = [-I; Wt1]. Not the shifted Woodbury form
// S = I + V^T B^{-1} U of the row border: with a bulk row in the border, A00 << s and that form
// loses digits in I + (A00/s - I) (measured 1e-11 .. 1e-8 relative on the shell, against 1e-15
// for the Schur form). Exact (no approximation); X stores the raw entries A_ik.
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "conatus/lapack.hpp"

namespace conatus {

template <class T>
struct Csr {
    int n = 0;
    std::vector<int> indptr, indices;
    std::vector<T> data;

    // y = alpha * A x (+ y if accumulate)
    void matvec(const T* x, T* y, T alpha = T(1), bool accumulate = false) const {
        for (int i = 0; i < n; ++i) {
            T s = T(0);
            for (int p = indptr[i]; p < indptr[i + 1]; ++p) s += data[p] * x[indices[p]];
            y[i] = accumulate ? y[i] + alpha * s : alpha * s;
        }
    }
};

// alpha * A + beta * B on the union sparsity pattern (same n).
template <class T>
Csr<T> csr_axpby(T alpha, const Csr<T>& A, T beta, const Csr<T>& B) {
    if (A.n != B.n) throw std::invalid_argument("csr_axpby: size mismatch");
    Csr<T> C;
    C.n = A.n;
    C.indptr.assign(A.n + 1, 0);
    std::vector<T> row(A.n, T(0));
    std::vector<char> seen(A.n, 0);
    std::vector<int> cols;
    for (int i = 0; i < A.n; ++i) {
        cols.clear();
        auto add = [&](const Csr<T>& M, T a) {
            for (int p = M.indptr[i]; p < M.indptr[i + 1]; ++p) {
                int j = M.indices[p];
                if (!seen[j]) { seen[j] = 1; cols.push_back(j); }
                row[j] += a * M.data[p];
            }
        };
        add(A, alpha);
        add(B, beta);
        std::sort(cols.begin(), cols.end());
        for (int j : cols) {
            C.indices.push_back(j);
            C.data.push_back(row[j]);
            row[j] = T(0);
            seen[j] = 0;
        }
        C.indptr[i + 1] = static_cast<int>(C.indices.size());
    }
    return C;
}

// Unfactored bordered-banded layout of a square CSR matrix: shared by the CPU solver and the
// batched GPU solver (single source of truth for the permutation, band and border).
// kl_min / ku_min widen the band (common band over a batch of modes).
template <class T>
struct BorderedLayout {
    int n = 0, r = 0, kl = 0, ku = 0, ldab = 1;
    std::vector<int> perm;   // perm[i] = original row placed at position i (border rows first)
    std::vector<T> ab;       // LAPACK band storage (ldab x n, column-major), unfactored B
    std::vector<T> Vt;       // r x n (row-major): border rows of P A minus diag(s)
                             // (column border: [diag(s), -A01], see the top of this file)
    // column border (empty unless col_border): A10 by column, column k < r has the entries
    // xptr[k]..xptr[k+1]-1, row (position in P A, >= r) xrow[q], value xval[q] = (P A)_{xrow[q], k};
    // A00 (r x r, row-major) and s (the diagonal of the border block of B)
    bool col_border = false;
    std::vector<int> xptr, xrow;
    std::vector<T> xval, A00, s;
};

// Column border, transposed solves: y_k -= (A10^T y)_k / s_k for k < r (rows >= r untouched).
template <class T>
void colborder_transpose(const std::vector<int>& xptr, const std::vector<int>& xrow,
                         const std::vector<T>& xval, const std::vector<T>& sd, T* y) {
    const int r = static_cast<int>(xptr.size()) - 1;
    for (int k = 0; k < r; ++k) {
        T acc = T(0);
        for (int q = xptr[k]; q < xptr[k + 1]; ++q) acc += xval[q] * y[xrow[q]];
        y[k] -= acc / sd[k];
    }
}

// Band of the bulk rows; with col_border, entries in the first r columns are not counted (they
// go to the column border X, see make_bordered_layout).
template <class T>
void measure_band(const Csr<T>& A, const std::vector<int>& border_rows, int& kl, int& ku,
                  bool col_border = false) {
    const int n = A.n, r = static_cast<int>(border_rows.size());
    std::vector<char> is_border(n, 0);
    for (int i : border_rows) {
        if (i < 0 || i >= n) throw std::invalid_argument("border row out of range");
        if (is_border[i]) throw std::invalid_argument("duplicate border row");
        is_border[i] = 1;
    }
    std::vector<int> perm = border_rows;
    for (int i = 0; i < n; ++i)
        if (!is_border[i]) perm.push_back(i);
    kl = ku = 0;
    for (int i = r; i < n; ++i) {
        int o = perm[i];
        for (int p = A.indptr[o]; p < A.indptr[o + 1]; ++p) {
            if (A.data[p] == T(0)) continue;
            int j = A.indices[p];
            if (col_border && j < r) continue;
            kl = std::max(kl, i - j);
            ku = std::max(ku, j - i);
        }
    }
}

template <class T>
BorderedLayout<T> make_bordered_layout(const Csr<T>& A, const std::vector<int>& border_rows,
                                       int kl_min = 0, int ku_min = 0, bool col_border = false) {
    BorderedLayout<T> Lo;
    Lo.n = A.n;
    Lo.r = static_cast<int>(border_rows.size());
    Lo.col_border = col_border;
    int kl, ku;
    measure_band(A, border_rows, kl, ku, col_border);   // validates border rows
    Lo.kl = std::max(kl, kl_min);
    Lo.ku = std::max(ku, ku_min);
    const int n = Lo.n, r = Lo.r;
    std::vector<char> is_border(n, 0);
    for (int i : border_rows) is_border[i] = 1;
    Lo.perm = border_rows;
    for (int i = 0; i < n; ++i)
        if (!is_border[i]) Lo.perm.push_back(i);
    Lo.ldab = 2 * Lo.kl + Lo.ku + 1;
    Lo.ab.assign(static_cast<std::size_t>(Lo.ldab) * n, T(0));
    Lo.Vt.assign(static_cast<std::size_t>(r) * n, T(0));
    auto AB = [&](int i, int j) -> T& {
        return Lo.ab[static_cast<std::size_t>(j) * Lo.ldab + (Lo.kl + Lo.ku + i - j)];
    };
    std::vector<T> sdiag(r, T(1));                         // s_k of the border rows
    std::vector<std::vector<std::pair<int, T>>> xcol(col_border ? r : 0);
    if (col_border) Lo.A00.assign(static_cast<std::size_t>(r) * r, T(0));
    for (int i = 0; i < n; ++i) {
        int o = Lo.perm[i];
        if (i < r) {
            double s = 0.0;
            for (int p = A.indptr[o]; p < A.indptr[o + 1]; ++p) {
                s = std::max(s, static_cast<double>(std::abs(A.data[p])));
                Lo.Vt[static_cast<std::size_t>(i) * n + A.indices[p]] += A.data[p];
            }
            if (s == 0.0) s = 1.0;
            AB(i, i) = T(s);
            sdiag[i] = T(s);
            if (col_border) {
                // V^T row = [s e_i, -A01 row]; A00 kept exact for the Schur complement
                T* v = Lo.Vt.data() + static_cast<std::size_t>(i) * n;
                for (int j = 0; j < n; ++j) {
                    if (j < r) Lo.A00[static_cast<std::size_t>(i) * r + j] = v[j];
                    v[j] = (j < r) ? T(0) : -v[j];
                }
                v[i] = T(s);
            } else {
                Lo.Vt[static_cast<std::size_t>(i) * n + i] -= T(s);
            }
        } else {
            for (int p = A.indptr[o]; p < A.indptr[o + 1]; ++p) {
                if (A.data[p] == T(0)) continue;
                const int j = A.indices[p];
                if (col_border && j < r) xcol[j].push_back({i, A.data[p]});
                else AB(i, j) += A.data[p];
            }
        }
    }
    if (col_border) {
        Lo.xptr.assign(r + 1, 0);
        for (int k = 0; k < r; ++k) {
            // rows ascending (filled in row order; one entry per (row, column) of a CSR matrix)
            for (const auto& [i, v] : xcol[k]) {
                Lo.xrow.push_back(i);
                Lo.xval.push_back(v);
            }
            Lo.xptr[k + 1] = static_cast<int>(Lo.xrow.size());
        }
        Lo.s = sdiag;
    }
    return Lo;
}

template <class T>
class BorderedBanded {
public:
    // col_border: see "Column border" at the top of this file (the first r columns leave the band).
    BorderedBanded(const Csr<T>& A, const std::vector<int>& border_rows, bool col_border = false)
        : n_(A.n) {
        BorderedLayout<T> Lo = make_bordered_layout(A, border_rows, 0, 0, col_border);
        col_border_ = col_border;
        xptr_ = std::move(Lo.xptr);
        xrow_ = std::move(Lo.xrow);
        xval_ = std::move(Lo.xval);
        A00_ = std::move(Lo.A00);
        sd_ = std::move(Lo.s);
        r_ = Lo.r;
        kl_ = Lo.kl;
        ku_ = Lo.ku;
        ldab_ = Lo.ldab;
        perm_ = std::move(Lo.perm);
        ab_ = std::move(Lo.ab);
        Vt_ = std::move(Lo.Vt);
        ipiv_.assign(n_, 0);
        int info = lapack::gbtrf(n_, kl_, ku_, ab_.data(), ldab_, ipiv_.data());
        if (info < 0)       // an illegal argument: a layout bug, never a usable factorization
            throw std::logic_error("BorderedBanded: gbtrf rejected argument " + std::to_string(-info));
        if (info > 0)
            throw std::runtime_error(
                "BorderedBanded: banded part is singular at pivot " + std::to_string(info) +
                "; the non-border rows must form an invertible banded operator "
                "(check the row/column ordering of the plan)");
        if (r_ > 0) {
            W_.assign(static_cast<std::size_t>(n_) * r_, T(0));
            if (col_border_) {
                // W = [-I; W1], W1 = A11^{-1} A10 (B is block diagonal: the top stays 0, then -I);
                // S = A00 - A01 W1 = A00 + V^T[:, r:] W1 (Schur complement, see the top)
                for (int k = 0; k < r_; ++k)
                    for (int q = xptr_[k]; q < xptr_[k + 1]; ++q)
                        W_[static_cast<std::size_t>(k) * n_ + xrow_[q]] = xval_[q];
                lapack::gbtrs('N', n_, kl_, ku_, r_, ab_.data(), ldab_, ipiv_.data(), W_.data(), n_);
                for (int k = 0; k < r_; ++k)
                    for (int i = 0; i < r_; ++i) W_[static_cast<std::size_t>(k) * n_ + i] = (i == k) ? T(-1) : T(0);
                S_ = schur(W_);
            } else {
                // W = B^{-1} U, U = first r unit vectors.
                for (int k = 0; k < r_; ++k) W_[static_cast<std::size_t>(k) * n_ + k] = T(1);
                lapack::gbtrs('N', n_, kl_, ku_, r_, ab_.data(), ldab_, ipiv_.data(), W_.data(), n_);
                S_ = capacitance(Vt_, W_);
            }
            sipiv_.assign(r_, 0);
            cond_ = factor_small(S_, sipiv_);
        }
    }

    int n() const { return n_; }
    int kl() const { return kl_; }
    int ku() const { return ku_; }
    int rank() const { return r_; }
    bool col_border() const { return col_border_; }
    int col_border_nnz() const { return static_cast<int>(xrow_.size()); }
    double capacitance_cond() const { return cond_; }

    // Solve A X = B in place; b holds nrhs column vectors of length n contiguously.
    void solve(T* b, int nrhs) const {
        std::vector<T> y(n_);
        std::vector<T> z(r_);
        for (int c = 0; c < nrhs; ++c) {
            T* bc = b + static_cast<std::size_t>(c) * n_;
            for (int i = 0; i < n_; ++i) y[i] = bc[perm_[i]];
            lapack::gbtrs('N', n_, kl_, ku_, 1, ab_.data(), ldab_, ipiv_.data(), y.data(), n_);
            if (r_ > 0) {
                for (int k = 0; k < r_; ++k) {
                    T s = T(0);
                    const T* vk = Vt_.data() + static_cast<std::size_t>(k) * n_;
                    for (int i = 0; i < n_; ++i) s += vk[i] * y[i];
                    z[k] = s;
                }
                lapack::getrs('N', r_, 1, S_.data(), sipiv_.data(), z.data());
                if (col_border_) std::fill(y.begin(), y.begin() + r_, T(0));   // x0 = z (W top = -I)
                for (int k = 0; k < r_; ++k) {
                    const T* wk = W_.data() + static_cast<std::size_t>(k) * n_;
                    for (int i = 0; i < n_; ++i) y[i] -= wk[i] * z[k];
                }
            }
            std::copy(y.begin(), y.end(), bc);
        }
    }

    // Solve A^T X = B in place (plain transpose, no conjugation): the discrete adjoint path.
    // A = P^T (Bt + U V^T)  =>  A^T x = b  <=>  (Bt^T + V U^T) y = b, x = P^T y.
    void solve_transpose(T* b, int nrhs) const {
        ensure_transpose();
        std::vector<T> y(n_);
        std::vector<T> z(r_);
        for (int c = 0; c < nrhs; ++c) {
            T* bc = b + static_cast<std::size_t>(c) * n_;
            std::copy(bc, bc + n_, y.begin());
            lapack::gbtrs('T', n_, kl_, ku_, 1, ab_.data(), ldab_, ipiv_.data(), y.data(), n_);
            if (col_border_) colborder_transpose(xptr_, xrow_, xval_, sd_, y.data());
            if (r_ > 0) {
                for (int k = 0; k < r_; ++k) z[k] = y[k];  // U^T y
                lapack::getrs('N', r_, 1, St_.data(), stipiv_.data(), z.data());
                if (col_border_) std::fill(y.begin(), y.begin() + r_, T(0));   // x0 = z (Wt top = -I)
                for (int k = 0; k < r_; ++k) {
                    const T* wk = Wt_.data() + static_cast<std::size_t>(k) * n_;
                    for (int i = 0; i < n_; ++i) y[i] -= wk[i] * z[k];
                }
            }
            for (int i = 0; i < n_; ++i) bc[perm_[i]] = y[i];
        }
    }

private:
    // S = I + Vt W (r x r, column-major).
    std::vector<T> capacitance(const std::vector<T>& Vt, const std::vector<T>& W) const {
        std::vector<T> S(static_cast<std::size_t>(r_) * r_, T(0));
        for (int a = 0; a < r_; ++a)
            for (int b = 0; b < r_; ++b) {
                T s = (a == b) ? T(1) : T(0);
                const T* va = Vt.data() + static_cast<std::size_t>(a) * n_;
                const T* wb = W.data() + static_cast<std::size_t>(b) * n_;
                for (int i = 0; i < n_; ++i) s += va[i] * wb[i];
                S[static_cast<std::size_t>(b) * r_ + a] = s;
            }
        return S;
    }

    // Column border: S = A00 + V^T[:, r:] W1 = A00 - A01 A11^{-1} A10 (r x r, column-major).
    std::vector<T> schur(const std::vector<T>& W) const {
        std::vector<T> S(static_cast<std::size_t>(r_) * r_, T(0));
        for (int a = 0; a < r_; ++a)
            for (int b = 0; b < r_; ++b) {
                T acc = A00_[static_cast<std::size_t>(a) * r_ + b];
                const T* va = Vt_.data() + static_cast<std::size_t>(a) * n_;
                const T* wb = W.data() + static_cast<std::size_t>(b) * n_;
                for (int i = r_; i < n_; ++i) acc += va[i] * wb[i];
                S[static_cast<std::size_t>(b) * r_ + a] = acc;
            }
        return S;
    }

    // LU-factor S in place; return its 1-norm condition number (explicit inverse, r is small).
    double factor_small(std::vector<T>& S, std::vector<int>& piv) const {
        auto norm1 = [&](const std::vector<T>& X) {
            double m = 0.0;
            for (int b = 0; b < r_; ++b) {
                double s = 0.0;
                for (int a = 0; a < r_; ++a) s += std::abs(X[static_cast<std::size_t>(b) * r_ + a]);
                m = std::max(m, s);
            }
            return m;
        };
        double nS = norm1(S);
        int info = lapack::getrf(r_, S.data(), piv.data());
        if (info < 0)
            throw std::logic_error("BorderedBanded: getrf rejected argument " + std::to_string(-info));
        if (info > 0)
            throw std::runtime_error(
                "BorderedBanded: capacitance matrix is singular: the boundary rows are not "
                "independent of the banded operator (ill-posed or rank-deficient BCs)");
        std::vector<T> inv(static_cast<std::size_t>(r_) * r_, T(0));
        for (int k = 0; k < r_; ++k) inv[static_cast<std::size_t>(k) * r_ + k] = T(1);
        lapack::getrs('N', r_, r_, S.data(), piv.data(), inv.data());
        return nS * norm1(inv);
    }

    void ensure_transpose() const {
        if (tr_ready_ || r_ == 0) return;
        St_.assign(static_cast<std::size_t>(r_) * r_, T(0));
        if (col_border_) {
            // Wt = [-I; Wt1], Wt1 = A11^{-T} A01^T (A01 = -V^T[:, r:]); St = diag(1/s) S^T with
            // S recomputed unfactored (Schur complement, see the top of this file)
            Wt_.assign(static_cast<std::size_t>(n_) * r_, T(0));
            for (int k = 0; k < r_; ++k)
                for (int i = r_; i < n_; ++i)
                    Wt_[static_cast<std::size_t>(k) * n_ + i] = -Vt_[static_cast<std::size_t>(k) * n_ + i];
            lapack::gbtrs('T', n_, kl_, ku_, r_, ab_.data(), ldab_, ipiv_.data(), Wt_.data(), n_);
            for (int k = 0; k < r_; ++k)
                for (int i = 0; i < r_; ++i) Wt_[static_cast<std::size_t>(k) * n_ + i] = (i == k) ? T(-1) : T(0);
            const std::vector<T> S = schur(W_);
            for (int a = 0; a < r_; ++a)
                for (int b = 0; b < r_; ++b)       // St(a, b) = S(b, a) / s_a, column-major
                    St_[static_cast<std::size_t>(b) * r_ + a] = S[static_cast<std::size_t>(a) * r_ + b] / sd_[a];
        } else {
            // Wt = B^{-T} V  (V = Vt^T, stored as r columns of length n = rows of Vt).
            Wt_ = Vt_;
            lapack::gbtrs('T', n_, kl_, ku_, r_, ab_.data(), ldab_, ipiv_.data(), Wt_.data(), n_);
            // St = I + U^T Wt: (a, b) entry = Wt[b][a].
            for (int a = 0; a < r_; ++a)
                for (int b = 0; b < r_; ++b)
                    St_[static_cast<std::size_t>(b) * r_ + a] =
                        (a == b ? T(1) : T(0)) + Wt_[static_cast<std::size_t>(b) * n_ + a];
        }
        stipiv_.assign(r_, 0);
        factor_small(St_, stipiv_);
        tr_ready_ = true;
    }

    int n_ = 0, r_ = 0, kl_ = 0, ku_ = 0, ldab_ = 1;
    bool col_border_ = false;
    std::vector<int> xptr_, xrow_;      // column border (see the top of this file)
    std::vector<T> xval_, A00_, sd_;
    std::vector<int> perm_, ipiv_, sipiv_;
    std::vector<T> ab_, Vt_, W_, S_;
    double cond_ = 1.0;
    mutable bool tr_ready_ = false;
    mutable std::vector<T> Wt_, St_;
    mutable std::vector<int> stipiv_;
};

}  // namespace conatus
