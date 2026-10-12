// IMEX timesteppers executed in C++ (spec §17, §106: no timestep loop in Python).
// Problem: M dX/dt + L X = F(X, t). Same schemes and conventions as Dedalus 3:
//  - multistep SBDF1..4 (constant dt): a0 M X^n + L X^n = sum_j c_j F^{n-j} - sum_{j>=1} a_j M X^{n-j}
//    (a_j include 1/dt); iteration i uses order min(i+1, q) (startup).
//  - Runge–Kutta IMEX (RK111, RK222, RK443, ARS tableaux): stage i solves
//    (M + dt H_ii L) X_i = M X_0 + dt sum_{j<i} (A_ij F_j - H_ij L X_j).
// Factorizations are cached by their LHS coefficients: RK443 has all H_ii = 1/2, so a single
// factorization serves every stage ( P1); L X_0 is skipped when H[:,0] = 0 (P2).
#pragma once

#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "conatus/banded.hpp"

namespace conatus {

struct RKTableau {
    std::vector<std::vector<double>> A, H;
    std::vector<double> c;
    int stages() const { return static_cast<int>(c.size()) - 1; }
};

inline RKTableau rk_tableau(const std::string& name) {
    if (name == "RK111") return {{{0, 0}, {1, 0}}, {{0, 0}, {0, 1}}, {0, 1}};
    if (name == "RK222") {
        double g = (2.0 - std::sqrt(2.0)) / 2.0, d = 1.0 - 1.0 / g / 2.0;
        return {{{0, 0, 0}, {g, 0, 0}, {d, 1 - d, 0}}, {{0, 0, 0}, {0, g, 0}, {0, 1 - g, g}},
                {0, g, 1}};
    }
    if (name == "RK443")
        return {{{0, 0, 0, 0, 0},
                 {1. / 2, 0, 0, 0, 0},
                 {11. / 18, 1. / 18, 0, 0, 0},
                 {5. / 6, -5. / 6, 1. / 2, 0, 0},
                 {1. / 4, 7. / 4, 3. / 4, -7. / 4, 0}},
                {{0, 0, 0, 0, 0},
                 {0, 1. / 2, 0, 0, 0},
                 {0, 1. / 6, 1. / 2, 0, 0},
                 {0, -1. / 2, 1. / 2, 1. / 2, 0},
                 {0, 3. / 2, -3. / 2, 1. / 2, 1. / 2}},
                {0, 1. / 2, 2. / 3, 1. / 2, 1}};
    throw std::invalid_argument("unknown RK scheme: " + name);
}

// Order q of a scheme named "SBDF<q>", q = 1..4; anything else after "SBDF" is refused
// (std::stoi accepted "SBDF2x" as SBDF2 and reported "SBDFx" as a bare "stoi" error).
inline int sbdf_order(const std::string& scheme) {
    const std::string tail = scheme.substr(4);
    if (tail.size() != 1 || tail[0] < '1' || tail[0] > '4')
        throw std::invalid_argument("unknown SBDF scheme '" + scheme + "': SBDF1, SBDF2, SBDF3 or SBDF4");
    return tail[0] - '0';
}

// Constant-step SBDF coefficients (a scaled by dt: a_j / dt is the Dedalus a_j), c.
inline std::pair<std::vector<double>, std::vector<double>> sbdf_coefficients(int order) {
    switch (order) {
        case 1: return {{1, -1}, {0, 1}};
        case 2: return {{1.5, -2, 0.5}, {0, 2, -1}};
        case 3: return {{11. / 6, -3, 1.5, -1. / 3}, {0, 3, -3, 1}};
        case 4: return {{25. / 12, -4, 3, -4. / 3, 0.25}, {0, 4, -6, 4, -1}};
    }
    throw std::invalid_argument("SBDF order must be 1..4");
}

template <class T>
class ImexIntegrator {
public:
    using Rhs = std::function<void(const T* x, double t, T* out)>;

    ImexIntegrator(Csr<T> M, Csr<T> L, std::vector<int> border_rows, std::string scheme, double dt)
        : M_(std::move(M)), L_(std::move(L)), border_(std::move(border_rows)),
          scheme_(std::move(scheme)), dt_(dt) {
        if (M_.n != L_.n) throw std::invalid_argument("M and L sizes differ");
        if (!(dt_ > 0)) throw std::invalid_argument("dt must be > 0");
        n_ = M_.n;
        if (scheme_.rfind("SBDF", 0) == 0) {
            order_ = sbdf_order(scheme_);
        } else {
            tab_ = rk_tableau(scheme_);
        }
    }

    ImexIntegrator(const ImexIntegrator&) = delete;
    ImexIntegrator& operator=(const ImexIntegrator&) = delete;

    void set_rhs(Rhs f) { F_ = std::move(f); }
    int n() const { return n_; }
    int n_factorizations() const { return static_cast<int>(factors_.size()); }
    long iteration() const { return iteration_; }
    const std::string& scheme() const { return scheme_; }

    // Advance x (length n) in place by nsteps; returns the final time.
    double run(T* x, double t, long nsteps) {
        for (long s = 0; s < nsteps; ++s) {
            if (order_) step_sbdf(x, t); else step_rk(x, t);
            t += dt_;
            ++iteration_;
        }
        return t;
    }

private:
    const BorderedBanded<T>& factor(double alpha, double beta) {
        auto key = std::make_pair(alpha, beta);
        auto it = factors_.find(key);
        if (it != factors_.end()) return *it->second;
        Csr<T> lhs = csr_axpby(T(alpha), M_, T(beta), L_);
        auto f = std::make_unique<BorderedBanded<T>>(lhs, border_);
        auto& ref = *f;
        factors_.emplace(key, std::move(f));
        return ref;
    }

    void eval_F(const T* x, double t, T* out) {
        if (F_) F_(x, t, out); else std::fill(out, out + n_, T(0));
    }

    void step_sbdf(T* x, double t) {
        std::vector<T> MX(n_), FX(n_);
        M_.matvec(x, MX.data());
        eval_F(x, t, FX.data());
        MXh_.push_front(std::move(MX));
        Fh_.push_front(std::move(FX));
        if (static_cast<int>(MXh_.size()) > order_) { MXh_.pop_back(); Fh_.pop_back(); }
        int q = std::min<int>(order_, static_cast<int>(MXh_.size()));
        auto [a, c] = sbdf_coefficients(q);
        std::vector<T> rhs(n_, T(0));
        for (int j = 1; j <= q; ++j) {
            const auto& mx = MXh_[j - 1];
            const auto& fx = Fh_[j - 1];
            for (int i = 0; i < n_; ++i) rhs[i] += T(c[j]) * fx[i] - T(a[j] / dt_) * mx[i];
        }
        const auto& lu = factor(a[0] / dt_, 1.0);
        lu.solve(rhs.data(), 1);
        std::copy(rhs.begin(), rhs.end(), x);
    }

    void step_rk(T* x, double t) {
        const int s = tab_.stages();
        const auto& A = tab_.A;
        const auto& H = tab_.H;
        std::vector<std::vector<T>> X(s + 1), F(s + 1), LX(s + 1);
        X[0].assign(x, x + n_);
        std::vector<T> MX0(n_);
        M_.matvec(x, MX0.data());
        auto needF = [&](int j) { for (int i = j + 1; i <= s; ++i) if (A[i][j] != 0) return true; return false; };
        auto needLX = [&](int j) { for (int i = j + 1; i <= s; ++i) if (H[i][j] != 0) return true; return false; };
        std::vector<T> rhs(n_);
        for (int i = 1; i <= s; ++i) {
            int j = i - 1;
            if (needF(j)) { F[j].resize(n_); eval_F(X[j].data(), t + tab_.c[j] * dt_, F[j].data()); }
            if (needLX(j)) { LX[j].resize(n_); L_.matvec(X[j].data(), LX[j].data()); }
            rhs = MX0;
            for (int k = 0; k < i; ++k) {
                if (A[i][k] != 0)
                    for (int m = 0; m < n_; ++m) rhs[m] += T(dt_ * A[i][k]) * F[k][m];
                if (H[i][k] != 0)
                    for (int m = 0; m < n_; ++m) rhs[m] -= T(dt_ * H[i][k]) * LX[k][m];
            }
            const auto& lu = factor(1.0, dt_ * H[i][i]);
            lu.solve(rhs.data(), 1);
            X[i] = rhs;
        }
        std::copy(X[s].begin(), X[s].end(), x);
    }

    Csr<T> M_, L_;
    std::vector<int> border_;
    std::string scheme_;
    double dt_;
    int n_ = 0, order_ = 0;
    RKTableau tab_;
    Rhs F_;
    long iteration_ = 0;
    std::deque<std::vector<T>> MXh_, Fh_;
    std::map<std::pair<double, double>, std::unique_ptr<BorderedBanded<T>>> factors_;
};

}  // namespace conatus
