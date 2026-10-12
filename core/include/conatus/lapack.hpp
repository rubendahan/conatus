// Minimal LAPACK declarations (Fortran ABI, 32-bit integers: OpenBLAS from conda-forge).
#pragma once

#include <complex>
#include <cstddef>

extern "C" {
void dgbtrf_(const int* m, const int* n, const int* kl, const int* ku, double* ab, const int* ldab,
             int* ipiv, int* info);
void dgbtrs_(const char* trans, const int* n, const int* kl, const int* ku, const int* nrhs,
             const double* ab, const int* ldab, const int* ipiv, double* b, const int* ldb,
             int* info, std::size_t trans_len);
void zgbtrf_(const int* m, const int* n, const int* kl, const int* ku, std::complex<double>* ab,
             const int* ldab, int* ipiv, int* info);
void zgbtrs_(const char* trans, const int* n, const int* kl, const int* ku, const int* nrhs,
             const std::complex<double>* ab, const int* ldab, const int* ipiv,
             std::complex<double>* b, const int* ldb, int* info, std::size_t trans_len);
void dgetrf_(const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info);
void dgetrs_(const char* trans, const int* n, const int* nrhs, const double* a, const int* lda,
             const int* ipiv, double* b, const int* ldb, int* info, std::size_t trans_len);
void zgetrf_(const int* m, const int* n, std::complex<double>* a, const int* lda, int* ipiv,
             int* info);
void zgetrs_(const char* trans, const int* n, const int* nrhs, const std::complex<double>* a,
             const int* lda, const int* ipiv, std::complex<double>* b, const int* ldb, int* info,
             std::size_t trans_len);
}

namespace conatus::lapack {

inline int gbtrf(int n, int kl, int ku, double* ab, int ldab, int* ipiv) {
    int info = 0;
    dgbtrf_(&n, &n, &kl, &ku, ab, &ldab, ipiv, &info);
    return info;
}
inline int gbtrf(int n, int kl, int ku, std::complex<double>* ab, int ldab, int* ipiv) {
    int info = 0;
    zgbtrf_(&n, &n, &kl, &ku, ab, &ldab, ipiv, &info);
    return info;
}
inline int gbtrs(char tr, int n, int kl, int ku, int nrhs, const double* ab, int ldab,
                 const int* ipiv, double* b, int ldb) {
    int info = 0;
    dgbtrs_(&tr, &n, &kl, &ku, &nrhs, ab, &ldab, ipiv, b, &ldb, &info, 1);
    return info;
}
inline int gbtrs(char tr, int n, int kl, int ku, int nrhs, const std::complex<double>* ab,
                 int ldab, const int* ipiv, std::complex<double>* b, int ldb) {
    int info = 0;
    zgbtrs_(&tr, &n, &kl, &ku, &nrhs, ab, &ldab, ipiv, b, &ldb, &info, 1);
    return info;
}
inline int getrf(int n, double* a, int* ipiv) {
    int info = 0;
    dgetrf_(&n, &n, a, &n, ipiv, &info);
    return info;
}
inline int getrf(int n, std::complex<double>* a, int* ipiv) {
    int info = 0;
    zgetrf_(&n, &n, a, &n, ipiv, &info);
    return info;
}
inline int getrs(char tr, int n, int nrhs, const double* a, const int* ipiv, double* b) {
    int info = 0;
    dgetrs_(&tr, &n, &nrhs, a, &n, ipiv, b, &n, &info, 1);
    return info;
}
inline int getrs(char tr, int n, int nrhs, const std::complex<double>* a, const int* ipiv,
                 std::complex<double>* b) {
    int info = 0;
    zgetrs_(&tr, &n, &nrhs, a, &n, ipiv, b, &n, &info, 1);
    return info;
}

}  // namespace conatus::lapack
