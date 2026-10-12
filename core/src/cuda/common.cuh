// Shared helpers for the CUDA backend (device code only).
#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

#include <complex>
#include <cuda/std/complex>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace conatus::cuda {

inline void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(e));
}
inline void check_fft(cufftResult r, const char* what) {
    if (r != CUFFT_SUCCESS)
        throw std::runtime_error(std::string("cuFFT error ") + std::to_string(int(r)) + " in " + what);
}
#define CN_CUDA(call) ::conatus::cuda::check((call), #call)
#define CN_CUFFT(call) ::conatus::cuda::check_fft((call), #call)
#define CN_LAUNCH_CHECK(name) ::conatus::cuda::check(cudaGetLastError(), name)

// Device-side scalar type for a host type: double -> double, std::complex<double> ->
// cuda::std::complex<double> (same layout).
template <class T> struct dev_type { using type = T; };
template <> struct dev_type<std::complex<double>> { using type = ::cuda::std::complex<double>; };
template <class T> using dev_t = typename dev_type<T>::type;

template <class T>
inline dev_t<T>* dptr(T* p) { return reinterpret_cast<dev_t<T>*>(p); }
template <class T>
inline const dev_t<T>* dptr(const T* p) { return reinterpret_cast<const dev_t<T>*>(p); }

__host__ __device__ inline double absval(double x) { return x < 0 ? -x : x; }
__host__ __device__ inline double absval(::cuda::std::complex<double> z) { return ::cuda::std::abs(z); }

inline unsigned blocks_for(std::size_t n, unsigned threads = 256) {
    return static_cast<unsigned>((n + threads - 1) / threads);
}

}  // namespace conatus::cuda
