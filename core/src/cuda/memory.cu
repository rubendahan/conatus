// Device / pinned memory spaces and a simple caching pool for device allocations.
#include <cuda_runtime.h>

#include <cstring>
#include <map>
#include <mutex>

#include "common.cuh"
#include "conatus/cuda.hpp"
#include "conatus/memory.hpp"

namespace conatus {

bool cuda_backend_compiled() { return true; }

namespace {
// Freed device blocks are cached by exact (64-byte rounded) size and reused; never returned to the
// driver implicitly (pool_trim does it on request).
struct Pool {
    std::mutex m;
    std::multimap<std::size_t, void*> free_blocks;
    std::size_t live = 0, cached = 0, n_alloc = 0, n_reuse = 0;
};
Pool& pool() {
    static Pool* p = new Pool();   // leaked on purpose: no destruction-order issue at exit
    return *p;
}
}  // namespace

namespace detail {

void* space_alloc(MemorySpace s, std::size_t bytes) {
    void* p = nullptr;
    if (s == MemorySpace::PINNED) {
        cuda::check(cudaMallocHost(&p, bytes), "cudaMallocHost");
        return p;
    }
    if (s == MemorySpace::CUDA_DEVICE) {
        auto& P = pool();
        std::lock_guard<std::mutex> lock(P.m);
        auto it = P.free_blocks.find(bytes);
        if (it != P.free_blocks.end()) {
            p = it->second;
            P.free_blocks.erase(it);
            P.cached -= bytes;
            ++P.n_reuse;
        } else {
            cuda::check(cudaMalloc(&p, bytes), "cudaMalloc");
            ++P.n_alloc;
        }
        P.live += bytes;
        return p;
    }
    throw std::invalid_argument("space_alloc: host memory is allocated inline");
}

void space_free(MemorySpace s, void* p, std::size_t bytes) noexcept {
    if (!p) return;
    if (s == MemorySpace::PINNED) { cudaFreeHost(p); return; }
    if (s == MemorySpace::CUDA_DEVICE) {
        auto& P = pool();
        std::lock_guard<std::mutex> lock(P.m);
        P.free_blocks.emplace(bytes, p);
        P.live -= bytes;
        P.cached += bytes;
    }
}

void space_zero(MemorySpace s, void* p, std::size_t bytes) {
    if (s == MemorySpace::CUDA_DEVICE) cuda::check(cudaMemset(p, 0, bytes), "cudaMemset");
    else std::memset(p, 0, bytes);
}

void space_copy(MemorySpace d, void* dst, MemorySpace s, const void* src, std::size_t bytes) {
    bool dd = d == MemorySpace::CUDA_DEVICE, sd = s == MemorySpace::CUDA_DEVICE;
    cudaMemcpyKind kind = dd ? (sd ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice)
                             : (sd ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost);
    cuda::check(cudaMemcpy(dst, src, bytes, kind), "cudaMemcpy");
}

}  // namespace detail

namespace cuda {

bool available() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) { cudaGetLastError(); return false; }
    return n > 0;
}

std::string device_name() {
    if (!available()) return "none";
    int dev = 0;
    CN_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CN_CUDA(cudaGetDeviceProperties(&prop, dev));
    return std::string(prop.name) + " (sm_" + std::to_string(prop.major) + std::to_string(prop.minor) + ")";
}

void synchronize() { CN_CUDA(cudaDeviceSynchronize()); }

PoolStats pool_stats() {
    auto& P = pool();
    std::lock_guard<std::mutex> lock(P.m);
    return {P.live, P.cached, P.n_alloc, P.n_reuse};
}

void pool_trim() {
    auto& P = pool();
    std::lock_guard<std::mutex> lock(P.m);
    for (auto& kv : P.free_blocks) cudaFree(kv.second);
    P.free_blocks.clear();
    P.cached = 0;
}

}  // namespace cuda
}  // namespace conatus
