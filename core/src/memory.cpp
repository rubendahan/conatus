// Host-only build: device and pinned memory spaces need the CUDA backend (CONATUS_CUDA=ON).
#include "conatus/memory.hpp"

#include <cstring>

namespace conatus {

#ifndef CONATUS_WITH_CUDA
bool cuda_backend_compiled() { return false; }

namespace detail {
[[noreturn]] static void no_cuda(MemorySpace s) {
    throw std::runtime_error(std::string("memory space '") + to_string(s) +
                             "' needs the CUDA backend: rebuild with CONATUS_CUDA=ON (core/build.sh)");
}
void* space_alloc(MemorySpace s, std::size_t) { no_cuda(s); }
void space_free(MemorySpace, void*, std::size_t) noexcept {}
void space_zero(MemorySpace s, void*, std::size_t) { no_cuda(s); }
void space_copy(MemorySpace d, void* dst, MemorySpace s, const void* src, std::size_t bytes) {
    if (d != MemorySpace::HOST) no_cuda(d);
    if (s != MemorySpace::HOST) no_cuda(s);
    std::memcpy(dst, src, bytes);
}
}  // namespace detail
#endif

}  // namespace conatus
