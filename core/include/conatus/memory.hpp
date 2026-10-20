// Buffers and memory spaces.
// HOST buffers are allocated inline (64-byte aligned). PINNED and CUDA_DEVICE buffers go through the
// CUDA backend (core/src/cuda/memory.cu, device allocations through a simple caching pool); without
// the backend (CONATUS_CUDA=OFF) asking for them raises an explicit error.
// Copies between spaces are always explicit (copy / copy_from_host / copy_to_host): never implicit.
#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

namespace conatus {

enum class MemorySpace { HOST, PINNED, CUDA_DEVICE };

inline const char* to_string(MemorySpace s) {
    switch (s) {
        case MemorySpace::HOST: return "host";
        case MemorySpace::PINNED: return "pinned";
        case MemorySpace::CUDA_DEVICE: return "cuda_device";
    }
    return "?";
}

namespace detail {
// Implemented in core/src/cuda/memory.cu (CUDA backend) or core/src/memory.cpp (stubs that raise).
void* space_alloc(MemorySpace s, std::size_t bytes);
void space_free(MemorySpace s, void* p, std::size_t bytes) noexcept;
void space_zero(MemorySpace s, void* p, std::size_t bytes);
void space_copy(MemorySpace dst_space, void* dst, MemorySpace src_space, const void* src,
                std::size_t bytes);
}  // namespace detail

bool cuda_backend_compiled();

// Owning, typed buffer in one memory space. Move-only; never reallocates behind the caller's back.
// operator[] is only meaningful for HOST / PINNED buffers.
template <class T>
class Buffer {
public:
    Buffer() = default;
    explicit Buffer(std::size_t n, MemorySpace space = MemorySpace::HOST) : n_(n), space_(space) {
        if (!n) return;
        if (n > (static_cast<std::size_t>(-1) - 63) / sizeof(T))    // n * sizeof(T) would wrap
            throw std::length_error("Buffer: " + std::to_string(n) + " elements overflow size_t");
        bytes_ = ((n * sizeof(T) + 63) / 64) * 64;
        if (space == MemorySpace::HOST) {
            ptr_ = static_cast<T*>(std::aligned_alloc(64, bytes_));
            if (!ptr_) throw std::bad_alloc();
            std::memset(static_cast<void*>(ptr_), 0, bytes_);
        } else {
            ptr_ = static_cast<T*>(detail::space_alloc(space, bytes_));
            try {
                detail::space_zero(space, ptr_, bytes_);
            } catch (...) {          // a throwing constructor runs no destructor: release here
                detail::space_free(space, ptr_, bytes_);
                throw;
            }
        }
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& o) noexcept { swap(o); }
    Buffer& operator=(Buffer&& o) noexcept { if (this != &o) { release(); swap(o); } return *this; }
    ~Buffer() { release(); }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    std::size_t size() const { return n_; }
    MemorySpace space() const { return space_; }
    T& operator[](std::size_t i) { return ptr_[i]; }
    const T& operator[](std::size_t i) const { return ptr_[i]; }
    void zero() {
        if (!n_) return;
        if (space_ == MemorySpace::HOST) std::memset(static_cast<void*>(ptr_), 0, n_ * sizeof(T));
        else detail::space_zero(space_, ptr_, n_ * sizeof(T));
    }

private:
    void release() noexcept {
        if (ptr_) {
            if (space_ == MemorySpace::HOST) std::free(ptr_);
            else detail::space_free(space_, ptr_, bytes_);
        }
        ptr_ = nullptr;
        n_ = bytes_ = 0;
    }
    void swap(Buffer& o) noexcept {
        std::swap(ptr_, o.ptr_); std::swap(n_, o.n_); std::swap(bytes_, o.bytes_);
        std::swap(space_, o.space_);
    }
    T* ptr_ = nullptr;
    std::size_t n_ = 0, bytes_ = 0;
    MemorySpace space_ = MemorySpace::HOST;
};

// Explicit copies (sizes must match exactly).
template <class T>
void copy(Buffer<T>& dst, const Buffer<T>& src) {
    if (dst.size() != src.size()) throw std::invalid_argument("copy: size mismatch");
    if (dst.size())
        detail::space_copy(dst.space(), dst.data(), src.space(), src.data(), sizeof(T) * dst.size());
}
template <class T>
void copy_from_host(Buffer<T>& dst, const T* host, std::size_t n) {
    if (n != dst.size()) throw std::invalid_argument("copy_from_host: size mismatch");
    if (n) detail::space_copy(dst.space(), dst.data(), MemorySpace::HOST, host, sizeof(T) * n);
}
template <class T>
void copy_to_host(T* host, const Buffer<T>& src, std::size_t n) {
    if (n != src.size()) throw std::invalid_argument("copy_to_host: size mismatch");
    if (n) detail::space_copy(MemorySpace::HOST, host, src.space(), src.data(), sizeof(T) * n);
}

}  // namespace conatus
