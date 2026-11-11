// Static partition of independent tasks over std::threads (no OpenMP: the same code builds on
// every platform of the wheels). Used by the batched CPU solves: system k is solved by one
// thread with exactly the arithmetic of the serial loop, so the result does not depend on the
// number of threads (bit for bit; tests/core/test_core_threads.py).
#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace conatus {

// Call f(k) for k in [0, n), split into at most `threads` contiguous chunks, one std::thread per
// chunk after the first (which runs on the calling thread). threads <= 1 or n <= 1: a plain loop.
// The first exception thrown by any task is rethrown on the calling thread after every thread
// has joined; tasks of other chunks still run to completion.
template <class F>
void parallel_for(std::size_t n, int threads, F&& f) {
    const std::size_t t = std::min<std::size_t>(n, threads > 1 ? static_cast<std::size_t>(threads) : 1);
    if (t <= 1) {
        for (std::size_t k = 0; k < n; ++k) f(k);
        return;
    }
    std::exception_ptr first;
    std::mutex mu;
    auto chunk = [&](std::size_t c) {
        const std::size_t lo = n * c / t, hi = n * (c + 1) / t;
        try {
            for (std::size_t k = lo; k < hi; ++k) f(k);
        } catch (...) {
            std::lock_guard<std::mutex> lock(mu);
            if (!first) first = std::current_exception();
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(t - 1);
    for (std::size_t c = 1; c < t; ++c) {
        try {
            pool.emplace_back(chunk, c);
        } catch (...) {          // no thread available (std::system_error): run the chunk here
            chunk(c);
        }
    }
    chunk(0);
    for (auto& th : pool) th.join();
    if (first) std::rethrow_exception(first);
}

}  // namespace conatus
