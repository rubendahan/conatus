// nanobind module conatus._core : thin bindings, no numerics here.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <complex>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "conatus/banded.hpp"
#include "conatus/imex.hpp"
#include "conatus/memory.hpp"
#ifdef CONATUS_WITH_CUDA
#include "conatus/cuda.hpp"
#endif

#ifndef CONATUS_VERSION
#error "CONATUS_VERSION is set by CMake from python/conatus/__init__.py (core/CMakeLists.txt)"
#endif

namespace nb = nanobind;
using cplx = std::complex<double>;

namespace {

template <class T>
using Arr2 = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
template <class T>
using Arr1 = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using IArr1 = nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// Product of array dimensions times the element size, checked for overflow before any allocation
// (review, CUDA risks: the products were formed unchecked; a wrapped size allocates a small buffer
// that the copies then overrun).
inline size_t checked_count(std::initializer_list<size_t> dims, size_t elem_size) {
    size_t n = 1;
    for (auto d : dims) {
        if (d != 0 && n > std::numeric_limits<size_t>::max() / d)
            throw std::overflow_error("array dimensions overflow size_t");
        n *= d;
    }
    if (elem_size != 0 && n > std::numeric_limits<size_t>::max() / elem_size)
        throw std::overflow_error("array size in bytes overflows size_t");
    return n;
}

// An element count passed to the C++ solvers as int (LAPACK 32-bit integers).
inline int checked_int(size_t v, const char* what) {
    if (v > size_t(std::numeric_limits<int>::max()))
        throw std::overflow_error(std::string(what) + " exceeds the 32-bit LAPACK range");
    return int(v);
}

// Fresh owned numpy array.
template <class T>
nb::ndarray<nb::numpy, T> new_array(std::initializer_list<size_t> shape) {
    const size_t n = checked_count(shape, sizeof(T));
    T* p = new T[n ? n : 1]();
    nb::capsule owner(p, [](void* q) noexcept { delete[] static_cast<T*>(q); });
    return nb::ndarray<nb::numpy, T>(p, shape, owner);
}

// Every structural property the C++ side indexes with is checked here: an
// out-of-range column index or a decreasing indptr used to be written straight into the band
// storage / V^T (heap writes out of bounds), the C++ never re-checks them.
template <class T>
conatus::Csr<T> to_csr(int n, IArr1 indptr, IArr1 indices, Arr1<T> data) {
    if (n < 0) throw std::invalid_argument("CSR: n must be >= 0, got " + std::to_string(n));
    if (indptr.shape(0) != size_t(n) + 1) throw std::invalid_argument("indptr must have n+1 entries");
    conatus::Csr<T> A;
    A.n = n;
    A.indptr.assign(indptr.data(), indptr.data() + indptr.shape(0));
    A.indices.assign(indices.data(), indices.data() + indices.shape(0));
    A.data.assign(data.data(), data.data() + data.shape(0));
    if (A.indices.size() != A.data.size() || A.indptr.front() != 0 ||
        size_t(A.indptr.back()) != A.data.size())
        throw std::invalid_argument("inconsistent CSR arrays (indptr[0] = 0, indptr[n] = nnz = "
                                    "len(indices) = len(data) required)");
    for (int i = 0; i < n; ++i)
        if (A.indptr[i + 1] < A.indptr[i])
            throw std::invalid_argument("CSR: indptr decreases at row " + std::to_string(i));
    for (std::size_t p = 0; p < A.indices.size(); ++p)
        if (A.indices[p] < 0 || A.indices[p] >= n)
            throw std::invalid_argument("CSR: column index " + std::to_string(A.indices[p]) +
                                        " outside [0, " + std::to_string(n) + ")");
    return A;
}

// The host IMEX integrator plus the owning reference to its Python right-hand side (see
// bind_linalg, finding 43). tp_traverse reports that reference to the garbage collector,
// tp_clear drops it (and the borrowing std::function first) when a cycle is collected.
template <class T>
struct ImexPy : conatus::ImexIntegrator<T> {
    using conatus::ImexIntegrator<T>::ImexIntegrator;
    nb::object py_rhs;

    static int tp_traverse(PyObject* self, visitproc visit, void* arg) {
#if PY_VERSION_HEX >= 0x03090000
        Py_VISIT(Py_TYPE(self));
#endif
        if (!nb::inst_ready(self)) return 0;
        ImexPy* p = nb::inst_ptr<ImexPy>(self);
        Py_VISIT(p->py_rhs.ptr());
        return 0;
    }
    static int tp_clear(PyObject* self) {
        ImexPy* p = nb::inst_ptr<ImexPy>(self);
        p->set_rhs({});                              // nothing borrows the callable any more
        p->py_rhs.reset();
        return 0;
    }
    static PyType_Slot slots[];
};
template <class T>
PyType_Slot ImexPy<T>::slots[] = {{Py_tp_traverse, (void*)ImexPy<T>::tp_traverse},
                                  {Py_tp_clear, (void*)ImexPy<T>::tp_clear},
                                  {0, nullptr}};

template <class T>
void bind_linalg(nb::module_& m, const char* suffix) {
    using BB = conatus::BorderedBanded<T>;
    std::string bb = std::string("BorderedBanded") + suffix;
    nb::class_<BB>(m, bb.c_str())
        .def("__init__", [](BB* self, int n, IArr1 indptr, IArr1 indices, Arr1<T> data,
                            std::vector<int> border, bool col_border) {
            new (self) BB(to_csr<T>(n, indptr, indices, data), border, col_border);
        }, nb::arg("n"), nb::arg("indptr"), nb::arg("indices"), nb::arg("data"), nb::arg("border"),
           nb::arg("col_border") = false)
        .def_prop_ro("n", &BB::n)
        .def_prop_ro("kl", &BB::kl)
        .def_prop_ro("ku", &BB::ku)
        .def_prop_ro("rank", &BB::rank)
        .def_prop_ro("col_border", &BB::col_border)
        .def_prop_ro("col_border_nnz", &BB::col_border_nnz)
        .def_prop_ro("capacitance_cond", &BB::capacitance_cond)
        .def("solve", [](const BB& s, Arr2<T> b, bool transpose) {
            // b: (nrhs, n) row-major = n x nrhs column-major.
            size_t nr = b.shape(0), n = b.shape(1);
            if (n != size_t(s.n())) throw std::invalid_argument("rhs length != n");
            const int nrhs = checked_int(nr, "number of right-hand sides");
            auto x = new_array<T>({nr, n});
            std::memcpy(x.data(), b.data(), sizeof(T) * nr * n);
            if (transpose) s.solve_transpose(x.data(), nrhs); else s.solve(x.data(), nrhs);
            return x;
        }, nb::arg("b"), nb::arg("transpose") = false);

    // Batch over modes/groups: one factorization per system, solves looped in C++.
    std::string bt = std::string("BorderedBandedBatch") + suffix;
    struct Batch {
        Batch() = default;
        Batch(const Batch&) = delete;
        Batch& operator=(const Batch&) = delete;
        std::vector<std::unique_ptr<BB>> sys;
    };
    nb::class_<Batch>(m, bt.c_str())
        .def("__init__", [](Batch* self, std::vector<std::tuple<int, IArr1, IArr1, Arr1<T>>> mats,
                            std::vector<int> border, bool col_border) {
            new (self) Batch();
            for (auto& [n, ip, ix, d] : mats)
                self->sys.push_back(std::make_unique<BB>(to_csr<T>(n, ip, ix, d), border, col_border));
        }, nb::arg("mats"), nb::arg("border"), nb::arg("col_border") = false)
        .def("__len__", [](const Batch& b) { return b.sys.size(); })
        .def_prop_ro("kl", [](const Batch& b) {      // common band (max over the systems)
            int k = 0;
            for (const auto& s : b.sys) k = std::max(k, s->kl());
            return k;
        })
        .def_prop_ro("ku", [](const Batch& b) {
            int k = 0;
            for (const auto& s : b.sys) k = std::max(k, s->ku());
            return k;
        })
        .def("solve", [](const Batch& bt, Arr2<T> b, bool transpose) {
            // b: (batch, n); every system must have the same n. transpose: A_k^T x = b_k
            // (plain transpose, no conjugation; same factors, D8/).
            size_t B = b.shape(0), n = b.shape(1);
            if (B != bt.sys.size()) throw std::invalid_argument("batch size mismatch");
            auto x = new_array<T>({B, n});
            std::memcpy(x.data(), b.data(), sizeof(T) * B * n);
            for (size_t k = 0; k < B; ++k) {
                if (size_t(bt.sys[k]->n()) != n) throw std::invalid_argument("rhs length != n");
                if (transpose) bt.sys[k]->solve_transpose(x.data() + k * n, 1);
                else bt.sys[k]->solve(x.data() + k * n, 1);
            }
            return x;
        }, nb::arg("b"), nb::arg("transpose") = false);

    // The Python callback F is owned by the bound object (ImexPy::py_rhs), not by the C++
    // std::function (which only borrows it): the type then exposes it to the garbage collector
    // (tp_traverse / tp_clear below), so a callback that refers to its integrator is a cycle the
    // collector sees and frees (review finding 43; it leaked before).
    using IM = ImexPy<T>;
    std::string im = std::string("Imex") + suffix;
    nb::class_<IM>(m, im.c_str(), nb::type_slots(ImexPy<T>::slots))
        .def("__init__", [](IM* self, int n, IArr1 mp, IArr1 mi, Arr1<T> md, IArr1 lp, IArr1 li,
                            Arr1<T> ld, std::vector<int> border, std::string scheme, double dt) {
            new (self) IM(to_csr<T>(n, mp, mi, md), to_csr<T>(n, lp, li, ld), border, scheme, dt);
        })
        .def("set_rhs", [](IM& s, nb::object f) {
            if (f.is_none()) {                       // F = 0
                s.set_rhs({});
                s.py_rhs.reset();
                return;
            }
            if (!PyCallable_Check(f.ptr())) throw nb::type_error("set_rhs: a callable F(x, t) or None");
            int n = s.n();
            s.py_rhs = f;                            // the owning reference, visible to the GC
            PyObject* fp = f.ptr();                  // borrowed: valid while py_rhs holds it
            s.set_rhs([fp, n](const T* x, double t, T* out) {
                size_t shape[1] = {size_t(n)};
                nb::ndarray<nb::numpy, const T> xv(x, 1, shape, nb::handle());
                nb::object r = nb::borrow<nb::object>(fp)(xv, t);
                auto ra = nb::cast<nb::ndarray<const T, nb::ndim<1>, nb::c_contig>>(r);
                if (ra.shape(0) != size_t(n)) throw std::invalid_argument("rhs callback: wrong length");
                std::memcpy(out, ra.data(), sizeof(T) * n);
            });
        }, nb::arg("f").none())
        .def("reset", [](IM& s) { s.reset(); },
             "Forget the multistep history and the iteration count (factorizations kept); the "
             "next run() starts a new trajectory.")
        .def_prop_ro("history_length", [](const IM& s) { return s.history_length(); })
        .def("run", [](IM& s, Arr1<T> x0, double t0, long nsteps) {
            size_t n = x0.shape(0);
            if (n != size_t(s.n())) throw std::invalid_argument("state length != n");
            auto x = new_array<T>({n});
            std::memcpy(x.data(), x0.data(), sizeof(T) * n);
            double t = s.run(x.data(), t0, nsteps);
            return nb::make_tuple(x, t);
        }, nb::arg("x0"), nb::arg("t0"), nb::arg("nsteps"))
        .def_prop_ro("n_factorizations", [](const IM& s) { return s.n_factorizations(); })
        .def_prop_ro("iteration", [](const IM& s) { return s.iteration(); })
        .def_prop_ro("scheme", [](const IM& s) { return s.scheme(); });
}


#ifdef CONATUS_WITH_CUDA
// ---- CUDA backend ------------------------------------------------------------------------------
// DeviceArray: a 2D (rows, cols) array resident on the device. Host <-> device copies happen only in
// from_host / to_host (explicit, §57); every GPU operation takes and returns DeviceArrays.
template <class T>
struct DeviceArray {
    conatus::Buffer<T> buf;
    size_t rows = 0, cols = 0;
    DeviceArray(size_t r, size_t c)
        : buf(checked_count({r, c}, sizeof(T)), conatus::MemorySpace::CUDA_DEVICE), rows(r), cols(c) {}
};

template <class T>
std::unique_ptr<DeviceArray<T>> dev_from_host(Arr2<T> a) {
    auto d = std::make_unique<DeviceArray<T>>(a.shape(0), a.shape(1));
    conatus::copy_from_host(d->buf, a.data(), d->rows * d->cols);
    return d;
}

template <class T>
void bind_device_array(nb::module_& m, const char* name) {
    using DA = DeviceArray<T>;
    nb::class_<DA>(m, name)
        .def_static("from_host", &dev_from_host<T>, nb::arg("array"))
        .def_static("zeros", [](size_t r, size_t c) { return std::make_unique<DA>(r, c); })
        .def("to_host", [](const DA& d) {
            auto h = new_array<T>({d.rows, d.cols});
            conatus::copy_to_host(h.data(), d.buf, d.rows * d.cols);
            return h;
        })
        .def("copy", [](const DA& d) {
            auto o = std::make_unique<DA>(d.rows, d.cols);
            conatus::copy(o->buf, d.buf);
            return o;
        })
        .def_prop_ro("shape", [](const DA& d) { return std::make_tuple(d.rows, d.cols); });
}

template <class T>
void bind_cuda_cheb(nb::module_& m) {
    using DA = DeviceArray<T>;
    m.def("cheb_forward", [](const DA& f, size_t N) {
        auto c = std::make_unique<DA>(f.rows, N);
        conatus::cuda::cheb_forward<T>(f.buf.data(), c->buf.data(), f.rows, f.cols, N);
        return c;
    }, nb::arg("grid"), nb::arg("N"));
    m.def("cheb_backward", [](const DA& c, size_t M) {
        auto f = std::make_unique<DA>(c.rows, M);
        conatus::cuda::cheb_backward<T>(c.buf.data(), f->buf.data(), c.rows, c.cols, M);
        return f;
    }, nb::arg("coef"), nb::arg("M"));
}

template <class T>
std::vector<conatus::Csr<T>> csr_list(std::vector<std::tuple<int, IArr1, IArr1, Arr1<T>>>& mats) {
    std::vector<conatus::Csr<T>> out;
    out.reserve(mats.size());
    for (auto& [n, ip, ix, d] : mats) out.push_back(to_csr<T>(n, ip, ix, d));
    return out;
}

template <class T>
void bind_cuda_linalg(nb::module_& m, const char* suffix) {
    using DA = DeviceArray<T>;
    using BB = conatus::cuda::BandedBatch<T>;
    std::string bb = std::string("BandedBatch") + suffix;
    nb::class_<BB>(m, bb.c_str())
        .def("__init__", [](BB* self, std::vector<std::tuple<int, IArr1, IArr1, Arr1<T>>> mats,
                            std::vector<int> border, bool pivot, bool level_schedule, bool level_merge,
                            bool col_border) {
            new (self) BB(csr_list<T>(mats), border, pivot, level_schedule, level_merge, col_border);
        }, nb::arg("mats"), nb::arg("border_rows"), nb::arg("pivot") = true,
           nb::arg("level_schedule") = true, nb::arg("level_merge") = true,
           nb::arg("col_border") = false)
        .def_prop_ro("col_border", &BB::col_border)
        .def_prop_ro("col_border_nnz", &BB::col_border_nnz)
        .def("solve_mapped_ptr", [](const BB& s, uintptr_t rhs, uintptr_t rows, uintptr_t x,
                                    uintptr_t cols, uintptr_t stream) {
            if (rhs == 0 || x == 0) throw std::invalid_argument("BandedBatch.solve_mapped_ptr: null device pointer");
            s.solve_mapped(reinterpret_cast<const T*>(rhs), reinterpret_cast<const long long*>(rows),
                           reinterpret_cast<T*>(x), reinterpret_cast<const long long*>(cols),
                           reinterpret_cast<void*>(stream));
        }, nb::arg("rhs"), nb::arg("rows"), nb::arg("x"), nb::arg("cols"), nb::arg("stream") = 0,
           "Solve with the gather/scatter fused in the kernel: b_m[i] = rhs[rows[m, i]], "
           "x[cols[m, i]] = x_m[i] (rows/cols: int64 (batch, n) device arrays, 0 = contiguous).")
        .def("solve_split_ptr", [](const BB& s, uintptr_t re, uintptr_t rows_c, uintptr_t rows_s,
                                   uintptr_t x, uintptr_t cols_c, uintptr_t cols_s, uintptr_t stream) {
            auto L = [](uintptr_t p) { return reinterpret_cast<const long long*>(p); };
            s.solve_split(reinterpret_cast<const double*>(re), L(rows_c), L(rows_s),
                          reinterpret_cast<double*>(x), L(cols_c), L(cols_s),
                          reinterpret_cast<void*>(stream));
        }, nb::arg("re"), nb::arg("rows_c"), nb::arg("rows_s"), nb::arg("x"), nb::arg("cols_c"),
           nb::arg("cols_s"), nb::arg("stream") = 0,
           "Complex solve of real data (complexified pairs): b = re[rows_c] + i re[rows_s], "
           "x[cols_c] = Re, x[cols_s] = Im (int64 (batch, n) device maps).")
        .def("probe_level_solve", &BB::probe_level_solve,
             "Diagnostics: per-block clock64 stamps of one level-scheduled solve (batch x 10)")
        .def_prop_ro("level_scheduled", &BB::level_scheduled)
        .def_prop_ro("level_depth_forward", &BB::level_depth_forward)
        .def_prop_ro("level_depth_backward", &BB::level_depth_backward)
        .def_prop_ro("level_steps_forward", &BB::level_steps_forward)
        .def_prop_ro("level_steps_backward", &BB::level_steps_backward)
        .def_prop_ro("level_merged", &BB::level_merged)
        .def_prop_ro("level_stream_bytes", &BB::level_stream_bytes)
        .def_prop_ro("level_note", &BB::level_note)
        .def("build_transpose_schedule", &BB::build_transpose_schedule,
             "Build the transposed level-scheduled streams (prepare_transpose included)")
        .def("release_band", &BB::release_band,
             "Free the band LU once the forward and transposed level schedules exist")
        .def_prop_ro("level_transposed", &BB::level_transposed)
        .def_prop_ro("band_released", &BB::band_released)
        .def_prop_ro("level_t_stream_bytes", &BB::level_t_stream_bytes)
        .def_prop_ro("level_t_note", &BB::level_t_note)
        .def_prop_ro("level_t_steps", [](const BB& s) {
            return std::make_tuple(s.level_t_steps_u(), s.level_t_steps_l()); })
        .def("solve", [](const BB& s, DA& b) {
            if (b.rows != size_t(s.batch()) || b.cols != size_t(s.n()))
                throw std::invalid_argument("BandedBatch.solve: expected a (batch, n) DeviceArray");
            s.solve(b.buf.data());
        }, nb::arg("b"), "Solve in place on the device.")
        .def("solve_ptr", [](const BB& s, uintptr_t ptr, int batch, int n, uintptr_t stream) {
            if (batch != s.batch() || n != s.n())
                throw std::invalid_argument("BandedBatch.solve_ptr: expected (batch, n) matching the factorization");
            if (ptr == 0) throw std::invalid_argument("BandedBatch.solve_ptr: null device pointer");
            s.solve(reinterpret_cast<T*>(ptr), reinterpret_cast<void*>(stream));
        }, nb::arg("ptr"), nb::arg("batch"), nb::arg("n"), nb::arg("stream") = 0,
           "Solve in place on caller-owned device memory (C-contiguous (batch, n), e.g. a CuPy array; "
           "launched on ``stream`` (a cudaStream_t as int; 0 = legacy default), capturable in a CUDA graph).")
        .def("solve_transpose", [](const BB& s, DA& b) {
            if (b.rows != size_t(s.batch()) || b.cols != size_t(s.n()))
                throw std::invalid_argument("BandedBatch.solve_transpose: expected a (batch, n) DeviceArray");
            s.solve_transpose(b.buf.data());
        }, nb::arg("b"), "Solve A^T x = b in place (plain transpose, same factors).")
        .def("prepare_transpose", [](const BB& s) { s.prepare_transpose(); },
             "Build the transposed Woodbury data now (otherwise done at the first transposed solve; "
             "call it before capturing a CUDA graph).")
        .def("solve_transpose_ptr", [](const BB& s, uintptr_t ptr, int batch, int n, uintptr_t stream) {
            if (batch != s.batch() || n != s.n())
                throw std::invalid_argument("BandedBatch.solve_transpose_ptr: expected (batch, n) matching the factorization");
            if (ptr == 0) throw std::invalid_argument("BandedBatch.solve_transpose_ptr: null device pointer");
            s.solve_transpose(reinterpret_cast<T*>(ptr), reinterpret_cast<void*>(stream));
        }, nb::arg("ptr"), nb::arg("batch"), nb::arg("n"), nb::arg("stream") = 0,
           "Transposed solve in place on caller-owned device memory (see solve_ptr).")
        .def_prop_ro("n", &BB::n)
        .def_prop_ro("kl", &BB::kl)
        .def_prop_ro("ku", &BB::ku)
        .def_prop_ro("rank", &BB::rank)
        .def_prop_ro("batch", &BB::batch)
        .def_prop_ro("pivot", &BB::pivot)
        .def_prop_ro("factor_ms", &BB::factor_ms)
        .def_prop_ro("capacitance_cond_max", &BB::capacitance_cond_max);

    using IM = conatus::cuda::Imex<T>;
    std::string im = std::string("Imex") + suffix;
    nb::class_<IM>(m, im.c_str())
        .def("__init__", [](IM* self, std::vector<std::tuple<int, IArr1, IArr1, Arr1<T>>> M,
                            std::vector<std::tuple<int, IArr1, IArr1, Arr1<T>>> L,
                            std::vector<int> border, std::string scheme, double dt, bool pivot) {
            new (self) IM(csr_list<T>(M), csr_list<T>(L), border, scheme, dt, pivot);
        }, nb::arg("M"), nb::arg("L"), nb::arg("border_rows"), nb::arg("scheme"), nb::arg("dt"),
           nb::arg("pivot") = true)
        .def("set_pointwise_rhs", [](IM& s, nb::object g, nb::object h) {
            size_t N = checked_count({size_t(s.n()), size_t(s.batch())}, sizeof(T));
            std::vector<T> gv, hv;
            auto grab = [&](nb::object o, std::vector<T>& v) {
                if (o.is_none()) return false;
                auto a = nb::cast<nb::ndarray<const T, nb::c_contig, nb::device::cpu>>(o);
                if (a.size() != N) throw std::invalid_argument("pointwise rhs: expected batch*n values");
                v.assign(a.data(), a.data() + N);
                return true;
            };
            bool hg = grab(g, gv), hh = grab(h, hv);
            s.set_pointwise_rhs(hg ? gv.data() : nullptr, hh ? hv.data() : nullptr);
        }, nb::arg("g").none(), nb::arg("h").none())
        .def("run", [](IM& s, DA& x, double t0, long nsteps) {
            if (x.rows != size_t(s.batch()) || x.cols != size_t(s.n()))
                throw std::invalid_argument("Imex.run: expected a (batch, n) DeviceArray");
            return s.run(x.buf.data(), t0, nsteps);
        }, nb::arg("x"), nb::arg("t0"), nb::arg("nsteps"))
        .def_prop_ro("n_factorizations", &IM::n_factorizations)
        .def_prop_ro("iteration", &IM::iteration)
        .def_prop_ro("scheme", &IM::scheme);
}

void bind_cuda(nb::module_& root) {
    nb::module_ m = root.def_submodule("cuda", "CUDA backend");
    m.def("available", &conatus::cuda::available);
    m.def("device_name", &conatus::cuda::device_name);
    m.def("synchronize", &conatus::cuda::synchronize);
    m.def("pool_stats", []() {
        auto p = conatus::cuda::pool_stats();
        return std::make_tuple(p.live_bytes, p.cached_bytes, p.n_alloc, p.n_reuse);
    });
    m.def("pool_trim", &conatus::cuda::pool_trim);
    m.def("plan_cache_size", &conatus::cuda::plan_cache_size);
    bind_device_array<double>(m, "DeviceArrayD");
    bind_device_array<cplx>(m, "DeviceArrayZ");
    using DD = DeviceArray<double>;
    using DZ = DeviceArray<cplx>;
    m.def("fft_c2c", [](const DZ& x, bool forward) {
        auto y = std::make_unique<DZ>(x.rows, x.cols);
        conatus::cuda::fft_c2c(x.buf.data(), y->buf.data(), x.rows, x.cols, forward);
        return y;
    }, nb::arg("x"), nb::arg("forward") = true);
    m.def("rfft", [](const DD& x) {
        auto y = std::make_unique<DZ>(x.rows, x.cols / 2 + 1);
        conatus::cuda::rfft(x.buf.data(), y->buf.data(), x.rows, x.cols);
        return y;
    });
    m.def("irfft", [](const DZ& X, size_t n) {
        if (X.cols != n / 2 + 1) throw std::invalid_argument("irfft: expected n/2+1 columns");
        auto y = std::make_unique<DD>(X.rows, n);
        conatus::cuda::irfft(X.buf.data(), y->buf.data(), X.rows, n);
        return y;
    }, nb::arg("X"), nb::arg("n"));
    bind_cuda_cheb<double>(m);
    bind_cuda_cheb<cplx>(m);
    m.def("realfourier_forward", [](const DD& f, size_t N) {
        auto c = std::make_unique<DD>(f.rows, N);
        conatus::cuda::realfourier_forward(f.buf.data(), c->buf.data(), f.rows, f.cols, N);
        return c;
    });
    m.def("realfourier_backward", [](const DD& c, size_t M) {
        auto f = std::make_unique<DD>(c.rows, M);
        conatus::cuda::realfourier_backward(c.buf.data(), f->buf.data(), c.rows, c.cols, M);
        return f;
    });
    m.def("complexfourier_forward", [](const DZ& f, size_t N) {
        auto c = std::make_unique<DZ>(f.rows, N);
        conatus::cuda::complexfourier_forward(f.buf.data(), c->buf.data(), f.rows, f.cols, N);
        return c;
    });
    m.def("complexfourier_backward", [](const DZ& c, size_t M) {
        auto f = std::make_unique<DZ>(c.rows, M);
        conatus::cuda::complexfourier_backward(c.buf.data(), f->buf.data(), c.rows, c.cols, M);
        return f;
    });
    bind_cuda_linalg<double>(m, "D");
    bind_cuda_linalg<cplx>(m, "Z");
    using A3 = nb::ndarray<const cplx, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
    m.def("dense_lu_reference", [](A3 A, Arr2<cplx> b, int repeats) {
        int batch = int(A.shape(0)), n = int(A.shape(1));
        if (A.shape(2) != size_t(n) || b.shape(0) != size_t(batch) || b.shape(1) != size_t(n))
            throw std::invalid_argument("dense_lu_reference: A (batch, n, n), b (batch, n)");
        auto r = conatus::cuda::dense_lu_reference(A.data(), b.data(), batch, n, repeats);
        auto x = new_array<cplx>({size_t(batch), size_t(n)});
        std::memcpy(x.data(), r.x.data(), sizeof(cplx) * r.x.size());
        return nb::make_tuple(x, r.factor_ms, r.solve_ms);
    }, nb::arg("A"), nb::arg("b"), nb::arg("repeats") = 10);
}
#endif

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "Conatus C++ core (host backend; CUDA backend in the `cuda` submodule when compiled)";
    // the package version, passed by CMake from python/conatus/__init__.py (one source)
    m.attr("__version__") = CONATUS_VERSION;

    // No host FFT: the CPU transforms are NumPy/SciPy (pocketfft) in Python; the FFTW-based
    // fft_c2c/rfft/irfft/cheb_*/realfourier_*/complexfourier_* were test-only and were removed
    // with FFTW (GPL) from every build.

    bind_linalg<double>(m, "D");
    bind_linalg<cplx>(m, "Z");
    m.def("cuda_compiled", &conatus::cuda_backend_compiled);
#ifdef CONATUS_WITH_CUDA
    bind_cuda(m);
#endif
}
