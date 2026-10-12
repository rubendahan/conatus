# Conatus C++ core: static library + nanobind module conatus._core.
find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# LAPACK/BLAS from ONE prefix: the conda env ($CONDA_PREFIX) by default, or
# -DCONATUS_DEPS_PREFIX=<prefix> (conda-build passes $PREFIX). No system fallback on purpose
# (NO_DEFAULT_PATH): a /usr/lib reference BLAS would silently change the numerics/timings.
# Each path can still be given explicitly (-DLAPACK_LIB=... -DBLAS_LIB=...; with pip:
# -Ccmake.define.LAPACK_LIB=...), which skips the search. The core has no FFT of its own: the
# production transforms are NumPy/SciPy (pocketfft) on the CPU and cuFFT on the GPU (FFTW, GPL,
# was linked only by test-only host transforms and was dropped).
set(CONATUS_DEPS_PREFIX "$ENV{CONDA_PREFIX}" CACHE PATH
    "Prefix holding LAPACK and BLAS (default: $CONDA_PREFIX)")
set(_cn_deps_hint
    "Conatus looks for LAPACK/BLAS in CONATUS_DEPS_PREFIX='${CONATUS_DEPS_PREFIX}' only.\n"
    "Fix: activate a conda env that has them\n"
    "  (micromamba|conda) install -c conda-forge liblapack libblas\n"
    "or point CMake at them: -DCONATUS_DEPS_PREFIX=<prefix>, or "
    "-DLAPACK_LIB=<liblapack.so> -DBLAS_LIB=<libblas.so>\n"
    "  (through pip: pip install . -Ccmake.define.CONATUS_DEPS_PREFIX=<prefix>).")
if(CONATUS_DEPS_PREFIX)   # empty prefix: search nothing (never "/lib", i.e. the system)
    find_library(LAPACK_LIB lapack PATHS "${CONATUS_DEPS_PREFIX}/lib" NO_DEFAULT_PATH)
    find_library(BLAS_LIB blas PATHS "${CONATUS_DEPS_PREFIX}/lib" NO_DEFAULT_PATH)
endif()
foreach(_cn_var LAPACK_LIB BLAS_LIB)
    if(NOT ${_cn_var})
        message(FATAL_ERROR "Conatus: ${_cn_var} not found.\n" ${_cn_deps_hint})
    endif()
endforeach()
message(STATUS "Conatus deps: LAPACK=${LAPACK_LIB} BLAS=${BLAS_LIB}")
# Directories of the libraries actually found (build-tree rpath, also right with explicit paths).
set(_cn_dep_dirs)
foreach(_cn_lib ${LAPACK_LIB} ${BLAS_LIB})
    get_filename_component(_cn_dir "${_cn_lib}" DIRECTORY)
    list(APPEND _cn_dep_dirs "${_cn_dir}")
endforeach()
list(REMOVE_DUPLICATES _cn_dep_dirs)

add_library(conatus_core STATIC src/memory.cpp)
target_include_directories(conatus_core PUBLIC include)
target_link_libraries(conatus_core PUBLIC ${LAPACK_LIB} ${BLAS_LIB})
target_compile_features(conatus_core PUBLIC cxx_std_20)
set_target_properties(conatus_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(conatus_core PRIVATE -Wall -Wextra)

if(CONATUS_CUDA)
    # CUDA toolkit from the conda env (Spack is not mounted on the GPU nodes, E2).
    find_package(CUDAToolkit REQUIRED)
    add_library(conatus_cuda STATIC src/cuda/memory.cu src/cuda/transforms.cu src/cuda/banded.cu
                                    src/cuda/imex.cu)
    target_include_directories(conatus_cuda PUBLIC include)
    target_compile_features(conatus_cuda PUBLIC cxx_std_20)
    target_compile_options(conatus_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set_target_properties(conatus_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON
                          CUDA_SEPARABLE_COMPILATION OFF)
    target_link_libraries(conatus_cuda PUBLIC CUDA::cudart CUDA::cufft CUDA::cublas
                          ${LAPACK_LIB} ${BLAS_LIB})
    target_compile_definitions(conatus_core PUBLIC CONATUS_WITH_CUDA)
    target_compile_definitions(conatus_cuda PUBLIC CONATUS_WITH_CUDA)
    target_link_libraries(conatus_core PUBLIC conatus_cuda)
endif()

nanobind_add_module(_core NB_STATIC bindings/module.cpp)
target_link_libraries(_core PRIVATE conatus_core)
# conatus._core.__version__ = the package version (top-level CMakeLists.txt reads it from
# python/conatus/__init__.py)
target_compile_definitions(_core PRIVATE CONATUS_VERSION="${CONATUS_VERSION}")
# LD_LIBRARY_PATH is cleared on purpose (env.sh, E3): the module always carries an rpath.
# * Build tree (core/build.sh, tests from $CN_BUILD): the directories of the found libraries,
#   absolute (unchanged behaviour: $CONDA_PREFIX/lib in the dev env).
# * Installed (pip install . / wheel / conda package): conatus/_core*.so lands in
#   <prefix>/lib/pythonX.Y/site-packages/conatus/, so "$ORIGIN/../../.." is <prefix>/lib of the env
#   it is installed INTO (relocatable: a wheel built in env A and installed in env B uses B's LAPACK).
#   No absolute fallback to the build env: if the target env lacks lapack the import fails
#   loudly instead of silently loading another env's libraries. Override with
#   -DCONATUS_INSTALL_RPATH=... (e.g. a venv whose libraries live elsewhere).
set(CONATUS_INSTALL_RPATH "$ORIGIN/../../.." CACHE STRING "RPATH of the installed conatus._core")
set_target_properties(_core PROPERTIES
    BUILD_RPATH "${_cn_dep_dirs}"
    INSTALL_RPATH "${CONATUS_INSTALL_RPATH}")
# pip install . (scikit-build-core): the module goes into the conatus package of the wheel.
install(TARGETS _core LIBRARY DESTINATION conatus)
# The conda-forge gcc appends its own <build env>/lib to every link through its specs file, which
# CMake does not see: after the install-time rpath rewrite the installed module would still end with
# the BUILD env's lib as a last-resort search path (a silent fallback to another env's LAPACK/BLAS).
# Reset the installed module's RPATH to exactly CONATUS_INSTALL_RPATH.
install(CODE "file(RPATH_SET FILE \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/conatus/$<TARGET_FILE_NAME:_core>\"
                            NEW_RPATH \"${CONATUS_INSTALL_RPATH}\")")
