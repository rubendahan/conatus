cmake_minimum_required(VERSION 3.24)
# One version for the package and its compiled core ( Z6): read
# from __version__ in python/conatus/__init__.py, the single source pyproject also reads (the core
# said "0.0.1", hard-coded, next to a 0.1.0 package). CMake's project(VERSION) takes numbers only
# (no "rc1"), so the full string is passed to the module as CONATUS_VERSION; the numeric part is
# the project version. Editing __init__.py re-runs the configure step (CMAKE_CONFIGURE_DEPENDS).
set(_cn_init "${CMAKE_CURRENT_SOURCE_DIR}/python/conatus/__init__.py")
file(STRINGS "${_cn_init}" _cn_version_line REGEX "^__version__ *= *\"[^\"]+\"")
string(REGEX REPLACE "^__version__ *= *\"([^\"]+)\".*" "\\1" CONATUS_VERSION "${_cn_version_line}")
if(NOT CONATUS_VERSION MATCHES "^[0-9]+\\.[0-9]+")
    message(FATAL_ERROR "Conatus: no __version__ = \"X.Y...\" line in ${_cn_init}")
endif()
string(REGEX MATCH "^[0-9]+(\\.[0-9]+)*" _cn_version_numeric "${CONATUS_VERSION}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_cn_init}")
project(conatus VERSION ${_cn_version_numeric} LANGUAGES CXX)
message(STATUS "Conatus version ${CONATUS_VERSION}")

option(CONATUS_CUDA "Build the CUDA backend" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CONATUS_CUDA)
    # A100 (sm_80) and L40S (sm_89); nvcc from the conda env, host compiler = the env g++ 13.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80 89)
    endif()
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    enable_language(CUDA)
endif()

add_subdirectory(core)
