# Installation

Conatus 0.1.0 is installed **from a binary wheel** (no compiler needed) or **from source**. The CPU
wheel is on PyPI (`pip install conatus`); the CUDA 12 wheel (beta) is a file attached to the
[GitHub release](https://github.com/rubendahan/conatus/releases/tag/v0.1.0). The compiled C++ core, inside
both wheels, is required for time integration; the GPU backend is optional.


## Requirements

- Linux x86-64, Python 3.11, 3.12 or 3.13 (`requires-python = ">=3.11,<3.14"`). **Tested**:
  3.12 runs the whole test suite; on 3.11, 3.12 and 3.13 the wheel installs, imports, solves a
  problem and passes `doctor` in a clean manylinux_2_28 image (a newer Python is refused by pip until
  it is tested). 3.12 is recommended.
- NumPy, SciPy, SymPy and h5py (the pip dependencies; h5py writes the outputs and checkpoints).
  With these the IR and the eigenvalue, boundary-value, Newton and continuation solvers work.
- For the tutorials: matplotlib (the `tutorials` extra, below). The optional Dedalus check of tutorial 14 (`t14b_dedalus_check.py`) needs Dedalus.
- **The compiled C++20 core** (`conatus._core`: inside the binary wheels; from source, built with
  CMake + nanobind, a C++20 compiler, GCC ≥ 13, and LAPACK and BLAS from the conda environment):
  **required for time integration**
  (`m.simulate(...)` refuses with `MissingDependencyError` without it) and for the banded solves.
  `python -m conatus doctor` lists it as required.
- For GPUs: an NVIDIA GPU (the CUDA wheel, beta, is built for compute capability 8.0 and 8.9: A100 and L40S), the
  CUDA toolkit (12.x) and CuPy: see [GPU (CUDA) install](#gpu-cuda-install) for the tested versions
  and the exact commands.

## From a binary wheel (recommended)

Two flavours, for Linux x86-64 with glibc 2.28 or newer (RHEL, Rocky or Alma 8, Ubuntu 20.04 and
later) and CPython 3.11, 3.12 or 3.13:

| Flavour | Where | Contains |
|---|---|---|
| CPU | PyPI: `pip install conatus` | the compiled core (CPU), OpenBLAS with LAPACK |
| CUDA 12 (beta) | a `+cu12` wheel attached to the GitHub release | the same, plus the CUDA core (A100 sm_80, L40S sm_89) |

In a fresh virtual environment (no conda, no compiler):

```bash
python3.12 -m venv venv && . venv/bin/activate
pip install "conatus[tutorials]"
python -m conatus --version     # conatus, compiled core and Python versions
python -m conatus doctor        # what this machine can run, and what is missing
```

In the release test (a fresh virtual environment on a cluster node, dependencies downloaded from
PyPI) the install took under 30 s.

The CUDA wheel pulls the NVIDIA runtime wheels (CUDA runtime, cuFFT, cuBLAS: about 700 MB) and,
with the `gpu` extra, CuPy: see [GPU (CUDA) install](#gpu-cuda-install). `pip show conatus` says
which flavour is installed (`0.1.0` from PyPI, `0.1.0+cu12` for the CUDA wheel).

**Threads.** NumPy, SciPy and the core each load their own OpenBLAS, and each starts by default
one thread per CPU, so the three compete for the same CPUs. On a node where the process could use
8 CPUs, a Floquet computation of the test set took about 20 minutes with the defaults, 26 s with
`OMP_NUM_THREADS=1` and 16 s with `OMP_NUM_THREADS=4`. Set `export OMP_NUM_THREADS=<half the
CPUs>` before starting Python; Conatus does not change thread counts for you, and
`python -m conatus doctor` prints the BLAS libraries loaded, their thread counts and the value to set.

## From source: install the package

The same recipe as the README: a conda environment for the compilers and libraries, then
`pip install ".[tutorials]"`, which compiles the core, installs it inside the package and adds
what the tutorials need (matplotlib). About two minutes on a cluster node with a warm package cache.

```bash
git clone https://github.com/rubendahan/conatus && cd conatus
micromamba create -n conatus -c conda-forge python=3.12 gxx_linux-64=13 liblapack libblas
micromamba activate conatus
pip install ".[tutorials]"      # CPU core + matplotlib (the GPU build: next section)
python -m conatus doctor        # what this machine can run, and what is missing
```

`pip install .` alone (without the extra) installs the library without matplotlib; the
tutorials then stop at their first figure, and `doctor` prints the line that installs it.

An installed package writes nothing inside its environment: the live-monitoring spool
(`sim.watch`) defaults to `$XDG_CACHE_HOME/conatus/telemetry` (`~/.cache/conatus/telemetry`);
set `CN_TELEMETRY`, `CN_CACHE` or `CN_TMP` to move it, and `CN_RUNS` (or `root=`) for the run
registry. `doctor` prints these directories on its first line.

`pip wheel .` builds a wheel of the same package (for another machine with the same
environment).

## GPU (CUDA) install

The GPU backend (`backend="cuda"`) needs three things: the core compiled with CUDA
(`CONATUS_CUDA=ON`), CuPy for the same CUDA major version, and an NVIDIA GPU with a recent enough
driver. Compiling needs no GPU (a CPU or login node is fine); running does.

**Tested combination** (October 2026; recipe 1 on both GPUs, recipe 2 on an A100): CUDA 12.9 (nvcc 12.9.86, cuFFT
11.4.1, cuBLAS 12.9.2), CuPy 14.2.0, GCC 13, Python 3.12, NVIDIA driver 580, on A100 (compute
capability 8.0) and L40S (8.9), the two architectures the core is built for by default. Other
12.x toolkits are expected to work but are not tested; CUDA 11 and 13 are not.

Recipe 1, conda (the development environment; one CuPy, from conda-forge):

```bash
micromamba create -n conatus -c conda-forge "python=3.12" gxx_linux-64=13 liblapack libblas \
    cuda-version=12.9 cuda-nvcc cuda-cudart-dev libcufft-dev libcublas-dev cupy-core
micromamba activate conatus
pip install . -Ccmake.define.CONATUS_CUDA=ON \
    -Ccmake.define.CMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc \
    -Ccmake.define.CMAKE_CUDA_HOST_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ \
    -Ccmake.define.CUDAToolkit_ROOT=$CONDA_PREFIX
```

Do not add the `gpu` extra here: it would install a second CuPy from pip next to conda's.

Recipe 0, the CUDA binary wheel (beta; no compiler, no CUDA toolkit; the driver must support
CUDA 12.9, i.e. driver ≥ 575 or an older 12.x driver through CUDA minor-version compatibility).
Download the `conatus-0.1.0+cu12-cp3XX-…whl` file of your Python from the
[GitHub release](https://github.com/rubendahan/conatus/releases/tag/v0.1.0), then:

```bash
python3.12 -m venv venv && . venv/bin/activate
pip install "./conatus-0.1.0+cu12-cp312-cp312-<platform>.whl[gpu,tutorials]"   # the file you downloaded
```

Recipe 2, a wheel for a plain virtual environment (the CUDA libraries then come as pip
wheels, no CUDA module and no `LD_LIBRARY_PATH` needed at run time). Build the wheel once in the
environment of recipe 1 (compiling needs `nvcc`), then install it anywhere with the `gpu` extra:

```bash
pip wheel . -w wheels/ -Ccmake.define.CONATUS_CUDA=ON \
    -Ccmake.define.CMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc \
    -Ccmake.define.CMAKE_CUDA_HOST_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ \
    -Ccmake.define.CUDAToolkit_ROOT=$CONDA_PREFIX            # in the conda environment
python -m venv venv && venv/bin/pip install "$(ls wheels/conatus-*.whl)[gpu]"   # anywhere
```

The `gpu` extra installs `cupy-cuda12x`; the tested set was CuPy 14.2.0 with the CUDA 12.9 runtime,
cuFFT 11.4.1 and cuBLAS 12.9.2 wheels. Building directly against a
system CUDA toolkit (`-Ccmake.define.CUDAToolkit_ROOT=/usr/local/cuda`) is expected to work but is
not tested.

Another GPU architecture: add `-Ccmake.define.CMAKE_CUDA_ARCHITECTURES=90` (for example); the
default is `80;89`. Then, **on a GPU node** (in a batch job on a cluster):

```bash
python -m conatus doctor     # "GPU (backend='cuda')": ok, with the device's compute capability
```

`doctor` names what is missing: CuPy, a core built without CUDA, or no visible GPU (each case is
said separately; the same text appears when `backend="cuda"` is refused).

## From source: development tree

To work on Conatus itself, build the core in place and use the package from the tree:

```bash
micromamba create -n conatus-dev -f ci/environment.yml      # the development environment
micromamba activate conatus-dev
cmake -S . -B build/core -G Ninja -DCMAKE_BUILD_TYPE=Release -DCONATUS_CUDA=OFF
cmake --build build/core -j 8
export CN_BUILD=$PWD/build PYTHONPATH=$PWD/python:$PYTHONPATH
ruff check . && python -m pytest -q -m "not slow and not gpu"   # lint + fast tests
```

On a cluster, keep the environment and the build on a scratch or project file system rather than
in a small home directory, and run anything heavy (tests, GPU work) in batch jobs.

## Check your installation

```bash
python -m conatus doctor   # compiled core, GPU: what this machine can run, what to do otherwise
python -m conatus --version   # conatus, compiled core and Python versions: the first line of a bug report
```

`doctor` marks each line `[ok]`, `[missing]` (required: the exit code is 1), `[warning]` (something
may be wrong: a core built for another version, little disk space, an untested Python) or
`[optional]` (an optional component that is absent: GPU, CuPy, Dedalus, ipywidgets,
an undeclared run registry; nothing is wrong).

```python
import conatus as cn
m = cn.Model(cn.Box(x=cn.walls(0, 1)), name="heat", kappa=1.0)
T = m.scalar("T")
m.equations("dt(T) - kappa*lap(T) = 0")
m.at("walls", T=0)
print(m.explain())
```

then run {doc}`tutorials/01_orr_sommerfeld` (under a minute; eigenvalues only, so it runs even
from a source tree whose core is not built yet). For the GPU path, the
second half of {doc}`tutorials/02_rayleigh_benard` runs the same problem on the device and
compares with the CPU.

## Building this documentation

The site is built with Sphinx, MyST and Furo in a **separate** environment, so that documentation
tools never modify the compute environment:

```bash
bash docs/site/build.sh --setup   # once: creates the docs environment (docs/site/requirements.txt)
bash docs/site/build.sh           # HTML pages, then open html/index.html in the output folder
```

Tutorial outputs are produced by executing the scripts (`docs/site/run_tutorials.py`) and are
included verbatim in the pages.
