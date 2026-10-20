# Contributing to Conatus

Issues and pull requests are welcome. A few rules keep the library trustworthy:

1. **No silent substitution.** Never change a basis, resolution, boundary condition, dealiasing
   factor, time scheme, tolerance, inner product or adjoint definition behind the user's back.
   The code *suggests* (an error with a `fix:` line, a warning, `explain()`); it never modifies.
   Errors derive from `conatus.errors.ConatusError` and say what, where, why and what to do.
2. **Tests with every change.** `ruff check .` and `pytest -m "not slow and not gpu"` must pass
   before a pull request is merged. Long or GPU tests are marked `slow` / `gpu`.
3. **Validate by residual.** An eigenmode, a solution or an orbit is accepted on a reported
   residual (and manufactured solutions, observed convergence order, spectral tails), never on a
   raw value alone. A comparison with a published number cites the source.
4. **Every bug gets a test**, written so that it fails without the fix.
5. **Every key object has `explain()`.** Public names go through `conatus.expert`.
6. English everywhere (code, comments, docs); type hints and short docstrings; small commits with
   imperative messages.

## Building and testing from a clone

The C++ core is compiled with CMake (a C++20 compiler, GCC 13 or newer, and LAPACK/BLAS; the
development environment of `ci/environment.yml` has all of them):

```bash
micromamba create -n conatus-dev -f ci/environment.yml && micromamba activate conatus-dev
cmake -S . -B build/core -G Ninja -DCMAKE_BUILD_TYPE=Release -DCONATUS_CUDA=OFF
cmake --build build/core -j 8
export CN_BUILD=$PWD/build PYTHONPATH=$PWD/python:$PYTHONPATH
ruff check . && python -m pytest -q -m "not slow and not gpu"
```

`pip install -e ".[dev,tutorials]"` builds the same core into an editable install instead. Add
`-DCONATUS_CUDA=ON` (with `nvcc` in the environment) for the GPU backend; the install page of the
documentation lists the tested CUDA versions.

There is no continuous integration on this repository yet: run the commands above before opening a
pull request, and say in it which ones you ran.
