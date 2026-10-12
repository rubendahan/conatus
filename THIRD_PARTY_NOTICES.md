# Third-party notices

Conatus itself is distributed under the BSD 3-Clause licence (`LICENSE`, Ruben Dahan and
contributors). Some files that ship with it come from other projects, under their own licences.
Each is listed below: its version, its licence (SPDX identifier), where it ships, and where its
licence text is. The texts are in `licenses/` in the repository, in the source distribution, and in
`.dist-info/licenses/licenses/` of every wheel. How this list was made and checked:
.

Nothing here is copied into the Conatus sources (`python/conatus/`, `core/`): the components below
are linked, bundled by the build, or served by the documentation and the website.

## 1. Binary wheels (`conatus-<version>+cpu` and `+cu12`)

`auditwheel repair` copies these shared libraries into `conatus.libs/` of the wheel
(`packaging/README.md`); the compiled core `conatus._core` links them. They are the conda-forge
builds of the wheel build environment (`packaging/build_wheels.sh`: OpenBLAS, pthreads build).
No GPL library is bundled: FFTW, which earlier development builds linked for test-only transforms,
was dropped from every build in 0.1.0.

| Component | Version | Licence | Licence text |
|---|---|---|---|
| OpenBLAS (`libopenblas`), the OpenBLAS Project | 0.3.34 | BSD-3-Clause | `licenses/BSD-3-Clause_openblas.txt` |
| LAPACK (reference, built into OpenBLAS), University of Tennessee et al. | 3.11 | BSD-3-Clause (LAPACK modified BSD) | `licenses/BSD-3-Clause_lapack.txt` |
| GNU Fortran runtime (`libgfortran`), GCC | 16.2.0 | GPL-3.0-only WITH GCC-exception-3.1 | `licenses/GPL-3.0-only.txt`, `licenses/GCC-exception-3.1.txt` |
| GNU quad-precision library (`libquadmath`), GCC | 16.2.0 | LGPL-2.1-or-later | `licenses/LGPL-2.1-or-later.txt` |

Compiled into `conatus._core` itself:

| Component | Version | Licence | Licence text |
|---|---|---|---|
| nanobind, Wenzel Jakob | 3.1.0 | BSD-3-Clause | `licenses/BSD-3-Clause_nanobind.txt` |
| tsl::robin_map (part of nanobind), Thibaut Goetghebuer-Planchon | (nanobind 3.1.0) | MIT | `licenses/MIT_robin-map.txt` |
| GCC C++ and support runtimes (`-static-libstdc++ -static-libgcc`) | 13.4 | GPL-3.0-only WITH GCC-exception-3.1 | `licenses/GPL-3.0-only.txt`, `licenses/GCC-exception-3.1.txt` |

**Corresponding source of the GPL and LGPL components.** `libgfortran` and
`libquadmath` (GCC 16.2.0): <https://gcc.gnu.org/> and the conda-forge recipe
<https://github.com/conda-forge/ctng-compilers-feedstock>. The libraries in `conatus.libs/` are
those builds unmodified, linked dynamically (a user may replace them).

The CUDA flavour does **not** bundle NVIDIA libraries: `libcudart`, `libcufft`, `libcublas` come
from the `nvidia-*-cu12` wheels it requires, under the NVIDIA CUDA Toolkit licence of those
packages. CuPy (MIT), NumPy, SciPy, SymPy and h5py (BSD-3-Clause) are dependencies installed by
pip, not bundled.

## 2. Documentation site (`docs/site`, built with Sphinx)

Served from `_static/vendor/`, fetched from the npm registry and verified by
`docs/site/vendor/fetch.py` against `docs/site/vendor/lock.json` (version and sha512 of each
tarball). Each package's licence text is copied next to its files (`_static/vendor/<package>/LICENSE`).

| Component | Version | Licence | Licence text |
|---|---|---|---|
| MathJax (`mathjax`) | 4.1.3 | Apache-2.0 | `licenses/Apache-2.0.txt`; `_static/vendor/mathjax/LICENSE` |
| MathJax New Computer Modern font (`@mathjax/mathjax-newcm-font`) | 4.1.3 | Apache-2.0 (declared in its `package.json`; the tarball ships no licence file, so the MathJax text of the same release is copied, `LICENSE.note.txt` says so) | `licenses/Apache-2.0.txt`; `_static/vendor/@mathjax/mathjax-newcm-font/LICENSE` |
| IBM Plex Sans (`@fontsource/ibm-plex-sans`), IBM Corp. | 5.3.0 | OFL-1.1 | `licenses/OFL-1.1_ibm-plex-sans.txt`; `_static/vendor/fonts/ibm-plex-sans/LICENSE` |
| JetBrains Mono (`@fontsource/jetbrains-mono`), the JetBrains Mono Project Authors | 5.3.0 | OFL-1.1 | `licenses/OFL-1.1_jetbrains-mono.txt`; `_static/vendor/fonts/jetbrains-mono/LICENSE` |
| IM Fell English (`@fontsource/im-fell-english`) | 5.3.0 | OFL-1.1 | `licenses/OFL-1.1_im-fell-english.txt`; `_static/vendor/fonts/im-fell-english/LICENSE` |
| IM Fell English SC (`@fontsource/im-fell-english-sc`) | 5.3.0 | OFL-1.1 | `licenses/OFL-1.1_im-fell-english.txt` (same text); `_static/vendor/fonts/im-fell-english-sc/LICENSE` |

The glyphs of the MathJax font are generated from the New Computer Modern fonts (Antonis
Tsolomitis), which CTAN distributes under the GUST Font License; whether that licence also covers
the generated web fonts is an open question to settle with the MathJax project
("Open items").

Written into the built site by the documentation tools (not stored in the repository):

| Component | Version | Licence |
|---|---|---|
| Sphinx (`searchtools.js`, `doctools.js`, `basic.css`, ...) | 9.1.0 | BSD-2-Clause |
| Snowball stemmers (`base-stemmer.js`, `english-stemmer.js`, shipped by Sphinx) | (Sphinx 9.1.0) | BSD-3-Clause |
| Furo theme | 2025.12.19 | MIT (its `furo.js.LICENSE.txt` ships next to it) |
| sphinx-design | 0.7.0 | MIT |
| sphinx-copybutton, with clipboard.js 2.0.8 (Zeno Rocha) | 0.5.2 | MIT |

## 3. Website (`web/`)

Bundled by Vite into the built site. The bundles drop
the licence comments, so `web/build.sh` copies this file and `licenses/`
into the build output (`/THIRD_PARTY_NOTICES.md`, `/licenses/`).

| Component | Version | Licence | Copyright |
|---|---|---|---|
| React (`react`), with `scheduler` 0.23.2 | 18.3.1 | MIT | Facebook, Inc. and its affiliates |
| React DOM (`react-dom`) | 18.3.1 | MIT | Facebook, Inc. and its affiliates |
| React Router (`react-router`, `react-router-dom`), with `@remix-run/router` 1.23.4 | 6.30.6 | MIT | React Training LLC 2015–2019, Remix Software Inc. 2020–2021, Shopify Inc. 2022–2023 |
| MiniSearch (`minisearch`, website search) | 7.2.0 | MIT | 2022 Luca Ongaro |
| `loose-envify` 1.4.0, `js-tokens` 4.0.0 (dependencies of React, build-time transform) | — | MIT | Andres Suarez; Simon Lydell |
| The four @fontsource faces of section 2 (`.woff2` files in the bundle) | 5.3.0 | OFL-1.1 | as in section 2 |

The MIT licence text, with each copyright line above in place of its notice:

```text
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

Build tools that do not ship (Vite, TypeScript, Lighthouse, Puppeteer and the rest of
`devDependencies`) are not listed.

## 4. Dedalus

Dedalus (GPL-3.0, the Dedalus Project developers) is not distributed with Conatus and the
`conatus` package never imports it. It runs, when installed, in test and benchmark oracles and in
the optional check script of tutorial 14 (`docs/site/tutorials/code/t14b_dedalus_check.py`). Two
benchmark oracles, `benchmarks/dedalus/rb2d.py` and `benchmarks/dedalus/poisson2d.py`, follow the
set-up of a Dedalus example and say so in their header; `benchmarks/` is excluded from the source
distribution and is not in the wheel. The official Dedalus examples used by the translator tests
are fetched at test time into `scratch/` and never stored in the repository.
