<div align="center">

# 🌀 Conatus

### *Define a PDE model, then simulate it, compute its stability and search for its invariant solutions.*

Conatus is experimental research software for fluid instability and dynamical systems.

**[Website](https://rubendahan.github.io/conatus/)** ·
**[Getting started](https://rubendahan.github.io/conatus/tutorials/getting-started/)** ·
**[Tutorials](https://rubendahan.github.io/conatus/tutorials/)** ·
**[Published cases](https://rubendahan.github.io/conatus/published/)** ·
**[Install](https://rubendahan.github.io/conatus/install/)**

![Python](https://img.shields.io/badge/Python-3.11%20|%203.12%20|%203.13-3776AB?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Linux-x86--64-FCC624?logo=linux&logoColor=black)
![CUDA](https://img.shields.io/badge/CUDA%2012-experimental-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/license-BSD--3-success)
![Version](https://img.shields.io/badge/version-0.2.1-blue)

</div>

## 📦 Install

```bash
pip install conatus==0.2.1   # CPU wheel for Linux x86-64, compiled core included
python -m conatus doctor         # what this machine can run, and what is missing
```

CUDA support (one NVIDIA GPU, CUDA 12 wheel attached to the
[v0.2.1 release](https://github.com/rubendahan/conatus/releases/tag/v0.2.1)) is experimental.
Building from source and the GPU wheel are described on the
[install page](https://rubendahan.github.io/conatus/install/).

## 📐 A first calculation: when does plane Poiseuille flow become unstable?

Small wavy disturbances of the flow between two plates obey the Orr-Sommerfeld equation. This
script writes that equation, computes the least stable mode at one Reynolds number, then searches
for the critical Reynolds number. Save it as `critical_reynolds.py` and run `python critical_reynolds.py`
([`web/home/critical_reynolds.py`](web/home/critical_reynolds.py)).

```python
# Linear stability of plane Poiseuille flow U(y) = 1 - y^2 between walls at y = -1 and y = 1.
# Orr-Sommerfeld equation for v(y) exp(i alpha (x - c t)): a mode grows when Im(c) > 0.
import conatus as cn
from conatus.expert import critical

m = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="plane Poiseuille", Re=1e4, alpha=1.0)
m.scalar("v")
m.given(U=lambda y: 1 - y**2)
m.eigenvalue("c", growth="imag")
m.define("L2(f) = dy(dy(f)) - alpha**2*f")
m.equations("L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + c*L2(v) = 0")
m.at("walls", "v = 0", "dy(v) = 0")

# Least stable mode at Re = 10000, alpha = 1 (modes kept only if found again at N = 96)
res = m.eigenvalues(N=64, drift=96, drift_tol=1e-6)
i = res.values.imag.argmax()
print(f"c = {res.values[i]:.8f}   residual below 1e-15: {res.residuals[i] < 1e-15}")

# Critical Reynolds number: the lowest Re at which Im(c) = 0 for some alpha
P = m.parametric(N=64, over=("Re", "alpha"), drift=96, drift_tol=1e-6)
crit = critical(P, "Re", bracket=(5000, 7000), over={"alpha": (0.98, 1.06)}, xtol=1e-9)
print(f"Re_c = {crit.value:.2f}   alpha_c = {crit.over['alpha']:.5f}")
```

It prints:

```text
c = 0.23752649+0.00373967j   residual below 1e-15: True
Re_c = 5772.22   alpha_c = 1.02055
```

Orszag (1971) gives c = 0.23752649 + 0.00373967i at Re = 10000, alpha = 1, and Re_c = 5772.22 at
alpha_c = 1.02056 (the script's 1.02055 differs in the last digit). The residual is the backward error of the computed eigenpair. With `drift=96`, a
mode is kept only if it is found again with 96 Chebyshev polynomials instead of 64. The
[stability tutorial](https://rubendahan.github.io/conatus/docs/tutorials/01_orr_sommerfeld.html)
continues from this script.

## 🧭 What else it does

- Time-dependent simulations of the same kind of model, with implicit-explicit time stepping
  ([first steps](https://rubendahan.github.io/conatus/tutorials/getting-started/), ten minutes).
- Channels, periodic boxes, disks, annuli, pipes, spheres, balls and spherical shells.
- Adjoint modes, transient growth, and Floquet analysis of time-periodic base flows.
- Steady states, travelling waves and periodic orbits by Newton-Krylov; continuation through folds
  and bifurcations.
- Lyapunov exponents, covariant Lyapunov vectors, tracers and dye in a simulated flow.

[`examples/`](https://github.com/rubendahan/conatus/tree/main/examples) holds further cases, each
checked against a published value. Four published studies are reproduced by the scripts of
[`examples/published/`](https://github.com/rubendahan/conatus/tree/main/examples/published): the
Orr-Sommerfeld spectrum of Orszag (1971), linearised pipe flow up to Re = 10⁷ (Meseguer & Trefethen
2003), the onset of Rayleigh-Bénard convection and the Kuramoto-Sivashinsky solutions of Cvitanović,
Davidchack & Siminos (2010). Each prints, for every published value, the computed value and the gap,
and its recorded output is in the same directory. The
[Published cases](https://rubendahan.github.io/conatus/published/) page draws them.

## 📚 Getting started

1. [Install](https://rubendahan.github.io/conatus/install/) and run `python -m conatus doctor`.
2. [First steps](https://rubendahan.github.io/conatus/tutorials/getting-started/): the Burgers equation in ten minutes.
3. [A stability calculation](https://rubendahan.github.io/conatus/docs/tutorials/01_orr_sommerfeld.html): the example above, with the eigenfunction and a resolution check.
4. [One problem from onset to simulation](https://rubendahan.github.io/conatus/docs/tutorials/00_getting_started.html): Rayleigh-Bénard convection.

The [documentation](https://rubendahan.github.io/conatus/documentation/) has the user guide, the
API reference and the numerical methods.

## ⚠️ Status and limits

- **Young API.** The interface can still change. Changes come with a deprecation warning and a line
  in the [changelog](https://github.com/rubendahan/conatus/blob/main/CHANGELOG.md).
- **CPU speed.** The CPU path is a NumPy reference implementation. It suits eigenvalue problems,
  small grids and setting a run up. Large grids can be slow on it.
- **One process.** 0.2.1 runs a simulation on one CPU process or one GPU. No multi-GPU or MPI yet.

## 🔧 Optional tools

- **Studio**: watch a running simulation in a browser (`pip install "conatus[studio]"`,
  [Studio guide](https://rubendahan.github.io/conatus/docs/guide/studio.html)). A session needs
  both its cookie and a key that the app keeps, so another service on the same machine cannot reuse
  the cookie.
- **Coding agents**: `python -m conatus agent` describes the installation and the API in JSON;
  [`AGENTS.md`](https://github.com/rubendahan/conatus/blob/main/AGENTS.md) is the page to give them.

## 📄 License

BSD-3-Clause, see [LICENSE](https://github.com/rubendahan/conatus/blob/main/LICENSE). Bundled third-party components and their licences are
listed in [THIRD_PARTY_NOTICES.md](https://github.com/rubendahan/conatus/blob/main/THIRD_PARTY_NOTICES.md), with the full texts in
[`licenses/`](https://github.com/rubendahan/conatus/tree/main/licenses). To cite Conatus, see [CITATION.cff](https://github.com/rubendahan/conatus/blob/main/CITATION.cff).

Made by Ruben Dahan. Conatus is developed with AI coding agents under the direction of one researcher.
