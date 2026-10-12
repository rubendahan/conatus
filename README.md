<div align="center">

# 🌀 Conatus

### *Spectral PDEs and dynamical systems, from equations written as on paper.*

Write the geometry, the unknowns and the equations as text. Conatus builds the spectral
discretisation, then lets you simulate the model or take apart its dynamics.

**[Website](https://rubendahan.github.io/conatus/)** ·
**[Tutorials](https://rubendahan.github.io/conatus/tutorials/)** ·
**[Install](https://rubendahan.github.io/conatus/install/)**

![Python](https://img.shields.io/badge/Python-3.11%20|%203.12%20|%203.13-3776AB?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Linux-x86--64-FCC624?logo=linux&logoColor=black)
![CUDA](https://img.shields.io/badge/CUDA%2012-beta-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/license-BSD--3-success)
![Version](https://img.shields.io/badge/version-0.1.0-blue)

<img src="https://raw.githubusercontent.com/rubendahan/conatus/main/web/public/gallery/kelvin-helmholtz-dye.webp" alt="Kelvin-Helmholtz billows in a shear layer, dye field computed with Conatus" width="85%"/>

*Kelvin-Helmholtz billows rolling up in a shear layer, with a passive dye (`examples/shear/kelvin_helmholtz.py`).*

</div>

## What it does

You describe a problem the way you would on a blackboard. Conatus picks the bases, places the
boundary conditions, splits implicit and explicit terms, and then tells you what it did:
`explain()` prints every numerical choice, so nothing happens behind your back.

From one model you can:

- 🌊 **Simulate** it in time, on CPU or on an NVIDIA GPU (beta).
- 📈 **Find its eigenvalues**, each with its residual, adjoint modes, and a check that a finer
  resolution finds the same mode again.
- ⚡ **Measure non-modal growth**: optimal perturbations and transient amplification.
- 🔁 **Study periodic problems** with Floquet and Hill spectra.
- 🎯 **Converge invariant solutions** (equilibria, periodic orbits) with Newton-Krylov.
- 🧭 **Follow branches** with continuation and locate bifurcations.
- 🌪️ **Quantify chaos** with Lyapunov spectra and covariant vectors.

## 📐 Stability: where does convection start?

Rayleigh-Bénard onset between rigid plates. The normal-mode problem, written as on paper
(from tutorial 0):

```python
import conatus as cn
from conatus.expert import critical

onset = cn.Model(cn.Box(z=cn.walls(-0.5, 0.5)), name="convection onset", Ra=1700.0, k=3.1, Pr=1)
onset.scalar("w"), onset.scalar("theta")       # w(z), theta(z) times exp(i k x + s t)
onset.eigenvalue("s", growth="real")
onset.define("L(f) = dz(dz(f)) - k**2*f")
onset.equations("""
    momentum: s*L(w) = Pr*L(L(w)) - Pr*Ra*k**2*theta
    heat:     s*theta = w + L(theta)
""")
onset.at("walls", "w = 0", "dz(w) = 0", "theta = 0")

P = onset.parametric(N=32, over=("Ra", "k"), drift=48, drift_tol=1e-6)   # modes confirmed at N=48
crit = critical(P, "Ra", bracket=(1500, 2000), over={"k": (2.8, 3.4)}, xtol=1e-9)
print(f"Ra_c = {crit.value:.4f}   k_c = {crit.over['k']:.5f}")
```

```text
Ra_c = 1707.7618   k_c = 3.11632
```

The classical value (Reid & Harris, 1958) is Ra_c = 1707.762 at k_c = 3.117.

## 🌡️ Simulation: the same convection, in time

```python
import numpy as np
import conatus as cn

box = cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5))
m = cn.Model(box, name="Rayleigh-Benard 2D", Ra=2500, Pr=1)
u, p, T = m.velocity("u"), m.pressure("p"), m.scalar("T")
m.equations("""
    momentum:   dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)
    continuity: div(u) = 0
    heat:       dt(T) - lap(T) = -u·grad(T)
""")
m.at("bottom", u=0, T=+0.5)
m.at("top", u=0, T=-0.5)

sim = m.simulate(N=(32, 24), scheme="RK443", dt=5e-4)    # backend="cuda" on a GPU (beta)
sim.initial(T=lambda x, z: -z + 1e-6 * np.cos(np.pi * x) * np.cos(np.pi * z))
sim.run(until=0.8)
print(sim.explain())
```

The small perturbation grows at the rate of the leading eigenvalue of the linearised model
(σ = 5.5425 at Ra = 2500), then saturates into rolls that carry the same heat flux through both
plates.

## 📦 Install

```bash
pip install conatus          # CPU wheel, compiled core included, no compiler needed
python -m conatus doctor     # what this machine can run, and what is missing
```

**GPU (beta).** For an NVIDIA A100 or L40S class card with a driver that supports CUDA 12.9.
The CUDA wheel is not on PyPI: grab the one matching your Python from the
[v0.1.0 release](https://github.com/rubendahan/conatus/releases/tag/v0.1.0) and install it with CuPy:

```bash
pip install "$(echo ./conatus-0.1.0+cu12-cp312-*.whl)[gpu]"   # the file you downloaded (CPython 3.12 here)
python -m conatus doctor                                      # says whether the CUDA core sees a GPU
```

**From source.** `pip install .` with LAPACK and BLAS available. Details on the
[install page](https://rubendahan.github.io/conatus/install/).

## 📚 Learn more

- The [website](https://rubendahan.github.io/conatus/) has the tutorials, the user guide, the API
  reference and the maths behind each method.
- [`examples/`](https://github.com/rubendahan/conatus/tree/main/examples) holds ready-to-run cases: convection, shear flows, patterns,
  bifurcations.

## ⚠️ Good to know

- **CPU speed.** The CPU path is a NumPy reference implementation. It is great for eigenvalue
  problems, small grids and setting a run up, but slower than established CPU codes on large
  grids. The speed comes from the GPU path, which is still in beta.
- **One GPU.** 0.1.0 runs a simulation on one CPU process or one GPU. No multi-GPU or MPI yet.
- **Young API.** The 0.1 interface can still change. Any change comes with a deprecation warning
  and a line in the [changelog](https://github.com/rubendahan/conatus/blob/main/CHANGELOG.md).

## 📄 License

BSD-3-Clause, see [LICENSE](https://github.com/rubendahan/conatus/blob/main/LICENSE). Bundled third-party components and their licences are
listed in [THIRD_PARTY_NOTICES.md](https://github.com/rubendahan/conatus/blob/main/THIRD_PARTY_NOTICES.md), with the full texts in
[`licenses/`](https://github.com/rubendahan/conatus/tree/main/licenses). To cite Conatus, see [CITATION.cff](https://github.com/rubendahan/conatus/blob/main/CITATION.cff).

Made by Ruben Dahan.
