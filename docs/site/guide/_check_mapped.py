"""Executable examples of the guide page ``mapped.md`` (channels between two surfaces).

Same conventions as ``_check_features.py``: every code block of the page is a region of this file
(``# [name]`` ... ``# [/name]``, included with ``literalinclude``, ``:dedent: 4``); what it printed
is written to ``docs/site/guide/outputs/<function>.txt`` with ``--sync`` and included as is; the
asserts after each example are the checks. Not a Sphinx source (leading underscore).

Run (CPU, about 10 s):
    python docs/site/guide/_check_mapped.py [name ...] [--sync]
"""
from __future__ import annotations

import contextlib
import io
import sys
import time
import traceback
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"


def mapped_stokes() -> None:
    """mapped.md, *A wavy wall*: Couette flow over z = eps cos x, first-order perturbation."""
    # [mapped-stokes]
    import numpy as np

    import conatus as cn

    eps = 0.01                                     # wall amplitude (wavelength 2 pi, gap 1)
    box = cn.MappedBox(x=cn.periodic(2 * np.pi),
                       z=cn.between(bottom=lambda x: eps * np.cos(x), top=1.0))
    m = cn.Model(box, name="Couette over a wavy wall")
    m.velocity("u")
    m.pressure("p")
    m.equations("grad(p) - lap(u) = 0", "div(u) = 0")      # Stokes flow
    m.at("bottom", u=0)                            # no slip on the wavy wall
    m.at("top", u=(1, 0))                          # the upper plate slides at speed 1
    sol = m.solve((16, 24))                        # (N_x, N_z)

    U, X, Z = sol.grid("u"), sol.coordinate("x"), sol.coordinate("z")
    print(sol.explain())
    print("grid:", U.shape, "| z of the bottom row runs from", f"{Z[:, 0].min():.4f}",
          "to", f"{Z[:, 0].max():.4f}")
    # [/mapped-stokes]
    # [mapped-perturbation]
    # first order in eps: psi_1 = phi(z) cos x with (D^2 - 1)^2 phi = 0,
    # phi(0) = phi(1) = 0, phi'(0) = -1 (no slip moved to z = 0), phi'(1) = 0
    def rows(z):
        ch, sh = np.cosh(z), np.sinh(z)
        return np.array([ch, z * ch, sh, z * sh]), np.array([sh, ch + z * sh, ch, sh + z * ch])

    co = np.linalg.solve([rows(0)[0], rows(1)[0], rows(0)[1], rows(1)[1]], [0, 0, -1, 0])
    u1 = np.tensordot(co, rows(Z)[1], 1) * np.cos(X)
    rem = np.abs(U[0] - Z - eps * u1).max()
    print(f"max |u_x - (z + eps u_1)| = {rem:.2e} = {rem / eps**2:.2f} eps^2")
    # [/mapped-perturbation]
    assert sol.max_residual < 1e-14
    assert rem < 5 * eps ** 2


def mapped_explain() -> None:
    """mapped.md, *What the map adds*: the geometry and the plan explain the chain rule."""
    # [mapped-explain]
    import numpy as np

    import conatus as cn

    box = cn.MappedBox(x=cn.periodic(2 * np.pi),
                       z=cn.between(bottom=lambda x: 0.1 * np.cos(x), top=1.0))
    m = cn.Model(box, name="Poisson")
    m.scalar("u")
    m.given("f", lambda x, z: np.sin(x) * z)
    m.equations("lap(u) = f")
    m.at("walls", u=0)
    print(box.explain())
    print(m.discretize((24, 16)).explain())
    # [/mapped-explain]
    text = m.discretize((24, 16)).explain()
    assert "metric terms implicit" in text and "Fourier bandwidth" in text


def mapped_convergence() -> None:
    """mapped.md, *Convergence*: manufactured solution, spectral decay of the error."""
    # [mapped-convergence]
    import numpy as np

    import conatus as cn

    eps = 0.05
    h = lambda x: eps * np.cos(x)
    ue = lambda x, z: np.exp(np.sin(x)) * np.sin(z) + z**2      # exact solution
    fe = lambda x, z: ((np.cos(x)**2 - np.sin(x) - 1) * np.exp(np.sin(x)) * np.sin(z)
                       + 2)                                     # its Laplacian
    for N in (16, 20, 24, 28):
        m = cn.Model(cn.MappedBox(x=cn.periodic(2 * np.pi), z=cn.between(bottom=h, top=1.0)))
        m.scalar("u")
        m.given("f", fe)
        m.equations("lap(u) = f")
        m.at("bottom", u=lambda x: ue(x, h(x)))
        m.at("top", u=lambda x: ue(x, 1.0))
        s = m.solve((N, N))
        err = np.abs(s.grid("u") - ue(s.coordinate("x"), s.coordinate("z"))).max()
        print(f"N = {N:2d}: max error {err:.1e}, residual {s.max_residual:.1e}")
    # [/mapped-convergence]
    assert err < 1e-13


def mapped_neumann() -> None:
    """mapped.md, *Conditions on a curved wall*: derivative along the true normal."""
    import numpy as np

    import conatus as cn
    eps = 0.1
    box = cn.MappedBox(x=cn.periodic(2 * np.pi),
                       z=cn.between(bottom=lambda x: eps * np.cos(x), top=1.0))
    # [mapped-neumann]
    slope = lambda x: -eps * np.sin(x)                 # b'(x)
    norm = lambda x: np.sqrt(1 + slope(x)**2)
    m = cn.Model(box, name="heat flux through a wavy wall")
    m.scalar("T")
    m.given("nx", lambda x: slope(x) / norm(x))       # outward unit normal of z = b(x)
    m.given("nz", lambda x: -1 / norm(x))
    m.equations("lap(T) = 0")
    m.at("bottom", "nx*dx(T) + nz*dz(T) = 1")          # unit flux into the fluid
    m.at("top", T=0)
    sol = m.solve((24, 16))
    print(sol.explain())
    # [/mapped-neumann]
    assert sol.max_residual < 1e-13


EXAMPLES = {"mapped_stokes": mapped_stokes, "mapped_explain": mapped_explain,
            "mapped_convergence": mapped_convergence, "mapped_neumann": mapped_neumann}


def main(argv: list[str]) -> int:
    sync = "--sync" in argv
    names = [a for a in argv if not a.startswith("--")] or list(EXAMPLES)
    bad = 0
    for n in names:
        buf = io.StringIO()
        t0 = time.time()
        try:
            with contextlib.redirect_stdout(buf):
                EXAMPLES[n]()
            print(f"PASS {n} ({time.time() - t0:.1f} s)")
            if sync:
                OUT.mkdir(exist_ok=True)
                (OUT / f"{n}.txt").write_text(buf.getvalue())
        except Exception:  # noqa: BLE001 -- report every failing example, then fail
            bad += 1
            print(f"FAIL {n}\n{buf.getvalue()}{traceback.format_exc()}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
