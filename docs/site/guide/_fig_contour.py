"""Figure of the contour guide (docs/site/guide/dynamics/contour.md): how the eigenvalue count samples
the boundary. Run in a job (no computation on a login or session node):

    python docs/site/guide/_fig_contour.py docs/site/guide/media/contour_sampling.png

A 6 x 6 pencil L + z M with known eigenvalues (M = I, L similar to -diag(lambda)), one of them
2e-3 from the right edge of the box. The determinant is passed as a callable that records where
the walk evaluates it. Left: the box, the evaluation points and the eigenvalues. Right: the
accumulated phase of det T along the boundary, in turns; its final value is the count. The walk
is an adaptive test on samples, not a proof (the guide says which checks it applies).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import conatus as cn  # noqa: E402
from conatus.contour import count  # noqa: E402

HERE = Path(__file__).resolve().parent
plt.style.use(HERE.parent / "tutorial_media.mplstyle")
RED, BLUE, INK, SMOKE = "#C8463A", "#4C6FB0", "#77736B", "#8A8A86"

lam = np.array([0.30 + 0.20j, -0.40 + 0.45j, 0.10 - 0.30j, 0.798 + 0.10j, 1.30 + 0.10j, -0.2 - 0.9j])
rng = np.random.default_rng(7)
S = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
L = -S @ np.diag(lam) @ np.linalg.inv(S)
M = np.eye(6)
region = cn.box(re=(-0.8, 0.8), im=(-0.6, 0.6))
seen: list[complex] = []


def T(z):
    seen.append(complex(z))
    return L + z * M


res = count(T, region, poles=0)
inside = [v for v in lam if -0.8 < v.real < 0.8 and -0.6 < v.imag < 0.6]
assert res.count == len(inside), (res.count, len(inside))

# accumulated phase along the boundary, in the order of the walk (anticlockwise from -0.8-0.6i)
corners = [-0.8 - 0.6j, 0.8 - 0.6j, 0.8 + 0.6j, -0.8 + 0.6j]
per = sum(abs(corners[(k + 1) % 4] - corners[k]) for k in range(4))


def arclength(z: complex) -> float:
    s = 0.0
    for k in range(4):
        a, b = corners[k], corners[(k + 1) % 4]
        t = ((z - a) * np.conj(b - a)).real / abs(b - a) ** 2
        if 0 <= t <= 1 and abs(a + t * (b - a) - z) < 1e-12:
            return s + t * abs(b - a)
        s += abs(b - a)
    return np.nan


pts = sorted((z for z in set(seen) if np.isfinite(arclength(z))), key=arclength)   # nan keys break a sort
pts.append(pts[0])
ph = np.unwrap([np.angle(np.prod(z - lam)) for z in pts])      # det T = prod(z - lambda) here
turns = (ph - ph[0]) / (2 * np.pi)
s = np.array([arclength(z) for z in pts[:-1]] + [per])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.2))
a1.plot([c.real for c in corners + corners[:1]], [c.imag for c in corners + corners[:1]], color=INK, lw=0.8)
a1.plot([z.real for z in pts], [z.imag for z in pts], ".", ms=3, color=BLUE, label=f"{res.evaluations} evaluations")
a1.plot(lam.real, lam.imag, "x", ms=7, mew=1.6, color=RED, label="eigenvalues")
a1.annotate(f"shortest step {res.smallest_step:.1e}\nof the perimeter", (res.smallest_at.real, res.smallest_at.imag),
            xytext=(8, -28), textcoords="offset points", fontsize=8, color=INK)
a1.set_aspect("equal")
a1.set_xlim(-1.0, 1.45)
a1.set_ylim(-1.0, 0.75)
a1.set_xlabel("Re z")
a1.set_ylabel("Im z")
a1.legend(loc="lower right", fontsize=8)
a2.plot(s / per, turns, "-", color=BLUE)
a2.axhline(res.count, color=INK, ls=":", lw=0.9)
a2.set_xlabel("position along the boundary (fraction of the perimeter)")
a2.set_ylabel("phase of det T (turns)")
fig.tight_layout()
out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "media" / "contour_sampling.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print(f"count {res.count}, evaluations {res.evaluations}, smallest step {res.smallest_step:.2e} "
      f"at {res.smallest_at:.4f}, winding error {res.winding_error:.1e} -> {out}")
