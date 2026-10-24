# Onset of Rayleigh–Bénard convection: the marginal curves

**References.** Lord Rayleigh (1916), On convection currents in a horizontal layer of fluid, when the
higher temperature is on the under side, *Phil. Mag.* 32, 529–546, doi:10.1080/14786441608635602.
S. Chandrasekhar (1961), *Hydrodynamic and Hydromagnetic Stability*, Oxford University Press, ch. II.

**Published values used.** For two stress-free walls, Rayleigh's closed form
Ra(k) = (k² + π²)³ / k², with minimum 27π⁴/4 at k = π/√2. For two rigid walls, Ra_c = 1707.762 at
k_c = 3.117; for a rigid bottom and a stress-free top, Ra_c = 1100.65 at k_c = 2.682 (Chandrasekhar
1961, ch. II). Typed in `run.py` from the book's text.

**What is computed.** The onset problem for the vertical velocity w(z) and the temperature θ(z) of a
mode exp(ikx + st) in a layer 0 < z < 1 (Boussinesq, Pr = 1; the marginal curve of this stationary
onset does not depend on Pr), with 32 Chebyshev modes, every mode checked again with 48. For each
pair of walls: the neutral Rayleigh number at 25 wavenumbers k = 1 … 7 and the minimum (Ra_c, k_c).

**Gap.** The relative gap |Ra − Ra_ref| / Ra_ref along the whole stress-free curve, against the
closed form, and at the critical points of the rigid cases, against the digits Chandrasekhar prints.

**Result** (`output.txt`): along the stress-free curve the largest relative gap to the closed form is
9.1 × 10⁻¹⁶. Two rigid walls: Ra_c = 1707.761777 at k_c = 3.116324 (relative gaps 1.3 × 10⁻⁷ and
2.2 × 10⁻⁴). The computed k_c rounds to 3.116, where Chandrasekhar prints 3.117. Rigid bottom and stress-free
top: Ra_c = 1100.649607 at k_c = 2.682322, within the rounding of 1100.65 and 2.682.

**Run.** From the repository root:

    python examples/published/rayleigh_benard_onset/run.py

It prints the comparison and the three curves, and writes `result.json` in the working directory.
`bash examples/published/run.sh rayleigh_benard_onset` records `output.txt` and `result.json` here.
Recorded run: 30 s of wall time on 4 threads of an AMD EPYC 9654 96-Core Processor (Conatus 0.2.1, Python 3.12.14, 2026-10-24).
