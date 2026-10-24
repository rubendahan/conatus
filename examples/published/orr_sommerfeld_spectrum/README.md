# Orr–Sommerfeld spectrum of plane Poiseuille flow

**Reference.** S. A. Orszag (1971), Accurate solution of the Orr–Sommerfeld stability equation,
*J. Fluid Mech.* 50, 689–703, doi:10.1017/S0022112071002842.

**Published values used.** 32 of Orszag's eigenvalues c for Re = 10⁴, α = 1, typed in `run.py`
from Table 1 of S. Gana (2026), *Advancements in spectral collocation methods for high-order
eigenvalue problems*, arXiv:2608.29932, column "λₙ ([23])", which reproduces Orszag's values next to
Gana's own computation; the index n is the row of that table. Re_c = 5772.22 is from the abstract of
Orszag (1971); α_c = 1.02056 is the value later papers quote from it.

**What is computed.** Every eigenvalue c of the Orr–Sommerfeld equation for U = 1 − y² between rigid
walls at y = ±1, with 128 Chebyshev modes; a mode is kept only if it is found again with 160 modes
(to 10⁻⁶). Then the critical Reynolds number, minimised over α.

**Gap.** |c_Conatus − c_Orszag|, with c_Conatus the nearest computed eigenvalue not already matched.
A value is *within rounding* when its real and imaginary parts are both within half a unit of the
last digit Orszag prints (5 to 8 decimals).

**Result** (`output.txt`): 27 of the 32 eigenvalues are within rounding of Orszag's digits and all 32
within one unit of his last digit; the median gap is 5.3 × 10⁻⁸, the largest 7.1 × 10⁻⁶ (n = 20, a
value Orszag prints to 5 decimals). Of the five values outside rounding, three (n = 14, 20, 24) lie
near the point where the branches of the spectrum meet, where the eigenvalues are most sensitive;
the other two (n = 6, 10) differ by one unit in the last digit. Re_c = 5772.2218 agrees with
Orszag's 5772.22 to its printed digits; α_c = 1.020547 rounds to 1.02055, and the quoted
1.02056 differs from it by 1.3 × 10⁻⁵.

**Run.** From the repository root:

    python examples/published/orr_sommerfeld_spectrum/run.py

It prints the comparison and writes `result.json` in the working directory.
`bash examples/published/run.sh orr_sommerfeld_spectrum` records `output.txt` and `result.json`
here.
Recorded run: 7 s of wall time on 4 threads of an AMD EPYC 9654 96-Core Processor (Conatus 0.2.0, Python 3.12.14, 2026-10-23).
