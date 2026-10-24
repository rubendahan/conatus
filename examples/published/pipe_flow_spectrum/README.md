# Linearised pipe flow from Re = 10² to 10⁷

**Reference.** A. Meseguer & L. N. Trefethen (2003), Linearized pipe flow to Reynolds number 10⁷,
*J. Comput. Phys.* 186, 178–197, doi:10.1016/S0021-9991(03)00029-9.

**Published values used.** Table 1: the rightmost eigenvalue of Hagen–Poiseuille flow for the
azimuthal and axial wavenumbers (n, k) = (1, 0), and the rightmost centre mode and wall mode for
(1, 1), at R = 10², 10³, …, 10⁷, to ten decimals (17 values; no centre mode at R = 10²). The authors
expect all but possibly the last digit to be correct. Typed in `run.py` from the table. Units: pipe
radius and centreline velocity.

**What is computed.** The eigenvalues s of the Navier–Stokes equations linearised about
U = 1 − r² (primitive variables, no-slip wall, regular at the axis) for one Fourier pair (n, k), at
two radial resolutions per Reynolds number (32 and 48 at R = 10², up to 256 and 320 at R = 10⁷). An
eigenvalue is kept only if both resolutions find it to 10⁻⁸; the value at the finer one is reported,
with the difference between the two. Conatus writes modes as exp(inφ + ikz + st); the table lists the
complex conjugate, which is what is compared. The centre mode is the rightmost (1, 1) eigenvalue with
|Im s| > 0.75, the wall mode the rightmost one with |Im s| < 0.6.

**Gap.** |s_Conatus − λ̄_MT| for each tabulated λ_MT. *Within rounding*: real and imaginary parts
both within 5 × 10⁻¹¹ (half a unit of the tenth decimal).

**Result** (`output.txt`): 15 of the 17 tabulated eigenvalues are within rounding of the ten decimals;
the largest gap for R = 10² to 10⁴ is 6.3 × 10⁻¹¹. The two others are the centre modes at R = 10⁶ and
10⁷ (gaps 1.8 × 10⁻¹⁰ and 3.9 × 10⁻¹⁰, two and four units of the tenth decimal); there Conatus's two
resolutions agree to better than 10⁻¹². These eigenvalues are very sensitive to perturbations of the
operator: Meseguer & Trefethen note, next to Table 1, that the numbers in this part of the complex
plane are exact eigenvalues of operators perturbed by less than 10⁻¹⁰. A difference of a few units
of the tenth decimal is of that size, so this comparison cannot say which of the two values is
closer to the exact eigenvalue.

**Run.** From the repository root:

    python examples/published/pipe_flow_spectrum/run.py

It prints the comparison and writes `result.json` in the working directory.
`bash examples/published/run.sh pipe_flow_spectrum` records `output.txt` and `result.json` here.
Recorded run: 174 s of wall time on 4 threads of an AMD EPYC 9654 96-Core Processor (Conatus 0.2.1, Python 3.12.14, 2026-10-24).
