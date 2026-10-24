# Kuramoto–Sivashinsky equilibria and travelling waves, L = 22

**Reference.** P. Cvitanović, R. L. Davidchack & E. Siminos (2010), On the state space geometry of
the Kuramoto–Sivashinsky flow in a periodic domain, *SIAM J. Appl. Dyn. Syst.* 9, 1–33,
doi:10.1137/070705623 (arXiv:0709.2944).

**Published values used.** Table 5.1: the leading eigenvalues μ ± iν of the equilibria E1, E2, E3
and of the relative equilibria TW1, TW2 (4 decimals, 32 entries; a pair μ ± iν is one entry). Table 5.2: the energies of the five
solutions. Section 5: the wave speeds c = 0.73699 and 0.34954. All are typed in `run.py` from the
paper's tables and text.

**What is computed.** The equation u_t + u u_x + u_xx + u_xxxx = 0 on a periodic interval of
length 22. Each solution is found by Newton–Krylov–hookstep from a fixed seed with 32 Fourier
modes, then converged again with 64 modes (residual below 10⁻¹²). The leading eigenvalues are those
of the Jacobian of the right-hand side, in the frame moving with the wave for TW1 and TW2.

**Gap.** For each tabulated eigenvalue: |λ_Conatus − λ_published|, with λ_Conatus the nearest
computed eigenvalue not already matched. A value is *within rounding* when its real and imaginary
parts are both within half a unit of the last printed digit (5 × 10⁻⁵; 5 × 10⁻⁶ for the speeds).

**Result** (`output.txt`): all 39 published values (32 eigenvalues, 5 energies, 2 speeds) are
within rounding; the largest gap is 6.0 × 10⁻⁵ (both parts within 5 × 10⁻⁵).

**Run.** From the repository root:

    python examples/published/kuramoto_sivashinsky_l22/run.py

It prints the comparison and writes `result.json` in the working directory.
`bash examples/published/run.sh kuramoto_sivashinsky_l22` records `output.txt` and `result.json`
here.
Recorded run: 14 s of wall time on 4 threads of an AMD EPYC 9654 96-Core Processor (Conatus 0.2.0, Python 3.12.14, 2026-10-23).
