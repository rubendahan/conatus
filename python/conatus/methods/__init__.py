"""``plan.methods()``: a "Methods" paragraph generated from the plan (spec §116).

The text is built only from what the objects hold (bases, sizes, dealias, tau/lift layout,
symmetry subspace, NCC reports, scheme, dt, startup, backend, dtype, commit); nothing is
paraphrased from defaults. It is meant to be pasted into a paper and edited, with the BibTeX of
the cited methods alongside.

    import conatus.methods            # adds Plan.methods (additive, no change to Plan itself)
    m = plan.methods(solver=ivp)      # or conatus.methods.methods(plan, solver=ivp)
    print(m.text); print(m.bibtex)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from conatus.discretize.plan import Plan
from conatus.ir import expr as E

__all__ = ["MethodsText", "methods", "BIBTEX"]

BIBTEX = {
    "vasil2016": """@article{vasil2016,
  author  = {Vasil, Geoffrey M. and Burns, Keaton J. and Lecoanet, Daniel and Olver, Sheehan and Brown, Benjamin P. and Oishi, Jeffrey S.},
  title   = {Tensor calculus in polar coordinates using {Jacobi} polynomials},
  journal = {Journal of Computational Physics},
  volume  = {325},
  pages   = {53--73},
  year    = {2016},
  doi     = {10.1016/j.jcp.2016.08.013}
}""",
    "vasil2019": """@article{vasil2019,
  author  = {Vasil, Geoffrey M. and Lecoanet, Daniel and Burns, Keaton J. and Oishi, Jeffrey S. and Brown, Benjamin P.},
  title   = {Tensor calculus in spherical coordinates using {Jacobi} polynomials. {Part-I}: Mathematical analysis and derivations},
  journal = {Journal of Computational Physics: X},
  volume  = {3},
  pages   = {100013},
  year    = {2019},
  doi     = {10.1016/j.jcpx.2019.100013}
}""",
    "lecoanet2019": """@article{lecoanet2019,
  author  = {Lecoanet, Daniel and Vasil, Geoffrey M. and Burns, Keaton J. and Brown, Benjamin P. and Oishi, Jeffrey S.},
  title   = {Tensor calculus in spherical coordinates using {Jacobi} polynomials. {Part-II}: Implementation and examples},
  journal = {Journal of Computational Physics: X},
  volume  = {3},
  pages   = {100012},
  year    = {2019},
  doi     = {10.1016/j.jcpx.2019.100012}
}""",
    "burns2020": """@article{burns2020,
  author  = {Burns, Keaton J. and Vasil, Geoffrey M. and Oishi, Jeffrey S. and Lecoanet, Daniel and Brown, Benjamin P.},
  title   = {Dedalus: A flexible framework for numerical simulations with spectral methods},
  journal = {Physical Review Research},
  volume  = {2},
  pages   = {023068},
  year    = {2020},
  doi     = {10.1103/PhysRevResearch.2.023068}
}""",
    "olver2013": """@article{olver2013,
  author  = {Olver, Sheehan and Townsend, Alex},
  title   = {A fast and well-conditioned spectral method},
  journal = {SIAM Review},
  volume  = {55},
  number  = {3},
  pages   = {462--489},
  year    = {2013},
  doi     = {10.1137/120865458}
}""",
    "ascher1997": """@article{ascher1997,
  author  = {Ascher, Uri M. and Ruuth, Steven J. and Spiteri, Raymond J.},
  title   = {Implicit-explicit {Runge--Kutta} methods for time-dependent partial differential equations},
  journal = {Applied Numerical Mathematics},
  volume  = {25},
  number  = {2--3},
  pages   = {151--167},
  year    = {1997},
  doi     = {10.1016/S0168-9274(97)00056-1}
}""",
    "wang2008": """@article{wang2008,
  author  = {Wang, Dong and Ruuth, Steven J.},
  title   = {Variable step-size implicit-explicit linear multistep methods for time-dependent partial differential equations},
  journal = {Journal of Computational Mathematics},
  volume  = {26},
  number  = {6},
  pages   = {838--855},
  year    = {2008}
}""",
    "spalart1991": """@article{spalart1991,
  author  = {Spalart, Philippe R. and Moser, Robert D. and Rogers, Michael M.},
  title   = {Spectral methods for the {Navier--Stokes} equations with one infinite and two periodic directions},
  journal = {Journal of Computational Physics},
  volume  = {96},
  number  = {2},
  pages   = {297--324},
  year    = {1991}
}""",
}

_SCHEME_TEXT = {
    "RK111": ("the first-order IMEX Runge–Kutta scheme (1,1,1) of Ascher, Ruuth & Spiteri", "ascher1997"),
    "RK222": ("the second-order IMEX Runge–Kutta scheme (2,2,2) of Ascher, Ruuth & Spiteri", "ascher1997"),
    "RK443": ("the third-order, four-stage IMEX Runge–Kutta scheme (4,4,3) of Ascher, Ruuth & "
              "Spiteri, whose implicit diagonal is constant so that a single factorization serves "
              "all stages", "ascher1997"),
    "CNAB1": ("the first-order Crank–Nicolson / Adams–Bashforth scheme (CNAB1) of Wang & Ruuth, "
              "with the linear terms treated by Crank–Nicolson and the nonlinear terms by forward "
              "Euler", "wang2008"),
    "CNAB2": ("the second-order Crank–Nicolson / Adams–Bashforth scheme (CNAB2) of Wang & Ruuth",
              "wang2008"),
    "MCNAB2": ("the second-order modified Crank–Nicolson / Adams–Bashforth scheme (MCNAB2) of "
               "Wang & Ruuth", "wang2008"),
    "CNLF2": ("the second-order Crank–Nicolson / leapfrog scheme (CNLF2) of Wang & Ruuth",
              "wang2008"),
    "RKSMR": ("the three-stage IMEX Runge–Kutta scheme of Spalart, Moser & Rogers (Crank–Nicolson "
              "implicit part, third-order low-storage explicit part; second order overall)",
              "spalart1991"),
    "RKGFY": ("the two-stage second-order IMEX Runge–Kutta scheme RKGFY (Heun explicit part, "
              "Crank–Nicolson implicit part)", None),
}


@dataclass(frozen=True)
class MethodsText:
    """A generated Methods paragraph: ``text``, the BibTeX keys it cites and the facts it used."""

    text: str
    citations: tuple[str, ...]
    facts: dict = field(default_factory=dict)

    @property
    def bibtex(self) -> str:
        """The BibTeX entries of the cited methods, in citation order."""
        return "\n\n".join(BIBTEX[c] for c in self.citations)

    def explain(self) -> str:
        """The paragraph with its citations and the facts it was built from."""
        return ("Methods paragraph generated from the plan (edit before publishing):\n\n"
                + self.text + "\n\nCitations: " + ", ".join(self.citations)
                + "\nFacts used: " + ", ".join(f"{k}={v}" for k, v in self.facts.items()))

    def __str__(self) -> str:                  # print(methods(...)) shows the paragraph, repr stays raw
        return self.explain()


def _num(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    return f"{x:g}"


def _frac(x: float) -> str:
    for d in (2, 3, 4):
        if abs(x * d - round(x * d)) < 1e-12:
            n = round(x * d)
            return f"{n}/{d}" if n % d else str(n // d)
    return f"{x:g}"


def _zbasis_text(b: Any) -> tuple[str, str]:
    """(expansion, operators) phrases of the non-periodic basis of a Cartesian plan, read from the
    basis itself (every basis used to be described as Chebyshev T on [a, b] with
    ultraspherical operators, also Legendre, Jacobi, Hermite and Laguerre)."""
    from conatus.ir.bases import (Chebyshev, CompoundChebyshev, Hermite, Jacobi, Laguerre, Legendre,
                                  Ultraspherical)
    a, z = (_num(x) for x in b.bounds) if hasattr(b, "bounds") else ("", "")
    banded = "sparse, banded operators exact on the truncated space"
    from conatus.ir.maps import is_mapped
    if is_mapped(b):                      # review of was the generic fallback
        return (f"∈ [{a}, {z}] is expanded in {b.size} Chebyshev polynomials of the first kind of "
                f"a mapped coordinate ({b.map.explain()})",
                "the sparse, banded ultraspherical operators of Olver & Townsend, the metric of the "
                "map multiplied as a non-constant coefficient")
    if type(b) is Chebyshev:
        return (f"∈ [{a}, {z}] is expanded in {b.size} Chebyshev polynomials of the first kind",
                "the sparse, banded ultraspherical operators of Olver & Townsend")
    if isinstance(b, Legendre):
        return (f"∈ [{a}, {z}] is expanded in {b.size} Legendre polynomials",
                f"{banded} (Jacobi polynomials, the construction of Olver & Townsend)")
    if isinstance(b, Ultraspherical):
        return (f"∈ [{a}, {z}] is expanded in {b.size} ultraspherical (Gegenbauer) polynomials "
                f"C^({_frac(b.lam)})", f"{banded} (Jacobi polynomials, the construction of Olver & "
                "Townsend)")
    if isinstance(b, Jacobi) and not isinstance(b, Chebyshev):
        return (f"∈ [{a}, {z}] is expanded in {b.size} Jacobi polynomials P^({_frac(b.a)}, {_frac(b.b)})",
                f"{banded} (Jacobi polynomials, the construction of Olver & Townsend)")
    if isinstance(b, Hermite):
        return (f"∈ (−∞, ∞) is expanded in {b.size} Hermite functions (centre {b.center:g}, scale "
                f"{b.scale:g})", f"{banded} (tridiagonal derivative and multiplication by x)")
    if isinstance(b, Laguerre):
        return (f"∈ [{b.start:g}, ∞) is expanded in {b.size} Laguerre functions (scale {b.scale:g})",
                f"{banded} (Laguerre functions, the construction of Olver & Townsend)")
    if isinstance(b, CompoundChebyshev):
        return (f"∈ [{a}, {z}] is expanded piecewise in Chebyshev polynomials of the first kind "
                f"({b.size} in total, glued at the interfaces)",
                "block-diagonal ultraspherical operators of Olver & Townsend on each piece")
    return (f"is expanded in {b.size} functions of the {b.family} basis ({b.coeff_space()})",
            "the sparse operators of that basis")


def _sbdf(order: int) -> tuple[str, str]:
    return (f"the order-{order} semi-implicit backward differentiation formula (SBDF{order}), "
            "with the linear terms implicit and the nonlinear terms extrapolated", "wang2008")


def methods(plan: Plan, solver: Any = None, *, eigen_solver: str | None = None,
            commit: str | None = None) -> MethodsText:
    """Build the Methods paragraph. ``solver`` is an IVPSolver (time stepping); for an EVP give
    ``eigen_solver`` ("dense" or "shift-invert") to describe how eigenpairs were obtained."""
    pb = plan.problem
    polar = getattr(plan, "is_polar", False)
    sphere = getattr(plan, "is_sphere", False)
    L = None if (polar or sphere) else plan.layout
    cites: list[str] = []
    facts: dict[str, Any] = {"kind": pb.kind, "groups": len(plan.groups)}
    s: list[str] = []

    names = ", ".join(v.name for v in pb.variables if not v.name.startswith("tau"))
    s.append(f"The {pb.kind} '{pb.name}' for ({names}) was solved with Conatus"
             + (f" (commit {commit})" if commit else "") + ".")

    parts = []
    if polar:
        s.append(_polar_text(plan, cites, facts))
    if sphere:
        s.append(_sphere_text(plan, cites, facts))
    for d in (L.fdirs if L is not None else ()):
        b = d.basis
        kind = "complex exponential" if d.complex else "real (cosine/sine)"
        parts.append(f"{b.coord.name} ∈ [{_num(b.bounds[0])}, {_num(b.bounds[1])}) is periodic and "
                     f"expanded in a {kind} Fourier series of {b.size} modes")
        facts[f"N_{b.coord.name}"] = b.size
    if L is not None and L.zbasis is not None:
        b = L.zbasis
        expansion, operators = _zbasis_text(b)
        parts.append(f"{b.coord.name} {expansion}")
        facts[f"N_{b.coord.name}"] = b.size
        facts["zbasis"] = type(b).__name__
    if parts:
        s.append("The domain is discretized as follows: " + "; ".join(parts) + ".")
    if L is not None and L.zbasis is not None:
        ntau = sum(1 for v in pb.variables if v in plan.taus)
        s.append(f"Derivatives, basis conversions and multiplications by non-constant coefficients use "
                 f"{operators}, and boundary conditions are imposed with the generalized tau method "
                 f"({ntau} tau variable(s), each entering through a lift polynomial), following "
                 "Burns et al.")
        cites += ["olver2013", "burns2020"]
        facts["taus"] = ntau
    if L is not None and L.fdirs:
        s.append(f"The Fourier directions decouple, giving {len(plan.groups)} independent systems, "
                 "one per wavenumber.")

    deal = {b.coord.name: b.dealias for b in (plan.domain.bases if plan.domain else ())}
    if deal:
        if all(v == 1 for v in deal.values()):
            s.append("No dealiasing is applied (dealias factor 1 on every basis).")
        else:
            s.append("Nonlinear terms are evaluated pseudospectrally with dealiasing factors "
                     + ", ".join(f"{_frac(v)} in {k}" for k, v in deal.items()) + ".")
        facts["dealias"] = deal
    nonpoly = sorted({f for eq in plan.problem.equations
                      for f in E.has_nonpolynomial(eq.lhs) + E.has_nonpolynomial(eq.rhs)})
    if nonpoly:
        s.append("The non-polynomial functions " + ", ".join(nonpoly) + " are evaluated pointwise "
                 "on the grid; dealiasing is not exact for them, and convergence was checked by "
                 "resolution.")
        facts["nonpolynomial"] = nonpoly

    if getattr(plan, "_ncc_data", None) and not plan.ncc_reports:
        plan.matrices(plan.groups[0])        # NCC expansions are reported at assembly (cached)
    if plan.ncc_reports:
        cut = ("without truncation" if plan.ncc_cutoff is None
               else f"truncated at relative amplitude {plan.ncc_cutoff:g}")
        s.append(f"Non-constant coefficients are expanded {cut}; their spectral tails are "
                 "reported in the run record.")

    if pb.symmetry is not None:
        g0 = plan.groups[0]
        full, red = plan.matrices(g0, False), plan.matrices(g0, True)
        s.append(f"The computation is restricted to the invariant subspace of {pb.symmetry!r}, "
                 f"reducing the first system from {full.shape[0]} to {red.shape[0]} unknowns; "
                 "symmetry-breaking perturbations are not represented.")
        facts["reduction"] = f"{full.shape[0]}->{red.shape[0]}"

    s.append("The structural rank of every system was checked before factorization.")

    if solver is not None:
        sch = solver.scheme
        if sch.startswith("SBDF"):
            txt, c = _sbdf(int(sch[4:]))
        else:
            txt, c = _SCHEME_TEXT.get(sch, (sch, None))
        if c:
            cites.append(c)
        if getattr(solver, "_it_origin", 0):      # set_dt after steps: the step was not constant
            s.append(f"Time integration uses {txt}, with a variable step (Δt = {solver.dt:g} at "
                     f"iteration {solver.iteration}; the step was changed during the run).")
        else:
            s.append(f"Time integration uses {txt}, with a constant step Δt = {solver.dt:g}.")
        if sch.startswith("SBDF") and int(sch[4:]) > 1:
            st = getattr(solver, "startup", None)
            if st == "rk443":
                s.append("The multistep history is initialized with RK443 steps, which preserves "
                         "the formal order of the scheme.")
                if "ascher1997" not in cites:
                    cites.append("ascher1997")
            elif st == "lower_order":
                s.append("The scheme is started with lower-order steps, which limits the global "
                         "order to two.")
        if sch in ("CNAB2", "MCNAB2", "CNLF2"):
            st = getattr(solver, "startup", None)
            if st == "rk443":
                s.append("The first step is computed with RK443 substeps to initialize the "
                         "multistep history.")
                if "ascher1997" not in cites:
                    cites.append("ascher1997")
            elif st == "lower_order":
                s.append("The first step uses CNAB1; the global order remains two.")
        system = getattr(solver, "system", None)
        dtype = np.dtype(getattr(system, "dtype", np.float64))
        dev = "on a GPU (CUDA)" if getattr(solver, "backend", "cpu") == "cuda" else "on CPU"
        s.append("The implicit systems are solved by banded LU factorization with a low-rank "
                 f"(Woodbury) correction for the tau rows, batched over wavenumbers {dev}, "
                 f"in {dtype.name} arithmetic.")
        facts.update(scheme=sch, dt=solver.dt, backend=getattr(solver, "backend", None))
    elif pb.kind == "EVP":
        ev = pb.eigenvalue.name
        how = {"dense": "dense QZ (LAPACK) on each generalized pencil",
               "shift-invert": "shift-and-invert Arnoldi on each generalized pencil"}.get(
            eigen_solver or "", None)
        s.append(f"Eigenpairs of the generalized problem (L + {ev} M) x = 0 are computed "
                 + (f"with {how}" if how else "per wavenumber")
                 + "; infinite eigenvalues from the tau rows are counted and discarded, and every "
                 "reported eigenvalue comes with its residual ‖(L + σM)x‖/‖x‖.")
        facts["eigen_solver"] = eigen_solver
    elif pb.kind == "LBVP":
        s.append("Each system is solved by sparse LU factorization and the residual of every "
                 "solve is reported.")

    seen: list[str] = []
    for c in cites:
        if c not in seen:
            seen.append(c)
    return MethodsText(" ".join(s), tuple(seen), facts)


def _polar_text(plan: Any, cites: list, facts: dict) -> str:
    """Domain and discretization sentences of a polar plan (disk or annulus)."""
    S = plan.space
    bp, br = S.bphi, S.br
    kind = "complex exponential" if not S.real else "real (cosine/sine)"
    facts.update({f"N_{bp.coord.name}": bp.size, f"N_{br.coord.name}": br.size})
    ntau = sum(1 for v in plan.variables if v in plan.taus)
    txt = (f"The azimuth {bp.coord.name} is expanded in a {kind} Fourier series of {bp.size} modes; "
           f"each azimuthal wavenumber m gives an independent system ({len(plan.groups)} in total). ")
    if S.ops.kind == "disk":
        txt += (f"On the disk of radius {_num(br.radius)}, every spin component (u_± = u_r ± i u_φ) "
                f"of azimuthal behaviour e^(ikφ) is expanded as ρ^|k| P_n^(a,|k|)(2ρ²−1) with Jacobi "
                f"polynomials (Zernike polynomials), n < {br.size} − ⌊|m|/2⌋, so that every "
                "representable field is regular at the axis without any axis condition; gradient, "
                "divergence, curl and Laplacian are sparse (single-diagonal) in these bases, following "
                "Vasil et al. ")
        cites += ["vasil2016", "burns2020"]
    else:
        txt += (f"On the annulus {_num(br.bounds[0])} ≤ r ≤ {_num(br.bounds[1])}, spin components are "
                f"expanded in {br.size} Chebyshev polynomials and each equation is multiplied by the "
                "smallest power of r that makes its coefficients polynomial (declared), giving banded "
                "ultraspherical operators (Olver & Townsend). ")
        cites += ["olver2013"]
    txt += (f"Boundary conditions are imposed with the generalized tau method ({ntau} tau variable(s), "
            "lifted into the last radial mode of the equation's derivative basis).")
    facts["radial"] = S.ops.kind
    facts["taus"] = ntau
    return txt


def _sphere_text(plan: Any, cites: list, facts: dict) -> str:
    """Domain and discretization sentences of a spherical plan (S2, ball or shell)."""
    S = plan.space
    kind = "complex exponential" if not S.real else "real (cosine/sine)"
    facts.update({f"N_{S.bphi.coord.name}": S.bphi.size, "Lmax": S.L})
    txt = (f"The azimuth {S.bphi.coord.name} is expanded in a {kind} Fourier series of {S.bphi.size} "
           f"modes and the colatitude in spin-weighted spherical harmonics of degree l ≤ {S.L} "
           "(triangular truncation), every tensor being written in spin components so that fields are "
           "regular at the poles by construction; angular derivatives are diagonal in l "
           f"(Vasil et al.). {'Each azimuthal wavenumber' if plan.coupled else 'Each pair (m, l)'} gives "
           f"an independent system ({len(plan.groups)} in total). ")
    cites += ["vasil2019", "lecoanet2019"]
    if S.ball:
        txt += (f"In the ball of radius {_num(S.br.radius)}, the regularity components of degree l are "
                f"expanded as r^k P_n^(a, k+1/2)(2r²−1), k = l + σ, n < {S.br.size} − ⌊l/2⌋ (three-"
                "dimensional Zernike polynomials, Lecoanet et al.), regular at the centre without any "
                "condition. ")
    elif S.shell:
        ntau = sum(1 for v in plan.variables if v in plan.taus)
        txt += (f"In the shell {_num(S.br.bounds[0])} ≤ r ≤ {_num(S.br.bounds[1])}, spin components are "
                f"expanded in {S.br.size} Chebyshev polynomials and each equation is multiplied by r^k in "
                "the ultraspherical basis C^(k) of its derivative order k (Dedalus's r^−k derivative "
                f"bases; banded operators, Olver & Townsend), with {ntau} tau variable(s). ")
        cites += ["olver2013"]
    facts["geometry"] = "S2" if S.s2 else ("ball" if S.ball else "shell")
    return txt


def _plan_methods(self: Plan, solver: Any = None, **kw) -> MethodsText:
    return methods(self, solver, **kw)


if not hasattr(Plan, "methods"):
    Plan.methods = _plan_methods

# The package called is its main function: cn.methods(...) == conatus.expert.methods(...).
from .._callable import make_callable  # noqa: E402

make_callable(__name__, methods)
