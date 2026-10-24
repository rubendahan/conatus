"""Time schemes of the IVP solvers: the catalogue (SBDF, Crank-Nicolson multistep and IMEX
Runge-Kutta families), their coefficients and tableaux, and the validation of (scheme, startup,
bc_time) shared by every IVP solver. Split out of :mod:`conatus.timestep.integrator`, which
re-exports every name (coefficients and orders)."""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ..errors import OptionError, did_you_mean


SBDF_SCHEMES = ("SBDF1", "SBDF2", "SBDF3", "SBDF4")
CN_SCHEMES = ("CNAB1", "CNAB2", "MCNAB2", "CNLF2")
RK_SCHEMES = ("RK111", "RK222", "RK443", "RKSMR", "RKGFY")
SCHEMES = ("SBDF1", "SBDF2", "SBDF3", "SBDF4", "RK111", "RK222", "RK443",
           "CNAB1", "CNAB2", "MCNAB2", "CNLF2", "RKSMR", "RKGFY")
STARTUPS = ("lower_order", "rk443")
# time-dependent boundary data: "auto" (default) = "rk" when g'(t) is exact, else
# "stage" (declared); "rk" = stage values from the scheme applied to g'(t); "stage" = g(t_stage);
# "step" = the Dedalus convention (g carried by F with the explicit weights)
BC_TIMES = ("auto", "rk", "stage", "step")


# (family, implicit part, explicit part, global IMEX order, reference): explain() and the docs.
_SCHEME_INFO = {
    "SBDF1": ("multistep", "backward Euler", "forward Euler", 1, "Wang & Ruuth 2008"),
    "SBDF2": ("multistep", "BDF2", "2-step extrapolation", 2, "Wang & Ruuth 2008"),
    "SBDF3": ("multistep", "BDF3", "3-step extrapolation", 3, "Wang & Ruuth 2008"),
    "SBDF4": ("multistep", "BDF4", "4-step extrapolation", 4, "Wang & Ruuth 2008"),
    "CNAB1": ("multistep", "Crank–Nicolson", "forward Euler", 1, "Wang & Ruuth 2008, eq. 2.5 with γ = 1/2"),
    "CNAB2": ("multistep", "Crank–Nicolson", "2-step Adams–Bashforth", 2,
              "Wang & Ruuth 2008, eq. 2.9"),
    "MCNAB2": ("multistep", "modified Crank–Nicolson (9/16, 6/16, 1/16)",
               "2-step Adams–Bashforth", 2, "Wang & Ruuth 2008, eq. 2.10"),
    "CNLF2": ("multistep", "Crank–Nicolson over 2 dt", "leapfrog", 2,
              "Wang & Ruuth 2008, eq. 2.11"),
    "RK111": ("Runge–Kutta (1 stage)", "backward Euler", "forward Euler", 1,
              "Ascher, Ruuth & Spiteri 1997, (1,1,1)"),
    "RK222": ("Runge–Kutta (2 stages)", "L-stable SDIRK2", "2-stage Runge–Kutta", 2,
              "Ascher, Ruuth & Spiteri 1997, (2,2,2)"),
    "RK443": ("Runge–Kutta (4 stages)", "L-stable SDIRK3", "4-stage Runge–Kutta", 3,
              "Ascher, Ruuth & Spiteri 1997, (4,4,3)"),
    "RKSMR": ("Runge–Kutta (3 stages)", "Crank–Nicolson per stage (order 2)",
              "Wray's low-storage RK3 (order 3)", 2,
              "Spalart, Moser & Rogers 1991, J. Comput. Phys. 96, Appendix"),
    "RKGFY": ("Runge–Kutta (2 stages)", "Crank–Nicolson (order 2)", "Heun (order 2)", 2,
              "tableau c = (0, 1, 1), no literature reference"),
}


def scheme_family(name: str) -> str:
    """Family of a scheme: "sbdf", "cn" (Crank–Nicolson multistep) or "rk" (IMEX Runge–Kutta)."""
    if name in SBDF_SCHEMES:
        return "sbdf"
    if name in CN_SCHEMES:
        return "cn"
    if name in RK_SCHEMES:
        return "rk"
    raise OptionError(f"unknown scheme {name!r}; available: {SCHEMES}",
                      fix="pick one of the listed schemes." + did_you_mean(name, SCHEMES))


def scheme_description(name: str, effective: int | None = None) -> str:
    """One line: family, implicit / explicit parts, global order, reference. ``effective``: the
    global order of the run when its start-up or its boundary data lower it (audit invariance): the line then gives the formal order of the scheme and that effective
    order (the next line of explain() says why)."""
    if name not in _SCHEME_INFO:         # audit inputs: a bare KeyError
        raise OptionError(f"scheme_description({name!r}): unknown scheme",
                          fix="one of " + ", ".join(_SCHEME_INFO) + "."
                              + did_you_mean(str(name), _SCHEME_INFO))
    fam, imp, exp, p, ref = _SCHEME_INFO[name]
    order = (f"global order {p}" if effective is None or effective >= p else
             f"formal order {p}, effective global order {effective}")
    return f"{name}: IMEX {fam}, implicit {imp} on L, explicit {exp} on F; {order} ({ref})"


def sbdf_coefficients(q: int) -> tuple[list[float], list[float]]:
    """(a, c) for constant-step SBDF-q: a0 M X^n + dt L X^n = dt sum c_j F - sum a_j M X^{n-j}."""
    if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or q not in (1, 2, 3, 4):
        raise OptionError(f"sbdf_coefficients({q!r}): the SBDF order is 1, 2, 3 or 4",
                          fix="sbdf_coefficients(2) for SBDF2")
    return {1: ([1, -1], [0, 1]),
            2: ([1.5, -2, 0.5], [0, 2, -1]),
            3: ([11 / 6, -3, 1.5, -1 / 3], [0, 3, -3, 1]),
            4: ([25 / 12, -4, 3, -4 / 3, 0.25], [0, 4, -6, 4, -1])}[q]


def cn_coefficients(name: str) -> tuple[list[float], list[float], list[float]]:
    """(a, b, c) of a constant-step CN-family multistep scheme (Wang & Ruuth 2008), ``a`` in units
    of 1/dt:  sum_j (a_j/dt) M X^{n+1-j} + sum_j b_j L X^{n+1-j} = sum_{j>=1} c_j F^{n+1-j},
    j = 0..s (c_0 unused); s = len(a) - 1 history levels."""
    if name == "CNAB1":                                   # eq. 2.5, γ = 1/2
        return [1.0, -1.0], [0.5, 0.5], [0.0, 1.0]
    if name == "CNAB2":                                   # eq. 2.9
        return [1.0, -1.0, 0.0], [0.5, 0.5, 0.0], [0.0, 1.5, -0.5]
    if name == "MCNAB2":                                  # eq. 2.10
        return [1.0, -1.0, 0.0], [9 / 16, 6 / 16, 1 / 16], [0.0, 1.5, -0.5]
    if name == "CNLF2":                                   # eq. 2.11
        return [0.5, 0.0, -0.5], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]
    raise OptionError(f"{name!r} is not a CN-family multistep scheme ({', '.join(CN_SCHEMES)})")


def rk_tableau(name: str):
    """(A, H, c) of an IMEX Runge–Kutta scheme in cumulative form
    M X_i = M X_0 + dt sum_j A_ij F_j - dt sum_j H_ij L X_j (X_s = next state)."""
    if name == "RK111":
        return [[0, 0], [1, 0]], [[0, 0], [0, 1]], [0, 1]
    if name == "RK222":
        g = (2 - np.sqrt(2)) / 2
        d = 1 - 1 / g / 2
        return ([[0, 0, 0], [g, 0, 0], [d, 1 - d, 0]], [[0, 0, 0], [0, g, 0], [0, 1 - g, g]],
                [0, g, 1])
    if name == "RK443":
        return ([[0, 0, 0, 0, 0], [1 / 2, 0, 0, 0, 0], [11 / 18, 1 / 18, 0, 0, 0],
                 [5 / 6, -5 / 6, 1 / 2, 0, 0], [1 / 4, 7 / 4, 3 / 4, -7 / 4, 0]],
                [[0, 0, 0, 0, 0], [0, 1 / 2, 0, 0, 0], [0, 1 / 6, 1 / 2, 0, 0],
                 [0, -1 / 2, 1 / 2, 1 / 2, 0], [0, 3 / 2, -3 / 2, 1 / 2, 1 / 2]],
                [0, 1 / 2, 2 / 3, 1 / 2, 1])
    if name == "RKSMR":
        # Spalart, Moser & Rogers (1991), Appendix: substep i (i = 1, 2, 3) is
        #   X_i = X_{i-1} + dt [gamma_i F_{i-1} + zeta_i F_{i-2} - alpha_i L X_{i-1} - beta_i L X_i]
        # (Crank–Nicolson-like implicit part, Wray's RK3 explicit part), summed into cumulative
        # rows. Explicit part order 3, implicit part order 2 (tests/ivp/test_new_schemes.py).
        al, be = (29 / 96, -3 / 40, 1 / 6), (37 / 160, 5 / 24, 1 / 6)
        ga, ze = (8 / 15, 5 / 12, 3 / 4), (-17 / 60, -5 / 12)
        A = [[0, 0, 0, 0], [ga[0], 0, 0, 0], [ga[0] + ze[0], ga[1], 0, 0],
             [ga[0] + ze[0], ga[1] + ze[1], ga[2], 0]]
        H = [[0, 0, 0, 0], [al[0], be[0], 0, 0], [al[0], be[0] + al[1], be[1], 0],
             [al[0], be[0] + al[1], be[1] + al[2], be[2]]]
        return A, H, [0, 8 / 15, 2 / 3, 1]
    if name == "RKGFY":
        # RKGFY: Heun predictor-corrector on F, Crank–Nicolson on L
        return ([[0, 0, 0], [1, 0, 0], [1 / 2, 1 / 2, 0]],
                [[0, 0, 0], [1 / 2, 1 / 2, 0], [1 / 2, 0, 1 / 2]], [0, 1, 1])
    if name in SBDF_SCHEMES or name in CN_SCHEMES:
        raise OptionError(f"{name} is a multistep scheme: it has no Runge–Kutta tableau",
                          fix=f"use an IMEX Runge–Kutta scheme ({', '.join(RK_SCHEMES)}) here")
    raise OptionError(f"unknown RK scheme {name!r}; available: {RK_SCHEMES}",
                      fix="pick one of the listed schemes." + did_you_mean(name, RK_SCHEMES))


def setup_scheme(obj: Any, scheme: str, startup: str, *, where: str,
                 families: tuple[str, ...] = ("sbdf", "cn", "rk"), fix: str = "",
                 bc_time: str = "auto") -> None:
    """Validate (scheme, startup) and set the scheme attributes shared by every IVP solver
    (Cartesian, polar, spherical, distributed): ``family``, ``order`` (SBDF order, 0 otherwise),
    ``_rk`` (RK tableau or None), ``_cn`` (CN coefficients or None), ``_nhist`` (history levels of
    a multistep scheme, 0 for RK) and the L X history ``_LXh``. A solver that implements only
    some families lists them in ``families``: the others are refused explicitly."""
    if scheme not in SCHEMES:
        raise OptionError(f"unknown scheme {scheme!r}; available: {SCHEMES}",
                          where=where,
                          why="the time scheme is an explicit choice",
                          fix="pick one of the IMEX schemes listed (RK222/RK443/RKSMR/RKGFY "
                              "multistage, SBDF1-4/CNAB1-2/MCNAB2/CNLF2 multistep)."
                              + did_you_mean(scheme, SCHEMES))
    if startup not in STARTUPS:
        raise OptionError(f"startup must be one of {STARTUPS}", where=where,
                          fix="startup='lower_order' (Dedalus convention) or startup='rk443'")
    if bc_time not in BC_TIMES:
        raise OptionError(f"bc_time={bc_time!r}: one of {BC_TIMES}", where=where,
                          why="when time-dependent boundary data are imposed is a numerical choice",
                          fix="bc_time='auto' (default: 'rk' when g'(t) is exact, else 'stage'), "
                              "'rk' (stage values from the scheme applied to g'(t)), 'stage' "
                              "(g at every stage time) or 'step' (the Dedalus convention, bit for "
                              "bit)" + did_you_mean(str(bc_time), BC_TIMES))
    obj._bc_request = bc_time          # resolved lazily into IVPSolver.bc_time (needs the system)
    fam = scheme_family(scheme)
    if fam not in families:
        ok = [s for s in SCHEMES if scheme_family(s) in families]
        raise OptionError(f"scheme {scheme} ({fam} family) is not implemented here",
                          where=where, why="the scheme is not supported here",
                          fix=fix or "use one of " + ", ".join(ok))
    obj.family = fam
    obj.order = int(scheme[4:]) if fam == "sbdf" else 0
    obj._rk = rk_tableau(scheme) if fam == "rk" else None
    obj._cn = cn_coefficients(scheme) if fam == "cn" else None
    obj._nhist = obj.order if fam == "sbdf" else (len(obj._cn[0]) - 1 if fam == "cn" else 0)
    obj._LXh = deque()


def require_family(solver: Any, families: tuple[str, ...], where: str, fix: str) -> None:
    """Refuse (OptionError) a solver whose scheme family is not in ``families``."""
    fam = getattr(solver, "family", None) or scheme_family(solver.scheme)
    if fam not in families:
        raise OptionError(f"{where}: scheme {solver.scheme} ({fam} family) is not supported",
                          where=where, why="the scheme is not supported here", fix=fix)
