"""KS equilibria continued in the domain length L (adapter over conatus.ecs)."""
import numpy as np
import pytest

from conatus.continuation.core import ContinuationSettings
from conatus.continuation.ks import KSDomainContinuation
from conatus.ecs import NewtonKrylovHookstep, find_equilibrium, ks_solver

N = 32


@pytest.fixture(scope="module")
def e1():
    ks = ks_solver(L=22.0, N=N, dealias=1.5, scheme="RK443", timestep=0.01)
    nk = NewtonKrylovHookstep(tol=1e-12, max_iterations=40, gmres_rtol=1e-3, gmres_max_iterations=60,
                              delta0=1.0, delta_min=1e-10, delta_max=10.0)
    c = np.zeros(N)
    c[3] = -1.0
    return find_equilibrium(ks, ks.system.gather_state([c]), nk).X


def adapter():
    return KSDomainContinuation(N=N, dealias=1.5, dp_fd=1e-6, marginal_tol=1e-8, eig_count=8)


def test_residual_jacobian_and_energy(e1):
    kc = adapter()
    x = kc.active(e1, 22.0)
    assert np.max(np.abs(kc.residual(x, 22.0))) < 1e-12
    assert abs(kc.energy(x, 22.0) - 0.2609) < 6e-5          # Cvitanović et al. 2010, Table 5.2
    rng = np.random.default_rng(0)
    d, h = rng.normal(size=x.size), 1e-6
    fd = (kc.residual(x + h * d, 22.0) - kc.residual(x - h * d, 22.0)) / (2 * h)
    assert np.linalg.norm(kc.jacobian(x, 22.0) @ d - fd) < 1e-6 * np.linalg.norm(fd)
    # L enters the residual (derivative coefficients 2πn/L): G_L is nonzero at an equilibrium
    assert np.linalg.norm(kc.residual(x, 22.1)) > 1e-3


def test_marginal_translation_mode_is_not_counted(e1):
    kc = adapter()
    ev = kc.eigenvalues(kc.active(e1, 22.0), 22.0)
    assert np.sum(ev.real > 0) == 4                         # two unstable pairs (Table 5.1)
    assert np.any(ev == 0)                                  # translation, |Re| <= marginal_tol
    assert "marginal" in kc.explain()


def test_start_outside_subspace_refused(e1):
    kc = adapter()
    bad = kc.at(22.0).ops.translate(e1, 0.37)               # not odd about x = 0
    cfg = ContinuationSettings(ds=0.05, ds_min=1e-6, ds_max=0.2, adapt=1.4, fast_iterations=3,
                               tolerance=1e-10, max_newton=12, max_points=2, dp_fd=1e-6)
    with pytest.raises(ValueError, match="antisymmetric"):
        kc.run(bad, 22.0, cfg, direction=-1, L_bounds=(18.0, 26.0))


@pytest.mark.slow
def test_e3_ends_in_pitchfork_off_zero_at_6pi():
    """E3 (3-cell state) continued down in L reaches u = 0 at L = 2π·3 (Cvitanović, Davidchack &
    Siminos 2010, §2: n-cell states are born in pitchforks off u = 0 when L/2π crosses n)."""
    ks = ks_solver(L=22.0, N=N, dealias=1.5, scheme="RK443", timestep=0.01)
    nk = NewtonKrylovHookstep(tol=1e-12, max_iterations=40, gmres_rtol=1e-3, gmres_max_iterations=60,
                              delta0=1.0, delta_min=1e-10, delta_max=10.0)
    c = np.zeros(N)
    c[7] = -2.0
    X0 = find_equilibrium(ks, ks.system.gather_state([c]), nk).X
    kc = adapter()
    cfg = ContinuationSettings(ds=0.05, ds_min=1e-6, ds_max=0.25, adapt=1.4, fast_iterations=3,
                               tolerance=1e-10, max_newton=12, max_points=30, dp_fd=1e-6)
    br = kc.continuation(cfg, eigen=False).run(kc.active(X0, 22.0), 22.0, direction=-1,
                                               p_bounds=(18.0, 26.0))
    Lmin = min(q.p for q in br.points)
    assert abs(Lmin - 6 * np.pi) < 1e-3
    q = min(br.points, key=lambda q: q.p)
    assert kc.energy(q.x, q.p) < 1e-3                       # amplitude -> 0 at the pitchfork


def test_nonpositive_length_is_a_corrector_failure():
    from conatus.continuation.ks import KSDomainContinuation
    kc = KSDomainContinuation(N=16, dealias=1.5, dp_fd=1e-6, marginal_tol=1e-8, eig_count=4)
    with pytest.raises(RuntimeError, match="must be > 0"):
        kc.at(-2.1)


@pytest.mark.slow
def test_e1_folds_match_kevrekidis_nicolaenko_scovel(e1):
    """E1 (GLMRT) continued down from L = 22 is the BT branch of Kevrekidis, Nicolaenko & Scovel
    (1990, SIAM J. Appl. Math. 50, 760–790, p. 774): turning point K at α = 34.1692 and collision
    with the trimodal branch at I, α = 36.235. Their equation u_t + 4u_xxxx + α(u_xx + ½u_x²) = 0 on
    2π is Greene–Kim's, so L = π√α (derivation)."""
    from conatus.continuation.core import ContinuationSettings as S
    cfg = S(ds=0.05, ds_min=1e-6, ds_max=0.25, adapt=1.4, fast_iterations=3, tolerance=1e-10,
            max_newton=12, max_points=60, dp_fd=1e-6)
    br = adapter().run(e1, 22.0, cfg, direction=-1, L_bounds=(18.0, 26.0), eigen=False)
    alphas = sorted((f.p / np.pi) ** 2 for f in br.folds())
    assert any(abs(a - 34.1692) < 1e-4 for a in alphas), alphas       # K (6 digits published)
    assert any(abs(a - 36.235) < 5e-4 for a in alphas), alphas        # I (5 digits published)
