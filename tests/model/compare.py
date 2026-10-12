"""Exact comparison of a native model with a hand-written IR problem (bit identity).

The two problems may order their unknowns and equations differently (the native layer keeps
the user's declaration order): the comparison matches rows by (equation, component, occurrence)
and columns by (variable, component, occurrence) through explicit, documented maps
(native name -> hand-written name, native equation index -> hand-written index), then requires
**exact** equality (``np.array_equal``, no tolerance) of the permuted L and M, and of the
right-hand sides evaluated on the same random state.
"""
from __future__ import annotations

import numpy as np


def _keys(labels, rename):
    """(name, component, kind, occurrence) keys; ``rename(name)`` returns a name, or a dict
    {component tuple: (name, component tuple)} when a vector unknown or equation of one problem
    is written as scalars in the other (a native velocity u <-> hand-written u, v, w)."""
    seen: dict = {}
    out = []
    for lab in labels:
        r = rename(lab[0])
        name, comp = r[tuple(lab[1])] if isinstance(r, dict) else (r, tuple(lab[1]))
        k = (name, tuple(comp), lab[2])
        n = seen.get(k, 0)
        seen[k] = n + 1
        out.append(k + (n,))
    return out


def permutation(gm_native, gm_hand, var_map: dict, eq_map: dict):
    """(row perm, col perm) taking the native group system to the hand-written order."""
    rk_n = _keys(gm_native.rows, lambda i: eq_map[i])
    rk_h = _keys(gm_hand.rows, lambda i: i)
    ck_n = _keys(gm_native.cols, lambda v: var_map.get(v, v))
    ck_h = _keys(gm_hand.cols, lambda v: v)
    assert sorted(rk_n) == sorted(rk_h), "row labels differ"
    assert sorted(ck_n) == sorted(ck_h), "column labels differ"
    ri = {k: i for i, k in enumerate(rk_n)}
    ci = {k: i for i, k in enumerate(ck_n)}
    return np.array([ri[k] for k in rk_h]), np.array([ci[k] for k in ck_h])


def assert_same_matrices(plan_native, plan_hand, var_map: dict, eq_map: dict, groups=None):
    """Exact equality of L and M in every group (or the given pairs), up to the permutation.
    Returns (number of groups, number of nonzeros compared, whether the permutation is the
    identity)."""
    pairs = groups or list(zip(plan_native.groups, plan_hand.groups))
    nnz, identity = 0, True
    for gn, gh in pairs:
        a, b = plan_native.matrices(gn), plan_hand.matrices(gh)
        rp, cp = permutation(a, b, var_map, eq_map)
        identity &= bool(np.array_equal(rp, np.arange(len(rp))) and np.array_equal(cp, np.arange(len(cp))))
        for X, Y in ((a.L, b.L), (a.M, b.M)):
            Xp = X.tocsr()[rp][:, cp].toarray()
            Yd = Y.toarray()
            assert Xp.shape == Yd.shape
            assert np.array_equal(Xp, Yd), f"max |diff| = {np.abs(Xp - Yd).max():.3e} in {a.label}"
            nnz += Y.nnz
    return len(pairs), nnz, identity


def assert_same_rhs(solver_native, solver_hand, var_map: dict, eq_map: dict, seed: int = 0):
    """Exact equality of F(X) (explicit right-hand sides) per equation on a random state.
    The state is drawn for the hand-written variables and copied to the native ones by name."""
    from conatus.timestep.spectral import SpectralEvaluator  # noqa: F401  (import check)
    rng = np.random.default_rng(seed)
    sh = solver_hand.system
    coeffs = [rng.standard_normal(s) for s in sh.var_shapes]
    by_name = {v.name: c for v, c in zip(sh.variables, coeffs)}
    sn = solver_native.system
    inv = {v: k for k, v in var_map.items()}
    cn = [by_name[var_map.get(v.name, v.name)] for v in sn.variables]
    Xh, Xn = sh.gather_state(coeffs), sn.gather_state(cn)
    ev_h, ev_n = solver_hand.evaluator, solver_native.evaluator
    data_h = dict(solver_hand._ncc)
    data_h.update(zip(sh.variables, sh.scatter(Xh)))
    data_n = dict(solver_native._ncc)
    data_n.update(zip(sn.variables, sn.scatter(Xn)))
    eqs_h = solver_hand.plan.problem.equations
    eqs_n = solver_native.plan.problem.equations
    checked = 0
    for i, eq in enumerate(eqs_n):
        j = eq_map[i]
        rn, rh = eq.rhs, eqs_h[j].rhs
        if rn.is_zero() and rh.is_zero():
            continue
        a = ev_n.evaluate(rn, data_n, dict(solver_native.plan.params))
        b = ev_h.evaluate(rh, data_h, dict(solver_hand.plan.params))
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and np.array_equal(a, b), \
            f"RHS of native [{i}] vs hand [{j}]: max |diff| = {np.abs(a - b).max():.3e}"
        checked += 1
    Fn, Fh = solver_native.F(Xn, 0.0), solver_hand.F(Xh, 0.0)
    del inv
    return checked, float(np.abs(Fh).max()), (Fn, Fh)


def assert_same_F(solver_native, solver_hand, seed: int = 0):
    """Identity permutation (same unknown and equation order): exact equality of the full
    explicit right-hand side F(X, t=0) on one random state vector. Returns max |F|."""
    n = solver_hand.X.shape[0]
    assert solver_native.X.shape[0] == n
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n).astype(solver_hand.X.dtype)
    Fn, Fh = solver_native.F(X.copy(), 0.0), solver_hand.F(X.copy(), 0.0)
    assert np.array_equal(Fn, Fh), f"max |dF| = {np.abs(Fn - Fh).max():.3e}"
    return float(np.abs(Fh).max())


def rhs_difference(solver_native, solver_hand, var_map: dict, eq_map: dict, *, t: float = 0.0,
                   seed: int = 0) -> tuple[int, float, float]:
    """Explicit right-hand sides per equation on one random state at time ``t`` (the time
    parameter of both solvers), with component maps: a ``var_map``/``eq_map`` value may be
    {component: (name, ())} (a native vector written as scalars by hand). Returns (equations
    compared, max |native - hand|, max |hand|); the caller states the tolerance (0 for bit
    identity, a few ulps when the vector form adds the same products in another order)."""
    rng = np.random.default_rng(seed)
    sh, sn = solver_hand.system, solver_native.system
    coeffs = [rng.standard_normal(s) for s in sh.var_shapes]
    by_name = {v.name: c for v, c in zip(sh.variables, coeffs, strict=True)}
    cn = []
    for v, shape in zip(sn.variables, sn.var_shapes, strict=True):
        r = var_map.get(v.name, v.name)
        if isinstance(r, dict):
            arr = np.zeros(shape)
            for comp, (name, _) in r.items():
                # same shape, no broadcasting of a hand-written scalar into a component
                assert arr[comp].shape == by_name[name].shape, (v.name, comp, name)
                arr[comp] = by_name[name]
            assert len(r) == shape[0], f"{v.name}: {len(r)} of {shape[0]} components mapped"
            cn.append(arr)
        else:
            assert by_name[r].shape == tuple(shape), (v.name, r)
            cn.append(by_name[r])
    Xh, Xn = sh.gather_state(coeffs), sn.gather_state(cn)
    data_h = dict(solver_hand._ncc)
    data_h.update(zip(sh.variables, sh.scatter(Xh), strict=True))
    data_n = dict(solver_native._ncc)
    data_n.update(zip(sn.variables, sn.scatter(Xn), strict=True))
    ph, pn = dict(solver_hand.plan.params), dict(solver_native.plan.params)
    for sv, prm in ((solver_hand, ph), (solver_native, pn)):
        if sv.time_parameter:
            prm[sv.time_parameter] = t
    eqs_h, eqs_n = solver_hand.plan.problem.equations, solver_native.plan.problem.equations
    checked, worst, scale = 0, 0.0, 0.0
    for i, eq in enumerate(eqs_n):
        r = eq_map[i]
        pairs = ([((c,), j) for (c,), (j, _) in r.items()] if isinstance(r, dict) else [((), r)])
        a_full = None if eq.rhs.is_zero() else np.asarray(
            solver_native.evaluator.evaluate(eq.rhs, data_n, pn))
        for comp, j in pairs:
            rh = eqs_h[j].rhs
            if a_full is None and rh.is_zero():
                continue
            b = np.asarray(solver_hand.evaluator.evaluate(rh, data_h, ph))
            a = np.zeros_like(b) if a_full is None else a_full[comp]
            assert a.shape == b.shape, f"RHS shapes native [{i}]{comp} {a.shape} vs hand [{j}] {b.shape}"
            worst = max(worst, float(np.abs(a - b).max()))
            scale = max(scale, float(np.abs(b).max()))
            checked += 1
    return checked, worst, scale
